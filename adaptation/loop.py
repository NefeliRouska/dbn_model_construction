"""
loop.py -- the MAPE-K adaptation loop: Monitor, Analyze, Plan, Execute, once
per second, on the live chain or on the simulator.

  Monitor   monitor.query_now (live) / SimulatedChain.read: one row per second
  Analyze   Analyzer: act only after the cooldown (COOLDOWN s after an action), and only when the chain is
            failing (completion tp3/load below FAIL_BELOW, median of the last
            FAIL_SECONDS seconds: the services stall for 1-2 s now and then,
            ~1% of seconds in the training data, and a mean would call that a
            failure) or the load has moved by more than LOAD_CHANGE since the
            last plan. The first plan waits FAIL_SECONDS seconds, so the
            Monitor already sees the load that was set.
  Plan      planner.Planner.decide (expected SLO utility over 12,320 configurations)
  Execute   Executor: the API calls of data_collection/experiment_orchestrator.py,
            cores lowered before others are raised so the budget of 8 holds
  Knowledge the model, the planner's settled chain and the log of every second

The load follows a program: the training patterns one after another
(workload_rps of the orchestrator), each from its own start.

Every second is written to a JSON-lines log (one dict per second, with the
decision when there was one); summary() turns a log into the run's metrics.

Usage:
    python adaptation/loop.py --sim [--minutes 60] [--planner dbn|minrule|static|vpa]
    python adaptation/loop.py --live --minutes 60      # on the lab machine only
    python adaptation/loop.py --live --minutes 10 --patterns periodic   # a short test
"""
import argparse
import json
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "dbn"))
sys.path.insert(0, str(REPO / "data_collection"))
sys.path.insert(0, str(HERE))
import build_model as B  # noqa: E402
import planner as PL  # noqa: E402
from check_monitor import DEFAULT_MODEL  # noqa: E402

RESULTS = REPO / "results" / "adaptation_loop"

FAIL_BELOW, FAIL_SECONDS = 0.97, 5     # failing: median completion of the last 5 s below 0.97
LOAD_CHANGE = 0.10                     # re-plan when the load moved by more than 10%
COOLDOWN = 10                          # seconds after an action before the next one: on the testbed a
                                       # change settles within 5-8 s (calibrate_live.py), and Sedlak et
                                       # al. act every 10 s
RECHECK = 10                           # seconds between plans that end without an action
PATTERNS = ("periodic", "once_in_a_lifetime", "unpredictable")
START = dict(cores_1=3, cores_2=3, cores_3=2, data_quality_1=600, data_quality_2=500, data_quality_3=400)


def completion(row):
    return min(row["throughput_3"] / row["rps"], 1.0) if row["rps"] > 0 else 1.0


def utility(row):
    """The planner's utility, measured: the SLO fulfillment of this second."""
    q = sum(min(row[f"data_quality_{i}"] / PL.QUALITY_TARGET, 1.0) for i in (1, 2, 3))
    return (PL.W_COMPLETION * completion(row) + PL.W_QUALITY * q) / (PL.W_COMPLETION + 3 * PL.W_QUALITY)


class LoadProgram:
    """The load at second t of the episode: each pattern for `seconds`, from its own start."""

    def __init__(self, patterns=PATTERNS, seconds=3600, seed=783122):
        from experiment_orchestrator import workload_rps
        self.workload_rps, self.patterns, self.seconds, self.seed = workload_rps, patterns, seconds, seed
        self.duration = len(patterns) * seconds

    def __call__(self, t):
        k = min(int(t // self.seconds), len(self.patterns) - 1)
        return self.patterns[k], self.workload_rps(self.patterns[k], t - k * self.seconds, self.seed)


class Analyzer:
    def __init__(self, fail_below=FAIL_BELOW, fail_seconds=FAIL_SECONDS, load_change=LOAD_CHANGE,
                 cooldown=COOLDOWN, recheck=RECHECK):
        self.fail_below, self.fail_seconds, self.load_change = fail_below, fail_seconds, load_change
        self.cooldown, self.recheck = cooldown, recheck
        self.recent, self.last_action, self.last_plan, self.planned_load = [], -np.inf, -np.inf, None

    def update(self, t, row):
        self.recent = (self.recent + [completion(row)])[-self.fail_seconds:]

    def reason(self, t, row):
        """Why the loop should plan now, or None."""
        if t - self.last_action < self.cooldown or len(self.recent) < self.fail_seconds:
            return None
        if self.planned_load is None:
            return "start"
        if abs(row["rps"] - self.planned_load) > self.load_change * max(self.planned_load, 1.0):
            return "load changed"
        failing = np.median(self.recent) < self.fail_below
        if failing and t - self.last_plan >= self.recheck:
            return "failing"
        return None

    def planned(self, t, row, acted):
        self.last_plan, self.planned_load = t, row["rps"]
        if acted:
            self.last_action = t


class Executor:
    """Apply a configuration through the services' REST API."""

    def __init__(self):
        from experiment_orchestrator import SERVICES, set_cores, set_quality
        self.services = [SERVICES[s] for s in ("svc1", "svc2", "svc3")]
        self.set_cores, self.set_quality = set_cores, set_quality

    def apply(self, current, config):
        """current None (first call): every setting is sent, the smallest core counts first."""
        if current is None:
            current = {c: -1 for c in config}
        # lower cores first, then raise, so c1 + c2 + c3 never exceeds the budget
        order = sorted((1, 2, 3), key=lambda i: config[f"cores_{i}"] - current[f"cores_{i}"])
        for i in order:
            if config[f"cores_{i}"] != current[f"cores_{i}"]:
                self.set_cores(self.services[i - 1], int(config[f"cores_{i}"]))
        for i in (1, 2, 3):
            if config[f"data_quality_{i}"] != current[f"data_quality_{i}"]:
                self.set_quality(self.services[i - 1], int(config[f"data_quality_{i}"]))


class LiveChain:
    """The real chain: Monitor = Prometheus, Execute = REST API, load = /change_rps."""

    def __init__(self):
        import monitor as M
        from experiment_orchestrator import SERVICES, SOURCE_CLIENT_ID, set_rps
        self.query_now, self.executor = M.query_now, Executor()
        self.set_rps = lambda rps: set_rps(SERVICES["svc1"], SOURCE_CLIENT_ID, int(rps))
        self.next_tick, self.config = time.time(), None

    def set_load(self, rps):
        self.set_rps(rps)

    def apply(self, config):
        self.executor.apply(self.config, config)

    def read(self):
        self.next_tick += 1.0
        time.sleep(max(0.0, self.next_tick - time.time()))
        ts, row = self.query_now()
        self.config = {c: row[c] for c in PL.SETTINGS}
        return ts, row


def peak_rss_mb():
    """Peak resident memory of this process (the loop, with the model and the planner) in MB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10        # bytes on macOS, KB on Linux


def run(system, plan, program, log_path, start=START):
    """
    One episode. plan(load, current, row) -> decision dict (planner.Planner.decide's
    shape) or None for a planner that never acts; start: the configuration
    applied before the first second.
    """
    analyzer, last_load, t0 = Analyzer(), None, None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    system.apply(start)
    with open(log_path, "w") as log:
        for t in range(program.duration):
            pattern, load = program(t)
            if load != last_load:
                system.set_load(load)
                last_load = load
            ts, row = system.read()
            t0 = ts if t0 is None else t0
            analyzer.update(t, row)
            rec = dict(t=t, ts=ts, pattern=pattern, load_set=load, **row,
                       completion=completion(row), utility=utility(row),
                       cpu_s=time.process_time(), rss_mb=peak_rss_mb())
            why = analyzer.reason(t, row)
            if why is not None and plan is not None:
                current = {c: int(row[c]) for c in PL.SETTINGS}
                t_plan = time.perf_counter()
                d = plan(row["rps"], current, row=row)
                rec.update(event=why, plan_ms=round((time.perf_counter() - t_plan) * 1000, 1),
                           act=d["act"], new_config=d["config"] if d["act"] else None,
                           predicted_completion=d["completion"], predicted_utility=d["utility"],
                           current_predicted_completion=d["completion_current"])
                if d["act"]:
                    system.apply(d["config"])
                analyzer.planned(t, row, d["act"])
            log.write(json.dumps(rec, default=float) + "\n")
    return log_path


def summary(log_path):
    """The run's metrics, overall and per load pattern."""
    df = pd.read_json(log_path, lines=True)
    df["cores"] = df[["cores_1", "cores_2", "cores_3"]].sum(axis=1)
    df["quality"] = df[["data_quality_1", "data_quality_2", "data_quality_3"]].mean(axis=1)
    df["violation"] = df["completion"] < 1.0
    acts = df[df.get("act", pd.Series(False, index=df.index)).fillna(False).astype(bool)]

    def block(g):
        return pd.Series(dict(
            seconds=len(g), slo_fulfillment=g["utility"].mean(), completion=g["completion"].mean(),
            violation_share=g["violation"].mean(), mean_quality=g["quality"].mean(),
            core_seconds=g["cores"].sum(), actions=int(g.index.isin(acts.index).sum())))
    out = df.groupby("pattern", sort=False).apply(block, include_groups=False)
    out.loc["all"] = block(df)
    return out


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sim", action="store_true")
    mode.add_argument("--live", action="store_true")
    ap.add_argument("--minutes", type=float, default=60, help="minutes per load pattern")
    ap.add_argument("--patterns", default=",".join(PATTERNS), help="comma-separated, in order")
    ap.add_argument("--planner", default="dbn", choices=["dbn", "minrule", "static", "vpa"])
    ap.add_argument("--bins", type=int, default=PL.PLANNER_BINS, help="bins of the planner's chain (0 = the model's)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    model = B.Model.load(str(DEFAULT_MODEL))
    import baselines as BL
    patterns = tuple(args.patterns.split(","))
    program = LoadProgram(patterns, seconds=int(args.minutes * 60))
    start, plan = START, None
    if args.planner == "dbn":
        plan = PL.Planner(model, PL.load_chain(PL.chain_path(DEFAULT_MODEL, args.bins or None))).decide
    elif args.planner == "minrule":
        plan = BL.MinRulePlanner(model).decide
    elif args.planner in ("static", "vpa"):
        start, u = BL.static_best(model, [program(t)[1] for t in range(program.duration)])
        print(f"start configuration {start} (static_best), mean min-rule utility over the program {u:.3f}")
        if args.planner == "vpa":
            plan = BL.VpaPlanner(model).decide

    if args.sim:
        from simulator import SimulatedChain
        system = SimulatedChain(model.capacity, start, seed=args.seed)
    else:
        system = LiveChain()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bins = f"_{args.bins or 20}bins" if args.planner == "dbn" else ""
    name = f"{'sim' if args.sim else 'live'}_{args.planner}{bins}_{int(args.minutes)}min_seed{args.seed}{args.tag}_{stamp}"
    log = run(system, plan, program, RESULTS / f"{name}.jsonl", start)
    print(f"log: {log}")
    print(summary(log).round(3).to_string())


if __name__ == "__main__":
    main()
