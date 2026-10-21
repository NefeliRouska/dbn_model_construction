"""
run_grid.py -- run several planners on the testbed (or the simulator) one
after another, unattended: the short live check and the full experiment.

For every repetition, every planner runs once through the given load patterns
(loop.py, one JSON-lines log per run). The order of the planners is shuffled
per repetition (seeded), so that no planner always runs first or last.
Between two runs the chain is drained: the load is set to DRAIN_RPS with a
high-capacity configuration until the queues of services 2 and 3 are below
DRAIN_QUEUE (at most DRAIN_MAX s; a WARNING is logged if they are still
longer), so that every run starts from the same state.

A run that fails is reported and the grid goes on; a progress line per run is
written to the grid's own log, so an interruption loses at most the run in
progress, and --skip-done resumes a grid without repeating finished runs.

Usage (on the lab machine, services running):
    python adaptation/run_grid.py --live --patterns check --minutes 10 --repeats 1 --no-shuffle
    python adaptation/run_grid.py --live --minutes 60 --repeats 3            # the full experiment
"""
import argparse
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import loop as L  # noqa: E402

PLANNERS = ("dbn", "minrule", "static", "vpa", "tabpfn", "rask")
ALL_PATTERNS = "periodic,once_in_a_lifetime,unpredictable,diurnal,bursty"
DRAIN_RPS, DRAIN_QUEUE, DRAIN_MAX = 100, 1000, 300     # RASK can leave a backlog of 10,000 and more
DRAIN_CONFIG = dict(cores_1=3, cores_2=3, cores_3=2, data_quality_1=100, data_quality_2=100, data_quality_3=100)


def drain(log):
    import monitor as M
    chain = L.LiveChain()
    chain.set_load(DRAIN_RPS)
    _, row = M.query_now()
    chain.executor.apply({c: L.setting(row[c]) for c in L.PL.SETTINGS}, DRAIN_CONFIG)
    t0 = time.time()
    while time.time() - t0 < DRAIN_MAX:
        time.sleep(5)
        _, row = M.query_now()
        if row["buffer_size_2"] < DRAIN_QUEUE and row["buffer_size_3"] < DRAIN_QUEUE and time.time() - t0 >= 15:
            break
    ok = row["buffer_size_2"] < DRAIN_QUEUE and row["buffer_size_3"] < DRAIN_QUEUE
    log(f"  {'drained' if ok else 'WARNING: not drained'} in {time.time() - t0:.0f} s "
        f"(queues {row['buffer_size_2']:.0f} / {row['buffer_size_3']:.0f})")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--sim", action="store_true")
    ap.add_argument("--planners", default=",".join(PLANNERS))
    ap.add_argument("--patterns", default=ALL_PATTERNS)
    ap.add_argument("--minutes", type=float, default=60, help="minutes per load pattern")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-done", default=None, help="a previous grid log: skip the runs it lists as done")
    args = ap.parse_args()

    planners = args.planners.split(",")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    L.RESULTS.mkdir(parents=True, exist_ok=True)
    grid_log = L.RESULTS / f"grid_{'live' if args.live else 'sim'}_{stamp}{args.tag}.log"
    done = set()
    if args.skip_done:
        done = {line.split("DONE ")[1].split(" ->")[0].strip() for line in open(args.skip_done) if "DONE " in line}

    def log(msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(grid_log, "a") as f:
            f.write(line + "\n")

    log(f"grid: planners {planners}, patterns {args.patterns}, {args.minutes} min each, {args.repeats} repetitions")
    for rep in range(args.repeats):
        order = list(planners)
        if not args.no_shuffle:
            random.Random(1000 + rep).shuffle(order)
        log(f"repetition {rep}: order {order}")
        for p in order:
            key = f"{p} seed{rep}"
            if key in done:
                log(f"SKIP {key} (done in {args.skip_done})")
                continue
            if args.live:
                drain(log)
            cmd = [sys.executable, str(HERE / "loop.py"), "--live" if args.live else "--sim", "--planner", p,
                   "--patterns", args.patterns, "--minutes", str(args.minutes), "--seed", str(rep)]
            if args.tag:
                cmd += ["--tag", args.tag]
            t0 = time.time()
            log(f"START {key}")
            r = subprocess.run(cmd, capture_output=True, text=True)
            out = [line for line in r.stdout.splitlines() if line.startswith("log:")]
            if r.returncode == 0:
                log(f"DONE {key} -> {out[-1][5:] if out else '?'} ({(time.time() - t0) / 60:.1f} min)")
            else:
                log(f"FAILED {key} (exit {r.returncode}): {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
    log("grid finished")


if __name__ == "__main__":
    main()
