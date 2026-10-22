"""
run_grid.py -- run the planners on the testbed (or the simulator) one after
another, unattended: the short live check and the full experiment.

One run = one planner on ONE load pattern (loop.py, one JSON-lines log per
run), as in Sedlak et al., where every pattern was a run of its own: nothing a
planner leaves behind in one pattern (a backlog, RASK's learned curves) carries
into the next. Every repetition runs every (planner, pattern) pair once, in an
order shuffled per repetition (seeded), so that no pair always runs first or
last.

Before every run the chain is drained to empty: the load is switched off and a
high-capacity configuration applied until the queues of services 2 and 3 are
below DRAIN_QUEUE, for at least DRAIN_MIN and at most DRAIN_MAX seconds. If
they are still longer (RASK can leave several 100,000 requests queued), the
three service containers are restarted, which empties their queues, and the
drain is repeated. Every drain is logged, so every run's starting state is on
record.

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
DRAIN_QUEUE, DRAIN_MIN, DRAIN_MAX = 100, 15, 1800
DRAIN_CONFIG = dict(cores_1=3, cores_2=3, cores_3=2, data_quality_1=100, data_quality_2=100, data_quality_3=100)
SERVICE_CONTAINERS = [f"elastic-workbench-linked-service-{i}" for i in (1, 2, 3)]
READY_TIMEOUT = 180


def _read():
    import monitor as M
    try:
        _, row = M.query_now()
        return row if len(row) == 15 else None
    except Exception:
        return None


def _drain_once(log, max_s):
    chain = L.LiveChain()
    chain.set_load(0)
    row = _read()
    current = {c: L.setting(row[c]) for c in L.PL.SETTINGS} if row else None
    chain.executor.apply(current, DRAIN_CONFIG)
    t0 = time.time()
    while True:
        time.sleep(5)
        row = _read()
        empty = row is not None and row["buffer_size_2"] < DRAIN_QUEUE and row["buffer_size_3"] < DRAIN_QUEUE
        if (empty and time.time() - t0 >= DRAIN_MIN) or time.time() - t0 >= max_s:
            return empty, row, time.time() - t0


def _restart_services(log):
    subprocess.run(["docker", "restart"] + SERVICE_CONTAINERS, capture_output=True, text=True)
    t0 = time.time()
    while time.time() - t0 < READY_TIMEOUT:
        time.sleep(5)
        if _read() is not None:
            log(f"  services restarted, metrics back after {time.time() - t0:.0f} s")
            return True
    log("  WARNING: services restarted but metrics not back")
    return False


def drain(log):
    empty, row, secs = _drain_once(log, DRAIN_MAX)
    q = f"queues {row['buffer_size_2']:.0f} / {row['buffer_size_3']:.0f}" if row else "no metrics"
    if empty:
        log(f"  drained in {secs:.0f} s ({q})")
        return
    log(f"  not drained in {secs:.0f} s ({q}); restarting the service containers")
    _restart_services(log)
    empty, row, secs = _drain_once(log, 300)
    q = f"queues {row['buffer_size_2']:.0f} / {row['buffer_size_3']:.0f}" if row else "no metrics"
    log(f"  {'drained' if empty else 'WARNING: not drained'} after restart in {secs:.0f} s ({q})")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--sim", action="store_true")
    ap.add_argument("--planners", default=",".join(PLANNERS))
    ap.add_argument("--patterns", default=ALL_PATTERNS)
    ap.add_argument("--minutes", type=float, default=60, help="minutes per run (one load pattern)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-done", default=None, help="a previous grid log: skip the runs it lists as done")
    args = ap.parse_args()

    planners, patterns = args.planners.split(","), args.patterns.split(",")
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

    log(f"grid: planners {planners}, patterns {patterns}, {args.minutes} min per run, {args.repeats} repetitions, "
        f"{len(planners) * len(patterns) * args.repeats} runs")
    for rep in range(args.repeats):
        order = [(p, pat) for p in planners for pat in patterns]
        if not args.no_shuffle:
            random.Random(1000 + rep).shuffle(order)
        log(f"repetition {rep}: order {[f'{p}/{pat}' for p, pat in order]}")
        for k, (p, pat) in enumerate(order):
            key = f"{p} {pat} seed{rep}"
            if key in done:
                log(f"SKIP {key} (done in {args.skip_done})")
                continue
            if args.live:
                drain(log)
            cmd = [sys.executable, str(HERE / "loop.py"), "--live" if args.live else "--sim", "--planner", p,
                   "--patterns", pat, "--minutes", str(args.minutes), "--seed", str(rep)]
            if args.tag:
                cmd += ["--tag", args.tag]
            t0 = time.time()
            log(f"START {key} ({k + 1}/{len(order)} of repetition {rep})")
            r = subprocess.run(cmd, capture_output=True, text=True)
            out = [line for line in r.stdout.splitlines() if line.startswith("log:")]
            if r.returncode == 0:
                log(f"DONE {key} -> {out[-1][5:] if out else '?'} ({(time.time() - t0) / 60:.1f} min)")
            else:
                log(f"FAILED {key} (exit {r.returncode}): {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
    log("grid finished")


if __name__ == "__main__":
    main()
