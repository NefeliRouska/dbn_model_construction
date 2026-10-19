"""
calibrate_live.py -- before the loop runs on the testbed: does the live chain
still behave like the one the data were recorded on, and how long does it take
to settle after a change?

At a fixed load, a few configurations from the training schedule are applied
one after another and held for HOLD seconds, read through the Monitor every
second. Per configuration:

  settle_s     seconds after the change until throughput_3 stays within 5%
               of its mean over the last 60 s
  tp1 / tp3    mean over the last 60 s
  expected     min(load, capacity_1) with the model's capacity function
  training     mean throughput_3 of the training runs with the same
               configuration (any load), for reference

Changes the testbed: it sets the load and the settings of the three services.

Usage (on the lab machine, services running):
    python adaptation/calibrate_live.py [--load 300] [--hold 120]
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dbn"))
sys.path.insert(0, str(HERE))
import build_model as B  # noqa: E402
import loop as L  # noqa: E402
import monitor as M  # noqa: E402
import planner as PL  # noqa: E402
from check_monitor import DEFAULT_MODEL  # noqa: E402

# (cores_1..3, quality_1..3): service 1 is the bottleneck in the first four
CONFIGS = [(3, 3, 1, 700, 700, 200), (3, 3, 1, 900, 900, 200), (1, 3, 3, 900, 600, 600),
           (1, 3, 3, 300, 300, 300), (3, 3, 1, 100, 100, 100)]


def settle_time(tp, window=60, tol=0.05):
    ref = np.mean(tp[-window:])
    off = np.abs(np.asarray(tp) - ref) > tol * max(ref, 1.0)
    late = np.flatnonzero(off)
    return int(late[-1] + 1) if len(late) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=int, default=300)
    ap.add_argument("--hold", type=int, default=120)
    args = ap.parse_args()

    model = B.Model.load(str(DEFAULT_MODEL))
    train = pd.concat([model.load_frame(HERE.parent / "data" / f) for f in model.meta["files"]])
    chain = L.LiveChain()
    chain.set_load(args.load)
    out = L.RESULTS / f"calibration_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    _, row = M.query_now()
    current, rows = {c: int(row[c]) for c in PL.SETTINGS}, []
    with open(out, "w") as log:
        for cfg in CONFIGS:
            config = dict(zip(PL.SETTINGS, cfg))
            chain.executor.apply(current, config)
            current, t_change, tp1, tp3 = config, time.time(), [], []
            for k in range(args.hold):
                time.sleep(max(0.0, t_change + k + 1 - time.time()))
                ts, row = M.query_now()
                log.write(json.dumps(dict(config=cfg, second=k + 1, ts=ts, **row)) + "\n")
                tp1.append(row["throughput_1"]), tp3.append(row["throughput_3"])
            cap = B.add_capacity(pd.DataFrame([config]), model.capacity)
            same = (train[PL.SETTINGS] == pd.Series(config)).all(axis=1)
            rows.append(dict(config=cfg, settle_s=settle_time(tp3), tp1=np.mean(tp1[-60:]), tp3=np.mean(tp3[-60:]),
                             expected=min(args.load, cap["capacity_1"].iloc[0]),
                             training_tp3=train.loc[same, "throughput_3"].mean() if same.any() else np.nan,
                             training_rps=train.loc[same, "rps"].mean() if same.any() else np.nan))
            print(pd.DataFrame(rows[-1:]).round(1).to_string(index=False, header=len(rows) == 1), flush=True)
    print(f"log: {out}")


if __name__ == "__main__":
    main()
