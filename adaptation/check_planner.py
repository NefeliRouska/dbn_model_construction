"""
check_planner.py -- offline checks of the Plan step.

  1. speed        time of one decision over the 12,320 candidates
  2. choices      what the planner picks at loads from 100 to 600 req/s,
                  starting from a mid configuration
  3. bottleneck   1 core at quality 800 for service 1 (~34 req/s) under a
                  load of 200: the settled chain must not deliver more than
                  service 1 can process
  4. completion   predicted E[phi_completion] against the measured
                  min(throughput_3 / load, 1) of 30 s windows of the recorded
                  runs (the run's own configuration and mean load), separately
                  for settled windows and windows with a draining second.
                  The final model was fitted on these runs, so this is a
                  consistency check of the planner's arithmetic (units, bins,
                  capacities), not a validation of the model.

Checks 2 to 4 are run with the model's own chain and with the settled chain
(planner.py), side by side.

Usage: python adaptation/check_planner.py [bins of the settled chain, 0 = the model's; default 50]

"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dbn"))
sys.path.insert(0, str(HERE))
import build_model as B  # noqa: E402
import data as D  # noqa: E402
import planner as PL  # noqa: E402
from check_monitor import DEFAULT_MODEL  # noqa: E402

DATA = HERE.parent / "data"
START = dict(cores_1=3, cores_2=3, cores_3=2, data_quality_1=600, data_quality_2=500, data_quality_3=400)
BOTTLENECK = dict(cores_1=1, cores_2=6, cores_3=1, data_quality_1=800, data_quality_2=700, data_quality_3=200)


def check(name, planner, df, drain):
    print(f"\n===== {name}")
    print("[choices] from", START)
    for load in (100, 200, 300, 400, 500, 600):
        d = planner.decide(load, START)
        c = d["config"]
        print(f"  load {load:3d}: act={d['act']!s:5}  cores {c['cores_1']},{c['cores_2']},{c['cores_3']}  "
              f"quality {c['data_quality_1']},{c['data_quality_2']},{c['data_quality_3']}  "
              f"E[completion] {d['completion']:.3f} (current {d['completion_current']:.3f})  "
              f"U {d['utility']:.3f} (current {d['utility_current']:.3f})")

    odd = B.add_capacity(pd.DataFrame([BOTTLENECK]), planner.model.capacity)
    P = planner.throughput_3(200, odd)[0]
    print(f"[bottleneck] capacity_1 {odd['capacity_1'].iloc[0]:.0f} req/s, load 200: "
          f"E[throughput_3] {P @ planner.bin_value:.0f} req/s")

    pred = np.empty(len(df))
    for load, g in df.groupby(D.LOAD_COL):
        _, comp, _ = planner.score(load, g)
        pred[df.index.get_indexer(g.index)] = comp
    meas = np.minimum(df["throughput_3"].to_numpy() / df[D.LOAD_COL].to_numpy(), 1.0)
    for label, m in (("settled", ~drain), ("draining", drain)):
        print(f"[completion] {label:8s} {m.sum():6d} windows: mean predicted {pred[m].mean():.3f}, "
              f"measured {meas[m].mean():.3f}, MAE {np.abs(pred[m] - meas[m]).mean():.3f}")
    bins = pd.cut(pred[~drain], [0, .5, .8, .9, .95, .99, 1.0001])
    t = pd.DataFrame({"predicted": pred[~drain], "measured": meas[~drain], "bin": bins})
    print(t.groupby("bin", observed=True).agg(n=("measured", "size"), predicted=("predicted", "mean"),
                                               measured=("measured", "mean")).round(3).to_string())


def main(n_bins=PL.PLANNER_BINS):
    model = B.Model.load(str(DEFAULT_MODEL))
    settled = PL.Planner(model, PL.load_chain(PL.chain_path(DEFAULT_MODEL, n_bins)))
    print(f"{len(settled.cands)} candidates")

    t0 = time.perf_counter()
    for _ in range(5):
        settled.decide(300, START)
    print(f"[speed] {(time.perf_counter() - t0) / 5 * 1000:.0f} ms per decision")

    frames = [model.load_frame(DATA / f) for f in model.meta["files"]]
    df, drain = PL.settled_windows(frames)
    keep = (df[D.LOAD_COL] > 0).to_numpy()
    df, drain = df[keep], drain[keep]

    check("model's own chain", PL.Planner(model), df, drain)
    check(f"settled chain (planner.py), {len(settled.edges) + 1} bins", settled, df, drain)


if __name__ == "__main__":
    main((int(sys.argv[1]) or None) if len(sys.argv) > 1 else PL.PLANNER_BINS)
