"""
whatif_tabpfn_capacity.py -- the unseen-configuration test of whatif_study.py,
with TabPFN given the SAME inputs as the capacity BN.

In whatif_study.py TabPFN sees (rps, cores_i, data_quality_i), while the best
BN sees (rps, capacity_i): the comparison mixes a better model with a better
input. This script adds the missing arm:

    tabpfn (capacities)          TabPFNRegressor on (rps, capacity_1..3)
    tabpfn (cores, quality)      TabPFNRegressor on (rps, cores_i, data_quality_i), as in whatif_study.py

and, for a self-contained table on exactly the same test rows,

    capacity chain, noisy-min prior   the best BN of whatif_study.py
    boosting (capacities)             gradient boosting on (rps, capacity_1..3)

Folds, aggregation (30 s rows), capacity rate (learned on each fold's training
runs), bins, the 10 000-sample TabPFN context and the scoring are those of
whatif_study.py, whose functions are imported, not copied. whatif_study.py
itself is not changed; results go to results/whatif_capacity/.

Needs the tabpfn package and, for v3.5, the Prior Labs licence key
(export TABPFN_TOKEN=...). Meant for the GPU workstation.

Usage:
    python analysis/whatif_tabpfn_capacity.py --csv data/dbn_wide_*.csv [--tabpfn v3.5]
"""
import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation as A  # noqa: E402
import data as D  # noqa: E402
import whatif_study as W  # noqa: E402

RESULTS_DIR = A.REPO_ROOT / "results" / "whatif_capacity"


def run_one(csv, args):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    df = D.aggregate(D.load_wide(csv, "throughput_3", verbose=False), args.granularity)
    stem = Path(csv).stem.replace("dbn_wide_", "")
    rows = []
    for fi, (train, test) in enumerate(D.make_temporal_folds(df, k=5, scheme="expanding")):
        rate, line = D.fit_capacity_rate(train)
        train, test = D.add_capacity(train, rate, line), D.add_capacity(test, rate, line)
        seen = set(map(tuple, train[W.CONFIG[1:]].drop_duplicates().to_numpy()))
        new_cfg = np.array([tuple(r) not in seen for r in test[W.CONFIG[1:]].to_numpy()])

        pooled = np.concatenate([train[c].to_numpy() for c in [D.LOAD_COL] + W.TARGETS])
        edges = np.linspace(pooled.min(), pooled.max(), args.n_bins + 1)[1:-1]
        centre = {}
        for v in W.TARGETS:
            c = W.codes_for(train[v].to_numpy(), edges)
            s = np.bincount(c, weights=train[v].to_numpy(), minlength=args.n_bins)
            k = np.bincount(c, minlength=args.n_bins)
            mid = np.r_[edges[0], (edges[1:] + edges[:-1]) / 2, edges[-1]]
            centre[v] = np.where(k > 0, s / np.maximum(k, 1), mid)

        preds = {}
        capm = W.chain(train, test, edges, lambda i: [(f"capacity_{i}", "flow")], args.n_bins, noisy_min=True)
        preds["capacity chain, noisy-min prior"] = {v: (P, P @ centre[v]) for v, P in capm.items()}
        preds["boosting (capacities)"] = {
            v: (None, HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                                    random_state=0).fit(train[W.CAPS], train[v]).predict(test[W.CAPS]))
            for v in W.TARGETS}
        ctx = train if len(train) <= 10000 else train.sample(10000, random_state=0)
        for name, cols in ((f"tabpfn {args.tabpfn} (capacities)", W.CAPS),
                           (f"tabpfn {args.tabpfn} (cores, quality)", W.CONFIG)):
            preds[name] = {}
            for v in W.TARGETS:
                m = TabPFNRegressor.create_default_for_version(ModelVersion(args.tabpfn))
                m.fit(ctx[cols].to_numpy(), ctx[v].to_numpy())
                # median, as in whatif_study.py (absolute error; skewed throughput)
                preds[name][v] = (None, m.predict(test[cols].to_numpy(), output_type="median"))

        for model, per_target in preds.items():
            for v, (P, point) in per_target.items():
                y = test[v].to_numpy()
                yc = W.codes_for(y, edges)
                pred_bin = P.argmax(1) if P is not None else W.codes_for(point, edges)
                d = pd.DataFrame({
                    "run_id": test["run_id"].to_numpy(), "new_config": new_cfg, "n": 1.0,
                    "ae": np.abs(point - y), "correct": (pred_bin == yc).astype(float),
                }).groupby(["run_id", "new_config"], as_index=False).sum()
                d.insert(0, "target", v)
                d.insert(0, "model", model)
                d.insert(0, "fold", fi + 1)
                d.insert(0, "dataset", stem)
                rows.append(d)
        print(f"[{stem} fold {fi + 1}] test rows {len(test)}, unseen configuration: {new_cfg.mean():.0%}", flush=True)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--tabpfn", default="v3.5")
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--granularity", type=int, default=30)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    tag = f"whatif_capacity_{args.tabpfn}_run{datetime.now():%Y%m%d_%H%M%S}_g{A._git_version()}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out = pd.concat([run_one(c, args) for c in args.csv], ignore_index=True)
    out.to_csv(RESULTS_DIR / f"{tag}_runs.csv.gz", index=False)
    new = out[out.new_config]
    g = new.groupby(["target", "model"])[["n", "ae", "correct"]].sum()
    summary = pd.DataFrame({"acc": g.correct / g.n, "mae": g.ae / g.n, "n": g.n}).reset_index()
    summary.to_csv(RESULTS_DIR / f"{tag}_summary_unseen.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", lambda x: f"{x:.3f}"):
        print("\n== unseen configurations only, pooled over the datasets given ==")
        print(summary.to_string(index=False))
    print(f"\nSaved {RESULTS_DIR / (tag + '_runs.csv.gz')}\n      {RESULTS_DIR / (tag + '_summary_unseen.csv')}")


if __name__ == "__main__":
    main()
