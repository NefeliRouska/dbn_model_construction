"""
transfer_study.py -- does a model learned under one workload hold under another?

The three datasets were collected with the same sequence of configurations and
different load patterns (periodic, constant with one spike, random steps). For
every fold the model is trained on the training runs of dataset X and tested
on the test runs of dataset Y (bins are fitted on the training data, so on X):

    diagonal      X = Y, the usual setting
    off-diagonal  another workload, same configurations
    pooled        trained on the training runs of all three datasets

Reported per (train, test): accuracy and log-loss of the DBN and of the AR
table trained on the same data.

Usage:
    python analysis/transfer_study.py --csv data/dbn_wide_A.csv data/dbn_wide_B.csv data/dbn_wide_C.csv
"""
import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
import ablation as A  # noqa: E402
import data as D  # noqa: E402

RESULTS_DIR = A.REPO_ROOT / "results" / "transfer"
OFFSET = 100000          # run ids of dataset k are shifted by k * OFFSET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--set", nargs="*", default=["ar_order=2", "other_lags=2"], metavar="KEY=VALUE")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    names = args.names or [Path(c).stem.replace("dbn_wide_", "")[:15] for c in args.csv]

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = type(A.DEFAULTS[k])(v)
    cfg = A.make_cfg(**overrides)

    frames = []
    for k, csv in enumerate(args.csv):
        df = A.load_frame(csv, cfg["target"], cfg["granularity"]).copy()
        df["run_id"] = df["run_id"] + k * OFFSET
        frames.append(df)
    cols = [c for c in frames[0].columns if all(c in f.columns for f in frames)]
    both = pd.concat([f[cols] for f in frames], ignore_index=True)
    S = A.Samples(both, cfg["target"], cfg["horizon"], cfg["cross_run"], A.MAX_HISTORY[cfg["granularity"]])
    dataset = S.run // OFFSET
    local_run = S.run % OFFSET

    folds = D.make_temporal_folds(frames[0][D.META_COLS], k=cfg["folds"])
    rows = []
    for fi, (tr_meta, te_meta) in enumerate(folds):
        tr_runs, te_runs = tr_meta["run_id"].unique(), te_meta["run_id"].unique()
        in_tr = np.isin(local_run, tr_runs) & np.isin(S.run_i % OFFSET, tr_runs)
        in_te = np.isin(local_run, te_runs)
        sources = [(names[k], dataset == k) for k in range(len(names))] + [("pooled", np.ones(len(dataset), bool))]
        for src, src_mask in sources:
            for k, dst in enumerate(names):
                res, info = A.run_fold(S, cfg, in_tr & src_mask, in_te & (dataset == k))
                for model in ("model", "ar"):
                    summ = res[model][0]
                    rows.append({"fold": fi + 1, "train": src, "test": dst, "model": model,
                                 "n": summ["n_all"], "acc": summ["acc_all"], "logloss": summ["logloss_all"],
                                 "acc_change": summ["acc_change"], "parents": ",".join(info["parents"])})
        print(f"[fold {fi + 1}] done", flush=True)

    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"transfer_run{datetime.now():%Y%m%d_%H%M%S}_g{A._git_version()}"
    out.to_csv(RESULTS_DIR / f"{tag}.csv", index=False)

    def table(metric, model):
        d = out[out.model == model]
        return d.groupby(["train", "test"]).apply(lambda g: np.average(g[metric], weights=g["n"])) \
            .unstack("test").round(3)

    with pd.option_context("display.width", 200):
        for metric in ("acc", "logloss", "acc_change"):
            print(f"\n== DBN {metric} (rows: trained on, columns: tested on) ==")
            print(table(metric, "model").to_string())
        print("\n== AR accuracy ==")
        print(table("acc", "ar").to_string())
        print("\n== DBN accuracy minus AR accuracy ==")
        print((table("acc", "model") - table("acc", "ar")).round(3).to_string())


if __name__ == "__main__":
    main()
