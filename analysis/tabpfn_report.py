"""
tabpfn_report.py -- summary of analysis/tabpfn_baseline.py over all datasets.

Per setting (granularity, horizon, bins) and model: accuracy on the bins,
log-loss, MAE of the model's continuous forecast and accuracy on the steps
where the target leaves its bin; then the paired difference between the DBN
("model") and every other model on the same test samples, with a 95% interval
from a bootstrap over runs.

Usage:
    python analysis/tabpfn_report.py [--version v2]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation as A  # noqa: E402
from ablation_report import newest  # noqa: E402

RESULTS_DIR = A.REPO_ROOT / "results" / "tabpfn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    args = ap.parse_args()
    study = f"tabpfn-{args.version}"

    runs = []
    for dataset, f in newest(study, "runs.csv.gz", RESULTS_DIR).items():
        d = pd.read_csv(f)
        d["run_id"] = dataset + ":" + d["run_id"].astype(str)
        d["dataset"] = dataset
        runs.append(d)
    runs = pd.concat(runs, ignore_index=True)

    g = runs.groupby(["config", "model"])
    table = pd.DataFrame({
        "datasets": g["dataset"].nunique(),
        "n_test": g["n"].sum().astype(int),
        "acc": g["correct"].sum() / g["n"].sum(),
        "logloss": g["ll"].sum() / g["n"].sum(),
        "mae": g["aeh"].sum() / g["n"].sum(),
        "acc_change": g["correct_change"].sum() / g["n_change"].sum(),
    }).reset_index().rename(columns={"config": "setting"})
    # hgb gives a point forecast: its log-loss is not a probabilistic score
    table.loc[table.model.isin(["hgb", "hgb_ctx", "persistence"]), "logloss"] = float("nan")

    rows = []
    for setting in table.setting.unique():
        r = runs[runs.config == setting]
        for other in sorted(set(r.model) - {"model"}):
            cmp = A.compare_to_reference(r, setting, "model", setting, other, n_boot=10000)
            rows.append({"setting": setting, "dbn_minus": other, **cmp})
    diffs = pd.DataFrame(rows)

    table.to_csv(RESULTS_DIR / f"summary_{study}.csv", index=False)
    diffs.to_csv(RESULTS_DIR / f"summary_{study}_dbn_vs_others.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))
        print()
        print(diffs[["setting", "dbn_minus", "n_runs", "d_acc", "d_acc_lo", "d_acc_hi", "p_acc",
                     "d_logloss", "d_logloss_lo", "d_logloss_hi", "d_maeh", "d_maeh_lo", "d_maeh_hi"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
