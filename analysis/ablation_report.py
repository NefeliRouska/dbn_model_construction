"""
ablation_report.py -- effect and significance of every option tested by
dbn/ablation.py, pooled over the datasets.

For one study (e.g. `ofat`) it reads the newest result files of every dataset
in results/ablation/ and writes one row per configuration:

    acc, logloss, maeh        the configuration's model, mean over test samples
    acc_change                accuracy on the steps where the target leaves its bin
    d_*_vs_ar                 difference to the AR(1) table scored on the same
                              bins and samples (the "does the DBN beat AR" number)
    d_*_vs_ref                difference to the reference configuration
                              (only meaningful when the bins are the same)
    *_lo / *_hi / p_*         95% interval and two-sided p-value of that
                              difference from a paired bootstrap. The unit that
                              is resampled is the position in the configuration
                              schedule (the three datasets share one schedule,
                              so run k of each dataset is drawn together)
    p_holm_*                  the p-values corrected for the number of
                              configurations compared (Holm)

Usage:
    python analysis/ablation_report.py --study ofat
    python analysis/ablation_report.py --study bins_granularity --ref-config self
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
import ablation as A  # noqa: E402


N_BOOT = 10000


def schedule_point(run_id):
    """'<dataset>:<run>' -> run: the position in the configuration schedule the datasets share."""
    return str(run_id).split(":", 1)[-1]


def newest(study, suffix, directory):
    """Newest file of every dataset for one study."""
    out = {}
    for f in sorted(glob.glob(str(directory / f"{study}_*_{suffix}"))):
        dataset = Path(f).name[len(study) + 1:].split("_run")[0]
        out[dataset] = f
    return out


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load(study, directory):
    folds, runs = [], []
    for dataset, f in newest(study, "folds.csv", directory).items():
        d = pd.read_csv(f)
        d.insert(0, "dataset", dataset)
        folds.append(d)
    for dataset, f in newest(study, "runs.csv.gz", directory).items():
        d = pd.read_csv(f)
        d["run_id"] = dataset + ":" + d["run_id"].astype(str)   # runs of different datasets stay distinct
        d.insert(0, "dataset", dataset)
        runs.append(d)
    return pd.concat(folds, ignore_index=True), pd.concat(runs, ignore_index=True)


def report(study, directory, ref_config="ref", model="model"):
    folds, runs = load(study, directory)
    if "error" in folds.columns:
        failed = folds[folds["error"].notna()]
        for r in failed.itertuples():
            print(f"[FAILED] {r.dataset} {r.config}: {r.error}")
        folds = folds[folds["error"].isna()] if len(failed) else folds
    configs = list(dict.fromkeys(folds["config"]))
    rows = []
    for cfg in configs:
        f = folds[(folds.config == cfg) & (folds.model == model)]
        r = runs[(runs.config == cfg)]
        m = r[r.model == model]
        row = {"config": cfg,
               "acc": m.correct.sum() / m.n.sum(),
               "logloss": m.ll.sum() / m.n.sum(),
               "maeh": m.aeh.sum() / m.n.sum(),
               "acc_change": m.correct_change.sum() / max(m.n_change.sum(), 1),
               "acc_xrun": m.correct_xrun.sum() / m.n_xrun.sum() if m.n_xrun.sum() else np.nan,
               "acc_ar": r[r.model == "ar"].correct.sum() / r[r.model == "ar"].n.sum(),
               "acc_persistence": r[r.model == "persistence"].correct.sum() / r[r.model == "persistence"].n.sum(),
               "maeh_ar": r[r.model == "ar"].aeh.sum() / r[r.model == "ar"].n.sum(),
               "mae_persistence_continuous": np.average(f.mae_persistence_continuous, weights=f.n_all),
               "n_parents": f.n_parents.mean(),
               "unseen_cfg_frac": f.test_unseen_config_frac.mean(),
               "parents_last_fold": "; ".join(sorted(set(f[f.fold == f.fold.max()].parents)))}
        for ds, g in f.groupby("dataset"):
            row[f"acc[{ds[:8]}]"] = g.acc_all.mean()
            row[f"logloss[{ds[:8]}]"] = g.logloss_all.mean()
        cmp_ar = A.compare_to_reference(r, cfg, model, cfg, "ar", n_boot=N_BOOT, cluster=schedule_point)
        row.update({f"{k}_vs_ar": v for k, v in cmp_ar.items() if k != "n_runs"})
        row["n_runs"] = cmp_ar["n_runs"]
        if ref_config != "self" and cfg != ref_config and ref_config in configs:
            cmp_ref = A.compare_to_reference(runs, cfg, model, ref_config, model, n_boot=N_BOOT,
                                             cluster=schedule_point)
            row.update({f"{k}_vs_ref": v for k, v in cmp_ref.items() if k != "n_runs"})
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in [c for c in out.columns if c.startswith("p_")]:
        ok = out[col].notna()
        out.loc[ok, col.replace("p_", "p_holm_", 1)] = holm(out.loc[ok, col])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--dir", default=str(A.RESULTS_DIR))
    ap.add_argument("--ref-config", default="ref",
                    help="configuration the others are compared with ('self' = only vs AR)")
    ap.add_argument("--model", default="model")
    args = ap.parse_args()

    out = report(args.study, Path(args.dir), args.ref_config, args.model)
    dest = Path(args.dir) / f"summary_{args.study}.csv"
    out.to_csv(dest, index=False)
    show = [c for c in ["config", "acc", "acc_ar", "logloss", "maeh", "mae_persistence_continuous",
                        "acc_change", "acc_xrun", "n_parents",
                        "d_acc_vs_ar", "d_logloss_vs_ar", "p_holm_logloss_vs_ar",
                        "d_acc_vs_ref", "d_logloss_vs_ref", "p_holm_logloss_vs_ref"] if c in out.columns]
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 60,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(out[show].to_string(index=False))
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
