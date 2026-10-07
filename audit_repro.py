"""
AUDIT ONLY -- independent reproduction of the headline numbers of
docs/2026-10-07_fixes_ablation_tabpfn.md using the repo's own harness.
Writes results to audit_out/ ; touches nothing else.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "dbn"))
import ablation as A

OUT = Path(__file__).resolve().parent / "audit_out"
OUT.mkdir(exist_ok=True)

CSVS = {
    "periodic": "data/dbn_wide_20260313_192652_seed783122.csv",
    "oial": "data/dbn_wide_20260316_105011_seed783122.csv",
    "unpred": "data/dbn_wide_20260319_094316_seed783122.csv",
}

CFGS = {
    "ref_1s_20b": dict(granularity=1, n_bins=20),
    "ref_1s_50b": dict(granularity=1, n_bins=50),
    "ref_30s_4b": dict(granularity=30, n_bins=4),
    "ref_30s_20b": dict(granularity=30, n_bins=20),
    "best_1s_20b": dict(granularity=1, n_bins=20, ar_order=2, other_lags=2, on_change="chain"),
    "best_30s_4b": dict(granularity=30, n_bins=4, ar_order=2, other_lags=2, on_change="chain"),
    "bic_bdeu_1s_20b": dict(granularity=1, n_bins=20, engine="hc-bic", prior="bdeu"),
    "nofr_1s_20b": dict(granularity=1, n_bins=20, cross_run=False),
}

rows, runs_all = [], []
for dname, csv in CSVS.items():
    for cname, kw in CFGS.items():
        cfg = A.make_cfg(**kw)
        t0 = time.time()
        try:
            r, t = A.run_config(csv, cfg)
        except Exception as e:
            print(f"{dname} {cname} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        rr = pd.DataFrame(r)
        rr.insert(0, "dataset", dname)
        rr.insert(1, "cfgname", cname)
        rows.append(rr)
        for tab in t:
            tab.insert(0, "dataset", dname)
            tab.insert(1, "cfgname", cname)
        runs_all.append(pd.concat(t, ignore_index=True))
        m = rr.groupby("model")[["acc_all", "logloss_all", "mae_all", "maeh_all", "acc_change"]].mean()
        print(f"{dname:9s} {cname:18s} ({time.time()-t0:5.0f}s) "
              f"DBN acc={m.loc['model','acc_all']:.4f} ll={m.loc['model','logloss_all']:.4f} "
              f"acc@change={m.loc['model','acc_change']:.3f} | "
              f"AR acc={m.loc['ar','acc_all']:.4f} ll={m.loc['ar','logloss_all']:.4f} | "
              f"pers acc={m.loc['persistence','acc_all']:.4f}", flush=True)
        pd.concat(rows, ignore_index=True).to_csv(OUT / "audit_folds.csv", index=False)
        pd.concat(runs_all, ignore_index=True).to_csv(OUT / "audit_runs.csv.gz", index=False)

folds = pd.concat(rows, ignore_index=True)
runs = pd.concat(runs_all, ignore_index=True)

# pooled over the three datasets, as the report does
summ = []
for cname in folds.cfgname.unique():
    f = folds[folds.cfgname == cname]
    r = runs[runs.cfgname == cname]
    m = r[r.model == "model"]
    ar = r[r.model == "ar"]
    per = r[r.model == "persistence"]
    cmp_ar = A.compare_to_reference(r, "ref", "model", "ref", "ar", n_boot=10000)
    summ.append(dict(cfgname=cname,
                     acc=m.correct.sum() / m.n.sum(), logloss=m.ll.sum() / m.n.sum(),
                     mae_ev=m.ae.sum() / m.n.sum(), maeh=m.aeh.sum() / m.n.sum(),
                     acc_ar=ar.correct.sum() / ar.n.sum(), logloss_ar=ar.ll.sum() / ar.n.sum(),
                     maeh_ar=ar.aeh.sum() / ar.n.sum(),
                     acc_pers=per.correct.sum() / per.n.sum(),
                     acc_change=m.correct_change.sum() / m.n_change.sum(),
                     acc_xrun=m.correct_xrun.sum() / m.n_xrun.sum(),
                     n_runs=cmp_ar["n_runs"],
                     d_acc=cmp_ar["d_acc"], d_acc_lo=cmp_ar["d_acc_lo"], d_acc_hi=cmp_ar["d_acc_hi"]))
out = pd.DataFrame(summ)
out.to_csv(OUT / "audit_summary.csv", index=False)
print(out.to_string(index=False))
