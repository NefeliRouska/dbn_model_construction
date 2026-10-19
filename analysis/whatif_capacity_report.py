"""
whatif_capacity_report.py -- 95% intervals for the unseen-configuration test of
analysis/whatif_tabpfn_capacity.py.

For every throughput, the capacity BN (noisy-min prior) is compared with each
other model on the test rows whose configuration never occurred in training,
pooled over the load patterns. Paired bootstrap, 10 000 resamples; the unit is
the position in the configuration schedule (run k of every pattern is drawn
together, as in ablation_report.py). Positive accuracy difference / negative
MAE difference = the capacity BN is better.

Reads the newest results/whatif_capacity/*_runs.csv.gz; writes
<same name>_intervals.csv next to it.

Usage:
    python analysis/whatif_capacity_report.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = "capacity chain, noisy-min prior"
N_BOOT = 10000


def main():
    path = sorted(glob.glob(str(ROOT / "results/whatif_capacity/*_runs.csv.gz")))[-1]
    d = pd.read_csv(path)
    d = d[d.new_config]
    rows = []
    for target, g in d.groupby("target"):
        per = g.groupby(["model", "run_id"])[["n", "correct", "ae"]].sum()   # run_id = schedule position
        ref = per.loc[REFERENCE]
        for model in sorted(set(g.model) - {REFERENCE}):
            other = per.loc[model]
            pos = ref.index.intersection(other.index)
            a, b = ref.loc[pos], other.loc[pos]
            draws = np.random.default_rng(0).integers(0, len(pos), size=(N_BOOT, len(pos)))
            out = {"target": target, "model": model, "positions": len(pos)}
            for col, name in (("correct", "acc"), ("ae", "mae")):
                va, vb, na, nb = a[col].to_numpy(), b[col].to_numpy(), a.n.to_numpy(), b.n.to_numpy()
                point = va.sum() / na.sum() - vb.sum() / nb.sum()
                boot = va[draws].sum(1) / na[draws].sum(1) - vb[draws].sum(1) / nb[draws].sum(1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                out.update({f"{name}_bn": va.sum() / na.sum(), f"{name}_other": vb.sum() / nb.sum(),
                            f"d_{name}": point, f"d_{name}_lo": lo, f"d_{name}_hi": hi})
            rows.append(out)
    res = pd.DataFrame(rows)
    out_path = path.replace("_runs.csv.gz", "_intervals.csv")
    res.to_csv(out_path, index=False)
    with pd.option_context("display.width", 220):
        print(f"capacity BN minus each model, unseen configurations ({Path(path).name})\n")
        print(res.round(3).to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
