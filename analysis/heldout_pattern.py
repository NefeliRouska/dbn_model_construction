"""
heldout_pattern.py -- is the recommended configuration an artefact of being
chosen on the data it is scored on?

The option studies (ablation.py) chose the modelling options on all three load
patterns, and study_recommended then scored the result on the same three. Here
the choice is made WITHOUT one pattern and scored on that pattern only:

    round   choose the options on            score on
    1       periodic + once-in-a-lifetime    unpredictable
    2       periodic + unpredictable         once-in-a-lifetime
    3       once-in-a-lifetime + unpredictable  periodic

Choosing = coordinate ascent from ablation.DEFAULTS: for each option in
OPTIONS, every value is scored with the other options fixed and the best one is
kept; passes repeat until no option changes (at most MAX_PASSES). The score is
cross-validated accuracy on the bins, pooled over the test runs of the two
choosing patterns (ties within TIE broken by log-loss, then by keeping the
current value). Everything else (folds, bins, metrics) is exactly ablation.py's.

On the held-out pattern the chosen configuration is scored next to the AR table
and to ablation.RECOMMENDED, with a 95% run-level bootstrap interval of the
difference to AR (one pattern: the run is the unit; no schedule clustering).

Only reads the data and imports ablation.py / data.py; writes to
results/heldout/. Nothing in the existing pipeline is changed.

Usage:
    python analysis/heldout_pattern.py                 # throughput_3, all three rounds
    python analysis/heldout_pattern.py --quick         # small option set, one pass (smoke test)
    python analysis/heldout_pattern.py --target avg_p_latency_1
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
import ablation as A  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "heldout"

PATTERNS = {
    "periodic": "dbn_wide_20260313_192652_seed783122.csv",
    "once-in-a-lifetime": "dbn_wide_20260316_105011_seed783122.csv",
    "unpredictable": "dbn_wide_20260319_094316_seed783122.csv",
}

# The options study_recommended's choices were made over. Values that change
# the task itself (bins, resolution, horizon, the target's discretisation) are
# fixed: they change what accuracy measures, not how well it is reached.
# "reconf" bundles the two options that only make sense together.
RECONF = {"same": dict(capacity=False, on_change="same"),
          "chain": dict(capacity=False, on_change="chain"),
          "capchain": dict(capacity=True, on_change="capchain")}
OPTIONS = {
    "ar_order": [1, 2, 3],
    "other_lags": [1, 2, 3],
    "velocity": ["none", "binned"],
    "pool": ["core", "local", "all"],
    "engine": ["cv-forward", "hc-bic", "hc-aic", "hc-bdeu", "cmi", "lasso", "dynotears"],
    "prior": ["backoff", "bdeu"],
    "max_parents": [2, 3, 4, 6],
    "reconf": list(RECONF),
}
QUICK_OPTIONS = {"other_lags": [1, 2], "prior": ["backoff", "bdeu"], "reconf": ["same", "capchain"]}
MAX_PASSES = 3
TIE = 1e-4


def make(choice, target):
    """ablation config from a {option: value} choice."""
    kw = {k: v for k, v in choice.items() if k != "reconf"}
    kw.update(RECONF[choice["reconf"]])
    disc = "quantile" if target.startswith("buffer") else "uniform"
    return A.make_cfg(target=target, disc=disc, **kw)


def pooled(tables, model):
    t = tables[tables.model == model]
    n = t.n.sum()
    return t.correct.sum() / n, t.ll.sum() / n


_cache = {}


def score(csvs, choice, target):
    """Pooled cross-validated accuracy and log-loss of one choice over csvs (memoised)."""
    key = (tuple(csvs), tuple(sorted(choice.items())), target)
    if key not in _cache:
        cfg = make(choice, target)
        tables = pd.concat([pd.concat(A.run_config(c, cfg)[1]) for c in csvs], ignore_index=True)
        _cache[key] = pooled(tables, "model")
    return _cache[key]


def choose(csvs, target, options, log):
    """Coordinate ascent over options, scored on csvs only."""
    choice = {"ar_order": A.DEFAULTS["ar_order"], "other_lags": A.DEFAULTS["other_lags"],
              "velocity": A.DEFAULTS["velocity"], "pool": A.DEFAULTS["pool"],
              "engine": A.DEFAULTS["engine"], "prior": A.DEFAULTS["prior"],
              "max_parents": A.DEFAULTS["max_parents"], "reconf": "same"}
    for p in range(1, MAX_PASSES + 1):
        changed = False
        for opt, values in options.items():
            current = score(csvs, choice, target)
            best_v, best = choice[opt], current
            for v in values:
                if v == choice[opt]:
                    continue
                s = score(csvs, {**choice, opt: v}, target)
                log.append(dict(pass_=p, option=opt, value=str(v), acc=s[0], logloss=s[1]))
                if s[0] > best[0] + TIE or (abs(s[0] - best[0]) <= TIE and s[1] < best[1]):
                    best_v, best = v, s
            if best_v != choice[opt]:
                print(f"    pass {p}: {opt} {choice[opt]} -> {best_v}  (acc {current[0]:.4f} -> {best[0]:.4f})",
                      flush=True)
                choice[opt] = best_v
                changed = True
        if not changed:
            break
    return choice


def evaluate(csv, cfg):
    """Held-out scores of one configuration on one pattern."""
    rows, tables = A.run_config(csv, cfg)
    t = pd.concat(tables, ignore_index=True)
    acc, ll = pooled(t, "model")
    acc_ar, _ = pooled(t, "ar")
    f = pd.DataFrame(rows)
    m = f[f.model == "model"]
    cmp = A.compare_to_reference(t, A.cfg_id(cfg), "model", A.cfg_id(cfg), "ar", n_boot=10000)
    return dict(acc=acc, logloss=ll, acc_ar=acc_ar, acc_change=(m.acc_change * m.n_change).sum() / m.n_change.sum(),
                d_acc_vs_ar=cmp["d_acc"], d_acc_lo=cmp["d_acc_lo"], d_acc_hi=cmp["d_acc_hi"], n_runs=cmp["n_runs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--target", default="throughput_3")
    ap.add_argument("--quick", action="store_true", help="small option set, one pass: a smoke test")
    args = ap.parse_args()
    options = QUICK_OPTIONS if args.quick else OPTIONS
    global MAX_PASSES
    MAX_PASSES = 1 if args.quick else MAX_PASSES
    files = {p: str(Path(args.data_dir) / f) for p, f in PATTERNS.items()}
    tag = (f"heldout_{args.target}{'_quick' if args.quick else ''}_run{datetime.now():%Y%m%d_%H%M%S}"
           f"_g{A._git_version()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results, trace = [], []
    for held in PATTERNS:
        chosen_on = [p for p in PATTERNS if p != held]
        print(f"\n=== round: choose on {' + '.join(chosen_on)}, score on {held}", flush=True)
        t0, log = time.perf_counter(), []
        choice = choose([files[p] for p in chosen_on], args.target, options, log)
        trace += [dict(held_out=held, **r) for r in log]
        chosen = make(choice, args.target)
        rec = A.recommended_cfg(args.target, numeric=False)
        r_ch, r_rec = evaluate(files[held], chosen), evaluate(files[held], rec)
        results.append(dict(held_out=held, chosen_on=" + ".join(chosen_on),
                            chosen=", ".join(f"{k}={v}" for k, v in choice.items()),
                            same_as_recommended=A.cfg_id(chosen) == A.cfg_id(rec),
                            **r_ch, rec_acc=r_rec["acc"], rec_logloss=r_rec["logloss"],
                            minutes=(time.perf_counter() - t0) / 60))
        print(f"    chosen: {results[-1]['chosen']}")
        print(f"    on {held}: chosen acc={r_ch['acc']:.4f} (AR {r_ch['acc_ar']:.4f}, gain "
              f"{r_ch['d_acc_vs_ar']:+.4f} [{r_ch['d_acc_lo']:+.4f}, {r_ch['d_acc_hi']:+.4f}])  |  "
              f"recommended acc={r_rec['acc']:.4f}", flush=True)
        pd.DataFrame(results).to_csv(OUT_DIR / f"{tag}.csv", index=False)
        pd.DataFrame(trace).to_csv(OUT_DIR / f"{tag}_trace.csv", index=False)
    print(f"\nSaved {OUT_DIR / (tag + '.csv')}\n      {OUT_DIR / (tag + '_trace.csv')}")


if __name__ == "__main__":
    main()
