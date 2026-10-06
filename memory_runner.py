"""
memory_runner.py

Runs YOUR pipeline unchanged (run_one -> feature selection, discretizer,
build_dbn_model_2s, evaluate, persistence / AR-DBN / Static BN SI) for ONE
fixed configuration, and only flips your existing memory switches between
conditions. Numbers are therefore directly comparable to your logs.

Fixed: markov feature selection, one score, one discretizer, one K.
Varied: N_LAGS, USE_CONTROL_FLAG, USE_VELOCITY_FEATURES.

Usage:
  python memory_runner.py --csv share/metrics/dbn_wide_XXXX.csv
  python memory_runner.py --csv ... --n_bins 6 8 --horizon 3
"""
import argparse
import contextlib
import importlib
import io
import sys

import numpy as np
import pandas as pd

TARGET = "throughput_3"

# name, N_LAGS, USE_CONTROL_FLAG, USE_VELOCITY_FEATURES
CONDITIONS = [
    ("A0 self only (~AR-DBN)",                       0, False, False),
    ("A1 +lag1",                                     1, False, False),
    ("A2 +lag1,lag2",                                2, False, False),
    ("A3 +lag1..lag3",                               3, False, False),
    ("B1 +ctrl",                                     0, True,  False),
    ("B2 +lag1 +ctrl  (reference)",                  1, True,  False),
    ("B3 +lag1,lag2 +ctrl",                          2, True,  False),
    ("F1 B2 + lag1,vel1 of other selected vars",     1, True,  True),
]
REF = "B2 +lag1 +ctrl  (reference)"


def import_pipeline(module_name, csv_path):
    saved = sys.argv
    sys.argv = [saved[0], "--csv", csv_path]   # the pipeline parses argv at import
    try:
        P = importlib.import_module(module_name)
    finally:
        sys.argv = saved
    return P


def mean_or_nan(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def run_bins(P, raw, folds, args, nb):
    per = {name: [None] * len(folds) for name, *_ in CONDITIONS}
    nparents = {name: [] for name, *_ in CONDITIONS}
    base = [None] * len(folds)

    print(f"\n[SETUP] bins={nb} disc={args.disc} score={args.score} K={args.k} "
          f"horizon={args.horizon} folds={len(folds)} (fs=markov)")

    for fi, (tr, te) in enumerate(folds):
        for name, nl, ctrl, vel in CONDITIONS:
            P.N_LAGS = nl
            P.USE_CONTROL_FLAG = ctrl
            P.USE_VELOCITY_FEATURES = vel
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    out = P.run_one(raw, "markov", args.disc, args.score,
                                    args.k, nb, train_raw=tr, test_raw=te)
            except Exception as e:
                print(f"  fold {fi+1} | {name}: FAILED: {e}")
                if args.verbose:
                    print(buf.getvalue()[-1500:])
                continue
            res = out[0]
            per[name][fi] = dict(ll=res["log_loss"], acc=res["accuracy"], f1=res["f1"])
            try:
                nparents[name].append(len(out[2].get_parents(f"{TARGET}_t1")))
            except Exception:
                pass
            if base[fi] is None:   # baselines do not depend on the memory switches
                base[fi] = dict(pers_acc=out[3], pers_f1=out[4],
                                ar_acc=out[7], ar_f1=out[8], ar_ll=out[11],
                                si_acc=out[12], si_f1=out[13], si_ll=out[16])
        got = sum(per[n][fi] is not None for n, *_ in CONDITIONS)
        print(f"  fold {fi+1}/{len(folds)} done ({got}/{len(CONDITIONS)} conditions ok)")

    rows = []
    for name, *_ in CONDITIONS:
        recs = per[name]
        ok = [i for i, r in enumerate(recs) if r is not None and base[i] is not None]
        if not ok:
            continue
        ll = np.array([recs[i]["ll"] for i in ok])
        ar = np.array([base[i]["ar_ll"] for i in ok])
        ref = np.array([per[REF][i]["ll"] if per[REF][i] is not None else np.nan for i in ok])
        d_ar, d_ref = ll - ar, ll - ref
        rows.append(dict(
            condition=name,
            n_parents=mean_or_nan(nparents[name]),
            log_loss=float(ll.mean()), ll_std=float(ll.std(ddof=1)) if len(ll) > 1 else 0.0,
            acc=mean_or_nan([recs[i]["acc"] for i in ok]),
            f1=mean_or_nan([recs[i]["f1"] for i in ok]),
            d_vs_AR=float(d_ar.mean()), folds_better_AR=f"{int((d_ar < 0).sum())}/{len(ok)}",
            d_vs_REF=float(np.nanmean(d_ref)) if not np.all(np.isnan(d_ref)) else float("nan"),
            folds_better_REF=f"{int((d_ref < 0).sum())}/{int((~np.isnan(d_ref)).sum())}",
        ))
    out_df = pd.DataFrame(rows)

    b = [x for x in base if x is not None]
    extra = [
        dict(condition="PERSISTENCE (your code)", acc=mean_or_nan([x["pers_acc"] for x in b]),
             f1=mean_or_nan([x["pers_f1"] for x in b])),
        dict(condition="AR-DBN (your code)", log_loss=mean_or_nan([x["ar_ll"] for x in b]),
             acc=mean_or_nan([x["ar_acc"] for x in b]), f1=mean_or_nan([x["ar_f1"] for x in b])),
        dict(condition="STATIC BN SI (your code; same-moment, NOT a forecast)",
             log_loss=mean_or_nan([x["si_ll"] for x in b]),
             acc=mean_or_nan([x["si_acc"] for x in b]), f1=mean_or_nan([x["si_f1"] for x in b])),
    ]
    out_df = pd.concat([out_df, pd.DataFrame(extra)], ignore_index=True)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== RESULTS bins={nb} horizon={args.horizon} "
          f"(d_vs_* < 0 means lower log-loss than the reference) ===")
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    fname = f"memory_runner_h{args.horizon}_b{nb}_{args.disc}_{args.score}.csv"
    out_df.to_csv(fname, index=False)
    print(f"\nSaved {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pipeline_module", default="test_DBN_new_era_AR_fixed")
    ap.add_argument("--n_bins", type=int, nargs="+", default=[6])
    ap.add_argument("--disc", default="classic_uniform")
    ap.add_argument("--score", default="bic")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    P = import_pipeline(args.pipeline_module, args.csv)
    P.TARGET = TARGET
    P.HORIZON = args.horizon
    P.TARGET_PARENTS_ONLY = True

    raw = P.load_and_clean(args.csv, TARGET)
    raw = P.aggregate_to_modeling_granularity(raw)
    folds = P.make_temporal_folds(raw, k=args.folds)
    print(f"[DATA] rows after aggregation: {len(raw)}")
    for nb in args.n_bins:
        run_bins(P, raw, folds, args, nb)


if __name__ == "__main__":
    main()