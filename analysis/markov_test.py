"""
markov_test.py
==============
Standalone diagnostic script to test whether your system
approximately satisfies a first-order Markov assumption.

Run this ONCE before the DBN sweep to justify your DBN design.

What it does:
  - Loads and cleans your data the same way the sweep does (dbn/data.py)
  - Aggregates rows to match the DBN modeling granularity
  - Builds two comparable feature sets using the SAME prediction rows:
      First order:  state at time t
      Second order: state at time t AND t-1
    The state includes the target's own value (INCLUDE_TARGET_HISTORY): a
    first-order model of throughput is allowed to see the current throughput.
    t-1, t and t+1 always lie in the same run.
  - Trains a Random Forest classifier on each
  - Compares accuracy over expanding-window folds made of whole runs; the
    target's bins are fitted on the training part of each fold

Interpretation:
  - Second-order improvement < 2%  -> first-order assumption is adequate
  - Improvement 2-5%              -> weak memory effect, acknowledge as limitation
  - Improvement > 5%              -> substantial memory beyond one step

Important:
  This is an empirical diagnostic, not a formal proof of the Markov property.

Usage:
    python analysis/markov_test.py --csv data/dbn_wide_<stem>.csv [--granularity 30]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
import data as D  # noqa: E402


# ============================================================
# CONFIG
# ============================================================

TARGETS = ["throughput_1", "throughput_2", "throughput_3"]

MODELING_GRANULARITY_SEC = 30

N_BINS_TO_TEST = [4, 10, 20]

CV_FOLDS = 5

INCLUDE_TARGET_HISTORY = True

EXCLUDE_OTHER_THROUGHPUTS = True

RANDOM_STATE = 0


# ============================================================
# LOAD + CLEAN
# ============================================================

def load_and_clean(path):
    """Chronological rows with run_id / pos / timestamp bookkeeping."""
    return D.load_wide(path, verbose=False)


def aggregate(df):
    return D.aggregate(df, MODELING_GRANULARITY_SEC)


# ============================================================
# FEATURE BUILDERS
# ============================================================

def get_predictor_cols(df, target):
    return [
        c for c in D.feature_cols(df)
        if (c != target or INCLUDE_TARGET_HISTORY)
        and not (
            EXCLUDE_OTHER_THROUGHPUTS
            and c.startswith("throughput_")
            and c != target
        )
    ]


def build_features(df, target):
    """
    One row per t that has both a previous and a next row in its run.

    Returns (X_first, X_second, y, run_id):
      X_first  = state at t
      X_second = state at t and at t-1
      y        = target at t+1
    """
    cols = get_predictor_cols(df, target)
    g = df.groupby("run_id")

    X_t = df[cols]
    X_tm1 = g[cols].shift(1)
    X_tm1.columns = [f"{c}_lag1" for c in cols]
    y = g[target].shift(-1)

    ok = (X_tm1.notna().all(axis=1) & y.notna()).to_numpy()
    X1 = X_t[ok].reset_index(drop=True)
    X2 = pd.concat([X_t, X_tm1], axis=1)[ok].reset_index(drop=True)
    return X1, X2, y[ok].reset_index(drop=True), df["run_id"][ok].reset_index(drop=True)


# ============================================================
# MAIN TEST
# ============================================================

def run_markov_test(df, target, n_bins):
    X1, X2, y, run_id = build_features(df, target)

    folds = D.make_temporal_folds(pd.DataFrame({"run_id": run_id}), k=CV_FOLDS)
    scores1, scores2 = [], []

    for train_part, test_part in folds:
        train = run_id.isin(train_part["run_id"].unique()).to_numpy()
        test = run_id.isin(test_part["run_id"].unique()).to_numpy()

        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
        y_train = kbd.fit_transform(y[train].to_numpy().reshape(-1, 1)).astype(int).ravel()
        y_test = kbd.transform(y[test].to_numpy().reshape(-1, 1)).astype(int).ravel()

        for X, scores in ((X1, scores1), (X2, scores2)):
            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            clf.fit(X[train], y_train)
            scores.append(float(np.mean(clf.predict(X[test]) == y_test)))

    score1 = float(np.mean(scores1))
    score2 = float(np.mean(scores2))

    return score1, score2, score2 - score1


def interpret(improvement):
    if improvement < 0.02:
        return "HOLDS — second-order features do not meaningfully improve accuracy; first-order DBN is adequate"
    elif improvement < 0.05:
        return "BORDERLINE — second-order features improve accuracy by 2-5%; weak memory effect, acknowledge in paper"
    else:
        return "VIOLATED — second-order features improve accuracy by > 5%; memory beyond one step is substantial"


def main():
    print("=" * 60)
    print("MARKOV PROPERTY DIAGNOSTIC")
    print("=" * 60)
    print()

    global MODELING_GRANULARITY_SEC
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--granularity", type=int, default=MODELING_GRANULARITY_SEC,
                        help="seconds per row")
    args = parser.parse_args()
    MODELING_GRANULARITY_SEC = args.granularity

    raw = load_and_clean(args.csv)
    raw = aggregate(raw)

    print(f"Rows after aggregation: {len(raw)}")
    print(f"Granularity: {MODELING_GRANULARITY_SEC} seconds per row")
    print(f"CV method: {CV_FOLDS} expanding-window folds of whole runs")
    print()

    if len(raw) < CV_FOLDS + 3:
        raise ValueError(
            "Not enough rows after aggregation for the chosen number of CV folds. "
            "Reduce CV_FOLDS or use a smaller aggregation window."
        )

    results = []

    for target in TARGETS:
        if target not in raw.columns:
            print(f"[SKIP] {target} not found in data")
            print()
            continue

        print(f"Target: {target}")
        print("-" * 40)

        for n_bins in N_BINS_TO_TEST:
            try:
                score1, score2, improvement = run_markov_test(raw, target, n_bins)
                verdict = interpret(improvement)

                print(f"  n_bins={n_bins}")
                print(f"    First order  X(t):        {score1:.3f}")
                print(f"    Second order X(t), X(t-1): {score2:.3f}")
                print(f"    Improvement:              {improvement:+.3f}")
                print(f"    Verdict: {verdict}")
                print()

                results.append({
                    "target": target,
                    "n_bins": n_bins,
                    "score_first_order": score1,
                    "score_second_order": score2,
                    "improvement": improvement,
                    "verdict": verdict,
                })

            except Exception as e:
                print(f"  n_bins={n_bins}")
                print(f"    [ERROR] {e}")
                print()

    if len(results) == 0:
        print("No valid results were produced.")
        return

    out = pd.DataFrame(results)
    stem = Path(args.csv).stem.replace("dbn_wide_", "")
    out_path = (Path(__file__).resolve().parent.parent / "results"
                / f"markov_test_results_{stem}_{MODELING_GRANULARITY_SEC}s.csv")
    out.to_csv(out_path, index=False)

    print("=" * 60)
    print(f"Results saved to {out_path}")
    print()

    print("OVERALL SUMMARY")
    print("-" * 40)

    avg_improvement = float(out["improvement"].mean())
    max_improvement = float(out["improvement"].max())

    print(f"Average second-order improvement: {avg_improvement:+.3f}")
    print(f"Maximum second-order improvement: {max_improvement:+.3f}")
    print(f"Overall verdict based on average: {interpret(avg_improvement)}")


if __name__ == "__main__":
    main()