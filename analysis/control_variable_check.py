"""
control_variable_check.py

Checks whether changes in control variables (cores_*, data_quality_*)
come with changes in throughput_3 more often than baseline.

Rows are chronological and a configuration only changes between two runs, so
"a control changed" means: this row is the first of a new run. The question is
whether throughput_3 is in a different bin on that row than on the row before
it (the last one of the previous run).

If control changes predict throughput changes, that's a real
early-warning signal persistence structurally cannot use -- persistence
only ever looks at throughput_3's own past value.

Usage:
    python control_variable_check.py --csv path/to/dbn_wide_*.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
import data as D  # noqa: E402

TARGET = "throughput_3"
MODELING_GRANULARITY_SEC = 30


def load_and_aggregate(csv_path):
    """Chronological rows, aggregated inside runs (see dbn/data.py)."""
    df = D.load_wide(csv_path, TARGET, verbose=False)
    return D.aggregate(df, MODELING_GRANULARITY_SEC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    raw = load_and_aggregate(args.csv)

    control_cols = [c for c in raw.columns
                    if c.lower().startswith(("cores_", "data_quality_"))]

    if not control_cols:
        print("No cores_* or data_quality_* columns found in this CSV.")
        return

    print(f"[DATA] {len(raw)} rows after aggregation")
    print(f"[CONTROL COLUMNS FOUND] {control_cols}\n")

    # "acted" = did ANY control variable change value from the
    # previous row to this one
    control_diff = raw[control_cols].diff().abs().sum(axis=1)
    acted = (control_diff > 1e-9)

    # discretize throughput_3 into 4 bins, same as the main pipeline,
    # purely so "did it change" means "did it cross a bin boundary",
    # not "did it move by 0.0001"
    target_bins = pd.qcut(raw[TARGET], 4, labels=False, duplicates="drop")
    # did the target change bin between the previous row and this one, i.e.
    # over the same step as the (possible) control change
    changes_next = (target_bins.shift(1) != target_bins)

    # drop the first row (no previous row to compare against)
    df_check = pd.DataFrame({
        "acted": acted,
        "changes_next": changes_next,
    }).iloc[1:]

    summary = df_check.groupby("acted")["changes_next"].agg(["mean", "count"])
    summary.index = summary.index.map({False: "no control change", True: "control changed"})

    print("=== RESULT ===")
    print(summary.rename(columns={"mean": "P(throughput changes bin on this step)", "count": "n_rows"}))

    rate_no_action = summary.loc["no control change", "mean"]
    rate_action = summary.loc["control changed", "mean"] if "control changed" in summary.index else float("nan")

    print()
    if np.isnan(rate_action):
        print("No rows with a control change found -- cannot compare.")
    else:
        ratio = rate_action / rate_no_action if rate_no_action > 0 else float("inf")
        print(f"Baseline change rate (no control action): {rate_no_action:.1%}")
        print(f"Change rate across a control action:       {rate_action:.1%}")
        print(f"Ratio: {ratio:.2f}x")
        print()
        if ratio > 2:
            print("STRONG signal: control changes are followed by throughput "
                  "changes much more often than baseline. This is a real, "
                  "usable early-warning signal persistence cannot access.")
        elif ratio > 1.3:
            print("MODEST signal: some elevated change rate after control "
                  "actions, worth investigating further but not dramatic.")
        else:
            print("NO signal: control changes are not followed by "
                  "throughput changes any more often than baseline. "
                  "This path does not look promising.")

    # Also check per individual control column, in case one dominates
    print("\n=== PER-COLUMN BREAKDOWN ===")
    for col in control_cols:
        col_diff = raw[col].diff().abs()
        col_acted = (col_diff > 1e-9).iloc[1:]
        if col_acted.sum() == 0:
            print(f"{col}: never changes in this data, skipping")
            continue
        rate = changes_next.iloc[1:][col_acted].mean()
        n = col_acted.sum()
        print(f"{col}: change rate after action = {rate:.1%} (n={n} actions) "
              f"vs baseline {rate_no_action:.1%}")


if __name__ == "__main__":
    main()