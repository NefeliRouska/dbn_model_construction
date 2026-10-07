"""
Raw long-format Prometheus dump -> wide table (one row per second).

Output columns: run_id, run_tag, timestamp, <one column per series>.

Rows are in CHRONOLOGICAL order. `run_id` numbers the measurement windows
(one per configuration applied by the orchestrator) in the order they were
executed: 0, 1, 2, ...  A new run starts whenever consecutive timestamps are
more than RUN_GAP_SEC apart, so two runs that happen to share a run_tag are
still told apart.

The file is read as a stream, so a multi-GB dump does not have to fit in
memory as a DataFrame of strings.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

# These labels often change but are not helpful for stable identity.
# Keep container_id / operation / device etc.
IGNORE_KEYS = {"__name__"}

# The orchestrator queries Prometheus with a 1 s step and waits >= 30 s
# (stabilisation) between two measurement windows.
RUN_GAP_SEC = 5.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Prometheus dump to wide DBN CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to raw Prometheus dump CSV"
    )
    return parser.parse_args()


def signature(metric_name: str, labels: dict) -> str:
    """
    Build a deterministic identity signature for a Prometheus time series:
    metric_name + sorted(label_key=label_value) after removing IGNORE_KEYS.
    """
    kept = {k: str(v) for k, v in labels.items() if k not in IGNORE_KEYS}
    items = sorted(kept.items())
    return metric_name + "|" + "|".join(f"{k}={v}" for k, v in items)


def read_long(path: Path):
    """Stream the dump into compact arrays (run_tag / series are integer-coded)."""
    tags, series = {}, {}
    tag_idx, series_idx, ts, val = [], [], [], []

    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        required = {"metric_name", "timestamp", "value", "labels_json"}
        missing = required - set(header)
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found: {header}")
        col = {name: i for i, name in enumerate(header)}
        has_run_tag = "run_tag" in col

        for row in reader:
            tag = row[col["run_tag"]] if has_run_tag else ""
            key = (row[col["metric_name"]], row[col["labels_json"]])
            tag_idx.append(tags.setdefault(tag, len(tags)))
            series_idx.append(series.setdefault(key, len(series)))
            ts.append(float(row[col["timestamp"]]))
            try:
                val.append(float(row[col["value"]]))
            except ValueError:
                val.append(np.nan)

    return (tags, series,
            np.asarray(tag_idx, dtype=np.int32), np.asarray(series_idx, dtype=np.int32),
            np.asarray(ts, dtype=float), np.asarray(val, dtype=float))


def name_series(series: dict):
    """
    Per-metric indices from sorted unique signatures (deterministic), e.g.
    throughput_1 / _2 / _3 for the three services.
    """
    sig = {key: signature(key[0], json.loads(key[1])) for key in series}
    by_metric = {}
    for (metric, _), s in sig.items():
        by_metric.setdefault(metric, set()).add(s)

    sig_to_var = {}
    for metric in sorted(by_metric):
        for i, s in enumerate(sorted(by_metric[metric])):
            sig_to_var[s] = (f"{metric}_{i + 1}", metric, i + 1)

    names, mapping_rows, seen = [None] * len(series), [], set()
    for key, idx in series.items():
        variable, metric, i = sig_to_var[sig[key]]
        names[idx] = variable
        if variable not in seen:
            seen.add(variable)
            mapping_rows.append({
                "variable": variable,
                "metric_name": metric,
                "series_idx": i,
                "signature": sig[key],
                "labels_json": key[1],
            })
    mapping = pd.DataFrame(mapping_rows).sort_values(["metric_name", "series_idx"])
    return names, mapping


def main():
    args = parse_args()
    in_path = Path(args.csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Cannot find {in_path.resolve()}")
    out_wide = in_path.parent / in_path.name.replace("prom_dump_all_", "dbn_wide_")
    out_map = in_path.parent / in_path.name.replace("prom_dump_all_", "dbn_variable_mapping_")

    tags, series, tag_idx, series_idx, ts, val = read_long(in_path)
    names, mapping = name_series(series)

    long = pd.DataFrame({
        "timestamp": ts,
        "variable": pd.Categorical.from_codes(series_idx, names) if len(set(names)) == len(names)
        else pd.Series(np.asarray(names, dtype=object)[series_idx]),
        "value": val,
    })
    wide = long.pivot_table(
        index="timestamp",
        columns="variable",
        values="value",
        aggfunc="mean",   # if duplicates at same time exist
        observed=True,
    ).sort_index()
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.reset_index()

    # Chronological run numbering (see module docstring).
    run_id = (wide["timestamp"].diff() > RUN_GAP_SEC).cumsum().astype(int)
    wide.insert(0, "run_id", run_id)

    tag_names = np.empty(len(tags), dtype=object)
    for tag, i in tags.items():
        tag_names[i] = tag
    first_tag = pd.Series(tag_idx).groupby(ts).first()
    wide.insert(1, "run_tag", tag_names[first_tag.reindex(wide["timestamp"]).to_numpy()])

    out_wide.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_wide, index=False)
    mapping.to_csv(out_map, index=False)

    n_runs = int(wide["run_id"].nunique())
    print("Wrote wide DBN table:", out_wide)
    print("Wrote variable mapping:", out_map)
    print(f"Wide shape: {wide.shape[0]} rows × {wide.shape[1]} columns, "
          f"{n_runs} runs ({len(tags)} distinct run tags)")
    run_len = wide.groupby("run_id").size()
    print(f"Rows per run: min={run_len.min()} median={int(run_len.median())} max={run_len.max()}")
    if wide.shape[1] > 2000:
        print("Note: you have many series/columns. DBNs may need dimensionality reduction/aggregation later.")


if __name__ == "__main__":
    main()
