"""
check_monitor.py -- does the live Monitor give the model exactly what it was
trained on?

The raw Prometheus dump is replayed second by second through
monitor.row_from_samples, as if each second had come from an instant query,
and compared with the table the model was trained on (Model.load_frame on the
wide CSV, i.e. data_prep/convert_prom_dump.py + dbn/data.py).

  1. names        instance-based names (monitor.py) == the converter's names
                  (dbn_variable_mapping_*.csv), for every model input series
  2. values       every model input column, second by second
  3. predictions  Model.predict_next fed with Monitor rows == Model.predict on
                  the training pipeline's inputs, for random within-run steps
                  and for reconfiguration steps

Usage:
    python adaptation/check_monitor.py --dump data/prom_dump_all_<stem>.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dbn"))
sys.path.insert(0, str(HERE))
import build_model as B  # noqa: E402
import data as D  # noqa: E402
import monitor as M  # noqa: E402

DEFAULT_MODEL = HERE.parent / "results" / "models" / "dbn_model_1s_h1_20bins_3datasets_20261015_151408_g9731e35.pkl"
SETTINGS = [f"{p}_{i}" for p in ("cores", "data_quality") for i in (1, 2, 3)]


def check_names(mapping_csv):
    mapping = pd.read_csv(mapping_csv)
    mapping = mapping[mapping["metric_name"].isin(M.METRICS)]
    bad = [(r.variable, M.variable_name(json.loads(r.labels_json))) for r in mapping.itertuples()
           if M.variable_name(json.loads(r.labels_json)) != r.variable]
    print(f"[names] {len(mapping)} series, {len(bad)} mismatches {bad[:5]}")
    return not bad


def replay(dump):
    """{timestamp (ms): row} from the dump, one Monitor row per second."""
    labels_of, by_ts = {}, {}
    with open(dump, newline="") as f:
        reader = csv.reader(f)
        col = {name: i for i, name in enumerate(next(reader))}
        for row in reader:
            if row[col["metric_name"]] not in M.METRICS:
                continue
            lj = row[col["labels_json"]]
            labels = labels_of.get(lj) or labels_of.setdefault(lj, json.loads(lj))
            ts = round(float(row[col["timestamp"]]) * 1000)
            by_ts.setdefault(ts, []).append((labels, row[col["value"]]))
    return {ts: M.row_from_samples(s) for ts, s in by_ts.items()}


def check_values(rows, frame, cols):
    ts = np.round(frame["timestamp"].to_numpy() * 1000).astype(np.int64)
    missing = [t for t in ts if t not in rows]
    live = pd.DataFrame([rows.get(t, {}) for t in ts], columns=cols)
    ok = True
    print(f"[values] {len(frame)} training rows, {len(missing)} without a Monitor row, "
          f"{len(rows) - len(frame) + len(missing)} Monitor rows not in training (dropped rows)")
    for c in cols:
        a, b = live[c].to_numpy(float), frame[c].to_numpy(float)
        diff = np.nanmax(np.abs(a - b)) if np.isfinite(a).any() else np.nan
        n_nan = int(np.isnan(a).sum())
        good = n_nan == len(missing) and diff <= 1e-9 * max(1.0, np.nanmax(np.abs(b)))
        ok &= bool(good)
        print(f"  {c:18s} max |diff| {diff:.3g}  missing {n_nan}  {'OK' if good else 'MISMATCH'}")
    return ok and not missing, live


def check_predictions(model, frame, live, n, seed=0):
    inputs, index = model.inputs_from_frame(frame)
    # predict_next calls a step a reconfiguration when the settings change; the
    # training pipeline calls every run boundary one (6 of 559 boundaries per
    # experiment keep the same settings). Live, the settings are the right rule.
    s = frame[SETTINGS].to_numpy()
    changed = (s[index["row_t"]] != s[index["row_next"]]).any(axis=1)
    print(f"[predictions] run boundaries without a settings change: "
          f"{int((index['reconfigured'].to_numpy() & ~changed).sum())} (scored as ordinary steps)")
    rng = np.random.default_rng(seed)
    within, cross = np.flatnonzero(~changed), np.flatnonzero(changed)
    picks = np.r_[rng.choice(within, min(n, len(within)), replace=False),
                  rng.choice(cross, min(n // 10 or 1, len(cross)), replace=False)]
    ref = model.predict({k: v[picks] for k, v in inputs.items()}, changed[picks])
    worst = 0.0
    for k, p in enumerate(picks):
        t, nxt = index["row_t"].iloc[p], index["row_next"].iloc[p]
        recent = live.iloc[t - model.n_lags + 1: t + 1]
        got = model.predict_next(recent, live.iloc[nxt][SETTINGS].to_dict())
        for v in got:
            worst = max(worst, float(np.abs(got[v]["probs"][0] - ref[v]["probs"][k]).max()))
    ok = worst < 1e-12
    print(f"[predictions] {len(picks)} steps (incl. {min(n // 10 or 1, len(cross))} reconfigurations), "
          f"all {len(ref)} variables: "
          f"max |prob diff| {worst:.3g}  {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--n", type=int, default=2000, help="random steps for the prediction check")
    args = ap.parse_args()

    dump = Path(args.dump)
    wide = dump.with_name(dump.name.replace("prom_dump_all_", "dbn_wide_"))
    mapping = dump.with_name(dump.name.replace("prom_dump_all_", "dbn_variable_mapping_"))
    model = B.Model.load(args.model)

    ok = check_names(mapping)
    frame = model.load_frame(wide)
    cols = [c for c in frame.columns if c in {M.variable_name({"__name__": m, "instance": f"linked-service-{i}"})
                                              for m in M.METRICS for i in (1, 2, 3)} | {D.LOAD_COL}]
    print(f"[replay] reading {dump.name} ...", flush=True)
    rows = replay(dump)
    good, live = check_values(rows, frame, cols)
    ok &= good
    live[["run_id", "pos", "timestamp"]] = frame[["run_id", "pos", "timestamp"]].to_numpy()
    ok &= check_predictions(model, frame, live, args.n)
    print("ALL OK" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
