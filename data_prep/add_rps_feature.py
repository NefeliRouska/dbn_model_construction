#!/usr/bin/env python3
"""
add_rps_feature.py

Reconstructs the experiment RPS signal for each 30-second window in a wide-format
DBN dataset and appends it as a new column `rps`.

RPS logic is a direct port from the experiment orchestrator (experiment_orchestrator.py).
All constants must match those in the orchestrator exactly.

NOTE -- prefer the measured load. `buffer_size_1` in the dumps takes exactly the
values the orchestrator sends to /change_rps, at the moment they take effect;
dbn/data.py exposes it as `rps`. This script rebuilds the INTENDED pattern,
which differs from what was applied: the controller only re-evaluates the
pattern every 20 s, and not at all between two measurement windows.

Clock alignment: the pattern runs on the orchestrator's clock, which starts
when the orchestrator is launched -- before the containers are restarted and
before the first stabilisation period -- while the first row of the dump is
the start of the first measurement window. The offset between the two is not
recorded, so it is estimated: the shift (0..600 s) that makes the rebuilt
pattern agree best with `--align-to` (default buffer_size_1). Use `--offset`
to set it by hand.

Usage:
    python add_rps_feature.py \
        --csv dbn_wide_20260313_192652_seed783122.csv \
        --pattern periodic \
        --out dbn_wide_20260313_192652_seed783122_with_rps.csv

Patterns:
    periodic        Cosine-based, period=3000s, range [150,350], quantised to 60s steps.
                    Starts at peak (350 RPS) at elapsed=0. Seed not used.
    oial            200 RPS baseline; spike to 600 RPS at [40min, 43min); back to 200.
    unpredictable   random.Random(seed*100000 + slot).randint(100,500), new slot every 600s.
"""

import argparse
import random
import sys
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Constants — keep in sync with experiment_orchestrator.py
# ============================================================

# Periodic
PERIODIC_LOW              = 150
PERIODIC_HIGH             = 350
PERIODIC_SINE_PERIOD_SEC  = 3000   # 50 minutes
PERIODIC_UPDATE_SEC       = 60     # 1 minute
PERIODIC_START_AT_PEAK    = True   # True -> cosine (starts at 350)

# Once-in-a-lifetime
OIAL_BASE       = 200
OIAL_SPIKE      = 600
OIAL_BEFORE_SEC = 2400   # 40 min before spike
OIAL_SPIKE_SEC  = 180    # 3 min spike

# Unpredictable
UNPREDICTABLE_MIN      = 100
UNPREDICTABLE_MAX      = 500
UNPREDICTABLE_STEP_SEC = 600   # 10 minutes


# ============================================================
# RPS signal reconstruction (ported from orchestrator)
# ============================================================

def rps_periodic(elapsed_seconds: np.ndarray) -> np.ndarray:
    mean      = (PERIODIC_LOW + PERIODIC_HIGH) / 2.0    # 250
    amplitude = (PERIODIC_HIGH - PERIODIC_LOW) / 2.0    # 100
    sampled_time = (elapsed_seconds // PERIODIC_UPDATE_SEC) * PERIODIC_UPDATE_SEC
    angle        = 2.0 * np.pi * sampled_time / PERIODIC_SINE_PERIOD_SEC
    rps          = mean + amplitude * np.cos(angle)
    rps_int = np.round(rps).astype(int)
    return np.clip(rps_int, PERIODIC_LOW, PERIODIC_HIGH)


def rps_oial(elapsed_seconds: np.ndarray) -> np.ndarray:
    spike_start = float(OIAL_BEFORE_SEC)
    spike_end   = float(OIAL_BEFORE_SEC + OIAL_SPIKE_SEC)
    return np.where(
        (elapsed_seconds >= spike_start) & (elapsed_seconds < spike_end),
        float(OIAL_SPIKE),
        float(OIAL_BASE),
    )


def rps_unpredictable(elapsed_seconds: np.ndarray, seed: int) -> np.ndarray:
    slots        = (elapsed_seconds // UNPREDICTABLE_STEP_SEC).astype(int)
    unique_slots = np.unique(slots)
    slot_to_rps  = {}
    for s in unique_slots:
        local_rng = random.Random(seed * 100000 + int(s))
        slot_to_rps[int(s)] = local_rng.randint(UNPREDICTABLE_MIN, UNPREDICTABLE_MAX)
    return np.array([slot_to_rps[int(s)] for s in slots])


# ============================================================
# Timestamp helpers
# ============================================================

CANDIDATE_TIMESTAMP_COLS = ["timestamp", "time", "Timestamp", "Time", "datetime"]


def find_timestamp_col(df: pd.DataFrame) -> str:
    for name in CANDIDATE_TIMESTAMP_COLS:
        if name in df.columns:
            return name
    for col in df.columns:
        if "time" in col.lower() or "stamp" in col.lower():
            return col
    raise ValueError(
        f"Cannot find a timestamp column. Columns present: {list(df.columns)}"
    )


def to_elapsed_seconds(series: pd.Series) -> np.ndarray:
    """Convert a timestamp series to elapsed seconds from experiment start.
    Tries numeric (Unix epoch) first, falls back to datetime string parsing.
    """
    # Try numeric first — timestamps here are Unix epoch floats.
    # pd.to_datetime() must NOT be tried first: it silently mis-parses
    # epoch floats as nanosecond-scale datetimes, giving elapsed ~ 0.
    vals = pd.to_numeric(series, errors="coerce").values.astype(float)
    if not np.isnan(vals).all():
        return vals - vals.min()

    # Fallback for human-readable datetime strings
    ts      = pd.to_datetime(series)
    elapsed = (ts - ts.min()).dt.total_seconds().values
    return elapsed.astype(float)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Add reconstructed RPS feature to a wide-format DBN CSV."
    )
    p.add_argument("--csv",     required=True,
                   help="Path to input wide CSV file.")
    p.add_argument("--pattern", required=True,
                   choices=["periodic", "oial", "unpredictable"],
                   help="Workload pattern: periodic | oial | unpredictable")
    p.add_argument("--seed",    type=int, default=783122,
                   help="Random seed used by the experiment orchestrator (default: 783122).")
    p.add_argument("--out",     default=None,
                   help="Output path. Defaults to <stem>_with_rps.csv alongside input.")
    p.add_argument("--align-to", default="buffer_size_1",
                   help="Measured-load column used to estimate the clock offset "
                        "(default: buffer_size_1). Ignored if --offset is given.")
    p.add_argument("--offset",  type=float, default=None,
                   help="Seconds between orchestrator start and the first row of the dump.")
    return p.parse_args()


def rps_for(pattern, elapsed, seed):
    if pattern == "periodic":
        return rps_periodic(elapsed)
    if pattern == "oial":
        return rps_oial(elapsed)
    if pattern == "unpredictable":
        return rps_unpredictable(elapsed, seed)
    raise ValueError(f"Unknown pattern: {pattern}")


def estimate_offset(pattern, elapsed, seed, measured, max_offset=600):
    """Shift of the orchestrator clock that best reproduces the measured load."""
    best_offset, best_match = 0.0, -1.0
    for offset in range(0, max_offset + 1):
        match = float(np.mean(np.abs(rps_for(pattern, elapsed + offset, seed) - measured) <= 1))
        if match > best_match:
            best_offset, best_match = float(offset), match
    return best_offset, best_match


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] Input CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else csv_path.with_name(
        csv_path.stem + "_with_rps" + csv_path.suffix
    )

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")

    ts_col = find_timestamp_col(df)
    print(f"  Timestamp column : '{ts_col}'")

    if "run_tag" in df.columns:
        print(f"  run_tag column   : present (ignored for RPS reconstruction)")

    elapsed = to_elapsed_seconds(df[ts_col])
    print(f"  Elapsed range    : {elapsed.min():.1f}s – {elapsed.max():.1f}s  "
          f"({elapsed.max() / 3600:.2f} h)")

    if args.offset is not None:
        offset = args.offset
        print(f"  Clock offset     : {offset:.0f}s (given)")
    elif args.align_to in df.columns:
        measured = pd.to_numeric(df[args.align_to], errors="coerce").to_numpy(dtype=float)
        match0 = float(np.mean(np.abs(rps_for(args.pattern, elapsed, args.seed) - measured) <= 1))
        offset, match = estimate_offset(args.pattern, elapsed, args.seed, measured)
        print(f"  Clock offset     : {offset:.0f}s (estimated from '{args.align_to}': "
              f"{match:.1%} of rows agree, {match0:.1%} without the offset)")
    else:
        offset = 0.0
        print(f"  [WARN] '{args.align_to}' not in the table: no clock alignment, "
              f"the pattern may be shifted by tens of seconds", file=sys.stderr)

    rps_values = rps_for(args.pattern, elapsed + offset, args.seed)

    print(f"  RPS range        : [{rps_values.min()}, {rps_values.max()}]")
    print(f"  Unique RPS values: {len(np.unique(rps_values))}")

    # Insert rps immediately after timestamp column
    insert_pos = df.columns.get_loc(ts_col) + 1
    df.insert(insert_pos, "rps", rps_values)

    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()