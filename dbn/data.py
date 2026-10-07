"""
Shared data layer: load a wide table, keep track of which run and which second
every row belongs to, aggregate, split into folds and build (t, t+h) pairs.

Everything downstream of this module works on DataFrames that carry three
bookkeeping columns next to the variables:

    run_id     measurement window (one configuration of cores / data quality),
               numbered chronologically
    pos        position of the row inside its run (0, 1, 2, ...) at the current
               modelling granularity
    timestamp  Unix time of the (first second of the) row

Rows are always in chronological order. A "next step" is only ever taken
inside a run, unless cross-run pairs are explicitly requested: between two runs
the orchestrator changes the configuration and waits >= 30 s without recording.
"""
import re

import numpy as np
import pandas as pd

META_COLS = ["run_id", "pos", "timestamp"]

# Consecutive timestamps further apart than this start a new run
# (same constant as data_prep/convert_prom_dump.py).
RUN_GAP_SEC = 5.0

# Arrival rate offered to the first service. buffer_size_1 takes exactly the
# values the orchestrator sends to /change_rps, at the moment service 1 has
# them: it is the rate setpoint as seen by service 1 (for the other services
# the same metric is their queue backlog), not an independent measurement of
# the traffic. It is the best record of the load in the dumps.
LOAD_SOURCE_COL = "buffer_size_1"
LOAD_COL = "rps"

# Variables the operator (or the workload generator) sets from outside. Their
# value at t+h is an input of the prediction, not something to predict.
CONTROL_PREFIXES = ("cores_", "data_quality_", "capacity_")

# A service is saturated in a run when it emits less than this share of what
# it receives (run means).
SATURATED_BELOW = 0.97

_COUNTER_RE = re.compile(r"^(?P<base>.+)_total_(?P<idx>\d+)$")


def feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def exog_cols(columns, mode):
    """Columns treated as known at t+h. mode: 'none' | 'controls' | 'controls+load'."""
    if mode == "none":
        return []
    out = [c for c in columns if c.lower().startswith(CONTROL_PREFIXES)]
    if mode == "controls+load":
        out += [c for c in columns if c == LOAD_COL]
    elif mode != "controls":
        raise ValueError("exog mode must be none | controls | controls+load")
    return out


def _counters_to_rates(df):
    """
    Cumulative Prometheus counters (..._total_N) grow with wall-clock time, so
    their level only encodes "how long has the experiment been running". Replace
    them with their per-second increase inside each run.
    """
    renamed = {}
    for c in list(df.columns):
        m = _COUNTER_RE.match(c)
        if not m:
            continue
        rate = df.groupby("run_id")[c].diff()
        rate = rate.where(rate >= 0)                      # counter reset -> missing
        rate = rate.groupby(df["run_id"]).transform(lambda s: s.bfill().ffill())
        df[c] = rate
        renamed[c] = f"{m.group('base')}_rate_{m.group('idx')}"
    return df.rename(columns=renamed)


def load_wide(path, target=None, load_source=LOAD_SOURCE_COL, counters="rate",
              variance_threshold=0.01, verbose=True):
    """
    Read a wide table written by data_prep/convert_prom_dump.py.

    load_source: column to expose as `rps` (default buffer_size_1; the source
        column is then dropped so the load is not in the table twice). Pass
        None to keep the table as it is (e.g. to use a reconstructed `rps`
        column written by data_prep/add_rps_feature.py).
    counters: 'rate' (per-second increase), 'drop', or 'keep'.
    """
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(f"{path}: no 'timestamp' column, cannot order rows in time")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    # Tables written before the converter recorded run_id: derive it.
    if "run_id" not in df.columns:
        df.insert(0, "run_id", (df["timestamp"].diff() > RUN_GAP_SEC).cumsum().astype(int))
    df["pos"] = df.groupby("run_id").cumcount()

    for col in list(df.columns):
        if col not in META_COLS and "time" in col.lower():
            df.drop(columns=[col], inplace=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    df = df.select_dtypes(include=[np.number])

    if load_source is not None and load_source in df.columns:
        if LOAD_COL in df.columns and verbose:
            print(f"[DATA] replacing existing '{LOAD_COL}' column with {load_source}")
        df[LOAD_COL] = df[load_source]
        if load_source != LOAD_COL:
            df = df.drop(columns=[load_source])

    if counters == "rate":
        df = _counters_to_rates(df)
    elif counters == "drop":
        df = df.drop(columns=[c for c in df.columns if _COUNTER_RE.match(c)])
    elif counters != "keep":
        raise ValueError("counters must be rate | drop | keep")

    df = df.dropna().reset_index(drop=True)
    df["pos"] = df.groupby("run_id").cumcount()

    if target is not None and target not in df.columns:
        raise ValueError(f"TARGET '{target}' not found after cleaning.")

    feats = [c for c in feature_cols(df) if c != target]
    var = df[feats].var(ddof=0)
    dropped = [c for c in feats if not var[c] > variance_threshold]
    if dropped and verbose:
        print(f"[VARIANCE FILTER] Dropped {len(dropped)} near-constant columns: {dropped}")
    kept = [c for c in feats if c not in dropped] + ([target] if target is not None else [])
    return df[META_COLS + kept]


def aggregate(df, every_n_seconds):
    """
    Mean over windows of `every_n_seconds` seconds, aligned to the start of each
    run. A trailing window shorter than that is dropped, so no window ever mixes
    two runs or covers less time than the others. every_n_seconds=1 is a no-op.
    """
    if every_n_seconds <= 1:
        return df.reset_index(drop=True)
    win = df["pos"] // every_n_seconds
    g = df.groupby([df["run_id"], win], sort=True)
    out = g.mean()
    out["timestamp"] = g["timestamp"].min()
    full = g.size() == every_n_seconds
    out = out[full.to_numpy()]
    out["run_id"] = out.index.get_level_values(0)
    out["pos"] = out.index.get_level_values(1)
    out = out.sort_values(["timestamp"]).reset_index(drop=True)
    return out[META_COLS + feature_cols(out)]


def make_temporal_folds(df, k=5, scheme="expanding"):
    """
    Split whole runs into folds; a run is never cut in two.

    scheme='expanding': fold i trains on the first chunk(s) and tests on the
        next one (k+1 chronological chunks -> k folds), as in the original sweep.
    scheme='blocked': k chronological chunks; each is the test set once and the
        other k-1 are the training set. Every row is tested exactly once and
        every configuration block is seen in training in k-1 folds.
    """
    runs = df["run_id"].to_numpy()
    uniq, first = np.unique(runs, return_index=True)   # runs are contiguous and sorted
    n_chunks = k + 1 if scheme == "expanding" else k
    targets = np.linspace(0, len(df), n_chunks + 1)[1:-1]
    cuts = sorted({int(first[np.argmin(np.abs(first - t))]) for t in targets})
    bounds = [0] + cuts + [len(df)]

    folds = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if scheme == "expanding":
            if i == 0:
                continue
            train, test = df.iloc[:lo], df.iloc[lo:hi]
        elif scheme == "blocked":
            train = pd.concat([df.iloc[:lo], df.iloc[hi:]])
            test = df.iloc[lo:hi]
        else:
            raise ValueError("scheme must be expanding | blocked")
        if len(train) < 2 or len(test) < 2:
            continue
        folds.append((train.reset_index(drop=True), test.reset_index(drop=True)))
    return folds


def pair_index(run_id, pos, horizon=1, cross_run=False):
    """
    Row positions (i, j) such that row j is `horizon` steps after row i in the
    same run. With cross_run=True, also pairs the last available row of every
    run with the first available row of the following run (the configuration
    change; the real time gap is then the stabilisation period, not `horizon`).

    Returns (i, j, is_cross).
    """
    run_id = np.asarray(run_id, dtype=np.int64)
    pos = np.asarray(pos, dtype=np.int64)
    if len(run_id) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([], dtype=bool)

    span = int(pos.max()) + horizon + 2
    key = run_id * span + pos
    order = np.argsort(key, kind="stable")
    skey = key[order]
    want = key + horizon
    loc = np.searchsorted(skey, want)
    loc_c = np.minimum(loc, len(skey) - 1)
    hit = (skey[loc_c] == want) & (loc < len(skey))
    i = np.nonzero(hit)[0]
    j = order[loc_c[hit]]
    is_cross = np.zeros(len(i), dtype=bool)

    if cross_run:
        s_run = run_id[order]
        last_of_run = np.nonzero(np.r_[s_run[1:] != s_run[:-1], True])[0]
        first_of_run = np.nonzero(np.r_[True, s_run[1:] != s_run[:-1]])[0]
        # consecutive run ids only: both sides of the change must be in this frame
        ok = s_run[first_of_run[1:]] == s_run[last_of_run[:-1]] + 1
        ci = order[last_of_run[:-1][ok]]
        cj = order[first_of_run[1:][ok]]
        i = np.r_[i, ci]
        j = np.r_[j, cj]
        is_cross = np.r_[is_cross, np.ones(len(ci), dtype=bool)]

    return i.astype(int), j.astype(int), is_cross


def two_slice_frame(df, pairs):
    """Columns <v>_t (row i) next to <v>_t1 (row j) for every pair."""
    i, j = pairs[0], pairs[1]
    df_t = df.iloc[i].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[j].reset_index(drop=True).add_suffix("_t1")
    return pd.concat([df_t, df_t1], axis=1)


def fit_capacity_rate(df, runs=None):
    """
    Processing rate of ONE core as a function of data quality.

    In these services the capacity is, to a very good approximation,
    cores x rate(data_quality), with the same rate function for the three
    services (they run the same code). It is learned from run means:

      - in a run where a service is saturated its throughput IS its capacity:
        rate = median of throughput / cores over those runs, per quality level;
      - in a run where it is not saturated its throughput is a lower bound of
        its capacity. At low quality settings services almost never saturate;
        the rate is at least the largest unsaturated throughput / cores
        observed at that level, and exactly that when fewer than three
        saturated runs exist;
      - a higher quality setting never makes a core faster: the rate is made
        non-increasing in data quality.

    Levels with no run at all are filled from a straight line fitted to
    log(rate) against log(data_quality).

    runs: run ids to learn from (default: all runs of df).
    Returns ({quality: rate}, (slope, intercept) of the log-log line).
    """
    if runs is not None:
        df = df[df["run_id"].isin(runs)]
    r = df.groupby("run_id").mean(numeric_only=True)
    samples = []
    for i in (1, 2, 3):
        up = r[LOAD_COL] if i == 1 else r[f"throughput_{i - 1}"]
        samples.append(pd.DataFrame({
            "dq": r[f"data_quality_{i}"], "rate": r[f"throughput_{i}"] / r[f"cores_{i}"],
            "saturated": r[f"throughput_{i}"] < SATURATED_BELOW * up}))
    samples = pd.concat(samples)
    sat = samples[samples["saturated"]].groupby("dq")["rate"].agg(["median", "size"])
    # An unsaturated service emits exactly what it receives, which cannot exceed
    # its capacity: the largest such throughput per core is a (noise-free) lower
    # bound of the rate. It decides wherever saturation was rarely observed, and
    # wherever the runs flagged as saturated contradict it.
    bound = samples[~samples["saturated"]].groupby("dq")["rate"].max()
    measured = sat.loc[sat["size"] >= 3, "median"]
    rate = pd.concat([measured, bound], axis=1).max(axis=1)
    if rate.isna().any() or len(rate) == 0:
        rate = rate.fillna(samples.groupby("dq")["rate"].max())
    rate = rate.sort_index()[::-1].cummax()[::-1]          # non-increasing in data quality

    levels = np.unique(np.concatenate([df[f"data_quality_{i}"].unique() for i in (1, 2, 3)]))
    if len(rate) >= 2:
        slope, intercept = np.polyfit(np.log(rate.index.to_numpy(dtype=float)), np.log(rate.to_numpy()), 1)
    else:
        slope, intercept = 0.0, float(np.log(rate.iloc[0]))
    return ({float(q): float(rate[q]) if q in rate.index else float(np.exp(intercept + slope * np.log(q)))
             for q in levels}, (float(slope), float(intercept)))


def add_capacity(df, rate, line=None):
    """Columns capacity_i = cores_i x rate(data_quality_i): what service i can emit per second."""
    df = df.copy()

    def lookup(q):
        if q in rate:
            return rate[q]
        slope, intercept = line
        return float(np.exp(intercept + slope * np.log(q)))

    for i in (1, 2, 3):
        df[f"capacity_{i}"] = df[f"cores_{i}"] * df[f"data_quality_{i}"].map(lookup)
    return df

