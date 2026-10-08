"""
build_model.py -- fit the final DBN on all the data and save it as one model.

=============================================================================
1. WHERE THIS FILE SITS (read this first)
=============================================================================
The repo has three kinds of scripts. They share code, but they answer
different questions:

    QUESTION                                   SCRIPT                 OUTPUT
    -----------------------------------------  ---------------------  -------------------------
    "Which modelling options are good?"        dbn/ablation.py        scores (results/ablation)
      Cross-validation: fit on the training
      runs of a fold, score on its test runs.
      Nothing is kept except the scores.

    "Give me THE model, to use it."            dbn/build_model.py     one file (results/models)
      Fit ONE configuration on ALL the runs,     (this file)
      for every variable, and save tables,
      bin edges and parents.

    "Draw / inspect a full pgmpy network."     dbn/dbn_sweep.py       pgmpy models, slow

So the work flow is:

    step 1  ablation.py       decide the configuration        (done: A.RECOMMENDED)
    step 2  ablation.py       report its cross-validated score (--study recommended)
    step 3  build_model.py    fit that configuration on everything -> the final model

The score to quote for the model built in step 3 is the one of step 2. A model
fitted on all the data has no data left to be tested on, so this file prints
only an IN-SAMPLE accuracy, as a sanity check, never as a result.

What is shared with ablation.py (imported, not copied):
    - the configuration: A.recommended_cfg(variable). study_recommended scores
      exactly this dictionary; this file fits exactly this dictionary.
    - loading / aggregation / run bookkeeping (data.py), the samples
      (A.Samples), the discretisation (A.Binner), the parent search
      (A.select_parents), the table (A.fit_table -> A.CPT).
What is specific to this file:
    - "training rows" are all rows, not the training part of a fold;
    - the fitted pieces are taken out of the harness and stored in a small
      object (Node) that predicts from RAW values, without Samples / Binner;
    - the three throughputs, three latencies and two queues are put together
      in one object (Model) with one predict().

`--check` proves the link: it fits a model on the training runs of the last
fold with this file, predicts the test runs through the saved-model code path,
and compares with what ablation.py reports for the same fold. The two must
agree to rounding error. Run it after any change to this file or ablation.py.

=============================================================================
2. WHAT THE MODEL IS
=============================================================================
A two-slice DBN, stored as one conditional probability table per variable:

    P( v(t+1) | parents of v )        v in A.SYSTEM_VARIABLES

Parents of v are chosen by the structure search among
    - the state at t and t-1 of v's own service and of the flow entering it
      (pool="local": a service only depends on itself and on what it receives),
    - the exogenous inputs AT t+1: cores_i, data_quality_i, capacity_i, rps.
      They are set from outside (operator / workload), so they are given, not
      predicted, and they never have parents.
Every parent is therefore OBSERVED when the prediction is made. That is why
one table per variable is the whole model: given slice t, the variables at
t+1 are independent of each other, and predicting any of them is one table
look-up (no inference engine, microseconds). The union of the per-variable
parent sets is the edge list of the DBN (see Model.structure()).

Two additions on top of the table:
    - capacity_i = cores_i x rate(data_quality_i): how much service i can
      process per second. `rate` is learned from the data
      (data.fit_capacity_rate) and stored in the model.
    - at a RECONFIGURATION (cores or data quality change between t and t+1)
      the past of a throughput describes the old configuration. For those
      steps the throughputs are predicted by inference in the service chain
          rps -> throughput_1 -> throughput_2 -> throughput_3
          capacity_i -> throughput_i
      with the upstream throughputs summed out (on_change="capchain").
    - a numeric forecast next to the bin distribution: v(t) + the median
      change seen in training for that parent configuration ("cg_median").

What the model does NOT contain: edges among variables of the same time step
inside a run, and tables for the exogenous inputs. Multi-step forecasts
therefore need either unrolling (dbn/rollout.py, good for ~2 s) or a model
built for that horizon (--horizon 5 builds P(v(t+5) | state at t)).

One limitation to know: each variable has its own bin edges for its parents
(the harness scores one variable at a time, and the throughputs are binned on
the edges of the variable being predicted). The interface of the model is
therefore raw values in, distribution + number out, and the tables can be
combined freely on that interface. A sampled bin of one table is not a bin of
another one; to unroll, feed the numeric forecast (or a sampled bin's value)
back in as a raw value, or use rollout.py, which uses one set of bins.

=============================================================================
3. HOW TO USE IT
=============================================================================
Build (about 1 minute per variable and dataset at 1 s):

    # the recommended model, trained on the three datasets together
    python dbn/build_model.py --csv data/dbn_wide_*.csv

    # other choices; --set takes any option of ablation.DEFAULTS
    python dbn/build_model.py --csv data/dbn_wide_X.csv --n-bins 50
    python dbn/build_model.py --csv data/dbn_wide_X.csv --targets throughput_3 --set max_parents=5
    python dbn/build_model.py --csv data/dbn_wide_X.csv --granularity 30 --n-bins 4

    # verify that this file and ablation.py describe the same model
    python dbn/build_model.py --csv data/dbn_wide_X.csv --check

    # print what a saved model contains
    python dbn/build_model.py --show results/models/<name>.pkl

Files written to results/models/:
    <name>.pkl             the model (pickle; needs dbn/ on the Python path to load)
    <name>_structure.csv   one row per edge: child, parent, time slice, what it is used for
    <name>_summary.csv     one row per variable: parents, table size, in-sample accuracy

Use from Python:

    import sys; sys.path.insert(0, "dbn")
    import build_model as B

    m = B.Model.load("results/models/<name>.pkl")

    # (a) on recorded data: every (t, t+1) pair of a wide table
    df = m.load_frame("data/dbn_wide_X.csv")          # same preparation as in training
    inputs, index = m.inputs_from_frame(df)           # raw parent values per pair
    out = m.predict(inputs, index["reconfigured"])
    out["throughput_3"]["value"]                      # numeric forecast, one per pair
    out["throughput_3"]["probs"]                      # (pairs x bins) distribution
    out["throughput_3"]["bin"]                        # most probable bin
    m.nodes["throughput_3"].y_edges                   # the inner bin edges

    # (b) online: the last seconds of telemetry + what is planned for the next one
    out = m.predict_next(recent,                      # DataFrame, newest row last, >= m.n_lags rows,
                                                      # columns = variable names (rps, throughput_1, ...)
                         nxt={"rps": 180, "cores_1": 2, "data_quality_1": 800, ...})

    m.structure()                                     # edge list as a DataFrame
    m.to_pgmpy()                                      # the same graph as a pgmpy network (for figures)

To change WHAT is built, change the configuration, not this file: test the new
option in ablation.py first (a study), then pass it here with --set, or update
A.RECOMMENDED once it is the new choice.
"""
import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation as A  # noqa: E402
import data as D  # noqa: E402

MODELS_DIR = A.REPO_ROOT / "results" / "models"
OFFSET = 100000          # run ids of the k-th csv are shifted by k * OFFSET (as in transfer_study.py)
MIN_COUNT_MEDIAN = 5     # parent configurations seen less often use the per-bin median (as in A.numeric_models)


# ============================================================
# DATA: one frame for all variables
# ============================================================
def make_frame(csvs, granularity, pool="local"):
    """
    The wide tables of `csvs`, one after the other, at `granularity` seconds per
    row. Run ids are made unique across files; the gap between the last run of
    one file and the first of the next is larger than 1, so no (t, t+1) pair is
    ever formed across two files.

    ablation.py loads one frame per target (A.load_frame); the rows are the
    same, here there is simply one frame for all targets.
    """
    frames = []
    for k, csv in enumerate(csvs):
        df = D.aggregate(D.load_wide(csv, None, verbose=False), granularity)
        df["run_id"] = df["run_id"] + k * OFFSET
        frames.append(df)
    cols = [c for c in frames[0].columns if all(c in f.columns for f in frames)]
    df = pd.concat([f[cols] for f in frames], ignore_index=True)
    if pool != "all":        # process / GC metrics are only candidates with pool="all"
        df = df[D.META_COLS + [c for c in D.feature_cols(df) if A.is_core(c)]]
    return df


def add_capacity(df, capacity):
    """capacity_i columns from a stored (rate, line) pair; no-op when the model has no capacity node."""
    return df if capacity is None else D.add_capacity(df, capacity["rate"], capacity["line"])


def raw_inputs(df, i, j, n_lags):
    """
    The naming convention of the model's inputs, in one place.

    For row positions i (time t) and j (time t+h) of a frame, returns
        "<variable>@t"      value at t
        "<variable>@t-1"    value one row earlier (... up to n_lags-1)
        "<variable>@t+h"    value at the predicted step
    for every variable of the frame. Rows i-1, i-2, ... must belong to the same
    run as row i (the callers guarantee it).

    These are the names ablation.py uses for its candidate columns, except that
    ablation.py calls the variable being predicted "y" ("y@t"); here every
    variable keeps its own name, so one dictionary serves all the tables.
    """
    out = {}
    for c in D.feature_cols(df):
        x = df[c].to_numpy(dtype=float)
        for lag in range(n_lags):
            out[f"{c}@t" + (f"-{lag}" if lag else "")] = x[i - lag]
        out[f"{c}@t+h"] = x[j]
    return out


# ============================================================
# ENCODERS: raw value -> state index, taken out of A.Binner
# ============================================================
def encoder_of(binner, S, col):
    """How `col` was discretised by the harness: ("levels", values) or ("edges", inner edges)."""
    binner.code(col, slice(0, 1))                      # makes sure the variable has been fitted
    key = S.var_of(col)
    return ("levels", binner.levels[key]) if key in binner.levels else ("edges", binner.edges[key])


def encode(enc, x):
    """Same arithmetic as A.Binner.code: nearest level for categorical variables, bin index otherwise."""
    kind, a = enc
    x = np.asarray(x, dtype=float)
    if kind == "levels":
        idx = np.clip(np.searchsorted(a, x), 0, len(a) - 1)
        left = np.clip(idx - 1, 0, len(a) - 1)
        return np.where(np.abs(x - a[left]) < np.abs(x - a[idx]), left, idx).astype(np.int64)
    return np.searchsorted(a, x, side="right").astype(np.int64)


def public(col, target):
    """ablation.py's name of a column ('y@t-1') -> the model's name ('throughput_3@t-1')."""
    return target + col[1:] if col.split("@")[0] == "y" else col


# ============================================================
# ONE VARIABLE
# ============================================================
class Node:
    """
    Everything needed to predict one variable from raw values:

        target      name of the variable
        cfg         the ablation configuration it was fitted with
        parents     input names, in the order of the table's dimensions
        encoders    {input name: encoder}; the target's own lags use y_edges
        cpt         A.CPT: counts per parent configuration + prior (prior="backoff":
                    rare configurations fall back to the autoregressive table)
        y_edges     inner bin edges of the target; bin k is [y_edges[k-1], y_edges[k])
        bin_value   the value a bin stands for (mean of the training values in it)
        ar_table    P(bin at t+1 | bin at t), the AR reference, kept for comparison
        med_*       the numeric read-out: median change per parent configuration
        chain       None, or the list of stages used at a reconfiguration
    """

    @classmethod
    def fit(cls, S, cfg, tr):
        """
        Fit on the samples `tr` (boolean mask over S). This is the first half of
        A.run_fold -- same calls, same order -- followed by taking the fitted
        pieces out of the harness objects.
        """
        unsupported = [k for k, bad in (("target_mode", cfg["target_mode"] != "level"),
                                        ("velocity", cfg["velocity"] != "none"),
                                        ("relative", cfg["relative"]), ("level", cfg["level"]),
                                        ("on_change", cfg["on_change"] == "static")) if bad]
        if unsupported:
            raise NotImplementedError(
                f"build_model.py cannot export a model with {unsupported} set: these options add "
                "derived input columns (or a second table) that raw_inputs()/Node.predict() do not "
                "build. They were not retained in A.RECOMMENDED; extend those two functions first.")
        self = cls()
        self.target, self.cfg = S.target, dict(cfg)
        binner = A.Binner(S, cfg, tr)
        c = A.Coder(S, cfg, binner, tr)
        self.y_edges, self.n_y = binner.y_edges, binner.n_y

        # --- structure: which candidate columns become parents (A.select_parents) ---
        _, self.ar_table = A.ar_model(c("y@t"), c.y_fit, c.n_fit, cfg["ess"])
        cands = A.candidate_columns(S, cfg)
        forced = [f for f in cfg["forced"] if f in S.cols]
        parents, self.search_info = A.select_parents(S, cfg, tr, c, cands, forced)
        self.candidates = [public(p, S.target) for p in cands]

        # --- parameters: the table (A.fit_table) ---
        self.cpt = A.fit_table(cfg, c, parents, self.ar_table)
        self.parents = [public(p, S.target) for p in parents]
        self.encoders = {public(p, S.target): encoder_of(binner, S, p) for p in parents}
        self.encoders.setdefault(f"{S.target}@t", ("edges", self.y_edges))

        # --- the value of a bin (as in A.run_fold) ---
        sums = np.bincount(c.y, weights=S.y1[tr], minlength=self.n_y)
        cnt = np.bincount(c.y, minlength=self.n_y)
        e = self.y_edges
        centers = np.r_[e[:1], (e[1:] + e[:-1]) / 2, e[-1:]] if len(e) else np.array([S.y1[tr].mean()])
        self.bin_value = np.where(cnt > 0, sums / np.maximum(cnt, 1), centers)

        # --- numeric read-out (the "cg_median" of A.numeric_models) ---
        codes = [c(p) for p in parents]
        g = pd.DataFrame({"key": A._keys([k for k, _ in codes], [n for _, n in codes]),
                          "bin": c.y0, "d": (S.y1 - S.y0)[tr]})
        by_key = g.groupby("key")["d"].agg(["median", "count"])       # index is sorted
        self.med_keys = by_key.index.to_numpy()
        self.med_val, self.med_n = by_key["median"].to_numpy(), by_key["count"].to_numpy()
        self.med_by_bin = g.groupby("bin")["d"].median().reindex(range(self.n_y)).fillna(0.0).to_numpy()

        # --- reconfiguration steps: the service chain (the fitting part of A.chain_predict) ---
        self.chain = None
        if cfg["on_change"] in ("chain", "capchain") and S.target.startswith("throughput_"):
            self.chain = self._fit_chain(S, cfg, binner, c)
        self.n_train = int(tr.sum())
        return self

    def _fit_chain(self, S, cfg, binner, c):
        """
        One table per service from the load down to the target, all values at
        the same instant:  P(throughput_i | upstream flow, configuration of i).
        With on_change="capchain" the configuration is capacity_i, and the bin
        of min(upstream, capacity) is added as a deterministic parent that
        centres the prior ("a service emits what it receives or what it can
        process, whichever is smaller").
        """
        K, last = self.n_y, A.service_of(S.target)
        noisy_min = cfg["on_change"] == "capchain"
        stages = []
        for i in range(1, last + 1):
            up = f"{D.LOAD_COL}@t+h" if i == 1 else f"throughput_{i - 1}@t+h"
            ctrl = [f"capacity_{i}@t+h"] if noisy_min else [f"cores_{i}@t+h", f"data_quality_{i}@t+h"]
            y = c.y if i == last else c(f"throughput_{i}@t+h")[0]
            par = [c(col) for col in [up] + ctrl]
            if noisy_min:
                codes = [k for k, _ in par] + [np.minimum(par[0][0], par[1][0])]
                cpt = A.CPT(codes, [n for _, n in par] + [K], y, K, "backoff", cfg["ess"],
                            (2, A.noisy_min_table(K)))
            else:
                cpt = A.CPT([k for k, _ in par], [n for _, n in par], y, K, "bdeu", cfg["ess"])
            stages.append(dict(service=i, up=up, ctrl=ctrl, cpt=cpt, noisy_min=noisy_min,
                               enc={col: encoder_of(binner, S, col) for col in [up] + ctrl}))
        return stages

    # ---- prediction ----
    def required_inputs(self):
        cols = set(self.parents) | {f"{self.target}@t"}
        for k, st in enumerate(self.chain or []):
            # only the load is read; the upstream throughputs at t+h are summed out
            cols |= set(st["ctrl"]) | ({st["up"]} if k == 0 else set())
        return sorted(cols)

    def _predict_chain(self, inputs, rows):
        """P(target | load, configuration) with the upstream throughputs summed out."""
        n, K, P_up = int(rows.sum()), self.n_y, None
        for st in self.chain:
            ctrl = [encode(st["enc"][col], inputs[col][rows]) for col in st["ctrl"]]

            def table(up_code, st=st, ctrl=ctrl):
                codes = [up_code] + ctrl + ([np.minimum(up_code, ctrl[0])] if st["noisy_min"] else [])
                return st["cpt"].predict(codes, n)
            if P_up is None:                               # the load is observed
                P_up = table(encode(st["enc"][st["up"]], inputs[st["up"]][rows]))
            else:                                          # sum over the unobserved upstream flow
                P_up = sum(P_up[:, [s]] * table(np.full(n, s)) for s in range(K))
        return P_up

    def predict(self, inputs, reconfigured=None):
        """
        inputs: {name: array}, at least self.required_inputs() (see raw_inputs).
        reconfigured: boolean array, True where cores / data quality change
            between t and t+h (default: nowhere).
        Returns a dict:
            probs   (n x n_bins) P(bin of target at t+h)
            bin     most probable bin
            value   numeric forecast in the target's units
        """
        y0 = np.asarray(inputs[f"{self.target}@t"], dtype=float)
        n = len(y0)
        reconf = np.zeros(n, dtype=bool) if reconfigured is None else np.asarray(reconfigured, dtype=bool)
        codes = [encode(self.encoders[p], inputs[p]) for p in self.parents]
        P = self.cpt.predict(codes, n)

        # inside a run: y(t) + median change seen for this parent configuration
        key = A._keys(codes, self.cpt.cards)
        loc = np.clip(np.searchsorted(self.med_keys, key), 0, len(self.med_keys) - 1)
        known = (self.med_keys[loc] == key) & (self.med_n[loc] >= MIN_COUNT_MEDIAN)
        y0_bin = np.searchsorted(self.y_edges, y0, side="right")
        value = y0 + np.where(known, self.med_val[loc], self.med_by_bin[y0_bin])

        # at a reconfiguration: the chain for throughputs; the expected bin value as the number
        if reconf.any():
            if self.chain is not None:
                P[reconf] = self._predict_chain(inputs, reconf)
            value[reconf] = P[reconf] @ self.bin_value
        return dict(probs=P, bin=P.argmax(1), value=value)

    def predict_ar(self, inputs):
        """The AR reference on the same bins: P(bin at t+h | bin at t)."""
        return self.ar_table[np.searchsorted(self.y_edges, inputs[f"{self.target}@t"], side="right")]

    def code_target(self, y):
        return np.searchsorted(self.y_edges, np.asarray(y, dtype=float), side="right")

    def edges(self):
        """Rows of the structure table for this variable."""
        rows = []
        for p in self.parents:
            var, when = p.split("@")
            rows.append(dict(child=self.target, parent=var, parent_slice=when, input=p,
                             used="within a run"))
        for st in self.chain or []:
            child = f"throughput_{st['service']}"
            for col in [st["up"]] + st["ctrl"]:
                rows.append(dict(child=child, parent=col.split("@")[0], parent_slice="t+h", input=col,
                                 used=f"reconfiguration chain of {self.target}"))
        return rows


# ============================================================
# THE WHOLE MODEL
# ============================================================
class Model:
    """
    nodes      {variable: Node}
    meta       how it was built (files, granularity, horizon, commit, date)
    capacity   None or {"rate": {quality: req/s per core}, "line": (slope, intercept)}
    n_lags     rows of history a prediction needs
    """

    def __init__(self, nodes, meta, capacity):
        self.nodes, self.meta, self.capacity = nodes, meta, capacity
        lags = [int(p.split("@t-")[1]) for n in nodes.values() for p in n.parents if "@t-" in p]
        self.n_lags = 1 + max(lags, default=0)

    # ---- inputs ----
    def load_frame(self, csv):
        """A wide table prepared as the training data was (rps, rates, aggregation, capacity)."""
        df = make_frame([csv] if isinstance(csv, (str, Path)) else list(csv),
                        self.meta["granularity"], self.meta["pool"])
        return add_capacity(df, self.capacity)

    def inputs_from_frame(self, df, min_history=None):
        """
        Every (t, t+h) pair of a prepared frame, including the pairs that
        straddle a reconfiguration (last row of a run -> first row of the next).

        Returns (inputs, index): `inputs` for predict(); `index` is a DataFrame
        with, per pair, the frame positions of t and t+h, the run of t+h, and
        `reconfigured`.
        """
        run, pos = df["run_id"].to_numpy(), df["pos"].to_numpy()
        i, j, cross = D.pair_index(run, pos, self.meta["horizon"], cross_run=True)
        keep = pos[i] >= (min_history or self.n_lags) - 1
        i, j, cross = i[keep], j[keep], cross[keep]
        order = np.argsort(j, kind="stable")
        i, j, cross = i[order], j[order], cross[order]
        index = pd.DataFrame({"row_t": i, "row_next": j, "run_id": run[j], "reconfigured": cross})
        return raw_inputs(df, i, j, self.n_lags), index

    def required_inputs(self, nodes=None):
        return sorted({c for n in (nodes or self.nodes) for c in self.nodes[n].required_inputs()})

    # ---- prediction ----
    def predict(self, inputs, reconfigured=None, nodes=None):
        """{variable: {"probs", "bin", "value"}} for the requested variables (default: all)."""
        return {n: self.nodes[n].predict(inputs, reconfigured) for n in (nodes or self.nodes)}

    def predict_next(self, recent, nxt, nodes=None):
        """
        One prediction from live data.

        recent: DataFrame with the last rows of telemetry, newest last, at the
            model's granularity; columns named as in the wide table after
            loading (rps, throughput_1, buffer_size_2, cores_1, ...). At least
            self.n_lags rows, all from the current configuration.
        nxt: the exogenous inputs of the step to predict: rps, cores_i,
            data_quality_i. Anything missing is assumed unchanged.
        A change of cores / data quality between the last row and `nxt` is
        detected here and treated as a reconfiguration.
        """
        if len(recent) < self.n_lags:
            raise ValueError(f"need the last {self.n_lags} rows, got {len(recent)}")
        last = recent.iloc[-1]
        step = pd.DataFrame([{**last.to_dict(), **nxt}])
        config = [c for c in recent.columns if c.startswith(("cores_", "data_quality_"))]
        reconfigured = bool(any(step[c].iloc[0] != last[c] for c in config))
        hist = add_capacity(recent.iloc[-self.n_lags:].reset_index(drop=True), self.capacity)
        step = add_capacity(step, self.capacity)
        inputs = {}
        for c in hist.columns:
            if c in D.META_COLS:
                continue
            x = hist[c].to_numpy(dtype=float)
            for lag in range(self.n_lags):
                inputs[f"{c}@t" + (f"-{lag}" if lag else "")] = x[[-1 - lag]]
            inputs[f"{c}@t+h"] = step[c].to_numpy(dtype=float)
        return self.predict(inputs, np.array([reconfigured]), nodes)

    # ---- inspection ----
    def structure(self):
        """Edge list of the DBN: one row per (child, parent, time slice of the parent)."""
        rows = [r for n in self.nodes.values() for r in n.edges()]
        return pd.DataFrame(rows).drop_duplicates(["child", "input", "used"]).reset_index(drop=True)

    def summary(self):
        rows = []
        for n in self.nodes.values():
            rows.append(dict(variable=n.target, n_bins=n.n_y, parents=json.dumps(n.parents),
                             table_configs_possible=n.cpt.q, table_configs_seen=len(n.cpt.uniq),
                             chain_at_reconfiguration=n.chain is not None, n_train=n.n_train,
                             config=A.cfg_id(n.cfg)))
        return pd.DataFrame(rows)

    def to_pgmpy(self):
        """
        The within-run graph as a pgmpy BayesianNetwork (structure only, for
        drawing): nodes <variable>_t, <variable>_tm1 (t-1) and <variable>_t1
        (the predicted step; exogenous parents are in that slice too).
        """
        from pgmpy.models import BayesianNetwork
        name = {"t": "_t", "t+h": "_t1"}
        edges = []
        for n in self.nodes.values():
            for p in n.parents:
                var, when = p.split("@")
                suffix = name.get(when) or "_tm" + when.split("-")[1]
                edges.append((var + suffix, n.target + "_t1"))
        return BayesianNetwork(edges)

    # ---- storage ----
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ============================================================
# BUILD
# ============================================================
def build(df, cfgs, train_runs=None, capacity_runs=None, verbose=True):
    """
    df: frame from make_frame (without capacity columns).
    cfgs: {variable: ablation configuration}; same granularity and horizon for all.
    train_runs: run ids to fit on (default: all -- the final model).
    capacity_runs: run ids the capacity rate is learned from (default: train_runs).
    """
    first = next(iter(cfgs.values()))
    if any((c["granularity"], c["horizon"], c["pool"] == "all") !=
           (first["granularity"], first["horizon"], first["pool"] == "all") for c in cfgs.values()):
        raise ValueError("all variables of one model must share granularity, horizon and pool")
    all_runs = df["run_id"].unique()
    train_runs = all_runs if train_runs is None else train_runs
    capacity = None
    if any(c["capacity"] for c in cfgs.values()):
        rate, line = D.fit_capacity_rate(df, train_runs if capacity_runs is None else capacity_runs)
        capacity = {"rate": rate, "line": line}
        df = add_capacity(df, capacity)

    nodes = {}
    for target, cfg in cfgs.items():
        S = A.Samples(df, target, cfg["horizon"], cfg["cross_run"],
                      A.MAX_HISTORY.get(cfg["granularity"], 3))
        tr, _ = A.split_masks(S, train_runs, train_runs)
        nodes[target] = Node.fit(S, cfg, tr)
        if verbose:
            print(f"  {target:16s} <- {nodes[target].parents}", flush=True)
    meta = dict(granularity=first["granularity"], horizon=first["horizon"], pool=first["pool"],
                created=f"{datetime.now():%Y-%m-%d %H:%M:%S}", commit=A._git_version(),
                n_train_runs=int(len(train_runs)))
    return Model(nodes, meta, capacity), df


def score(model, inputs, index, df):
    """Accuracy, log-loss and absolute error of the model and of AR on the given pairs."""
    out = model.predict(inputs, index["reconfigured"].to_numpy())
    rows = []
    for name, node in model.nodes.items():
        y1 = df[name].to_numpy(dtype=float)[index["row_next"].to_numpy()]
        y = node.code_target(y1)
        P, k = out[name]["probs"], np.arange(len(y))
        Par = node.predict_ar(inputs)
        rows.append(dict(variable=name, n=len(y),
                         acc=float((P.argmax(1) == y).mean()),
                         logloss=float(-np.log(np.clip(P[k, y], 1e-12, 1)).mean()),
                         mae=float(np.abs(out[name]["value"] - y1).mean()),
                         acc_ar=float((Par.argmax(1) == y).mean()),
                         mae_persistence=float(np.abs(inputs[f"{name}@t"] - y1).mean())))
    return pd.DataFrame(rows)


def check(csv, cfgs):
    """
    The link with ablation.py, verified. For the last cross-validation fold:
      reference: A.run_fold on ablation.py's own samples (what a study reports);
      here:      build() on the training runs, Model.predict() on the test runs
                 from raw values, as a saved model would be used.
    """
    first = next(iter(cfgs.values()))
    df = make_frame([csv], first["granularity"], first["pool"])
    folds = D.make_temporal_folds(df[D.META_COLS], k=first["folds"], scheme=first["cv"])
    train_runs, test_runs = folds[-1][0]["run_id"].unique(), folds[-1][1]["run_id"].unique()
    # ablation.py learns the capacity rate on the first chunk of runs (A.load_frame)
    model, df = build(df, cfgs, train_runs, capacity_runs=folds[0][0]["run_id"].unique(), verbose=False)
    inputs, index = model.inputs_from_frame(df, A.MAX_HISTORY.get(first["granularity"], 3))
    te = index["run_id"].isin(test_runs).to_numpy()
    mine = score(model, {k: v[te] for k, v in inputs.items()}, index[te].reset_index(drop=True), df)

    worst = 0.0
    print(f"{'variable':16s} {'':10s} {'accuracy':>10s} {'log-loss':>10s} {'abs error':>10s}")
    for _, r in mine.iterrows():
        cfg = cfgs[r["variable"]]
        S = A.get_samples(csv, cfg)
        tr_m, te_m = A.split_masks(S, *A.fold_runs(S, cfg)[-1])
        res, _ = A.run_fold(S, {**cfg, "numeric": True}, tr_m, te_m)
        ref = (res["model"][0]["acc_all"], res["model"][0]["logloss_all"], res["cg_median"][0]["maeh_all"])
        got = (r["acc"], r["logloss"], r["mae"])
        worst = max(worst, *(abs(a - b) for a, b in zip(ref, got)))
        print(f"{r['variable']:16s} {'ablation':10s} {ref[0]:10.6f} {ref[1]:10.6f} {ref[2]:10.6f}")
        print(f"{'':16s} {'this file':10s} {got[0]:10.6f} {got[1]:10.6f} {got[2]:10.6f}")
    print(f"\nlargest difference: {worst:.2e}  ->  {'OK' if worst < 1e-9 else 'MISMATCH'}")
    return worst


def parse_overrides(pairs):
    out = {}
    for kv in pairs:
        k, v = kv.split("=", 1)
        if k not in A.DEFAULTS:
            raise SystemExit(f"--set {k}: not an option of ablation.DEFAULTS")
        d = A.DEFAULTS[k]
        out[k] = json.loads(v.lower()) if d is None or isinstance(d, bool) else type(d)(v)
    return out


def main():
    ap = argparse.ArgumentParser(description="Fit the final DBN on all the data and save it "
                                             "(see the header of this file).")
    ap.add_argument("--csv", nargs="+", help="wide table(s); several are pooled into one training set")
    ap.add_argument("--targets", nargs="+", default=list(A.SYSTEM_VARIABLES))
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--granularity", type=int, default=1, help="seconds per row")
    ap.add_argument("--horizon", type=int, default=1, help="rows ahead")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override an option of the recommended configuration (any key of ablation.DEFAULTS)")
    ap.add_argument("--name", default=None, help="file name of the model (default: built from the options)")
    ap.add_argument("--check", action="store_true",
                    help="compare with ablation.py on the last fold of ONE csv, then exit")
    ap.add_argument("--show", metavar="MODEL.pkl", help="print the content of a saved model, then exit")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200, "display.max_colwidth", 120)

    if args.show:
        m = Model.load(args.show)
        print(json.dumps(m.meta, indent=1), "\n")
        print(m.summary().to_string(index=False), "\n")
        print(m.structure().to_string(index=False))
        return
    if not args.csv:
        ap.error("--csv is required")

    overrides = dict(n_bins=args.n_bins, granularity=args.granularity, horizon=args.horizon,
                     **parse_overrides(args.set))
    cfgs = {v: A.recommended_cfg(v, **overrides) for v in args.targets}

    if args.check:
        if len(args.csv) != 1:
            ap.error("--check takes one csv (ablation.py scores one dataset at a time)")
        raise SystemExit(0 if check(args.csv[0], cfgs) < 1e-9 else 1)

    print(f"Fitting {len(cfgs)} variables on {len(args.csv)} file(s), all runs:")
    df = make_frame(args.csv, args.granularity, next(iter(cfgs.values()))["pool"])
    model, df = build(df, cfgs)
    model.meta.update(files=[Path(c).name for c in args.csv], overrides=overrides)

    name = args.name or (f"dbn_model_{args.granularity}s_h{args.horizon}_{args.n_bins}bins_"
                         f"{len(args.csv)}datasets_{datetime.now():%Y%m%d_%H%M%S}_g{model.meta['commit']}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    model.save(path)
    model.structure().to_csv(MODELS_DIR / f"{name}_structure.csv", index=False)

    # Sanity check of the saved file: reload it and predict the data it was fitted on.
    # IN-SAMPLE numbers: they show the file works, they are not a result.
    inputs, index = Model.load(path).inputs_from_frame(df)
    insample = score(Model.load(path), inputs, index, df).set_index("variable")
    summary = model.summary().set_index("variable").join(
        insample.add_prefix("insample_")).reset_index()
    summary.to_csv(MODELS_DIR / f"{name}_summary.csv", index=False)
    print("\nIn-sample check of the saved model (NOT a score; for scores run "
          "`ablation.py --study recommended`):")
    print(insample[["n", "acc", "acc_ar", "mae", "mae_persistence"]].round(4).to_string())
    print(f"\nSaved {path}\n      {MODELS_DIR / (name + '_structure.csv')}"
          f"\n      {MODELS_DIR / (name + '_summary.csv')}")


if __name__ == "__main__":
    # Run main() from the imported module, not from "__main__": pickle stores the
    # module a class comes from, and a model saved by "__main__.Model" could only
    # be loaded by this script itself, not by `import build_model`.
    import build_model
    build_model.main()
