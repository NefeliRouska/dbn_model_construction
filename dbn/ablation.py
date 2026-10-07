"""
ablation.py -- one harness for every modelling option of the target's
transition model.

What is modelled
----------------
The quantity every experiment in this repo scores is

    P( target at t+h  |  everything observed up to t  [, controls / load at t+h] )

In a 2-slice DBN whose slice t is fully observed, that distribution is fully
determined by the PARENT SET of target_{t+h} and its conditional probability
table (as long as those parents are observed; an unobserved same-slice parent
would have to be marginalised out). So every structure-learning engine, every
"how long is the state" choice and every discretisation choice acts on the
prediction through one thing only: which discretised columns end up as parents
of the target, and how the table is estimated. This file models exactly that,
with numpy counting instead of generic inference, which makes it fast enough to
run at 1 s granularity with many bins. dbn_sweep.py remains the full-graph
pipeline (pgmpy); `--check-pgmpy` verifies both give the same numbers.

A configuration (see DEFAULTS) is one point in the option space; a study is a
list of configurations. Every configuration is evaluated on the same test
samples, and per-run sums are stored so that two configurations can be compared
with a paired, run-level bootstrap (compare_to_reference).

Usage
-----
    python dbn/ablation.py --csv data/dbn_wide_<stem>.csv --study ofat
    python dbn/ablation.py --csv ... --study bins_granularity
    python dbn/ablation.py --list-studies
"""
import argparse
import itertools
import json
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

import data as D

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "ablation"

# ============================================================
# OPTION SPACE
# ============================================================
DEFAULTS = dict(
    target="throughput_3",
    # --- data ---
    granularity=1,          # seconds per row
    horizon=1,              # rows ahead
    cross_run=True,         # also learn / test the configuration changes
    pool="core",            # core: service metrics | all: + process / GC metrics
                            # | local: the target's own service + what flows into it
    cv="expanding",         # expanding | blocked
    folds=5,
    # --- state ---
    ar_order=1,             # target values available: y@t ... y@t-(p-1)
    other_lags=1,           # slices of the other variables available (1 = only @t)
    velocity="none",        # none | sign | binned   (v@t - v@t-1 of every variable)
    relative=False,         # add v@t - y@t for the upstream throughputs and the load
    exog="controls+load",   # none | controls | controls+load   (values at t+h)
    capacity=False,         # add capacity_i = cores_i x rate(data_quality_i) as exogenous
                            # variables (rate learned on the first chunk of runs, which is
                            # training data in every expanding fold; see data.fit_capacity_rate)
    # --- discretisation ---
    n_bins=20,              # target (and throughput family when shared_bins)
    disc="uniform",         # target: uniform | quantile | kmeans
    n_bins_x=None,          # other variables (None -> n_bins)
    disc_x="quantile",      # other variables: uniform | quantile | kmeans
    shared_bins=True,       # throughput_1/2 and rps use the target's bin edges
    # --- structure ---
    engine="cv-forward",    # ar | fixed | hc-bic | hc-bic-obs | hc-aic | hc-bdeu
                            # | cv-forward | cmi | lasso | dynotears
    max_parents=4,
    forced=("y@t",),        # parents that are always in
    fixed_parents=(),       # for engine=fixed
    # --- CPT ---
    target_mode="level",    # level: P(bin of y') | delta: P(bin of y' - bin of y)
    prior="backoff",        # bdeu (what pgmpy fits) | backoff (to the AR table)
    ess=10.0,
    numeric=False,          # also score continuous read-outs of the same model (see numeric_models)
    on_change="same",       # what predicts the first step after a reconfiguration:
                            # same: the model above | static: a second table over the
                            # new configuration | chain: inference in the service chain
                            # | capchain: the chain with capacity nodes and a noisy-min
                            #   prior (needs capacity=True)
)

MAX_HISTORY = {1: 5, 5: 5, 10: 3, 30: 3}   # rows of history reserved per granularity

EXPERT_CHAIN = ("y@t", "throughput_2@t", "buffer_size_3@t", "cores_3@t+h", "data_quality_3@t+h")


def cfg_id(cfg):
    """Short string listing every option that differs from DEFAULTS."""
    diff = {k: v for k, v in cfg.items() if DEFAULTS.get(k) != v}
    return "ref" if not diff else ",".join(f"{k}={v}" for k, v in sorted(diff.items()))


def make_cfg(**kw):
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown options: {unknown}")
    return {**DEFAULTS, **kw}


# ============================================================
# SAMPLES: one row per (t, t+h) pair, continuous values
# ============================================================
def is_core(col):
    return col.split("_")[0] in ("throughput", "cores", "data", "buffer", "avg", "capacity") \
        or col == D.LOAD_COL


class Samples:
    """
    Continuous candidate columns for every (t, t+h) pair of one dataset at one
    granularity / horizon. Column names:
        v@t, v@t-1, ...   value of v at t, t-1, ...
        d.v@t             v@t - v@t-1
        v@t+h             value at the predicted step (exogenous variables only)
        y@t ...           the target's own history
    """

    def __init__(self, df, target, horizon, cross_run, max_history):
        self.target = target
        feats = D.feature_cols(df)
        run, pos = df["run_id"].to_numpy(), df["pos"].to_numpy()
        i, j, cross = D.pair_index(run, pos, horizon, cross_run)
        keep = pos[i] >= max_history - 1            # same samples for every history length
        i, j, cross = i[keep], j[keep], cross[keep]
        order = np.argsort(j, kind="stable")
        self.i, self.j, self.cross = i[order], j[order], cross[order]
        self.run = run[self.j]                      # a sample belongs to the run of its target
        self.run_i = run[self.i]
        V = {c: df[c].to_numpy(dtype=float) for c in feats}
        self.vars = feats
        self.cols = {}
        for c in feats:
            name = "y" if c == target else c
            for lag in range(max_history):
                # rows are contiguous inside a run and pos[i] >= max_history-1
                self.cols[f"{name}@t" + (f"-{lag}" if lag else "")] = V[c][self.i - lag]
            self.cols[f"d.{name}@t"] = V[c][self.i] - V[c][self.i - 1]
            self.cols[f"{c}@t+h"] = V[c][self.j]
        self.y1 = V[target][self.j]
        self.y0 = V[target][self.i]
        self.family = [c for c in feats if c != target
                       and (c.startswith("throughput_") or c == D.LOAD_COL)]
        for c in self.family:                       # distance of the upstream flow to the target
            self.cols[f"g.{c}@t"] = V[c][self.i] - self.y0

    def var_of(self, col):
        """Underlying variable of a candidate column (for sharing bin edges)."""
        base = col.split("@")[0]
        kind = {"d.": "vel", "g.": "gap"}.get(base[:2], "")
        base = base[2:] if kind else base
        return (self.target if base == "y" else base), kind


def service_of(col):
    """Index of the service a column belongs to (throughput_3 -> 3), None for rps etc."""
    tail = col.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def local_pool(S):
    """
    Domain constraint: a service's output can only depend on that service's own
    state and settings and on what the previous stage sends it (the load, for
    the first service).
    """
    i = service_of(S.target)
    upstream = D.LOAD_COL if i == 1 else f"throughput_{i - 1}"
    return [v for v in S.vars if is_core(v) and (service_of(v) == i or v == upstream)]


def variable_pool(S, cfg):
    if cfg["pool"] == "local":
        return local_pool(S)
    return [v for v in S.vars if cfg["pool"] == "all" or is_core(v)]


def candidate_columns(S, cfg):
    exog = set(D.exog_cols(S.vars, cfg["exog"]))
    controls = set(D.exog_cols(S.vars, "controls"))
    pool = variable_pool(S, cfg)
    cands = []
    for v in pool:
        name = "y" if v == S.target else v
        n_lags = cfg["ar_order"] if v == S.target else cfg["other_lags"]
        cands += [f"{name}@t" + (f"-{lag}" if lag else "") for lag in range(n_lags)]
        # controls never move inside a run, so their velocity carries nothing
        if cfg["velocity"] != "none" and v not in controls:
            cands.append(f"d.{name}@t")
        if v in exog:
            cands.append(f"{v}@t+h")
        if cfg["relative"] and v in S.family:
            cands.append(f"g.{v}@t")
    return cands


# ============================================================
# DISCRETISATION (fit on training samples only)
# ============================================================
def _edges(x, n_bins, method):
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.array([])
    if method == "uniform":
        return np.linspace(lo, hi, n_bins + 1)[1:-1]
    if method == "quantile":
        return np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    if method == "kmeans":
        from sklearn.cluster import KMeans
        sub = x if len(x) <= 20000 else np.random.default_rng(0).choice(x, 20000, replace=False)
        k = min(n_bins, len(np.unique(sub)))
        centers = np.sort(KMeans(n_clusters=k, n_init=3, random_state=0)
                          .fit(sub.reshape(-1, 1)).cluster_centers_.ravel())
        return (centers[1:] + centers[:-1]) / 2
    raise ValueError(f"unknown discretiser {method}")


class Binner:
    """Per-variable bin edges; every lag of a variable uses that variable's edges."""

    def __init__(self, S, cfg, train):
        self.S, self.cfg, self.edges, self.levels = S, cfg, {}, {}
        self.train = train
        nb, nbx = cfg["n_bins"], cfg["n_bins_x"] or cfg["n_bins"]
        y_train = np.r_[S.y1[train], S.y0[train]]
        self.y_edges = _edges(y_train, nb, cfg["disc"])
        self.n_y = len(self.y_edges) + 1
        # The throughputs and the load are the same quantity measured at different
        # points of the chain: one set of bin edges for all of them. It is the
        # target's own edges when the target is one of them.
        flow = {v for v in S.vars if v.startswith(("throughput_", "capacity_")) or v == D.LOAD_COL}
        self.shared_edges = {S.target: self.y_edges}
        if cfg["shared_bins"]:
            if S.target in flow:
                edges = self.y_edges
            else:
                pooled = np.concatenate([S.cols[f"{v}@t"][train] for v in sorted(flow)
                                         if not v.startswith("capacity_")])
                edges = _edges(pooled, nbx, "uniform")
            self.shared_edges.update({v: edges for v in flow if v != S.target})
        self.shared = set(self.shared_edges)
        self.nbx = nbx

    def _symmetric_edges(self, x, n_side):
        """Bins mirrored around a dead zone at exactly zero."""
        nz = np.abs(x[x != 0])
        half = _edges(nz, n_side, "quantile") if len(nz) else np.array([])
        return np.unique(np.r_[-half[::-1], -1e-9, 1e-9, half])

    def _fit_var(self, var, kind):
        key = (var, kind)
        if key in self.edges or key in self.levels:
            return
        S, cfg = self.S, self.cfg
        name = "y" if var == S.target else var
        if kind == "vel":
            x = S.cols[f"d.{name}@t"][self.train]
            self.edges[key] = np.array([-1e-9, 1e-9]) if cfg["velocity"] == "sign" \
                else self._symmetric_edges(x, 2)
            return
        if kind == "gap":
            self.edges[key] = self._symmetric_edges(S.cols[f"g.{name}@t"][self.train],
                                                    max(2, self.nbx // 4))
            return
        if var in self.shared:
            self.edges[key] = self.shared_edges[var]
            return
        x = S.cols[f"{name}@t"][self.train]
        uniq = np.unique(x)
        if len(uniq) <= max(self.nbx, 12):
            self.levels[key] = uniq                 # already categorical (cores, quality, ...)
        else:
            self.edges[key] = _edges(x, self.nbx, cfg["disc_x"])

    def code(self, col, rows=slice(None)):
        key = self.S.var_of(col)
        self._fit_var(*key)
        x = self.S.cols[col][rows]
        if key in self.levels:
            lv = self.levels[key]
            idx = np.clip(np.searchsorted(lv, x), 0, len(lv) - 1)
            left = np.clip(idx - 1, 0, len(lv) - 1)
            idx = np.where(np.abs(x - lv[left]) < np.abs(x - lv[idx]), left, idx)   # nearest level
            return idx.astype(np.int64), len(lv)
        e = self.edges[key]
        return np.searchsorted(e, x, side="right").astype(np.int64), len(e) + 1

    def code_y(self, y):
        return np.searchsorted(self.y_edges, y, side="right").astype(np.int64)


# ============================================================
# CONDITIONAL PROBABILITY TABLE
# ============================================================
def _keys(codes, cards):
    """Mixed-radix index of a parent configuration."""
    key = np.zeros(len(codes[0]) if codes else 0, dtype=np.int64)
    for c, k in zip(codes, cards):
        key = key * k + c
    return key


def _counts(key, y, n_y):
    """Unique parent configurations and their (config x target-state) counts."""
    uniq, inv = np.unique(key, return_inverse=True)
    N = np.bincount(inv * n_y + y, minlength=len(uniq) * n_y).reshape(len(uniq), n_y)
    return uniq, N.astype(float)


def local_score(N, q, n_y, n, score, ess=10.0):
    """Decomposable score of the target given a parent set (same formulas as pgmpy)."""
    if score in ("bic", "aic", "bic-obs"):
        Nj = N.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = np.nansum(N * (np.log(N) - np.log(Nj)))
        # bic-obs charges only the parent configurations that occur in the data
        k = (n_y - 1) * (N.shape[0] if score == "bic-obs" else q)
        score = "bic" if score == "bic-obs" else score
        return ll - (0.5 * np.log(n) * k if score == "bic" else k)
    if score == "bdeu":
        a_j, a_jk = ess / q, ess / (q * n_y)
        Nj = N.sum(1)
        return float(np.sum(gammaln(a_j) - gammaln(a_j + Nj))
                     + np.sum(gammaln(a_jk + N) - gammaln(a_jk)))
    raise ValueError(score)


class CPT:
    def __init__(self, codes, cards, y, n_y, prior="bdeu", ess=10.0, backoff=None):
        """
        backoff: (index of the fallback parent among `codes`, its table P[state, y])
        used as the prior mean when prior='backoff' -- unseen or rare parent
        configurations then fall back to the autoregressive table instead of to
        a uniform distribution.
        """
        self.cards, self.n_y, self.prior, self.ess, self.backoff = cards, n_y, prior, ess, backoff
        self.q = float(np.prod([float(c) for c in cards])) if cards else 1.0
        self.uniq, self.N = _counts(_keys(codes, cards) if codes else np.zeros(len(y), np.int64), y, n_y)

    def predict(self, codes, n):
        key = _keys(codes, self.cards) if codes else np.zeros(n, np.int64)
        loc = np.clip(np.searchsorted(self.uniq, key), 0, len(self.uniq) - 1)
        seen = self.uniq[loc] == key
        N = np.where(seen[:, None], self.N[loc], 0.0)
        if self.prior == "backoff" and self.backoff is not None:
            idx, table = self.backoff
            alpha = self.ess * table[codes[idx]]
        else:
            alpha = np.full((1, self.n_y), self.ess / (self.q * self.n_y))
        P = N + alpha
        return P / P.sum(1, keepdims=True)


# ============================================================
# PARENT-SET SEARCH ("structure learning" for the target)
# ============================================================
def greedy_search(code_fn, cands, forced, y, n_y, score, max_parents, ess):
    """
    Hill climbing on the target's parent set with a decomposable score. All
    candidates are in an earlier slice or exogenous, so no choice can create a
    cycle: this is the sequence of moves HillClimbSearch makes for this node.
    """
    n = len(y)

    def s(parents):
        codes, cards = zip(*[code_fn(p) for p in parents]) if parents else ((), ())
        _, N = _counts(_keys(codes, cards) if codes else np.zeros(n, np.int64), y, n_y)
        q = float(np.prod([float(c) for c in cards])) if cards else 1.0
        return local_score(N, q, n_y, n, score, ess)

    parents = list(forced)
    best = s(parents)
    while True:
        moves = []
        if len(parents) < max_parents:
            moves += [("add", c) for c in cands if c not in parents]
        moves += [("del", p) for p in parents if p not in forced]
        scored = []
        for op, c in moves:
            trial = parents + [c] if op == "add" else [p for p in parents if p != c]
            scored.append((s(trial), trial))
        if not scored:
            break
        top, trial = max(scored, key=lambda t: t[0])
        if top <= best + 1e-4:
            break
        best, parents = top, trial
    return parents


def cv_forward(c_tr, c_va, cands, forced, max_parents, cfg, ar_table):
    """Forward selection on the log-loss of an inner validation split."""
    def loss(parents):
        tr = [c_tr(p) for p in parents]
        bo = (parents.index("y@t"), ar_table) if "y@t" in parents and len(parents) > 1 else None
        cpt = CPT([c for c, _ in tr], [k for _, k in tr], c_tr.y_fit, c_tr.n_fit,
                  cfg["prior"], cfg["ess"], bo)
        P = c_va.to_bins(cpt.predict([c_va(p)[0] for p in parents], c_va.n))
        return -np.mean(np.log(np.clip(P[np.arange(c_va.n), c_va.y], 1e-12, 1)))

    parents = list(forced)
    best = loss(parents)
    while len(parents) < max_parents:
        scored = [(loss(parents + [c]), c) for c in cands if c not in parents]
        if not scored:
            break
        top, c = min(scored)
        if top >= best - 1e-4:
            break
        best, parents = top, parents + [c]
    return parents


def cmi_rank(code_fn, cands, forced, y, n_y, max_parents):
    """Top candidates by conditional mutual information I(y'; x | forced)."""
    n = len(y)
    fc = [code_fn(p) for p in forced]
    zkey = _keys([c for c, _ in fc], [k for _, k in fc]) if fc else np.zeros(n, np.int64)
    _, z = np.unique(zkey, return_inverse=True)
    nz = z.max() + 1

    def H(key):
        _, cnt = np.unique(key, return_counts=True)
        p = cnt / n
        return -np.sum(p * np.log(p))

    out = []
    for c in cands:
        if c in forced:
            continue
        x, k = code_fn(c)
        zx = z * k + x
        # I(Y;X|Z) = H(Y,Z) + H(X,Z) - H(X,Y,Z) - H(Z)
        out.append((H(z * n_y + y) + H(zx) - H(zx * n_y + y) - H(z), c))
    out.sort(reverse=True)
    return list(forced) + [c for _, c in out[:max(0, max_parents - len(forced))]]


def lasso_rank(S, cands, forced, rows, max_parents):
    """Order in which candidates enter the lasso path of a linear model for y(t+h)."""
    from sklearn.linear_model import lars_path
    rng = np.random.default_rng(0)
    idx = np.nonzero(rows)[0]
    idx = idx if len(idx) <= 50000 else rng.choice(idx, 50000, replace=False)
    X = np.column_stack([S.cols[c][idx] for c in cands])
    sd = X.std(0)
    ok = sd > 0
    X = (X[:, ok] - X[:, ok].mean(0)) / sd[ok]
    names = [c for c, o in zip(cands, ok) if o]
    y = S.y1[idx] - S.y1[idx].mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, active, _ = lars_path(X, y, method="lasso", max_iter=4 * max_parents)
    parents = list(forced)
    for a in active:
        if names[a] not in parents and len(parents) < max_parents:
            parents.append(names[a])
    return parents


def dynotears_parents(S, cfg, rows, forced, max_parents, lam=0.02):
    """
    DYNOTEARS on the standardised continuous variables. Slice "t" of the SVAR is
    the predicted step; the lagged block holds the `max(ar_order, other_lags)`
    slices before it. Parents of the target = its strongest lagged edges, plus
    same-slice edges from exogenous variables (the only same-slice values known
    at prediction time).
    """
    from dynotears import dynotears
    exog = D.exog_cols(S.vars, cfg["exog"])
    controls = set(D.exog_cols(S.vars, "controls"))
    pool = variable_pool(S, cfg)
    p = max(cfg["ar_order"], cfg["other_lags"])
    idx = np.nonzero(rows)[0]

    def col(v, lag):                       # lag 0 = predicted step
        name = "y" if v == S.target else v
        if lag == 0:
            return S.y1[idx] if v == S.target else S.cols[f"{v}@t+h"][idx]
        return S.cols[f"{name}@t" + (f"-{lag - 1}" if lag > 1 else "")][idx]

    def z(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else x * 0

    X = np.column_stack([z(col(v, 0)) for v in pool])
    L = np.column_stack([z(col(v, lag)) for lag in range(1, p + 1) for v in pool])
    d = len(pool)
    # exogenous variables have no parents; nothing but exogenous variables may be
    # a same-slice parent of anything we want to read off as "observed"
    is_ctrl = np.array([v in controls or v == D.LOAD_COL for v in pool])
    forbid_w = np.zeros((d, d), dtype=bool)
    forbid_w[:, is_ctrl] = True
    forbid_a = np.zeros((p * d, d), dtype=bool)
    forbid_a[:, is_ctrl] = True
    for lag in range(p):                   # ... except their own persistence
        for k in np.nonzero(is_ctrl)[0]:
            forbid_a[lag * d + k, k] = lag > 0
    W, A = dynotears(X, L, lambda_w=lam, lambda_a=lam, forbid_w=forbid_w, forbid_a=forbid_a)

    t = pool.index(S.target)
    weights = {}
    for lag in range(1, p + 1):
        for k, v in enumerate(pool):
            name = "y" if v == S.target else v
            n_lags = cfg["ar_order"] if v == S.target else cfg["other_lags"]
            if lag <= n_lags:
                weights[f"{name}@t" + (f"-{lag - 1}" if lag > 1 else "")] = abs(A[(lag - 1) * d + k, t])
    for k, v in enumerate(pool):
        if v in exog:
            weights[f"{v}@t+h"] = abs(W[k, t])
    hidden = {v: abs(W[k, t]) for k, v in enumerate(pool) if v not in exog and abs(W[k, t]) > 1e-3}
    ranked = sorted((w, c) for c, w in weights.items() if w > 1e-3 and c not in forced)[::-1]
    parents = list(forced) + [c for _, c in ranked[:max(0, max_parents - len(forced))]]
    return parents, {"dynotears_weights": {c: round(float(w), 4) for w, c in ranked[:12]},
                     "dynotears_unobservable_same_slice_parents":
                         {v: round(float(w), 4) for v, w in hidden.items()}}


# ============================================================
# METRICS
# ============================================================
def macro_f1(y, p, n_y):
    cm = np.bincount(y * n_y + p, minlength=n_y * n_y).reshape(n_y, n_y)
    tp = np.diag(cm).astype(float)
    support, pred = cm.sum(1), cm.sum(0)
    present = (support + pred) > 0
    f1 = np.where(2 * tp + (support - tp) + (pred - tp) > 0,
                  2 * tp / np.maximum(2 * tp + (support - tp) + (pred - tp), 1), 0.0)
    return float(f1[present].mean())


def per_sample(P, y_code, y1, bin_value, y0, y0_code):
    """
    Per-sample losses of a probabilistic prediction over the target's bins.
    ae:  |E[bin value] - y'|, the table read as a point forecast.
    aeh: the same forecast expressed as a change and applied to the observed
         y(t): y(t) + E[bin value] - value of y(t)'s bin. A model that predicts
         "same bin" then returns y(t) itself, so the width of a bin is not
         counted as an error.
    """
    n = len(y_code)
    pred = P.argmax(1)
    ev = P @ bin_value
    return dict(
        correct=(pred == y_code).astype(float),
        ll=-np.log(np.clip(P[np.arange(n), y_code], 1e-12, 1.0)),
        ae=np.abs(ev - y1),
        aeh=np.abs(y0 + ev - bin_value[y0_code] - y1),
        pred=pred,
    )


def summarise(L, masks):
    out = {}
    for name, m in masks.items():
        k = int(m.sum())
        out[f"n_{name}"] = k
        for key, label in (("correct", "acc"), ("ll", "logloss"), ("ae", "mae"), ("aeh", "maeh")):
            out[f"{label}_{name}"] = float(L[key][m].mean()) if k else np.nan
    return out


def per_run_table(L, run, masks):
    df = pd.DataFrame({"run_id": run, "n": 1.0, "correct": L["correct"], "ll": L["ll"],
                       "ae": L["ae"], "aeh": L["aeh"]})
    for name in ("change", "event", "xrun"):
        df[f"n_{name}"] = masks[name].astype(float)
        df[f"correct_{name}"] = L["correct"] * masks[name]
        df[f"ae_{name}"] = L["ae"] * masks[name]
    return df.groupby("run_id", as_index=False).sum()


# ============================================================
# ONE CONFIGURATION ON ONE FOLD
# ============================================================
def split_masks(S, train_runs, test_runs):
    tr = np.isin(S.run, train_runs) & np.isin(S.run_i, train_runs)
    te = np.isin(S.run, test_runs)
    return tr, te


def ar_model(code_y0, y, n_y, ess):
    """P(y' | y) table; also the fallback prior of prior='backoff'."""
    c0, k0 = code_y0
    ar = CPT([c0], [k0], y, n_y, "bdeu", ess)
    return ar, ar.predict([np.arange(k0)], k0)


class Coder:
    """Discretised columns and fitting target of one set of rows (memoised)."""

    def __init__(self, S, cfg, binner, rows):
        self.S, self.cfg, self.binner, self.rows = S, cfg, binner, rows
        self.n = int(rows.sum())
        self.memo = {}
        self.n_y = binner.n_y
        self.y = binner.code_y(S.y1[rows])          # bin of y(t+h)
        self.y0 = binner.code_y(S.y0[rows])         # bin of y(t)
        if cfg["target_mode"] == "delta":
            self.n_fit = 2 * self.n_y - 1
            self.y_fit = self.y - self.y0 + self.n_y - 1
        else:
            self.n_fit, self.y_fit = self.n_y, self.y

    def __call__(self, col):
        if col not in self.memo:
            self.memo[col] = self.binner.code(col, self.rows)
        return self.memo[col]

    def to_bins(self, P):
        """Distribution over the fitted target -> distribution over the bins of y(t+h)."""
        if self.cfg["target_mode"] != "delta":
            return P
        idx = np.arange(self.n_y)[None, :] - self.y0[:, None] + self.n_y - 1
        Pb = np.take_along_axis(P, idx, axis=1)
        return Pb / np.maximum(Pb.sum(1, keepdims=True), 1e-300)


def select_parents(S, cfg, tr, code_tr, cands, forced):
    """The structure-learning step: which candidate columns become parents of the target."""
    engine, info = cfg["engine"], {}
    y, n_fit, mp = code_tr.y_fit, code_tr.n_fit, cfg["max_parents"]
    if engine == "ar":
        parents = ["y@t" + (f"-{lag}" if lag else "") for lag in range(cfg["ar_order"])]
    elif engine == "fixed":
        parents = [p for p in cfg["fixed_parents"] if p in S.cols]
    elif engine in ("hc-bic", "hc-bic-obs", "hc-aic", "hc-bdeu"):
        parents = greedy_search(code_tr, cands, forced, y, n_fit, engine[3:], mp, cfg["ess"])
    elif engine == "cmi":
        parents = cmi_rank(code_tr, cands, forced, y, n_fit, mp)
    elif engine == "lasso":
        parents = lasso_rank(S, cands, forced, tr, mp)
    elif engine == "dynotears":
        parents, info = dynotears_parents(S, cfg, tr, forced, mp)
    elif engine == "cv-forward":
        # inner split: the last 20% of the training runs decide which parents to add
        runs = np.unique(S.run[tr])
        inner_va = np.isin(S.run, runs[int(0.8 * len(runs)):]) & tr
        inner_tr = tr & ~inner_va
        b2 = Binner(S, cfg, inner_tr)
        c_in, c_va = Coder(S, cfg, b2, inner_tr), Coder(S, cfg, b2, inner_va)
        _, table = ar_model(c_in("y@t"), c_in.y_fit, c_in.n_fit, cfg["ess"])
        parents = cv_forward(c_in, c_va, cands, forced, mp, cfg, table)
    else:
        raise ValueError(f"unknown engine {engine}")
    return parents, info


def fit_table(cfg, code_tr, parents, ar_table):
    tr_c = [code_tr(p) for p in parents]
    # with y@t as the only parent the table IS the autoregressive one: no prior on top
    bo = (parents.index("y@t"), ar_table) if "y@t" in parents and len(parents) > 1 else None
    return CPT([c for c, _ in tr_c], [k for _, k in tr_c], code_tr.y_fit, code_tr.n_fit,
               cfg["prior"], cfg["ess"], bo)


def numeric_models(S, cfg, binner, tr, te, parents):
    """
    Continuous forecasts of y(t+h) from the SAME parents the structure search
    selected. The table P(bin' | parents) can only move the forecast by whole
    bins; these keep the discrete parents and change what the target node is.

      cg_mean    conditional-Gaussian node: for every parent configuration, the
                 mean change y' - y seen in training (shrunk towards the mean
                 for that bin of y when the configuration is rare).
      cg_median  the same with the median (the best constant for absolute error).
      clg        conditional linear-Gaussian node: the parents that are flows
                 (throughputs, load, y itself) enter as continuous regressors,
                 the other parents (queues, latency, cores, quality) select the
                 regime; one linear model per regime, a global one as fallback.
      linear     linear-Gaussian node: one linear model on the continuous values
                 of all selected parents (what a DYNOTEARS / Gaussian BN node is).
    """
    c_tr, c_te = Coder(S, {**cfg, "target_mode": "level"}, binner, tr), \
        Coder(S, {**cfg, "target_mode": "level"}, binner, te)
    d_tr, y0_te = (S.y1 - S.y0)[tr], S.y0[te]
    out = {}

    codes_tr = [c_tr(p) for p in parents]
    cards = [k for _, k in codes_tr]
    key_tr = _keys([c for c, _ in codes_tr], cards)
    key_te = _keys([c_te(p)[0] for p in parents], cards)
    y0bin_tr, y0bin_te = c_tr.y0, c_te.y0
    g = pd.DataFrame({"key": key_tr, "bin": y0bin_tr, "d": d_tr})
    by_bin = g.groupby("bin")["d"].agg(["mean", "median"]).reindex(range(binner.n_y)).fillna(0.0)
    by_key = g.groupby("key")["d"].agg(["sum", "count", "median"])
    loc = by_key.reindex(key_te)
    n = loc["count"].fillna(0).to_numpy()
    prior_mean = by_bin["mean"].to_numpy()[y0bin_te]
    out["cg_mean"] = y0_te + (loc["sum"].fillna(0).to_numpy() + cfg["ess"] * prior_mean) / (n + cfg["ess"])
    out["cg_median"] = y0_te + np.where(n >= 5, loc["median"].fillna(0).to_numpy(),
                                        by_bin["median"].to_numpy()[y0bin_te])

    def design(cols, rows):
        # flows relative to the current value of the target, plus that value
        X = [np.ones(int(rows.sum())), S.y0[rows]]
        X += [S.cols[c][rows] - S.y0[rows] for c in cols if c != "y@t"]
        return np.column_stack(X)

    def ridge(X, y, lam=1e-3):
        A_ = X.T @ X + lam * len(y) * np.eye(X.shape[1])
        return np.linalg.solve(A_, X.T @ y)

    cont = [p for p in parents if not p.startswith(("d.", "g."))]
    X_tr, X_te = design(cont, tr), design(cont, te)
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1.0
    mu[0], sd[0] = 0.0, 1.0
    out["linear"] = y0_te + ((X_te - mu) / sd) @ ridge((X_tr - mu) / sd, d_tr)

    flows = [p for p in parents if p == "y@t" or
             (S.var_of(p)[1] == "" and (S.var_of(p)[0] in S.family or S.var_of(p)[0] == S.target))]
    regime = [p for p in parents if p not in flows]
    F_tr, F_te = design(flows, tr), design(flows, te)
    fm, fs = F_tr.mean(0), F_tr.std(0)
    fs[fs == 0] = 1.0
    fm[0], fs[0] = 0.0, 1.0
    F_tr, F_te = (F_tr - fm) / fs, (F_te - fm) / fs
    pred = F_te @ ridge(F_tr, d_tr)
    if regime:
        r_tr = _keys([c_tr(p)[0] for p in regime], [c_tr(p)[1] for p in regime])
        r_te = _keys([c_te(p)[0] for p in regime], [c_tr(p)[1] for p in regime])
        order = np.argsort(r_tr, kind="stable")
        uniq, start, count = np.unique(r_tr[order], return_index=True, return_counts=True)
        for u, s0, k in zip(uniq, start, count):
            if k < 200:
                continue
            rows = order[s0:s0 + k]
            hit = r_te == u
            if hit.any():
                pred[hit] = F_te[hit] @ ridge(F_tr[rows], d_tr[rows])
    out["clg"] = y0_te + pred
    return out


def noisy_min_table(n_bins, spread=0.1):
    """Prior of a flow given the bin of min(upstream, capacity): that bin, a little on its neighbours."""
    T = np.eye(n_bins) * (1 - spread)
    for k in range(n_bins):
        nb = [j for j in (k - 1, k + 1) if 0 <= j < n_bins]
        T[k, nb] += spread / len(nb)
    return T


def chain_predict(S, cfg, binner, tr, te):
    """
    The service chain written as a same-slice Bayesian network and queried by
    inference, for the question "what will throughput_3 be under THIS
    configuration and load":

        rps -> throughput_1 -> throughput_2 -> throughput_3
        cores_i, data_quality_i -> throughput_i

    Three tables with three parents each, learned from every training sample
    (all values taken at the same instant). The upstream throughputs of the
    predicted step are not observed, so they are summed out:
        P(tp3 | rps, cfg) = sum_{tp1, tp2} P(tp1 | rps, cfg1) P(tp2 | tp1, cfg2) P(tp3 | tp2, cfg3)
    Needs shared_bins (all throughputs on the target's bins).
    """
    level = {**cfg, "target_mode": "level"}
    c_tr, c_te = Coder(S, level, binner, tr), Coder(S, level, binner, te)
    K, n = binner.n_y, c_te.n
    last = service_of(S.target)
    chain = [(f"{D.LOAD_COL}@t+h" if i == 1 else f"throughput_{i - 1}@t+h",
              None if i == last else f"throughput_{i}@t+h", str(i)) for i in range(1, last + 1)]
    P_up = None
    noisy_min = cfg["on_change"] == "capchain"
    for up, node, i in chain:
        ctrl = [f"capacity_{i}@t+h"] if noisy_min else [f"cores_{i}@t+h", f"data_quality_{i}@t+h"]
        y = c_tr.y if node is None else c_tr(node)[0]
        par = [c_tr(c) for c in [up] + ctrl]
        ctrl_te = [c_te(c)[0] for c in ctrl]
        if noisy_min:
            # A service emits what it receives or what it can process, whichever is
            # smaller. min(upstream, capacity) enters as a deterministic extra parent:
            # it adds no parent combination, it centres the prior of the table.
            codes = [c for c, _ in par] + [np.minimum(par[0][0], par[1][0])]
            cpt = CPT(codes, [k for _, k in par] + [K], y, K, "backoff", cfg["ess"],
                      (2, noisy_min_table(K)))

            def predict(up_code, cpt=cpt, ctrl_te=ctrl_te):
                return cpt.predict([up_code] + ctrl_te + [np.minimum(up_code, ctrl_te[0])], n)
        else:
            cpt = CPT([c for c, _ in par], [k for _, k in par], y, K, "bdeu", cfg["ess"])

            def predict(up_code, cpt=cpt, ctrl_te=ctrl_te):
                return cpt.predict([up_code] + ctrl_te, n)
        if P_up is None:                                   # the load is observed
            P_up = predict(c_te(up)[0])
        else:                                              # sum over the unobserved upstream flow
            P_up = sum(P_up[:, [s]] * predict(np.full(n, s)) for s in range(K))
    return P_up


def run_fold(S, cfg, tr, te, extra_models=()):
    """Returns {model_name: (summary dict, per-run table)}, info dict."""
    binner = Binner(S, cfg, tr)
    code_tr, code_te = Coder(S, cfg, binner, tr), Coder(S, cfg, binner, te)
    n_y, n_te = binner.n_y, code_te.n
    ytr, yte, y0te = code_tr.y, code_te.y, code_te.y0
    # value a bin stands for = mean of the training targets that fall in it
    sums = np.bincount(ytr, weights=S.y1[tr], minlength=n_y)
    cnt = np.bincount(ytr, minlength=n_y)
    e = binner.y_edges
    centers = np.r_[e[:1], (e[1:] + e[:-1]) / 2, e[-1:]] if len(e) else np.array([S.y1[tr].mean()])
    bin_value = np.where(cnt > 0, sums / np.maximum(cnt, 1), centers)

    masks = {
        "all": np.ones(n_te, dtype=bool),
        "within": ~S.cross[te],
        "xrun": S.cross[te],
        "change": yte != y0te,                                 # target leaves its bin
        "stay": yte == y0te,
        "event": (S.y1[te] != S.y0[te]) & ~S.cross[te],        # target moves at all (bin-free)
    }

    _, ar_table = ar_model(code_tr("y@t"), code_tr.y_fit, code_tr.n_fit, cfg["ess"])
    cands = candidate_columns(S, cfg)
    forced = [f for f in cfg["forced"] if f in S.cols]

    t0 = time.perf_counter()
    parents, info = select_parents(S, cfg, tr, code_tr, cands, forced)
    info["search_sec"] = time.perf_counter() - t0
    cpt = fit_table(cfg, code_tr, parents, ar_table)
    te_codes = [code_te(p)[0] for p in parents]
    P = code_te.to_bins(cpt.predict(te_codes, n_te))
    te_key = _keys(te_codes, cpt.cards) if parents else np.zeros(n_te, np.int64)
    info.update(parents=parents, n_parents=len(parents), n_y=n_y,
                cpt_configs_possible=cpt.q, cpt_configs_seen=len(cpt.uniq),
                test_unseen_config_frac=float(np.mean(~np.isin(te_key, cpt.uniq))))

    if cfg["on_change"] == "static" and masks["xrun"].any():
        # At a configuration change y(t) describes the old configuration. Predict
        # those samples from a second table over the values known at t+h (and the
        # queues left behind), learned from all training samples.
        level = {**cfg, "target_mode": "level", "prior": "bdeu"}
        lv_tr, lv_te = Coder(S, level, binner, tr), Coder(S, level, binner, te)
        static_cands = [c for c in cands if c.endswith("@t+h") or c.startswith("buffer_size_")]
        sp, _ = select_parents(S, {**level, "engine": cfg["engine"] if cfg["engine"].startswith("hc")
                                   or cfg["engine"] == "cv-forward" else "hc-bic-obs",
                                   "forced": ()}, tr, lv_tr, static_cands, [])
        sc = fit_table(level, lv_tr, sp, None)
        Ps = sc.predict([lv_te(p)[0] for p in sp], n_te)
        P = np.where(masks["xrun"][:, None], Ps, P)
        info["static_parents"] = sp
    elif cfg["on_change"] in ("chain", "capchain") and masks["xrun"].any() \
            and S.target.startswith("throughput_"):
        P = np.where(masks["xrun"][:, None], chain_predict(S, cfg, binner, tr, te), P)
    elif cfg["on_change"] not in ("same", "static", "chain", "capchain"):
        raise ValueError("on_change must be same | static | chain | capchain")

    out = {}
    run_te, y1te, y0c = S.run[te], S.y1[te], S.y0[te]

    def add(name, Pm, point=None):
        L = per_sample(Pm, yte, y1te, bin_value, y0c, y0te)
        if point is not None:           # model with a native continuous forecast
            L["aeh"] = np.abs(point - y1te)
        summ = summarise(L, masks)
        summ["f1_all"] = macro_f1(yte, L["pred"], n_y)
        out[name] = (summ, per_run_table(L, run_te, masks))

    add("model", P)
    if cfg["numeric"]:
        # the reconfiguration steps keep the bin-based forecast (chain inference
        # when on_change='chain'); inside a run the read-outs below replace it
        fallback = np.where(masks["xrun"], P @ bin_value, y0c + P @ bin_value - bin_value[y0te])
        for name, point in numeric_models(S, cfg, binner, tr, te, parents).items():
            add(name, P, np.where(masks["xrun"], fallback, point))
    # AR(1) on the bins: the reference every configuration is compared with
    lv = Coder(S, {**cfg, "target_mode": "level"}, binner, tr)
    ar, _ = ar_model(lv("y@t"), lv.y, n_y, cfg["ess"])
    add("ar", ar.predict([code_te("y@t")[0]], n_te))
    # persistence: all mass on the current bin (its log-loss is not meaningful)
    Pp = np.zeros((n_te, n_y))
    Pp[np.arange(n_te), y0te] = 1.0
    add("persistence", Pp)
    # bin-free reference: persistence on the continuous value
    info["mae_persistence_continuous"] = float(np.abs(y0c - y1te).mean())
    info["mae_persistence_continuous_event"] = float(np.abs(y0c - y1te)[masks["event"]].mean()) \
        if masks["event"].any() else np.nan
    for name, fn in extra_models:
        # fn returns bin probabilities, or (bin probabilities, continuous forecast)
        res = fn(S, cfg, tr, te, binner, cands, parents)
        add(name, *res) if isinstance(res, tuple) else add(name, res)
    return out, info


# ============================================================
# REFERENCE MODELS ON THE SAME INPUTS (not Bayesian networks)
# ============================================================
def integer_target(S):
    """True when the target only takes whole values (a count per second at 1 s)."""
    return bool(np.all(S.y1 == np.round(S.y1)))


def _point_to_probs(pred, binner, integer=False):
    # A continuous forecast of 199.97 for a count that is 200 belongs to the
    # bin of 200, not to the one below it.
    code = binner.code_y(np.round(pred) if integer else pred)
    P = np.full((len(pred), binner.n_y), 1e-6)
    P[np.arange(len(pred)), code] = 1.0
    return P / P.sum(1, keepdims=True)


def hgb_model(S, cfg, tr, te, binner, cands, parents=None):
    """
    Gradient boosting on the continuous candidates, predicting the change
    y(t+h) - y(t). A point forecast: its accuracy and MAE are meaningful, its
    log-loss is not (all mass is put on one bin).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.column_stack([S.cols[c] for c in cands])
    m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                      random_state=0)
    m.fit(X[tr], S.y1[tr] - S.y0[tr])
    point = S.y0[te] + m.predict(X[te])
    return _point_to_probs(point, binner, integer_target(S)), point


# ============================================================
# STUDIES
# ============================================================
def study_ofat():
    """One factor at a time around the reference configuration (DEFAULTS)."""
    c = [make_cfg()]
    # discretisation
    c += [make_cfg(n_bins=b) for b in (4, 6, 10, 30, 50, 100)]
    c += [make_cfg(disc=d) for d in ("quantile", "kmeans")]
    c += [make_cfg(disc_x=d) for d in ("uniform", "kmeans")]
    c += [make_cfg(n_bins_x=b) for b in (6, 10)]
    c += [make_cfg(shared_bins=False)]
    # length of the state
    c += [make_cfg(ar_order=p) for p in (2, 3, 5)]
    c += [make_cfg(other_lags=p) for p in (2, 3, 5)]
    c += [make_cfg(ar_order=3, other_lags=3)]
    c += [make_cfg(velocity=v) for v in ("sign", "binned")]
    c += [make_cfg(velocity="binned", ar_order=2, other_lags=2)]
    c += [make_cfg(relative=True), make_cfg(target_mode="delta"),
          make_cfg(target_mode="delta", relative=True)]
    # what is known about t+h, which variables are offered
    c += [make_cfg(exog=e) for e in ("none", "controls")]
    c += [make_cfg(pool="all")]
    # structure learning
    engines = ("ar", "hc-bic", "hc-bic-obs", "hc-aic", "hc-bdeu", "cmi", "lasso", "dynotears")
    c += [make_cfg(engine=e) for e in engines]
    c += [make_cfg(engine=e, prior="bdeu") for e in engines + ("cv-forward",)]
    c += [make_cfg(engine="fixed", fixed_parents=EXPERT_CHAIN)]
    c += [make_cfg(max_parents=m) for m in (2, 3, 5, 6)]
    c += [make_cfg(forced=())]
    # table estimation
    c += [make_cfg(ess=e) for e in (1.0, 3.0, 30.0, 100.0)]
    # configuration changes
    c += [make_cfg(cross_run=False), make_cfg(on_change="static"), make_cfg(on_change="chain")]
    return c


def study_bins_granularity():
    """Bins x granularity: reference model, and the pgmpy-equivalent (hc-bic + bdeu)."""
    c = []
    for g, b in itertools.product((1, 5, 10, 30), (4, 6, 10, 20, 30, 50)):
        c.append(make_cfg(granularity=g, n_bins=b))
        c.append(make_cfg(granularity=g, n_bins=b, engine="hc-bic", prior="bdeu"))
    return c


def study_horizon():
    """How far ahead: horizons in rows, at several granularities."""
    c = [make_cfg(horizon=h) for h in (1, 2, 5, 10, 30)]
    c += [make_cfg(granularity=g, horizon=h) for g, h in ((5, 1), (5, 2), (5, 6), (10, 1), (10, 3), (30, 1))]
    return c


def study_interactions():
    """Factorial over the options that interact: bins x state length x engine x table size."""
    c = []
    states = ((1, 1, "none"), (2, 2, "none"), (2, 2, "binned"), (3, 3, "binned"))
    for eng, nb, st, rel, mp in itertools.product(("cv-forward", "hc-aic"), (10, 20, 30, 50),
                                                  states, (False, True), (4, 6)):
        c.append(make_cfg(engine=eng, n_bins=nb, ar_order=st[0], other_lags=st[1],
                          velocity=st[2], relative=rel, max_parents=mp))
    return c


def study_cv():
    """Expanding-window versus blocked cross-validation."""
    return [make_cfg(cv=s, engine=e, prior=p) for s in ("expanding", "blocked")
            for e, p in (("cv-forward", "backoff"), ("hc-aic", "backoff"), ("hc-bic", "bdeu"))]


def study_refs():
    """Reference configurations for comparisons with non-BN models (use --with-hgb)."""
    return [make_cfg(granularity=g, n_bins=b) for g, b in ((1, 20), (1, 50), (30, 4), (30, 20))]


def study_config_change():
    """What predicts the first step after a reconfiguration (on_change), by granularity and bins."""
    return [make_cfg(granularity=g, n_bins=b, on_change=oc)
            for g, b in itertools.product((1, 30), (4, 10, 20))
            for oc in ("same", "static", "chain")]


def study_best():
    """The options that helped, combined: longer state + chain inference at reconfigurations."""
    c = []
    for g, b in ((1, 10), (1, 20), (1, 30), (1, 50), (30, 4), (30, 10), (30, 20)):
        c.append(make_cfg(granularity=g, n_bins=b))
        c.append(make_cfg(granularity=g, n_bins=b, ar_order=2, other_lags=2, on_change="chain"))
        c.append(make_cfg(granularity=g, n_bins=b, ar_order=3, other_lags=3, velocity="binned",
                          max_parents=6, on_change="chain"))
    return c


def study_targets():
    """The same model for every non-exogenous variable of the system (one table per variable)."""
    c = []
    for v in ("throughput_1", "throughput_2", "throughput_3", "avg_p_latency_1", "avg_p_latency_2",
              "avg_p_latency_3", "buffer_size_2", "buffer_size_3"):
        disc = "quantile" if v.startswith("buffer") else "uniform"
        for nb in (10, 20):
            c.append(make_cfg(target=v, n_bins=nb, disc=disc))
            c.append(make_cfg(target=v, n_bins=nb, disc=disc, ar_order=2, other_lags=2, on_change="chain"))
            c.append(make_cfg(target=v, n_bins=nb, disc=disc, engine="hc-bic", prior="bdeu"))
    return c


def study_constraints():
    """Domain constraints on the structure: restricted candidates and forced parents."""
    up = "throughput_2@t"
    ctrl = ("cores_3@t+h", "data_quality_3@t+h")
    c = [make_cfg(), make_cfg(pool="local"), make_cfg(pool="local", other_lags=2, ar_order=2),
         make_cfg(other_lags=2, ar_order=2),
         make_cfg(forced=("y@t", up)), make_cfg(forced=("y@t", up, "buffer_size_3@t")),
         make_cfg(forced=("y@t", up) + ctrl, max_parents=5),
         make_cfg(forced=("y@t", up) + ctrl, max_parents=6),
         make_cfg(forced=("y@t",) + ctrl), make_cfg(forced=("y@t",) + ctrl, max_parents=5),
         make_cfg(exog="none"), make_cfg(exog="none", pool="local")]
    c += [make_cfg(engine=e, pool="local") for e in ("hc-bic", "hc-aic", "hc-bdeu", "dynotears")]
    return c


def study_numeric():
    """Continuous read-outs of the DBN (conditional Gaussian / linear-Gaussian target node)."""
    c = []
    for nb in (10, 20, 50):
        c.append(make_cfg(n_bins=nb, numeric=True, on_change="chain"))
        c.append(make_cfg(n_bins=nb, numeric=True, on_change="chain", ar_order=2, other_lags=2))
        c.append(make_cfg(n_bins=nb, numeric=True, on_change="chain", ar_order=3, other_lags=3,
                          max_parents=6))
    c.append(make_cfg(granularity=30, n_bins=10, numeric=True, on_change="chain"))
    c.append(make_cfg(granularity=30, n_bins=10, numeric=True, on_change="chain", ar_order=2, other_lags=2))
    return c


def study_capacity():
    """A learned capacity node per service, inside a run and at reconfigurations."""
    c = []
    for g, b in itertools.product((1, 30), (4, 10, 20)):
        base = dict(granularity=g, n_bins=b, ar_order=2, other_lags=2)
        c.append(make_cfg(**base, on_change="chain"))
        c.append(make_cfg(**base, capacity=True, on_change="chain"))
        c.append(make_cfg(**base, capacity=True, on_change="capchain"))
        c.append(make_cfg(**base, capacity=True, on_change="capchain", pool="local"))
    for t in ("throughput_1", "throughput_2"):
        c.append(make_cfg(target=t, ar_order=2, other_lags=2, on_change="chain"))
        c.append(make_cfg(target=t, ar_order=2, other_lags=2, capacity=True, on_change="capchain"))
    return c


STUDIES = {"capacity": study_capacity, "targets": study_targets, "constraints": study_constraints, "numeric": study_numeric,
           "best": study_best, "config_change": study_config_change, "ofat": study_ofat, "bins_granularity": study_bins_granularity, "horizon": study_horizon,
           "interactions": study_interactions, "cv": study_cv, "refs": study_refs}


# ============================================================
# DRIVER
# ============================================================
_frame_cache = {}


def load_frame(csv, target, granularity, capacity=False):
    key = (str(csv), target, granularity) + (("capacity",) if capacity else ())
    if key not in _frame_cache:
        df = D.aggregate(D.load_wide(csv, target, verbose=False), granularity)
        if capacity:
            first_chunk = D.make_temporal_folds(df[D.META_COLS], k=5)[0][0]["run_id"].unique()
            rate, line = D.fit_capacity_rate(df, first_chunk)
            df = D.add_capacity(df, rate, line)
            df = df[D.META_COLS + [c for c in D.feature_cols(df) if c != target] + [target]]
        _frame_cache[key] = df
    return _frame_cache[key]


_sample_cache = {}


def get_samples(csv, cfg):
    key = (str(csv), cfg["target"], cfg["granularity"], cfg["horizon"], cfg["cross_run"], cfg["capacity"])
    if key not in _sample_cache:
        df = load_frame(csv, cfg["target"], cfg["granularity"], cfg["capacity"])
        S = Samples(df, cfg["target"], cfg["horizon"], cfg["cross_run"],
                    MAX_HISTORY.get(cfg["granularity"], 3))
        S.meta = df[D.META_COLS]
        _sample_cache[key] = S
    return _sample_cache[key]


def fold_runs(S, cfg):
    folds = D.make_temporal_folds(S.meta, k=cfg["folds"], scheme=cfg["cv"])
    return [(tr["run_id"].unique(), te["run_id"].unique()) for tr, te in folds]


def run_config(csv, cfg, extra_models=()):
    """All folds of one configuration. Returns (rows, per-run tables)."""
    S = get_samples(csv, cfg)
    rows, tables = [], []
    for fi, (train_runs, test_runs) in enumerate(fold_runs(S, cfg)):
        tr, te = split_masks(S, train_runs, test_runs)
        res, info = run_fold(S, cfg, tr, te, extra_models)
        for model, (summ, table) in res.items():
            rows.append({"config": cfg_id(cfg), "model": model, "fold": fi + 1,
                         "n_train": int(tr.sum()), **summ,
                         **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                            for k, v in info.items()}})
            table.insert(0, "fold", fi + 1)
            table.insert(0, "model", model)
            table.insert(0, "config", cfg_id(cfg))
            tables.append(table)
    return rows, tables


def compare_to_reference(per_run, config, model="model", ref_config=None, ref_model="ar",
                         n_boot=2000, seed=0):
    """
    Paired comparison on the runs both were tested on. The unit of resampling is
    the run (rows inside a run are strongly dependent; runs are separated by a
    reconfiguration). Returns the difference (candidate - reference) of
    accuracy, log-loss and MAE with a 95% bootstrap interval and the two-sided
    bootstrap p-value; negative is better for log-loss and MAE.
    """
    ref_config = config if ref_config is None else ref_config
    a = per_run[(per_run.config == config) & (per_run.model == model)].set_index("run_id")
    b = per_run[(per_run.config == ref_config) & (per_run.model == ref_model)].set_index("run_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(common), size=(n_boot, len(common)))
    out = {"n_runs": len(common)}
    for metric, name in (("correct", "acc"), ("ll", "logloss"), ("ae", "mae"), ("aeh", "maeh")):
        da, db, na, nb = a[metric].to_numpy(), b[metric].to_numpy(), a["n"].to_numpy(), b["n"].to_numpy()
        point = da.sum() / na.sum() - db.sum() / nb.sum()
        boot = da[draws].sum(1) / na[draws].sum(1) - db[draws].sum(1) / nb[draws].sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        out.update({f"d_{name}": point, f"d_{name}_lo": lo, f"d_{name}_hi": hi,
                    f"p_{name}": max(p, 1.0 / n_boot)})
    return out


def check_against_pgmpy(csv, n_bins=10):
    """
    Same parent set, same discretised data: the table / score code of this file
    against pgmpy (BayesianEstimator with a BDeu prior, VariableElimination,
    BicScore). Prints the largest disagreement.
    """
    from pgmpy.estimators import BayesianEstimator, BicScore
    from pgmpy.inference import VariableElimination
    from pgmpy.models import BayesianNetwork

    cfg = make_cfg(granularity=30, n_bins=n_bins, prior="bdeu")
    S = get_samples(csv, cfg)
    train_runs, test_runs = fold_runs(S, cfg)[-1]
    tr, te = split_masks(S, train_runs, test_runs)
    binner = Binner(S, cfg, tr)
    c_tr, c_te = Coder(S, cfg, binner, tr), Coder(S, cfg, binner, te)
    parents = ["y@t", "throughput_2@t", "cores_3@t+h"]
    names = {p: f"p{i}" for i, p in enumerate(parents)}

    cpt = fit_table(cfg, c_tr, parents, None)
    P_here = cpt.predict([c_te(p)[0] for p in parents], c_te.n)

    frame = pd.DataFrame({names[p]: c_tr(p)[0] for p in parents})
    frame["y1"] = c_tr.y
    states = {names[p]: list(range(c_tr(p)[1])) for p in parents}
    states["y1"] = list(range(c_tr.n_y))
    model = BayesianNetwork([(names[p], "y1") for p in parents])
    model.fit(frame, estimator=BayesianEstimator, prior_type="BDeu",
              equivalent_sample_size=cfg["ess"], state_names=states)
    infer = VariableElimination(model)
    E = np.column_stack([c_te(p)[0] for p in parents])
    uniq, inv = np.unique(E, axis=0, return_inverse=True)
    P_u = np.array([infer.query(["y1"], evidence={names[p]: int(v) for p, v in zip(parents, row)},
                                show_progress=False).values for row in uniq])
    P_pgmpy = P_u[np.asarray(inv).ravel()]
    print(f"probabilities: max |difference| = {np.abs(P_here - P_pgmpy).max():.2e} "
          f"over {c_te.n} test samples, {len(uniq)} distinct parent configurations")

    key = _keys([c_tr(p)[0] for p in parents], [c_tr(p)[1] for p in parents])
    _, N = _counts(key, c_tr.y, c_tr.n_y)
    q = float(np.prod([c_tr(p)[1] for p in parents]))
    here = local_score(N, q, c_tr.n_y, c_tr.n, "bic")
    theirs = BicScore(frame, state_names=states).local_score("y1", [names[p] for p in parents])
    print(f"BIC local score: here {here:.3f}  pgmpy {theirs:.3f}")


def _git_version():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--study", default="ofat", choices=sorted(STUDIES))
    ap.add_argument("--list-studies", action="store_true")
    ap.add_argument("--check-pgmpy", action="store_true",
                    help="compare this file's tables and scores with pgmpy on one fold, then exit")
    ap.add_argument("--with-hgb", action="store_true",
                    help="also fit the gradient-boosting reference on every configuration")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override a default for every configuration of the study, e.g. n_bins=30")
    args = ap.parse_args()

    if args.list_studies:
        for name, fn in STUDIES.items():
            print(f"{name}: {len(fn())} configurations -- {fn.__doc__.strip() if fn.__doc__ else ''}")
        return
    if not args.csv:
        ap.error("--csv is required")
    if args.check_pgmpy:
        check_against_pgmpy(args.csv)
        return

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = type(DEFAULTS[k])(v) if DEFAULTS[k] is not None and not isinstance(DEFAULTS[k], bool) \
            else json.loads(v.lower())
    DEFAULTS.update(overrides)
    configs = STUDIES[args.study]()

    stem = Path(args.csv).stem.replace("dbn_wide_", "")
    tag = f"{args.study}_{stem}_run{datetime.now():%Y%m%d_%H%M%S}_g{_git_version()}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    extra = (("hgb", hgb_model),) if args.with_hgb else ()

    rows, tables, seen = [], [], set()
    for n, cfg in enumerate(configs):
        cid = cfg_id(cfg)
        if cid in seen:
            continue
        seen.add(cid)
        t0 = time.perf_counter()
        try:
            r, t = run_config(args.csv, cfg, extra)
        except Exception as e:                       # keep the study going, report at the end
            print(f"[{n + 1}/{len(configs)}] {cid}: FAILED {type(e).__name__}: {e}", flush=True)
            rows.append({"config": cid, "model": "model", "error": f"{type(e).__name__}: {e}"})
            continue
        rows += r
        tables += t
        m = pd.DataFrame(r)
        g = m.groupby("model")[["acc_all", "logloss_all", "mae_all", "acc_change"]].mean()
        print(f"[{n + 1}/{len(configs)}] {cid}  ({time.perf_counter() - t0:.0f}s)\n"
              f"    model acc={g.loc['model', 'acc_all']:.4f} ll={g.loc['model', 'logloss_all']:.4f} "
              f"mae={g.loc['model', 'mae_all']:.2f} acc@change={g.loc['model', 'acc_change']:.3f} | "
              f"AR acc={g.loc['ar', 'acc_all']:.4f} ll={g.loc['ar', 'logloss_all']:.4f} "
              f"mae={g.loc['ar', 'mae_all']:.2f} | persistence acc={g.loc['persistence', 'acc_all']:.4f}",
              flush=True)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{tag}_folds.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{tag}_folds.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(RESULTS_DIR / f"{tag}_runs.csv.gz", index=False)
    print(f"\nSaved {RESULTS_DIR / (tag + '_folds.csv')}\nSaved {RESULTS_DIR / (tag + '_runs.csv.gz')}")


if __name__ == "__main__":
    main()
