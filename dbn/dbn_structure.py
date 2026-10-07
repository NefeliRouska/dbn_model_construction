import numpy as np
import pandas as pd

from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, StructureScore
from pgmpy.models import BayesianNetwork

from data import pair_index, two_slice_frame


# ----------------------------
# CONFIG DEFAULTS
# ----------------------------
MAX_INDEGREE   = 4
HC_MAX_ITER    = 25000
HC_TABU_LENGTH = 100
HC_EPSILON     = 1e-4
HC_USE_CACHE   = True

# A conditional probability table with more columns than this is refused
# before pgmpy tries to allocate it (columns = product of the parents' numbers
# of states). 2e6 columns x 50 states is already ~1 GB.
MAX_CPT_COLUMNS = 2_000_000


def consecutive_pairs(n_rows, horizon=1):
    """(t, t+horizon) row pairs of a single uninterrupted series."""
    i = np.arange(max(n_rows - horizon, 0))
    return i, i + horizon


# ============================================================
# AIC SCORE
# ============================================================
class AicScoreCustom(StructureScore):
    """
    Discrete AIC score: maximize LL - k.
    Equivalent to minimizing AIC = -2LL + 2k.
    """

    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.state_names = {c: sorted(data[c].unique()) for c in data.columns}
        self.card = {c: len(self.state_names[c]) for c in data.columns}

    def local_score(self, variable, parents):
        df = self.data
        r_i = self.card[variable]

        if not parents:
            counts = (
                df[variable]
                .value_counts()
                .reindex(self.state_names[variable], fill_value=0)
                .to_numpy()
            )
            total = counts.sum()

            with np.errstate(divide="ignore", invalid="ignore"):
                p = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
                # Bug fix: np.log(p, where=(p > 0)) without out= leaves
                # uninitialized memory where p == 0, and nansum does NOT
                # catch that (it only skips real NaNs). Zero-init the
                # output explicitly so skipped positions are a defined 0.
                p_log = np.zeros_like(p)
                np.log(p, where=(p > 0), out=p_log)
                ll = np.sum(counts * p_log)

            return float(ll - (r_i - 1))

        group_cols = list(parents) + [variable]
        ct = df.groupby(group_cols).size().reset_index(name="n")
        pa_counts = ct.groupby(list(parents))["n"].sum().reset_index(name="n_pa")
        merged = ct.merge(pa_counts, on=list(parents), how="left")

        n = merged["n"].to_numpy(dtype=float)
        n_pa = merged["n_pa"].to_numpy(dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = n / n_pa
            frac_log = np.zeros_like(frac)
            np.log(frac, where=(frac > 0), out=frac_log)
            ll = np.sum(n * frac_log)

        q_i = 1
        for p in parents:
            q_i *= self.card[p]

        return float(ll - (r_i - 1) * q_i)


def make_score(score_name: str, df: pd.DataFrame):
    score_name = score_name.lower().strip()

    if score_name == "bic":
        return BicScore(df)

    if score_name == "aic":
        return AicScoreCustom(df)

    raise ValueError("score_name must be 'bic' or 'aic'")


# ============================================================
# BLACKLIST: DOMAIN CAUSAL ORDERING
# ============================================================
def build_blacklist(df):
    """
    Encodes domain causal ordering as forbidden edges.

    Layer 0: control parameters (cores_, data_quality_) and the offered load (rps)
    Layer 1: container resource metrics
    Layer 2: performance outcomes (throughput_, avg_p_latency_, buffer_size_)
    Layer 3: failure/error indicators

    Rules:
      - Layer 0 has no incoming edges.
      - Later layers cannot cause earlier layers.
      - throughput_i is not a parent of throughput_j for j < i.
    """
    all_vars = list(df.columns)

    def has_prefix(c, prefixes):
        c = c.lower()
        return any(c.startswith(p) for p in prefixes)

    def has_substring(c, subs):
        c = c.lower()
        return any(s in c for s in subs)

    layer0 = [v for v in all_vars if has_prefix(v, ["cores_", "data_quality_"]) or v == "rps"]

    layer1 = [
        v for v in all_vars
        if has_prefix(
            v,
            [
                "container_cpu_",
                "container_memory_",
                "container_network_",
                "container_fs_",
                "container_blkio_",
            ],
        )
    ]

    layer2 = [
        v for v in all_vars
        if has_prefix(v, ["throughput_", "avg_p_latency_", "buffer_size_"])
    ]

    layer3 = [v for v in all_vars if has_substring(v, ["fail", "oom", "scrape_error"])]

    layers = [layer0, layer1, layer2, layer3]
    layer_index = {v: i for i, layer in enumerate(layers) for v in layer}

    black = []

    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black.append((parent, child))

    for parent in all_vars:
        for child in all_vars:
            if parent == child:
                continue
            if (
                parent in layer_index
                and child in layer_index
                and layer_index[parent] > layer_index[child]
            ):
                black.append((parent, child))

    # Data flows down the chain: within a slice, a later service's throughput
    # is not a parent of an earlier one's.
    throughputs = sorted(v for v in all_vars if v.lower().startswith("throughput_"))
    for i, earlier in enumerate(throughputs):
        for later in throughputs[i + 1:]:
            black.append((later, earlier))

    return black


# ============================================================
# INTRA-SLICE EDGES
# ============================================================
def learn_intra_edges(
    df,
    score_name="bic",
    max_indegree=MAX_INDEGREE,
    max_iter=HC_MAX_ITER,
    tabu_length=HC_TABU_LENGTH,
    epsilon=HC_EPSILON,
    use_cache=HC_USE_CACHE,
):
    """
    Learn within-slice dependencies from single-slice data.
    """
    blacklist = build_blacklist(df)

    est = HillClimbSearch(df, use_cache=use_cache)
    best = est.estimate(
        scoring_method=make_score(score_name, df),
        max_indegree=max_indegree,
        black_list=blacklist,
        max_iter=max_iter,
        tabu_length=tabu_length,
        epsilon=epsilon,
        show_progress=False,
    )

    return list(best.edges())


# ============================================================
# INTER-SLICE EDGES
# ============================================================
def build_inter_only_blacklist(nodes, exog_cols=()):
    """
    Only allow edges of the form X_t -> Y_t1.
    Self-transition edges X_t -> X_t1 are allowed.
    Exogenous variables (set from outside the system) take no parent other
    than their own previous value, and may point at the other variables of
    their own slice (E_t1 -> Y_t1): their t1 value is known when the
    prediction is made, so it competes with the X_t parents for the same
    max_indegree slots.
    """
    black = []
    exog = set(exog_cols)

    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            black.append((f"{a}_t", f"{b}_t"))
            black.append((f"{a}_t1", f"{b}_t"))
            if not (a in exog and b not in exog):
                black.append((f"{a}_t1", f"{b}_t1"))
            if b in exog:
                black.append((f"{a}_t", f"{b}_t1"))

    return black


def learn_inter_edges_only(
    df,
    score_name="bic",
    horizon=1,
    max_indegree=MAX_INDEGREE,
    max_iter=HC_MAX_ITER,
    tabu_length=HC_TABU_LENGTH,
    epsilon=HC_EPSILON,
    use_cache=HC_USE_CACHE,
    pairs=None,
    exog_cols=(),
    extra_blacklist=(),
):
    """
    Learn inter-slice edges X_t -> Y_t1, where "t1" means t+horizon
    rather than strictly the next row. horizon=1 reproduces the
    original one-step-ahead behavior exactly.

    pairs: (rows at t, rows at t1). Default: consecutive rows of df, which is
    only right when df is one uninterrupted series.
    """
    nodes = list(df.columns)

    if pairs is None:
        pairs = consecutive_pairs(len(df), horizon)
    df_2s = two_slice_frame(df, pairs)

    blacklist = build_inter_only_blacklist(nodes, exog_cols) + list(extra_blacklist)

    est = HillClimbSearch(df_2s, use_cache=use_cache)
    best = est.estimate(
        scoring_method=make_score(score_name, df_2s),
        max_indegree=max_indegree,
        black_list=blacklist,
        max_iter=max_iter,
        tabu_length=tabu_length,
        epsilon=epsilon,
        show_progress=False,
    )

    return [(u, v) for (u, v) in best.edges() if v.endswith("_t1")]


# ============================================================
# GUARANTEE TARGET'S SELF-LOOP EDGE
# ============================================================
def ensure_self_loops(columns, inter_edges, target=None):
    """
    Force X_t -> X_t1 for a specific variable (target), or for every
    variable if target is not given.

    Why: the greedy inter-edge search sometimes drops the direct
    self-transition edge for the target variable entirely, leaving
    the model unable to reproduce even "predict the same as last
    step" for it. Confirmed reproducible on both this telemetry data
    and separately on Sachs benchmark data.

    Scoped to target rather than every variable: forcing a self-loop
    adds a parent to that variable's _t1 CPD, which shrinks the
    effective data per cell. Paying that cost only for the variable
    being predicted, not for every variable in the dataset.
    """
    existing = set(inter_edges)
    forced = []

    vars_to_force = [target] if target is not None else list(columns)

    for v in vars_to_force:
        edge = (f"{v}_t", f"{v}_t1")
        if edge not in existing:
            forced.append(edge)

    return inter_edges + forced


# ============================================================
# FIT 2-SLICE DBN
# ============================================================
def backoff_cpd(model, df_2s, state_names, node, fallback_parent, ess):
    """
    Table of `node` with a Dirichlet prior centred on P(node | fallback_parent)
    instead of on the uniform distribution: `ess` pseudo-observations per
    parent configuration, distributed as the one-parent table says. A parent
    configuration that was never (or rarely) seen in training then predicts
    what the autoregressive model would, not "all states equally likely".
    """
    parents = sorted(model.get_parents(node))
    cards = [len(state_names[p]) for p in parents]
    n_states, k = len(state_names[node]), len(state_names[fallback_parent])

    counts = np.zeros((k, n_states))
    np.add.at(counts, (df_2s[fallback_parent].to_numpy(), df_2s[node].to_numpy()), 1.0)
    counts += ess / (k * n_states)                       # BDeu-smoothed one-parent table
    table = counts / counts.sum(axis=1, keepdims=True)   # [state of fallback parent, state of node]

    # pgmpy orders the columns of a CPD as the product of the (sorted) parents'
    # states, last parent varying fastest
    fallback_state = np.unravel_index(np.arange(int(np.prod(cards))), cards)[parents.index(fallback_parent)]
    pseudo_counts = ess * table[fallback_state].T        # (n_states, n_configurations)

    estimator = BayesianEstimator(model, df_2s, state_names=state_names)
    return estimator.estimate_cpd(node, prior_type="dirichlet", pseudo_counts=pseudo_counts)


def fit_consistent_2slice_bn(df, intra_edges, inter_edges, horizon=1, pairs=None,
                             state_cards=None, protect=None, backoff=None, ess=10):
    """
    Build and fit a two-slice DBN, where "t1" means t+horizon.

    CPDs are fitted once from the full two-slice dataframe. We do NOT
    refit intra CPDs separately, because t1 nodes may have both
    intra-slice parents and inter-slice parents; replacing their CPDs
    with intra-only CPDs would make the CPD parent sets inconsistent
    with the graph.

    state_cards: {column: number of states}. Pass the discretiser's bin count
    so that a state that only shows up in the test data is still a valid
    state of the model. Default: the largest value seen in df, plus one.

    protect: node whose parent set must not be touched. If the table of any
    other node would exceed MAX_CPT_COLUMNS, parents are dropped from it (and
    reported); if the protected node's table is too large, this raises.

    backoff: (node, fallback_parent) -- estimate that node's table with
    backoff_cpd instead of the uniform BDeu prior.
    """
    intra_t = [(f"{u}_t", f"{v}_t") for (u, v) in intra_edges]
    intra_t1 = [(f"{u}_t1", f"{v}_t1") for (u, v) in intra_edges]

    edges_2s = intra_t + intra_t1 + inter_edges

    if pairs is None:
        pairs = consecutive_pairs(len(df), horizon)
    df_2s = two_slice_frame(df, pairs)

    model_2s = BayesianNetwork(edges_2s)

    # Explicit state names prevent silent state-order or missing-bin issues.
    state_cards = state_cards or {}
    state_names_2s = {}
    for v in df.columns:
        n_states = max(int(df[v].max()) + 1, int(state_cards.get(v, 0)))
        states = list(range(n_states))
        state_names_2s[f"{v}_t"] = states
        state_names_2s[f"{v}_t1"] = states

    def cpt_columns(node):
        return float(np.prod([float(len(state_names_2s[p]))
                              for p in model_2s.get_parents(node)]))

    inter_set = set(inter_edges)
    for node in list(model_2s.nodes()):
        if cpt_columns(node) <= MAX_CPT_COLUMNS:
            continue
        if node == protect:
            raise RuntimeError(
                f"CPT of {node} needs {cpt_columns(node):.3g} parent configurations "
                f"({len(model_2s.get_parents(node))} parents) for {len(df_2s)} training rows; "
                f"refusing to fit (limit {MAX_CPT_COLUMNS:.0e})."
            )
        # Any other node: a t1 node inherits its same-slice parents and gets
        # lagged ones on top. Drop same-slice parents first, largest first,
        # until the table fits; the self-loop is kept.
        base = node.rsplit("_", 1)[0]
        removable = sorted(
            (p for p in model_2s.get_parents(node) if p != f"{base}_t"),
            key=lambda p: ((p, node) in inter_set, -len(state_names_2s[p])),
        )
        dropped = []
        while cpt_columns(node) > MAX_CPT_COLUMNS and removable:
            p = removable.pop(0)
            model_2s.remove_edge(p, node)
            dropped.append(p)
        print(f"[CPT LIMIT] {node}: dropped parents {dropped}")
    edges_2s = list(model_2s.edges())

    model_2s.fit(
        df_2s,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10,
        state_names=state_names_2s,
    )

    # (with the fallback parent as the only parent the table already is the
    # autoregressive one, and the BDeu fit above is kept)
    if (backoff is not None and backoff[1] in model_2s.get_parents(backoff[0])
            and len(model_2s.get_parents(backoff[0])) > 1):
        model_2s.add_cpds(backoff_cpd(model_2s, df_2s, state_names_2s, backoff[0], backoff[1], ess))

    return model_2s, edges_2s


# ============================================================
# MAIN BUILDER
# ============================================================
def build_dbn_model_2s(
    df_ready,
    score_name="bic",
    target=None,
    lag_cols=None,
    control_col=None,
    target_parents_only=True,
    horizon=1,
    max_indegree=MAX_INDEGREE,
    max_iter=HC_MAX_ITER,
    tabu_length=HC_TABU_LENGTH,
    epsilon=HC_EPSILON,
    use_cache=HC_USE_CACHE,
    pairs=None,
    exog_cols=(),
    state_cards=None,
    cpt_prior="bdeu",
    extra_inter_blacklist=(),
):
    """
    lag_cols: list of column names already present in df_ready holding
    the target's own past values (e.g. ["throughput_3_lag1"]). Each
    gets its own forced edge into target_t1. These are lags relative
    to the CURRENT row (t), and are unaffected by horizon.

    control_col: name of a single column already present in df_ready
    holding a binary "did a control variable change last step" flag.
    Gets its own forced edge into target_t1, same mechanism as
    lag_cols -- this is information persistence and AR-DBN cannot see
    (a deliberate system action just happened), distinct from lag_cols
    (more history of the target's own past value).

    target_parents_only: which parents target_t1 may have.
      True / "ar_only": exactly {target_t} plus lag_cols plus control_col.
          The learned structure then has no effect on the prediction of
          the target: the model is an autoregression on the target's bins.
      "observed": everything that is observed at prediction time -- any
          X_t -> target_t1 or E_t1 -> target_t1 (E exogenous, its t1 value
          is given as evidence) edge found by the two-slice search, and
          the forced lag / control edges. Same-slice edges from other
          variables are dropped: their t1 value is not known when the
          prediction is made.
      False / "learned": whatever structure learning returns.

    pairs: (rows at t, rows at t1) used for everything that looks one
    step ahead; see data.pair_index. Default: consecutive rows.

    exog_cols: columns set from outside the system (cores, data quality,
    offered load). They get no learned parents and may be same-slice
    parents of the target in "observed" mode.

    state_cards: {column: number of states}, see fit_consistent_2slice_bn.

    extra_inter_blacklist: further forbidden edges for the two-slice search,
    as (parent node, child node) with the _t / _t1 suffixes.

    cpt_prior: "bdeu" (uniform prior, what pgmpy fits by default) or
    "backoff" (target_t1's table falls back to P(target_t1 | target_t), see
    backoff_cpd).

    horizon: how many rows ahead "t1" points to. horizon=1 is the
    original one-step-ahead model.
    """
    lag_cols = list(lag_cols) if lag_cols else []
    exclude_cols = list(lag_cols)
    if control_col is not None:
        exclude_cols = exclude_cols + [control_col]

    # Lag/control columns are excluded from the general within-slice
    # search. They only ever appear as forced parents of target_t1.
    intra_df = df_ready.drop(columns=exclude_cols) if exclude_cols else df_ready

    intra = learn_intra_edges(
        intra_df,
        score_name=score_name,
        max_indegree=max_indegree,
        max_iter=max_iter,
        tabu_length=tabu_length,
        epsilon=epsilon,
        use_cache=use_cache,
    )

    inter = learn_inter_edges_only(
        df_ready,
        score_name=score_name,
        horizon=horizon,
        max_indegree=max_indegree,
        max_iter=max_iter,
        tabu_length=tabu_length,
        epsilon=epsilon,
        use_cache=use_cache,
        pairs=pairs,
        exog_cols=exog_cols,
        extra_blacklist=extra_inter_blacklist,
    )

    inter = ensure_self_loops(df_ready.columns, inter, target=target)

    if target is not None:
        for lag_col in lag_cols:
            forced = (f"{lag_col}_t", f"{target}_t1")
            if forced not in inter:
                inter = inter + [forced]
        if control_col is not None:
            forced = (f"{control_col}_t", f"{target}_t1")
            if forced not in inter:
                inter = inter + [forced]

    mode = {True: "ar_only", False: "learned"}.get(target_parents_only, target_parents_only)
    if mode not in ("ar_only", "observed", "learned"):
        raise ValueError("target_parents_only must be True/'ar_only', 'observed' or False/'learned'")

    if mode == "ar_only" and target is not None:
        keep = {f"{target}_t"} | {f"{lag_col}_t" for lag_col in lag_cols}
        if control_col is not None:
            keep.add(f"{control_col}_t")
        intra = [(u, v) for (u, v) in intra if v != target]
        inter = [(u, v) for (u, v) in inter
                 if v != f"{target}_t1" or u in keep]

    if mode == "observed" and target is not None:
        # same-slice parents of the target come from the two-slice search
        # (exogenous ones only), not from the single-slice graph
        intra = [(u, v) for (u, v) in intra if v != target]

    if mode == "ar_only" or not exog_cols:
        # no exogenous same-slice parents unless they are explicitly allowed
        inter = [(u, v) for (u, v) in inter if u.endswith("_t")]

    backoff = (f"{target}_t1", f"{target}_t") if (cpt_prior == "backoff" and target is not None) else None
    model_2s, edges = fit_consistent_2slice_bn(df_ready, intra, inter, horizon=horizon,
                                               pairs=pairs, state_cards=state_cards,
                                               protect=f"{target}_t1" if target is not None else None,
                                               backoff=backoff)

    return model_2s, edges, intra, inter