"""
rollout.py -- the DBN as one model of the whole chain, rolled forward in time.

State of the system:  throughput_1, throughput_2, throughput_3, buffer_size_2,
                      buffer_size_3   (each at t and t-1)
Exogenous inputs:     rps, cores_i, data_quality_i

Every state variable has one table P(v(t+1) | parents), with parents chosen
among the state at t and t-1 and the exogenous inputs at t+1. Together the
tables are a two-slice DBN, so the model can be unrolled: sample t+1 from t,
t+2 from the sampled t+1, and so on (a particle approximation of the DBN's
h-step-ahead distribution).

Structures compared
    learned    parents searched among the whole state
    local      parents restricted to the variable's own service and the flow
               that enters it (the physical chain)
    chain      'local' + a same-slice edge from the upstream throughput
               (throughput_{i-1}(t+1) -> throughput_i(t+1)), sampled in chain order
    anchored   'chain' + the service's own cores and data quality forced in as
               parents. Inside a run they are constant, so they add nothing one
               step ahead (and the search never picks them); over many steps
               they are what ties the throughput to the level the configuration
               allows, instead of letting it drift bin by bin.

    capacity   'chain' + the service's learned capacity
               (cores x rate(data_quality), see data.fit_capacity_rate) forced in
               as a parent: the anchor of 'anchored' as one variable on the flow
               bins instead of two labels
    level      'local' + a slow state variable per throughput: its exponential
               moving average ("the level it has been running at"), forced in as
               a parent and updated deterministically at every step. A table
               over bins of v(t) alone forgets the level after one noisy step;
               this gives the unrolled model something to return to.
    level?     the same, with the moving average offered to the search instead
               of forced

Forecasts compared at every horizon h, on the same test origins
    rollout:<structure>   one-step tables unrolled h times
    direct                a separate table P(v(t+h) | state at t, inputs at t+h)
    static_chain          no dynamics at all: P(throughput_i | upstream flow,
                          cores_i, data_quality_i) chained from the load at t+h,
                          the upstream flows summed out (what the configuration
                          and the load alone say)
    ar                    P(v(t+h) | v(t))
    persistence           v(t+h) = v(t)

Future load: 'known' (the rps of the next h seconds is given, as cores and data
quality are) or 'frozen' (rps is assumed to stay at its current value).

Usage:
    python dbn/rollout.py --csv data/dbn_wide_<stem>.csv
"""
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import ablation as A
import data as D

RESULTS_DIR = A.REPO_ROOT / "results" / "rollout"

FLOWS = [D.LOAD_COL, "throughput_1", "throughput_2", "throughput_3"]
STATE = ["throughput_1", "throughput_2", "throughput_3", "buffer_size_2", "buffer_size_3"]
TARGETS = ["throughput_1", "throughput_2", "throughput_3"]
HORIZONS = [1, 2, 5, 10, 30]
EMA_ALPHA = 0.1                      # weight of the newest second in the moving average


def ema_name(v):
    return f"ema_{v}"


def add_levels(df):
    """Exponential moving average of every throughput, restarted at the beginning of each run."""
    df = df.copy()
    for v in TARGETS:
        df[ema_name(v)] = df.groupby("run_id")[v].transform(
            lambda x: x.ewm(alpha=EMA_ALPHA, adjust=False).mean())
    return df


def service(v):
    return A.service_of(v)


def upstream(v):
    i = service(v)
    return D.LOAD_COL if i == 1 else f"throughput_{i - 1}"


def allowed(v, structure, exog):
    """(state variables, exogenous variables) that may be parents of v."""
    if structure == "learned":
        return STATE, exog
    i = service(v)
    structure = structure.rstrip("?")
    own = [s for s in STATE if service(s) == i]
    up = upstream(v) if v.startswith("throughput_") else f"throughput_{i - 1}"
    state = own + ([up] if up in STATE else [])
    if structure == "level" and v.startswith("throughput_"):
        state = state + [ema_name(v)]
    ex = [e for e in exog if service(e) == i or (e == D.LOAD_COL and up == D.LOAD_COL)]
    return state, ex


# ------------------------------------------------------------------
# one consistent discretisation of every variable
# ------------------------------------------------------------------
class Codes:
    def __init__(self, df, train_rows, n_bins):
        self.code, self.card, self.value = {}, {}, {}
        self.flow_edges = None
        pooled = np.concatenate([df[v].to_numpy()[train_rows] for v in FLOWS])
        flow_edges = np.linspace(pooled.min(), pooled.max(), n_bins + 1)[1:-1]
        self.flow_edges = flow_edges
        for v in A.D.feature_cols(df):
            x = df[v].to_numpy(dtype=float)
            xt = x[train_rows]
            if v in FLOWS or v.startswith(("ema_", "capacity_")):
                c, k = np.searchsorted(flow_edges, x, side="right"), n_bins
            elif len(np.unique(xt)) <= max(n_bins, 12):
                lv = np.unique(xt)
                idx = np.clip(np.searchsorted(lv, x), 0, len(lv) - 1)
                left = np.clip(idx - 1, 0, len(lv) - 1)
                c, k = np.where(np.abs(x - lv[left]) < np.abs(x - lv[idx]), left, idx), len(lv)
            else:
                e = np.unique(np.quantile(xt, np.linspace(0, 1, n_bins + 1)[1:-1]))
                c, k = np.searchsorted(e, x, side="right"), len(e) + 1
            self.code[v], self.card[v] = c.astype(np.int64), k
            s = np.bincount(c[train_rows], weights=xt, minlength=k)
            n = np.bincount(c[train_rows], minlength=k)
            self.value[v] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
            self.value[v] = pd.Series(self.value[v]).interpolate(limit_direction="both").to_numpy()


def parse(name, target):
    """Harness column name -> (variable, 0 | 1 | 'next')."""
    var, when = name.split("@")
    var = target if var == "y" else var
    return var, {"t": 0, "t-1": 1, "t+h": "next"}[when]


def select_structure(csv, v, structure, tr_runs, te_runs, n_bins, horizon, exog):
    """Parent names for v from the harness's search, restricted to what can be rolled forward."""
    cfg = A.make_cfg(target=v, n_bins=n_bins, horizon=horizon, cross_run=False, ar_order=2, capacity=True,
                     other_lags=2, disc="quantile" if v.startswith("buffer") else "uniform")
    S = A.get_samples(csv, cfg)
    tr, _ = A.split_masks(S, tr_runs, te_runs)
    state, ex = allowed(v, "local" if structure in ("chain", "anchored", "capacity") else structure, exog)
    cands = []
    for u in state:
        name = "y" if u == v else u
        cands += [f"{name}@t"] if u.startswith("ema_") else [f"{name}@t", f"{name}@t-1"]
    cands += [f"{e}@t+h" for e in ex]
    if D.LOAD_COL in ex:
        cands.append(f"{D.LOAD_COL}@t")
    forced = ["y@t"]
    if structure in ("chain", "anchored", "capacity") and v.startswith("throughput_") and horizon == 1:
        forced.append(f"{upstream(v)}@t+h")            # same-slice edge from the upstream flow
    if structure == "level" and v.startswith("throughput_"):
        forced.append(f"{ema_name(v)}@t")
    if structure == "capacity" and v.startswith("throughput_"):
        forced.append(f"capacity_{service(v)}@t+h")
    if structure == "anchored" and v.startswith("throughput_"):
        i = service(v)
        forced += [f"cores_{i}@t+h", f"data_quality_{i}@t+h"]
        cfg = {**cfg, "max_parents": 5}
    binner = A.Binner(S, cfg, tr)
    parents, _ = A.select_parents(S, cfg, tr, A.Coder(S, cfg, binner, tr), cands, forced)
    A._sample_cache.clear()                      # one Samples object is ~0.5 GB at 1 s
    return [parse(p, v) for p in parents]


class Table:
    def __init__(self, C, v, parents, i, j, ess=10.0):
        self.v, self.parents, self.n_y = v, parents, C.card[v]
        codes = [self.gather(C, p, i, j) for p in parents]
        cards = [C.card[p[0]] for p in parents]
        y = C.code[v][j]
        backoff = None
        if (v, 0) in parents and len(parents) > 1:
            _, table = A.ar_model((C.code[v][i], C.card[v]), y, self.n_y, ess)
            backoff = (parents.index((v, 0)), table)
        self.cpt = A.CPT(codes, cards, y, self.n_y, "backoff" if backoff else "bdeu", ess, backoff)

    @staticmethod
    def gather(C, p, i, j):
        var, when = p
        return C.code[var][j] if when == "next" else C.code[var][i - when]

    def predict(self, codes):
        return self.cpt.predict(codes, len(codes[0]))


def sample(P, rng):
    cum = np.cumsum(P, axis=1)
    return np.minimum((rng.random((len(P), 1)) * cum[:, -1:] > cum).sum(1), P.shape[1] - 1)


def rollout(C, tables, order, origins, horizons, load_future, exog, n_particles, rng, raw):
    """Particle roll-out. Returns {h: {v: P over bins (n_origins x n_bins)}}."""
    N, M = len(origins), n_particles
    rep = np.repeat(np.arange(N), M)
    cur = {v: np.repeat(C.code[v][origins], M) for v in STATE}
    prev = {v: np.repeat(C.code[v][origins - 1], M) for v in STATE}
    level = {ema_name(v): np.repeat(raw[ema_name(v)][origins], M) for v in TARGETS}
    out = {}
    for k in range(1, max(horizons) + 1):
        def exo(e, offset):
            # value of an exogenous input at t + offset (controls are constant inside a run)
            if e == D.LOAD_COL and load_future == "frozen":
                return np.repeat(C.code[e][origins], M)
            return np.repeat(C.code[e][origins + offset], M)

        new = {}
        for v in order:
            codes = []
            for var, when in tables[v].parents:
                if var in exog:
                    codes.append(exo(var, k if when == "next" else k - 1 - when))
                elif var.startswith("ema_"):
                    codes.append(np.searchsorted(C.flow_edges, level[var], side="right"))
                elif when == "next":
                    codes.append(new[var])             # same-slice parent, already sampled
                else:
                    codes.append(cur[var] if when == 0 else prev[var])
            new[v] = sample(tables[v].predict(codes), rng)
        prev, cur = cur, new
        for v in TARGETS:                              # deterministic update of the slow state
            level[ema_name(v)] = (1 - EMA_ALPHA) * level[ema_name(v)] + EMA_ALPHA * C.value[v][cur[v]]
        if k in horizons:
            out[k] = {}
            for v in TARGETS:
                counts = np.bincount(rep * C.card[v] + cur[v], minlength=N * C.card[v]).reshape(N, -1)
                # (smoothed for the log-loss, raw frequencies for the point forecast)
                out[k][v] = ((counts + 1.0 / C.card[v]) / (M + 1.0), counts / M)
    return out


def static_chain(C, i, j, origins, h, load_future, ess=10.0):
    """P(throughput_k at t+h | load and configuration at t+h), upstream flows summed out."""
    K, n = C.card["throughput_1"], len(origins)
    at = origins + h
    load = C.code[D.LOAD_COL][origins if load_future == "frozen" else at]
    out, P_up = {}, None
    for k in (1, 2, 3):
        up, v = (D.LOAD_COL if k == 1 else f"throughput_{k - 1}"), f"throughput_{k}"
        ctrl = [f"cores_{k}", f"data_quality_{k}"]
        cpt = A.CPT([C.code[up][j]] + [C.code[c][j] for c in ctrl],
                    [C.card[up]] + [C.card[c] for c in ctrl], C.code[v][j], K, "bdeu", ess)
        ctrl_te = [C.code[c][at] for c in ctrl]
        if P_up is None:
            P_up = cpt.predict([load] + ctrl_te, n)
        else:
            P_up = sum(P_up[:, [s]] * cpt.predict([np.full(n, s)] + ctrl_te, n) for s in range(K))
        out[v] = P_up
    return out


def metrics(P, C, v, origins, h, raw):
    P, P_point = P if isinstance(P, tuple) else (P, P)
    y = C.code[v][origins + h]
    y0 = C.code[v][origins]
    ev = P_point @ C.value[v]
    point = raw[v][origins] + ev - C.value[v][y0]
    return dict(correct=(P_point.argmax(1) == y).astype(float),
                ll=-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)),
                ae=np.abs(point - raw[v][origins + h]),
                change=(y != y0).astype(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--n-origins", type=int, default=3000, help="test origins per fold")
    ap.add_argument("--particles", type=int, default=100)
    ap.add_argument("--structures", nargs="+",
                    default=["learned", "local", "chain", "anchored", "capacity", "level", "level?"])
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    df = add_levels(A.load_frame(args.csv, "throughput_3", 1, capacity=True))
    # the harness builds its candidate columns from this frame, for every target
    for v in STATE:
        A._frame_cache[(str(args.csv), v, 1)] = df[D.META_COLS + [c for c in D.feature_cols(df) if c != v] + [v]]
        A._frame_cache[(str(args.csv), v, 1, "capacity")] = A._frame_cache[(str(args.csv), v, 1)]
    raw = {v: df[v].to_numpy(dtype=float) for v in D.feature_cols(df)}
    run, pos = df["run_id"].to_numpy(), df["pos"].to_numpy()
    exog = D.exog_cols(D.feature_cols(df), "controls+load")
    H = max(HORIZONS)
    stem = Path(args.csv).stem.replace("dbn_wide_", "")
    tag = f"rollout_b{args.n_bins}_{stem}_run{datetime.now():%Y%m%d_%H%M%S}_g{A._git_version()}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, structures = [], []

    folds = D.make_temporal_folds(df[D.META_COLS], k=5)
    for fi, (tr_meta, te_meta) in enumerate(folds):
        tr_runs, te_runs = tr_meta["run_id"].unique(), te_meta["run_id"].unique()
        in_tr, in_te = np.isin(run, tr_runs), np.isin(run, te_runs)
        C = Codes(df, in_tr, args.n_bins)
        room = np.zeros(len(run), dtype=int)                 # rows left in the run after this one
        room[:] = (df.groupby("run_id")["pos"].transform("max").to_numpy() - pos)

        def pairs(h, rows_ok):
            i = np.nonzero(rows_ok & (pos >= 1) & (room >= h))[0]
            return i, i + h

        i1, j1 = pairs(1, in_tr)
        origins = np.nonzero(in_te & (pos >= 4) & (room >= H))[0]
        if len(origins) > args.n_origins:
            origins = np.sort(np.random.default_rng(fi).choice(origins, args.n_origins, replace=False))
        run_o = run[origins]

        def record(name, load, h, v, P):
            m = metrics(P, C, v, origins, h, raw)
            d = pd.DataFrame({"run_id": run_o, "n": 1.0, **m}).groupby("run_id", as_index=False).sum()
            d.insert(0, "horizon", h)
            d.insert(0, "target", v)
            d.insert(0, "load_future", load)
            d.insert(0, "method", name)
            d.insert(0, "fold", fi + 1)
            rows.append(d)

        # --- references and direct h-step tables ---
        for h in HORIZONS:
            ih, jh = pairs(h, in_tr)
            for v in TARGETS:
                Pp = np.zeros((len(origins), C.card[v]))
                Pp[np.arange(len(origins)), C.code[v][origins]] = 1.0
                record("persistence", "known", h, v, Pp)
                ar = Table(C, v, [(v, 0)], ih, jh)
                record("ar", "known", h, v, ar.predict([C.code[v][origins]]))
                parents = select_structure(args.csv, v, "learned", tr_runs, te_runs, args.n_bins, h, exog)
                t = Table(C, v, parents, ih, jh)
                codes = [Table.gather(C, p, origins, origins + h) for p in parents]
                record("direct", "known", h, v, t.predict(codes))
            for load in ("known", "frozen"):
                for v, P in static_chain(C, i1, j1, origins, h, load).items():
                    record("static_chain", load, h, v, P)
                structures.append({"fold": fi + 1, "structure": f"direct_h{h}", "node": v,
                                   "parents": [f"{a}@{b}" for a, b in parents]})

        # --- one-step structures, rolled out ---
        for structure in args.structures:
            tables = {}
            for v in STATE:
                parents = select_structure(args.csv, v, structure, tr_runs, te_runs, args.n_bins, 1, exog)
                tables[v] = Table(C, v, parents, i1, j1)
                structures.append({"fold": fi + 1, "structure": structure, "node": v,
                                   "parents": [f"{a}@{b}" for a, b in parents]})
            for load in ("known",) if structure.startswith("level") else ("known", "frozen"):
                res = rollout(C, tables, STATE, origins, HORIZONS, load, exog, args.particles, rng, raw)
                for h in HORIZONS:
                    for v in TARGETS:
                        record(f"rollout:{structure}", load, h, v, res[h][v])
            print(f"[fold {fi + 1}] {structure} done", flush=True)

        out = pd.concat(rows, ignore_index=True)
        out.to_csv(RESULTS_DIR / f"{tag}_runs.csv.gz", index=False)
        pd.DataFrame(structures).to_csv(RESULTS_DIR / f"{tag}_structures.csv", index=False)

    out = pd.concat(rows, ignore_index=True)
    g = out.groupby(["target", "load_future", "method", "horizon"])[["n", "correct", "ll", "ae"]].sum()
    summ = pd.DataFrame({"acc": g.correct / g.n, "logloss": g.ll / g.n, "mae": g.ae / g.n})
    summ.to_csv(RESULTS_DIR / f"{tag}_summary.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        for metric in ("acc", "logloss", "mae"):
            print(f"\n== {metric} ==")
            print(summ[metric].unstack("horizon").round(3).to_string())


if __name__ == "__main__":
    main()
