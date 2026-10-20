"""
planner.py -- the Plan step of the adaptation loop: score every allowed
configuration with the DBN and pick the one with the highest expected utility.

Utility: the SLO fulfillment of Sedlak et al. (arXiv 2510.06882), adapted to
the chain. Each SLO is met to the share phi = min(measured / target, 1). In the
paper every service has a completion SLO (weight 1.0) and a quality SLO (0.5);
in the chain completion is measured once, end to end, so it carries the weight
of the three services' completion SLOs:

    completion   tp_3 / load, target 1, weight 3 x 1.0
    quality_i    q_i / 800,   target 1, weight 0.5   (one per service)

    U(a | load) = (3.0 * E[phi_completion | do(a), load] + 0.5 * sum_i phi_quality_i) / 4.5

(With weight 1.0 the three quality SLOs outweigh completion, and at high load
the planner kept quality and dropped most requests.)

E[phi_completion] comes from the reconfiguration chain of the model,
P(throughput_3 | load, capacity_1..3) with the upstream throughputs summed out
(same structure, bins and noisy-min prior), with its three tables refitted for
the question the planner asks -- what a configuration delivers once settled:

  - on WINDOW-second means (30 s, as in Part 1's unseen-configurations study)
    instead of single seconds: a queue of a few hundred requests lets a service
    emit far more than it receives for a few seconds, and the per-second
    tables learned those bursts;
  - with PLANNER_BINS bins of 10 req/s instead of the model's 20 of 25: with
    20, a capacity of 181 for a load of 196 falls in the load's bin and looks
    like full completion (simulator: 16% of seconds short with 20 bins, 7%
    with 50; 100 bins choose the same and take 3.5 times longer);
  - without windows that contain a second in which service 2 or 3 is DRAINING
    a backlog: its queue holds at least QUEUE requests and is shrinking. A
    draining service emits more than it receives (in 87% / 77% of those
    seconds for services 2 / 3, against 2% when a long queue is growing and
    ~1% otherwise). Growing queues stay in: they are what an overloaded
    service looks like.

With the model's own per-second chain, one run with a 47,000-request backlog
made the table say "service 2 emits ~263 req/s although service 1 sends ~34",
and the search over 12,320 candidates found and exploited that entry (expected
throughput_3 of 242 req/s behind a service that can pass 34). That is right
for predicting the next second (Part 1), not for planning. The Part 1 model
itself is unchanged.

The load is the one at t; the load after the change is not known (commit 01de53d).

Candidates are the training support: ordered core triples with c_i >= 1 and
c1 + c2 + c3 <= 8 (56), times quality triples from {100, ..., 1000} with
q1 >= q2 >= q3 (220): 12,320 configurations.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "dbn"))
import ablation as A  # noqa: E402
import build_model as B  # noqa: E402
import data as D  # noqa: E402

TOTAL_CORES = 8
QUALITIES = list(range(100, 1001, 100))
SETTINGS = [f"{p}_{i}" for p in ("cores", "data_quality") for i in (1, 2, 3)]

W_COMPLETION, W_QUALITY, QUALITY_TARGET = 3.0, 0.5, 800.0
DELTA = 0.01          # act only if the best candidate beats the current one by more than this
QUEUE, LOOKBACK = 1000, 10   # draining: queue >= QUEUE and smaller than LOOKBACK seconds earlier
WINDOW = 30                  # seconds per row of the planner's tables
PLANNER_BINS = 50            # 10 req/s; the model's 20 bins (25 req/s) hide shortfalls inside a bin


def draining(frame, queue=QUEUE, lookback=LOOKBACK):
    """
    Boolean per row of one frame: service 2 or 3 is emptying a backlog. The
    queue is compared with LOOKBACK seconds earlier, or, in the first LOOKBACK
    seconds of a run (no earlier value), with LOOKBACK seconds later.
    """
    out = np.zeros(len(frame), dtype=bool)
    for i in (2, 3):
        b = frame[f"buffer_size_{i}"]
        g = b.groupby(frame["run_id"])
        before, after = g.shift(lookback), g.shift(-lookback)
        shrinking = (b < before) | (before.isna() & (after < b))
        out |= ((b >= queue) & shrinking).to_numpy()
    return out


def settled_windows(frames, window=WINDOW, queue=QUEUE, lookback=LOOKBACK):
    """WINDOW-second means of the frames, and per window whether it contains a draining second."""
    out = []
    for f in frames:
        f = f.assign(draining=draining(f, queue, lookback).astype(float))
        out.append(D.aggregate(f, window))
    df = pd.concat(out, ignore_index=True)
    return df, (df["draining"] > 0).to_numpy()


def settled_chain(model, frames, window=WINDOW, queue=QUEUE, lookback=LOOKBACK, n_bins=None):
    """
    The three chain tables of throughput_3, refitted on settled WINDOW-second
    means: P(throughput_i | upstream flow, capacity_i), same noisy-min prior
    and prior weight as the model's own chain.

    n_bins: None = the model's bins (20 of 25 req/s); otherwise n_bins of
    equal width over the training range of throughput_3, shared by the load,
    the throughputs and the capacities as in the model.

    Returns (chain, info); chain = {tables, edges, bin_value}.
    """
    node = model.nodes["throughput_3"]
    df, drain = settled_windows(frames, window, queue, lookback)
    df = df[~drain]
    if n_bins is None:
        edges = node.y_edges
    else:
        y = df["throughput_3"].to_numpy()
        edges = np.linspace(y.min(), y.max(), n_bins + 1)[1:-1]
    K, ess = len(edges) + 1, node.cfg["ess"]
    code = lambda x: np.searchsorted(edges, np.asarray(x, dtype=float), side="right")  # noqa: E731
    tables = []
    for i in (1, 2, 3):
        up = code(df[D.LOAD_COL] if i == 1 else df[f"throughput_{i - 1}"])
        cap = code(df[f"capacity_{i}"])
        tables.append(A.CPT([up, cap, np.minimum(up, cap)], [K, K, K], code(df[f"throughput_{i}"]),
                            K, "backoff", ess, (2, A.noisy_min_table(K))))
    y3, b3 = df["throughput_3"].to_numpy(), code(df["throughput_3"])
    cnt = np.bincount(b3, minlength=K)
    centers = np.r_[edges[:1], (edges[1:] + edges[:-1]) / 2, edges[-1:]]
    bin_value = np.where(cnt > 0, np.bincount(b3, weights=y3, minlength=K) / np.maximum(cnt, 1), centers)
    chain = dict(tables=tables, edges=edges, bin_value=bin_value)
    return chain, dict(settled_windows=int((~drain).sum()), all_windows=len(drain), window=window,
                       queue=queue, lookback=lookback, n_bins=K)


def chain_path(model_path, n_bins=None):
    stem = Path(model_path).stem + "_planner_chain" + (f"_{n_bins}bins" if n_bins else "")
    return Path(model_path).with_name(stem + ".pkl")


def build_settled_chain(model_path, out=None, window=WINDOW, queue=QUEUE, lookback=LOOKBACK, n_bins=None):
    """Fit the settled tables on the model's training files and save them next to the model."""
    model = B.Model.load(str(model_path))
    frames = [model.load_frame(REPO / "data" / f) for f in model.meta["files"]]
    chain, info = settled_chain(model, frames, window, queue, lookback, n_bins)
    out = Path(out or chain_path(model_path, n_bins))
    pd.to_pickle(dict(chain=chain, info=info, model=Path(model_path).name), out)
    return out, info


def load_chain(path):
    return pd.read_pickle(path)["chain"]


def candidates():
    """DataFrame of the 12,320 allowed configurations, columns SETTINGS."""
    cores = [c for c in itertools.product(range(1, TOTAL_CORES + 1), repeat=3) if sum(c) <= TOTAL_CORES]
    quals = [q for q in itertools.product(QUALITIES, repeat=3) if q[0] >= q[1] >= q[2]]
    rows = [(*c, *q) for c in cores for q in quals]
    return pd.DataFrame(rows, columns=SETTINGS)


class Planner:
    def __init__(self, model, chain=None, delta=DELTA):
        """chain: the settled chain (settled_chain / load_chain); None = the model's own chain."""
        self.model, self.chain, self.delta = model, chain, delta
        self.node = model.nodes["throughput_3"]
        if self.node.chain is None:
            raise ValueError("the model has no reconfiguration chain for throughput_3")
        self.edges = self.node.y_edges if chain is None else chain["edges"]
        self.bin_value = self.node.bin_value if chain is None else chain["bin_value"]
        self.cands = B.add_capacity(candidates(), model.capacity)
        self.phi_quality = np.minimum(self.cands[[f"data_quality_{i}" for i in (1, 2, 3)]].to_numpy()
                                      / QUALITY_TARGET, 1.0).sum(axis=1)

    def throughput_3(self, load, configs):
        """P(bin of throughput_3) for each row of `configs` (with capacity columns) under `load`."""
        n = len(configs)
        if self.chain is None:
            inputs = {f"{D.LOAD_COL}@t": np.full(n, float(load))}
            for i in (1, 2, 3):
                inputs[f"capacity_{i}@t+h"] = configs[f"capacity_{i}"].to_numpy(dtype=float)
            return self.node._predict_chain(inputs, np.ones(n, dtype=bool))
        edges = self.edges
        K = len(edges) + 1
        code = lambda x: np.searchsorted(edges, np.asarray(x, dtype=float), side="right")  # noqa: E731
        P_up = None
        for i, cpt in zip((1, 2, 3), self.chain["tables"]):
            cap = code(configs[f"capacity_{i}"])
            table = lambda up, cpt=cpt, cap=cap: cpt.predict([up, cap, np.minimum(up, cap)], n)  # noqa: E731
            if P_up is None:                               # the load is observed
                P_up = table(np.full(n, code(load)))
            else:                                          # sum over the unobserved upstream flow
                P_up = sum(P_up[:, [s]] * table(np.full(n, s)) for s in range(K))
        return P_up

    def expected_completion(self, load, P):
        phi = np.minimum(self.bin_value / max(float(load), 1e-9), 1.0)
        return P @ phi

    def score(self, load, configs=None, phi_quality=None):
        configs = self.cands if configs is None else configs
        if phi_quality is None:
            phi_quality = np.minimum(configs[[f"data_quality_{i}" for i in (1, 2, 3)]].to_numpy()
                                     / QUALITY_TARGET, 1.0).sum(axis=1)
        P = self.throughput_3(load, configs)
        completion = self.expected_completion(load, P)
        utility = (W_COMPLETION * completion + W_QUALITY * phi_quality) / (W_COMPLETION + 3 * W_QUALITY)
        return utility, completion, P

    def decide(self, load, current, row=None, observed=None):
        """
        current: {setting: value} of the configuration in place; row (the
        Monitor's row) is not used here, only by planners that read telemetry.
        observed: the completion measured for the current configuration (the
        loop passes it when the chain is failing). The current configuration
        is then scored by what it does, not by what the model predicts for it,
        so a configuration the model is too optimistic about is not kept.
        Returns a dict: act (bool), config (the one to apply, or the current one),
        utility / completion of the chosen and of the current configuration,
        and the predicted distribution of throughput_3 for the chosen one.

        Ties in utility are broken by fewer settings changed, then fewer cores.
        """
        U, comp, P = self.score(load, self.cands, self.phi_quality)
        cur = B.add_capacity(pd.DataFrame([current])[SETTINGS], self.model.capacity)
        U_cur, comp_cur, P_cur = self.score(load, cur)
        if observed is not None:
            q_cur = np.minimum(cur[[f"data_quality_{i}" for i in (1, 2, 3)]].to_numpy() / QUALITY_TARGET, 1.0).sum(axis=1)
            comp_cur = np.array([float(observed)])
            U_cur = (W_COMPLETION * comp_cur + W_QUALITY * q_cur) / (W_COMPLETION + 3 * W_QUALITY)

        X = self.cands[SETTINGS].to_numpy()
        changed = (X != cur[SETTINGS].to_numpy()).sum(axis=1)
        if observed is not None:
            # the current configuration among the candidates gets its observed utility too; otherwise
            # the best candidate can be the current configuration itself (by its prediction), and the
            # loop would re-send the settings it already has instead of trying the next best
            U = U.copy()
            U[changed == 0] = U_cur[0]
        n_cores = X[:, :3].sum(axis=1)
        best = np.lexsort((n_cores, changed, -np.round(U, 12)))[0]

        act = bool(U[best] - U_cur[0] > self.delta)
        k = best if act else None
        return dict(
            act=act,
            config={c: int(v) for c, v in zip(SETTINGS, X[best] if act else cur[SETTINGS].to_numpy()[0])},
            utility=float(U[best] if act else U_cur[0]), completion=float(comp[best] if act else comp_cur[0]),
            utility_current=float(U_cur[0]), completion_current=float(comp_cur[0]),
            utility_best=float(U[best]), completion_best=float(comp[best]),
            probs=None if P is None else (P[k] if act else P_cur[0]), bin_value=self.bin_value,
        )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fit and save the planner's settled chain for a model.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--queue", type=int, default=QUEUE)
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    ap.add_argument("--bins", type=int, default=PLANNER_BINS, help="0 = the model's bins")
    args = ap.parse_args()
    path, info = build_settled_chain(args.model, window=args.window, queue=args.queue, lookback=args.lookback,
                                     n_bins=args.bins or None)
    print(f"wrote {path}: {info['settled_windows']} of {info['all_windows']} {info['window']} s windows kept, {info['n_bins']} bins "
          f"({info['settled_windows'] / info['all_windows']:.1%}; dropped: a second with queue >= "
          f"{info['queue']} shrinking over {info['lookback']} s)")
