"""
whatif_study.py -- "what throughput will this configuration give under this
load?", for configurations the model has not been trained on.

No history is used: every model sees only the load (rps) and the cores and data
quality of the three services, and predicts the throughput of each service.
Rows are 30-second means inside runs; folds are whole runs in time order, and
because the experiment visits each combination of cores for ten consecutive
runs, most test combinations were never seen in training.

Models
  table chain      P(throughput_i | upstream flow, cores_i, data_quality_i) for the
                   three services, chained from the load, upstream flows summed
                   out (uniform prior). Cores and quality are just labels to it.
  capacity chain   the same network with ONE change: (cores_i, data_quality_i)
                   are first turned into a capacity node,
                   capacity_i = cores_i x rate(data_quality_i), with the rate
                   function learned from the saturated runs of the training
                   data (dbn/data.py). Tables P(throughput_i | upstream flow,
                   capacity_i) are then chained as before.
                   "noisy-min prior": the prior of those tables is centred on
                   min(upstream, capacity) instead of being uniform.
  min rule         no tables: throughput_i = min(upstream, capacity_i). The
                   deterministic bottleneck formula with the learned capacities.
  boosting         gradient boosting on (rps, cores, quality); and on (rps, capacities)
  tabpfn           TabPFNRegressor on (rps, cores, quality)   [--tabpfn VERSION]

Usage:
    python analysis/whatif_study.py --csv data/dbn_wide_<stem>.csv [--tabpfn v3.5]
"""
import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
import ablation as A  # noqa: E402
import data as D  # noqa: E402

RESULTS_DIR = A.REPO_ROOT / "results" / "whatif"
TARGETS = ["throughput_1", "throughput_2", "throughput_3"]
CONFIG = [D.LOAD_COL] + [f"{p}_{i}" for i in (1, 2, 3) for p in ("cores", "data_quality")]
CAPS = [D.LOAD_COL] + [f"capacity_{i}" for i in (1, 2, 3)]


def codes_for(x, edges):
    return np.searchsorted(edges, x, side="right").astype(np.int64)


def levels_for(x, levels):
    idx = np.clip(np.searchsorted(levels, x), 0, len(levels) - 1)
    left = np.clip(idx - 1, 0, len(levels) - 1)
    return np.where(np.abs(x - levels[left]) < np.abs(x - levels[idx]), left, idx).astype(np.int64)


def noisy_min_table(n_bins, spread=0.1):
    """Prior of a flow node given min(upstream, capacity): that bin, a little on its neighbours."""
    T = np.eye(n_bins) * (1 - spread)
    for k in range(n_bins):
        nb = [j for j in (k - 1, k + 1) if 0 <= j < n_bins]
        T[k, nb] += spread / len(nb)
    return T


def chain(train, test, edges, parents_of, n_bins, ess=10.0, noisy_min=False):
    """
    Chained tables; parents_of(i) -> list of (column, 'flow' | 'level'). Returns {target: P}.

    noisy_min (capacity parents only): a service emits what it receives or what
    it can process, whichever is smaller. The table is learned from data as
    usual, but its prior is centred on the bin of min(upstream, capacity)
    instead of being uniform, so a combination never seen in training predicts
    what the bottleneck rule says.
    """
    def code(df, col, kind):
        if kind == "flow":
            return codes_for(df[col].to_numpy(), edges), n_bins
        lv = np.unique(train[col].to_numpy())
        return levels_for(df[col].to_numpy(), lv), len(lv)

    out, P_up, n = {}, None, len(test)
    for i in (1, 2, 3):
        up = D.LOAD_COL if i == 1 else f"throughput_{i - 1}"
        extra = parents_of(i)
        tr = [code(train, up, "flow")] + [code(train, c, k) for c, k in extra]
        y = codes_for(train[f"throughput_{i}"].to_numpy(), edges)
        te_extra = [code(test, c, k)[0] for c, k in extra]
        if noisy_min:
            # min(upstream, capacity) as a deterministic extra parent: it adds no
            # parent combination, it only tells the prior where to put its mass
            tr_codes = [c for c, _ in tr] + [np.minimum(tr[0][0], tr[1][0])]
            cpt = A.CPT(tr_codes, [k for _, k in tr] + [n_bins], y, n_bins, "backoff", ess,
                        (2, noisy_min_table(n_bins)))

            def predict(up_code):
                return cpt.predict([up_code] + te_extra + [np.minimum(up_code, te_extra[0])], n)
        else:
            cpt = A.CPT([c for c, _ in tr], [k for _, k in tr], y, n_bins, "bdeu", ess)

            def predict(up_code):
                return cpt.predict([up_code] + te_extra, n)
        if P_up is None:
            P_up = predict(code(test, up, "flow")[0])
        else:
            P_up = sum(P_up[:, [s]] * predict(np.full(n, s)) for s in range(n_bins))
        out[f"throughput_{i}"] = P_up
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--granularity", type=int, default=30)
    ap.add_argument("--cv", default="expanding", choices=["expanding", "blocked"])
    ap.add_argument("--tabpfn", default=None, help="TabPFN model version to include, e.g. v2 or v3.5")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = D.aggregate(D.load_wide(args.csv, "throughput_3", verbose=False), args.granularity)
    stem = Path(args.csv).stem.replace("dbn_wide_", "")
    tag = f"whatif_{args.cv}_{stem}_run{datetime.now():%Y%m%d_%H%M%S}_g{A._git_version()}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, rates = [], []

    for fi, (train, test) in enumerate(D.make_temporal_folds(df, k=5, scheme=args.cv)):
        rate, line = D.fit_capacity_rate(train)
        train, test = D.add_capacity(train, rate, line), D.add_capacity(test, rate, line)
        rates.append({"fold": fi + 1, **{f"rate_dq{int(q)}": v for q, v in rate.items()},
                      "loglog_slope": line[0]})
        seen = set(map(tuple, train[CONFIG[1:]].drop_duplicates().to_numpy()))
        new_cfg = np.array([tuple(r) not in seen for r in test[CONFIG[1:]].to_numpy()])

        pooled = np.concatenate([train[c].to_numpy() for c in [D.LOAD_COL] + TARGETS])
        edges = np.linspace(pooled.min(), pooled.max(), args.n_bins + 1)[1:-1]
        centre = {}
        for v in TARGETS:
            c = codes_for(train[v].to_numpy(), edges)
            s = np.bincount(c, weights=train[v].to_numpy(), minlength=args.n_bins)
            k = np.bincount(c, minlength=args.n_bins)
            mid = np.r_[edges[0], (edges[1:] + edges[:-1]) / 2, edges[-1]]
            centre[v] = np.where(k > 0, s / np.maximum(k, 1), mid)

        preds = {}        # model -> {target: (P or None, point)}
        tab = chain(train, test, edges, lambda i: [(f"cores_{i}", "level"), (f"data_quality_{i}", "level")],
                    args.n_bins)
        preds["table chain"] = {v: (P, P @ centre[v]) for v, P in tab.items()}
        cap = chain(train, test, edges, lambda i: [(f"capacity_{i}", "flow")], args.n_bins)
        preds["capacity chain"] = {v: (P, P @ centre[v]) for v, P in cap.items()}
        capm = chain(train, test, edges, lambda i: [(f"capacity_{i}", "flow")], args.n_bins, noisy_min=True)
        preds["capacity chain, noisy-min prior"] = {v: (P, P @ centre[v]) for v, P in capm.items()}
        flow, mins = test[D.LOAD_COL].to_numpy(dtype=float), {}
        for i in (1, 2, 3):
            flow = np.minimum(flow, test[f"capacity_{i}"].to_numpy())
            mins[f"throughput_{i}"] = (None, flow.copy())
        preds["min rule"] = mins
        for name, cols in (("boosting (cores, quality)", CONFIG), ("boosting (capacities)", CAPS)):
            preds[name] = {}
            for v in TARGETS:
                m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                                  random_state=0).fit(train[cols], train[v])
                preds[name][v] = (None, m.predict(test[cols]))
        if args.tabpfn:
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
            preds[f"tabpfn {args.tabpfn}"] = {}
            ctx = train if len(train) <= 10000 else train.sample(10000, random_state=0)
            for v in TARGETS:
                m = TabPFNRegressor.create_default_for_version(ModelVersion(args.tabpfn))
                m.fit(ctx[CONFIG].to_numpy(), ctx[v].to_numpy())
                # the median, as for boosting (absolute error): the throughput under a
                # given configuration is skewed (a queue being drained), and the mean
                # of TabPFN's predictive distribution is far off here
                preds[f"tabpfn {args.tabpfn}"][v] = (None, m.predict(test[CONFIG].to_numpy(),
                                                                     output_type="median"))

        for model, per_target in preds.items():
            for v, (P, point) in per_target.items():
                y = test[v].to_numpy()
                yc = codes_for(y, edges)
                pred_bin = P.argmax(1) if P is not None else codes_for(point, edges)
                d = pd.DataFrame({
                    "run_id": test["run_id"].to_numpy(), "new_config": new_cfg, "n": 1.0,
                    "ae": np.abs(point - y), "correct": (pred_bin == yc).astype(float),
                    "ll": -np.log(np.clip(P[np.arange(len(y)), yc], 1e-12, 1)) if P is not None else np.nan,
                }).groupby(["run_id", "new_config"], as_index=False).sum(min_count=1)
                d.insert(0, "target", v)
                d.insert(0, "model", model)
                d.insert(0, "fold", fi + 1)
                rows.append(d)
        print(f"[fold {fi + 1}] test rows {len(test)}, of which unseen configuration: {new_cfg.mean():.0%}",
              flush=True)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(RESULTS_DIR / f"{tag}_runs.csv.gz", index=False)
    pd.DataFrame(rates).to_csv(RESULTS_DIR / f"{tag}_rates.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", lambda x: f"{x:.3f}"):
        for label, sub in (("all test rows", out), ("unseen configurations only", out[out.new_config])):
            g = sub.groupby(["target", "model"])[["n", "ae", "correct", "ll"]].sum(min_count=1)
            t = pd.DataFrame({"mae": g.ae / g.n, "acc": g.correct / g.n, "logloss": g.ll / g.n})
            print(f"\n== {label} ==")
            print(t.unstack(0).swaplevel(axis=1).sort_index(axis=1).to_string())
    print(pd.DataFrame(rates).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
