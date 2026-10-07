"""
adaptation_study.py -- a regime the model has never seen appears: can the DBN
notice it, and how fast does it recover if its tables are updated?

Regimes are the bottleneck regimes of analysis/regime_study.py (none / s1 / s23).

Part 1, detection with the whole network. Every non-exogenous variable has its
own table; its surprise is -log p(what happened). The alarm score of a run (or
of a step) is the sum of the surprises of a set of nodes:
    target only            throughput_3
    throughputs            throughput_1 + _2 + _3
    all nodes              + the three latencies and the two queue lengths
(a) run-level: models trained without regime r; AUC of the run's mean score
    for telling runs of r from runs of the regimes it knows. Two scores:
      surprise          mean of -log p(observed)
      |miscalibration|  absolute value of the mean of
                        (-log p(observed)) - (entropy of the prediction),
                        i.e. how far the surprise is from what the model itself
                        expected. A new regime can be EASIER than the known ones
                        (less surprise than expected); this score sees that too.
(a') how many seconds it takes: a two-sided CUSUM on the per-second
    miscalibration (surprise minus predicted entropy, summed over the nodes).
    Its reference level, slack and threshold are set on half of the test runs
    of the known regimes so that 5% of them raise an alarm at some point of
    their 6 minutes; reported are the false-alarm rate on the other half, the
    share of runs of the unseen regime that raise an alarm, and the median
    number of seconds until they do.
(b) step-level: models that are not told the configuration; AUC of the score
    on the first step after a reconfiguration against ordinary steps, and the
    share of reconfigurations caught at 1% false alarms.

Part 2, adaptation. The model for throughput_3 is trained on the first half of
the runs WITHOUT regime r, then the second half arrives run by run:
    static        never updated
    updated       every 5 runs the table is re-estimated on everything seen so
                  far (the structure is kept)
    restructured  as 'updated', and the parents are searched again every 20 runs
    informed      had regime r in its training data from the start, then updated
Accuracy on the runs of regime r, by how many runs of r had been seen before.

Usage:
    python analysis/adaptation_study.py --csv data/dbn_wide_<stem>.csv
"""
import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation as A  # noqa: E402
from regime_study import bottleneck_of_runs, predict, score  # noqa: E402

RESULTS_DIR = A.REPO_ROOT / "results" / "adaptation"
NODES = ["throughput_3", "throughput_2", "throughput_1", "avg_p_latency_1", "avg_p_latency_2",
         "avg_p_latency_3", "buffer_size_2", "buffer_size_3"]
SETS = {"target only": NODES[:1], "throughputs": NODES[:3], "all nodes": NODES}
BUCKETS = [(0, 0, "0"), (1, 5, "1-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 40, "21-40"),
           (41, 10 ** 6, ">40")]


def node_cfg(base, v, **kw):
    return A.make_cfg(**{**base, "target": v, "disc": "quantile" if v.startswith("buffer") else "uniform",
                         **kw})


def surprise(S, cfg, tr, te):
    binner = A.Binner(S, cfg, tr)
    c_tr, c_te = A.Coder(S, cfg, binner, tr), A.Coder(S, cfg, binner, te)
    _, ar_table = A.ar_model(c_tr("y@t"), c_tr.y_fit, c_tr.n_fit, cfg["ess"])
    parents, _ = A.select_parents(S, cfg, tr, c_tr, A.candidate_columns(S, cfg), list(cfg["forced"]))
    P = predict(S, cfg, binner, c_tr, c_te, parents, ar_table)
    entropy = -np.sum(P * np.log(np.clip(P, 1e-12, 1.0)), axis=1)
    return score(P, c_te.y)[1], entropy


def cusum_alarm_times(x, run, mu, slack, threshold):
    """First position (seconds into the run) at which a two-sided CUSUM exceeds threshold; inf if never."""
    out = {}
    order = np.argsort(run, kind="stable")
    bounds = np.flatnonzero(np.r_[True, run[order][1:] != run[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        z = x[order[a:b]] - mu
        up = dn = 0.0
        t_alarm = np.inf
        for t, v in enumerate(z):
            up = max(0.0, up + v - slack)
            dn = max(0.0, dn - v - slack)
            if up > threshold or dn > threshold:
                t_alarm = t + 1
                break
        out[run[order[a]]] = t_alarm
    return out


def cusum_rows(fi, r, run, novel, per_node):
    rows = []
    known_runs = np.unique(run[~novel])
    rng = np.random.default_rng(0)
    calib = set(rng.choice(known_runs, len(known_runs) // 2, replace=False).tolist())
    is_calib = np.isin(run, list(calib))
    for name, nodes in SETS.items():
        x = sum(per_node[v][0] - per_node[v][1] for v in nodes)
        mu, slack = x[is_calib].mean(), 0.5 * x[is_calib].std()
        # threshold: 5% of the calibration runs raise an alarm
        lo, hi = 0.0, 50.0 * max(slack, 1e-9) * 400
        cal_idx = is_calib
        for _ in range(30):
            mid = (lo + hi) / 2
            t = cusum_alarm_times(x[cal_idx], run[cal_idx], mu, slack, mid)
            if np.mean(np.isfinite(list(t.values()))) > 0.05:
                lo = mid
            else:
                hi = mid
        times = cusum_alarm_times(x[~is_calib], run[~is_calib], mu, slack, hi)
        novel_runs = set(np.unique(run[novel]).tolist())
        t_new = np.array([t for k, t in times.items() if k in novel_runs])
        t_old = np.array([t for k, t in times.items() if k not in novel_runs])
        if len(t_new) and len(t_old):
            rows.append({"fold": fi + 1, "unseen_regime": r, "nodes": name,
                         "false_alarm_runs": float(np.isfinite(t_old).mean()),
                         "detected_runs": float(np.isfinite(t_new).mean()),
                         "median_seconds_to_alarm": float(np.median(t_new[np.isfinite(t_new)]))
                         if np.isfinite(t_new).any() else np.nan,
                         "n_new_runs": len(t_new)})
    return rows


def detection(csv, base):
    """Returns (run-level rows, step-level rows, CUSUM rows)."""
    run_scores, step_scores, keys = {}, {}, {}
    for v in NODES:
        for mode, cfg in (("regime", node_cfg(base, v)), ("step", node_cfg(base, v, exog="none"))):
            S = A.get_samples(csv, cfg)
            label = bottleneck_of_runs(A.load_frame(csv, v, cfg["granularity"])).reindex(S.run).to_numpy()
            for fi, (train_runs, test_runs) in enumerate(A.fold_runs(S, cfg)):
                tr, te = A.split_masks(S, train_runs, test_runs)
                if mode == "step":
                    step_scores.setdefault(fi, {})[v] = surprise(S, cfg, tr, te)[0]
                    keys[("step", fi)] = S.cross[te]
                    continue
                for r in sorted(set(label)):
                    tr_not = tr & (label != r)
                    if (tr & (label == r)).sum() < 2000 or tr_not.sum() < 2000 or not (label[te] == r).any():
                        continue
                    run_scores.setdefault((fi, r), {})[v] = surprise(S, cfg, tr_not, te)
                    keys[("regime", fi, r)] = (S.run[te], label[te] == r)
            A._sample_cache.clear()
        print(f"[detection] {v} done", flush=True)

    run_rows, step_rows, cusum = [], [], []
    for (fi, r), per_node in run_scores.items():
        run, novel = keys[("regime", fi, r)]
        cusum += cusum_rows(fi, r, run, novel, per_node)
        for name, nodes in SETS.items():
            s = pd.DataFrame({"run": run, "novel": novel,
                              "s": sum(per_node[v][0] for v in nodes),
                              "excess": sum(per_node[v][0] - per_node[v][1] for v in nodes)}) \
                .groupby("run").mean()
            if 0 < (s["novel"] > 0.5).sum() < len(s):
                run_rows.append({"fold": fi + 1, "unseen_regime": r, "nodes": name,
                                 "auc_surprise": roc_auc_score(s["novel"] > 0.5, s["s"]),
                                 "auc_miscalibration": roc_auc_score(s["novel"] > 0.5, s["excess"].abs())})
    for fi, per_node in step_scores.items():
        change = keys[("step", fi)]
        for name, nodes in SETS.items():
            s = sum(per_node[v] for v in nodes)
            thr = np.quantile(s[~change], 0.99)
            step_rows.append({"fold": fi + 1, "score": name, "auc": roc_auc_score(change, s),
                              "caught_at_1pct_false_alarms": float((s[change] > thr).mean())})
    return pd.DataFrame(run_rows), pd.DataFrame(step_rows), pd.DataFrame(cusum)


def adaptation(csv, base):
    cfg = node_cfg(base, "throughput_3")
    S = A.get_samples(csv, cfg)
    label = bottleneck_of_runs(A.load_frame(csv, "throughput_3", cfg["granularity"])).reindex(S.run).to_numpy()
    runs = np.unique(S.run)
    first, stream = runs[: len(runs) // 2], runs[len(runs) // 2:]
    in_first = np.isin(S.run, first) & np.isin(S.run_i, first)
    cands, forced = A.candidate_columns(S, cfg), list(cfg["forced"])
    rows = []

    for r in sorted(set(label)):
        start = {"static": in_first & (label != r), "updated": in_first & (label != r),
                 "restructured": in_first & (label != r), "informed": in_first}
        if start["static"].sum() < 2000:
            continue
        binner = A.Binner(S, cfg, in_first)                     # one discretisation for all variants
        parents = {}
        for name, rows_tr in start.items():
            c = A.Coder(S, cfg, binner, rows_tr)
            parents[name], _ = A.select_parents(S, cfg, rows_tr, c, cands, forced)
        seen = np.zeros(len(S.run), dtype=bool)                 # stream samples already observed
        seen_r = 0
        for b in range(0, len(stream), 5):
            block = stream[b:b + 5]
            te = np.isin(S.run, block)
            c_te = A.Coder(S, cfg, binner, te)
            hit = label[te] == r
            for name in start:
                tr = start[name] if name == "static" else (start[name] | seen)
                c_tr = A.Coder(S, cfg, binner, tr)
                if name == "restructured" and b % 20 == 0 and b > 0:
                    parents[name], _ = A.select_parents(S, cfg, tr, c_tr, cands, forced)
                _, ar_table = A.ar_model(c_tr("y@t"), c_tr.y_fit, c_tr.n_fit, cfg["ess"])
                correct, ll = score(predict(S, cfg, binner, c_tr, c_te, parents[name], ar_table), c_te.y)
                if hit.any():
                    rows.append({"regime": r, "variant": name, "runs_of_regime_seen": seen_r,
                                 "n": int(hit.sum()), "correct": correct[hit].sum(), "ll": ll[hit].sum()})
            seen |= te & np.isin(S.run_i, np.r_[first, stream])
            seen_r += int(len(np.unique(S.run[te][hit])))
        print(f"[adaptation] regime {r} done", flush=True)
    out = pd.DataFrame(rows)
    out["bucket"] = ""
    for lo, hi, name in BUCKETS:
        out.loc[out.runs_of_regime_seen.between(lo, hi), "bucket"] = name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--part", choices=["detection", "adaptation", "both"], default="both")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    base = dict(ar_order=2, other_lags=2)
    stem = Path(args.csv).stem.replace("dbn_wide_", "")
    tag = f"adaptation_{stem}_run{datetime.now():%Y%m%d_%H%M%S}_g{A._git_version()}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.part in ("adaptation", "both"):
        ad = adaptation(args.csv, base)
        ad.to_csv(RESULTS_DIR / f"{tag}_adaptation.csv", index=False)
        g = ad.groupby(["regime", "variant", "bucket"])[["n", "correct", "ll"]].sum()
        order = [b for _, _, b in BUCKETS]
        print(f"\n=== {stem} | accuracy on the new regime, by runs of it seen so far ===")
        print((g.correct / g.n).unstack("bucket").reindex(columns=order).round(3).to_string())
    if args.part in ("detection", "both"):
        run_rows, step_rows, cusum = detection(args.csv, base)
        run_rows.to_csv(RESULTS_DIR / f"{tag}_regime_detection.csv", index=False)
        cusum.to_csv(RESULTS_DIR / f"{tag}_regime_cusum.csv", index=False)
        print(f"\n=== {stem} | CUSUM on per-second miscalibration: unseen regime ===")
        print(cusum.groupby(["unseen_regime", "nodes"])[["false_alarm_runs", "detected_runs",
                                                         "median_seconds_to_alarm"]].mean().round(2).to_string())
        step_rows.to_csv(RESULTS_DIR / f"{tag}_reconfiguration_detection.csv", index=False)
        print(f"\n=== {stem} | run-level AUC for an unseen regime ===")
        print(run_rows.groupby(["unseen_regime", "nodes"])[["auc_surprise", "auc_miscalibration"]]
              .mean().unstack("nodes").round(3).to_string())
        print(f"\n=== {stem} | step-level detection of unannounced reconfigurations ===")
        print(step_rows.groupby("score")[["auc", "caught_at_1pct_false_alarms"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
