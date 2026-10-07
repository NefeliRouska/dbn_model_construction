"""
regime_study.py -- does the model change with the operating regime, and if so,
is it the structure (which parents) or only the values (the tables)?

Two definitions of "regime", both known per sample:

  bottleneck   where the chain saturates during the run:
                 none   every service keeps up with what it receives
                 s1     service 1 is the first one that cannot
                 s23    service 2 or 3 is
               (decided per run from the run's mean throughputs: a service is
               saturated when it emits < 97% of what it receives)
  load         low / mid / high third of the offered load (rps) of the dataset

For the target (default throughput_3) and every fold it compares, on the test
samples of each regime:

  global             one structure and one table learned from all regimes
  +regime parent     the same search, with the regime label forced in as a parent
  regime tables      the global structure, one table per regime
  regime structure   structure and table learned per regime
  leave-regime-out   learned without any sample of that regime (what happens
                     when a regime appears that the model has never seen)

and it records the parents selected per regime (is the structure stable?) and
how well the model's own surprise (-log p of what happened) separates the runs
of an unseen regime from the others (can a regime change be detected?).

It also treats every reconfiguration as an unannounced fault and measures how
well the surprise detects it on the first step (change_detection).

Usage:
    python analysis/regime_study.py --csv data/dbn_wide_<stem>.csv
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
import ablation as A  # noqa: E402
import data as D  # noqa: E402

RESULTS_DIR = A.REPO_ROOT / "results" / "regimes"
SATURATED_BELOW = 0.97


def bottleneck_of_runs(df):
    """run_id -> none | s1 | s23."""
    r = df.groupby("run_id")[[D.LOAD_COL, "throughput_1", "throughput_2", "throughput_3"]].mean()
    s1 = r["throughput_1"] < SATURATED_BELOW * r[D.LOAD_COL]
    s2 = r["throughput_2"] < SATURATED_BELOW * r["throughput_1"]
    s3 = r["throughput_3"] < SATURATED_BELOW * r["throughput_2"]
    return pd.Series(np.where(s1, "s1", np.where(s2 | s3, "s23", "none")), index=r.index)


def regime_labels(S, df, kind):
    if kind == "bottleneck":
        return bottleneck_of_runs(df).reindex(S.run).to_numpy()
    load = S.cols[f"{D.LOAD_COL}@t+h"]
    lo, hi = np.quantile(load, [1 / 3, 2 / 3])
    if hi <= lo:                       # constant load: one regime only
        return np.full(len(load), "mid")
    return np.where(load <= lo, "low", np.where(load <= hi, "mid", "high"))


def predict(S, cfg, binner, c_tr, c_te, parents, ar_table):
    cpt = A.fit_table(cfg, c_tr, parents, ar_table)
    return c_te.to_bins(cpt.predict([c_te(p)[0] for p in parents], c_te.n))


def score(P, y):
    return (P.argmax(1) == y).astype(float), -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))


def run(csv, cfg, kind):
    S = A.get_samples(csv, cfg)
    df = A.load_frame(csv, cfg["target"], cfg["granularity"])
    label = regime_labels(S, df, kind)
    regimes = sorted(set(label))
    codes = {r: i for i, r in enumerate(regimes)}
    S.cols["regime@t"] = np.array([codes[r] for r in label], dtype=float)
    cands = A.candidate_columns(S, cfg)
    forced = list(cfg["forced"])
    rows, parent_rows, detect_rows = [], [], []

    for fi, (train_runs, test_runs) in enumerate(A.fold_runs(S, cfg)):
        tr, te = A.split_masks(S, train_runs, test_runs)
        binner = A.Binner(S, cfg, tr)                      # one discretisation for every variant

        def fit(rows_tr, forced_=forced, cands_=cands):
            """Structure + table from the training samples in rows_tr; predictions for all of te."""
            c_tr, c_te = A.Coder(S, cfg, binner, rows_tr), A.Coder(S, cfg, binner, te)
            _, ar_table = A.ar_model(c_tr("y@t"), c_tr.y_fit, c_tr.n_fit, cfg["ess"])
            parents, _ = A.select_parents(S, cfg, rows_tr, c_tr, cands_, forced_)
            return parents, predict(S, cfg, binner, c_tr, c_te, parents, ar_table)

        c_te = A.Coder(S, cfg, binner, te)
        y = c_te.y
        lab_te, run_te = label[te], S.run[te]

        g_parents, P_global = fit(tr)
        _, P_ctx = fit(tr, forced_=forced + ["regime@t"])
        # AR table as the common reference
        c_all = A.Coder(S, {**cfg, "target_mode": "level"}, binner, tr)
        ar, _ = A.ar_model(c_all("y@t"), c_all.y, binner.n_y, cfg["ess"])
        P_ar = ar.predict([c_te("y@t")[0]], c_te.n)

        P_tables, P_struct, P_loo = P_global.copy(), P_global.copy(), P_global.copy()
        parent_rows.append({"fold": fi + 1, "regime": "ALL", "parents": g_parents,
                            "n_train": int(tr.sum())})
        for r in regimes:
            tr_r, tr_not = tr & (label == r), tr & (label != r)
            hit = lab_te == r
            if tr_r.sum() < 2000 or not hit.any():
                continue
            # global structure, this regime's table
            c_r = A.Coder(S, cfg, binner, tr_r)
            _, ar_r = A.ar_model(c_r("y@t"), c_r.y_fit, c_r.n_fit, cfg["ess"])
            P_tables[hit] = predict(S, cfg, binner, c_r, c_te, g_parents, ar_r)[hit]
            # this regime's own structure
            r_parents, P_r = fit(tr_r)
            P_struct[hit] = P_r[hit]
            parent_rows.append({"fold": fi + 1, "regime": r, "parents": r_parents,
                                "n_train": int(tr_r.sum())})
            # never seen this regime
            if tr_not.sum() >= 2000:
                _, P_not = fit(tr_not)
                P_loo[hit] = P_not[hit]
                # detection: per-run mean surprise of the model that has not seen regime r
                s_dbn, s_ar = score(P_not, y)[1], None
                c_not = A.Coder(S, {**cfg, "target_mode": "level"}, binner, tr_not)
                ar_not, _ = A.ar_model(c_not("y@t"), c_not.y, binner.n_y, cfg["ess"])
                s_ar = score(ar_not.predict([c_te("y@t")[0]], c_te.n), y)[1]
                per_run = pd.DataFrame({"run": run_te, "novel": hit, "dbn": s_dbn, "ar": s_ar}) \
                    .groupby("run").mean()
                if 0 < per_run["novel"].sum() < len(per_run):
                    novel = per_run["novel"] > 0.5
                    detect_rows.append({
                        "fold": fi + 1, "unseen_regime": r, "n_runs_novel": int(novel.sum()),
                        "n_runs_known": int((~novel).sum()),
                        "auc_dbn_surprise": roc_auc_score(novel, per_run["dbn"]),
                        "auc_ar_surprise": roc_auc_score(novel, per_run["ar"]),
                        "surprise_known": per_run.loc[~novel, "dbn"].mean(),
                        "surprise_novel": per_run.loc[novel, "dbn"].mean()})

        variants = {"ar": P_ar, "global": P_global, "+regime parent": P_ctx,
                    "regime tables": P_tables, "regime structure": P_struct,
                    "leave-regime-out": P_loo}
        for name, P in variants.items():
            correct, ll = score(P, y)
            for r in regimes + ["ALL"]:
                hit = np.ones(len(y), bool) if r == "ALL" else lab_te == r
                if hit.any():
                    rows.append({"fold": fi + 1, "regime": r, "variant": name, "n": int(hit.sum()),
                                 "correct": correct[hit].sum(), "ll": ll[hit].sum()})
    return pd.DataFrame(rows), pd.DataFrame(parent_rows), pd.DataFrame(detect_rows)


def change_detection(csv, cfg):
    """
    A reconfiguration as a fault to detect: the model is NOT told the new
    configuration (exog='none') and its surprise, -log p(observed bin), is used
    as an alarm score on every step. AUC of that score for telling the first
    step after a reconfiguration from an ordinary step, and the share of
    reconfigurations caught at a threshold that raises a false alarm on 1% of
    ordinary steps. The same for the AR table's surprise.
    """
    cfg = {**cfg, "exog": "none", "cross_run": True}
    S = A.get_samples(csv, cfg)
    cands = A.candidate_columns(S, cfg)
    rows = []
    for fi, (train_runs, test_runs) in enumerate(A.fold_runs(S, cfg)):
        tr, te = A.split_masks(S, train_runs, test_runs)
        binner = A.Binner(S, cfg, tr)
        c_tr, c_te = A.Coder(S, cfg, binner, tr), A.Coder(S, cfg, binner, te)
        _, ar_table = A.ar_model(c_tr("y@t"), c_tr.y_fit, c_tr.n_fit, cfg["ess"])
        parents, _ = A.select_parents(S, cfg, tr, c_tr, cands, list(cfg["forced"]))
        s_dbn = score(predict(S, cfg, binner, c_tr, c_te, parents, ar_table), c_te.y)[1]
        ar, _ = A.ar_model(c_tr("y@t"), c_tr.y, binner.n_y, cfg["ess"])
        s_ar = score(ar.predict([c_te("y@t")[0]], c_te.n), c_te.y)[1]
        change = S.cross[te]
        for name, sc in (("dbn", s_dbn), ("ar", s_ar)):
            thr = np.quantile(sc[~change], 0.99)
            rows.append({"fold": fi + 1, "detector": name, "n_changes": int(change.sum()),
                         "n_ordinary": int((~change).sum()), "auc": roc_auc_score(change, sc),
                         "caught_at_1pct_false_alarms": float((sc[change] > thr).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--set", nargs="*", default=["ar_order=2", "other_lags=2"], metavar="KEY=VALUE")
    ap.add_argument("--only-detection", action="store_true",
                    help="only the reconfiguration-detection part")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = type(A.DEFAULTS[k])(v)
    cfg = A.make_cfg(**overrides)
    stem = Path(args.csv).stem.replace("dbn_wide_", "")
    tag = f"regimes_{stem}_run{datetime.now():%Y%m%d_%H%M%S}_g{A._git_version()}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for gran in (1, 30):
        det = change_detection(args.csv, {**cfg, "granularity": gran})
        det.insert(0, "granularity", gran)
        det.to_csv(RESULTS_DIR / f"{tag}_reconfiguration_detection_{gran}s.csv", index=False)
        print(f"\n=== {stem} | detecting reconfigurations from surprise, {gran} s ===")
        print(det.groupby("detector")[["auc", "caught_at_1pct_false_alarms"]].mean().round(3).to_string())
    if args.only_detection:
        return

    for kind in ("bottleneck", "load"):
        scores, parents, detect = run(args.csv, cfg, kind)
        for name, d in (("scores", scores), ("parents", parents), ("detection", detect)):
            d.insert(0, "regime_kind", kind)
            if name == "parents":
                d["parents"] = d["parents"].apply(json.dumps)
            d.to_csv(RESULTS_DIR / f"{tag}_{kind}_{name}.csv", index=False)
        g = scores.groupby(["regime", "variant"])[["n", "correct", "ll"]].sum()
        out = pd.DataFrame({"acc": g.correct / g.n, "logloss": g.ll / g.n}).round(4)
        print(f"\n=== {stem} | regimes by {kind} ===")
        print(out["acc"].unstack(1).to_string())
        print(out["logloss"].unstack(1).to_string())
        for r, grp in parents.groupby("regime"):
            print(f"  parents [{r}]:", " | ".join(sorted(set(grp.parents))))
        if len(detect):
            print(detect.groupby("unseen_regime")[["auc_dbn_surprise", "auc_ar_surprise",
                                                   "surprise_known", "surprise_novel"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
