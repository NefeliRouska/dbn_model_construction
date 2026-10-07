"""
tabpfn_baseline.py -- how far is the DBN from TabPFN on the same task?

Every model predicts throughput_3 at t+h from the same information (the
candidate columns of dbn/ablation.py: service metrics at t, and the controls
and load at t+h), on the same folds and the SAME test samples, and is scored on
the same bins:

    persistence        y(t+h) = y(t)
    ar                 P(bin' | bin)                      -- table on the target's bins
    model              the DBN configuration under test   -- dbn/ablation.py
    tabpfn             TabPFNRegressor on the continuous candidates, predicting
                       the change y(t+h) - y(t); its predictive distribution is
                       integrated over the bins to get P(bin') (with a continuity
                       correction when the target is a count, see _to_bins)
    tabpfn_parents     the same, given only the DBN's parent columns (what the
                       discretisation and the table cost, structure held equal)
    tabpfn_clf         TabPFNClassifier on the bins (only when there are <= 10)
    hgb                gradient boosting trained on ALL training samples
    hgb_ctx            gradient boosting trained on TabPFN's context only

TabPFN conditions on a context of at most `--context` training samples (drawn
at random, seeded); the test samples are a random `--n-test` per fold.

Model weights: by default TabPFN v2, the last version whose weights download
without a Hugging Face login. Newer versions (`--model-version v3.5`) are
gated: accept the terms on huggingface.co/Prior-Labs and run `hf auth login`
first.

Usage:
    python analysis/tabpfn_baseline.py --csv data/dbn_wide_<stem>.csv
"""
import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
import ablation as A  # noqa: E402

RESULTS_DIR = A.REPO_ROOT / "results" / "tabpfn"

# (granularity, horizon, n_bins) settings to compare on
SETTINGS = [(1, 1, 20), (1, 1, 50), (1, 1, 10), (1, 5, 20), (30, 1, 20), (30, 1, 4)]


class TabPFNModels:
    def __init__(self, version, device, n_context, seed=0):
        self.version, self.device, self.n_context, self.seed = version, device, n_context, seed
        self.cache = {}

    def _context(self, tr):
        idx = np.nonzero(tr)[0]
        if len(idx) > self.n_context:
            idx = np.random.default_rng(self.seed).choice(idx, self.n_context, replace=False)
        return idx

    def _regress(self, S, tr, te, cols):
        """Predictive distribution of y(t+h) - y(t); cached, it does not depend on the bins."""
        key = (id(S), tuple(cols), int(tr.sum()), int(te.sum()), int(np.nonzero(te)[0][:50].sum()))
        if key not in self.cache:
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
            X = np.column_stack([S.cols[c] for c in cols])
            ctx = self._context(tr)
            m = TabPFNRegressor.create_default_for_version(ModelVersion(self.version), device=self.device)
            m.fit(X[ctx], (S.y1 - S.y0)[ctx])
            if len(self.cache) >= 12:                                # bound the memory used
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = m.predict(X[te], output_type="full")
        return self.cache[key]

    def _to_bins(self, out, S, te, binner):
        import torch
        logits, crit = out["logits"], out["criterion"]
        y0 = S.y0[te]
        # P(bin k) = cdf(upper edge - y0) - cdf(lower edge - y0).
        # At 1 s the target is a count. A continuous density spreads "no change"
        # around zero, and when y0 sits exactly on a bin edge half of that mass
        # would fall into the bin below. Counts below edge e are the counts
        # <= ceil(e) - 1, so the density is cut at ceil(e) - 0.5.
        edges = binner.y_edges
        if A.integer_target(S):
            edges = np.ceil(edges) - 0.5
        ys = torch.tensor(edges[None, :] - y0[:, None], dtype=logits.dtype, device=logits.device)
        cdf = crit.cdf(logits, ys).cpu().numpy()
        cdf = np.column_stack([np.zeros(len(y0)), cdf, np.ones(len(y0))])
        P = np.clip(np.diff(cdf, axis=1), 1e-9, None)
        return P / P.sum(1, keepdims=True), y0 + np.asarray(out["median"])

    def all_candidates(self, S, cfg, tr, te, binner, cands, parents):
        return self._to_bins(self._regress(S, tr, te, cands), S, te, binner)

    def dbn_parents(self, S, cfg, tr, te, binner, cands, parents):
        return self._to_bins(self._regress(S, tr, te, parents), S, te, binner)

    def classifier(self, S, cfg, tr, te, binner, cands, parents):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        X = np.column_stack([S.cols[c] for c in cands])
        ctx = self._context(tr)
        m = TabPFNClassifier.create_default_for_version(ModelVersion(self.version), device=self.device)
        m.fit(X[ctx], binner.code_y(S.y1[ctx]))
        P = np.full((int(te.sum()), binner.n_y), 1e-9)
        P[:, m.classes_.astype(int)] = m.predict_proba(X[te])
        return P / P.sum(1, keepdims=True)

    def hgb_context(self, S, cfg, tr, te, binner, cands, parents):
        ctx_mask = np.zeros(len(tr), dtype=bool)
        ctx_mask[self._context(tr)] = True
        return A.hgb_model(S, cfg, ctx_mask, te, binner, cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model-version", default="v2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=2000, help="test samples per fold")
    ap.add_argument("--settings", nargs="*", default=None, metavar="gGs_hH_bB",
                    help="subset of the settings to run, e.g. g1s_h1_b20 g30s_h1_b4 (default: all)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="DBN options, e.g. engine=cv-forward max_parents=4")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = type(A.DEFAULTS[k])(v)

    T = TabPFNModels(args.model_version, args.device, args.context)
    stem = Path(args.csv).stem.replace("dbn_wide_", "")
    tag = f"tabpfn-{args.model_version}_{stem}_run{datetime.now():%Y%m%d_%H%M%S}_g{A._git_version()}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows, tables = [], []
    for si, (gran, horizon, n_bins) in enumerate(SETTINGS):
        cfg = A.make_cfg(granularity=gran, horizon=horizon, n_bins=n_bins, **overrides)
        S = A.get_samples(args.csv, cfg)
        setting = f"g{gran}s_h{horizon}_b{n_bins}"
        if args.settings and setting not in args.settings:
            continue
        for fi, (train_runs, test_runs) in enumerate(A.fold_runs(S, cfg)):
            tr, te = A.split_masks(S, train_runs, test_runs)
            idx = np.nonzero(te)[0]
            if len(idx) > args.n_test:
                idx = np.random.default_rng(fi).choice(idx, args.n_test, replace=False)
            te_sub = np.zeros(len(te), dtype=bool)
            te_sub[idx] = True

            extra = [("tabpfn", T.all_candidates), ("hgb", A.hgb_model), ("hgb_ctx", T.hgb_context)]
            if (gran, horizon, n_bins) == SETTINGS[0]:
                extra.append(("tabpfn_parents", T.dbn_parents))
            if n_bins <= 10:
                extra.append(("tabpfn_clf", T.classifier))
            t0 = time.perf_counter()
            res, info = A.run_fold(S, cfg, tr, te_sub, extra)
            for model, (summ, table) in res.items():
                rows.append({"setting": setting, "granularity": gran, "horizon": horizon,
                             "n_bins": n_bins, "model": model, "fold": fi + 1,
                             "n_train": int(tr.sum()), "n_context": min(int(tr.sum()), args.context),
                             **summ, "parents": ",".join(info["parents"]),
                             "mae_persistence_continuous": info["mae_persistence_continuous"]})
                table.insert(0, "fold", fi + 1)
                table.insert(0, "model", model)
                table.insert(0, "config", setting)
                tables.append(table)
            m = pd.DataFrame([r for r in rows if r["setting"] == setting and r["fold"] == fi + 1])
            print(f"[{setting} fold {fi + 1}] {time.perf_counter() - t0:.0f}s  "
                  + "  ".join(f"{r.model}: acc={r.acc_all:.3f} ll={r.logloss_all:.3f} mae={r.maeh_all:.2f}"
                              for r in m.itertuples()), flush=True)
            pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{tag}_folds.csv", index=False)
            pd.concat(tables, ignore_index=True).to_csv(RESULTS_DIR / f"{tag}_runs.csv.gz", index=False)
    print(f"\nSaved {RESULTS_DIR / (tag + '_folds.csv')}")


if __name__ == "__main__":
    main()
