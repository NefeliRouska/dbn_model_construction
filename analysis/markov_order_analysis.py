"""
markov_order_analysis.py

Standalone: measures held-out log-likelihood and accuracy of
throughput_3 as a function of Markov order (0 = persistence-only
self-loop, 1, 2, 3, ... lags), using the same 5-fold expanding-window
temporal CV as the main sweep. Lags never reach across two runs.

Deliberately isolated from feature selection, discretizer sweep, and
other variables: this answers "how much of throughput_3's own history
does throughput_3 need to see to predict itself well," not "what's
the best full model." That's a cleaner, more defensible number for
the paper than mixing it with a feature-selection question.

Usage:
    python markov_order_analysis.py --csv path/to/dbn_wide_*.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import log_loss, accuracy_score, f1_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
import data as D  # noqa: E402

TARGET = "throughput_3"
N_BINS = 20
MAX_ORDER = 5
MODELING_GRANULARITY_SEC = 1
K_FOLDS = 5
MAX_CPT_COLUMNS = 2_000_000


def load_and_aggregate(csv_path, target):
    """Chronological rows of the target with run_id / pos bookkeeping (see dbn/data.py)."""
    df = D.load_wide(csv_path, target, verbose=False)
    df = D.aggregate(df, MODELING_GRANULARITY_SEC)
    return df[D.META_COLS + [target]]


def make_temporal_folds(series, k=K_FOLDS):
    return D.make_temporal_folds(series, k=k)


def make_lagged(disc_series, run_id, order, max_order):
    """
    disc_series: 1D int array, already discretized; run_id: the run of each row.
    order=0: only the current value, predicting itself one step ahead
             (this reproduces AR-DBN / persistence-shaped model).
    order=k: current value plus k lags as evidence.
    Lags and the next value are taken inside a run. Rows that lack max_order
    lags are dropped for every order, so all orders are scored on the same rows.
    Returns a DataFrame with columns lag0..lagK and 'next' (target).
    """
    s = pd.Series(disc_series, dtype=float)
    g = s.groupby(np.asarray(run_id))
    cols = {f"lag{L}": g.shift(L) for L in range(max_order + 1)}
    cols["next"] = g.shift(-1)
    out = pd.DataFrame(cols).dropna().reset_index(drop=True).astype(int)
    return out[[f"lag{L}" for L in range(order + 1)] + ["next"]]


def fit_and_eval(train_lagged, test_lagged, order, n_states):
    parent_cols = [f"lag{L}" for L in range(order + 1)]
    edges = [(c, "next") for c in parent_cols]

    model = BayesianNetwork(edges)
    state_names = {c: list(range(n_states)) for c in parent_cols + ["next"]}
    model.fit(
        train_lagged,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10,
        state_names=state_names,
    )

    infer = VariableElimination(model)
    y_true, y_pred, y_prob = [], [], []
    cache = {}

    for _, row in test_lagged.iterrows():
        ev = {c: int(row[c]) for c in parent_cols}
        key = tuple(sorted(ev.items()))
        if key not in cache:
            q = infer.query(["next"], evidence=ev, show_progress=False)
            cache[key] = q.values / q.values.sum()
        probs = cache[key]

        y_true.append(int(row["next"]))
        y_pred.append(int(np.argmax(probs)))
        y_prob.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.clip(np.array(y_prob), 1e-12, 1.0)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    ll = log_loss(y_true, y_prob, labels=list(range(n_states)))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return ll, acc, f1


def main():
    global N_BINS, MODELING_GRANULARITY_SEC
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    parser.add_argument("--granularity", type=int, default=MODELING_GRANULARITY_SEC,
                        help="seconds per row")
    args = parser.parse_args()
    N_BINS, MODELING_GRANULARITY_SEC = args.n_bins, args.granularity

    raw = load_and_aggregate(args.csv, TARGET)
    folds = make_temporal_folds(raw)

    print(f"[DATA] {len(raw)} rows after aggregation, {len(folds)} folds\n")

    results = []

    for order in range(MAX_ORDER + 1):
        fold_lls, fold_accs, fold_f1s = [], [], []

        for fold_idx, (train_raw, test_raw) in enumerate(folds):
            # Discretize on train only, apply same edges to test.
            disc = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="uniform")
            disc.fit(train_raw[[TARGET]])
            train_disc = disc.transform(train_raw[[TARGET]]).astype(int).flatten()
            test_disc  = disc.transform(test_raw[[TARGET]]).astype(int).flatten()

            n_states = len(disc.bin_edges_[0]) - 1

            train_lagged = make_lagged(train_disc, train_raw["run_id"], order, MAX_ORDER)
            test_lagged  = make_lagged(test_disc, test_raw["run_id"], order, MAX_ORDER)

            if len(train_lagged) < 10 or len(test_lagged) < 10:
                continue

            # the table of 'next' has n_states ** (order + 1) columns
            if float(n_states) ** (order + 1) > MAX_CPT_COLUMNS:
                if fold_idx == 0:
                    print(f"order={order}  skipped: {n_states}^{order + 1} parent "
                          f"configurations (limit {MAX_CPT_COLUMNS:.0e})")
                continue

            try:
                ll, acc, f1 = fit_and_eval(train_lagged, test_lagged, order, n_states)
                fold_lls.append(ll)
                fold_accs.append(acc)
                fold_f1s.append(f1)
            except Exception as e:
                print(f"  order={order} fold={fold_idx+1} FAILED: {e}")

        if fold_lls:
            mean_ll  = np.mean(fold_lls)
            std_ll   = np.std(fold_lls, ddof=1) if len(fold_lls) > 1 else 0.0
            mean_acc = np.mean(fold_accs)
            mean_f1  = np.mean(fold_f1s)
            results.append((order, mean_ll, std_ll, mean_acc, mean_f1))
            print(f"order={order}  log_loss={mean_ll:.4f}±{std_ll:.4f}  "
                  f"acc={mean_acc:.3f}  f1={mean_f1:.3f}  "
                  f"(n_states={n_states})")

    print("\n=== SUMMARY ===")
    print(f"{'order':>5} {'log_loss':>10} {'acc':>7} {'f1':>7}")
    best_order = min(results, key=lambda r: r[1])[0]
    for order, ll, std, acc, f1 in results:
        marker = "  <-- best log_loss" if order == best_order else ""
        print(f"{order:>5} {ll:>10.4f} {acc:>7.3f} {f1:>7.3f}{marker}")

    print(f"\nBest-fitting Markov order by held-out log-loss: {best_order}")
    print("(order=0 means throughput_3's current value alone; "
          "order=k means current value plus k lags)")


if __name__ == "__main__":
    main()