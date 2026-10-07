"""
lstm_baseline.py
================
Publication-clean LSTM baseline for comparison against the learned DBN.

Fixes included:
  - Chronological train/test split, made of whole runs
  - Feature scaler fit only on training data
  - Target scaler fit only on training data
  - No future leakage; an input window never reaches across two runs
  - The target's own past is part of the input window (the persistence
    baseline it is compared with uses nothing else)
  - Mini-batch training
  - Reproducible random seeds
  - LSTM regression on scaled throughput
  - MAE on original throughput scale
  - Discretized accuracy using train-fitted bins
  - Persistence baseline on the same test rows

Usage:
    python analysis/lstm_baseline.py --csv data/dbn_wide_<stem>.csv [--granularity 30]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
import data as D  # noqa: E402


# ============================================================
# CONFIG — must match DBN sweep
# ============================================================

TARGETS = ["throughput_1", "throughput_2", "throughput_3"]

MODELING_GRANULARITY_SEC = 30

TRAIN_FRAC = 0.8

SEQUENCE_LENGTH = 5

N_BINS = 20

EPOCHS = 30

BATCH_SIZE = 256

HIDDEN_SIZE = 64

LEARNING_RATE = 1e-3

EXCLUDE_OTHER_THROUGHPUTS = True

RANDOM_STATE = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD + CLEAN + AGGREGATE
# ============================================================

def load_and_clean(path):
    """Chronological rows with run_id / pos / timestamp bookkeeping (dbn/data.py)."""
    return D.load_wide(path, verbose=False)


def aggregate(df):
    return D.aggregate(df, MODELING_GRANULARITY_SEC)


# ============================================================
# MODEL
# ============================================================

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.linear(last_hidden)


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_cols(df, target):
    # the target itself is an input: the window holds its last SEQUENCE_LENGTH values
    return [
        c for c in D.feature_cols(df)
        if not (
            EXCLUDE_OTHER_THROUGHPUTS
            and c.startswith("throughput_")
            and c != target
        )
    ]


# ============================================================
# SEQUENCE PREPARATION WITHOUT LEAKAGE
# ============================================================

def make_sequences(X, y, run_id, seq_len):
    """
    Window X[i-seq_len:i] predicts y[i]; only windows that lie, together with
    row i, inside one run are kept. Also returns the index i of every window.
    """
    idx = np.array([i for i in range(seq_len, len(X))
                    if run_id[i - seq_len] == run_id[i]], dtype=int)
    X_seqs = np.stack([X[i - seq_len:i] for i in idx]).astype(np.float32)
    return X_seqs, y[idx].astype(np.float32), idx


def prepare_data(df, target):
    feature_cols = get_feature_cols(df, target)

    X_raw = df[feature_cols].to_numpy(dtype=float)
    y_raw = df[target].to_numpy(dtype=float)
    run_id = df["run_id"].to_numpy()

    # first TRAIN_FRAC of the runs -> train, the rest -> test
    split_run = np.quantile(np.unique(run_id), TRAIN_FRAC)
    is_train = run_id <= split_run

    # Fit scalers only on training data
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaler.fit(X_raw[is_train])
    y_scaler.fit(y_raw[is_train].reshape(-1, 1))

    X_scaled = X_scaler.transform(X_raw)
    y_scaled = y_scaler.transform(y_raw.reshape(-1, 1)).flatten()

    X_seq, y_seq_scaled, idx = make_sequences(X_scaled, y_scaled, run_id, SEQUENCE_LENGTH)
    train = is_train[idx]

    return {
        "X_train": X_seq[train],
        "y_train_scaled": y_seq_scaled[train],
        "X_test": X_seq[~train],
        "y_test_raw": y_raw[idx[~train]],
        "y_prev_raw": y_raw[idx[~train] - 1],     # persistence forecast for the same rows
        "y_train_raw": y_raw[is_train],
        "y_scaler": y_scaler,
        "feature_cols": feature_cols,
    }


# ============================================================
# BASELINES
# ============================================================

def persistence_baseline(data):
    """
    Predict y(t) = y(t-1) on exactly the rows the LSTM is tested on.
    """
    y_test, y_pred = data["y_test_raw"], data["y_prev_raw"]

    mae = float(mean_absolute_error(y_test, y_pred))

    kbd = KBinsDiscretizer(
        n_bins=N_BINS,
        encode="ordinal",
        strategy="uniform"
    )

    kbd.fit(data["y_train_raw"].reshape(-1, 1))

    y_test_disc = kbd.transform(
        y_test.reshape(-1, 1)
    ).astype(int).flatten()

    y_pred_disc = kbd.transform(
        y_pred.reshape(-1, 1)
    ).astype(int).flatten()

    accuracy = float(np.mean(y_pred_disc == y_test_disc))

    return mae, accuracy


# ============================================================
# TRAIN + EVALUATE LSTM
# ============================================================

def run_lstm(data):

    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(
        data["y_train_scaled"],
        dtype=torch.float32
    ).unsqueeze(1)

    X_test = torch.tensor(data["X_test"], dtype=torch.float32)

    y_test_raw = data["y_test_raw"]
    y_scaler = data["y_scaler"]

    model = LSTMPredictor(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        output_size=1
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    model.train()

    generator = torch.Generator().manual_seed(RANDOM_STATE)

    for epoch in range(EPOCHS):
        order = torch.randperm(len(X_train), generator=generator)

        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]

            optimizer.zero_grad()

            pred = model(X_train[batch])

            loss = criterion(pred, y_train[batch])

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_test).squeeze().numpy()

    y_pred_raw = y_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()

    mae = float(mean_absolute_error(y_test_raw, y_pred_raw))

    # Discretized accuracy
    kbd = KBinsDiscretizer(
        n_bins=N_BINS,
        encode="ordinal",
        strategy="uniform"
    )

    kbd.fit(data["y_train_raw"].reshape(-1, 1))

    y_test_disc = kbd.transform(
        y_test_raw.reshape(-1, 1)
    ).astype(int).flatten()

    y_pred_disc = kbd.transform(
        y_pred_raw.reshape(-1, 1)
    ).astype(int).flatten()

    accuracy = float(np.mean(y_pred_disc == y_test_disc))

    diagnostics = {
        "y_test_min": float(np.min(y_test_raw)),
        "y_test_max": float(np.max(y_test_raw)),
        "y_pred_min": float(np.min(y_pred_raw)),
        "y_pred_max": float(np.max(y_pred_raw)),
        "final_train_loss": float(loss.item()),
        "n_train_sequences": len(data["X_train"]),
        "n_test_sequences": len(data["X_test"]),
        "n_features": len(data["feature_cols"]),
    }

    return mae, accuracy, diagnostics


# ============================================================
# MAIN
# ============================================================

def main():
    global MODELING_GRANULARITY_SEC, N_BINS
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--granularity", type=int, default=MODELING_GRANULARITY_SEC,
                        help="seconds per row")
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    args = parser.parse_args()
    MODELING_GRANULARITY_SEC, N_BINS = args.granularity, args.n_bins

    set_seed(RANDOM_STATE)

    print("=" * 60)
    print("LSTM BASELINE")
    print("=" * 60)

    raw = load_and_clean(args.csv)
    raw = aggregate(raw)

    print(f"Rows after aggregation: {len(raw)}")
    print(
        f"Sequence length: {SEQUENCE_LENGTH} steps = "
        f"{SEQUENCE_LENGTH * MODELING_GRANULARITY_SEC} seconds"
    )
    print(f"Train fraction: {TRAIN_FRAC}")
    print(f"Epochs: {EPOCHS}")
    print(f"Hidden size: {HIDDEN_SIZE}")
    print()

    results = []

    for target in TARGETS:
        if target not in raw.columns:
            print(f"[SKIP] {target} not found in data")
            print()
            continue

        print(f"Target: {target}")
        print("-" * 40)

        data = prepare_data(raw, target)
        lstm_mae, lstm_acc, diagnostics = run_lstm(data)
        pers_mae, pers_acc = persistence_baseline(data)

        print(f"  LSTM MAE:              {lstm_mae:.3f}")
        print(f"  LSTM accuracy:         {lstm_acc:.3f} (n_bins={N_BINS})")
        print()
        print(f"  Persistence MAE:       {pers_mae:.3f}")
        print(f"  Persistence accuracy:  {pers_acc:.3f} (n_bins={N_BINS})")
        print()
        print("  Diagnostics:")
        print(f"    Train sequences:     {diagnostics['n_train_sequences']}")
        print(f"    Test sequences:      {diagnostics['n_test_sequences']}")
        print(f"    Features:            {diagnostics['n_features']}")
        print(f"    Final train loss:    {diagnostics['final_train_loss']:.6f}")
        print(
            f"    y_test range:        "
            f"{diagnostics['y_test_min']:.3f} to {diagnostics['y_test_max']:.3f}"
        )
        print(
            f"    y_pred range:        "
            f"{diagnostics['y_pred_min']:.3f} to {diagnostics['y_pred_max']:.3f}"
        )
        print()

        results.append({
            "target": target,
            "lstm_mae": lstm_mae,
            "lstm_accuracy": lstm_acc,
            "persistence_mae": pers_mae,
            "persistence_accuracy": pers_acc,
            **diagnostics,
        })

    if results:
        out = pd.DataFrame(results)
        stem = Path(args.csv).stem.replace("dbn_wide_", "")
        out_path = (Path(__file__).resolve().parent.parent / "results"
                    / f"lstm_baseline_results_{stem}_{MODELING_GRANULARITY_SEC}s.csv")
        out.to_csv(out_path, index=False)

        print("=" * 60)
        print(f"Results saved to {out_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()