"""
make_figures.py -- the figures of the October 2026 reports, from the result
files in results/ (pooled over the datasets). Writes PNGs to docs/figures/.

Usage:
    python analysis/make_figures.py
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8985"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def line(ax, x, y, color, label, end_label=True, dy=0):
    ax.plot(x, y, color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    if end_label:
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(8, dy), textcoords="offset points",
                    va="center", color=INK, fontsize=9)


def newest(pattern, key):
    files = {}
    for f in sorted(glob.glob(str(ROOT / pattern))):
        files[key(Path(f).name)] = f
    return list(files.values())


def fig_bins():
    d = pd.read_csv(ROOT / "results/ablation/summary_bins_granularity.csv")
    d = d[~d.config.str.contains("hc-bic")].copy()
    d["g"] = d.config.str.extract(r"granularity=(\d+)").fillna(1).astype(int)
    d["b"] = d.config.str.extract(r"n_bins=(\d+)").fillna(20).astype(int)
    d = d.sort_values(["g", "b"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    one = d[d.g == 1]
    x = np.arange(len(one))
    line(axes[0], x, one.acc_ar.to_numpy(), NEUTRAL, "Autoregressive table", dy=-8)
    line(axes[0], x, one.acc.to_numpy(), BLUE, "DBN (reference configuration)", dy=8)
    axes[0].set_xticks(x, one.b)
    axes[0].set_xlabel("bins of throughput_3")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("More bins: AR falls faster than the DBN (1 s)")
    axes[0].legend(loc="lower left")
    for g, color in ((1, BLUE), (5, ORANGE), (10, AQUA), (30, YELLOW)):
        s = d[d.g == g]
        axes[1].plot(np.arange(len(s)), s.d_acc_vs_ar, color=color, marker="o", markeredgecolor=SURFACE,
                     markeredgewidth=1.5, label=f"{g} s rows")
        if g == 1:
            axes[1].annotate(f"+{s.d_acc_vs_ar.iloc[-1]:.3f}", (len(s) - 1, s.d_acc_vs_ar.iloc[-1]),
                             xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
    axes[1].set_xticks(np.arange(len(one)), one.b)
    axes[1].set_xlabel("bins of throughput_3")
    axes[1].set_ylabel("accuracy gain over AR")
    axes[1].set_title("The gain is at 1-second resolution")
    axes[1].legend(loc="upper left")
    for ax in axes:
        ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(OUT / "bins_and_granularity.png", dpi=160)


def fig_rollout():
    files = newest("results/rollout/rollout_b20_*_runs.csv.gz", lambda n: n.split("b20_")[1][:8])
    d = pd.concat([pd.read_csv(f) for f in files])
    d = d[(d.target == "throughput_3") & (d.load_future == "known")]
    g = d.groupby(["method", "horizon"])[["n", "correct"]].sum()
    acc = (g.correct / g.n).unstack("horizon")
    h = list(acc.columns)
    x = np.arange(len(h))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method, color, label, dy in (("ar", NEUTRAL, "AR table for that horizon", -9),
                                     ("direct", BLUE, "DBN table trained for that horizon", 9),
                                     ("rollout:local", ORANGE, "One-step DBN, unrolled", 0),
                                     ("rollout:level", AQUA, "Unrolled, with a slow level variable", 0)):
        line(ax, x, acc.loc[method].to_numpy(), color, label, dy=dy)
    ax.set_xticks(x, [f"{v} s" for v in h])
    ax.set_xlabel("seconds ahead")
    ax.set_ylabel("accuracy (20 bins)")
    ax.set_title("Unrolling the one-step model works for about two seconds")
    ax.legend(loc="lower left")
    ax.margins(x=0.1)
    fig.tight_layout()
    fig.savefig(OUT / "rollout.png", dpi=160)


def bars(ax, labels, values, highlight, fmt="{:.2f}"):
    y = np.arange(len(labels))[::-1]
    colors = [BLUE if h else NEUTRAL for h in highlight]
    ax.barh(y, values, height=0.55, color=colors)
    for yi, v in zip(y, values):
        ax.annotate(fmt.format(v), (v, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.grid(False)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)


def fig_whatif():
    files = newest("results/whatif/whatif_expanding_*_runs.csv.gz", lambda n: n.split("expanding_")[1][:8])
    d = pd.concat([pd.read_csv(f) for f in files])
    d = d[d.new_config & (d.target == "throughput_3")]
    g = d.groupby("model")[["n", "correct", "ae"]].sum()
    acc, mae = g.correct / g.n, g.ae / g.n
    order = ["table chain", "tabpfn v2", "boosting (cores, quality)", "boosting (capacities)",
             "capacity chain", "min rule", "capacity chain, noisy-min prior"]
    names = {"table chain": "BN, tables over cores and quality", "tabpfn v2": "TabPFN v2",
             "boosting (cores, quality)": "Gradient boosting", "boosting (capacities)": "Gradient boosting on capacities",
             "capacity chain": "BN with capacity nodes", "min rule": "min(upstream, capacity) rule",
             "capacity chain, noisy-min prior": "BN with capacity nodes, noisy-min prior"}
    order = [m for m in order if m in acc.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    hl = ["capacity" in m or m == "min rule" for m in order]
    bars(axes[0], [names[m] for m in order], acc[order].to_numpy(), hl)
    axes[0].set_title("Accuracy (20 bins)")
    bars(axes[1], [names[m] for m in order], mae[order].to_numpy(), hl, "{:.0f}")
    axes[1].set_title("Mean absolute error, requests/s")
    fig.suptitle("Predicting throughput_3 for configurations never seen in training", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "from load, cores and quality only; 30 s rows; blue = uses the learned capacity",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUT / "unseen_configurations.png", dpi=160)


def fig_detection():
    files = newest("results/adaptation/*_reconfiguration_detection.csv", lambda n: n.split("adaptation_")[1][:8])
    d = pd.concat([pd.read_csv(f) for f in files]).groupby("score")["caught_at_1pct_false_alarms"].mean()
    order = ["target only", "throughputs", "all nodes"]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bars(ax, ["throughput_3 only", "the three throughputs", "all eight nodes"], d[order].to_numpy(),
         [False, False, True], "{:.0%}")
    ax.set_title("Unannounced reconfigurations caught at 1% false alarms")
    fig.tight_layout()
    fig.savefig(OUT / "reconfiguration_detection.png", dpi=160)


def fig_targets():
    d = pd.read_csv(ROOT / "results/ablation/summary_recommended.csv")
    d = d[~d.config.str.contains("n_bins|granularity")].copy()
    d["target"] = d.config.str.extract(r"target=(\w+)").fillna("throughput_3")
    d = d.set_index("target").loc[["throughput_1", "throughput_2", "throughput_3", "avg_p_latency_1",
                                   "avg_p_latency_2", "avg_p_latency_3", "buffer_size_2", "buffer_size_3"]]
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hlines(y, d.acc_ar, d.acc, color=GRID, linewidth=3, zorder=1)
    ax.scatter(d.acc_ar, y, s=60, color=NEUTRAL, edgecolor=SURFACE, linewidth=1.5, zorder=2,
               label="Autoregressive table")
    ax.scatter(d.acc, y, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="DBN (recommended configuration)")
    for yi, a_, v in zip(y, d.acc_ar, d.acc):
        ax.annotate(f"+{v - a_:.3f}", (v, yi), xytext=(9, 0), textcoords="offset points", va="center",
                    fontsize=9)
    ax.set_yticks(y, [t.replace("avg_p_", "").replace("_", " ") for t in d.index])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("accuracy (20 bins, 1 s)")
    ax.set_title("Every variable of the system is predicted better than by its AR table")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(OUT / "all_targets.png", dpi=160)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_bins, fig_rollout, fig_whatif, fig_detection, fig_targets):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:
            print("skipped", f.__name__, type(e).__name__, e)
        plt.close("all")


if __name__ == "__main__":
    main()
