"""
make_figures.py -- the figures of the October 2026 reports, from the result
files in results/ (pooled over the datasets). Writes PNGs to docs/figures/.

Usage:
    python analysis/make_figures.py
"""
import glob
import json
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
    # TabPFN given the capacities (analysis/whatif_tabpfn_capacity.py): same folds and test rows
    cap_files = sorted(glob.glob(str(ROOT / "results/whatif_capacity/*_runs.csv.gz")))
    if cap_files:
        c = pd.read_csv(cap_files[-1])
        d = pd.concat([d, c[c.model.str.startswith("tabpfn") & c.model.str.contains("capacities")]])
    d = d[d.new_config & (d.target == "throughput_3")]
    g = d.groupby("model")[["n", "correct", "ae"]].sum()
    acc, mae = g.correct / g.n, g.ae / g.n
    tab = sorted(m for m in acc.index if m.startswith("tabpfn") and "capacities" not in m)
    tab_cap = sorted(m for m in acc.index if m.startswith("tabpfn") and "capacities" in m)
    order = ["table chain"] + tab + ["boosting (cores, quality)", "boosting (capacities)"] + tab_cap + \
            ["capacity chain", "min rule", "capacity chain, noisy-min prior"]
    names = {"table chain": "BN, tables over cores and quality",
             "boosting (cores, quality)": "Gradient boosting", "boosting (capacities)": "Gradient boosting on capacities",
             "capacity chain": "BN with capacity nodes", "min rule": "min(upstream, capacity) rule",
             "capacity chain, noisy-min prior": "BN with capacity nodes, noisy-min prior"}
    names.update({m: m.replace("tabpfn", "TabPFN") + " on cores and quality" for m in tab})
    names.update({m: m.replace("tabpfn", "TabPFN").replace(" (capacities)", " on capacities") for m in tab_cap})
    order = [m for m in order if m in acc.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(order) + 1.2), sharey=True)
    hl = ["capacit" in m or m == "min rule" for m in order]   # capacity chains, models on capacities, the min rule
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
    fig.suptitle("Unannounced reconfigurations caught at 1% false alarms", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "reconfiguration_detection.png", dpi=160)


TARGET_ORDER = ["throughput_1", "throughput_2", "throughput_3", "avg_p_latency_1", "avg_p_latency_2",
                "avg_p_latency_3", "buffer_size_2", "buffer_size_3"]


def target_label(t):
    return t.replace("avg_p_latency", "latency").replace("buffer_size", "buffer size").replace("_", "\n")


def fig_targets():
    d = pd.read_csv(ROOT / "results/ablation/summary_recommended.csv")
    d = d[~d.config.str.contains("n_bins|granularity")].copy()
    d["target"] = d.config.str.extract(r"target=(\w+)").fillna("throughput_3")
    d = d.set_index("target").loc[TARGET_ORDER]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.vlines(x, d.acc_ar, d.acc, color=GRID, linewidth=3, zorder=1)
    ax.scatter(x, d.acc_ar, s=60, color=NEUTRAL, edgecolor=SURFACE, linewidth=1.5, zorder=2,
               label="Autoregressive table")
    ax.scatter(x, d.acc, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="DBN (recommended configuration)")
    for xi, a_, v in zip(x, d.acc_ar, d.acc):
        ax.annotate(f"+{v - a_:.3f}", (xi, v), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=9)
    for b in (2.5, 5.5):                        # throughputs | latencies | buffer sizes
        ax.axvline(b, color=GRID, linewidth=1)
    ax.set_xticks(x, [target_label(t) for t in d.index])
    ax.set_ylabel("accuracy (20 bins, 1 s)")
    ax.margins(x=0.06, y=0.15)
    ax.set_title("Every variable of the system is predicted better than by its AR table")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2)
    fig.tight_layout()
    fig.savefig(OUT / "all_targets.png", dpi=160)


def run_gains():
    """Per test run: accuracy of the recommended DBN minus that of the AR table."""
    files = sorted(glob.glob(str(ROOT / "results/ablation/recommended_*_runs.csv.gz")))
    newest_files = {Path(f).name.split("recommended_")[1][:8]: f for f in files}.values()
    d = pd.concat([pd.read_csv(f).assign(ds=Path(f).name.split("recommended_")[1][:8]) for f in newest_files])
    d = d[~d.config.str.contains("n_bins|granularity") & d.model.isin(["model", "ar"])]
    d["target"] = d.config.str.extract(r"target=(\w+)")[0].fillna("throughput_3")
    d["acc"] = d.correct / d.n
    p = d.pivot_table(index=["target", "ds", "fold", "run_id"], columns="model", values="acc").dropna()
    return (p.model - p.ar).rename("gain").reset_index()


def fig_targets_runs():
    g = run_gains()
    tol = 0.5 / 361                             # less than one second of a run: a tie
    share = g.assign(better=g.gain > tol, worse=g.gain < -tol).groupby("target")[["better", "worse"]].mean()
    share["tied"] = 1 - share.better - share.worse
    share = share.loc[TARGET_ORDER]
    n_runs = g.groupby("target").size().iloc[0]
    x = np.arange(len(share))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bottom = np.zeros(len(share))
    for col, color, label in (("better", BLUE, "DBN better than AR"), ("tied", "#d9d8d4", "tied"),
                              ("worse", RED, "DBN worse than AR")):
        v = share[col].to_numpy()
        ax.bar(x, v, bottom=bottom, width=0.62, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for xi, b, h in zip(x, bottom, v):
            if h >= 0.06:
                ax.text(xi, b + h / 2, f"{h:.0%}", ha="center", va="center", fontsize=8.5,
                        color=SURFACE if col != "tied" else INK)
        bottom += v
    for b in (2.5, 5.5):
        ax.axvline(b, color=GRID, linewidth=1)
    ax.set_xticks(x, [target_label(t) for t in share.index])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(f"share of test runs ({n_runs} per variable)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_title("Run by run: how often the DBN beats its AR table")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=3)
    fig.tight_layout()
    fig.savefig(OUT / "all_targets_runs.png", dpi=160)


RED = "#e34948"

# One-factor options that keep the prediction task fixed (same bins of the
# target, same test samples), grouped. Bins and the target's discretisation are
# left out: they change what "accuracy" measures, not how well it is reached.
OPTION_GROUPS = (
    ("Memory", (
        ("other_lags=2", "other variables: 2 steps of history"),
        ("other_lags=3", "other variables: 3 steps of history"),
        ("ar_order=2", "target: 2 steps of its own history"),
        ("ar_order=5", "target: 5 steps of its own history"),
        ("velocity=binned", "velocity features (binned change)"),
    )),
    ("Choosing the parents (vs cross-validated forward search)", (
        ("engine=hc-bdeu", "hill climbing, BDeu score"),
        ("engine=cmi", "conditional mutual information"),
        ("engine=lasso", "Lasso ranking"),
        ("engine=hc-aic", "hill climbing, AIC"),
        ("engine=dynotears", "DYNOTEARS"),
        ("engine=hc-bic", "hill climbing, BIC"),
        ("engine=fixed,fixed_parents=('y@t', 'throughput_2@t', 'buffer_size_3@t', 'cores_3@t+h', "
         "'data_quality_3@t+h')", "expert parents, fixed by hand"),
        ("engine=ar", "no search: own past only (AR)"),
    )),
    ("Table prior (vs backoff to the AR table)", (
        ("prior=bdeu", "uniform BDeu prior"),
        ("engine=hc-bic,prior=bdeu", "uniform prior + BIC search (pgmpy defaults)"),
    )),
    ("Inputs", (
        ("pool=all", "all metrics as candidates (+ process / GC)"),
        ("exog=none", "next-step settings not given"),
        ("max_parents=2", "at most 2 parents"),
    )),
)


def fig_options():
    d = pd.read_csv(ROOT / "results/ablation/summary_ofat.csv").set_index("config")
    ref = d.loc["ref", "acc"]
    entries = []                                   # (label, config or None for a group header)
    for group, items in OPTION_GROUPS:
        entries.append((group, None))
        entries += [(label, cfg) for cfg, label in items if cfg in d.index]
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(entries) + 1.6))
    y = np.arange(len(entries))[::-1]
    for yi, (label, cfg) in zip(y, entries):
        if cfg is None:
            continue
        r = d.loc[cfg]
        dv, lo, hi = 100 * r.d_acc_vs_ref, 100 * r.d_acc_lo_vs_ref, 100 * r.d_acc_hi_vs_ref
        color = BLUE if lo > 0 else RED if hi < 0 else NEUTRAL
        ax.hlines(yi, lo, hi, color=color, linewidth=2, alpha=0.45)
        ax.scatter(dv, yi, s=55, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{dv:+.1f}", (hi, yi), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK)
    ax.set_yticks(y, [label for label, _ in entries], fontsize=9)
    for tick, (_, cfg) in zip(ax.get_yticklabels(), entries):
        if cfg is None:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.margins(x=0.08)
    ax.set_xlabel(f"change in accuracy vs the reference DBN (accuracy {ref:.3f}), percentage points")
    fig.suptitle("What each modelling choice does to the prediction of throughput_3", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "1 s rows, 20 bins, pooled over the three load patterns; bars = 95% intervals over "
             "schedule positions;\nblue = helps, red = hurts, grey = no clear effect",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(OUT / "option_effects.png", dpi=160)


def fig_tabpfn():
    path = ROOT / "results/tabpfn/summary_tabpfn-v3.5.csv"
    d = pd.read_csv(path).set_index(["setting", "model"])
    models = (("ar", "ar", "AR table", NEUTRAL),
              ("hgb", "hgb", "Gradient boosting", NEUTRAL),
              ("tabpfn_rich", "tabpfn_rich", "TabPFN v3.5, rich inputs", ORANGE),
              ("tabpfn_rec", "tabpfn_rec", "TabPFN v3.5, same inputs as the DBN", ORANGE),
              ("dbn_rec", "dbn_rec_median", "DBN, recommended", BLUE))
    settings = (("g1s_h1_b20", "1 s rows, 20 bins"), ("g30s_h1_b4", "30 s rows, 4 bins"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4), sharey=True)
    y = np.arange(len(models))[::-1]
    for row, (s, title) in enumerate(settings):
        for col, (metric, fmt, label) in enumerate((("acc", "{:.3f}", "accuracy (higher is better)"),
                                                    ("mae", "{:.1f}", "MAE, requests/s (lower is better)"))):
            ax = axes[row, col]
            for yi, (m_acc, m_mae, name, color) in zip(y, models):
                v = d.loc[(s, m_acc if metric == "acc" else m_mae), metric]
                ax.scatter(v, yi, s=60, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
                ax.annotate(fmt.format(v), (v, yi), xytext=(8, 0), textcoords="offset points", va="center",
                            fontsize=8.5)
            ax.set_yticks(y, [m[2] for m in models], fontsize=9)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            ax.margins(x=0.18, y=0.12)
            ax.set_title(f"{title}: {label}", fontsize=10)
    fig.suptitle("DBN against TabPFN and gradient boosting, throughput_3, identical test samples",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "pooled over the three load patterns; DBN MAE from its conditional-median read-out; "
             "clean run of 13 Oct 2026 (commit e48dbda)",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT / "tabpfn_comparison.png", dpi=160)


def fig_structure():
    files = sorted(glob.glob(str(ROOT / "results/models/*_structure.csv")))
    s = pd.read_csv(files[-1])
    names = {"avg_p_latency": "latency", "throughput": "throughput", "buffer_size": "buffer size",
             "data_quality": "quality", "cores": "cores", "capacity": "capacity"}

    def label(v):
        if v == "rps":
            return "load (rps)"
        base, _, i = v.rpartition("_")
        return f"{names.get(base, base)} {i}"

    # services 4.5 apart; settings above, throughput left, latency right-below,
    # each buffer between the service that fills it and the one that empties it
    pos = {"rps": (-2.1, 1.6)}
    for i in (1, 2, 3):
        x = 4.5 * (i - 1)
        pos.update({f"capacity_{i}": (x, 3.0), f"cores_{i}": (x + 1.45, 3.0), f"data_quality_{i}": (x + 2.5, 3.0),
                    f"throughput_{i}": (x, 1.6), f"avg_p_latency_{i}": (x + 1.95, 0.45)})
    pos.update({"buffer_size_2": (2.25, -0.9), "buffer_size_3": (6.75, -0.9)})
    exog_prefix = ("cores_", "data_quality_", "capacity_")

    def draw(ax, edges, chain=False):
        nodes = set(edges.child) | set(edges.parent)
        own = edges[edges.parent == edges.child].groupby("child").parent_slice.apply(
            lambda x: ", ".join(sorted(set(x), key=["t", "t-1", "t+h"].index)))
        slices = edges[edges.parent != edges.child].groupby(["parent", "child"]).parent_slice.apply(set)
        boxes = {}
        for v in nodes:
            x, y = pos[v]
            known = v.startswith(exog_prefix) or v == "rps"
            text = label(v) + (f"\nown past: {own[v]}" if v in own.index else "")
            boxes[v] = ax.text(x, y, text, ha="center", va="center", fontsize=8.5, linespacing=1.3, zorder=3,
                               bbox=dict(boxstyle="round,pad=0.45", fc="#e8f0fb" if known else SURFACE,
                                         ec=BLUE if known else MUTED, lw=1))
        for (p, c), sl in slices.items():
            known = p.startswith(exog_prefix) or p == "rps"
            if known:
                color, ls = BLUE, "-"
            elif "t+h" in sl:                       # same step, predicted along the chain
                color, ls = ORANGE, "-"
            else:
                color, ls = MUTED, "-" if "t" in sl else "--"
            # arrows start and end on the box outlines (patchA / patchB), so heads stay visible
            ax.annotate("", xy=pos[c], xytext=pos[p], zorder=2, arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.6, ls=ls, shrinkA=3, shrinkB=3, mutation_scale=12,
                patchA=boxes[p].get_bbox_patch(), patchB=boxes[c].get_bbox_patch(),
                connectionstyle="arc3,rad=0.08"))
            if not known and sl == {"t", "t-1"}:
                (x0, y0), (x1, y1) = pos[p], pos[c]
                flat = abs(y1 - y0) < 0.3
                f = 0.62 if flat else 0.3              # clear of the arrows that cross the midpoint
                ax.text(x0 + f * (x1 - x0) + (0 if flat else 0.2), y0 + f * (y1 - y0) - (0.32 if flat else 0),
                        "t, t-1", ha="center" if flat else "left", va="center", fontsize=7.5, color=MUTED)
        for i in (1, 2, 3):
            ax.text(4.5 * (i - 1) + 1.0, 3.75, f"service {i}", ha="center", fontsize=10, color=MUTED,
                    fontweight="bold")
        ax.set_xlim(-3.0, 12.2)
        ax.axis("off")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": (3.2, 1.5)})
    draw(axes[0], s[s.used == "within a run"])
    axes[0].set_ylim(-1.5, 4.1)
    axes[0].set_title("Mode 1: every normal second (settings unchanged), predicted from the current state of the system", loc="left")
    chain = s[s.used == "reconfiguration chain of throughput_3"]
    draw(axes[1], chain, chain=True)
    axes[1].set_ylim(1.0, 4.1)
    axes[1].set_title("Mode 2: the first second after a reconfiguration, predicted from the new settings, service by service", loc="left")
    handles = [plt.Line2D([], [], color=MUTED, lw=1.6, label="from the state at t"),
               plt.Line2D([], [], color=BLUE, lw=1.6, label="from a setting known for t+1"),
               plt.Line2D([], [], color=ORANGE, lw=1.6, label="predicted value at t+1, passed down the chain")]
    fig.legend(handles=handles, loc="lower center", ncols=4, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.suptitle("How the DBN predicts the next second: two working modes",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.canvas.draw()                  # place the box outlines before the arrows are clipped to them
    fig.savefig(OUT / "learned_structure.png", dpi=160)


def fig_whatif_dumbbell():
    """
    The unseen-configuration test as one row per model: accuracy (and MAE) with the raw
    settings -> with the capacities, so the effect of the input and the ranking of the
    models read separately. Same data as fig_whatif; written to a new file.
    """
    files = newest("results/whatif/whatif_expanding_*_runs.csv.gz", lambda n: n.split("expanding_")[1][:8])
    d = pd.concat([pd.read_csv(f) for f in files])
    cap_files = sorted(glob.glob(str(ROOT / "results/whatif_capacity/*_runs.csv.gz")))
    c = pd.read_csv(cap_files[-1])
    d = pd.concat([d, c[c.model.str.startswith("tabpfn") & c.model.str.contains("capacities")]])
    d = d[d.new_config & (d.target == "throughput_3")]
    g = d.groupby("model")[["n", "correct", "ae"]].sum()
    acc, mae = g.correct / g.n, g.ae / g.n
    tab = next(m for m in acc.index if m.startswith("tabpfn") and "capacities" not in m)
    tab_cap = next(m for m in acc.index if m.startswith("tabpfn") and "capacities" in m)
    # (label, raw-settings model, capacity model, extra capacity model drawn as a third point)
    rows = [("DBN, mode 2 (chain of tables)", "table chain", "capacity chain", "capacity chain, noisy-min prior"),
            ("TabPFN v3.5", tab, tab_cap, None),
            ("Gradient boosting", "boosting (cores, quality)", "boosting (capacities)", None)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, val, fmt, title in ((axes[0], acc, "{:.2f}", "Accuracy, 20 bins (higher is better)"),
                                (axes[1], mae, "{:.0f}", "Mean absolute error, req/s (lower is better)")):
        for yi, (label, raw, cap, extra) in zip(y, rows):
            ax.annotate("", xy=(val[cap], yi), xytext=(val[raw], yi),
                        arrowprops=dict(arrowstyle="-|>", color=GRID, lw=2.5, shrinkA=6, shrinkB=6))
            ax.scatter(val[raw], yi, s=70, color=NEUTRAL, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.scatter(val[cap], yi, s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(fmt.format(val[raw]), (val[raw], yi), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=MUTED)
            ax.annotate(fmt.format(val[cap]), (val[cap], yi), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
            if extra:
                ax.scatter(val[extra], yi, s=90, marker="D", color="#104281", edgecolor=SURFACE, linewidth=1.5,
                           zorder=4)
                ax.annotate(fmt.format(val[extra]), (val[extra], yi), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=8.5, color=INK, fontweight="bold")
        ax.axvline(val["min rule"], color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(f"min rule: {fmt.format(val['min rule'])}", (val["min rule"], y[-1] - 0.45),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED, va="center")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.margins(x=0.12)
        ax.set_ylim(-0.6, len(rows) - 0.3)
    axes[0].set_yticks(y, [r[0] for r in rows])
    handles = [plt.Line2D([], [], marker="o", ls="", color=NEUTRAL, markersize=8, label="raw settings (cores, quality)"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=8, label="capacities"),
               plt.Line2D([], [], marker="D", ls="", color="#104281", markersize=8,
                          label="capacities + noisy-min prior (DBN)"),
               plt.Line2D([], [], color=MUTED, ls="--", label="deterministic min rule, no learning")]
    fig.legend(handles=handles, loc="lower center", ncols=4, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.suptitle("Capacity helps every model on configurations never seen in training; "
                 "the DBN chain with the noisy-min prior is the most accurate",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.905, "throughput 3, 30 s rows, unseen combinations of cores and data quality, "
             "pooled over the three load patterns; each row: one model without -> with the capacities",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.88))
    fig.savefig(OUT / "unseen_configurations_v2.png", dpi=160)


def fig_whatif_steps():
    """
    The unseen-configuration test in two panels: (A) three versions of the DBN chain,
    (B) all models given the capacities. Neutral labels only; the reading is in the text.
    Same data as fig_whatif; written to a new file.
    """
    files = newest("results/whatif/whatif_expanding_*_runs.csv.gz", lambda n: n.split("expanding_")[1][:8])
    d = pd.concat([pd.read_csv(f) for f in files])
    c = pd.read_csv(sorted(glob.glob(str(ROOT / "results/whatif_capacity/*_runs.csv.gz")))[-1])
    d = pd.concat([d, c[c.model.str.startswith("tabpfn") & c.model.str.contains("capacities")]])
    d = d[d.new_config & (d.target == "throughput_3")]
    g = d.groupby("model")[["n", "correct", "ae"]].sum()
    acc, mae = g.correct / g.n, g.ae / g.n
    tab_cap = next(m for m in acc.index if m.startswith("tabpfn") and "capacities" in m)
    full = "capacity chain, noisy-min prior"
    DARK = "#104281"

    def panel(ax, rows, val, fmt, title, xlabel, ylabels=True):
        y = np.arange(len(rows))[::-1]
        for yi, (m, label, color) in zip(y, rows):
            ax.barh(yi, val[m], height=0.55, color=color)
            ax.annotate(fmt.format(val[m]), (val[m], yi), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=9)
        ax.set_yticks(y, [r[1] for r in rows] if ylabels else [""] * len(rows))
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.15)
        ax.set_xlim(left=0)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10.5, loc="left")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.4), gridspec_kw={"height_ratios": (3, 4)})
    versions = [("table chain", "cores and quality", "#9ec5f4"),
                ("capacity chain", "capacity", "#5598e7"),
                (full, "capacity + noisy-min prior", DARK)]
    panel(axes[0, 0], versions, acc, "{:.2f}", "A. DBN chain, three versions", "accuracy (20 bins)")
    panel(axes[0, 1], versions, mae, "{:.0f}", "", "mean absolute error (req/s)", ylabels=False)
    models = [(full, "DBN (capacity + noisy-min prior)", DARK), (tab_cap, "TabPFN v3.5", ORANGE),
              ("boosting (capacities)", "gradient boosting", NEUTRAL), ("min rule", "min rule (no learning)", "#c8c7c3")]
    panel(axes[1, 0], models, acc, "{:.2f}", "B. All models, capacity inputs", "accuracy (20 bins)")
    panel(axes[1, 1], models, mae, "{:.0f}", "", "mean absolute error (req/s)", ylabels=False)
    fig.suptitle("Throughput 3, configurations never seen in training", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    fig.text(0.01, 0.01, "inputs: load and the settings of the three services, no history; 30 s rows; "
             "pooled over the three load patterns", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95), h_pad=2.5)
    fig.savefig(OUT / "unseen_configurations_v3.png", dpi=160)


def parent_frequency():
    """
    Share of the cross-validation fits (5 folds x 3 patterns, recommended configuration,
    1 s, 20 bins) in which each variable was chosen as a parent of each target, any time
    slice. The target's own previous value (t-1) is reported as the column "own_t-1"; its
    own value at t is forced and therefore not counted.
    """
    files = newest("results/ablation/recommended_*_folds.csv", lambda n: n.split("recommended_")[1][:8])
    d = pd.concat([pd.read_csv(f) for f in files])
    d = d[(d.model == "model") & ~d.config.str.contains("n_bins|granularity")].copy()
    d["target"] = d.config.str.extract(r"target=(\w+)")[0].fillna("throughput_3")
    rows = []
    for t, g in d.groupby("target"):
        n = len(g)
        count = {}
        for ps in g.parents:
            chosen = set()
            for p in json.loads(ps):
                var = p.split("@")[0]
                if var == "y":
                    if p == "y@t-1":
                        chosen.add("own_t-1")
                    continue
                chosen.add(var)
            for v in chosen:
                count[v] = count.get(v, 0) + 1
        rows += [dict(target=t, parent=v, freq=k / n, fits=n) for v, k in count.items()]
    return pd.DataFrame(rows)


def service_of(v):
    tail = v.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def is_candidate(target, parent):
    """The local candidate pool of the recommended configuration (ablation.local_pool)."""
    if parent == "own_t-1":
        return True
    i = service_of(target)
    upstream = "rps" if i == 1 else f"throughput_{i - 1}"
    core = parent.split("_")[0] in ("throughput", "cores", "data", "buffer", "avg", "capacity") or parent == "rps"
    return core and parent != target and (service_of(parent) == i or parent == upstream)


def fig_parent_heatmap():
    f = parent_frequency()
    n_fits = int(f.fits.iloc[0])
    cols = ["own_t-1", "rps"] + [f"{p}_{i}" for i in (1, 2, 3) for p in
                                  ("throughput", "avg_p_latency", "buffer_size", "cores", "data_quality", "capacity")]
    cols = [c for c in cols if c != "buffer_size_1" and any(is_candidate(t, c) for t in TARGET_ORDER)]
    M = f.pivot(index="target", columns="parent", values="freq").reindex(index=TARGET_ORDER, columns=cols)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9ec5f4", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for r, t in enumerate(TARGET_ORDER):
        for c, p in enumerate(cols):
            if not is_candidate(t, p):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=SURFACE, edgecolor=GRID,
                                           hatch="////", linewidth=0))
                continue
            v = M.loc[t, p]
            v = 0.0 if pd.isna(v) else v
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=ramp(v), edgecolor=SURFACE, linewidth=2))
            if v > 0:
                ax.text(c, r, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color=SURFACE if v >= 0.55 else INK)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(TARGET_ORDER) - 0.5, -0.5)
    names = {"own_t-1": "own value\nat t-1", "rps": "load"}
    ax.set_xticks(range(len(cols)), [names.get(c, c.replace("avg_p_latency", "latency").replace("data_quality", "quality")
                                               .replace("buffer_size", "buffer size").replace("_", " ")) for c in cols],
                  rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(TARGET_ORDER)), [target_label(t).replace("\n", " ") for t in TARGET_ORDER], fontsize=9)
    ax.set_xlabel("candidate parent (any time slice)")
    ax.set_title(f"How often each parent was chosen across the {n_fits} cross-validation fits", pad=10)
    fig.text(0.01, 0.01, "recommended configuration, 1 s rows, 20 bins; 5 folds x 3 load patterns; every variable's own "
             "value at t is always a parent (its value at t-1 is the first column); hatched = not a candidate", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "parent_frequency_heatmap.png", dpi=160)


def fig_structure_consensus(min_freq=1 / 3):
    """The mode-1 structure drawn from the cross-validation fits: every edge chosen in at
    least `min_freq` of them, its width and label giving how often."""
    f = parent_frequency()
    n_fits = int(f.fits.iloc[0])
    names = {"avg_p_latency": "latency", "throughput": "throughput", "buffer_size": "buffer size",
             "data_quality": "quality", "cores": "cores", "capacity": "capacity"}

    def label(v):
        if v == "rps":
            return "load (rps)"
        base, _, i = v.rpartition("_")
        return f"{names.get(base, base)} {i}"

    pos = {"rps": (-2.1, 1.6)}
    for i in (1, 2, 3):
        x = 4.5 * (i - 1)
        pos.update({f"capacity_{i}": (x, 3.0), f"cores_{i}": (x + 1.45, 3.0), f"data_quality_{i}": (x + 2.5, 3.0),
                    f"throughput_{i}": (x, 1.6), f"avg_p_latency_{i}": (x + 1.95, 0.45)})
    pos.update({"buffer_size_2": (2.25, -0.9), "buffer_size_3": (6.75, -0.9)})
    exog = ("cores_", "data_quality_", "capacity_")
    edges = f[(f.parent != "own_t-1") & (f.freq >= min_freq - 1e-9)]
    own = f[f.parent == "own_t-1"].set_index("target").freq
    nodes = set(TARGET_ORDER) | set(edges.parent)

    fig, ax = plt.subplots(figsize=(12, 6.4))
    boxes = {}
    for v in nodes:
        x, y = pos[v]
        known = v.startswith(exog) or v == "rps"
        text = label(v) + (f"\nown t-1: {own.get(v, 0):.0%}" if v in TARGET_ORDER else "")
        boxes[v] = ax.text(x, y, text, ha="center", va="center", fontsize=8.5, linespacing=1.3, zorder=3,
                           bbox=dict(boxstyle="round,pad=0.45", fc="#e8f0fb" if known else SURFACE,
                                     ec=BLUE if known else MUTED, lw=1))
    for _, e in edges.iterrows():
        known = e.parent.startswith(exog) or e.parent == "rps"
        color = BLUE if known else MUTED
        ax.annotate("", xy=pos[e.target], xytext=pos[e.parent], zorder=2, arrowprops=dict(
            arrowstyle="-|>", color=color, lw=0.6 + 3.0 * e.freq, shrinkA=3, shrinkB=3, mutation_scale=12,
            patchA=boxes[e.parent].get_bbox_patch(), patchB=boxes[e.target].get_bbox_patch(),
            connectionstyle="arc3,rad=0.08"))
        (x0, y0), (x1, y1) = pos[e.parent], pos[e.target]
        k = 0.42                       # nearer the parent, so two opposite edges keep apart
        ax.text(x0 + k * (x1 - x0), y0 + k * (y1 - y0), f"{e.freq:.0%}", ha="center", va="center", fontsize=8,
                color=color, zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
    for i in (1, 2, 3):
        ax.text(4.5 * (i - 1) + 1.0, 3.75, f"service {i}", ha="center", fontsize=10, color=MUTED, fontweight="bold")
    ax.set_xlim(-3.0, 12.2)
    ax.set_ylim(-1.5, 4.1)
    ax.axis("off")
    handles = [plt.Line2D([], [], color=MUTED, lw=2, label="from the state at t or t-1"),
               plt.Line2D([], [], color=BLUE, lw=2, label="from a setting known for t+1"),
               plt.Line2D([], [], color=MUTED, lw=0.6 + 3.0 * 0.4, label="width = share of fits choosing the edge")]
    fig.legend(handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.suptitle(f"Edges chosen in at least {min_freq:.0%} of the {n_fits} cross-validation fits (mode 1)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.canvas.draw()                  # place the box outlines before the arrows are clipped to them
    fig.savefig(OUT / "learned_structure_consensus.png", dpi=160)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="names of the figure functions to run, e.g. fig_parent_heatmap")
    only = ap.parse_args().only
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_bins, fig_rollout, fig_whatif, fig_detection, fig_targets, fig_targets_runs, fig_options, fig_tabpfn,
              fig_structure, fig_structure_consensus, fig_parent_heatmap, fig_whatif_dumbbell, fig_whatif_steps):
        if only and f.__name__ not in only:
            continue
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:
            print("skipped", f.__name__, type(e).__name__, e)
        plt.close("all")


if __name__ == "__main__":
    main()
