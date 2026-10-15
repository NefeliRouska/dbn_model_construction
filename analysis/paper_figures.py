"""
paper_figures.py -- figures (and two LaTeX tables) for parts 1 and 2 of the paper.

Part 1, how the model was built
    p1_buildup                the recommended DBN built one decision at a time
    p1_options                one modelling option at a time, against the reference DBN
    p1_bins_resolution        bins x time resolution, and the target's discretisation
    p1_structure              the learned structure, two working modes (as make_figures.fig_structure)
    p1_structure_consensus    edges by how often the 15 cross-validation fits chose them (supplementary)
    p1_parent_frequency       every candidate parent x target (supplementary)
    tables/heldout_pattern    the held-out-pattern check
Part 2, comparison with other models
    p2_all_variables          the eight state variables against their AR tables
    p2_runs                   test runs by outcome against the AR table (supplementary)
    p2_comparison             DBN against persistence, AR, gradient boosting and TabPFN v3.5
    tables/comparison         the numbers of p2_comparison with 95% intervals

Titles describe what is plotted, never what it shows; the reading belongs in the text.
Reads only result files; writes only to docs/figures/paper/ (one PNG per figure).

Usage:
    python analysis/paper_figures.py              # everything
    python analysis/paper_figures.py p1_buildup   # selected figures
"""
import glob
import sys
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as M  # noqa: E402  (style, colours and a few readers; nothing is regenerated there)

ROOT = M.ROOT
OUT = ROOT / "docs" / "figures" / "paper"
TABLES = OUT / "tables"
ABL = ROOT / "results" / "ablation"
BLUE, ORANGE, NEUTRAL, MUTED, INK, GRID, SURFACE = M.BLUE, M.ORANGE, M.NEUTRAL, M.MUTED, M.INK, M.GRID, M.SURFACE
RED, DARK = M.RED, "#104281"
LIGHT_GREY = "#c8c7c3"
REC = "ar_order=2,capacity=True,numeric=True,on_change=capchain,other_lags=2,pool=local"   # throughput_3, 1 s, 20 bins


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def summary(study):
    return pd.read_csv(ABL / f"summary_{study}.csv").set_index("config")


def pp(x):
    return 100 * np.asarray(x, dtype=float)


# ============================================================
# PART 1
# ============================================================
def runs_of(study, config):
    """Per-run sums of one configuration of an ablation study, newest file of every load pattern."""
    files = sorted(glob.glob(str(ABL / f"{study}_*_runs.csv.gz")))
    newest = {Path(f).name[len(study) + 1:].split("_run")[0]: f for f in files}.values()
    d = pd.concat([pd.read_csv(f) for f in newest])
    return d[d.config == config]


def paired(d, k, n, n_boot=10000, seed=0):
    """DBN and AR accuracy on a subset (k = correct column, n = count column) and the 95% bootstrap
    interval of their difference; schedule positions (run_id) are resampled, the three load patterns together."""
    g = d.groupby(["model", "run_id"])[[k, n]].sum()
    m, a = g.loc["model"], g.loc["ar"].reindex(g.loc["model"].index)
    mk, mn, ak, an = m[k].to_numpy(), m[n].to_numpy(), a[k].to_numpy(), a[n].to_numpy()
    draws = np.random.default_rng(seed).integers(0, len(m), size=(n_boot, len(m)))
    diff = mk[draws].sum(1) / mn[draws].sum(1) - ak[draws].sum(1) / an[draws].sum(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return dict(dbn=mk.sum() / mn.sum(), ar=ak.sum() / an.sum(), lo=lo, hi=hi)


BUILDUP_STEPS = [("bins_granularity", "engine=hc-bic,prior=bdeu", "Standard DBN (BIC search, uniform prior)"),
                 ("ofat", "ref", "+ backoff prior, cross-validated parent search"),
                 ("constraints", "ar_order=2,other_lags=2", "+ two time steps of history"),
                 ("constraints", "ar_order=2,other_lags=2,pool=local", "+ candidates: own service and incoming flow"),
                 ("recommended", REC, "+ capacity chain at reconfigurations")]
BUILDUP_SUBSETS = [("correct", "n"), ("correct_change", "n_change"), ("correct_xrun", "n_xrun")]


def buildup_results():
    return {(i, j): paired(runs_of(st, cf), k, n) for i, (st, cf, _) in enumerate(BUILDUP_STEPS)
            for j, (k, n) in enumerate(BUILDUP_SUBSETS)}


def p1_buildup():
    """One row per decision, top to bottom; throughput 3, 1 s, 20 bins, same test samples throughout.
    Accuracy per panel against AR(1), with the 95% interval of the DBN-AR difference. Panel titles say which samples are scored,
    the legend sits at the bottom, everything else (row logic, data settings, how to read significance) is caption."""
    steps = BUILDUP_STEPS
    titles = [("A. All test samples\n ", (0.85, 0.95)),
              ("B. Bin changes\nbin at t+1 differs from bin at t", (0, 1)),
              ("C. Reconfigurations\nfirst second after new cores or data quality", (0, 1))]
    res = buildup_results()
    y = np.arange(len(steps))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True, gridspec_kw={"width_ratios": (1.2, 1, 1)})
    for j, (ax, (title, xlim)) in enumerate(zip(axes, titles)):
        ar = res[(0, j)]["ar"]
        ax.axvline(ar, color=NEUTRAL, linewidth=1.3, linestyle="--", zorder=1)
        ax.annotate(f"AR(1) {ar:.3f}" if j == 0 else f"AR(1) {ar:.2f}", (ar, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(4, 2), textcoords="offset points",
                    fontsize=8.5, color=MUTED, va="bottom")
        for i, yi in enumerate(y):
            r = res[(i, j)]
            ax.hlines(yi, r["ar"] + r["lo"], r["ar"] + r["hi"], color=BLUE, linewidth=2.5, alpha=0.45, zorder=2)
            ax.scatter(r["dbn"], yi, s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(f"{r['dbn']:.3f}" if j == 0 else f"{r['dbn']:.2f}",
                        (max(r["dbn"], r["ar"] + r["hi"]), yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5)
        ax.set_xlim(*xlim)
        ax.set_xlabel("accuracy, throughput 3 (next-second bin)")
        head, sub = title.split("\n")
        ax.set_title(head, fontsize=10.5, pad=30, loc="left")
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points",
                    fontsize=8.5, color=MUTED, va="bottom")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.margins(y=0.12)
    axes[0].set_yticks(y, [s_[2] for s_ in steps])
    fig.suptitle("Building the recommended DBN, one decision at a time", x=0.01, y=0.985,
                 ha="left", fontsize=12, fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=7, markeredgecolor=SURFACE, label="DBN"),
               plt.Line2D([], [], color=BLUE, lw=2.5, alpha=0.45,
                          label="95% interval of the DBN's difference to AR(1)"),
               plt.Line2D([], [], color=NEUTRAL, lw=1.3, ls="--", label="AR(1)")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.0), ncols=3, fontsize=8.5,
               handlelength=2.2, columnspacing=1.5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, "p1_buildup")


OPTION_GROUPS_P1 = M.OPTION_GROUPS[:1] + (
    (M.OPTION_GROUPS[1][0], tuple((cfg, "no search: own past only, AR(1)" if cfg == "engine=ar" else label)
                                  for cfg, label in M.OPTION_GROUPS[1][1])),
) + (
    ("Table prior (vs backoff to AR(1), strength 10)", (
        ("prior=bdeu", "uniform BDeu prior"),
        ("engine=hc-bic,prior=bdeu", "uniform prior + BIC search (pgmpy defaults)"),
        ("ess=1.0", "prior strength 1"),
        ("ess=3.0", "prior strength 3"),
        ("ess=30.0", "prior strength 30"),
        ("ess=100.0", "prior strength 100"),
    )),
) + M.OPTION_GROUPS[3:]


def p1_options():
    """Forest plot: one option changed at a time from the reference DBN (task kept fixed); rows sorted by
    effect within each group."""
    d = summary("ofat")
    ref = d.loc["ref", "acc"]
    entries = []
    for group, items in OPTION_GROUPS_P1:
        entries.append((group, None))
        rows = [(label, cfg) for cfg, label in items if cfg in d.index]
        entries += sorted(rows, key=lambda e: -d.loc[e[1], "d_acc_vs_ref"])
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(entries) + 2.2))
    y = np.arange(len(entries))[::-1]
    for yi, (label, cfg) in zip(y, entries):
        if cfg is None:
            continue
        r = d.loc[cfg]
        dv, lo, hi = pp(r.d_acc_vs_ref), pp(r.d_acc_lo_vs_ref), pp(r.d_acc_hi_vs_ref)
        color = BLUE if lo > 0 else RED if hi < 0 else NEUTRAL
        ax.hlines(yi, lo, hi, color=color, linewidth=2.5, alpha=0.45)
        ax.scatter(dv, yi, s=55, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if round(dv, 1) == 0:
            ax.annotate("0.0", (hi, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5)
        elif dv < 0:
            ax.annotate(f"{dv:+.1f}", (lo, yi), xytext=(-6, 0), textcoords="offset points", va="center",
                        ha="right", fontsize=8.5)
        else:
            ax.annotate(f"{dv:+.1f}", (hi, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5)
    ax.set_yticks(y, [label for label, _ in entries], fontsize=9)
    for tick, (_, cfg) in zip(ax.get_yticklabels(), entries):
        if cfg is None:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.annotate(f"reference DBN {ref:.3f}", (0, 1.0), xycoords=("data", "axes fraction"), xytext=(4, 2),
                textcoords="offset points", fontsize=8.5, color=MUTED, va="bottom")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.margins(x=0.08)
    ax.set_xlabel("accuracy minus reference DBN, throughput 3 (percentage points)")
    fig.suptitle("One modelling option changed at a time", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    dot = dict(marker="o", ls="", markersize=7, markeredgecolor=SURFACE)
    handles = [plt.Line2D([], [], color=BLUE, label="significantly better than the reference", **dot),
               plt.Line2D([], [], color=NEUTRAL, lw=2.5, alpha=0.45,
                          label="95% interval of the difference (bootstrap over schedule positions)"),
               plt.Line2D([], [], color=NEUTRAL, label="no significant difference", **dot),
               plt.Line2D([], [], color=INK, lw=1.2, label="reference DBN (difference 0)"),
               plt.Line2D([], [], color=RED, label="significantly worse than the reference", **dot)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.0), ncols=3, fontsize=8.5,
               handlelength=2.2, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save(fig, "p1_options")


def p1_bins_resolution():
    """Gain over AR by number of bins, one line per time resolution; reference and standard DBN; and the
    target's discretisation method. Gains, not accuracies: accuracy itself falls with bins for every model."""
    d = summary("bins_granularity")
    o = summary("ofat")
    bins = [4, 6, 10, 20, 30, 50]
    res = [(1, BLUE), (5, ORANGE), (10, M.AQUA), (30, M.YELLOW)]

    def cfg(g, b, standard):
        parts = (["engine=hc-bic"] if standard else []) + ([f"granularity={g}"] if g != 1 else []) + \
                ([f"n_bins={b}"] if b != 20 else []) + (["prior=bdeu"] if standard else [])
        return ",".join(parts) if parts else "ref"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": (1, 1, 0.75)})
    for ax, standard, title in ((axes[0], False, "A. DBN (reference configuration)"),
                                (axes[1], True, "B. Standard DBN (BIC search, uniform prior)")):
        for k, (g, color) in enumerate(res):
            x = np.arange(len(bins)) + (k - 1.5) * 0.09
            r = d.loc[[cfg(g, b, standard) for b in bins]]
            ax.errorbar(x, pp(r.d_acc_vs_ar), yerr=[pp(r.d_acc_vs_ar - r.d_acc_lo_vs_ar),
                                                    pp(r.d_acc_hi_vs_ar - r.d_acc_vs_ar)],
                        color=color, marker="o", markersize=5, markeredgecolor=SURFACE, linewidth=1.8,
                        capsize=0, elinewidth=1.2, label=f"{g} s rows")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(np.arange(len(bins)), bins)
        ax.set_xlabel("number of bins of throughput 3")
        ax.set_title(title, fontsize=10.5)
    lo = min(a.get_ylim()[0] for a in axes[:2])
    hi = max(a.get_ylim()[1] for a in axes[:2])
    for a in axes[:2]:
        a.set_ylim(lo, hi)
    axes[0].set_ylabel("accuracy minus AR table (percentage points)")
    axes[0].legend(loc="upper left", fontsize=9)

    ax = axes[2]
    disc = [("ref", "equal width (used)"), ("disc=quantile", "equal frequency"), ("disc=kmeans", "k-means")]
    y = np.arange(len(disc))[::-1]
    for yi, (c, label) in zip(y, disc):
        r = o.loc[c]
        ax.hlines(yi, pp(r.d_acc_lo_vs_ar), pp(r.d_acc_hi_vs_ar), color=BLUE, linewidth=2, alpha=0.45)
        ax.scatter(pp(r.d_acc_vs_ar), yi, s=50, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{pp(r.d_acc_vs_ar):+.1f}", (pp(r.d_acc_hi_vs_ar), yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, [lab for _, lab in disc])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.margins(x=0.2, y=0.25)
    ax.set_xlabel("minus AR (percentage points)")
    ax.set_title("C. Discretisation of the target", fontsize=10.5)
    fig.suptitle("Number of bins, time resolution and discretisation (throughput 3)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "pooled over the three load patterns; bars: 95% intervals (bootstrap over schedule "
             "positions); AR scored on the same bins and samples in every case; C: 1 s rows, 20 bins", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save(fig, "p1_bins_resolution")


def reuse(fn, name, title=None):
    """Run a make_figures function with its output redirected to this folder, as PNG."""
    orig_out, orig_save = M.OUT, matplotlib.figure.Figure.savefig
    captured = {}

    def grab(self, path, **kw):
        captured["fig"] = self

    try:
        matplotlib.figure.Figure.savefig = grab
        fn()
    finally:
        matplotlib.figure.Figure.savefig = orig_save
        M.OUT = orig_out
    fig = captured["fig"]
    if title is not None and fig._suptitle is not None:
        fig._suptitle.set_text(title)
    fig.canvas.draw()
    save(fig, name)


def p1_structure():
    reuse(M.fig_structure, "p1_structure", "The learned DBN: two working modes")


def p1_structure_consensus():
    reuse(M.fig_structure_consensus, "p1_structure_consensus")


def p1_parent_frequency():
    reuse(M.fig_parent_heatmap, "p1_parent_frequency")


def table_heldout():
    f = sorted(f for f in glob.glob(str(ROOT / "results/heldout/heldout_throughput_3_run*.csv"))
               if not f.endswith("_trace.csv"))[-1]
    h = pd.read_csv(f)
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Held-out pattern & Chosen on & Chosen config. & Recommended & AR & Gain over AR [95\% CI] \\",
             r"\midrule"]
    for _, r in h.iterrows():
        lines.append(f"{r.held_out} & {r.chosen_on} & {r.acc:.3f} & {r.rec_acc:.3f} & {r.acc_ar:.3f} & "
                     f"${r.d_acc_vs_ar:+.3f}$ [{r.d_acc_lo:.3f}, {r.d_acc_hi:.3f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "heldout_pattern.tex").write_text("\n".join(lines) + "\n")
    h.to_csv(TABLES / "heldout_pattern.csv", index=False)


# ============================================================
# PART 2
# ============================================================
def p2_all_variables():
    d = summary("recommended")
    d = d[~d.index.str.contains("n_bins|granularity")].copy()
    d["target"] = d.index.str.extract(r"target=(\w+)")[0].fillna("throughput_3").to_numpy()
    d = d.set_index("target").loc[M.TARGET_ORDER]
    x = np.arange(len(d))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": (1.2, 1)})
    ax = axes[0]
    ax.vlines(x, d.acc_ar, d.acc, color=GRID, linewidth=3, zorder=1)
    ax.scatter(x, d.acc_ar, s=55, color=NEUTRAL, edgecolor=SURFACE, linewidth=1.5, zorder=2, label="AR table")
    ax.scatter(x, d.acc, s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="DBN")
    ax.set_ylabel("accuracy")
    ax.set_title("A. Accuracy", fontsize=10.5)
    ax.legend(loc="lower left", ncols=2, fontsize=9)
    ax = axes[1]
    ax.vlines(x, pp(d.d_acc_lo_vs_ar), pp(d.d_acc_hi_vs_ar), color=BLUE, linewidth=2, alpha=0.45)
    ax.scatter(x, pp(d.d_acc_vs_ar), s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for xi, v, hi in zip(x, pp(d.d_acc_vs_ar), pp(d.d_acc_hi_vs_ar)):
        ax.annotate(f"{v:+.1f}", (xi, hi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8.5)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_ylabel("DBN minus AR\n(percentage points)")
    ax.set_title("B. Difference with 95% interval", fontsize=10.5)
    ax.set_ylim(min(0, pp(d.d_acc_lo_vs_ar).min()) - 0.4, pp(d.d_acc_hi_vs_ar).max() + 0.9)
    for a in axes:
        for b in (2.5, 5.5):
            a.axvline(b, color=GRID, linewidth=1)
        a.margins(x=0.05)
    axes[1].set_xticks(x, [M.target_label(t) for t in d.index])
    fig.suptitle("DBN and AR table for each state variable", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "recommended configuration; 1 s rows, 20 bins; pooled over the three load patterns; "
             "intervals: bootstrap over schedule positions", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "p2_all_variables")


def p2_runs():
    g = M.run_gains()
    tol = 0.5 / 361
    share = g.assign(better=g.gain > tol, worse=g.gain < -tol).groupby("target")[["better", "worse"]].mean()
    share["tied"] = 1 - share.better - share.worse
    share = share.loc[M.TARGET_ORDER]
    n_runs = g.groupby("target").size().iloc[0]
    x = np.arange(len(share))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bottom = np.zeros(len(share))
    for col, color, label in (("better", BLUE, "DBN higher than AR"), ("tied", "#d9d8d4", "equal"),
                              ("worse", RED, "DBN lower than AR")):
        v = share[col].to_numpy()
        ax.bar(x, v, bottom=bottom, width=0.62, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for xi, b, h in zip(x, bottom, v):
            if h >= 0.06:
                ax.text(xi, b + h / 2, f"{h:.0%}", ha="center", va="center", fontsize=8.5,
                        color=SURFACE if col != "tied" else INK)
        bottom += v
    for b in (2.5, 5.5):
        ax.axvline(b, color=GRID, linewidth=1)
    ax.set_xticks(x, [M.target_label(t) for t in share.index])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(f"share of test runs ({n_runs} per variable)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3, fontsize=9)
    fig.suptitle("Accuracy of DBN and AR table per test run", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "p2_runs")


# gradient boosting, TabPFN and the DBN read the same columns (ablation.information_columns)
COMPARISON_MODELS = [("persistence", "persistence", "persistence", LIGHT_GREY, "o"),
                     ("ar", "ar", "AR(1)", NEUTRAL, "o"),
                     ("hgb_rec", "hgb_rec", "gradient boosting", "#5f5e5a", "o"),
                     ("tabpfn_rec", "tabpfn_rec", "TabPFN v3.5", ORANGE, "o"),
                     ("dbn_rec", "dbn_rec_median", "DBN", BLUE, "o")]
SETTINGS = [("g1s_h1_b20", "1 s rows, 20 bins"), ("g30s_h1_b4", "30 s rows, 4 bins")]


def comparison_table(n_boot=2000):
    """Per setting and model: accuracy, log-loss, MAE with 95% bootstrap intervals over schedule positions.
    MAE uses column aeh, the error of the numeric forecast, as analysis/tabpfn_report.py does."""
    files = sorted(glob.glob(str(ROOT / "results/tabpfn/tabpfn-v3.5_*_runs.csv.gz")))
    newest = {Path(f).name.split("tabpfn-v3.5_")[1][:8]: f for f in files}.values()
    d = pd.concat([pd.read_csv(f) for f in newest])
    rows = []
    rng = np.random.default_rng(0)
    for setting, _ in SETTINGS:
        s = d[d.config == setting]
        for key, mae_key, label, _, _ in COMPARISON_MODELS:
            out = {"setting": setting, "model": label}
            for metric, m, col in (("acc", key, "correct"), ("logloss", key, "ll"), ("mae", mae_key, "aeh")):
                t = s[s.model == m].groupby("run_id")[["n", col]].sum()   # run_id = schedule position
                if t.empty or (metric == "logloss" and key in ("persistence", "hgb", "hgb_rec")):
                    out.update({metric: np.nan, f"{metric}_lo": np.nan, f"{metric}_hi": np.nan})
                    continue
                v, n = t[col].to_numpy(), t.n.to_numpy()
                draws = rng.integers(0, len(t), size=(n_boot, len(t)))
                boot = v[draws].sum(1) / n[draws].sum(1)
                out.update({metric: v.sum() / n.sum(), f"{metric}_lo": np.percentile(boot, 2.5),
                            f"{metric}_hi": np.percentile(boot, 97.5)})
            rows.append(out)
    return pd.DataFrame(rows)


def p2_comparison():
    t = comparison_table()
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "comparison.csv", index=False)
    lines = [r"\begin{tabular}{llccc}", r"\toprule", r"Setting & Model & Accuracy & Log-loss & MAE (req/s) \\",
             r"\midrule"]
    for setting, slabel in SETTINGS:
        for _, r in t[t.setting == setting].iterrows():
            cells = [f"{r[m]:.{p}f} [{r[m + '_lo']:.{p}f}, {r[m + '_hi']:.{p}f}]" if not np.isnan(r[m]) else "--"
                     for m, p in (("acc", 3), ("logloss", 3), ("mae", 1))]
            lines.append(f"{slabel} & {r.model} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (TABLES / "comparison.tex").write_text("\n".join(lines) + "\n")

    comparison_plot(t, (("acc", "{:.3f}", "accuracy"), ("logloss", "{:.2f}", "log-loss"),
                        ("mae", "{:.1f}", "MAE (req/s)")), "p2_comparison",
                    "DBN MAE from its conditional-median read-out; ")


def p2_comparison_main():
    """The main-text version: accuracy and log-loss only (MAE is in tables/comparison.tex)."""
    comparison_plot(comparison_table(), (("acc", "{:.3f}", "accuracy"), ("logloss", "{:.2f}", "log-loss")),
                    "p2_comparison_main", "")


def comparison_plot(t, metrics, name, note):
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.5 * len(metrics), 6.6), sharey=True, squeeze=False)
    y = np.arange(len(COMPARISON_MODELS))[::-1]
    for row, (setting, slabel) in enumerate(SETTINGS):
        s = t[t.setting == setting].set_index("model")
        for col, (metric, fmt, xlabel) in enumerate(metrics):
            ax = axes[row, col]
            for yi, (_, _, label, color, marker) in zip(y, COMPARISON_MODELS):
                v = s.loc[label, metric]
                if np.isnan(v):
                    ax.annotate("n/a", (0.02, yi), xycoords=("axes fraction", "data"), va="center",
                                fontsize=8, color=MUTED)
                    continue
                ax.hlines(yi, s.loc[label, f"{metric}_lo"], s.loc[label, f"{metric}_hi"], color=color, linewidth=2,
                          alpha=0.5)
                ax.scatter(v, yi, s=50, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.2, zorder=3)
                ax.annotate(fmt.format(v), (s.loc[label, f"{metric}_hi"], yi), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=8)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            ax.margins(x=0.22, y=0.1)
            if row == 1:
                ax.set_xlabel(xlabel)
            ax.set_title(f"{'AB'[row]}{col + 1}. {slabel}: {xlabel}", fontsize=10)
        axes[row, 0].set_yticks(y, [m[2] for m in COMPARISON_MODELS])
    fig.suptitle("DBN and other models on identical test samples (throughput 3)", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    fig.text(0.01, 0.01, f"2,000 test samples per fold, 5 folds x 3 load patterns; {note}bars: 95% intervals "
             "(bootstrap over schedule positions); n/a: point forecast, no probabilities", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, name)


ALL = [p1_buildup, p1_options, p1_bins_resolution, p1_structure, p1_structure_consensus, p1_parent_frequency,
       table_heldout, p2_all_variables, p2_runs, p2_comparison, p2_comparison_main]


def main():
    only = set(sys.argv[1:])
    for f in ALL:
        if only and f.__name__ not in only:
            continue
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, type(e).__name__, e)
        plt.close("all")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
