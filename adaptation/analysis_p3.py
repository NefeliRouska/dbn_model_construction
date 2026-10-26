"""
analysis_p3.py -- the analysis of the main adaptation experiment for the paper:
per-run metrics, paired comparisons against the DBN planner, and the Part 3
figures.

Input: the 90 run logs of the main experiment (results/adaptation_loop/main/,
or the archive main_experiment_20261022.tar.gz unpacked there). One run = one
planner on one load pattern for 3,600 s; the load trace of a pattern is the
same in every repetition, so runs are paired by (pattern, repetition).

Per run
  serving     slo_fulfillment (the planners' utility, measured every second),
              completion (throughput_3 / load, capped at 1), mean quality
  stability   changes_per_h      configurations sent per hour
              reversals_per_h    direction reversals of the applied settings
                                 (a setting goes up and later down, or the
                                 reverse), summed over the six settings
              core_moves_per_h   cores moved per hour (sum of |change|)
              the same three in the seconds of high load (load >= 0.4 x the
              maximum of the pattern, the split of Sedlak et al.), per hour
              of high load
  recovery    for every load step (load changes by >= JUMP of its value in one
              second) to a load the chain can serve (<= CEILING): seconds until
              the median completion of the next 5 s is >= RECOVERED again,
              censored at the next step or the end; mean over the steps of the
              run, and the number never recovered. Steps to a load above the
              ceiling (the 600 req/s spike) cannot be recovered by any planner
              and are analysed separately (spike)
  spike       the 180 s at 600 req/s of the once-in-a-lifetime pattern: each
              service takes at most 500 items per 1 s cycle (IoTService
              process_loop), so no configuration completes more than 500/600;
              completion, SLO fulfillment, quality and configuration changes
  prediction  for every configuration a planner sent with a prediction (DBN,
              min rule, TabPFN): its predicted completion minus the observed
              mean completion from 10 to 40 s after it was sent (or until the
              next change, if at least 10 s); mean absolute error and bias
  overhead    decision time, CPU cores and peak memory of the loop process

Comparisons: every planner against the DBN on the 15 (pattern, repetition)
pairs, paired bootstrap of the mean difference (10,000 resamples, 95%
percentile interval, two-sided p), Holm-corrected over the five comparisons
of a metric.

Figures (docs/figures/paper/, PNG; titles say what is plotted, legend at the
bottom, no comments on results):
  p3_fulfillment_stability   SLO fulfillment against configuration changes:
                             all 90 runs (per hour), and the overload spike
                             (per spike); one dot per planner, 95% intervals

Usage:
    python adaptation/analysis_p3.py [--logs results/adaptation_loop/main]
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "analysis"))
import make_figures as MF  # noqa: E402

OUT_TABLES = REPO / "results" / "adaptation_loop" / "main" / "analysis"
OUT_FIGS = REPO / "docs" / "figures" / "paper"
SETTINGS = [f"{p}_{i}" for p in ("cores", "data_quality") for i in (1, 2, 3)]
PLANNERS = {"dbn": ("DBN", "#2a78d6"), "minrule": ("min rule", "#eb6834"), "tabpfn": ("TabPFN v3.5", "#e87ba4"),
            "static": ("static", "#1baf7a"), "vpa": ("VPA", "#eda100"), "rask": ("RASK", "#008300")}
PATTERNS = ["periodic", "once_in_a_lifetime", "unpredictable", "diurnal", "bursty"]
HIGH_LOAD, JUMP, RECOVERED, RECOVER_WINDOW, CEILING = 0.4, 0.10, 0.97, 5, 500
PRED_FROM, PRED_TO, PRED_MIN = 10, 40, 10
N_BOOT, SEED = 10_000, 0
NAME_RE = re.compile(r"live_(?P<planner>[a-z]+)(?:_[a-z0-9]+)?_60min_seed(?P<rep>\d+)_")


def reversals(x):
    """Direction reversals of a step series (changes only)."""
    d = np.diff(np.asarray(x, dtype=float))
    d = d[d != 0]
    return int((np.sign(d[1:]) != np.sign(d[:-1])).sum()) if len(d) > 1 else 0


def stability(df, mask=None):
    """changes, reversals and core moves per hour, over the seconds in `mask`."""
    m = np.ones(len(df), dtype=bool) if mask is None else np.asarray(mask)
    hours = m.sum() / 3600
    if hours == 0:
        return dict(changes_per_h=np.nan, reversals_per_h=np.nan, core_moves_per_h=np.nan)
    acted = df.get("act", pd.Series(0, index=df.index)).fillna(0).astype(bool).to_numpy() & m
    rev = sum(reversals(df.loc[m, c]) for c in SETTINGS)
    moves = sum(np.abs(np.diff(df.loc[m, f"cores_{i}"].to_numpy(dtype=float))).sum() for i in (1, 2, 3))
    return dict(changes_per_h=acted.sum() / hours, reversals_per_h=rev / hours, core_moves_per_h=moves / hours)


def recovery(df):
    load = df["load_set"].to_numpy(dtype=float)
    comp = df["completion"].to_numpy(dtype=float)
    jumps = np.flatnonzero(np.abs(np.diff(load)) >= JUMP * np.maximum(load[:-1], 1.0)) + 1
    times, never = [], 0
    for k, t0 in enumerate(jumps):
        if load[t0] > CEILING:
            continue
        end = jumps[k + 1] if k + 1 < len(jumps) else len(df)
        rec = None
        for t in range(t0, end - RECOVER_WINDOW + 1):
            if np.median(comp[t:t + RECOVER_WINDOW]) >= RECOVERED:
                rec = t - t0
                break
        if rec is None:
            never += 1
            rec = end - t0
        times.append(rec)
    return dict(steps=len(times), recovery_s=float(np.mean(times)) if times else np.nan, never_recovered=never)


def spike(df):
    s = df[df["load_set"] > CEILING]
    if s.empty:
        return {}
    q = s[[f"data_quality_{i}" for i in (1, 2, 3)]].mean(axis=1)
    acted = s["act"].fillna(0).astype(bool).sum() if "act" in s else 0
    return dict(spike_completion=s["completion"].mean(), spike_slo=s["utility"].mean(), spike_quality=q.mean(),
                spike_changes=int(acted), spike_seconds=len(s))


def prediction(df):
    if "predicted_completion" not in df or "act" not in df:
        return dict(pred_n=0, pred_mae=np.nan, pred_bias=np.nan)
    comp = df["completion"].to_numpy(dtype=float)
    acts = np.flatnonzero(df["act"].fillna(0).astype(bool).to_numpy() & df["predicted_completion"].notna().to_numpy())
    errs = []
    for k, t in enumerate(acts):
        nxt = acts[k + 1] if k + 1 < len(acts) else len(df)
        lo, hi = t + PRED_FROM, min(t + PRED_TO, nxt)
        if hi - lo >= PRED_MIN:
            errs.append(float(df["predicted_completion"].iloc[t]) - comp[lo:hi].mean())
    errs = np.array(errs)
    return dict(pred_n=len(errs), pred_mae=float(np.abs(errs).mean()) if len(errs) else np.nan,
                pred_bias=float(errs.mean()) if len(errs) else np.nan)


def per_run(paths):
    rows = []
    for p in paths:
        m = NAME_RE.search(Path(p).name)
        df = pd.read_json(p, lines=True)
        pat = df["pattern"].iloc[0]
        high = (df["load_set"] >= HIGH_LOAD * df["load_set"].max()).to_numpy()
        q = df[[f"data_quality_{i}" for i in (1, 2, 3)]].mean(axis=1)
        row = dict(planner=m["planner"], rep=int(m["rep"]), pattern=pat, seconds=len(df),
                   slo_fulfillment=df["utility"].mean(), completion=df["completion"].mean(),
                   slo_high=df.loc[high, "utility"].mean(), completion_high=df.loc[high, "completion"].mean(),
                   mean_quality=q.mean(), high_share=high.mean())
        row |= stability(df)
        row |= {f"{k}_high": v for k, v in stability(df, high).items()}
        row |= recovery(df)
        row |= spike(df)
        row |= prediction(df)
        e = df[df["event"].notna()] if "event" in df else df.iloc[:0]
        row |= dict(decision_ms=e["plan_ms"].mean() if len(e) else np.nan,
                    loop_cpu_cores=(df["cpu_s"].iloc[-1] - df["cpu_s"].iloc[0]) / max(len(df) - 1, 1),
                    loop_peak_mb=df["rss_mb"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def paired(runs, metric, ref="dbn"):
    """Each planner against `ref` on the (pattern, rep) pairs: mean difference ref - other, 95% CI, p, Holm."""
    wide = runs.pivot_table(index=["pattern", "rep"], columns="planner", values=metric)
    rng = np.random.default_rng(SEED)
    out = []
    for p in [k for k in PLANNERS if k != ref and k in wide]:
        d = (wide[ref] - wide[p]).dropna().to_numpy()
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        boot = d[idx].mean(axis=1)
        pval = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        out.append(dict(metric=metric, planner=p, pairs=len(d), diff=d.mean(),
                        lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), p=pval))
    out = pd.DataFrame(out)
    order = np.argsort(out["p"].to_numpy())
    m, holm, prev = len(out), np.empty(len(out)), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (m - rank) * out["p"].iloc[i]))
        holm[i] = prev
    out["p_holm"] = holm
    return out


def boot_mean(x, rng):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    b = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fig_fulfillment_stability(runs, path):
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    panels = [("changes_per_h", "slo_fulfillment", "all runs", "configuration changes per hour", [0, 1, 10, 100]),
              ("spike_changes", "spike_slo", "overload spike (600 req/s, 180 s)",
               "configuration changes during the spike", [0, 1, 10])]
    for ax, (xcol, ycol, title, xlabel, ticks) in zip(axes, panels):
        seen, ys = {}, []
        for p, (label, color) in PLANNERS.items():
            r = runs[(runs["planner"] == p) & runs[ycol].notna()]
            if r.empty:
                continue
            xm, xl, xh = boot_mean(r[xcol], rng)
            ym, yl, yh = boot_mean(r[ycol], rng)
            # points that would overlap (static and VPA in the spike) are drawn side by side
            near = sum(1 for (x0, y0) in seen if x0 == xm and abs(y0 - ym) < 0.01)
            seen[(xm, ym)] = 1
            shift = mtransforms.offset_copy(ax.transData, fig=fig, x=11 * near, units="points")
            ax.plot([xl, xh], [ym, ym], "-", color=color, lw=2, alpha=0.5, solid_capstyle="butt", zorder=2,
                    transform=shift)
            ax.plot([xm, xm], [yl, yh], "-", color=color, lw=2, alpha=0.5, solid_capstyle="butt", zorder=2,
                    transform=shift)
            ax.plot([xm], [ym], "o", color=color, ms=7, mew=0, zorder=3, transform=shift)
            ys += [yl, yh]
        ax.set_title(title, fontsize=10, color=MF.INK, loc="left")
        ax.set_xlabel(xlabel, color=MF.MUTED)
        ax.set_xscale("symlog", linthresh=1)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(-0.3, ticks[-1] * 2.2)
        pad = 0.06 * (max(ys) - min(ys))
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    axes[0].set_ylabel("SLO fulfillment", color=MF.MUTED)
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=7, mew=0, label=l)
               for l, c in PLANNERS.values()]
    handles.append(Line2D([], [], color=MF.NEUTRAL, lw=2, alpha=0.55, label="95% interval"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(REPO / "results" / "adaptation_loop" / "main"))
    args = ap.parse_args()
    paths = sorted(glob.glob(str(Path(args.logs) / "live_*_main_*.jsonl")))
    runs = per_run(paths)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT_TABLES / "per_run.csv", index=False)
    print(f"{len(runs)} runs; per planner:", runs.groupby("planner").size().to_dict())

    cols = ["slo_fulfillment", "completion", "mean_quality", "changes_per_h", "reversals_per_h", "core_moves_per_h",
            "recovery_s", "never_recovered", "spike_completion", "spike_slo", "spike_quality", "spike_changes",
            "pred_mae", "pred_bias",
            "decision_ms", "loop_cpu_cores", "loop_peak_mb"]
    summary = runs.groupby("planner")[cols].mean().reindex([p for p in PLANNERS if p in set(runs.planner)])
    summary.to_csv(OUT_TABLES / "summary.csv")
    pd.set_option("display.width", 250)
    print("\n== means over the 15 runs of each planner\n" + summary.round(3).to_string())

    tests = pd.concat([paired(runs, m) for m in ["slo_fulfillment", "completion", "mean_quality", "changes_per_h",
                                                  "reversals_per_h", "core_moves_per_h", "recovery_s",
                                                  "pred_mae"]], ignore_index=True)
    tests.to_csv(OUT_TABLES / "paired_vs_dbn.csv", index=False)
    print("\n== DBN minus each planner, paired on (pattern, repetition), 95% bootstrap interval, Holm p\n"
          + tests.round(4).to_string(index=False))
    print("\nfigure:", fig_fulfillment_stability(runs, OUT_FIGS / "p3_fulfillment_stability.png"))


if __name__ == "__main__":
    main()
