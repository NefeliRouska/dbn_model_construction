"""
report.py -- tables and the figure of the adaptation runs, reported as in
Sedlak et al. (arXiv 2510.06882, Fig. 8): SLO fulfillment over time, mean and
standard deviation over repetitions, and split into high and low load.

Input: JSON-lines logs of loop.py (one per run). The planner is read from the
file name (<mode>_<planner>[_<bins>bins]_<minutes>min_seed<k>...). Runs of the
same planner and the same load program are repetitions.

  table     per planner and load pattern, and per planner and load level:
            SLO fulfillment, completion, share of seconds short, mean quality,
            core-seconds, actions per hour, decision time, and the loop's own
            overhead (CPU cores used, peak memory); mean (sd) over repetitions
  figure    one column per load pattern: the load on top, the SLO fulfillment
            (60 s means) of every planner below, mean line and +-1 sd band

Load level: the load relative to the highest load of the program; high when at
least HIGH_LOAD (0.4, the split of Sedlak et al.).

Usage:
    python adaptation/report.py results/adaptation_loop/sim_*_60min_*.jsonl [--out DIR] [--name NAME]
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
import make_figures as MF  # noqa: E402  (style and colours of the paper figures)

HIGH_LOAD = 0.4
SMOOTH = 60
# one colour per planner, fixed, in the categorical order of the paper figures
PLANNERS = {"dbn": ("DBN", "#2a78d6"), "minrule": ("min rule", "#eb6834"), "static": ("static", "#1baf7a"),
            "vpa": ("VPA", "#eda100"), "tabpfn": ("TabPFN v3.5", "#e87ba4"), "rask": ("RASK", "#008300")}
NAME_RE = re.compile(r"^(?P<mode>sim|live)_(?P<planner>[a-z]+)(?:_\d+bins)?_(?P<minutes>\d+)min_seed(?P<seed>\d+)")


def load_runs(paths):
    runs = []
    for p in map(Path, paths):
        m = NAME_RE.match(p.name)
        if m is None:
            print(f"skipped (name not understood): {p.name}")
            continue
        df = pd.read_json(p, lines=True)
        df["planner"], df["run"] = m["planner"], p.stem
        runs.append(df)
    df = pd.concat(runs, ignore_index=True)
    df["cores"] = df[["cores_1", "cores_2", "cores_3"]].sum(axis=1)
    df["quality"] = df[["data_quality_1", "data_quality_2", "data_quality_3"]].mean(axis=1)
    df["short"] = df["completion"] < 1.0
    df["load_level"] = np.where(df["load_set"] >= HIGH_LOAD * df["load_set"].max(), "high", "low")
    if "act" not in df:
        df["act"] = np.nan
    df["acted"] = df["act"].fillna(0).astype(bool)
    return df


def per_run(df, by):
    def one(g):
        dur = len(g)
        out = dict(slo_fulfillment=g["utility"].mean(), completion=g["completion"].mean(),
                   short_share=g["short"].mean(), mean_quality=g["quality"].mean(),
                   core_seconds=g["cores"].sum(), actions_per_h=g["acted"].sum() / dur * 3600,
                   decision_ms=g["plan_ms"].mean() if "plan_ms" in g else np.nan)
        if "cpu_s" in g:
            out["loop_cpu_cores"] = (g["cpu_s"].iloc[-1] - g["cpu_s"].iloc[0]) / max(dur - 1, 1)
            out["loop_peak_mb"] = g["rss_mb"].max()
        return pd.Series(out)
    return df.groupby(["planner", "run"] + by).apply(one, include_groups=False).reset_index()


def table(df, by):
    """mean (sd) over repetitions, one row per planner and `by`."""
    r = per_run(df, by)
    cols = [c for c in r.columns if c not in ["planner", "run"] + by]
    g = r.groupby(["planner"] + by)[cols]
    mean, sd, n = g.mean(), g.std(), g.size()
    out = mean.round(3).astype(str) + " (" + sd.round(3).fillna(0).astype(str) + ")"
    out.insert(0, "runs", n)
    order = [p for p in PLANNERS if p in out.index.get_level_values(0)]
    return out.reindex(order, level=0)


def figure(df, path):
    patterns = list(dict.fromkeys(df["pattern"]))
    fig, axes = plt.subplots(2, len(patterns), figsize=(4.2 * len(patterns), 5.2), sharex="col",
                             gridspec_kw=dict(height_ratios=[1, 2.4]), squeeze=False)
    for j, pat in enumerate(patterns):
        d = df[df["pattern"] == pat]
        top, ax = axes[0, j], axes[1, j]
        load = d.groupby("t")["load_set"].first()
        top.plot((load.index - load.index.min()) / 60, load.to_numpy(), color=MF.NEUTRAL, lw=1.5)
        top.set_title(pat.replace("_", "-"), fontsize=10, color=MF.INK)
        for name, (label, color) in PLANNERS.items():
            p = d[d["planner"] == name]
            if p.empty:
                continue
            curves = []
            for _, g in p.groupby("run"):
                s = g.set_index("t")["utility"]
                curves.append(s.groupby((s.index - s.index.min()) // SMOOTH).mean())
            c = pd.concat(curves, axis=1)
            x = (c.index.to_numpy() + 0.5) * SMOOTH / 60
            mu, sd = c.mean(axis=1).to_numpy(), c.std(axis=1).fillna(0).to_numpy()
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)
            ax.plot(x, mu, color=color, lw=2, label=label)
        ax.set_xlabel("minutes", color=MF.MUTED)
        if j == 0:
            top.set_ylabel("load (req/s)", color=MF.MUTED)
            ax.set_ylabel("SLO fulfillment", color=MF.MUTED)
    lo = min(a.get_ylim()[0] for a in axes[1])
    for a in axes[1]:
        a.set_ylim(max(0, lo), 1.0)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "adaptation_loop" / "report"))
    ap.add_argument("--name", default="report")
    args = ap.parse_args()
    df = load_runs(args.logs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    for label, by in (("by pattern", ["pattern"]), ("by load level", ["load_level"]), ("overall", [])):
        t = table(df, by)
        t.to_csv(out / f"{args.name}_{label.replace(' ', '_')}.csv")
        print(f"\n== {label}\n{t.to_string()}")
    print("\nfigure:", figure(df, out / f"{args.name}_fulfillment_over_time.png"))


if __name__ == "__main__":
    main()
