"""
explain_p3.py -- can a planner's model explain the configurations it applies?
Checked against what the testbed did, on every configuration change of the 90
runs of the main experiment (all planners).

For every change, the window from PRED_FROM to PRED_TO s after it was sent
(shorter if the next change came earlier, at least PRED_MIN s) gives the
observed mean load and throughputs of the three services. The OBSERVED
bottleneck is the first service whose throughput is below LIMITING x what it
receives (the load for service 1, the throughput of the service before it
otherwise), or "none" when the chain passes the load.

Two models explain the same configuration under the same (observed) load:
  DBN        the planner's settled chain, service by service:
             P(tp_1 | load, capacity_1), P(tp_2 | tp_1, capacity_2), ...,
             E[tp_i] from each stage; the predicted bottleneck is the first
             service with E[tp_i] < LIMITING x E[what it receives]
  min rule   tp_i = min(what it receives, capacity_i); the first service
             whose capacity is below LIMITING x what it receives
TabPFN predicts throughput_3 only (no per-service account) and RASK's
per-service curves treat the services as independent; neither names the
service that limits the chain, so they are not scored here.

Reported: agreement with the observed bottleneck (accuracy, and per observed
class), the error of each model's per-service throughputs, and the overload
spike as a case study (what each model says the configuration will do there).

Usage:
    python adaptation/explain_p3.py [--logs results/adaptation_loop/main]
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "dbn"))
sys.path.insert(0, str(HERE))
import build_model as B  # noqa: E402
import planner as PL  # noqa: E402
from analysis_p3 import NAME_RE, OUT_TABLES, PRED_FROM, PRED_MIN, PRED_TO, SETTINGS  # noqa: E402
from check_monitor import DEFAULT_MODEL  # noqa: E402

LIMITING = 0.97
CLASSES = ["none", "1", "2", "3"]


class Explainers:
    def __init__(self):
        self.model = B.Model.load(str(DEFAULT_MODEL))
        chain = PL.load_chain(PL.chain_path(DEFAULT_MODEL, PL.PLANNER_BINS))
        self.tables, self.edges, self.value = chain["tables"], chain["edges"], chain["bin_value"]
        self.K = len(self.edges) + 1

    def capacities(self, cfg):
        c = B.add_capacity(pd.DataFrame([{k: float(cfg[k]) for k in SETTINGS}]), self.model.capacity)
        return [float(c[f"capacity_{i}"].iloc[0]) for i in (1, 2, 3)]

    def code(self, x):
        return np.searchsorted(self.edges, np.atleast_1d(np.asarray(x, dtype=float)), side="right")

    def dbn(self, load, caps):
        """E[throughput_i] of every service, through the settled chain."""
        P, flows = None, []
        for i, cpt in enumerate(self.tables):
            cap = self.code(caps[i])
            if P is None:
                up = self.code(load)
                P = cpt.predict([up, cap, np.minimum(up, cap)], 1)[0]
            else:
                rows = np.array([cpt.predict([np.array([s]), cap, np.minimum(np.array([s]), cap)], 1)[0]
                                 for s in range(self.K)])
                P = P @ rows
            flows.append(float(P @ self.value))
        return flows

    @staticmethod
    def min_rule(load, caps):
        flows, f = [], load
        for c in caps:
            f = min(f, c)
            flows.append(f)
        return flows


def bottleneck(load, flows, receives=None):
    inputs = [load] + list(flows[:-1]) if receives is None else receives
    for i, (tp, inp) in enumerate(zip(flows, inputs)):
        if tp < LIMITING * inp:
            return str(i + 1)
    return "none"


def min_rule_bottleneck(load, caps):
    f = load
    for i, c in enumerate(caps):
        if c < LIMITING * f:
            return str(i + 1)
        f = min(f, c)
    return "none"


def windows(paths):
    for path in paths:
        m = NAME_RE.search(Path(path).name)
        df = pd.read_json(path, lines=True)
        if "act" not in df:
            continue
        acts = np.flatnonzero(df["act"].fillna(0).astype(bool).to_numpy())
        for k, t in enumerate(acts):
            nxt = acts[k + 1] if k + 1 < len(acts) else len(df)
            lo, hi = t + PRED_FROM, min(t + PRED_TO, nxt)
            if hi - lo < PRED_MIN:
                continue
            w = df.iloc[lo:hi]
            yield dict(planner=m["planner"], rep=int(m["rep"]), pattern=df["pattern"].iloc[0], t=int(t),
                       cfg=df["new_config"].iloc[t], load=w["rps"].mean(),
                       tp=[w[f"throughput_{i}"].mean() for i in (1, 2, 3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(REPO / "results" / "adaptation_loop" / "main"))
    args = ap.parse_args()
    ex = Explainers()
    rows = []
    for w in windows(sorted(glob.glob(str(Path(args.logs) / "live_*_main_*.jsonl")))):
        caps = ex.capacities(w["cfg"])
        dbn, mr = ex.dbn(w["load"], caps), ex.min_rule(w["load"], caps)
        obs = w["tp"]
        rows.append(dict(planner=w["planner"], rep=w["rep"], pattern=w["pattern"], t=w["t"], load=w["load"],
                         observed=bottleneck(w["load"], obs), dbn=bottleneck(w["load"], dbn),
                         min_rule=min_rule_bottleneck(w["load"], caps),
                         **{f"obs_tp{i + 1}": obs[i] for i in range(3)}, **{f"dbn_tp{i + 1}": dbn[i] for i in range(3)},
                         **{f"mr_tp{i + 1}": mr[i] for i in range(3)}, **{f"cap{i + 1}": caps[i] for i in range(3)}))
    r = pd.DataFrame(rows)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT_TABLES / "explanations.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{len(r)} configuration changes with a window of >= {PRED_MIN} s (all planners)")
    print("observed bottleneck:", r["observed"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict())

    acc = []
    for name in ("dbn", "min_rule"):
        a = dict(explainer=name, accuracy=(r[name] == r["observed"]).mean())
        for c in CLASSES:
            sub = r[r["observed"] == c]
            a[f"recall_{c}"] = (sub[name] == c).mean() if len(sub) else np.nan
        pre = "dbn" if name == "dbn" else "mr"
        for i in (1, 2, 3):
            a[f"mae_tp{i}"] = (r[f"{pre}_tp{i}"] - r[f"obs_tp{i}"]).abs().mean()
        acc.append(a)
    acc = pd.DataFrame(acc).set_index("explainer")
    acc.to_csv(OUT_TABLES / "explanation_accuracy.csv")
    print("\n== agreement with the observed bottleneck, and per-service throughput error (req/s)\n"
          + acc.round(3).to_string())
    for name in ("dbn", "min_rule"):
        print(f"\nconfusion ({name}: rows observed, columns explained)\n"
              + pd.crosstab(r["observed"], r[name]).reindex(index=CLASSES, columns=CLASSES).fillna(0)
              .astype(int).to_string())

    spike = r[r["load"] > 550]
    print(f"\n== overload spike: {len(spike)} changes; mean over them (req/s)")
    cols = ["load", "cap1", "cap2", "cap3", "obs_tp1", "obs_tp2", "obs_tp3", "dbn_tp1", "dbn_tp2", "dbn_tp3",
            "mr_tp1", "mr_tp2", "mr_tp3"]
    print(spike[cols].mean().round(0).to_string())
    print("explained bottleneck in the spike:", {n: spike[n].value_counts().to_dict() for n in ("observed", "dbn", "min_rule")})


if __name__ == "__main__":
    main()
