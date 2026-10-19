"""
baselines.py -- the planners the DBN is compared with that need no learning
beyond the capacity function. Each plugs into loop.run like Planner.decide:
plan(load, current, row=None) -> {act, config, completion, utility, ...}.

  MinRulePlanner  same candidates, utility and decision rule as the DBN
                  planner, but the outcome of a candidate is the deterministic
                  bottleneck formula tp3 = min(load, capacity_1, capacity_2,
                  capacity_3), with the capacity function of the final model.
                  Does the DBN add anything beyond the formula?
  static_best     one fixed configuration, never changed: the candidate with
                  the highest mean utility over the loads of the whole load
                  program under the min rule. It knows the load program in
                  advance (best in hindsight), but cannot react to it.
  VpaPlanner      the VPA-style baseline of Sedlak et al. (arXiv 2510.06882):
                  cores only, data quality stays at its start value, which is
                  the configuration of static_best (with any fixed quality the
                  result would mostly reflect that choice; this one gives the
                  VPA the best quality setting it could have known in advance). Each
                  service keeps between 5% and 15% spare capacity, judged on
                  its own arrivals (the load for service 1, the throughput of
                  the service before it otherwise) against
                  capacity_i = cores_i x r(q_i); one core at a time per
                  service, as their +-0.25 cores in whole cores. Like a VPA it
                  sees each service alone, not the chain.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dbn"))
sys.path.insert(0, str(HERE))
import build_model as B  # noqa: E402
import planner as PL  # noqa: E402

SLACK_LOW, SLACK_HIGH = 0.05, 0.15


class MinRulePlanner(PL.Planner):
    def __init__(self, model, delta=PL.DELTA):
        super().__init__(model, None, delta)

    def score(self, load, configs=None, phi_quality=None):
        configs = self.cands if configs is None else configs
        if phi_quality is None:
            phi_quality = np.minimum(configs[[f"data_quality_{i}" for i in (1, 2, 3)]].to_numpy()
                                     / PL.QUALITY_TARGET, 1.0).sum(axis=1)
        cap = configs[[f"capacity_{i}" for i in (1, 2, 3)]].to_numpy(dtype=float).min(axis=1)
        completion = np.minimum(cap / max(float(load), 1e-9), 1.0)
        utility = (PL.W_COMPLETION * completion + PL.W_QUALITY * phi_quality) / (PL.W_COMPLETION + 3 * PL.W_QUALITY)
        return utility, completion, None

    def decide(self, load, current, row=None, observed=None):
        d = super().decide(load, current, observed=observed)
        d["probs"] = None
        return d


def static_best(model, loads):
    """The candidate with the highest mean min-rule utility over `loads` (one value per second)."""
    p = MinRulePlanner(model)
    values, counts = np.unique(np.asarray(loads, dtype=float), return_counts=True)
    U = sum(c * p.score(v)[0] for v, c in zip(values, counts)) / counts.sum()
    best = int(np.argmax(U))
    return {c: int(v) for c, v in zip(PL.SETTINGS, p.cands[PL.SETTINGS].to_numpy()[best])}, float(U[best])


class VpaPlanner:
    def __init__(self, model, total=PL.TOTAL_CORES, low=SLACK_LOW, high=SLACK_HIGH):
        self.model, self.total, self.low, self.high = model, total, low, high

    def _capacity(self, config):
        return B.add_capacity(pd.DataFrame([config]), self.model.capacity)[[f"capacity_{i}" for i in (1, 2, 3)]] \
            .to_numpy()[0]

    def decide(self, load, current, row, observed=None):
        arrivals = np.array([load, row["throughput_1"], row["throughput_2"]], dtype=float)
        new = dict(current)
        cap = self._capacity(new)
        slack = 1 - arrivals / np.maximum(cap, 1e-9)
        # scale down first (frees cores), then up where spare capacity is short
        for i in np.argsort(-slack):
            c = f"cores_{i + 1}"
            if slack[i] > self.high and new[c] > 1:
                trial = dict(new, **{c: new[c] - 1})
                if 1 - arrivals[i] / self._capacity(trial)[i] >= self.low:
                    new = trial
        cap = self._capacity(new)
        slack = 1 - arrivals / np.maximum(cap, 1e-9)
        for i in np.argsort(slack):
            c = f"cores_{i + 1}"
            if slack[i] < self.low and sum(new[f"cores_{k}"] for k in (1, 2, 3)) < self.total:
                new = dict(new, **{c: new[c] + 1})
        act = new != current
        nan = float("nan")
        return dict(act=act, config=new, completion=nan, utility=nan, completion_current=nan,
                    utility_current=nan, probs=None)
