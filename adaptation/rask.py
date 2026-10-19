"""
rask.py -- RASK (Sedlak et al., arXiv 2510.06882), ported from its public code
(github.com/borissedlak/elastic-workbench, branch tsc-submission:
agent/RASK_Agent.py, agent/components/RASK.py, PolicySolverRASK.py,
SLORegistry.py) to the three-service chain, as faithfully as the chain allows.

What RASK does every cycle (EVERY = 10 s, its EVALUATION_CYCLE_DELAY):
  1. explore: in the first MAX_EXPLORE cycles (20, the paper's choice) it
     applies a random configuration (services in random order; cores drawn
     from what is left of the budget, as in call_all_ES_randomly; a random
     quality);
  2. otherwise it refits, on everything it has observed in this run (rows
     within COOLDOWN_S of a change dropped, as filter_rows_during_cooldown),
     one degree-2 polynomial regression per service:
         max_tp_i = 1000 / avg_p_latency_i x cores_i   ~   (cores_i, data_quality_i)
     (its definition for the QR service, which the chain's services are);
     degree 2 is the code's default (train_rask_models), degree 4 the best
     degree for the QR service in the paper's own regression study (E2),
  3. and maximises with SLSQP the sum over services of the per-service SLO
     fulfillment, with c1 + c2 + c3 = 8 (equality, as in the code), starting
     from its last assignment. Per service (its SLO config for QR):
         completion_i = max_tp_i / rps_i            target 1,   weight 1.0
         data_quality_i                             target 800, weight 0.5
         SLO-F_i = sum_s weight_s x smoothstep(value_s / target_s) / sum weights,
     0 if the predicted throughput is below 1 (its rule). The code uses
     smoothstep(x) = 3x^2 - 2x^3 of the clipped ratio; the paper writes
     min(x, 1). RASK decides with its code's version; the evaluation of every
     planner uses the paper's (loop.utility).

What the chain changes:
  - rps_i, the arrivals of service i, is measured: the load for service 1, the
    throughput of service i-1 otherwise. RASK treats the services as
    independent, as in the paper ("we assumed completely independent
    services"); it does not know that more cores at service 3 are useless when
    service 1 is the bottleneck. That assumption is what the chain tests.
  - cores are whole numbers on our services (rounded by largest remainder to
    sum 8, at least 1 each) and quality is rounded to steps of 100, the levels
    of our capacity function and training data.
  - its Gaussian exploration noise eta is 0 (the paper's choice).
  - one episode plays several load patterns; RASK explores once, at the start
    of the episode, and keeps everything it has learned across patterns
    (in the paper each pattern was a separate run). This favours RASK.
"""
import random
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

EVERY = 10
MAX_EXPLORE = 20
COOLDOWN_S = 2              # its ES cooldown, 2000 ms
TOTAL_CORES = 8
Q_MIN, Q_MAX, Q_TARGET, W_Q, W_C = 100, 1000, 800.0, 0.5, 1.0
SETTINGS = [f"{p}_{i}" for p in ("cores", "data_quality") for i in (1, 2, 3)]


def smoothstep(x):
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def round_cores(c, total=TOTAL_CORES):
    """Whole cores, at least 1 each, summing to `total` (largest remainder)."""
    c = np.maximum(np.asarray(c, dtype=float), 1.0)
    c = c / c.sum() * total
    base = np.maximum(np.floor(c).astype(int), 1)
    while base.sum() > total:
        base[np.argmax(base)] -= 1
    for i in np.argsort(-(c - np.floor(c))):
        if base.sum() >= total:
            break
        base[i] += 1
    return base


class RaskPlanner:
    every = EVERY

    def __init__(self, max_explore=MAX_EXPLORE, seed=0, degree=2):
        self.max_explore, self.rng, self.degree = max_explore, random.Random(seed), degree
        self.explored, self.history, self.last_change, self.last_x = 0, [], -np.inf, None

    # ---- what it observes, every second ----
    def observe(self, t, row):
        if t - self.last_change > COOLDOWN_S:
            self.history.append({k: row[k] for k in row if k.startswith(("cores_", "data_quality_", "avg_p_latency_"))})

    # ---- its models ----
    def _fit(self):
        models = []
        for i in (1, 2, 3):
            rows = [r for r in self.history if r[f"avg_p_latency_{i}"] > 0]
            if len(rows) < 3:
                return None
            X = np.array([[r[f"cores_{i}"], r[f"data_quality_{i}"]] for r in rows], dtype=float)
            y = np.array([1000.0 / r[f"avg_p_latency_{i}"] * r[f"cores_{i}"] for r in rows])
            poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            models.append((poly, LinearRegression().fit(poly.fit_transform(X), y)))
        return models

    @staticmethod
    def _slo_f(models, x, rps):
        total = 0.0
        for i, (poly, reg) in enumerate(models):
            q, c = x[2 * i], x[2 * i + 1]
            max_tp = float(reg.predict(poly.transform([[c, q]]))[0])
            tp = max_tp if max_tp > 1 else 0.0
            if tp < 1.0:
                continue
            completion = tp / rps[i] if rps[i] > 0 else 1.0
            total += (W_C * smoothstep(completion) + W_Q * smoothstep(q / Q_TARGET)) / (W_C + W_Q)
        return total

    # ---- one cycle ----
    def decide(self, load, current, row, observed=None):
        nan = float("nan")
        if self.explored < self.max_explore:
            self.explored += 1
            new, assigned = dict(current), 0
            order = [1, 2, 3]
            self.rng.shuffle(order)
            for k, i in enumerate(order):
                max_cores = max(1, TOTAL_CORES - assigned + (k - 2))
                new[f"cores_{i}"] = self.rng.randint(1, max_cores)
                new[f"data_quality_{i}"] = int(round(self.rng.randint(Q_MIN, Q_MAX), -2))
                assigned += new[f"cores_{i}"]
            event = "explore"
        else:
            models = self._fit()
            if models is None:
                return dict(act=False, config=current, completion=nan, utility=nan, completion_current=nan,
                            utility_current=nan, probs=None, rask="no data")
            rps = [load, row["throughput_1"], row["throughput_2"]]
            x0 = self.last_x if self.last_x is not None else \
                [v for i in (1, 2, 3) for v in ((Q_MIN + Q_MAX) / 2, TOTAL_CORES / 3)]
            bounds = [(Q_MIN, Q_MAX), (1, TOTAL_CORES)] * 3
            cons = [{"type": "eq", "fun": lambda x: TOTAL_CORES - (x[1] + x[3] + x[5])}]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(lambda x: -self._slo_f(models, x, rps), x0, method="SLSQP",
                               bounds=bounds, constraints=cons, options={"maxiter": 150})
            if not res.success:
                return dict(act=False, config=current, completion=nan, utility=nan, completion_current=nan,
                            utility_current=nan, probs=None, rask=f"solver: {res.message}")
            self.last_x = list(res.x)
            cores = round_cores([res.x[1], res.x[3], res.x[5]])
            new = {f"cores_{i}": int(cores[i - 1]) for i in (1, 2, 3)}
            new |= {f"data_quality_{i}": int(np.clip(round(res.x[2 * (i - 1)], -2), Q_MIN, Q_MAX)) for i in (1, 2, 3)}
            event = "optimise"
        act = new != {c: current[c] for c in SETTINGS}
        return dict(act=act, config=new, completion=nan, utility=nan, completion_current=nan,
                    utility_current=nan, probs=None, rask=event)

    def applied(self, t):
        self.last_change = t
