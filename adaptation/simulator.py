"""
simulator.py -- a stand-in for the three-service chain, so the whole loop can
be run and debugged offline, faster than real time.

It is NOT a model of the testbed to draw conclusions from; it only has to
behave plausibly enough to exercise every code path of the loop:

  capacity_i   cores_i x r(q_i), with r learned from the data (the model's
               capacity function), times a little noise each second
  service 1    emits min(load, capacity_1); the excess is not queued
               (buffer_size_1 is the load setpoint in the real data too)
  service 2,3  queue what arrives, emit min(queue + arrivals, capacity_i)
  settings     take effect APPLY_DELAY seconds after they are sent
  latency      the processing time of one item, 1000 / r(q) ms with noise, rounded
               to whole ms like the real metric (on the testbed 1000 / latency
               x cores matches the capacity: 22-23 ms at quality 700, where r is
               45 per core); RASK learns its capacities from it

It offers the same two calls as the live system: read() -> (timestamp, row in
the Monitor's columns) and apply(config) / set_load(rps).
"""
import numpy as np

APPLY_DELAY = 5
NOISE = 0.03


class SimulatedChain:
    def __init__(self, capacity, config, seed=0, t0=0.0):
        self.rate = capacity["rate"]
        self.rng = np.random.default_rng(seed)
        self.t = float(t0)
        self.config = dict(config)
        self.pending = []                       # (time it takes effect, config)
        self.load = 0.0
        self.queue = {2: 0.0, 3: 0.0}
        self.n_applied = 0

    # ---- the interface the loop uses ----
    def set_load(self, rps):
        self.load = float(rps)

    def apply(self, config):
        self.pending.append((self.t + APPLY_DELAY, dict(config)))
        self.n_applied += 1

    def read(self):
        """Advance one second and return (timestamp, row)."""
        self.t += 1.0
        while self.pending and self.pending[0][0] <= self.t:
            self.config = self.pending.pop(0)[1]
        c = self.config
        cap = {i: c[f"cores_{i}"] * self.rate[float(c[f"data_quality_{i}"])]
               * max(0.0, 1 + NOISE * self.rng.standard_normal()) for i in (1, 2, 3)}
        tp = {1: min(self.load, cap[1])}
        for i in (2, 3):
            avail = self.queue[i] + tp[i - 1]
            tp[i] = min(avail, cap[i])
            self.queue[i] = avail - tp[i]
        row = {"rps": self.load}
        for i in (1, 2, 3):
            row[f"throughput_{i}"] = round(tp[i])
            row[f"cores_{i}"] = c[f"cores_{i}"]
            row[f"data_quality_{i}"] = c[f"data_quality_{i}"]
            per_item = 1000.0 / self.rate[float(c[f"data_quality_{i}"])]
            row[f"avg_p_latency_{i}"] = max(1, round(per_item * max(0.5, 1 + NOISE * self.rng.standard_normal())))
        row["buffer_size_2"], row["buffer_size_3"] = round(self.queue[2]), round(self.queue[3])
        return self.t, row
