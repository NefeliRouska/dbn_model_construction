"""
load_patterns.py -- two load patterns the DBN has never seen, in the style of
the bursty and diurnal patterns of Sedlak et al. (arXiv 2510.06882, from
Google cluster traces), generated here with a fixed seed so that every planner
sees the same load.

Both give one value per second for one hour, between LOW and HIGH req/s,
inside the range of the training data (100-600). Unlike the training patterns,
which change the load at most once a minute, they change it every second.

  diurnal   one smooth "day" per hour: from LOW_DIURNAL at the start up to HIGH
            at the middle and back, plus slow noise (an AR(1) process with a
            correlation time of 2 min) and a little per-second jitter
  bursty    a low base around BASE_BURSTY that wanders slowly (AR(1), 5 min),
            with BUMPS soft bumps at random times (Gaussian in time, 40-120 s
            wide, 30-100 req/s high) and one surge (3-5 min wide, up to
            ~HIGH), as in the bursty trace of Sedlak et al.: no square jumps,
            but a load that can rise by ~300 req/s within ten minutes

load(name, t, seed) is the load at second t of the pattern; the whole hour is
generated once per (name, seed) and cached.
"""
from functools import lru_cache

import numpy as np

SECONDS = 3600
LOW, HIGH = 100, 500
LOW_DIURNAL = 230
BASE_BURSTY, BUMPS = 150, 4
PATTERNS = ("diurnal", "bursty")


def _slow_noise(rng, n, sd, tau=60.0):
    """AR(1) noise with standard deviation sd and correlation time tau seconds."""
    a = np.exp(-1.0 / tau)
    e = rng.standard_normal(n) * sd * np.sqrt(1 - a * a)
    x = np.empty(n)
    x[0] = rng.standard_normal() * sd
    for t in range(1, n):
        x[t] = a * x[t - 1] + e[t]
    return x


@lru_cache(maxsize=None)
def trace(name, seed=783122):
    rng = np.random.default_rng([seed, PATTERNS.index(name)])
    t = np.arange(SECONDS)
    if name == "diurnal":
        shape = LOW_DIURNAL + (HIGH - LOW_DIURNAL) * (1 - np.cos(2 * np.pi * t / SECONDS)) / 2
        x = shape + _slow_noise(rng, SECONDS, 12.0, tau=120.0) + rng.normal(0, 2, SECONDS)
    elif name == "bursty":
        x = BASE_BURSTY + _slow_noise(rng, SECONDS, 25.0, tau=300.0) + rng.normal(0, 2, SECONDS)
        for _ in range(BUMPS):
            x = x + rng.uniform(30, 100) * np.exp(-0.5 * ((t - rng.uniform(0, SECONDS)) / rng.uniform(40, 120)) ** 2)
        centre = rng.uniform(0.2, 0.8) * SECONDS
        x = x + rng.uniform(270, 330) * np.exp(-0.5 * ((t - centre) / rng.uniform(180, 300)) ** 2)
    else:
        raise ValueError(f"unknown pattern {name}")
    return np.clip(np.round(x), LOW, HIGH).astype(int)


def load(name, t, seed=783122):
    return int(trace(name, seed)[min(int(t), SECONDS - 1)])
