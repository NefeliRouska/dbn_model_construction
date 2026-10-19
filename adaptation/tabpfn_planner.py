"""
tabpfn_planner.py -- TabPFN v3.5 as the planner's outcome model, in place of
the DBN's settled chain. Everything else (candidates, utility, decision rule,
the loop) is the DBN planner's.

Same data and inputs as the DBN planner's chain: the settled 30 s windows of
planner.settled_windows (no draining backlog), inputs (load, capacity_1..3),
target throughput_3. TabPFN is conditioned on CONTEXT windows drawn at random
(the context size of the Part 2 comparison) and asked for the quantiles
QUANTILES of throughput_3. The expected completion of a candidate is the mean
over those quantiles of min(quantile / load, 1).

A forward pass for 12,320 candidates every decision would be too slow for a
1 s loop, so the expected completion is computed once, offline (GPU), for every
candidate at every load of LOADS, and saved; at run time the planner
interpolates linearly between the two nearest loads.

Build (on the GPU workstation; needs the TabPFN v3.5 licence):
    python adaptation/tabpfn_planner.py --build
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "dbn"))
sys.path.insert(0, str(HERE))
import build_model as B  # noqa: E402
import data as D  # noqa: E402
import planner as PL  # noqa: E402
from check_monitor import DEFAULT_MODEL  # noqa: E402

LOADS = np.arange(100, 601, 5)
CONTEXT, SEED, VERSION = 10_000, 0, "v3.5"
QUANTILES = list(np.round(np.arange(0.05, 0.96, 0.05), 2))
INPUTS = [D.LOAD_COL, "capacity_1", "capacity_2", "capacity_3"]
BATCH = 50_000


def grid_path(model_path=DEFAULT_MODEL):
    return Path(model_path).with_name(Path(model_path).stem + f"_tabpfn_{VERSION}_planner_grid.npz")


def build(model_path=DEFAULT_MODEL):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    model = B.Model.load(str(model_path))
    frames = [model.load_frame(REPO / "data" / f) for f in model.meta["files"]]
    df, drain = PL.settled_windows(frames)
    df = df[~drain].reset_index(drop=True)
    ctx = df.sample(n=min(CONTEXT, len(df)), random_state=SEED)
    reg = TabPFNRegressor.create_default_for_version(ModelVersion(VERSION))
    reg.fit(ctx[INPUTS].to_numpy(dtype=float), ctx["throughput_3"].to_numpy(dtype=float))

    cands = B.add_capacity(PL.candidates(), model.capacity)
    caps = cands[["capacity_1", "capacity_2", "capacity_3"]].to_numpy(dtype=float)
    completion = np.empty((len(LOADS), len(cands)), dtype=np.float32)
    t0 = time.time()
    X = np.column_stack([np.repeat(LOADS, len(cands)), np.tile(caps, (len(LOADS), 1))]).astype(float)
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), BATCH):
        q = np.asarray(reg.predict(X[s:s + BATCH], output_type="quantiles", quantiles=QUANTILES))  # (n_q, n)
        load = X[s:s + BATCH, 0]
        out[s:s + BATCH] = np.minimum(q / load, 1.0).mean(axis=0)
        print(f"  {min(s + BATCH, len(X))}/{len(X)} predictions, {time.time() - t0:.0f} s", flush=True)
    completion[:] = out.reshape(len(LOADS), len(cands))
    path = grid_path(model_path)
    np.savez_compressed(path, loads=LOADS, completion=completion,
                        settings=cands[PL.SETTINGS].to_numpy(dtype=np.int16),
                        meta=np.array([f"TabPFN {VERSION}, context {len(ctx)} of {len(df)} settled 30 s windows, "
                                       f"seed {SEED}, quantiles {QUANTILES[0]}..{QUANTILES[-1]}, model {Path(model_path).name}"]))
    print(f"wrote {path} ({time.time() - t0:.0f} s)")
    return path


class TabPFNPlanner(PL.Planner):
    """The DBN planner with the expected completion looked up in the TabPFN grid."""

    def __init__(self, model, grid=None, delta=PL.DELTA):
        super().__init__(model, None, delta)
        g = np.load(grid or grid_path(), allow_pickle=False)
        if not np.array_equal(g["settings"], self.cands[PL.SETTINGS].to_numpy()):
            raise ValueError("the TabPFN grid was built for other candidates")
        self.loads, self.grid, self.meta = g["loads"].astype(float), g["completion"], str(g["meta"][0])
        self.index = {tuple(r): k for k, r in enumerate(g["settings"].tolist())}

    def _completion(self, load):
        load = float(np.clip(load, self.loads[0], self.loads[-1]))
        j = int(np.clip(np.searchsorted(self.loads, load) - 1, 0, len(self.loads) - 2))
        w = (load - self.loads[j]) / (self.loads[j + 1] - self.loads[j])
        return (1 - w) * self.grid[j] + w * self.grid[j + 1]

    def score(self, load, configs=None, phi_quality=None):
        configs = self.cands if configs is None else configs
        if phi_quality is None:
            phi_quality = np.minimum(configs[[f"data_quality_{i}" for i in (1, 2, 3)]].to_numpy()
                                     / PL.QUALITY_TARGET, 1.0).sum(axis=1)
        allc = self._completion(load)
        if configs is self.cands:
            completion = allc.astype(float)
        else:
            rows = [self.index[tuple(int(v) for v in r)] for r in configs[PL.SETTINGS].to_numpy()]
            completion = allc[rows].astype(float)
        utility = (PL.W_COMPLETION * completion + PL.W_QUALITY * phi_quality) / (PL.W_COMPLETION + 3 * PL.W_QUALITY)
        return utility, completion, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    if args.build:
        build(args.model)
