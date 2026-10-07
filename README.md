# DBN model building

Dynamic Bayesian Network modelling of a three-service processing chain
(throughput, latency, cores, data quality, buffer size, request rate) —
research line of Nefeli Rouska.

Split out of [NefeliRouska/elastic-workbench](https://github.com/NefeliRouska/elastic-workbench),
a fork of Boris Sedlak's [elastic-workbench](https://github.com/borissedlak/elastic-workbench).
The workbench itself (services, agents, docker setup) stays there; this repo
holds the modelling code, its results, and its git history.

## Layout

| Path | What it is |
|---|---|
| `data_collection/experiment_orchestrator.py` | Drives the workbench: sets cores / data quality / **requests per second** per run and dumps Prometheus. Defines the workload patterns. Runs only inside a live elastic-workbench checkout. |
| `data_prep/convert_prom_dump.py` | Raw long-format Prometheus dump → wide table (`dbn_wide_*.csv`). |
| `data_prep/add_rps_feature.py` | Rebuilds the `rps` column from the orchestrator's workload pattern (the dumps do not record it). Constants must match the orchestrator. |
| `dbn/dbn_structure.py` | 2-slice DBN structure learning and scores (BIC / AIC). |
| `dbn/dbn_sweep.py` | Main pipeline: feature selection, discretisation, DBN fit, evaluation and baselines over the hyperparameter grid. Imports `dbn_structure`. |
| `dbn/memory_ablation.py` | Memory ablation (lags, control flag, velocity features) on top of `dbn_sweep`. |
| `dbn/joint_dbn.py` | Single joint DBN for all three throughput targets (not updated since April 2026). |
| `analysis/` | Markov-order analysis, Markov test, control-variable check, statistical significance, LSTM baseline, sweep visualisation. |
| `results/` | Outputs: `sweeps/` (result tables), `models/` (saved top models), `logs/`, `figures/`, `memory_ablation/`. |
| `archive/` | Superseded script versions, early static-BN work, old runs. See `archive/README.md`. |
| `data/` | Not tracked (≈13 GB). Raw dumps and the wide tables derived from them. |

## Setup

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Data

| Dump | Workload pattern | `--pattern` for `add_rps_feature.py` |
|---|---|---|
| `data/prom_dump_all_20260313_192652_seed783122.csv` | periodic | `periodic` |
| `data/prom_dump_all_20260316_105011_seed783122.csv` | once in a lifetime | `oial` |
| `data/prom_dump_all_20260319_094316_seed783122.csv` | unpredictable | `unpredictable` |

## Pipeline

    # 1. raw dump -> wide table (written next to the input, in data/)
    python data_prep/convert_prom_dump.py --csv data/prom_dump_all_<stem>.csv

    # 2. add the requests-per-second column
    python data_prep/add_rps_feature.py --csv data/dbn_wide_<stem>.csv --pattern <pattern> \
        --out data/dbn_wide_<stem>_with_rps.csv

    # 3. hyperparameter sweep
    python dbn/dbn_sweep.py --csv data/dbn_wide_<stem>_with_rps.csv \
        | tee results/logs/dbn_<stem>_with_rps_$(date +%Y%m%d_%H%M%S).log

    # 4. memory ablation for one fixed configuration
    python dbn/memory_ablation.py --csv data/dbn_wide_<stem>_with_rps.csv

`analysis/markov_test.py`, `analysis/lstm_baseline.py` and `dbn/joint_dbn.py`
still hard-code their input as `share/metrics/dbn_wide_20260310_160246_seed161312.csv`
(an earlier dataset that is not in `data/`); edit `CSV_PATH` before running them.

## Versioning

One file per thing. Do not copy a script to try a variant: change it and commit,
or branch. Git is the version history.

- Every sweep and ablation output carries the commit that produced it:
  `..._run<timestamp>_g<short-hash>`, with `-dirty` appended if there were
  uncommitted changes. Commit before a run you intend to keep.
- Tag states worth returning to (`git tag -a v0.2.0 -m "..."`).
- `v0-legacy-layout` is the flat layout as it came out of elastic-workbench.
- `v0.1.0` is the first reorganised state.

Results in `results/` dated before October 2026 were produced by earlier
script versions, now in `archive/sweep_versions/`; their file names carry no
commit hash.
