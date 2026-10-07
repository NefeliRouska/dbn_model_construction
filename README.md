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
| `data_prep/convert_prom_dump.py` | Raw long-format Prometheus dump → wide table (`dbn_wide_*.csv`), one row per second, in chronological order, with a `run_id` per measurement window. |
| `data_prep/add_rps_feature.py` | Rebuilds the *intended* workload pattern as an `rps` column. Not needed by the pipeline: the applied load is in the dumps as `buffer_size_1`, and `dbn/data.py` exposes that as `rps`. |
| `dbn/data.py` | Shared data layer: loading, run / position bookkeeping, counters → rates, aggregation inside runs, folds made of whole runs, (t, t+h) pairs. Everything else imports it. |
| `dbn/dbn_structure.py` | 2-slice DBN structure learning and scores (BIC / AIC). |
| `dbn/dbn_sweep.py` | Full-graph pipeline (pgmpy): feature selection, discretisation, DBN fit, evaluation and baselines over the hyperparameter grid. Imports `dbn_structure`. |
| `dbn/ablation.py` | Fast harness for every modelling option of the target's transition model (bins, granularity, horizon, length of the state, structure-learning engine, table prior, exogenous inputs, ...). Stores per-run results for paired significance tests. `--check-pgmpy` verifies it against pgmpy. |
| `dbn/dynotears.py` | DYNOTEARS structure learning (own implementation; causalnex does not install on Python 3.12). |
| `dbn/memory_ablation.py` | Memory ablation (lags, control flag, velocity features) on top of `dbn_sweep`. |
| `dbn/joint_dbn.py` | Single joint DBN for all three throughput targets (not updated since April 2026; still reads a hard-coded path). |
| `analysis/ablation_report.py` | Effect size and run-level bootstrap significance of every configuration of an `ablation.py` study, pooled over datasets. |
| `analysis/tabpfn_baseline.py`, `analysis/tabpfn_report.py` | DBN versus TabPFN (and gradient boosting) on identical inputs, folds and test samples; pooled summary with paired intervals. |
| `analysis/` (rest) | Markov-order analysis, Markov test, control-variable check, statistical significance of a sweep, LSTM baseline, sweep visualisation. |
| `results/` | Outputs: `sweeps/` (result tables), `models/` (saved top models), `logs/`, `figures/`, `memory_ablation/`, `ablation/`, `tabpfn/`. |
| `docs/` | Written reports. |
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

Each dump holds 560 runs of 361 seconds (one configuration of cores and data
quality per run), separated by ≥ 30 s of unrecorded stabilisation. All three
were collected with the same seed, so they share one sequence of configurations.

## Pipeline

    # 1. raw dump -> wide table (written next to the input, in data/)
    python data_prep/convert_prom_dump.py --csv data/prom_dump_all_<stem>.csv

    # 2. options of the target's transition model, one factor at a time
    #    (--list-studies shows the other studies)
    python dbn/ablation.py --csv data/dbn_wide_<stem>.csv --study ofat
    python analysis/ablation_report.py --study ofat      # pools every dataset run so far

    # 3. full-graph hyperparameter sweep (pgmpy)
    python dbn/dbn_sweep.py --csv data/dbn_wide_<stem>.csv --granularity 1 --n-bins 10 20 30 \
        | tee results/logs/dbn_<stem>_$(date +%Y%m%d_%H%M%S).log

    # 4. memory ablation for one fixed configuration
    python dbn/memory_ablation.py --csv data/dbn_wide_<stem>.csv

    # 5. comparison with TabPFN (needs `pip install tabpfn`; see the script's docstring
    #    for the model versions that require a Hugging Face login)
    python analysis/tabpfn_baseline.py --csv data/dbn_wide_<stem>.csv

Conventions shared by all of them (`dbn/data.py`):

- rows are chronological; a step ahead is taken inside a run, plus, where
  enabled, the transition from the last row of a run to the first row of the
  next one (the configuration change);
- `rps` is the applied load (`buffer_size_1`);
- cores, data quality and `rps` are exogenous: their value at the predicted
  step is an input of the model (`--exog`), not something it has to guess;
- folds are made of whole runs.

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
commit hash. They were also computed on wide tables whose rows were not in
chronological order (see `docs/`), so they are not comparable with later ones.
