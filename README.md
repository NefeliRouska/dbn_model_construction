# DBN model building

Dynamic Bayesian Network modelling of service metrics (throughput, latency,
cores, data quality, buffer size) — research line of Nefeli Rouska.

Split out of [NefeliRouska/elastic-workbench](https://github.com/NefeliRouska/elastic-workbench),
a fork of Boris Sedlak's [elastic-workbench](https://github.com/borissedlak/elastic-workbench).
The workbench itself (services, agents, docker setup, experiment drivers) stays
there; this repo holds only the modelling code, its results, and its git history.

## Data

`data/` is not tracked (≈13 GB). It holds the three raw Prometheus dumps:

    data/prom_dump_all_20260313_192652_seed783122.csv
    data/prom_dump_all_20260316_105011_seed783122.csv
    data/prom_dump_all_20260319_094316_seed783122.csv

## Pipeline

    # 1. raw long-format dump -> wide table (writes dbn_wide_*.csv next to the input)
    python share/metrics/converter_json_to_csv.py --csv data/prom_dump_all_<stem>.csv

    # 2. add the requests-per-second column (writes *_with_rps.csv)
    python add_rps_feature.py --csv data/dbn_wide_<stem>.csv --pattern <pattern>

    # 3. DBN hyperparameter sweep (writes dbn_k_sweep_results_* and saved_dbn_models_top5_*)
    python test_DBN_new_era_AR_fixed.py --csv data/dbn_wide_<stem>_with_rps.csv

`test_DBN_new_era_AR_fixed.py` (with `full_dynamic_bn_lag_memory.py`) is the
latest pipeline; `memory_runner.py` runs the memory ablation on top of it.
Earlier `test_DBN_*` / `full_dynamic_bn_*` files are previous iterations.
Some older scripts still hard-code `share/metrics/dbn_wide_*.csv` as input.
