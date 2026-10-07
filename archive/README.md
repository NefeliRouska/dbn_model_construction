# Archive

Superseded code and early outputs, kept for reference. Nothing here is used by
the current pipeline. Files were moved without edits, so scripts that import
each other by name or read files by relative path will not run from here; to run
one, check out the tag `v0-legacy-layout`, where everything sits flat at the
repo root as originally written.

## `sweep_versions/` — predecessors of `dbn/dbn_sweep.py`

Oldest to newest (date first added):

| File | Added |
|---|---|
| `test_DBN.py`, `test_DBN_traintest.py` | 2026-02-09 |
| `test_DBN_new_new.py`, `test_DBN_new_new_new.py` | 2026-02-16 |
| `test_DBN_new_new_new_json_variables.py` | 2026-03-05 |
| `test_DBN_new_new_new_json_variables_save.py` | 2026-03-17 |
| `test_DBN_new_new_new_json_variables_save_new.py` | 2026-04-22 |
| `test_DBN_new_new_new_json_variables_save_new_new.py` | 2026-05-08 |
| `test_DBN_new_new_new_json_variables_save_new_new_new.py` | 2026-05-15 |
| `..._save_new_new_new_kfold.py` | 2026-05-19 |
| `..._save_new_new_new_kfold_SBN.py` | 2026-05-21 |
| `test_DBN_new_era.py` | 2026-07-07 |
| `test_DBN_new_era_mormalization_discretization.py` | 2026-07-09 |
| `test_DBN_new_era_new_discretization.py` | 2026-07-11 |
| → `test_DBN_new_era_AR_fixed.py`, now `dbn/dbn_sweep.py` | 2026-07-22 |

## `structure_versions/` — predecessors of `dbn/dbn_structure.py`

| File | Added |
|---|---|
| `dynamic_bn_learn_combined.py` | 2025-12-04 |
| `full_dynamic_bn_learn_final.py`, `_final_new.py`, `_final_new_new.py` | 2026-02-09 |
| `full_dynamic_bn_new_new_new.py` | 2026-02-16 |
| `full_dynamic_bn_new_new_new_new.py` | 2026-04-22 |
| `full_dynamic_bn_new_new_new_new_self_loop.py` | 2026-09-14 |
| `full_dynamic_bn_new_new_new_new_self_loop_counter.py` | 2026-09-28 |
| → `full_dynamic_bn_lag_memory.py`, now `dbn/dbn_structure.py` | 2026-09-29 |

## `visualize_versions/` — predecessors of `analysis/visualize_sweep.py`

`DBN_visualize.py`, `DBN_visualize_v2.py`, `DBN_visualize_v4.py`
(→ `DBN_visualize_v5.py`, now `analysis/visualize_sweep.py`).

## `data_collection/` — predecessors of `data_collection/experiment_orchestrator.py`

`run_nominal_changes.py`, `run_changes_randomized*.py` (four steps towards the
automated orchestrator), plus the manual Prometheus export / perturbation /
merge helpers used for the first static-BN experiments.

## `early_bn/` — static Bayesian network work (late 2025)

Nominal-vs-perturbation BN comparison: `DBN_versions/` (`bn_learn*.py`),
`KL.py`, `Compare BN.py`, `visualize_bn_edges.py`, the learned edge lists in
`BN_edges/`, and the small merged datasets they read.

## `old_runs/`

Outputs from the first sweeps (March 2026, seed 161312 and the first
seed 783122 run).
