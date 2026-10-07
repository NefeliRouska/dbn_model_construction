# Handover: what changed in the DBN pipeline, and what was tested

For Nefeli. This is the living record of the October 2026 audit: every change to
your files, why it was made, what the results look like before and after, and
every test that was run. It is kept up to date; the newest entries are in the
log at the end.

- Branch: `audit-fixes` (not merged into `main`).
- Full result tables and discussion:
  [2026-10-07_fixes_ablation_tabpfn.md](2026-10-07_fixes_ablation_tabpfn.md) (fixes, option studies, TabPFN v2) and
  [2026-10-07_part2_regimes_rollout_whatif.md](2026-10-07_part2_regimes_rollout_whatif.md) (regimes, roll-out, unseen configurations, capacity node, transfer).
  This document does not repeat them; it tells you what is different and where to look.

---

## 1. The short version

- Your model could not beat the autoregressive baseline for reasons that had
  nothing to do with the idea. Three were in the data preparation, two in the
  modelling choices (§2).
- With those fixed, the DBN beats AR for `throughput_3` on all three datasets,
  and for every other throughput, latency and queue length as well.
- The pipeline's defaults have changed (§4). **Results produced before
  7 October 2026 are not comparable with new ones** and should not go in the paper.
- There is a second, fast tool next to your sweep (`dbn/ablation.py`). It is
  what the option studies were run with. Your sweep still works and gives
  consistent numbers.

---

## 2. What was wrong, in order of importance

1. **Rows were not in time order.** `convert_prom_dump.py` sorted the wide
   table by `run_tag` (a string: cores first, then data quality, then repetition)
   and the sweep then dropped both `run_tag` and `timestamp`. Everything that
   assumed "the next row is the next moment" — lags, the two-slice data, the
   temporal folds — was operating on runs in alphabetical order. The folds
   were effectively "train on `cores_1 = 1`, test on `cores_1 > 1`".
2. **Nothing knew where a run ends.** A run is 361 seconds; aggregation took
   blocks of 30 rows, so 8% of the 30-second rows averaged two different
   configurations, and every shift crossed run boundaries. (Your April
   `joint_dbn.py` handled runs correctly; that was lost in the sweep.)
3. **The model was never told the configuration it was predicting for.** The
   configuration changes at random every run. Seen from time t, the first row
   of the next run is unpredictable; persistence is right 97% of the time
   inside a run and nothing can improve the rest without knowing the new
   cores / data quality.
4. **`TARGET_PARENTS_ONLY = True` plus BIC.** The target's parents were limited
   to its own past, so the learned structure could not influence the
   prediction at all. Independently of that, BIC adds no parent to a node with
   many bins: it charges for every cell of the table, including the parent
   combinations that never occur.
5. **Uniform prior on the target's table.** When a search does add parents,
   most parent combinations of the test set were never seen in training, and
   pgmpy's BDeu prior then predicts "all bins equally likely". That is the
   collapse you saw with more than 3–4 parents. It is a property of the prior,
   not of the parents.

Smaller ones: the reconstructed `rps` was 40–60 s late (and the true load is
already in the dump as `buffer_size_1`); cumulative counters were used as
levels; states came from the training data only (the `FAIL: 8` folds); the
LSTM baseline and the Markov test did not give the model the target's own past.

---

## 3. File by file

| File | Status | What changed |
|---|---|---|
| `data_prep/convert_prom_dump.py` | rewritten | Chronological rows; `run_id` column (0, 1, 2, … in the order the runs were executed); `run_tag` and `timestamp` kept; reads the dump as a stream. Variable names (`throughput_3` etc.) are assigned exactly as before. **You must regenerate the wide tables with it.** |
| `data_prep/add_rps_feature.py` | changed | Estimates the offset between the orchestrator's clock and the first row (48 s / 37 s / 61 s for the three dumps) and applies it. No longer needed by the pipeline: see `rps` below. |
| `dbn/data.py` | **new** | The only place where data is loaded. `load_wide` (chronological order, `run_id`/`pos` bookkeeping, counters → per-second rates, `rps` = `buffer_size_1`), `aggregate` (inside runs), `make_temporal_folds` (whole runs), `pair_index` (which row is "t+h" of which), `exog_cols`. |
| `dbn/dbn_structure.py` | changed | Functions take explicit `pairs` instead of shifting rows. Exogenous variables: no learned parents; their t+1 value can be a parent. `target_parents_only` now has three values (`"ar_only"` = your old `True`, `"observed"` = new default, `"learned"` = your old `False`). Tables above 2·10⁶ columns are refused or pruned. `backoff_cpd` (new prior). Throughput ordering rule added to the blacklist. |
| `dbn/dbn_sweep.py` | changed | Uses `data.py`. New command-line options `--granularity`, `--n-bins`, `--horizon`, `--exog`, `--cv`. `run_one` rewritten so lags / velocities / baselines / evaluation all work on the same run-aware pairs. `evaluate` vectorised; reports accuracy at bin changes. Controls keep one state per value. MI on a 20 000-row sample. The grid loop and the result columns are as you wrote them, with a few columns added. |
| `dbn/memory_ablation.py` | small change | `--granularity`; sets the target's parents to "own history only", which is what it studies. |
| `dbn/ablation.py` | **new** | Fast harness for option studies (§5). |
| `dbn/dynotears.py` | **new** | DYNOTEARS implementation. |
| `dbn/joint_dbn.py` | untouched | Still reads the old hard-coded path. |
| `analysis/markov_order_analysis.py` | changed | Shared loader, lags inside runs, `--n-bins`, `--granularity`, guard against huge tables. |
| `analysis/control_variable_check.py` | changed | Shared loader; compares the step on which the configuration changes. |
| `analysis/markov_test.py` | changed | Target's own value is part of the state; run-aware; `--csv`. |
| `analysis/lstm_baseline.py` | changed | Target's own history in the input; windows inside runs; mini-batches; `--csv`. |
| `analysis/ablation_report.py`, `tabpfn_baseline.py`, `tabpfn_report.py` | **new** | Significance tables; TabPFN comparison. |
| `analysis/statistical_significance.py` | untouched, **do not use for the paper** | It treats every hyperparameter configuration as an independent observation. They share data and folds, so its p-values are far too small. Use `ablation_report.py`. |
| `analysis/visualize_sweep.py` | untouched | Reads sweep CSVs; still works on new ones. |
| `data_collection/experiment_orchestrator.py` | small change, **untested** | Writes `rps_log_<…>.csv` with every load value it sends and its own clock. Needs a run on the live workbench to confirm. |

The exact diff of any file: `git diff 4e6e3d1 audit-fixes -- <file>`.

---

## 4. Defaults that changed in `dbn_sweep.py`

| Setting | Before | Now | Why |
|---|---|---|---|
| `MODELING_GRANULARITY_SEC` | 30 | 1 (`--granularity`) | The predictable signal is a load change moving down the chain; it takes 1–2 s. |
| `N_BINS_VALUES` | 4, 6, 10 | 10, 20, 30 (`--n-bins`) | With 4–6 bins nothing ever crosses a bin; there is nothing to predict beyond "same". |
| `TARGET_PARENTS_ONLY` | `True` | `"observed"` | Lets the learned structure reach the target. |
| `N_LAGS` | 1 (forced) | 0 | A forced parent multiplies the table by the number of bins. Lags are better offered as candidates (ablation harness). |
| `USE_CONTROL_FLAG` | `True` | `False` | A configuration is constant inside a run; the exogenous inputs carry this information. |
| `USE_VELOCITY_FEATURES` | `True` | `False` | Forcing them gave tables with 10⁶–10²² columns. |
| `EXOG_MODE` | — | `controls+load` | Cores, data quality, `rps` at t+1 are given to the model. |
| `CROSS_RUN_PAIRS` | — | `True` | The run-to-run transitions are learned and scored, flagged separately. |
| `CPT_PRIOR` | BDeu (implicit) | `backoff` | Unseen parent combinations fall back to the AR table. |
| `rps` | reconstructed column | `buffer_size_1` | It is the load actually applied. |

---

## 5. Results before and after

Old numbers are from your logs in `results/logs/`. New numbers are pooled over
the three datasets.

| | Persistence | AR | DBN |
|---|---|---|---|
| **Before** — periodic, 30 s, 4 bins (`dbn_v7_20260722`, K=4, Markov, uniform, BIC) | 0.92 | 0.920 | 0.866 |
| **After** — 30 s, 4 bins, reference configuration of the harness | 0.920 | 0.920 | 0.921 |
| After — 30 s, 4 bins, + two lags + chain inference at reconfigurations | 0.920 | 0.920 | **0.949** |
| After — 1 s, 20 bins, reference configuration | 0.877 | 0.875 | **0.908** |
| After — 1 s, 20 bins, + two lags + chain | 0.877 | 0.875 | **0.919** |
| After — 1 s, 20 bins, + two lags, candidates limited to service 3 and its input | 0.877 | 0.875 | **0.921** |
| After — 1 s, 50 bins, + two lags + chain | 0.813 | 0.806 | **0.876** |
| After — your sweep itself (`dbn_sweep.py`, periodic, 1 s, 20 bins, K=4, uniform, AIC) | 0.888 | 0.887 | **0.912** |

Persistence and AR at 30 s / 4 bins barely move (0.92 before and after): the
mis-ordering did not change how persistent the signal is, it changed what the
model could learn.

Things you concluded earlier that no longer hold:

- "Order 1 is the optimum for throughput_3's own dynamics; a second lag is
  consistently worse." At 1 s with 20 bins, two values are better than one
  (held-out log-loss 0.382 against 0.394), and lags of the *other* variables
  help more (+0.011 accuracy).
- "Everything beyond 3–4 parents collapses under table sparsity." It collapses
  with the uniform prior. With the backoff prior 4–5 parents are fine.
- "Throughput changes 2.6× more often right after a control action." On
  ordered data: 70.7% of reconfiguration steps change the bin, against 3.6% of
  the other steps (30 s, quartile bins).
- "Other variables dilute the signal." `throughput_2(t)` and
  `buffer_size_3(t)` are selected in nearly every fit and are where the gain
  over AR comes from.

---

## 6. The ablation harness in one page

`dbn/ablation.py` models one thing: the table of the target at t+h given its
parents. When slice t is fully observed, that is all the prediction depends
on, so it gives the same numbers as the full pgmpy model with the same parents
(`--check-pgmpy` verifies this: differences of 10⁻¹⁶). It is fast because it
counts with numpy instead of running inference: a configuration takes seconds
at 1 s granularity.

    python dbn/ablation.py --list-studies
    python dbn/ablation.py --csv data/dbn_wide_<stem>.csv --study ofat
    python analysis/ablation_report.py --study ofat

- A *configuration* is a dictionary of options; `DEFAULTS` at the top of the
  file documents every one. A *study* is a list of configurations
  (`study_ofat`, `study_targets`, …): to test something new, add a function
  like those.
- Columns are named `variable@t`, `variable@t-1`, `d.variable@t` (velocity),
  `variable@t+h` (exogenous, at the predicted step), `y@t` (the target).
- Every run writes `<study>_<dataset>_run<time>_g<commit>_folds.csv` (one row
  per configuration, model and fold) and `…_runs.csv.gz` (per-run sums, used
  for significance; not in git, regenerate by re-running the study).
- `ablation_report.py` pools the datasets and gives, per configuration, the
  difference to AR and to the reference configuration with a 95% interval from
  a bootstrap over runs.

What it does not do: learn edges between the other variables. For the full
graph (and for figures of the learned network) use `dbn_sweep.py`.

---

## 7. Every test that was run

Outputs are in `results/ablation/` and `results/tabpfn/` unless stated.
"All" = the three datasets.

| # | Test | Data | Where | Finding |
|---|---|---|---|---|
| 1 | Audit checks on the raw dumps: run order, rows per run, mixed bins, load reconstruction, share of persistence errors at run changes | all | report §2 | basis for §2 above |
| 2 | Converter output: 560 runs × 361 rows, monotonic time | all | — | pass |
| 3 | Harness against pgmpy (probabilities, BIC) | periodic | `--check-pgmpy` | identical |
| 4 | DYNOTEARS on a synthetic SVAR | synthetic | — | all edges recovered |
| 5 | `ofat`: one option at a time, 60 configurations | all | `summary_ofat.csv`, report §5.3 | prior and search matter most; lags of other variables help |
| 6 | `bins_granularity`: 4–50 bins × 1/5/10/30 s | all | report §5.2 | gain over AR grows with bins, lives at 1 s |
| 7 | `horizon`: 1–30 steps | all | report §5.6 | gain fades after ~2 s |
| 8 | `cv`: expanding vs blocked folds | all | report §5.7 | same conclusion |
| 9 | `config_change`: plain / second table / chain inference at reconfigurations | all | report §5.5 | chain: 0.45 → 0.74 at 30 s / 4 bins |
| 10 | `interactions`: 128-configuration factorial | all | report §5.8 | state length and search interact little |
| 11 | `best`: combinations per bin count | all | report §5.8 | table in §5 above |
| 12 | `refs` with gradient boosting | all | report §5.9 | DBN ahead on bins, behind on MAE (table read-out) |
| 13 | `targets`: every non-exogenous variable as target | all | report §5.11 | all beat AR at 20 bins |
| 14 | `constraints`: restricted candidates, forced parents | all | report §5.12 | restricting helps (+0.007), forcing controls hurts |
| 15 | `numeric`: continuous target node on the same parents | all | report §5.13 | conditional median: 4.14 against 4.65 for persistence |
| 16 | TabPFN v2 vs DBN, six settings | all (two settings on periodic only) | report §7 | DBN level or slightly ahead on bins at 1 s; TabPFN better on MAE and at 30 s |
| 17 | `dbn_sweep.py` end to end, small grid, 30 s / 10 bins and 1 s / 20 bins | periodic | report §2 | runs without failed folds; 0.912 vs 0.887 at 1 s |
| 18 | Fitted pgmpy graph inspected: parents of exogenous nodes and of the throughputs | periodic | report §5.12 | exogenous nodes have no parents; AIC recovers the service chain |
| 19 | `memory_ablation.py` | periodic, 30 s, 10 bins | `results/memory_ablation/` | runs; forced velocity columns are refused (table too large) |
| 20 | `markov_order_analysis.py`, `control_variable_check.py`, `markov_test.py`, `lstm_baseline.py` | periodic | report §5.14 | all run; conclusions in §5 above |

| 21 | `regime_study.py`: one global model vs regime-specific tables / structures; regime never seen in training; detection from surprise | all | part 2 §1, `results/regimes/` | regimes change values, not the core structure; unseen regime −4 to −12 points; reconfigurations detected with AUC 0.88 |
| 22 | `dbn/rollout.py`: the chain as one model, unrolled 1–30 s, seven structures, vs direct tables | all | part 2 §2, `results/rollout/` | good to ~2 s, then direct tables win; a slow level variable helps |
| 23 | `whatif_study.py`: prediction for configurations never trained on | all | part 2 §3, `results/whatif/` | plain tables 0.45; capacity node with noisy-min prior 0.73; boosting 0.46–0.57 |
| 24 | `ablation.py --study capacity`: capacity node in the dynamic model | all | part 2 §4 | accuracy at reconfigurations 0.42 → 0.65 (1 s), 0.74 → 0.84 (30 s, 4 bins) |
| 25 | `transfer_study.py`: train on one workload, test on another; pooled | all | part 2 §5, `results/transfer/` | varying-load models transfer; pooling best |
| 26 | `adaptation_study.py`: detection with all nodes; recovery when tables are updated after a new regime appears | all | part 2 §9–10, `results/adaptation/` | reconfigurations caught 93% at 1% false alarms with all nodes (73% with one); updating tables recovers most of the loss |
| 27 | `ablation.py --study recommended`: recommended configuration for all eight variables, with numeric read-outs | all | part 2 §11 | every variable beats AR; conditional-median MAE below persistence for 7 of 8 |
| 28 | `ablation.py --study extras`: slow level variable in the one-step model; DYNOTEARS penalty | all | part 2 §12 | level: only helps at 5–10 s; DYNOTEARS best at λ = 0.02, still behind |
| 29 | `whatif_study.py --tabpfn v2` | all | part 2 §3 | TabPFN v2 does not generalise to unseen configurations for services 2 and 3 (0.41 against 0.73) |
| — | TabPFN v3.5 | — | part 2 §6 | **not run**: Prior Labs licence key missing on this machine |

Not re-run: the full 360-configuration sweep (about 20 hours per dataset at
1 s), `joint_dbn.py`, `visualize_sweep.py`, the figures in `results/figures/`.

---

## 8. What I would do next (in this order)

1. Regenerate anything you want in the paper from the new wide tables.
   `results/sweeps`, `results/models`, `results/figures` and `results/logs` are
   all from before the fix.
2. Use 1 s and 20–50 bins. Report accuracy and log-loss against AR **on the
   same bins**, accuracy at bin changes, and the MAE of the conditional-median
   read-out. Do not compare accuracies across different bin counts.
3. Use the capacity node (`capacity=True`, `on_change="capchain"`): it is what
   lets the model predict configurations it has not seen (part 2 §3–4). The
   chain as one unrolled model was built and tested (part 2 §2): it works for
   1–2 s ahead; for longer horizons train the tables for that horizon.
4. Collect one dump with a different seed (the three dumps share one
   configuration schedule) and record the 30 s stabilisation period, which is
   currently lost.
5. TabPFN v3.5 once access to the gated weights is set up.

---

## 9. Log

Newest first.

### 2026-10-07 (fourth pass, night)

- New: `analysis/adaptation_study.py` (tests 26); studies `recommended` and
  `extras` and options `level=True`, `dynotears_lambda` in `dbn/ablation.py`
  (tests 27–28); `--tabpfn` in `whatif_study.py` uses the median forecast
  (test 29; a first run with TabPFN's mean was discarded — the mean is far off
  for the skewed throughput of services 2 and 3).
- Part 2 report: sections 9–12 added, summary extended.
- Still no Prior Labs licence key on this machine: TabPFN v3.5 not run.

### 2026-10-07 (third pass, evening)

- New scripts: `analysis/regime_study.py`, `analysis/whatif_study.py`,
  `analysis/transfer_study.py`, `dbn/rollout.py` (tests 21–25).
- `dbn/data.py`: `fit_capacity_rate`, `add_capacity`; `capacity_*` columns are
  treated as exogenous. `dbn/ablation.py`: options `capacity=True` and
  `on_change="capchain"` (chain with capacity nodes and a noisy-min prior),
  study `capacity`.
- TabPFN v3.5 attempted. Hugging Face access works; the run is blocked by
  TabPFN's own licence step on ux.priorlabs.ai (no key in
  `~/.cache/tabpfn/auth_token`). This Python also lacks a certificate bundle,
  which made the library report it as a Hugging Face error: run TabPFN
  commands with `SSL_CERT_FILE=$(python -m certifi)`.
- Intermediate result files of scripts that were re-run while being developed
  were moved to `results/_superseded/` (git-ignored); they can be deleted.
- Nothing longer than a few minutes was left running. Jobs for the other
  machine are listed in part 2 §7.

### 2026-10-07 (second pass)

- New studies in `dbn/ablation.py`: `targets`, `constraints`, `numeric`
  (tests 13–15). New options: `pool="local"`, `numeric=True`; any variable can
  be `target`; the shared throughput bins and chain inference work for any
  throughput target.
- `dbn_structure.py` / `dbn_sweep.py`: the rule "a later service's throughput
  is not a parent of an earlier one's" is now applied in both searches. The
  function that builds it for the two-slice search (`_build_blacklist_two_slice`)
  existed but was never called.
- Fitted pgmpy graphs inspected (test 18).
- Report sections 5.11–5.13 added.

### 2026-10-07 (first pass)

- Audit; all fixes of §2–§4; harness, DYNOTEARS, report scripts; tests 1–12,
  16, 17, 19, 20. Committed as `600064d`.
- A first TabPFN run was discarded: continuous forecasts were being binned
  without accounting for the target being a count, which penalised TabPFN
  whenever the current value sat exactly on a bin edge.
