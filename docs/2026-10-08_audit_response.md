# Response to the external audit (`AUDIT.md`)

8 October 2026. An independent audit of the work on branch `audit-fixes` was
added to the repository as `AUDIT.md` (with `audit_repro.py` and `audit_out/`).
This note records, for each of its findings, whether it holds, how that was
checked, and what was changed. The reports of 7 October have been corrected in
place where a number or a sentence was wrong; this document holds the corrected
tables.

## What changes for the reader

1. **The comparison with TabPFN was not on equal inputs, and with equal inputs
   the DBN is not ahead of TabPFN v3.5.** The recommended DBN was given two
   time steps and the capacity node; TabPFN had one time step and no capacity.
   Given the same columns as the DBN, TabPFN v3.5 matches its bin accuracy at
   1 s (0.924 against 0.922 at 20 bins) and is clearly better in log-loss and
   as a number (2.8 against 3.7 requests/s). At 30 s the best TabPFN is ahead
   on every measure. Table in §C.
2. **Intervals should be read at the level of the configuration schedule.** The
   three datasets share one sequence of configurations, so they are three load
   patterns over the same 560 conditions, not 3 × 560 independent runs. All
   reports now resample the schedule position. Intervals are wider; no
   conclusion about the DBN against AR changes. Table in §D.
3. **Everything else stands.** The auditor reproduced the main result
   independently (0.9078 against 0.8751), found the data layer and the harness
   correct, and three of its code findings (A, K, L) turned out to be
   mistaken.

## Finding by finding

| # | Auditor's claim | Verdict | Check | Action |
|---|---|---|---|---|
| A | The sweep's AR baseline is conditioned on all variables; its training frame pairs t with t | **Not correct** | The baseline was instrumented: inference receives `throughput_3_t` as its only evidence, and its accuracy equals a `P(bin′ given bin)` table built independently on the same bins and pairs (0.8915 on the fold tested). `two_slice_frame` names its columns `_t` (row t) and `_t1` (row t+h). | None |
| L | `parents <= set(ev_nodes)` in `evaluate()` is always true, so the reduction never fires | **Not correct** | It is false whenever a parent is unobserved (`"learned"` mode with same-slice parents; the legacy evidence mode without the target). When true — always, in the default mode — the reduction *does* fire: that is why the AR baseline is queried with one evidence variable. | None |
| K | The blacklist bans exogenous self-loops and `ensure_self_loops` adds them back | **Not correct** | The blacklist loop skips `a == b`; `ensure_self_loops` is called with the target only. In a fitted graph the exogenous t+1 nodes have no parent or only their own previous value. | None |
| B | The 1 s small-grid sweep has no artefact; its AR (0.887) disagrees with the harness (0.875) | **Half correct** | The outputs had been written outside the repository. 0.887 is the periodic dataset alone (harness: 0.8873 for that dataset); 0.875 is the pooled figure. | Outputs added under `results/sweeps/dbn_sweep_smallgrid_*` and `results/logs/`; the report says which dataset the figure is for |
| C | The recommended DBN had more inputs than TabPFN; the `parents` column came from the wrong model | **Correct** | — | TabPFN re-run with the DBN's inputs and with a richer set; every model's inputs are recorded (`inputs` column); §C |
| D | Pooled intervals treat the datasets as independent | **Correct** | — | Schedule-level bootstrap in `ablation.compare_to_reference(cluster=…)`, used by both report scripts; all summaries regenerated; §D |
| E | The 1% false-alarm threshold was set on the steps it was scored on | **Correct** | — | Threshold set on a random half of the test runs, rates measured on the other half; §E |
| F | Every roll-out structure was offered the capacity node | **Correct**, small effect | 47 of 630 non-capacity structure rows had picked it; none of the `throughput_3` tables in the "local" or "level" rows | Re-run with capacity offered to the "capacity" structure only; §F |
| G | `results/markov_test_results.csv` and `lstm_baseline_results.csv` are stale | **Correct** | They are the pre-fix files | Current outputs added as `results/markov_test_results_<stem>_30s.csv` and `results/lstm_baseline_results_<stem>_30s.csv`; the old files are left as they were |
| H | `buffer_size_1` is the rate setpoint as seen by service 1, not a measurement | **Accepted** | The workbench source is not in this repository; the description is consistent with the data (it takes exactly the orchestrator's values) | Wording changed in `dbn/data.py` |
| I | The sweep writes its CSV only at the end | **Correct** (original design) | — | `dbn_sweep.py` now writes the CSV after every configuration |
| J | TabPFN v3.5 was run and not written up; "nothing is committed" is stale; MAE labelling | **Out of date / correct** | The audit was written while the v3.5 runs were in progress; they were written up the same evening (part 2 §6) | Stale statements corrected; the read-out behind each MAE is now stated |
| M, wording | "halves the loss", "every regime except one", "4–12 points", "5–13%" | **Correct** | — | Sentences corrected in part 2 |

## C. TabPFN v3.5 on equal inputs

`analysis/tabpfn_baseline.py` now groups models by the inputs they receive.
"Local inputs" are the candidate columns of the recommended DBN (the target's
service and its input flow at t and t−1, its controls, the load and the
capacity at t+h); "rich inputs" are every service metric at t and t−1, all
controls, the load and the three capacities; "reference inputs" are what TabPFN
had before (every service metric at t, controls and load at t+h). Same folds,
same test samples, pooled over the three datasets. MAE is each model's
continuous forecast (the DBN's conditional-median node).

| Setting | Model | Inputs | Accuracy | Log-loss | MAE (req/s) |
|---|---|---|---|---|---|
| 1 s, 20 bins | DBN, recommended | local | 0.922 | 0.270 | 3.72 |
| | TabPFN v3.5 | local | **0.924** | **0.217** | **2.78** |
| | DBN | rich | 0.920 | 0.275 | 3.74 |
| | TabPFN v3.5 | rich | 0.909 | 0.240 | 2.93 |
| | TabPFN v3.5 | reference | 0.906 | 0.272 | 3.52 |
| 1 s, 50 bins | DBN, recommended | local | 0.878 | 0.413 | 3.75 |
| | TabPFN v3.5 | local | **0.887** | **0.348** | **2.78** |
| | TabPFN v3.5 | rich | 0.862 | 0.391 | 2.93 |
| 1 s, 10 bins | DBN, recommended | local | 0.948 | 0.194 | 4.17 |
| | TabPFN v3.5 | local | **0.951** | **0.135** | **2.78** |
| | TabPFN v3.5 | rich | 0.944 | 0.146 | 2.93 |
| 30 s, 4 bins | DBN, recommended | local | 0.959 | 0.167 | 11.2 |
| | TabPFN v3.5 | local | 0.940 | 0.174 | 11.6 |
| | DBN | rich | 0.959 | 0.150 | 11.4 |
| | TabPFN v3.5 | rich | **0.969** | **0.095** | **6.3** |
| 30 s, 20 bins | DBN, recommended | local | 0.889 | 0.614 | 10.4 |
| | TabPFN v3.5 | local | 0.868 | 0.490 | 11.6 |
| | DBN | rich | 0.890 | 0.577 | 10.7 |
| | TabPFN v3.5 | rich | **0.892** | **0.374** | **6.3** |

DBN minus TabPFN v3.5 on the same inputs, schedule-level 95% intervals:

| Setting, inputs | Accuracy | Log-loss | MAE |
|---|---|---|---|
| 1 s, 10 bins, local | −0.003 [−0.008, +0.001] | +0.060 [0.043, 0.078] | +1.4 [1.0, 1.8] |
| 1 s, 20 bins, local | −0.002 [−0.006, +0.002] | +0.053 [0.036, 0.072] | +0.9 [0.7, 1.2] |
| 1 s, 50 bins, local | −0.008 [−0.014, −0.002] | +0.065 [0.042, 0.093] | +1.0 [0.6, 1.3] |
| 30 s, 4 bins, local | +0.018 [0.013, 0.024] | −0.007 [−0.024, +0.009] | −0.4 [−1.4, +0.6] |
| 30 s, 20 bins, local | +0.021 [0.014, 0.028] | +0.125 [0.074, 0.177] | −1.2 [−2.3, −0.2] |
| 30 s, 4 bins, DBN local vs TabPFN rich | −0.010 [−0.015, −0.005] | +0.072 [0.059, 0.086] | +4.9 [3.9, 5.9] |
| 30 s, 20 bins, DBN local vs TabPFN rich | −0.003 [−0.010, +0.004] | +0.240 [0.193, 0.290] | +4.1 [3.1, 5.2] |

What to say in the paper:

- **At 1 s, with the same inputs, TabPFN v3.5 and the DBN have the same bin
  accuracy** (TabPFN slightly ahead at 50 bins), and TabPFN gives better
  probabilities and a better numeric forecast (2.8 against 3.7 requests/s;
  persistence 4.5 on these samples). The earlier statement that the DBN was
  ahead at 1 s compared a DBN with two time steps against a TabPFN with one.
- **The inputs matter more than the model.** For both models the gain at 1 s
  comes from seeing the upstream flow at t and t−1; TabPFN with more columns
  (rich) is worse than TabPFN with the right few (local), 0.909 against 0.924.
  Choosing those columns is what the DBN's structure search did.
- **At 30 s, TabPFN v3.5 with rich inputs is the best model** on accuracy at
  4 bins, on log-loss and on numeric error (6.3 against 10–11 requests/s).
- **Still in the DBN's favour:** on configurations never seen in training the
  DBN with capacity nodes is ahead (0.73 against 0.58 accuracy, part 2 §3; in
  that test TabPFN was not given the capacities — gradient boosting with them
  reached 0.57); it uses all the training data rather than a 10 000-sample
  context; its tables are readable, updatable by counting, and give the
  detection and adaptation results of part 2 §9–10.

## D. Intervals at the level of the configuration schedule

Gain in accuracy over the AR table, `throughput_3`; 467 schedule positions in
the test folds.

| Configuration | DBN | AR | Gain | Run-level interval (as printed on 7 Oct) | Schedule-level interval |
|---|---|---|---|---|---|
| 1 s, 20 bins, reference | 0.908 | 0.875 | +0.033 | [0.028, 0.037] | [0.026, 0.039] |
| 1 s, 20 bins, two lags + chain | 0.919 | 0.875 | +0.044 | [0.038, 0.050] | [0.035, 0.052] |
| 1 s, 20 bins, recommended | 0.922 | 0.875 | +0.047 | [0.042, 0.053] | [0.040, 0.055] |
| 1 s, 50 bins, recommended | 0.877 | 0.806 | +0.071 | [0.065, 0.078] | [0.062, 0.080] |
| 1 s, 10 bins, two lags + chain | 0.940 | 0.924 | +0.016 | [0.011, 0.020] | [0.008, 0.023] |
| 30 s, 4 bins, two lags + chain | 0.949 | 0.920 | +0.029 | [0.026, 0.032] | [0.024, 0.034] |
| 30 s, 4 bins, recommended | 0.959 | 0.920 | +0.039 | [0.036, 0.041] | [0.034, 0.043] |
| 30 s, 20 bins, recommended | 0.889 | 0.837 | +0.052 | [0.049, 0.055] | [0.048, 0.056] |

Other targets, recommended configuration, 1 s, 20 bins (schedule-level):
throughput_1 +0.016 [0.013, 0.018]; throughput_2 +0.009 [0.004, 0.015];
avg_p_latency_1 +0.025 [0.022, 0.028]; avg_p_latency_2 +0.008 [0.006, 0.011];
avg_p_latency_3 +0.011 [0.008, 0.013]; buffer_size_2 +0.005 [0.004, 0.006];
buffer_size_3 +0.011 [0.008, 0.014].

One-factor effects against the reference (schedule-level): lags of the other
variables +0.011 [0.007, 0.014]; candidates limited to the service +0.007
[0.004, 0.011], with two lags +0.013 [0.009, 0.017]; uniform prior −0.041
[−0.050, −0.031]; BIC search −0.033 [−0.040, −0.026].

Not significant at this level (they were borderline before): the reference
DBN for `avg_p_latency_2` and `buffer_size_3`; limiting the reference model to
two parents.

The regenerated tables are `results/ablation/summary_*.csv` and
`results/tabpfn/summary_tabpfn-v3.5*.csv`.

## E. Detection with a held-out threshold

| Detector (unannounced reconfiguration, one step) | AUC | Caught at 1% false alarms — first version | — held-out threshold | Realised false alarms |
|---|---|---|---|---|
| `throughput_3`, DBN, 1 s | 0.880 | 73% | 74% | 1.4% |
| `throughput_3`, AR, 1 s | 0.884 | 65% | 68% | 1.1% |
| `throughput_3`, DBN, 30 s | 0.881 | 31% | 31% | 1.0% |
| three throughputs, 1 s | 0.915 | 84% | 86% | 1.1% |
| all eight nodes, 1 s | 0.984 | 93% | 94% | 1.0% |
| service 1 / 2 / 3 nodes, 1 s | 0.943 / 0.970 / 0.954 | 90% / 86% / 88% | 91% / 86% / 89% | 1.4% / 0.9% / 1.2% |

Pooled over datasets and folds the rates did not go down. At 30 s a fold has
about 90 reconfigurations and 900 ordinary steps, so a single fold's rate can
move a lot with the threshold (the auditor reports 42% → 7.5% for one fold);
the 30 s figure should be quoted with that caveat or not at all.

## F. Roll-out without the capacity confound

Accuracy for `throughput_3`, capacity offered only to the "capacity" structure:

| Forecast | 1 s | 2 s | 5 s | 10 s | 30 s |
|---|---|---|---|---|---|
| Direct table | 0.931 | 0.908 | 0.894 | 0.891 | 0.872 |
| Roll-out, own service + input | 0.934 | 0.912 | 0.884 | 0.855 | 0.767 |
| Roll-out, + slow level variable | 0.929 | 0.909 | 0.886 | 0.868 | 0.825 |
| Roll-out, + capacity node | 0.917 | 0.904 | 0.879 | 0.850 | 0.779 |

No number moved by more than 0.007; the conclusions of part 2 §2 are unchanged.

## Not done

- TabPFN v2 was not re-run with equal inputs (v3.5 supersedes it); the v2 table
  in part 2 §11 is marked as unequal.
- The what-if test (unseen configurations) was not re-run with the capacities
  given to TabPFN.
- Result files still carry `-dirty` hashes; rerun under a clean commit before
  quoting them in the paper (`scripts/run_long_jobs.sh studies`).
