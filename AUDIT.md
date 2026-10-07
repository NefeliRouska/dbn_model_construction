# Audit of the DBN pipeline (`dbn-model-building`, branch `audit-fixes`)

Scope: read-only audit of the Python code, the saved result files, and the claims made in
`README.md`, `docs/HANDOVER.md`, `docs/2026-10-07_fixes_ablation_tabpfn.md` and
`docs/2026-10-07_part2_regimes_rollout_whatif.md`.
Nothing in the repository was modified. Files created for the audit: this report,
`audit_repro.py`, `audit_out/` (independent re-runs). Both are additive and can be deleted.

Environment used: `.venv/bin/python` (Python 3.12, pgmpy 0.1.26, pandas 3.0.6, numpy 2.5.3);
data read from the regenerated wide tables in `data/`.

---

## 0. Executive summary

**The main result is real.** I re-implemented the evaluation independently (new script, not the
repo's own reporting path) and reproduced the headline numbers within rounding:

| Configuration (pooled, 3 datasets) | Reported | My independent re-run |
|---|---|---|
| 1 s / 20 bins: DBN accuracy | 0.908 | **0.9078** |
| 1 s / 20 bins: AR table | 0.875 | **0.8751** |
| 1 s / 20 bins: gain [95% CI] | +0.033 [0.028, 0.037] | **+0.0327 [+0.0280, +0.0373]** |
| 1 s / 20 bins: DBN log-loss | 0.338 | **0.3375** (AR 0.4869) |
| 1 s / 20 bins, two lags + chain | 0.919 | **0.9189** |
| 30 s / 4 bins, two lags + chain | 0.949 | **0.9493** |
| BIC + uniform prior = AR | 0.875 | **0.8744** (AR 0.8751) |
| 1 s / 50 bins: DBN / AR | 0.864 / 0.811 | **0.8623 / 0.8060** |

The data layer is sound (2 021 160 rows = 560 runs × 361 rows per dump; 201 600 within-run pairs,
exactly 1.000 s apart; 559 cross-run pairs at 30.2–30.5 s; monotone timestamps; folds are whole
runs; no run is shared by train and test). The ablation harness's `summary_*.csv` files agree
row-for-row with the `*_folds.csv`/`*_runs.csv.gz` files that produced them, and all 14 studies
produced exactly their advertised number of configurations with **zero failed folds**.

**But several things in the repository do not mean what the reports say they mean, and six of
them matter for the paper.** In order of importance:

| # | Finding | Severity |
|---|---|---|
| A | The `dbn_sweep.py` "AR-DBN" baseline is conditioned on **all** variables at `t` (its parent set is every variable), because of a tautological check in `evaluate()`; the intended AR-only evidence is never used. The number in the report §2 table labelled "AR acc = 0.887" is not a `P(bin' \| bin)` autoregression. | **major** |
| B | The 1 s sweep numbers in report §2 (and HANDOVER test 17) have **no result file and no log**; they cannot be reproduced, and the AR figure there disagrees with the harness's AR on identical bins. | **major** |
| C | The recommended DBN in the "DBN vs TabPFN" comparison is given **strictly more information** than TabPFN (2 lags + capacity node) while the module docstring claims equal information; the `parents` column in the TabPFN result files is written from the wrong model and hides this. | **major** |
| D | The pooled significance statement treats the three datasets as 3 × 467 independent runs, but they share **one** configuration schedule (same seed). A cluster bootstrap over the schedule makes two of the TabPFN significance claims disappear, and the "1 401 independent test runs" framing overstates the evidence. | **major** |
| E | **Detection rates in part 2 §1 and §9 are in-sample operating points**: the "1 % false alarms" threshold is a quantile of the *same* test fold whose detection rate is then reported. Re-done with a held-out threshold, the 30 s reconfiguration detection falls from 31 % to ≈26 %, and one fold from 42 % to 7.5 %. The AUC columns are unaffected. | **major** |
| F | **The roll-out ablation is confounded**: `dbn/rollout.py` gives `capacity_i@t+h` as a candidate parent to *every* structure, and it is selected in 92/675 structure rows — including `local` (11/75 node-folds) and `level`. The gain attributed to the "slow level variable" (0.765 → 0.821 at 30 s) is therefore not capacity-free. | **major** |
| G | `results/markov_test_results.csv` and `results/lstm_baseline_results.csv` are **stale and contradict** report §5.14 (markov +0.0064 vs the current script's +0.0133; LSTM MAE 51.95 vs the current 11.6). The document is right and the artifacts are wrong, but the artifacts are what a reader would find. | minor |
| H | The evidence for the load variable being the *applied* load (`buffer_size_1`) is weaker than stated; the within-run results turn out not to depend on it at all (I verified), so this is a documentation issue, not a result issue. | minor |
| I | `dbn_sweep.py` writes its result CSV **only at the end**, so the 30 s sweep that was stopped after 383/1080 configurations lost everything and had to be re-derived from logs — a 3-dataset × 386-configuration run that is not reproducible from the pipeline. | minor |
| J | **A TabPFN v3.5 comparison was run and never written up, and v3.5 changes the conclusion**: on the periodic dump the recommended DBN is still ahead at 1 s (0.927 vs 0.915) but **behind at 30 s** (0.960 vs 0.967 accuracy; MAE 11.4 vs 6.8 requests/s), while three documents say v3.5 was not run. Same finding covers "nothing is committed" vs committed `-dirty` result files and the `ae`/`aeh` labelling. | **major** |
| K | `ensure_self_loops` force-adds `X_t → X_t1` for every node, including exogenous ones, contradicting the stated rule "exogenous variables take no parent other than their own previous value". Harmless (exogenous `t+1` is always given as evidence) but wrong as stated. | minor |
| L | `evaluate()` in the sweep contains the tautology `parents <= set(ev_nodes)` — it always holds, so the intended "restrict the query to the target's parents" reduction never triggers. Harmless for the DBN (extra evidence cannot change the answer) but it is the mechanic behind finding A. | minor (cause of A) |
| M | Roll-out report §2's prose attributes `rollout:local`'s numbers to the `rollout:level` row; `rollout:level?` is an undocumented variant; `structures.csv` writes 50 duplicated `direct` rows instead of 15. | cosmetic |
| — | Overstated sentences in part 2: §1's "selected in every regime except one" (8/15, 13/15, 6/15 folds by regime) and "falls by 4–12 points" (3.6 for `s23`); §3's "5–13 % above the median" (2.6–3.1 % for quality 800–1000). | minor |

The letters are the section ids in §2. **A, C, D, E, F and J are the six findings that change how
a result should be read or reported**; the rest are documentation or hygiene. Nothing found indicates
that the headline DBN-beats-AR result is an artefact: I reproduced it independently, and its main
suspicious input (the load variable) turns out to make no difference (finding H).

---

## 1. Verified correct

1. **Data layer** (`dbn/data.py`): `load_wide` (chronological sort, run bookkeeping, counters →
   per-second rate, `rps` = `buffer_size_1`, variance filter), `aggregate` (aligned to the start of
   each run, drops the short trailing window so no window mixes two configurations),
   `make_temporal_folds` (whole runs, expanding or blocked), `pair_index` (exact `horizon` inside a
   run; the cross-run pair is flagged), `two_slice_frame`. Measured: 2 021 160 rows per dump;
   `pos[i] >= max_history - 1` prevents lags/velocities from reaching across a run boundary; all
   within-run `(t, t+1)` gaps are exactly 1.0 s; 559 cross-run gaps of 30.2–30.5 s.
2. **`dbn_structure.py`**: the intra/two-slice blacklists encode the stated causal ordering; the
   throughput-direction rule (a later service is not a parent of an earlier one) *is* applied in
   both searches (`build_blacklist`, `build_inter_only_blacklist` + `_build_blacklist_two_slice`);
   the `MAX_CPT_COLUMNS` guard refuses/warns instead of letting pgmpy allocate 10⁶-column tables.
   `AicScoreCustom`'s zero-init fix for `np.log(..., where=)` is correct (`nansum` would not have
   caught the uninitialised cells).
3. **`dbn/ablation.py` — the harness itself.** This is the strongest part of the repo. The
   reduction "with slice *t* fully observed, the target's prediction depends only on its parent set
   and table" is correct. Discretisation is fitted on training samples only. `Coder.to_bins`
   (level/delta) is correct, and `Samples` builds `y@t-1`, velocities and gaps only where the
   history lies inside the run. The `backoff` prior is a genuine improvement and is what lifts
   log-loss from 0.487 (AR) to 0.338.
4. **Result-file integrity.** For every study I cross-checked, the summary equals the source
   tables. Example (extras, 3 datasets): summary `0.9182 / 0.2919 / 5.8078` =
   sample-weighted recomputation from the folds `0.9182 / 0.2919 / 5.8078`. Same for
   `ofat`, `bins_granularity`, `best`, `targets`, `capacity`, `recommended`, `refs`.
   `dbn_sweep`'s partial 30 s summary (383 configurations, all with `folds_ok = 5`) reproduces the
   §13 table exactly (`4/AIC 0.9126 vs 0.9109`, `4/BIC 0.9127 vs 0.9103`, `10/AIC 0.8517 vs 0.8482`,
   `10/BIC 0.8478 vs 0.8471`, `20/AIC 0.7778 vs 0.7740`, `20/BIC 0.7740 vs 0.7730`).
5. **The load variable is not what makes the numbers.** I re-ran the reference configuration with
   `exog='none'`, with `exog='controls'`, and with the `rps` column removed from the candidate pool
   entirely. Across the three datasets the DBN accuracy is unchanged to ±0.0004
   (periodic 0.9155/0.9151/0.9155/0.9155; oial 0.9087/0.9147/0.9087/0.9087;
   unpredictable 0.8992/0.8991/0.8992/0.8992). Whatever the provenance of `buffer_size_1`, the
   reported gain over AR does not come from it.
6. **`dbn/rollout.py`** reproduces its table exactly from the per-run files for `persistence`,
   `ar`, `direct`, `rollout:learned`, `rollout:local`, `rollout:chain`, `rollout:capacity` and
   `rollout:level` (all horizons; e.g. `direct` 0.9308/0.9074/0.8943/0.8908/0.8720 vs the report's
   0.931/0.907/0.894/0.891/0.872).
7. **Part 2 §13** (partial 30 s sweep) is exact.
8. **Part 2 §1 (regimes) is reproducible bit-for-bit** from the current code: the 660 regime score
   rows, 110 parent rows and 75 detection rows in `results/regimes/` are identical on a re-run
   (max |Δaccuracy| = 0, max |Δlog-loss| = 2 × 10⁻¹⁶), as are the per-fold change-detection rates.
   Part 2 §9/§10 (`results/adaptation/`), §3 (`results/whatif/`, including the learned rate table
   199.5/116.7/77.2/55.9/44.8/34.4/28.6/25.5 for quality 300–1000 and the saturated medians
   177.0/109.3/71.5/52.2/41.4/33.4/27.8/24.8), §5 (`results/transfer/`), and fixes §5.14 all
   reproduce from the current scripts.

---

## 2. Findings in detail

### A — `dbn_sweep.py`'s "AR-DBN" baseline is not an AR baseline (major)

`dbn/dbn_sweep.py:1070-1092`

```python
edges = [(f"{v}_t", f"{v}_t1") for v in nodes]   # diagonal: every variable's own past
df_2s = D.two_slice_frame(train_ready, pairs_train)   # <- train_ready has BARE column names
model.fit(df_2s, ...)
res = evaluate(model, test_ready, pairs_test)
```

`train_ready`'s columns are `throughput_3`, `cores_1`, … — not `throughput_3_t` / `throughput_3_t1`.
The fit therefore does *not* pair `v(t)` with `v(t+1)`; it pairs `v(t)` with `v(t)` (identity) under
the `_t1` name. `pgmpy` accepts it because the column names match the model's node names, so the
conditional counts are the same number as a correctly built AR table — the fit is accidentally right.

The failure is at evaluation, `dbn/dbn_sweep.py:1005-1007`:

```python
parents = set(model_2s.get_parents(target_t1_node))
if parents and parents <= set(ev_nodes):        # ev_nodes keys are the parents themselves
    ev_nodes = {node: ev_nodes[node] for node in sorted(parents)}
```

For the AR model the target node's parent set is **every variable** (`X_t` for each `X`), and
`evaluate` builds `ev_nodes` from every column of `test_ready`, so all 43 of them are passed as
evidence. Verified by construction: `evidence columns evaluate() passes: [all 43 non-target
columns]`. The consequence is that the AR model — whose edges are diagonal — marginalises the extra
evidence away, so its prediction is `P(y(t+1) | y(t))`, but through a needlessly huge
VariableElimination query per distinct 43-tuple.

Why this matters: (i) the report's §2 table, its "DBN acc / AR acc" columns and HANDOVER test 17
present `0.887` as the number to beat; (ii) my own reconstruction of the AR table *on the sweep's
own bins* gives **0.8995** for one fold (persistence 0.8994), and the harness's own AR at
1 s / 20 bins pooled is **0.8751**; (iii) the comment in the code ("This is exact and keeps the
number of distinct queries small") describes a reduction that cannot fire.

Fix: build the AR frame so that the `_t`/`_t1` names hold `t`/`t+1` values, and make the
`parents <= set(ev_nodes)` test actually test what it says (e.g. compare against the model's
variable set minus the target, or drop the branch). Then re-run §2.

### B — The 1 s sweep numbers in §2 are not reproducible (major)

`docs/2026-10-07_fixes_ablation_tabpfn.md:73-85` and `docs/HANDOVER.md:200` report a completed
`dbn_sweep.py` small grid at 1 s / 20 bins (0.912 vs 0.887, 289 s / 156 s per configuration).
There is:

- no `results/sweeps/dbn_k_sweep_results_*_1s_*` file (only the old, pre-fix ones and the 30 s
  partial summary);
- no `results/logs/dbn_*_1s_*.log` (the only October logs are the three `_30s_20261007_192028.log`);
- no 1 s entries in `results/models/saved_dbn_models_top5_*` for that session.

I therefore cannot verify the §2 grid, and the one number in it that I *can* check against the
harness (AR at 1 s / 20 bins) disagrees by 1.2 accuracy points with the same quantity computed by
`ablation.py` (0.887 vs 0.875). The rest of §2 (30 s rows, relative ordering of discretisers) is
consistent with the harness.

Fix: re-run the 1 s grid and commit its outputs, or mark those rows as lost.

### C — The "DBN vs TabPFN" comparison in part 2 §11 is not information-equal (major)

`analysis/tabpfn_baseline.py:1-8` states: "Every model predicts `throughput_3` at `t+h` from the
same information (the candidate columns of `dbn/ablation.py` …)". Line 153 builds the setting's
candidate set once:

```python
cfg = A.make_cfg(granularity=gran, horizon=horizon, n_bins=n_bins, **overrides)
...
res, info = A.run_fold(S, cfg, tr, te_sub, extra)      # tabpfn uses cands of `cfg`
```

with `cfg` at its defaults (`ar_order=1, other_lags=1, pool="core", capacity=False`), while
lines 175-180 score the recommended configuration with

```python
RECOMMENDED = dict(ar_order=2, other_lags=2, pool="local", capacity=True, on_change="capchain",
                   numeric=True)
S_rec = A.get_samples(args.csv, rec)
res_rec, _ = A.run_fold(S_rec, rec, tr, te_sub)
```

`tabpfn` is given the candidate columns of `cfg` (22 columns, one lag of each variable, no
capacity); `dbn_rec` selects its parents from the candidate columns of `rec` (`2` lags, local pool,
`capacity_3`) — nine columns that TabPFN never sees. So the §11 sentence "With the recommended
configuration the DBN is ahead of TabPFN v2 in bin accuracy in all four settings" compares two
models with different inputs. The §7 comparison (reference configuration, `other_lags=1`, no
capacity) *is* information-equal, and that is the one to keep.

Compounding this, `analysis/tabpfn_baseline.py:185` writes `"parents": ",".join(info["parents"])`
inside the loop over **all** models, but `info` is the dict returned at line 172 for the *setting*
configuration; `res_rec` returns its own `info` which is discarded at line 178. Every `parents`
value in `results/tabpfn/*_folds.csv` is therefore the reference model's parent set, including the
rows for `dbn_rec` and `dbn_rec_median` — which is precisely the column a reader would use to check
that the comparison is fair.

Fix: give `tabpfn` and `dbn_rec` the same candidate columns (either restrict `dbn_rec` to the
setting's cands for this comparison, or give TabPFN the recommended candidate set as well), record
per-model `info`, and state the asymmetry if it is kept deliberately.

### D — The pooled confidence intervals overstate the evidence (major)

`analysis/ablation_report.py:66` pools the three datasets by prefixing `run_id` with the dataset
name, and `dbn/ablation.py:1188-1213` bootstraps **runs** independently. But all three dumps were
collected with the same seed and share one configuration schedule — I verified that
`cores_1..3` and `data_quality_1..3` are identical for all 560 runs in the three dumps — so run
*k* of dataset 1, dataset 2 and dataset 3 is the same experimental condition, not an independent
replication. The effective number of independent schedule points is 560, not 1 401.

A cluster bootstrap that resamples the schedule point (drawing the three datasets together)
materially changes conclusions in the TabPFN comparison:

- part 2 §11/§7, 1 s / 20 bins, model vs TabPFN accuracy: reported
  `+0.006 [+0.001, +0.012]` → cluster bootstrap `p = 0.065`, CI `[-0.0004, +0.0126]`;
- 30 s / 20 bins, model vs TabPFN accuracy: reported `-0.009 [-0.018, -0.000]` → `p = 0.145`.

The recommended-DBN (`dbn_rec`) accuracy claims do survive (p ≤ 0.0002), and none of the
`dbn_rec_median` MAE differences are significant at either level.

`docs/HANDOVER.md:9` and report §1 say "paired bootstrap over 1 401 test runs" without the caveat
that they are three copies of one schedule; report §8 does state the caveat, but the number in the
abstract does not. `docs/2026-10-07_fixes_ablation_tabpfn.md:119-123` also claims the p-values are
"Holm-corrected within a study" — `ablation_report.py` does that, `analysis/tabpfn_report.py` does
not (there are no `p_holm_*` columns), and 56/77 `p_acc` values sit at the 1e-4 bootstrap floor.

Fix: report the schedule-level (cluster) interval as the primary one, or say explicitly that the
run-level interval is conditional on these three dumps.

### E — Detection rates are in-sample operating points (major)

`analysis/regime_study.py:190` and `analysis/adaptation_study.py:176` both do

```python
thr = np.quantile(s[~change], 0.99)          # threshold on THIS test fold
... float((s[change] > thr).mean())          # detection rate on the SAME fold
```

so the "1 % false alarms" operating point is calibrated on the very steps it is scored on. The
sub-audit redid it with a leave-one-fold-out threshold (periodic dump, `throughput_3`): at 1 s the
in-sample and out-of-fold rates are 0.720 vs **0.739** (no optimism, the score distribution is
stable over ~33 k ordinary steps per fold), but at 30 s they are 0.334 vs **0.259**, and fold 5
goes from 0.419 to **0.075**. So part 2 §1's "DBN, 30 s: 31 %" is about 5 points optimistic, and
§9's step-level 93 % / 84 % / 73 % and per-service 86–90 % are same-fold operating points that
should be labelled as such. The AUCs in the same tables are threshold-free and unaffected. The
CUSUM part of `adaptation_study.py:105-127` *is* properly held out (calibrated on half the known
runs, false alarms reported on the other half) — that part is fine.

Fix: split the ordinary steps of the test folds into a calibration half and a reporting half, as
the CUSUM code already does.

### F — The roll-out structure ablation is confounded by the capacity node (major)

`dbn/rollout.py:150` builds the selection config with `capacity=True` for **every** structure:

```python
cfg = A.make_cfg(target=v, n_bins=n_bins, horizon=horizon, cross_run=False, ar_order=2,
                 capacity=True, other_lags=2, ...)
```

`dbn/data.py:37` makes `capacity_` a control prefix, so `capacity_i@t+h` is a legal exogenous
candidate in `allowed()` for every variant. It is in fact selected: 92 of the 675 rows of
`results/rollout/*_structures.csv` have a capacity parent, e.g.

    fold 1  learned  throughput_1  [throughput_1@0, capacity_1@next, throughput_2@0]
    fold 1  local    throughput_1  [throughput_1@0, capacity_1@next, throughput_1@1]
    fold 1  local    throughput_2  [throughput_2@0, capacity_2@next, throughput_2@1, cores_2@next]
    fold 1  level    throughput_2  [throughput_2@0, ema_throughput_2@0, throughput_2@1, capacity_2@next]

by structure: `learned` 15/75 node-folds, `local` 11/75, `chain` 6/75, `level` 5/75, `level?` 7/75.

So part 2 §2's rows "rollout:learned / local / chain / level" are all "…plus a capacity node", and
the improvement attributed to the slow level variable partly comes from a capacity parent. Only the
`capacity` row is clean. Costs nothing to fix (build the selection config with `capacity=False` and
add capacity only for the `capacity` structure), and it is worth re-running, because the level
variable is the one §8 wants to build on.

### G — Two result artifacts contradict the report (minor, but will be found)

- `results/markov_test_results.csv` (10:53) uses `n_bins` 3/4/5 and reports a mean improvement of
  **+0.0064**; the current `analysis/markov_test.py` tests `n_bins` {4,10,20}, writes
  `markov_test_results_{stem}_{gran}s.csv`, and gives **+0.0133** — which is what report §5.14
  quotes. The file the script writes exists nowhere in the repo.
- `results/lstm_baseline_results.csv` (12:39:56, the same second as the script's mtime) reports
  `throughput_3` MAE **51.95** against persistence 10.79 with `n_features = 72`; re-running the
  current script gives **11.606** against 4.170 with `n_features = 41` — again what the report
  quotes.

Both saved files are pre-fix artefacts, so the report's §5.14 numbers are right, but the two CSVs
that a reader will open say something different. Regenerate or delete them.
`control_variable_check.py` and `markov_order_analysis.py` write no file at all, so their §5.14
claims have no artifact either; both were re-run and reproduce (70.71 % / 3.58 %; log-loss
0.4623 / 0.3938 / 0.3820 / 0.3948 for orders 0–3).

### H — What `buffer_size_1` actually is (minor, documentation)

`dbn/data.py:29-33` says `buffer_size_1` "reports [the arrival rate] exactly … so it is the
reference for the load". In the workbench the metric is a gauge set from
`self.client_arrivals.get('buffer', 0)` (`elastic-workbench/iot_services/IoTService.py:75-76`) and
for service 1 the batch size is built from the same value
(`IoTService.py:125-133`); downstream services instead store *queue backlog* in the same field
(`Service_Wrapper.py:110-112`). So `buffer_size_1` is the orchestrator's offered-rate setpoint as
seen by service 1 — which is what the pipeline wants — but it is not an independent measurement of
the load, and the real setpoint history (`rps_log_*.csv`) is a change to the orchestrator that the
handover itself marks "untested".

Practically this does not affect the results (finding 5 above: removing the load changes nothing).
The docstring in `data.py` should say "the offered rate the orchestrator sends to service 1", not
"the load actually applied", and the claim in `docs/HANDOVER.md:61` that the reconstructed rps was
"40–60 s late (and the true load is already in the dump as `buffer_size_1`)" should be re-examined.

### I — The 30 s sweep lost its own results (minor)

`dbn_sweep.py` accumulates `all_results` and writes `RESULTS_CSV_PATH` only after the whole grid
(`main`), and the per-configuration model saving happens through `BestPerBinsTracker`; when the run
was interrupted at 383/1 080 configurations there was no CSV and no saved model, so part 2 §13's
table had to be reconstructed from stdout by hand
(`results/sweeps/dbn_k_sweep_partial_30s_from_logs_20261007.csv` — a reconstruction that reproduces
the table exactly, so the numbers are fine, but the artifact is not produced by the pipeline).
The logs contain 127 + 129 + 130 started configurations, i.e. slightly more than the 383 summarised.

Fix: flush the result row per configuration (append) and persist the tracker, as `ablation.py`
already does.

### J — Documentation integrity (minor)

1. Both reports say "nothing is committed" / "All changes are uncommitted on branch `audit-fixes`",
   but the working tree is clean and every result file is committed with a `-dirty` suffix
   (`docs/2026-10-07_fixes_ablation_tabpfn.md:628`, `docs/HANDOVER.md:8`). The `-dirty` marker means
   no committed hash describes the code that produced the committed numbers.
2. **TabPFN v3.5 was in fact run** on 7 Oct after the last commit, and — unlike the v2
   comparison — it is not kind to the DBN. A partial run is on disk:
   `results/tabpfn/tabpfn-v3.5_20260313_192652_seed783122_run20261007_204132_gbfbb3b9_folds.csv`
   (+ `_runs.csv.gz`, hidden by `.gitignore`) — untracked, and it covers **5 settings × 10 models ×
   5 folds** on the periodic dump (`g1s_h1_b20`, `b50`, `b10`, `g30s_h1_b20`, `g30s_h1_b4`), i.e. a
   real v3.5 comparison, not a smoke test. On the periodic dump (accuracy / MAE in requests/s; the
   DBN's MAE is its `dbn_rec_median` read-out, the same quantity part 2 §11 reports):

   | Setting | recommended DBN | TabPFN v3.5 | TabPFN v2, same cells |
   |---|---|---|---|
   | 1 s, 10 bins | **0.9532** / 4.50 | 0.9465 / 4.01 | 0.940 |
   | 1 s, 20 bins | **0.9266** / 3.85 | 0.9151 / 4.01 | 0.902 |
   | 1 s, 50 bins | **0.8932** / 3.64 | 0.8716 / 4.01 | 0.855 |
   | 30 s, 20 bins | 0.8685 / 10.29 | **0.8705** / **6.75** | 0.848 |
   | 30 s, 4 bins | 0.9602 / 11.43 | **0.9673** / **6.75** | 0.947 |

   At 1 s the DBN's bin accuracy still leads, but by less than against v2 and with a log-loss that
   is no longer better (0.2560 against 0.2589 at 20 bins); at 30 s **v3.5 beats the recommended DBN
   on both accuracy and MAE**. Part 2 §11's "with the recommended configuration the DBN is ahead of
   TabPFN in bin accuracy in all four settings" holds for v2 only, and the gap between the two
   TabPFN versions is of the same order as the effect the report is arguing about.

   The docs say v3.5 was not run in three places (`docs/HANDOVER.md:217`,
   `docs/2026-10-07_fixes_ablation_tabpfn.md:525-526`, part 2 §6) and list running it as the
   outstanding next step. This is the single most consequential documentation gap in the repo: the
   honest headline for v3.5 is "the DBN is level at 1 s and behind at 30 s".
   (A second dataset's v3.5 file, `..._20260316_105011_..._run20261007_210257_...`, and a
   `whatif` v3.5 run appeared *while this audit was being written* — both were still being written,
   so I did not use them. Note that work is evidently still happening in this working tree.)
3. The MAE column in report §5.9 and the "persistence 4.65 / DBN 5.85" lines are `aeh`
   (the change-anchored read-out), not `ae`. The raw `ae` for the same model is 8.40 against 8.95
   for persistence (1 s / 20 bins) — the table is correct as computed but is labelled "MAE" and the
   text ("The DBN's forecast is turned into a number as 'current value + expected change of bin
   value'") describes `aeh`; the two are not the same and the surrounding comparison against
   TabPFN/gradient-boosting MAE mixes them.
4. Section numbering: report §5.10 does not exist; part 2's §13 sits after §12 but §8 ("Open")
   follows §13; §7 of part 2 is referenced from §6. `results/ablation/summary_<study>.csv` is
   written with an underscore (`summary_ofat.csv`). HANDOVER §7's summary table for §5.8 has an
   extra `|` in the header row.
5. 1 s / 20 bins MAE for persistence appears as 4.65 in report §5.9, 4.51 in §7 and 4.65 in part 2
   §11; the AR table appears as 5.93 (§5.9) and 5.80 (§7). These are the same quantity computed
   on different sample subsets (all samples vs TabPFN's 2 000-per-fold subsample) — the 1 s
   comparison only scores ~6 % of each fold (2 000 / 33 558), on which the reference DBN is
   slightly optimistic (0.9171 vs 0.9154 accuracy). Worth stating explicitly.

### K — `ensure_self_loops` contradicts the exogenous rule (minor)

`dbn/dbn_structure.py:219-243` says exogenous variables "take no parent other than their own
previous value", and the two-slice blacklist bans `a_t → b_t1` when `b` is exogenous — i.e. it
bans `cores_1(t) → cores_1(t+1)`. But `build_dbn_model_2s:538` then calls
`ensure_self_loops(df_ready.columns, inter, target=target)`, which force-adds exactly that edge for
the target (verified: `ensure_self_loops(cols, [], target='throughput_3')` returns
`[('throughput_3_t', 'throughput_3_t1')]`, and with `target=None` it would add one for every
variable, exogenous included). In `dbn_sweep.py` the blacklist therefore never holds, and an
exogenous `t→t+1` edge ends up in the fitted graph. It changes nothing measurable (exogenous `t+1`
is always supplied as evidence) but the reported invariant in report §5.12 ("Cores, data quality
and `rps` have no parent other than their own previous value") is about a graph that the code then
adds an edge to — the sentence happens to be about the *learned* search, but it will read as a
property of the fitted model.

### L — `evaluate()`'s tautology (minor; cause of A)

`dbn/dbn_sweep.py:1006`: `if parents and parents <= set(ev_nodes)`. `ev_nodes` is a dict whose keys
are exactly `<c>_t` for the columns of `test_df`, and `parents` is a subset of those names, so the
condition is always true. The intended reduction ("when every parent of the target is in the
evidence, the rest of the evidence cannot change the answer") is correct but is not what is being
tested. It is harmless for the model built by `build_dbn_model_2s` (extra `t`-slice evidence cannot
change the answer with only inter-slice edges into `target_t1`) and is what makes the AR baseline
misbehave (A).

### M — Roll-out table row/column confusion (cosmetic)

Part 2 §2's rows are correct as tabulated — `rollout:local` 0.934/0.911/0.880/0.849/0.765,
`rollout:level` 0.929/0.909/0.884/0.866/**0.821** (recomputed from the per-run files), and the
level row really does improve at long horizons (30 s: accuracy 0.765 → 0.821, log-loss 0.826 →
0.701, MAE 20.4 → 16.3). The prose is what is wrong: "0.765 → 0.821 at 30 s, 0.849 → 0.866 at
10 s" is written as a comparison against `rollout:local`'s values, and the text says the level
variable "halves the loss" while quoting accuracy. The CSVs also contain a `rollout:level?`
variant that the report never mentions.

---

## 3. Things I could not verify

- TabPFN's internal `crit.cdf` / median semantics for the installed `tabpfn 9.1.0` (I did not run
  TabPFN — the model weights require network access). The bin-probability construction
  (`tabpfn_baseline.py:96-102`) is the right discretisation of a continuous predictive density onto
  `searchsorted(..., side="right")` bins, including the half-count correction, and the delegated
  check measured it: on the constant-load dump 31 % of samples sit exactly on a bin edge and the
  correction is what puts ~0.51 of the mass on the correct side instead of an exact 0.48/0.48 tie.
  The v3.5 file above is also unaudited; treat its numbers as provisional.
- The pre-correction TabPFN numbers quoted in §7 ("0.57–0.83 on that dataset"): the uncorrected
  code is not in git history (the file first appears in `600064d` already corrected), so this can
  only be checked mechanistically.
- `joint_dbn.py`: hard-coded to `share/metrics/dbn_wide_20260310_160246_seed161312.csv`, which is
  not in this repo, so it cannot run. The handover lists it as "untouched"; it is effectively dead
  code and should be archived or repaired.
- `analysis/lstm_baseline.py` on the other two dumps (periodic re-run reproduces the report's
  numbers; the saved CSV for the other two is the stale one from finding G).
- Part 2 §1's "about a quarter of reconfigurations do not move `throughput_3` out of its bin" — no
  result file records it, and it needs a per-sample bin comparison the detection code does not save.
- `dbn/ablation.py`, `tabpfn_baseline.py` and `adaptation_study.py` were not re-run end to end in
  this audit (only their saved summaries cross-checked, plus my own reproduction of the reference,
  best and cross-run configurations).

---

## 4. Suggested repair order

1. Fix `autoregressive_dbn_baseline`'s fit frame and the `evaluate()` parent test (findings A, L),
   then re-run the 1 s sweep grid and re-derive the 30 s summary (finding B) and commit the outputs.
2. Make the detection tables honest (finding E): calibrate the 1 %-false-alarm threshold on a
   held-out half of the ordinary steps, as the CUSUM code already does.
3. Decide the TabPFN protocol for the recommended configuration (finding C): equal candidate sets,
   per-model `info`, correct the `parents` column.
4. Write up the TabPFN v3.5 comparison that already exists on disk (finding J) — including the
   30 s result where it wins — and correct the three places that say it was not run.
5. Re-derive the pooled intervals with the schedule as the resampling unit (finding D), and either
   implement Holm in `tabpfn_report.py` or delete the claim that it is applied.
6. Re-run the roll-out study with `capacity=False` for the non-capacity structures (finding F).
7. Flush sweep results per configuration and persist the best-per-bins tracker (finding I).
8. Regenerate or delete the stale `markov_test_results.csv` / `lstm_baseline_results.csv`
   (finding G); correct the `ae`/`aeh` labelling, the overstated sentences and the "exogenous
   variables take no parent other than their own previous value" statement (findings J, M, K).
9. Commit before producing anything that goes in the paper, so the `-dirty` marker disappears.
