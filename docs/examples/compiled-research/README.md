# Compiled Research project

The Manifest owns preparation and its configured checks. Trainer reconstructs selected recorded rounds and validates their metrics.

The [research chapter](../../Research-Book.md) develops the project. The [documentation contract](../../Developer/Documentation-System.md) defines the proof required before this candidate is published as the complete book.

The search scenario also compares baseline and compressed preparation on 1,440 recorded hourly bars. It retains each prepared train/validation/test matrix, a fixed-order fractional feature, ternary forward-close labels, training-fitted robust statistics, an explicit feature drop and optional two-component PCA. Independent tests reconstruct the labels, differenced values, scaler statistics, PCA subspace and Ridge forecasts. The Python case owns separate partition preparation; the native Manifest case retains its admitted CCO context. These are preparation checks, not a feature-importance ranking or profitability claim.

The walk-forward scenario also runs `acceptance.yaml` against the complete recorded hourly source. It retains two rule-based trials, their 96-return tracks, fold timestamps and JSON/Markdown acceptance reports under `walk_forward/acceptance/`. Independent tests recompute Deflated Sharpe probability and PBO from those tracks and check the declared verdicts. The driver binds the YAML's repository-relative source filename through `--data-dir`; use the complete project when running from the downloaded bundle.

## Prerequisites

Use the project and input bundle identified by the example evidence, Python 3.10–3.13, and its matching Limen wheel. Selection needs the data extra. The full project needs `vaquum-limen[data,stats,boosting]`; Sobol uses statistics; factorization uses LightGBM.

## Run the selection scenario

Install the matching `vaquum-limen[data]` distribution. From the example bundle root, supply the directory containing the recorded input files:

<!-- example: compiled-selection-command source=docs/examples/commands.json region=compiled-selection scenario=selection interface=compiled role=executed -->
```bash
python compiled-research/research.py --scenario selection --data-dir inputs --output-dir results/compiled
```

From a repository checkout, work in `docs/examples/`, use `--data-dir ../../tests/fixtures` and put generated output under `../../.docs-evidence/`. The command checks the recorded file digest before research. It writes `summary.json`, `results.csv`, `metadata.json` and the compressed round artifacts. Trainer reconstructs the selected round from those artifacts and validates the replayed metrics; it does not restore saved model weights.

## Run the declaration through the CLI

From the recorded example bundle root, the same `selection.yaml` can run directly:

<!-- example: compiled-selection-yaml-command source=docs/examples/commands.json region=compiled-yaml scenario=selection interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/selection.yaml
```

The declaration uses the bundle's `inputs/` directory and writes `results/dev/compiled-cli`. The proof compares its two distinct candidate identities and validation returns with the Python driver's recorded run. Use a fresh output directory for a new experiment; the driver command above supplies its output explicitly.

## Continue through the research decisions

Search also executes all eight [common strategy baselines](../../Common-Strategies.md) on the recorded 2024 hourly source. Their desired signals share the same assessment window. The Python case owns next-open-to-close one-bar trades with a declared 2 bp round-trip cost; the Compiled case uses the native lagged held-position snapshot with 10 bp fee and 5 bp slippage per fill. Independent assertions check causal signals and each policy's arithmetic; these are different trading interpretations, not a cross-engine return ranking.

`--scenario all` runs the named scenarios in separate output directories. `--scenario` also accepts one name from the table. Use a new output directory for a fresh run; the recovery scenario creates its interrupted run and resumes it internally.

| Scenario | Decision and output |
|---|---|
| `search` | Compare traversal, preparation, eight common rules, budget reduction and mixed feedback under their declared populations. |
| `recovery` | Interrupt after two rounds, retain a checkpoint, resume without repeating a completed round. |
| `walk_forward` | Keep each test window after its fitting/validation observations, with purge and embargo. |
| `holdout` | Assess final observations only after selection; retain the assessment and its exposure limits. |
| `calibration` | Compare all four calibration/threshold modes and disclose validation reuse. |
| `execution` | Execute recorded observations under explicit costs, product rules and causal clocks. |
| `bars` | Form volume, trade and liquidity bars; omit an incomplete final bar. |
| `allocation` | Preserve rule and learned primaries, fit completed opportunities and cap each opening exposure under recorded execution costs. |
| `command_line` | Validate, profile, store, fork, run and resume a recorded campaign; back it up and restore it through a local Git remote. |

Factorized search uses the admitted deterministic LightGBM grid with independent fee and tree-count axes. The ordinary search modes use logistic regression; these are different declared comparisons. The walk-forward scenario reconstructs qualified folds and binds their complete identities into a Cohort. `allocation` adds batch meta-allocation with completed-opportunity training; it makes no live-promotion or recovery claim.

Traversal and recovery use the recorded January 2024 hourly source. Recovery compares resumed and uninterrupted native results and round records. Walk-forward uses its first 384 hourly observations, with a separate fitted classifier and probability calibration in each fold; validation return supplies the declared objective. The model-family diagnostics keep their quarter-hour population.

The project records its source digest and any supplied row selection in `project-input.json` before execution. It refuses a changed source or selection before resuming. This project check supplements Limen's native checkpoint and fold-declaration validation; it does not assert that the native checkpoint validates an arbitrary replacement DataFrame.

Allocation compares an unchanged rule primary and an unchanged logistic primary with their composed exposures. A separate prefix fit checks the learned primary's recorded out-of-fold identity. Training rows retain completion times, unavailable outcomes and exact source bindings. Each opportunity keeps its opening allocation until primary flat; the risk cap includes the declared stop and round-trip fee/slippage allowance. Paired January 1–8 cases isolate recorded funding on the same opportunity identities. These are batch research results; the allowance does not bound loss through a price gap.

`adaptive.yaml` adds a separate grid with a three-round resource limit declared through the budget reducer. The driver compiles its pruning configuration. The audit records the queue intervention, and the checkpoint retains the completed and remaining counts. The four-candidate traversal comparisons retain their original budgets.

Search also runs Ridge, DLinear and XGBoost on next-close returns, a random binary diagnostic baseline, and direction/sizing with both independent and conditional sizing. Each has a separate `family_*.yaml` definition. The regressors share recorded rows and the current return and five past returns; their RMSE is checked against the retained predictions and independently calculated labels. This is an execution comparison, not evidence of trading performance. Direction/sizing uses completed trade outcomes and signed exposure under its declared execution contract; Trainer reconstructs the selected round. The random baseline uses the driver's visible NumPy seed and makes no Trainer replay claim.

The family comparisons use `post_processing=False`. Native results, recording and Trainer reconstruction remain active. Limen's optional legacy post-experiment binary summaries currently fail on these continuous targets, tracked in [#1079](https://github.com/Vaquum/Limen/issues/1079); this project does not claim support for those summaries. The driver reads the recorded version-2 prediction payloads, checks their digests and counts, and retains their values beside the metrics in its summary.

The execution scenario also runs `funded.yaml` with recorded funding. The driver normalizes the pinned funding JSON into a Parquet source through the shared `recorded_inputs.py`, resolves that derived source's path in the run definition, and retains both the realized rates and mark prices in the funding ledger. No constant funding-rate scenario replaces those records. Both projects' bar scenarios use the same shared helper to turn the recorded 90-second trade slice into three CFR diagnostic bars. That short slice is not presented as a compiled training population.

The holdout scenario creates a separate Git project, reserves its final window before research, freezes selected fitted state and retains the final report. Local recorded files leave earlier exposure unknown. Preserve its project marker, history and assessment artifacts together.

The command-line scenario invokes the `limen` executable installed beside its Python interpreter. It inspects an installed template, then writes the project's recorded run definition, stores a production manifest, forks its lineage and verifies CSV/Parquet output after resume. Backup and `limen new --from` operate on a local Git remote. Every tracked campaign file must restore byte for byte. Git backup honors ignore rules: local runtime leases and guarded holdout history are deliberately excluded and require separate preservation. This scenario makes no live template-download or hosted-backup claim. The original recorded-input paths remain part of the saved definition and must be available for replay.

Real TabPFN model validation remains unproved under [#628](https://github.com/Vaquum/Limen/issues/628).

The JSON summary names the statistics used by each decision. Complete native metrics remain in the experiment CSVs. Some statistics are undefined for short or one-sided populations—for example, tail loss on a short fold—and retain their documented native representation there. Undefined statistics are not substituted with zero.
