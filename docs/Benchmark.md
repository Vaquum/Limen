# Assess the evidence behind a score

In Limen, benchmark is the prediction-quality layer between the raw experiment log and the trading backtest.

It measures signal activity, positive-class accuracy, and true-positive versus false-positive outcome separation. Benchmark comes before PnL compression so prediction quality remains visible before a trading rule is applied.

## Prerequisites

- a completed experiment log with predictions and targets
- retained round artifacts (`post_processing=True`) for direct UEL analysis
- a numeric outcome column such as `price_change` for return-separation fields

## Risk boundary

Benchmark output is research evidence, not investment advice, trading advice, regulatory approval, or a promise of future performance. Past performance is not predictive, digital-asset trading can result in total loss of capital, and a benchmark table can show statistical structure without proving that a strategy survives live execution, fees, slippage, or portfolio constraints.

## Where benchmark lives

Benchmark analytics are built on top of `Log`.

The confusion tables diagnose prediction quality; this is not an independent public benchmark suite and does not publish a leaderboard. Walk-forward sweeps additionally retain test return tracks and produce the selection report described below. The report assesses the evaluated sweep, without providing independent evidence of future performance.

The main surfaces are:

- `uel.experiment_confusion_metrics`
- `uel._log.experiment_confusion_metrics('price_change')`
- `uel._log.permutation_confusion_metrics(x='price_change', round_id=0)`

With `post_processing=True` and compatible native retained results, the first two expose the same explicit row analysis. Calling `Log.experiment_confusion_metrics()` without `x` instead reads captured metrics and preserves fraction units. It is a separate view; see [Log](Log.md#benchmark-surfaces).

## Benchmark workflow

The [Python project](examples/python-research/README.md) controls its validation
score in ordinary Python. Its classifier scenario separately retains labels,
probabilities and decision thresholds for all four calibration/tuning modes.
The [Compiled project](examples/compiled-research/README.md) records the native
classifier metrics and uses a declared validation-return objective. A caller's
continuous return score does not acquire binary confusion semantics merely by
entering the same loop.

### Python Research

<!-- example: benchmark-python-calibration source=docs/examples/python-research/scenarios.py region=calibration scenario=calibration interface=python role=executed -->
```python
def calibration(data_dir, output):
    bars = quarter_hour(data_dir)
    params = {'window': 4, 'alpha': .1}
    train, validation = project.prep(bars, params).values()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, random_state=42)).fit(train.select('momentum'), train['next_return'].to_numpy() > 0)
    modes = []
    for calibrate in (False, True):
        fitted = sklearn_probability_calibrator(model, validation.select('momentum'), validation['next_return'].to_numpy() > 0, method='sigmoid') if calibrate else model
        probabilities = fitted.predict_proba(validation.select('momentum'))[:, 1]
        for tune in (False, True):
            threshold, score = grid_threshold_optimizer(validation['next_return'].to_numpy() > 0, probabilities, metric=balanced_metric) if tune else (.5, None)
            state = SimpleNamespace(model=fitted, threshold=threshold)
            sensor = Sensor.from_python(fitted_state=state, params=params, infer=infer_classifier, map_output=lambda rows: rows, state_id=f'sigmoid={calibrate};threshold={tune}', fitting_cutoff=bars['datetime'][960], selection_cutoff=bars['datetime'][1440], output_id='next_bar_up')
            cohort = Cohort.from_python([sensor], aggregate='vote')
            tape = cohort.predict_all(bars[1428:1920])[12:]
            if any(point.prediction not in (0, 1) for point in tape):
                raise ValueError('Classifier inference must provide a binary decision for every assessment row')
            modes.append({'calibrated': calibrate, 'tuned': tune, 'threshold': threshold, 'validation_score': score, 'validation_probabilities': probabilities.tolist(), 'validation_labels': (validation['next_return'].to_numpy() > 0).astype(int).tolist(), 'inference_rows': len(tape), 'positive_predictions': sum(point.prediction == 1 for point in tape)})
    return {'modes': modes, 'validation_reused': True}
```

### Compiled Research

<!-- example: benchmark-compiled-calibration source=docs/examples/compiled-research/scenarios.py region=calibration scenario=calibration interface=compiled role=executed -->
```python
def calibration(data_dir, output):
    config = configuration('calibration', data_dir)
    loop = execute(config, output / 'experiment')
    sensors = Trainer(output / 'experiment').train(research_project.completed_log(loop)['id'].to_list())
    return {'modes': research_project.completed_log(loop).select('use_calibration', 'use_threshold', 'optimal_threshold', 'val_score', 'accuracy').to_dicts(), 'reconstructed_sensors': len(sensors), 'validation_reused': True}
```

These executed functions use helpers from their complete projects. The tests
recompute classifier decisions and the validation threshold score. Use the
source-bound `Log` reference below when inspecting native retained row evidence.

<!-- reference: benchmark-round-confusion source=limen/log/_permutation_confusion_metrics.py region=signature:permutation_confusion_metrics role=signature -->
```python
def permutation_confusion_metrics(self: Any, x: str, round_id: int, *, pred_col: str='predictions', actual_col: str='actuals', proba_col: str | None=None, threshold: float=0.5, outlier_quantiles: Sequence[float]=(0.01, 0.99), outlier_mode: str='filter', id_cols: dict[str, Any] | None=None) -> pd.DataFrame:
    ...
```

Use the experiment-wide table to compare rounds. Use the single-round table when one permutation deserves a closer look.

## What benchmark measures

The current benchmark table focuses on long-only prediction quality and outcome separation.

Key fields include:

- `pred_pos_rate_pct`
- `actual_pos_rate_pct`
- `precision_pct`
- `recall_pct`
- `pred_pos_count`
- `tp_count`
- `fp_count`
- `tp_x_mean`, `tp_x_median`
- `fp_x_mean`, `fp_x_median`
- `tp_fp_cohen_d`
- `tp_fp_ks`

When `x='price_change'`, the table reports the long-call rate, realized long-class accuracy, and realized price-change separation between true positives and false positives.

## How to read it

### Signal rate

`pred_pos_rate_pct` reports signal activity. High precision with a low positive rate can leave too few candidate bars for downstream use.

### Precision and recall

- `precision_pct` is the share of predicted positives that were true positives.
- `recall_pct` is the share of actual positives captured by the model.

Interpret precision and recall together. High precision with low recall means a selective signal. High recall with low precision means a noisy signal.

### TP versus FP quality

Limen's benchmark layer extends precision and recall with outcome-separation metrics.

- `tp_x_mean` and `tp_x_median` describe the chosen outcome inside true positives
- `fp_x_mean` and `fp_x_median` do the same for false positives
- `tp_fp_cohen_d` and `tp_fp_ks` estimate how separated those two distributions are

If TP and FP are not separated on the chosen outcome, a round can be statistically correct while adding little downstream signal value.

## Benchmark versus backtest

Benchmark and backtest are intentionally separate:

- benchmark asks whether the signal has predictive structure
- backtest asks whether that structure survives a concrete long-only trading rule with costs

This separation matters because the layers can disagree.

Common cases:

- a round can have high benchmark metrics but low trading economics
- a round can have modest benchmark metrics yet produce usable backtest behavior because of position profile and cost structure

Limen exposes both layers so one score does not hide either prediction quality or trading economics.

## Walk-forward selection report

The [Compiled project's recorded acceptance case](Experiment-Manifest.md#recorded-acceptance-example)
executes two rule-based trials and retains their complete return tracks and reports.
It runs inside the project's walk-forward scenario:

<!-- example: benchmark-recorded-acceptance source=docs/examples/compiled-research/scenarios.py region=acceptance scenario=walk_forward interface=compiled role=executed -->
```python
def acceptance(data_dir, output):
    config = configuration('acceptance', data_dir)
    loop = execute(config, output)
    report = json.loads((output / 'acceptance_report.json').read_text())
    return {'rounds': research_project.completed_log(loop).height, 'report': report,
            'artifacts': {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in (
                'trial_returns.parquet', 'acceptance_report.json', 'acceptance_report.md', 'round_data.jsonl.gz')}}
```

Its independent proof derives DSR and PBO from the retained returns and ordered
fold timestamps. These are native sweep reports. The Python project owns its
selection criteria and return ledger; it does not acquire native report artifacts
merely by using UEL or MSQ.

[Manifest-backed walk-forward validation](Experiment-Manifest.md#walk-forward-sweeps) fits each fold independently, purges rows before validation and test, and excludes embargoed rows from later training pools. Choose the purge to cover the target's forward label horizon. Each successful trial retains its test net-return track in `trial_returns.parquet`.

At sweep conclusion, Limen reads that artifact and writes `acceptance_report.json` and `acceptance_report.md` beside it. Without the artifact, no report is written. The report computes two statistics over the recorded per-trial out-of-sample return matrix:

- **Deflated Sharpe probability:** test the selected trial's observed per-bar Sharpe against the expected maximum from the evaluated trial count and trial-Sharpe variance, with skewness and kurtosis corrections. This applies equation 2 of [Bailey and López de Prado (2014)](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
- **Probability of backtest overfitting (PBO):** partition the return matrix into contiguous blocks and evaluate every balanced in-sample/out-of-sample block combination. Select the in-sample Sharpe winner, rank it out of sample, and count how often its relative rank falls in the bottom half. This follows the CSCV construction in [Bailey, Borwein, López de Prado and Zhu (2015)](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

The report's winner has the highest full-track per-bar Sharpe; exact ties prefer persisted trial order. The successful track count and the sample variance of trial Sharpes supply DSR's deflation inputs. PBO requires identical recorded test timestamps in each fold across trials; missing or different row evidence makes it unavailable. Its default two contiguous equal blocks evaluate both balanced combinations, giving only `0`, `0.5` or `1`. Optional `acceptance.pbo_blocks` selects 2, 4, 6 or 8 before the run; this configuration/metadata key becomes report `n_blocks`. Eight yields 70 nominal combinations and step `1/70`. The public PBO function remains independently configurable beyond the report cap.

The Sharpe units are per bar, with no annualization. The report uses net returns already recorded by execution; it adds no position replay or new cost assumptions. Optional [acceptance thresholds](Experiment-Manifest.md#acceptance-thresholds) define statistical acceptance gates whose verdicts appear in the report. A failed threshold verdict records a research result and never aborts the run. Without thresholds, the report still contains the statistics. Degenerate or unavailable statistics are reported as `null` with an explanation; any declared verdict using them is also `null`.

The partition is frozen; thresholds remain caller-supplied on regeneration, as today. Reassess `max_pbo` before changing S: a threshold is relative to partition resolution, so the same numeric threshold need not express the same selection criterion. Omitted or thresholds-only direct mappings inherit saved S; omission drops verdicts. Count conflicts and invalid saved counts raise before report writes. See [acceptance configuration](Experiment-Manifest.md#acceptance-thresholds) for admission and legacy rules.

The JSON and Markdown show `pbo_combinations`, `pbo_resolution` (nominal counting step), and `pbo_block_bars` (null only for invalid/unknown geometry). These are geometry, not proof that enumeration succeeded. Blocks partition stitched retained rows and may cross fold boundaries without extra purge/embargo; each half still has `n_bars/2` observations. More combinations do not add independent evidence. Sparse rule-based trials can keep finite full-track DSR while finer half samples have zero variance. A finer partition can make PBO unavailable when a half sample has zero variance; the chosen count is not changed to rescue a verdict. Null is not pass.

The public metrics are `limen.metrics.deflated_sharpe_ratio(returns, n_trials=..., trial_sharpe_variance=...)` and `limen.metrics.probability_of_backtest_overfitting(returns_matrix, n_blocks=...)`; the matrix has one row per trial and one column per bar. Sharpe uses the arithmetic mean divided by sample standard deviation (`ddof=1`). DSR requires at least four finite returns with positive variance. Its skewness and Pearson kurtosis use centered returns standardized by population standard deviation; excess kurtosis is Pearson kurtosis minus three. The supplied trial-Sharpe variance must be finite and nonnegative. One trial or zero trial-Sharpe variance sets the comparison benchmark to zero.

PBO requires at least two trials, an even block count of at least two, exact divisibility of the track length, and at least two bars per block. Every trial must have positive finite variance in every compared half. In-sample ties select the first input trial. Out-of-sample ties use average ascending ranks divided by `n_trials + 1`; relative ranks at or below `0.5` count toward PBO. Degenerate inputs raise rather than producing a hollow probability.

These statistics reuse the sweep's test observations to assess selection. They are not a second untouched holdout. The trial count includes successfully recorded tracks in this artifact; it omits failed trials and earlier unrecorded searches and estimates no effective independent trial count. Purging and embargo depend on the declared geometry and cannot repair a feature that already reads future data. DSR's trial-count approximation and moment correction do not establish independence, stationarity or a calibrated forecast of live success. CSCV redistributes recorded blocks and is not a new chronological walk-forward evaluation. Correlated trials, serial dependence and changing market regimes limit interpretation. The report provides no formal research falsification proof; passing verdicts establish neither causal validity nor future profitability.

### Comparing a recorded experiment family

`limen.experiment.family_acceptance_report.family_acceptance_report` assesses explicitly supplied walk-forward runs together. Supply existing experiment-directory `Path` objects and the project containing their committed-manifest store. The current reference signature is:

<!-- reference: benchmark-family-report source=limen/experiment/family_acceptance_report.py region=signature:family_acceptance_report role=signature -->
```python
def family_acceptance_report(directories: Sequence[Path], *, project_root: Path, n_blocks: int=2) -> dict[str, object]:
    ...
```

The returned dictionary is JSON-serializable and reads artifacts without modifying them. `runs` lists resolved source directories, recorded **run-start** times, runtime geometry, recording flags, implementation versions and available committed-manifest ancestry. Known starts sort ascending, unknown starts last, then resolved directory and persisted logical trial order. This report order governs winner ties and evidence-match references; it does not establish global trial occurrence or when results were inspected. Missing ancestors remain explicit, and the manifest index is not authoritative.

Retain the following profile when recording runs intended for this report:

Enable `uel.record_execution`, `uel.recording.alignment` and `uel.recording.fold_details` together. The complete [walk-forward project definition](examples/compiled-research/walk_forward.yaml) declares those settings.

The normal YAML runner passes these settings to execution. For a direct `UniversalExperimentLoop.run()` call, pass the same `record_execution` and `recording` arguments explicitly. Default lean recording omits folds/alignment; folds without execution recording omit snapshot market observations. Those trials remain in the inventory with unavailable reasons and their actual recording flags. Valid per-run statistics do not imply sufficient cross-run evidence.

`comparisons` groups equal ordered per-fold test timestamps and retained evaluation observations within one return convention. Snapshot groups require exactly matching nullable `market.ret`; initial nulls stay null. These are close-observation groups: the artifacts do not retain complete open/high/low identity, so **snapshot joint PBO is unavailable**, including barrier execution. Signed groups instead require matching retained execution-source bindings, rule/accounting conventions, product/currencies, ordered signal availability and partitions. Complete contract digests are validated separately; the full digest is not a comparison key, so searched fee/slippage changes remain eligible. Missing projected alignment is unavailable, not a corrupt complete contract. Source-hash mismatches leave comparability unproven across unknown dependency environments.

Declared sources are unverified context, not proof of consumed data. Fetch-window declarations do not automatically split matching retained observations. Each trial exposes its recorded configuration and runtime geometry; model, feature, target and cost variations remain alternatives when observations match. Lookback warm-up and target horizons can change retained rows and form separate groups. Recorded missing-row evidence is shown without inventing the cause. No intersection, padding, replay or timestamp reconstruction occurs.

Every source-qualified retained trial stays counted. Equal configuration, recorded version and results do not prove identical training inputs, fits or implementations, including `dev` and editable installs. `evidence_match_of` names the first matching recorded result in report order; it never removes that alternative. `repeat_of` remains null when true repeat identity is unproven. Duplicate directory aliases are listed and read once.

A group's DSR is available only when it contains **every supplied retained successful track**, all supplied runs have known track coverage, and at least two alternatives remain. Fragmented or incomplete pools receive no DSR rather than a more favorable partial selection correction; singleton groups receive no undeflated substitute. The full-track Sharpe winner, sample trial-Sharpe variance and member count remain visible. Flat candidates are retained and produce named metric errors. Supported signed PBO uses the recorded synchronous matrix; a PBO-only failure does not erase supported DSR. `n_blocks` must be an even integer of at least two, divide `n_bars` exactly and leave at least two bars per block; its default two blocks give only `0`, `0.5` or `1`. Family `n_blocks` is independent and never inherits per-run `pbo_blocks`; even a single-run family can differ from its source report.

Adjacent group counts are `n_family_trials`, `n_group_trials`, `n_trials_without_group`, `n_trials_outside_group` and `n_groups`. `coverage` distinguishes retained tracks, evidence matches, missing/incompatible evidence, known CSV failures and runs without terminal inventory. Existing CSV successes are reconciled with round records and return tracks; failed trials never receive fabricated returns. A directory without metadata raises; a valid run without returns reports unknown track coverage. Shared storage, digest, manifest-identity and return-track integrity failures raise. Present unusable timestamps remain an explained incompatible trial, rather than discarding other runs.

These are conditional assessments of supplied retained successful tracks. Deleted, crashed, pruned and unrecorded attempts remain unknown; failed attempts have no return tracks. Choosing a group adds selection, sequential researcher adaptation is uncorrected, and raw counts are not effective independent counts. Neither lineage nor declared fingerprints prove comparability or authenticate a provider. The report supplies no untouched holdout, leakage guarantee or forecast of profitability.

## Choosing `x`

`permutation_confusion_metrics()` and `experiment_confusion_metrics()` work on a chosen column `x`.

The default analysis column is `price_change` because it is available in the reconstructed prediction-performance table and gives a straightforward economic interpretation.

Other numeric columns are valid when they exist in the reconstructed round table and match the analysis question.

## Outlier handling

Single-round confusion summaries support outlier handling through:

- `outlier_quantiles=(lo, hi)`
- `outlier_mode='filter'` or `'winsor'`

This protects the TP/FP comparison from domination by extreme outcome values.

## Read next

- Continue to [Backtest](Backtest.md) to see how benchmark-quality signals translate into long-only trading economics.
- Continue to [Log](Log.md) for the broader post-run workflow around benchmark, backtest, and parameter correlation.

See [Reserve and assess final observations](Final-Holdout.md) for project research exclusion, frozen fitted-selection assessment, retained scenario evidence and the limits of local exposure history.
