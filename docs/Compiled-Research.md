# Define and run the experiment

The research definition determines what can vary, which observations can fit state, and which result selects a candidate. Python Research expresses those decisions in ordinary Python. Compiled Research expresses them in YAML and executes them with the CLI.

The [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) are independent starting points. Their models, units and assessment populations differ. Running both does not turn their scores into a controlled comparison.

## Prerequisites

Use the matching Limen wheel, its recorded input bundle and the extras stated on each project page. Run Compiled commands from the bundle root. Python Research supplies its own data and functions; its complete project gives the corresponding entry point.

## The declaration connects the research decisions

The feature, target, learner, and execution policy jointly define an experiment. A feature expresses information available at a decision time. A target supplies the observed outcome. The learner fits their relationship, and the execution policy determines how its output requests a position.

The preceding chapters develop [features](Feature-Design.md), [targets](Target-Design.md), [model families](Model-Selection.md), and [rule strategies](Rule-Research.md). This chapter connects those decisions in the complete research definition. The [workflow table](Experiment-Manifest.md#supported-research-workflows) identifies additional requirements for recovery, workers, folds, and promotion.

The three chronological populations serve different roles:

| Population | State or decision it can supply | Consequence |
|---|---|---|
| Training | Fitted targets, scalers, PCA, and model parameters | Later observations apply the retained fitted state. |
| Validation | Candidate ranking, probability calibration, and threshold selection | Every choice consumes the declared validation evidence. |
| Test | The later result measured under the chosen definition | Reading this result to revise the definition consumes it as research evidence. |

A reserved final assessment adds a separately protected population after those research choices. Its reservation precedes research, as developed in [final holdout](Final-Holdout.md).

## Own the Python experiment

The Python project supplies data, a parameter space and ordinary preparation and model functions to UEL. Its model function fits one candidate, scores validation and retains that fitted object. Selection wraps those same evaluated objects in Sensors and combines compatible forecasts in a Cohort.

The complete [selection function](examples/python-research/research.py) keeps each of those decisions visible. The [SFD reference](Single-File-Decoder.md#custom-sfds) describes its function signatures; the [UEL reference](Universal-Experiment-Loop.md#direct-python-run) describes execution, recording and recovery.

<!-- example: book-definition-python source=docs/examples/python-research/research.py region=experiment scenario=selection interface=python role=executed -->
```python
def run_research(bars, experiment_name):
    research = bars[:1440]
    uel, retained = run_trials(research, experiment_name)
    selected = choose_parameters(uel)
    sensors = selected_sensors(bars, selected, retained)
    cohort = Cohort.from_python(sensors)
    return cohort, uel, selected, retained
```

## Declare the Compiled experiment

`compiled-research/research.yaml` is the complete, composable worked declaration. It compares regularization, feature ablation and optional PCA over eight candidates. Incompatible decisions have separate complete YAML files: rule-based models, continuous targets, alternate bars, walk-forward folds and guarded final assessment.

Run from the recorded bundle root with its matching wheel and `data`, `stats` and `boosting` extras installed. The source paths below refer to the bundle's pinned `inputs/` directory.

### Fix the source and partitions

<!-- example: book-declaration-source source=docs/examples/compiled-research/research.yaml region=source scenario=search interface=compiled role=executed -->
```yaml
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_15m_20250101_20250531.parquet
```
<!-- example: book-declaration-partitions source=docs/examples/compiled-research/research.yaml region=partitions scenario=search interface=compiled role=executed -->
```yaml
    split_dates:
      train_start: "2025-01-01"
      train_end: "2025-01-15"
      val_start: "2025-01-15"
      val_end: "2025-01-18"
      test_start: "2025-01-18"
      test_end: "2025-01-22"
      val_predict_guard: true
      test_predict_guard: true
```

Dates are half-open UTC windows. Fit target and transform state on training; validation chooses among candidates and tunes the declared threshold. The test partition contributes neither choice. Reusing validation for calibration and selection consumes that evidence; it is not a fresh assessment.

### Define the information the model receives

<!-- example: book-declaration-features source=docs/examples/compiled-research/research.yaml region=features scenario=search interface=compiled role=executed -->
```yaml
    features:
      - func: limen.features.lag_range
        params:
          col: roc_1
          start: 1
          end: 3
          group: momentum
      - func: limen.features.fractional_diff
        params:
          d: 0.4
          cols: [close]
          threshold: 0.05
          group: price
```

Indicators add three past-return horizons; the features above add past-return lags and fixed-order fractional differentiation. Feature groups identify which added columns enter the model. The target shifts the next return into a train-fitted binary label. Its fitted cutoff is reused on later partitions.

<!-- example: book-declaration-preparation source=docs/examples/compiled-research/research.yaml region=preparation scenario=search interface=compiled role=executed -->
```yaml
    scaler:
      class: limen.scalers.RobustScaler
    feature_ablation:
      drop_count_key: feature_drop_count
      seed_key: feature_drop_seed
    pca_compression:
      enabled_param: auto_pca
      n_components_param: pca_k
```

Robust statistics and PCA fit on training. Ablation has its own explicit seed. PCA and feature dropping are candidate choices, not claims that either improves future performance. [Targets](Targets.md), [features](Features.md), [scalers](Scalers.md) and [PCA](Perturbation-Strategies.md) give the parameter and boundary details.

### Declare calibration and the objective

<!-- example: book-declaration-calibration source=docs/examples/compiled-research/research.yaml region=calibration scenario=search interface=compiled role=executed -->
```yaml
    calibration:
      probability_calibration:
        func: limen.calibration.sklearn_probability_calibrator
        params:
          method: sigmoid
      threshold_function:
        func: limen.calibration.grid_threshold_optimizer
        params: {}
```
<!-- example: book-declaration-objective source=docs/examples/compiled-research/research.yaml region=objective scenario=search interface=compiled role=executed -->
```yaml
    objective:
      metric: backtest_total_return
      direction: maximize
```

The classifier fits on training. Probability calibration and threshold choice use validation. The objective names validation event-ledger return explicitly; the same field later ranks candidates. [Calibration](Calibration.md) explains what the four on/off modes measure.

### Measure the decision under an execution policy

<!-- example: book-declaration-execution source=docs/examples/compiled-research/research.yaml region=execution scenario=search interface=compiled role=executed -->
```yaml
    backtest:
      source_interval_seconds: 900
      fee_bps: 7.0
      slip_bps: 3.0
      notional_rate: 0.5
      execution_lag_seconds: 0.0
      product:
        kind: cash_spot
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1.0e-9
        min_notional: 0.0
```

The declared product, source clock, fees, slippage and exposure determine the ledger. They are scenario assumptions. [Backtest](Backtest.md) develops their timing and arithmetic, including separate recorded-funding and allocation variants.

### Bound and record the search

<!-- example: book-declaration-parameters source=docs/examples/compiled-research/research.yaml region=parameters scenario=search interface=compiled role=executed -->
```yaml
  params:
    C: [0.5, 1.0]
    random_state: [42]
    feature_groups: [momentum|price]
    feature_drop_count: [0, 1]
    feature_drop_seed: [7]
    auto_pca: [false, true]
    pca_k: [2]
    use_calibration: [true]
    use_threshold: [true]
```
<!-- example: book-declaration-run source=docs/examples/compiled-research/research.yaml region=run scenario=search interface=compiled role=executed -->
```yaml
uel:
  output_path: compiled-research
  n_permutations: 8
  workers: 1
  search_strategy:
    type: grid
    shuffle: true
    seed: 17
  prep_each_round: true
  checkpoint_interval: 1
  feedback_interval: 1
  output_format: parquet
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
```

Eight candidates exhaust this finite grid. The search seed controls traversal; it does not replace a model seed. All recording groups are explicit so the artifacts retain predictions, alignment, ledgers and fold detail. Checkpoints preserve search state for CLI recovery.

<!-- example: book-declaration-command source=docs/examples/commands.json region=compiled-research scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/research.yaml
```

The CLI writes the unchanged definition, `metadata.json`, `results.csv`, `results.parquet`, compressed round records and checkpoint/audit state under `results/dev/compiled-research`. A round with a pipeline error does not count as a successful research result.

### Carry the selected behavior into inference

<!-- example: book-declaration-assessment source=docs/examples/compiled-research/research.yaml region=assessment scenario=search interface=compiled role=executed -->
```yaml
assessment:
  metric: val_backtest_total_return
  direction: maximize
  top_k: 2
  predict: true
```
<!-- example: book-declaration-assess-command source=docs/examples/commands.json region=compiled-assess scenario=search interface=compiled role=executed -->
```bash
limen assess compiled-research/research.yaml --results results/dev/compiled-research
```

Assessment selects two finite validation results, breaking metric ties by recorded identity. Trainer reconstructs those rounds under their original declaration and checks replayed metrics. The resulting Sensors retain fitted preprocessing and calibration; the Cohort combines their compatible outputs. `assessment.json` retains membership and `predictions.parquet` retains decisions, unavailable reasons and native availability values where supplied.

This reconstruction is a checked refit, rather than restoration of arbitrary saved weights. [Trainer](Trainer.md) and [Cohort](Cohort.md) document the artifact and inference contracts. Their optional Python APIs are reference material; the Compiled workflow remains YAML and CLI.

## Validation establishes the declared contract

Schema version `1.0` identifies the configuration language independently of the package version. Structural validation rejects unknown fields, incompatible types, broken references, and unsupported combinations. Native validation also checks callable binding, consumed search parameters, chronology, and component contracts. An editor schema supplies structural assistance while native validation establishes those additional checks.

Profiling loads real observations and samples preparation and fitting. Its completed and failed sample counts determine whether the estimate contains successful work. A zero-error process exit alone can accompany a profile with no successful samples. Dry-run checks compilation and setup without executing the candidate campaign.

Changing source, parameters, target geometry, or execution economics creates a changed research definition. The CLI retains the exact declaration used by the experiment and requires that declaration for assessment. Project storage and lineage preserve the relationship between successive definitions, as developed in the [complete workflow](End-to-End-Workflow.md).

## Continue the question

Continue to [controlled perturbations](Robustness-Research.md) to compare changes to the declared population, features, representation, target, and learner.

Use [search](Advanced-Search.md) to change traversal, [feedback](Reducers-And-Feedback.md) to bound the remaining work, and [walk-forward](Walk-Forward.md) to repeat the decision over later windows. Reserve final observations before research with the separate [guarded YAML project](Final-Holdout.md).

The [Manifest appendix](Experiment-Manifest.md) keeps complete field and component API detail. The [complete workflow](End-to-End-Workflow.md) connects each declaration to its native result files and current execution evidence.
