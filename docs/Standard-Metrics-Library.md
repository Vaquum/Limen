# Standard Metrics Library

Limen's metrics layer provides the low-level evaluation helpers used inside SFD model functions and reference architectures.

Task helpers return metric dictionaries. The DSR and PBO helpers return scalar advisory statistics with explicit input requirements. The [Python project](examples/python-research/README.md) uses a caller-owned validation return and the shared threshold metric; the [Compiled project](examples/compiled-research/README.md) records classification, regression and execution results through its declared architectures. Higher-level analysis lives in [Log](Log.md) and [Benchmark](Benchmark.md).

## Public Surface

The current public metrics exports are:

- `binary_metrics`
- `multiclass_metrics`
- `continuous_metrics`
- `balanced_metric`
- `safe_ovr_auc`
- `rule_based_metrics`
- `deflated_sharpe_ratio`
- `probability_of_backtest_overfitting`

## Import Pattern

The safest import style is to import the callable from its submodule:

| Callable import | Package-root export |
|---|---|
| `limen.metrics.binary_metrics.binary_metrics` | module |
| `limen.metrics.multiclass_metrics.multiclass_metrics` | module |
| `limen.metrics.continuous_metrics.continuous_metrics` | module |
| `limen.metrics.safe_ovr_auc.safe_ovr_auc` | module |
| `limen.metrics.rule_based_metrics.rule_based_metrics` | module |
| `limen.metrics.balanced_metric.balanced_metric` | callable |
| `limen.metrics.deflated_sharpe_ratio.deflated_sharpe_ratio` | callable |
| `limen.metrics.probability_of_backtest_overfitting.probability_of_backtest_overfitting` | callable |

Package-root imports preserve these module/callable boundaries:

`from limen.metrics import binary_metrics` therefore names the module. Import its `binary_metrics` member when you need the function.

This low-level metrics layer also sits underneath [Reference Architecture](Reference-Architecture.md). The class-based models add confusion and backtest fields later; these helpers stay at the smaller task-metric level.

## `binary_metrics(data, preds, probs)`

Computes the standard binary metrics used by Limen's binary reference models.

Expected inputs:

- `data['y_test']`
- predicted labels `preds`
- positive-class probabilities `probs`

Returns a dictionary with:

- `recall`
- `precision`
- `fpr`
- `auc`
- `accuracy`

Current API:

<!-- reference: metrics-binary source=limen/metrics/binary_metrics.py region=signature:binary_metrics role=signature -->
```python
def binary_metrics(data: dict[str, Any], preds: list[int] | npt.NDArray[np.integer[Any]], probs: list[float] | npt.NDArray[np.floating[Any]]) -> dict[str, Any]:
    ...
```

### Edge cases

`binary_metrics()` handles degenerate binary folds without raising. It returns `NaN` for `auc` when `y_test` contains one class and `NaN` for `fpr` when there are no negative examples. Precision and recall use `zero_division=0`.

## `multiclass_metrics(data, preds, probs, average='macro')`

<!-- reference: metrics-multiclass_metrics source=limen/metrics/multiclass_metrics.py region=signature:multiclass_metrics role=signature -->
```python
def multiclass_metrics(data: dict[str, Any], preds: list[int] | npt.NDArray[np.integer[Any]], probs: list[list[float]] | npt.NDArray[np.floating[Any]], average: str='macro') -> dict[str, Any]:
    ...
```

Computes multiclass classification metrics from:

- `data['y_test']`
- predicted labels
- class probabilities

Returns:

- `precision`
- `recall`
- `auc`
- `accuracy`

This helper uses `safe_ovr_auc()` instead of calling raw multiclass AUC directly.

## `continuous_metrics(data, preds)`

<!-- reference: metrics-continuous_metrics source=limen/metrics/continuous_metrics.py region=signature:continuous_metrics role=signature -->
```python
def continuous_metrics(data: dict[str, Any], preds: list[float] | npt.NDArray[np.floating[Any]]) -> dict[str, Any]:
    ...
```

Computes the current regression metrics from:

- `data['y_test']`
- continuous predictions `preds`

Returns:

- `bias`
- `mae`
- `rmse`
- `r2`
- `mape`

`mape` is reported in percent units.

MAPE divides by the target value. Rows with zero or near-zero financial returns can dominate or invalidate the percentage interpretation; use MAE/RMSE/bias for return series where the denominator can be zero or economically negligible.

## `balanced_metric(y_true, y_pred)`

`balanced_metric()` is Limen's compact binary score for cases where class balance matters.

Current formula:

`precision * sqrt(trade_rate)`

This rewards accurate positive calls while penalizing degenerate behavior such as never trading.

If there are no positive predictions, it returns `0.0`.

Current API:

<!-- reference: metrics-balanced source=limen/metrics/balanced_metric.py region=signature:balanced_metric role=signature -->
```python
def balanced_metric(y_true: Any, y_pred: Any) -> float:
    ...
```

## `safe_ovr_auc(y_true, probs)`

<!-- reference: metrics-safe_ovr_auc source=limen/metrics/safe_ovr_auc.py region=signature:safe_ovr_auc role=signature -->
```python
def safe_ovr_auc(y_true: npt.NDArray[Any] | list[int] | list[str], probs: npt.NDArray[np.floating[Any]] | list[list[float]]) -> float:
    ...
```

`safe_ovr_auc()` computes one-vs-rest AUC more defensively than a raw direct multiclass AUC call.

Its purpose is to make multiclass evaluation more stable when not every class is present in every fold.

If no valid class-vs-rest comparisons can be made, it returns `NaN`.

Boundary: when `probs` contains only the classes present in `y_true`, columns are read in the sorted-present-class order. When `probs` still contains full integer-label columns, the class label itself is used as the probability-column index. If columns cannot be aligned safely, the helper returns `NaN` instead of reading the wrong column.

## `rule_based_metrics`

This helper combines split-specific long/flat position statistics and supplied backtest metric dictionaries. It counts entries and position rates for train, validation and test, carries each split's backtest fields and reports drawdown dispersion when enough values exist. `is_stable` remains `False`; the retained Sharpe-threshold parameters are compatibility arguments, rather than a current acceptance rule.

<!-- reference: metrics-rule-based source=limen/metrics/rule_based_metrics.py region=signature:rule_based_metrics role=signature -->
```python
def rule_based_metrics(positions: dict[str, npt.NDArray[np.integer[Any]]], backtest_results: dict[str, dict[str, float]], sharpe_std_threshold: float=0.5, sharpe_degradation_threshold: float=0.3) -> dict[str, Any]:
    ...
```

## `deflated_sharpe_ratio`

DSR accepts one finite return track with at least four observations, a positive integer trial count and nonnegative finite trial-Sharpe variance. The return track and estimated Sharpe variance must be valid; invalid or degenerate inputs raise. The result is a per-bar probability, rather than an annualized Sharpe or a declaration that the selected strategy is acceptable. One trial uses a zero selection benchmark.

<!-- reference: metrics-dsr source=limen/metrics/deflated_sharpe_ratio.py region=signature:deflated_sharpe_ratio role=signature -->
```python
def deflated_sharpe_ratio(returns: npt.NDArray[np.float64], *, n_trials: int, trial_sharpe_variance: float) -> float:
    ...
```

## `probability_of_backtest_overfitting`

PBO accepts a finite trial-by-synchronous-bar matrix with at least two trials. Choose an even number of equal contiguous blocks, with at least two bars per block. Every half-sample trial needs positive finite variance. In-sample ties select the first trial; out-of-sample ties receive an average rank, and a median tie counts as overfit. Invalid input raises.

The helper returns the share of combinatorial symmetric cross-validation selections at or below the out-of-sample median rank. Cost grows with the block combinations. [Walk-forward selection reports](Benchmark.md#walk-forward-selection-report) define how retained experiment evidence can supply these statistics and how unavailable criteria are reported. Neither scalar function stops a run or creates an untouched holdout.

<!-- reference: metrics-pbo source=limen/metrics/probability_of_backtest_overfitting.py region=signature:probability_of_backtest_overfitting role=signature -->
```python
def probability_of_backtest_overfitting(returns_matrix: npt.NDArray[np.float64], *, n_blocks: int) -> float:
    ...
```

## Where These Helpers Fit

Both interfaces can use these helpers without adopting the other interface's methodology:

The Python calibration scenario calls the shared threshold optimizer with `balanced_metric`; its custom scorer remains caller-owned. The Compiled calibration scenario declares the same metric and trains all four fitted modes. The separate `calibration_assessment.yaml` uses the native event-return objective and reconstructs all four modes through `limen assess`. The metric-comparison campaign and the event-return campaign answer different selection questions. These are the same project sources used in the [calibration chapter](Calibration.md):

<!-- example: metrics-python-calibration source=docs/examples/python-research/scenarios.py region=calibration scenario=calibration interface=python role=executed -->
```python
def calibration(data_dir, output):
    bars = quarter_hour(data_dir)
    params = {'window': 4, 'alpha': .1}
    train, validation = project.prep(bars, params).values()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, random_state=42)).fit(train.select('momentum'), train['next_return'].to_numpy() > 0)
    modes, sensors = [], []
    for calibrate in (False, True):
        fitted = sklearn_probability_calibrator(model, validation.select('momentum'), validation['next_return'].to_numpy() > 0, method='sigmoid') if calibrate else model
        probabilities = fitted.predict_proba(validation.select('momentum'))[:, 1]
        for tune in (False, True):
            threshold, score = grid_threshold_optimizer(validation['next_return'].to_numpy() > 0, probabilities, metric=balanced_metric) if tune else (.5, None)
            state = SimpleNamespace(model=fitted, threshold=threshold)
            sensor = Sensor.from_python(fitted_state=state, params=params, infer=infer_classifier, map_output=lambda rows: rows, state_id=f'sigmoid={calibrate};threshold={tune}', fitting_cutoff=bars['datetime'][960], selection_cutoff=bars['datetime'][1440], output_id='next_bar_up')
            sensors.append(sensor)
            cohort = Cohort.from_python([sensor], aggregate='vote')
            tape = cohort.predict_all(bars[1428:1920])[12:]
            if any(point.prediction not in (0, 1) for point in tape):
                raise ValueError('Classifier inference must provide a binary decision for every assessment row')
            modes.append({'calibrated': calibrate, 'tuned': tune, 'threshold': threshold, 'validation_score': score, 'validation_probabilities': probabilities.tolist(), 'validation_labels': (validation['next_return'].to_numpy() > 0).astype(int).tolist(), 'inference_rows': len(tape), 'positive_predictions': sum(point.prediction == 1 for point in tape)})
    return {'modes': modes, 'decisions': [{key: mode[key] for key in ('calibrated', 'tuned', 'threshold', 'inference_rows', 'positive_predictions')} for mode in modes], 'validation_reused': True, 'composition': compose_selected(sensors, bars)}
```

<!-- example: metrics-compiled-calibration source=docs/examples/compiled-research/calibration.yaml region=calibration_config scenario=calibration interface=compiled role=executed -->
```yaml
    calibration:
      probability_calibration:
        func: limen.calibration.sklearn_probability_calibrator
        params:
          method: sigmoid
      threshold_function:
        func: limen.calibration.grid_threshold_optimizer
        params:
          metric: limen.metrics.balanced_metric.balanced_metric
```

The Compiled project's `search` scenario separately checks retained continuous-family predictions against independently computed RMSE. Multiclass helpers below remain reference APIs; these projects do not claim a multiclass research scenario.

Keep the measured task, evaluation population and output units explicit when adding a metric to either project.

## Read Next

- Continue to [Log](Log.md) for experiment-level analysis built on top of these low-level metrics.
- Continue to [Benchmark](Benchmark.md) for the benchmark layer that uses reconstructed round outputs rather than these raw helper functions directly.
- Continue to [Reference Architecture](Reference-Architecture.md) for the model layer that wraps these task metrics into richer evaluation payloads.

## Callable contracts

These entries give the exact public signatures and the source-owned argument, return and failure contracts. Use the project examples above for the research context; a signature is reference evidence, not a claim that every option ran in the base experiment.

### `limen.metrics.balanced_metric`

<!-- reference: api-limen-metrics-balanced-metric source=limen/metrics/balanced_metric.py region=signature:balanced_metric role=signature -->
```python
def balanced_metric(y_true: Any, y_pred: Any) -> float:
    ...
```

Compute balanced precision metric that accounts for trade rate.

Calculates precision * sqrt(trade_rate) to balance signal quality

with trade frequency. Higher scores indicate better balance between

accurate predictions and sufficient trading activity.

| Argument | Contract |
| --- | --- |
| y_true | Ground truth binary labels |
| y_pred | Predicted binary labels |

**Returns:** float: Balanced metric score (0.0 if no positive predictions)

### `limen.metrics.binary_metrics`

<!-- reference: api-limen-metrics-binary-metrics source=limen/metrics/binary_metrics.py region=signature:binary_metrics role=signature -->
```python
def binary_metrics(data: dict[str, Any], preds: list[int] | npt.NDArray[np.integer[Any]], probs: list[float] | npt.NDArray[np.floating[Any]]) -> dict[str, Any]:
    ...
```

Compute binary classification metrics from predictions and probabilities.

Degenerate one-class outcomes never crash: fpr is NaN when y_test has no

negatives, and auc is NaN when y_test contains a single class, mirroring

the safe_ovr_auc convention.

| Argument | Contract |
| --- | --- |
| data (`dict`) | Data dictionary with 'y_test' key containing true binary labels |
| preds (`list`) | Predicted binary class labels |
| probs (`list`) | Predicted class probabilities |

**Returns:** dict: Dictionary containing recall, precision, fpr, auc, and accuracy metrics

### `limen.metrics.continuous_metrics`

<!-- reference: api-limen-metrics-continuous-metrics source=limen/metrics/continuous_metrics.py region=signature:continuous_metrics role=signature -->
```python
def continuous_metrics(data: dict[str, Any], preds: list[float] | npt.NDArray[np.floating[Any]]) -> dict[str, Any]:
    ...
```

Compute regression metrics from continuous predictions.

NOTE: This function is experimental and may change in future versions.

| Argument | Contract |
| --- | --- |
| data (`dict`) | Data dictionary with 'y_test' key containing true continuous values |
| preds (`list`) | Predicted continuous values |

**Returns:** dict: Dictionary containing bias, mae, rmse, r2, and mape metrics

### `limen.metrics.deflated_sharpe_ratio`

<!-- reference: api-limen-metrics-deflated-sharpe-ratio source=limen/metrics/deflated_sharpe_ratio.py region=signature:deflated_sharpe_ratio role=signature -->
```python
def deflated_sharpe_ratio(returns: npt.NDArray[np.float64], *, n_trials: int, trial_sharpe_variance: float) -> float:
    ...
```

Return the per-bar DSR probability from Bailey and López de Prado (2014).

[Bailey and López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), equation 2.

Sharpe uses sample standard deviation; skew and Pearson kurtosis use

centered population moments. Four observations and positive return

variance are required. One trial has a zero selection benchmark.

### `limen.metrics.multiclass_metrics`

<!-- reference: api-limen-metrics-multiclass-metrics source=limen/metrics/multiclass_metrics.py region=signature:multiclass_metrics role=signature -->
```python
def multiclass_metrics(data: dict[str, Any], preds: list[int] | npt.NDArray[np.integer[Any]], probs: list[list[float]] | npt.NDArray[np.floating[Any]], average: str='macro') -> dict[str, Any]:
    ...
```

Compute multiclass classification metrics from predictions and probabilities.

| Argument | Contract |
| --- | --- |
| data (`dict`) | Data dictionary with 'y_test' key containing true class labels |
| preds (`list`) | Predicted class labels |
| probs (`list`) | Predicted class probabilities |
| average (`str`) | Averaging strategy for precision and recall |

**Returns:** dict: Dictionary containing precision, recall, auc, and accuracy metrics

### `limen.metrics.probability_of_backtest_overfitting`

<!-- reference: api-limen-metrics-probability-of-backtest-overfitting source=limen/metrics/probability_of_backtest_overfitting.py region=signature:probability_of_backtest_overfitting role=signature -->
```python
def probability_of_backtest_overfitting(returns_matrix: npt.NDArray[np.float64], *, n_blocks: int) -> float:
    ...
```

Return CSCV logit mass at or below zero using per-bar Sharpe ranks.

[Bailey and López de Prado](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf), algorithm 2.3.

Rows are trials, columns synchronous bars. Equal contiguous blocks need

at least two bars. In-sample ties select the first trial; out-of-sample

ties receive their average ascending rank. Median ties count as overfit.

### `limen.metrics.rule_based_metrics`

<!-- reference: api-limen-metrics-rule-based-metrics source=limen/metrics/rule_based_metrics.py region=signature:rule_based_metrics role=signature -->
```python
def rule_based_metrics(positions: dict[str, npt.NDArray[np.integer[Any]]], backtest_results: dict[str, dict[str, float]], sharpe_std_threshold: float=0.5, sharpe_degradation_threshold: float=0.3) -> dict[str, Any]:
    ...
```

Compute rule-based strategy metrics across train, val, and test splits.

| Argument | Contract |
| --- | --- |
| positions (`dict`) | Per-split position arrays (0/1) keyed by 'train', 'val', 'test' |
| backtest_results (`dict`) | Per-split backtest metric dicts keyed by 'train', 'val', 'test' |
| sharpe_std_threshold (`float`) | Legacy parameter retained for call compatibility |
| sharpe_degradation_threshold (`float`) | Legacy parameter retained for call compatibility |

**Returns:** dict: Tier 1 position stats, Tier 2 backtest metrics per split, Tier 3 stability metrics

### `limen.metrics.safe_ovr_auc`

<!-- reference: api-limen-metrics-safe-ovr-auc source=limen/metrics/safe_ovr_auc.py region=signature:safe_ovr_auc role=signature -->
```python
def safe_ovr_auc(y_true: npt.NDArray[Any] | list[int] | list[str], probs: npt.NDArray[np.floating[Any]] | list[list[float]]) -> float:
    ...
```

Compute one-vs-rest AUC safely handling missing classes.

| Argument | Contract |
| --- | --- |
| y_true (`np.ndarray \| list[int] \| list[str]`) | True class labels, shape (n_samples,) |
| probs (`np.ndarray \| list[list[float]]`) | Predicted probabilities, shape (n_samples, n_classes) |

**Returns:** float: Mean AUC across all valid class comparisons, or NaN if no valid AUC calculations can be made
