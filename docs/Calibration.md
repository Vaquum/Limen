# Calibration

Calibration controls two model-output surfaces: the reliability of predicted probabilities, and where the decision boundary sits. Limen's calibration layer is declarative — it is configured on the manifest and injected automatically into each experiment round.

## What Calibration Does

Binary classifiers return raw probabilities, but raw probabilities can be miscalibrated. A model output of `0.7` is not automatically a calibrated `70%` positive-class probability. Probability calibration fits a post-hoc correction on a held-out validation set so the output probabilities align with actual class frequencies.

Threshold optimization solves a different problem. The default decision boundary at `0.5` can be suboptimal when the target metric is asymmetric, such as when recall and precision have unequal importance. Limen's `grid_threshold_optimizer` sweeps a range of candidate thresholds, scores each one with a chosen metric, and selects the highest-scoring candidate.

The two steps are independent and can be used together or separately.

## The Four Modes

| `use_calibration` | `use_threshold` | Behavior |
|---|---|---|
| True | True | calibrate probabilities → optimize threshold |
| True | False | calibrate probabilities → decide at 0.5 |
| False | True | raw probabilities → optimize threshold |
| False | False | raw model output, no calibration injected |

Both flags default to `True`, so adding `with_calibration()` to a manifest with no search over these flags runs the fully calibrated path automatically.

When both are `False`, no `prediction_calibration_config` is injected into the architecture and the model behaves exactly as if calibration was never configured.

## Quick Start

The minimum setup: add `with_calibration()` to the manifest.

```python
from limen.calibration import grid_threshold_optimizer, sklearn_probability_calibrator
from limen.data import HistoricalData
from limen.experiment import Manifest, MLManifest
from limen.indicators import roc
from limen.metrics.balanced_metric import balanced_metric
from limen.scalers import LogRegScaler
from limen.sfd.reference_architecture import logreg_binary
from limen.targets import QuantileBinaryTarget

def manifest() -> Manifest:
    return (
        MLManifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'row_count_limit': 5000},
        )
        .set_split_config(8, 1, 2)
        .add_indicator(roc, period=14)
        .with_target_label(
            'quantile_flag',
            QuantileBinaryTarget,
            fit_params={'source_column': 'roc_14', 'quantile': 0.40},
            transform_params={'shift': -1},
        )
        .set_scaler(LogRegScaler)
        .with_reference_architecture(logreg_binary)
        .with_calibration()
        .probability_calibration(func=sklearn_probability_calibrator, method='isotonic')
        .threshold_function(func=grid_threshold_optimizer, metric=balanced_metric)
        .done()
    )
```

`with_calibration()` returns a `CalibrationBuilder`. After configuring the steps, call `.done()` to return to the manifest and finalize the pipeline.

`sklearn_probability_calibrator` wraps sklearn's `CalibratedClassifierCV` with `FrozenEstimator` to avoid the deprecation that came with sklearn 1.6. It accepts `method='isotonic'` (default) or `method='sigmoid'`.

`grid_threshold_optimizer` sweeps a bounded range of thresholds, scores each with the chosen metric, and returns the highest-scoring threshold. `balanced_metric` is Limen's built-in `precision * sqrt(trade_rate)` score, which balances signal quality with trade frequency — appropriate when both accuracy and trading activity matter.

## Threshold-Only Path

`threshold_function()` configures threshold optimization without probability recalibration. In an existing manifest chain, the calibration fragment is:

```python-fragment
.with_calibration()
.threshold_function(func=grid_threshold_optimizer, metric=balanced_metric)
.done()
```

The optimizer runs on the raw model probabilities. No calibration step is applied.

## Searching Over Calibration Parameters

String parameter values are treated as round-param references. This is the same resolution mechanism used elsewhere in the manifest.

```python
def params():
    return {
        'cal_method':      ['isotonic', 'sigmoid'],
        'threshold_min':   [0.20, 0.25, 0.30],
        'threshold_max':   [0.60, 0.65, 0.70],
        'threshold_step':  [0.05, 0.10],
    }

def manifest() -> Manifest:
    return (
        MLManifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'row_count_limit': 5000},
        )
        .set_split_config(8, 1, 2)
        .add_indicator(roc, period=14)
        .with_target_label(
            'quantile_flag',
            QuantileBinaryTarget,
            fit_params={'source_column': 'roc_14', 'quantile': 0.40},
            transform_params={'shift': -1},
        )
        .set_scaler(LogRegScaler)
        .with_reference_architecture(logreg_binary)
        .with_calibration()
        .probability_calibration(func=sklearn_probability_calibrator, method='cal_method')
        .threshold_function(
            func=grid_threshold_optimizer,
            metric=balanced_metric,
            threshold_min='threshold_min',
            threshold_max='threshold_max',
            threshold_step='threshold_step',
        )
        .done()
    )
```

At runtime, `'cal_method'` is substituted with the current round's `cal_method` value. Non-string values (floats, callables like `metric=balanced_metric`) pass through unchanged.

## On/Off Comparison Within a Single Experiment

`use_calibration` and `use_threshold` are consumed by `run_model()` before the architecture is called. The architecture never sees these flags — it only sees the resulting (possibly masked) `CalibrationConfig`.

Adding both to `params()` creates a 2×2 grid of calibration modes within a single experiment:

```python
def params():
    return {
        'use_calibration': [True, False],
        'use_threshold':   [True, False],
    }
```

All four modes produce comparable metrics in the same experiment log, so each step's effect is visible directly from the results.

## Custom Calibrators and Optimizers

The calibration layer is protocol-based. Any function that matches the expected signatures can be plugged in.

**Calibrator protocol** — `(clf, x_val, y_val, **params) -> fitted model with predict_proba()`:

```python
def passthrough_calibrator(clf, x_val, y_val, **params):
    return clf
```

Pass `passthrough_calibrator` as the `func=` value in `probability_calibration()`.

**Threshold optimizer protocol** — `(y_val, val_proba, **params) -> tuple[float, float]`:

```python
def fixed_threshold_optimizer(y_val, val_proba, **params) -> tuple[float, float]:
    return 0.5, 0.0
```

Pass `fixed_threshold_optimizer` as the `func=` value in `threshold_function()`.

The two protocols are independent. Limen's built-in calibrator can be paired with a custom optimizer, and a custom calibrator can be paired with Limen's built-in optimizer.

## What Gets Added to Results

When calibration is configured, `evaluate()` always includes these two keys so the experiment log schema stays stable across rounds (e.g. when `use_calibration`/`use_threshold` are searched as `[True, False]`):

| Key | Type | Meaning |
|---|---|---|
| `optimal_threshold` | `float` or `None` | the threshold chosen by the optimizer (`None` when calibration is off for this round) |
| `val_score` | `float` or `None` | the metric score at that threshold; `None` when no threshold function ran |

These appear in the experiment log alongside the standard binary metrics. `Trainer` validates them during reconstruction; `Sensor` does not expose an experiment-results mapping.

NOTE: `predict()` only includes these keys when calibration is active for the round. `evaluate()` always emits them to ensure every row in the CSV has a consistent schema.

`_preds` are computed using `optimal_threshold`: a test observation is predicted positive when its probability is at or above the threshold.

## `grid_threshold_optimizer` Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `threshold_min` | `0.20` | minimum threshold to test |
| `threshold_max` | `0.70` | maximum threshold to test |
| `threshold_step` | `0.05` | step size for the sweep |
| `default_threshold` | `0.35` | fallback when no threshold produces positive predictions |
| `metric` | `balanced_metric` | scoring function `(y_true, y_pred) -> float` |

The optimizer returns the `default_threshold` with score `0.0` when every candidate threshold predicts all negatives.

Threshold grids are caller-owned inputs: `threshold_step` must be positive and finite, bounds must be finite, and `threshold_min <= threshold_max`. The default-threshold fallback is for valid grids that produce no positive predictions, not a substitute for validating a malformed grid.

## Calibration at Inference

When a calibrated model is promoted to a `Sensor`, the calibrator is fitted once during training evaluation (on validation data) and stored inside the model instance. Subsequent `predict()` calls — including all sensor inference calls — reuse the stored calibrator without needing `x_val` or `y_val`. Calibrated sensors therefore work identically to uncalibrated ones from the caller's perspective: pass `x_test`, receive predictions.

`fit_calibrator` is the low-level function that encapsulates this fit step. It is called internally by `LogRegBinary.predict()` and `TabPFNBinary.predict()` on first use, and is also available as a public API for calibrator fitting outside the standard pipeline.

## Where To Look

- `limen/calibration/` — `fit_calibrator`, `sklearn_probability_calibrator`, `grid_threshold_optimizer`, `apply_calibrated_predict`, `CalibratorProtocol`, `ThresholdOptimizerProtocol`
- `limen.experiment.CalibrationConfig` — the resolved config dataclass injected into the architecture
- `CalibrationBuilder` — the fluent builder returned by `MLManifest.with_calibration()`; not a public import, obtain it only via the fluent chain

## Read Next

- Continue to [Experiment Manifest](Experiment-Manifest.md) for the full manifest builder reference.
- Continue to [Reference Architecture](Reference-Architecture.md) to see how calibration interacts with `predict()` and `evaluate()`.
- Continue to [Built-In SFDs](Built-In-SFDs.md) to see calibration wired up in the foundational `logreg_binary` and `tabpfn_binary` SFDs.
