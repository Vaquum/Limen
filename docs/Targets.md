# Targets

Target classes in Limen define how the target column is built during the manifest preparation pipeline. A target class is fitted once on the training split, then applied to validation and test without refitting.

Use this page to choose a target class, understand the convention each class must follow, or see what columns each class adds.

## How Targets Fit In The Pipeline

The manifest pipeline is split-first:

1. fetch and prepare raw data
2. split into train, validation, and test
3. apply indicators and features
4. fit the target class on the training split
5. apply the fitted instance to all splits using the stored state

This ensures that any statistics derived from the training data (such as quantile cutoffs) do not leak into validation or test.

## Using A Target Class

```python
from limen.targets import QuantileBinaryTarget

manifest.with_target_label(
    'quantile_flag',
    QuantileBinaryTarget,
    fit_params={'source_column': 'roc_{roc_period}', 'quantile': 'q'},
    transform_params={'shift': 'shift'},
)
```

`fit_params` are forwarded to `__init__` on the training split. `transform_params` are forwarded to `transform()` on every split. Both support round-param references such that `'q'` resolves to the current round's `q` value.

## Choosing A Target Class

| Class | Output type | Fitted on train | Shift default | Notes |
|---|---|---|---|---|
| `QuantileBinaryTarget` | binary `UInt8` | yes — quantile cutoff | `shift=0` | Positive label above the top-N quantile of the source column. |
| `ThresholdBinaryTarget` | binary `UInt8` | no | `shift=-1` | Positive label above a fixed numeric threshold. |
| `ForwardBreakoutTarget` | binary `UInt8` | no | `shift=-1` | Positive label if price rises at least `threshold` over the next `forward_periods` bars. |
| `NextReturnTarget` | continuous `Float64` | no | n/a | Percentage return over the next N bars. Use with regression architectures. |
| `RandomBinaryTarget` | binary `UInt8` | no | none | Uniformly random labels. Use as a noise benchmark. |
| `IdentityTarget` | existing column | no | none | Target column already present in the data. Validates the column exists on every split. |

## Reference

### `QuantileBinaryTarget`

```python
QuantileBinaryTarget(train_data, target_name, source_column, quantile)
```

Fits a quantile cutoff on the training split. Labels a bar as positive if the source column exceeds the `(1 - quantile)` quantile of the training distribution.

```python
.with_target_label(
    'quantile_flag',
    QuantileBinaryTarget,
    fit_params={'source_column': 'roc_4', 'quantile': 0.35},
    transform_params={'shift': -1},
)
```

| Parameter | Type | Description |
|---|---|---|
| `source_column` | `str` | Column used to compute the quantile threshold |
| `quantile` | `float` | Top-N fraction for positive label (`0.3` = top 30%) |
| `shift` | `int` | Periods to shift the label; `shift=0` means no shift |

### `ThresholdBinaryTarget`

```python
ThresholdBinaryTarget(train_data, target_name, source_column, threshold)
```

Labels a bar as positive if the source column exceeds a fixed threshold. No fitting step — the threshold is specified directly.

```python
.with_target_label(
    'above_zero',
    ThresholdBinaryTarget,
    fit_params={'source_column': 'roc_4', 'threshold': 0.0},
    transform_params={'shift': -1},
)
```

| Parameter | Type | Description |
|---|---|---|
| `source_column` | `str` | Column to compare against the threshold |
| `threshold` | `float` | Fixed value; bars above this are labeled 1 |
| `shift` | `int` | Periods to shift the label; default `-1` |

### `ForwardBreakoutTarget`

```python
ForwardBreakoutTarget(train_data, target_name)
```

Labels a bar as positive if the close price rises by at least `threshold` over the next `forward_periods` bars. No fitting step.

```python
.with_target_label(
    'breakout',
    ForwardBreakoutTarget,
    transform_params={'forward_periods': 24, 'threshold': 0.02, 'shift': -1},
)
```

| Parameter | Type | Description |
|---|---|---|
| `forward_periods` | `int` | Look-ahead window in bars; default `24` |
| `threshold` | `float` | Minimum return for positive label; default `0.02` (2%) |
| `shift` | `int` | Additional shift applied after labeling; default `-1` |

### `NextReturnTarget`

```python
NextReturnTarget(train_data, target_name)
```

Produces a continuous target as the percentage return over the next N bars. Use with regression architectures such as `xgboost_regressor`. No fitting step.

```python
.with_target_label(
    'next_return',
    NextReturnTarget,
    transform_params={'periods': 1, 'scale': 100.0},
)
```

| Parameter | Type | Description |
|---|---|---|
| `periods` | `int` | Look-ahead window in bars; default `1` |
| `scale` | `float` | Multiplier applied to the raw return; default `100.0` |

### `RandomBinaryTarget`

```python
RandomBinaryTarget(train_data, target_name)
```

Produces uniformly random binary labels. No fitting step. Use as a noise benchmark to verify that a trained model outperforms chance.

```python
.with_target_label('outcome', RandomBinaryTarget)
```

No parameters.

### `IdentityTarget`

```python
IdentityTarget(train_data, target_name)
```

For when the target column is already present in the data. Validates that the column exists on the training split and on every subsequent split. The data is returned unchanged; no new column is added.

```python
.with_target_label('label', IdentityTarget)
```

No parameters.

## Writing A Custom Target Class

A target class must follow this convention:

```python
class MyTarget:

    def __init__(self, train_data: pl.DataFrame, target_name: str, **fit_params) -> None:
        self.target_name = target_name
        # fit on train_data here if needed

    def transform(self, data: pl.DataFrame, **transform_params) -> pl.DataFrame:
        return data.with_columns(
            # ... compute label ...
            .alias(self.target_name)
        )
```

There is no base class to inherit from — the convention is enforced structurally, the same way scalers work. Any class that accepts `(train_data, target_name, **fit_params)` in `__init__` and exposes `transform(data, **transform_params) -> pl.DataFrame` is a valid target class.

## Read Next

- [Experiment Manifest](Experiment-Manifest.md) for how targets plug into the split-first pipeline
- [Scalers](Scalers.md) for train-fitted preprocessing that runs after target construction
- [Transforms](Transforms.md) for stateless post-model helpers
