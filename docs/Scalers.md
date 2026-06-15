# Scalers

Scalers are train-fitted preprocessing objects. A manifest fits the scaler on `x_train`, then reuses that fitted state to transform validation and test data without refitting.

This page covers scaler selection, `set_scaler()` and `set_scaler_from_params()` behavior, and the custom scaler interface.

## How Scalers Fit In Limen

The manifest pipeline is split-first:

1. fetch and prepare raw data
2. split into train, validation, and test
3. build indicators, features, and target columns
4. fit the configured scaler on `x_train`
5. apply that exact fitted scaler to `x_val` and `x_test`

That is why scalers live separately from the stateless helpers in [Transforms](Transforms.md).

## Choosing A Scaler

| Scaler | Fit | Inverse support | Notes |
|---|---|---|---|
| `LogRegScaler` | the reference logistic-regression style feature sets used in foundational flows | yes | Uses a fixed per-column rule map. Columns outside the rule map are left alone. |
| `LinearScaler` | mixed feature sets needing regex-based scaling rules | yes | Broadest built-in scaler rule surface. Supports `standard`, `log_standard`, `divide_100`, and `none`. |
| `RobustScaler` | outlier-heavy numeric features | yes | Uses median and IQR instead of mean and standard deviation. |
| `CausalRollingRobustScaler` | non-stationary features whose scale drifts over time | no | Median and IQR from a strictly trailing rolling window, so no look-ahead. Not row-wise invertible from the fitted scaler. |
| `RankGaussScaler` | numeric features that benefit from a Gaussianized shape | approximate | The inverse is only approximate because rank-based transforms are lossy. |

## Manifest Usage

### Fixed scaler

```python
from limen.scalers import LogRegScaler

manifest.set_scaler(LogRegScaler)
```

### Parameterized scaler choice

```python
manifest.set_scaler_from_params('scaler_type')

params = {
    'scaler_type': ['linear', 'robust', 'rank_gauss'],
}
```

The built-in registry currently maps:

| Key | Class |
|---|---|
| `'linear'` | `LinearScaler` |
| `'logreg'` | `LogRegScaler` |
| `'robust'` | `RobustScaler` |
| `'rank_gauss'` | `RankGaussScaler` |
| `'causal_rolling_robust'` | `CausalRollingRobustScaler` |

The registry itself is also a public export:

```python
from limen.scalers import SCALER_REGISTRY

sorted(SCALER_REGISTRY)
```

That is the lookup surface `set_scaler_from_params()` uses under the hood.

On live local manifest-prep runs in this repo, `set_scaler_from_params('scaler_type')` correctly resolved:

- `'robust'` to `RobustScaler`
- `'rank_gauss'` to `RankGaussScaler`

## Built-In Scalers

### `LogRegScaler`

`LogRegScaler(x_train)` uses a fixed column-to-rule mapping tailored to the classic Limen logistic-regression workflow.

- standardizes columns such as `open`, `close`, `atr`, `macd`, `roc`, and `returns`
- log-standardizes columns such as `volume`, `no_of_trades`, and liquidity fields
- divides `wilder_rsi` by `100`
- leaves columns such as `maker_ratio` unchanged

This is the logistic-regression-oriented scaler in the package. It remains the legacy default for old-style foundational SFDs and exposes fewer rules than `LinearScaler`.

### `LinearScaler`

`LinearScaler(x_train, rules=None, default='standard')` applies regex-driven scaling rules.

It supports:

- `standard`
- `log_standard`
- `divide_100`
- `none`

Use `set_scaler_from_params()` for explicit scaling-policy control or search-time scaler choice.

### `RobustScaler`

`RobustScaler(x_train, quantile_range=(0.25, 0.75))` applies:

```python
(x - median) / IQR
```

It skips datetime and non-numeric columns automatically and fits heavy-tail or outlier-distorted standardization cases.

### `RankGaussScaler`

`RankGaussScaler(x_train, n_quantiles=1000)` maps numeric columns to a quantile-Gaussian distribution through quantiles and the inverse normal CDF.

Use it when relative ordering matters more than preserving original spacing. Its inverse transform is approximate, not exact.

### CausalRollingRobustScaler

`CausalRollingRobustScaler(x_train, window=1000, quantile_range=(0.25, 0.75), clip=8.0, min_samples=50)` applies:

```python
(x - rolling_median) / rolling_IQR
```

The median and IQR for each row are taken from the `window` rows strictly before it (`.shift(1)`), so the transform never reads the current row or any future row. That makes it suited to non-stationary series whose scale drifts over time.

Rows with fewer than `min_samples` of trailing history fall back to the median and IQR fitted on `x_train`, so during warmup it degrades to plain `RobustScaler` behavior. The scaled output is clipped to `+/- clip`.

It skips datetime and non-numeric columns automatically. It provides no inverse transform: each row's scaling factors are derived from the data itself, so the transform is not row-wise invertible from the fitted scaler alone.

The fitted instance exposes `context_rows = window`. The experiment pipeline reads this property to size the scaler warm-up portion of the Context Carry-Over (CCO) block prepended to val and test splits, ensuring the scaler sees enough preceding rows to be fully warm from the very first split row.

Scalers without a `context_rows` property (such as `RobustScaler` and `StandardScaler`) contribute zero to the scaler warm-up. CCO may still run for those scalers when indicators produce leading warm-up rows — those rows are prepended as indicator context regardless of scaler type. The total CCO block is `indicator_warm_up_rows + scaler_context_rows`.

## Custom Scaler Contract

All custom scalers should follow the same interface:

```python
class IdentityScaler:
    def __init__(self, x_train: pl.DataFrame, **kwargs):
        self.columns = x_train.columns

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df
```

Optional inverse helper:

```python
def inverse_transform(df: pl.DataFrame, scaler: IdentityScaler) -> pl.DataFrame:
    return df
```

That contract is what makes the scaler usable from `Manifest.set_scaler()` and compatible with post-processing flows that need to return to the original scale.

## Practical Notes

- `RobustScaler`, `CausalRollingRobustScaler`, and `RankGaussScaler` automatically skip datetime and non-numeric columns.
- `LogRegScaler` and `LinearScaler` are rule-driven, so only columns matched by their rule sets are transformed.
- `LinearScaler` fits cases where new feature names are expected to appear frequently.
- If a prediction post-processing step needs original scale values, prefer a scaler with a meaningful inverse path.

## Read Next

- [Transforms](Transforms.md) for stateless preprocessing and post-model helpers
- [Experiment Manifest](Experiment-Manifest.md) for where scaling happens inside the split-first pipeline
- [Trainer](Trainer.md) for the retraining path that reconstructs manifests and preserves fitted preprocessing behavior
