# Scalers

Scalers are train-fitted preprocessing objects. A manifest fits the scaler on `x_train`, then reuses that fitted state to transform validation and test data without refitting.

Use this page when you need to choose a scaler, understand how `set_scaler()` and `set_scaler_from_params()` behave, or see the interface a custom scaler must follow.

## How Scalers Fit In Limen

The manifest pipeline is split-first:

1. fetch and prepare raw data
2. split into train, validation, and test
3. build indicators, features, and target columns
4. fit the configured scaler on `x_train`
5. apply that exact fitted scaler to `x_val` and `x_test`

That is why scalers live separately from the stateless helpers in [Transforms](Transforms.md).

## Choosing A Scaler

| Scaler | Best fit | Inverse support | Notes |
|---|---|---|---|
| `LogRegScaler` | the reference logistic-regression style feature sets used in foundational flows | yes | Uses a fixed per-column rule map. Columns outside the rule map are left alone. |
| `LinearScaler` | mixed feature sets where regex-based scaling rules are useful | yes | The most flexible built-in scaler. Supports `standard`, `log_standard`, `divide_100`, and `none`. |
| `RobustScaler` | outlier-heavy numeric features | yes | Uses median and IQR instead of mean and standard deviation. |
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

This is the most opinionated scaler in the package. It is a good default for old-style foundational SFDs, but less flexible than `LinearScaler`.

### `LinearScaler`

`LinearScaler(x_train, rules=None, default='standard')` applies regex-driven scaling rules.

It supports:

- `standard`
- `log_standard`
- `divide_100`
- `none`

Use it when you want explicit control over scaling policy or when scaler choice itself is part of the search space.

### `RobustScaler`

`RobustScaler(x_train, quantile_range=(0.25, 0.75))` applies:

```python
(x - median) / IQR
```

It skips datetime and non-numeric columns automatically and is usually the safest choice when heavy tails or outliers are distorting standardization.

### `RankGaussScaler`

`RankGaussScaler(x_train, n_quantiles=1000)` maps numeric columns to an approximately Gaussian distribution through quantiles and the inverse normal CDF.

Use it when relative ordering matters more than preserving original spacing. Its inverse transform is approximate, not exact.

## Custom Scaler Contract

All custom scalers should follow the same interface:

```python
class YourScaler:
    def __init__(self, x_train: pl.DataFrame, **kwargs):
        ...

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        ...
```

Optional inverse helper:

```python
def inverse_transform(df: pl.DataFrame, scaler: YourScaler) -> pl.DataFrame:
    ...
```

That contract is what makes the scaler usable from `Manifest.set_scaler()` and compatible with post-processing flows that need to return to the original scale.

## Practical Notes

- `RobustScaler` and `RankGaussScaler` automatically skip datetime and non-numeric columns.
- `LogRegScaler` and `LinearScaler` are rule-driven, so only columns matched by their rule sets are transformed.
- `LinearScaler` is the better choice when new feature names are expected to appear frequently.
- If a prediction post-processing step needs original scale values, prefer a scaler with a meaningful inverse path.

## Read Next

- [Transforms](Transforms.md) for stateless preprocessing and post-model helpers
- [Experiment Manifest](Experiment-Manifest.md) for where scaling happens inside the split-first pipeline
- [Trainer](Trainer.md) for the retraining path that reconstructs manifests and preserves fitted preprocessing behavior
