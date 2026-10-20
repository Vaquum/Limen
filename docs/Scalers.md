# Scalers

Scalers fit preprocessing state on training observations and reuse that state on later observations. Python Research owns that fit in ordinary Python; Compiled Research gives it to the Manifest.

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

## Fit once, then reuse

The [Python project](examples/python-research/README.md) fits a scikit-learn `StandardScaler` and Ridge model together. `run_research` retains that evaluated pipeline, and its Sensor uses the same fitted objects without fitting them again.

<!-- example: scalers-python-fit source=docs/examples/python-research/research.py region=fitted_scaler scenario=selection interface=python role=executed -->
```python
def fit(train, round_params):
    return make_pipeline(StandardScaler(), Ridge(alpha=round_params["alpha"])).fit(
        train.select("momentum"), train["next_return"].to_numpy()
    )
```

The [Compiled project](examples/compiled-research/README.md) resolves its scaler from the current round's `scaler_type`. Its selection configuration declares `scaler_type: [logreg]`; the Manifest fits `LogRegScaler` on `x_train` and reuses it on validation and test. Trainer later reconstructs the selected recorded definition with its original splits.

<!-- example: scalers-compiled-fit source=docs/examples/compiled-research/selection.yaml region=fitted_scaler scenario=selection interface=compiled role=executed -->
```yaml
    scaler:
      from_params: scaler_type
```

A fixed scaler uses `Manifest.set_scaler(transform_class, param_name, extra_params)`. A searched scaler uses `Manifest.set_scaler_from_params(param_name, extra_params)`, with a declared parameter whose values are registry keys. The current registry is:

<!-- reference: scalers-registry source=limen/scalers/registry.py region=definition:SCALER_REGISTRY role=definition -->
```python
SCALER_REGISTRY: dict[str, type] = {'linear': LinearScaler, 'logreg': LogRegScaler, 'robust': RobustScaler, 'rank_gauss': RankGaussScaler, 'causal_rolling_robust': CausalRollingRobustScaler}
```

The class definitions below describe alternative scaling policies. The two selection runs prove the fitted pipelines shown above; they do not claim to search every registered scaler.

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

`RobustScaler(x_train, quantile_range=(0.25, 0.75))` applies `(x - median) / IQR`.

It skips datetime and non-numeric columns automatically and fits heavy-tail or outlier-distorted standardization cases.

### `RankGaussScaler`

`RankGaussScaler(x_train, n_quantiles=1000)` maps numeric columns to a quantile-Gaussian distribution through quantiles and the inverse normal CDF.

Use it when relative ordering matters more than preserving original spacing. Its inverse transform is approximate, not exact.

### CausalRollingRobustScaler

`CausalRollingRobustScaler(x_train, window=1000, quantile_range=(0.25, 0.75), clip=8.0, min_samples=50)` applies `(x - rolling_median) / rolling_IQR`.

The median and IQR for each row are taken from the `window` rows strictly before it (`.shift(1)`), so the transform never reads the current row or any future row. That makes it suited to non-stationary series whose scale drifts over time.

Rows with fewer than `min_samples` of trailing history fall back to the median and IQR fitted on `x_train`, so during warmup it degrades to plain `RobustScaler` behavior. The scaled output is clipped to `+/- clip`.

It skips datetime and non-numeric columns automatically. It provides no inverse transform: each row's scaling factors are derived from the data itself, so the transform is not row-wise invertible from the fitted scaler alone.

The fitted instance exposes `context_rows = window`. The experiment pipeline reads this property to size the scaler warm-up portion of the Context Carry-Over (CCO) block prepended to val and test splits, ensuring the scaler sees enough preceding rows to be fully warm from the very first split row.

Scalers without a `context_rows` property (such as `RobustScaler` and `StandardScaler`) contribute zero to the scaler warm-up. CCO may still run for those scalers when indicators produce leading warm-up rows — those rows are prepended as indicator context regardless of scaler type. The total CCO block is `indicator_warm_up_rows + scaler_context_rows`.

## Custom Scaler Contract

A Manifest scaler accepts training features at construction and exposes `transform(df) -> pl.DataFrame`. Store learned statistics during construction; transform later splits with those stored statistics. There is no required base class. `RobustScaler` exposes the concrete package signatures:

<!-- reference: scalers-robust-constructor source=limen/scalers/robust_scaler.py region=signature:RobustScaler.__init__ role=signature -->
```python
def __init__(self, x_train: pl.DataFrame, quantile_range: tuple[float, float]=(0.25, 0.75)) -> None:
    ...
```

<!-- reference: scalers-robust-transform source=limen/scalers/robust_scaler.py region=signature:RobustScaler.transform role=signature -->
```python
def transform(self, df: pl.DataFrame) -> pl.DataFrame:
    ...
```

The package's inverse helper is a separate callable accepting the frame and fitted scaler:

<!-- reference: scalers-robust-inverse source=limen/scalers/robust_scaler.py region=signature:inverse_transform role=signature -->
```python
def inverse_transform(df: pl.DataFrame, scaler: RobustScaler) -> pl.DataFrame:
    ...
```

An inverse is optional and must reflect the information the transform retains. A rolling scaler cannot promise a row-wise inverse from fitted state alone. Python Research may instead use any caller-owned preprocessing object, including the pipeline above; it does not require adopting the Manifest scaler convention.

## Practical Notes

- `RobustScaler`, `CausalRollingRobustScaler`, and `RankGaussScaler` automatically skip datetime and non-numeric columns.
- `LogRegScaler` and `LinearScaler` are rule-driven, so only columns matched by their rule sets are transformed.
- `LinearScaler` fits cases where new feature names are expected to appear frequently.
- If a prediction post-processing step needs original scale values, prefer a scaler with a meaningful inverse path.

## Read Next

- [Transforms](Transforms.md) for stateless preprocessing and post-model helpers
- [Experiment Manifest](Experiment-Manifest.md) for where scaling happens inside the split-first pipeline
- [Trainer](Trainer.md) for the replay path that reconstructs manifests and preserves fitted preprocessing behavior
