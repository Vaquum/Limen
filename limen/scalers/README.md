# `limen.scalers`

> Fit train-only feature scaling and apply it consistently across validation and test data.

## Canonical docs

- [Scalers](../../docs/Scalers.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)

## What this package owns

Owns fitted scaler classes, the scaler registry, and the rule-based logic that decides how columns should be transformed.
Does **not** own raw feature creation or the experiment loop itself.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `LinearScaler` | Rule-based scaling over mixed market-data columns | Exported at the package root |
| `LogRegScaler` | Standard scaler used by logistic-regression SFDs | Exported at the package root |
| `RobustScaler` | Median and IQR scaling for outlier-heavy data | Exported at the package root |
| `CausalRollingRobustScaler` | Robust scaling that adapts to drift, with no look-ahead | Exported at the package root |
| `RankGaussScaler` | Rank-based Gaussianization | Exported at the package root |
| `SCALER_REGISTRY` | Resolve scalers by manifest parameter name | Used by `set_scaler_from_params()` |
| `build_rules`, `inverse_transform` | Customize or interpret `LinearScaler` behavior | Available from the module-level implementations |

## Adjacent modules

- `limen.experiment.Manifest` is the main consumer of this package.
- `limen.transforms` handles stateless transforms, which is a different stage from fitted scaling.
- `limen.features` and `limen.indicators` produce the columns that scalers later operate on.

## Quick orientation

```text
scalers/
├── linear_scaler.py                 # LinearScaler, rule helpers, inverse transform
├── logreg_scaler.py                 # Logistic-regression tuned scaling
├── robust_scaler.py                 # Median and IQR scaling
├── causal_rolling_robust_scaler.py  # Causal trailing-window median and IQR scaling
├── rank_gauss_scaler.py             # Rank-based Gaussianization
└── registry.py                      # SCALER_REGISTRY
```

## Things to know

- Scalers are fit on the training split and then reused on validation and test splits. That fit/apply discipline is part of why this package stays separate from `limen.transforms`.
- `LinearScaler` uses ordered regex rules. The first matching rule wins.
- Unrecognized columns fall through to the catch-all `none` rule and stay unchanged.
- Columns assigned `standard` or `log_standard` scaling must have a finite, nonzero training standard deviation. `LinearScaler` and `LogRegScaler` do not guard zero variance; division by zero produces non-finite output.

## Read next

- [Scalers](../../docs/Scalers.md)
- [Transforms](../../docs/Transforms.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
