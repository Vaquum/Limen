# `limen.scalers`

> Fit feature-scaling transformations on training data and apply them consistently to validation and test sets.

## Responsibilities

Provides scaler classes that are trained on the training split and then applied to all splits, preventing look-ahead bias.  Scalers integrate with the `Manifest` pipeline through `Manifest.set_scaler()`.

Does **not** own the data splits or control when `fit` vs `transform` is called — that sequencing is managed by `Manifest._apply_scaler()`.

## Key concepts

- **LinearScaler** – regex-rule-driven per-column Z-score or log-then-Z-score scaling; rules determine how each column is scaled based on its name
- **LogRegScaler** – logistic-regression-specific scaler variant
- **DEFAULT_SCALING_RULES** – regex → rule mapping shipped with `LinearScaler`; covers all standard OHLCV, indicator, and feature column names; falls back to `'none'` (passthrough) for anything unrecognised
- **build_rules()** – merge user overrides into the default rule set
- **get_scaling_rule()** – look up the rule for a single column name
- **inverse_transform()** – reverse a `LinearScaler` transformation; used when predictions are in scaled space

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `LinearScaler` | `linear_scaler.py` | Instantiate with `x_train`; call `.transform(df)` on each split |
| `inverse_transform()` | `linear_scaler.py` | Invert scaled predictions or targets back to original scale |
| `LogRegScaler` | `logreg_scaler.py` | Logistic-regression-oriented scaler variant |
| `build_rules()` | `linear_scaler.py` | Merge column-specific overrides with the default rule set |

## Dependencies

- **Internal:** none
- **External:** `polars`, `re`

## Quick orientation

```text
scalers/
├── linear_scaler.py   # LinearScaler, DEFAULT_SCALING_RULES, build_rules, get_scaling_rule, inverse_transform
└── logreg_scaler.py   # LogRegScaler
```

## Gotchas / things to know

- Scaling rules are matched in insertion order; the catch-all `'.*': 'none'` at the end of `DEFAULT_SCALING_RULES` ensures columns without an explicit rule are left unchanged
- `LinearScaler` stores per-column `means` and `stds` dicts; columns with `rule='none'` are not stored and not transformed
- When using `log_standard`, zero or negative values will produce `-inf` after `log1p`; ensure such columns are non-negative before scaling
- Pass `LinearScaler` to `Manifest.set_scaler()` — the manifest handles calling `fit` on training data and `transform` on all splits automatically
