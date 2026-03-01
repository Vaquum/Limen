# `limen.scalers`

> Fit and apply feature-scaling transformations to Polars DataFrames.

## Responsibilities

Owns stateful scaler classes that are fit on training data and then applied consistently to val/test splits.
Does **not** own the column selection logic for what to scale (that is embedded in `LinearScaler`'s rule set) or the experiment loop mechanics.

## Key concepts

- **LinearScaler** – regex-rule-driven scaler; maps each column name to a scaling strategy (`standard`, `log_standard`, `divide_100`, or `none`) and applies a z-score normalisation using training-set statistics.
- **LogRegScaler** – specialised scaler tuned for logistic regression inputs; extends or wraps `LinearScaler` with appropriate defaults for the LR feature space.
- **Scaling rules** – a `dict[regex_pattern → rule_name]` evaluated in order; the first matching pattern wins. The default ruleset in `linear_scaler.py` covers all common OHLCV and indicator column names.
- **`inverse_transform`** – module-level function in `linear_scaler.py`; reverses a `LinearScaler` transformation (useful for interpreting predictions in the original scale).

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `LinearScaler(x_train)` | `linear_scaler.py` | Fit on training split; then call `.transform(df)` on val/test |
| `LogRegScaler(x_train)` | `logreg_scaler.py` | Used by `Manifest.set_scaler(LogRegScaler)` in LR-based SFDs |
| `inverse_transform(df, scaler)` | `linear_scaler.py` | Reverse scaling for result interpretation |
| `build_rules(overrides)` | `linear_scaler.py` | Merge user column overrides into the default rule set |

## Dependencies

- **Internal:** consumed by `limen.experiment.manifest_core` via `Manifest.set_scaler()`
- **External:** `polars`

## Quick orientation
```text
scalers/
├── linear_scaler.py   # LinearScaler, build_rules, get_scaling_rule, inverse_transform
└── logreg_scaler.py   # LogRegScaler (LR-tuned defaults)
```

## Gotchas / things to know

- The default rule `r'.*': 'none'` is a catch-all at the end of `DEFAULT_SCALING_RULES`; any unrecognised column is left unscaled.
- Columns with zero standard deviation are skipped silently during fitting — their values are passed through unchanged.
- `Manifest.set_scaler()` wraps `LinearScaler` (or `LogRegScaler`) in a `FittedTransformEntry` that ensures fit-on-train / apply-to-all semantics automatically.
- Pass `overrides={'standard': ['my_col']}` to `build_rules()` to override the default behaviour for specific columns.
