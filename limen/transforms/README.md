# `limen.transforms`

> Hold lightweight transform helpers for target construction, data cleanup, calibration, and threshold selection.

## Canonical docs

- [Transforms](../../docs/Transforms.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)

## What this package owns

Owns stateless DataFrame transforms and post-model helpers such as probability calibration and threshold optimization.
Does **not** own fitted feature scaling, higher-level feature engineering, or experiment orchestration.

## Key entry points

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `shift_column_transform` | You need a forward-looking target column | Common inside manifest target blocks |
| `zscore_transform` | You want a simple frame-wide normalization step | Stateless and fast |
| `winsorize_transform`, `mad_transform`, `quantile_trim_transform` | You want lightweight clipping or row filtering before modeling | These do not carry their own train-only state |
| `calibrate_classifier` | You want better-calibrated classifier probabilities | Operates after model training |
| `optimize_binary_threshold` | You want a decision threshold tuned on validation data | Often paired with calibration |

## Adjacent modules

- `limen.features` and `limen.indicators` produce the columns these transforms often operate on.
- `limen.scalers` handles fitted train-only scaling, which is a separate concern from this package.
- `limen.experiment.Manifest` provides the most common place to attach target transforms.

## Quick orientation

```text
transforms/
├── shift_column_transform.py        # Forward-looking target shifts
├── zscore_transform.py              # Stateless DataFrame normalization
├── winsorize_transform.py           # Percentile clipping
├── mad_transform.py                 # MAD-based clipping
├── quantile_trim_transform.py       # Quantile-based row trimming
├── calibrate_classifier.py          # Probability calibration
└── optimize_binary_threshold.py     # Validation-based threshold search
```

## Things to know

- Most helpers in this package are intentionally stateless per call.
- If you need fit-on-train and apply-to-all semantics, use a manifest fit parameter or a scaler rather than assuming a transform will remember state.
- `shift_column_transform` with `shift=-1` creates a future-looking label and therefore introduces a trailing null row that later prep stages drop.
- Calibration and threshold helpers live here because they transform model outputs rather than raw market data.

## Read next

- [Transforms](../../docs/Transforms.md)
- [Scalers](../../docs/Scalers.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
