# `limen.transforms`

> Apply stateless or fitted column-level transformations to Polars DataFrames during data preparation.

## Responsibilities

Owns reusable transform functions that normalise, clip, shift, or calibrate individual columns in a DataFrame.
Does **not** own feature engineering (that lives in `limen.features`) or model fitting — transforms operate purely on data values, not on model parameters.

## Key concepts

- **Stateless transform** – takes a DataFrame and parameters, returns a transformed DataFrame; no fitting required (e.g. `zscore_transform`, `shift_column_transform`).
- **Fitted transform** – computes statistics on the training split and stores them; the same statistics are then applied to val/test splits via `Manifest`'s `add_fitted_transform` mechanism (e.g. `winsorize_transform`, `mad_transform`, `quantile_trim_transform`).
- **`shift_column_transform`** – shifts a target column by N bars to create a forward-looking label; `shift=-1` shifts labels one bar into the future relative to features.
- **`calibrate_classifier`** – wraps a trained sklearn classifier with Platt scaling (sigmoid calibration) to produce well-calibrated probabilities.
- **`optimize_binary_threshold`** – finds the classification threshold that maximises a given metric (e.g. F1) on validation data.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `zscore_transform(df)` | `zscore_transform.py` | Stateless Z-score of all numeric columns; used in early-stage prep |
| `shift_column_transform(df, shift, column)` | `shift_column_transform.py` | Forward-shift target column; added to manifest `with_target()` block |
| `winsorize_transform(df, ...)` | `winsorize_transform.py` | Clip extreme values at configurable percentile bounds |
| `mad_transform(df, ...)` | `mad_transform.py` | Median-absolute-deviation outlier clipping |
| `quantile_trim_transform(df, ...)` | `quantile_trim_transform.py` | Remove rows outside quantile bounds |
| `calibrate_classifier(model, x_val, y_val)` | `calibrate_classifier.py` | Post-training probability calibration |
| `optimize_binary_threshold(probs, y_val)` | `optimize_binary_threshold.py` | Find optimal classification decision threshold |

## Dependencies

- **Internal:** consumed by `limen.experiment.manifest_core` via `Manifest.with_target().add_transform()` and `add_fitted_transform()`
- **External:** `polars`, `scikit-learn` (for `calibrate_classifier` and `optimize_binary_threshold`)

## Quick orientation
```text
transforms/
├── zscore_transform.py              # Stateless Z-score normalisation
├── shift_column_transform.py        # Forward-shift a target column by N bars
├── winsorize_transform.py           # Percentile-based value clipping
├── mad_transform.py                 # MAD-based outlier clipping
├── quantile_trim_transform.py       # Row removal outside quantile range
├── calibrate_classifier.py          # Platt scaling for probability calibration
└── optimize_binary_threshold.py     # Threshold optimisation for binary classifiers
```

## Gotchas / things to know

- `shift_column_transform` with `shift=-1` creates a forward-looking label; the last row will have a null and is dropped by `drop_nulls()` in the manifest pipeline.
- Fitted transforms (winsorize, MAD) must be placed in `add_fitted_transform()` blocks in the manifest, not `add_transform()`, so that fit statistics are computed only on the training split.
- `zscore_transform` excludes the `datetime` column by default; pass `time_col=''` to disable this exclusion.
