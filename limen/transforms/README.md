# `limen.transforms`

> Apply lightweight data and model-output transformations used during preparation, calibration, and threshold selection.

## Responsibilities

Owns reusable helpers that normalise, clip, shift, calibrate, or threshold data and model outputs.
Does **not** own feature engineering (that lives in `limen.features`) or fit-on-train feature scaling (that lives in `limen.scalers`).

## Key concepts

- **DataFrame transform** - takes a DataFrame and parameters, returns a transformed DataFrame (e.g. `zscore_transform`, `shift_column_transform`, `winsorize_transform`).
- **Model-output helper** - operates on fitted classifiers or probability arrays rather than on raw DataFrames (e.g. `calibrate_classifier`, `optimize_binary_threshold`).
- **`shift_column_transform`** – shifts a target column by N bars to create a forward-looking label; `shift=-1` shifts labels one bar into the future relative to features.
- **`calibrate_classifier`** – wraps a trained sklearn classifier with isotonic or sigmoid calibration to produce better-calibrated probabilities.
- **`optimize_binary_threshold`** – finds the classification threshold that maximises a given metric (e.g. F1) on validation data.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `zscore_transform(df)` | `zscore_transform.py` | Stateless Z-score of all numeric columns; used in early-stage prep |
| `shift_column_transform(df, shift, column)` | `shift_column_transform.py` | Forward-shift target column; added to manifest `with_target()` block |
| `winsorize_transform(df, ...)` | `winsorize_transform.py` | Clip extreme values at configurable percentile bounds |
| `mad_transform(df, ...)` | `mad_transform.py` | Median-absolute-deviation outlier clipping |
| `quantile_trim_transform(df, ...)` | `quantile_trim_transform.py` | Remove rows outside quantile bounds |
| `calibrate_classifier(clf, x_val, y_val, x_sets)` | `calibrate_classifier.py` | Post-training probability calibration |
| `optimize_binary_threshold(y_val, y_val_proba, ...)` | `optimize_binary_threshold.py` | Find optimal classification decision threshold |

## Dependencies

- **Internal:** consumed by manifest target construction, custom prep flows, and reference architectures
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
- `winsorize_transform`, `mad_transform`, and `quantile_trim_transform` operate on whichever frame you pass in. If you need train-only fitting semantics, orchestrate that explicitly in the manifest or in custom prep logic.
- `zscore_transform` excludes the `datetime` column by default; pass `time_col=''` to disable this exclusion.
