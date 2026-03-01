# `limen.transforms`

> Apply stateless and fitted data transformations to Polars DataFrames during pipeline preparation.

## Responsibilities

Provides reusable column-level transforms — outlier clipping, normalisation, threshold optimisation, and classifier calibration — that slot into the `Manifest` pipeline as feature transforms or target transforms.

Does **not** own feature engineering logic or model training; transforms are purely data-shaping utilities.

## Key concepts

- **zscore_transform** – standardises all numeric columns (mean=0, std=1) in-place; excludes the datetime column
- **mad_transform** – median absolute deviation normalisation; more robust to outliers than Z-score
- **winsorize_transform** – clips extreme values at configurable percentile bounds
- **quantile_trim_transform** – removes rows whose values fall outside a specified quantile range
- **shift_column_transform** – shifts a column forward/backward by N bars (used to align targets with features)
- **calibrate_classifier** – applies Platt or isotonic calibration to a trained sklearn classifier to produce well-calibrated probabilities
- **optimize_binary_threshold** – finds the decision threshold that maximises a chosen metric (e.g. F1) on the validation set

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `zscore_transform()` | `zscore_transform.py` | Pass to `Manifest.add_feature()` to normalise a split |
| `mad_transform()` | `mad_transform.py` | Robust alternative to Z-score normalisation |
| `winsorize_transform()` | `winsorize_transform.py` | Clip outliers before model training |
| `quantile_trim_transform()` | `quantile_trim_transform.py` | Drop rows with out-of-range values |
| `shift_column_transform()` | `shift_column_transform.py` | Align a target column temporally with input features |
| `calibrate_classifier()` | `calibrate_classifier.py` | Calibrate probability outputs of a fitted classifier |
| `optimize_binary_threshold()` | `optimize_binary_threshold.py` | Search for the best decision threshold on validation data |

## Dependencies

- **Internal:** none
- **External:** `polars`, `numpy`, `scikit-learn` (`CalibratedClassifierCV`, `metrics`)

## Quick orientation

```text
transforms/
├── zscore_transform.py              # Global Z-score normalisation
├── mad_transform.py                 # MAD-based normalisation
├── winsorize_transform.py           # Percentile-based clipping
├── quantile_trim_transform.py       # Row-level outlier removal
├── shift_column_transform.py        # Temporal column shift
├── calibrate_classifier.py          # Probability calibration
└── optimize_binary_threshold.py     # Decision threshold search
```

## Gotchas / things to know

- `zscore_transform` computes statistics from the input DataFrame itself; when used inside the Manifest it will be applied per-split — fit on training, transform all — so use `Manifest.set_scaler()` with `LinearScaler` instead if you need train-fitted statistics applied to val/test
- `shift_column_transform` introduces leading `null` values; `Manifest.prepare_data()` calls `drop_nulls()` after transforms
- `calibrate_classifier` wraps a fitted estimator; the calibrated model replaces the original and must be re-serialised if persistence is required
