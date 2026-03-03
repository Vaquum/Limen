# `limen.utils`

> Provide cross-cutting utilities for parameter sampling, reporting, confidence filtering, and experiment result export.

## Responsibilities

Owns generic helpers that don't belong to a single domain: hyperparameter space management, formatted report generation, confidence-based prediction filtering, numpy data-dict conversion, and Optuna study export.
Does **not** own domain-specific logic such as feature engineering, model training, or backtesting.

## Key concepts

- **ParamSpace** – samples a fixed-size, deduplicated set of hyperparameter combinations from a `params` dict; used by the legacy `UniversalExperimentLoop.run()` path (the advanced loop uses `ParamDomain` + `SearchStrategy` instead).
- **confidence_filtering_system** – filters model predictions based on ensemble agreement (prediction variance), retaining only "confident" bars above a calibrated threshold.
- **data_dict_to_numpy** – converts the `data_dict` produced by the prep pipeline into numpy arrays suitable for sklearn estimators.
- **log_to_optuna_study** – exports experiment log results into an Optuna `Study` object for further analysis or visualisation.
- **Reporting helpers** – `format_report_header`, `format_report_section`, `format_report_footer` produce human-readable text blocks for experiment summaries.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `ParamSpace(params, n_permutations)` | `param_space.py` | Used internally by the basic UEL `.run()` to sample parameter combos |
| `data_dict_to_numpy(data_dict, ...)` | `data_dict_to_numpy.py` | Inside an SFD model function to get `x_train`, `y_train`, etc. as numpy arrays |
| `confidence_filtering_system(...)` | `confidence_filtering_system.py` | Post-prediction step to mask low-confidence bars |
| `log_to_optuna_study(log, ...)` | `log_to_optuna_study.py` | Convert a `Log.experiment_log` to an Optuna study for analysis |
| `format_report_header/section/footer` | `reporting.py` | Build formatted text summaries of experiment results |

## Dependencies

- **Internal:** `limen.metrics` (re-exported `binary_metrics`, `continuous_metrics`, `safe_ovr_auc` for convenience)
- **External:** `numpy`, `polars`, `scikit-learn`, `optuna` (for `log_to_optuna_study`)

## Quick orientation
```text
utils/
├── param_space.py                   # ParamSpace — deduped hyperparameter sampler
├── data_dict_to_numpy.py            # Convert data_dict to numpy train/val/test arrays
├── confidence_filtering_system.py   # Ensemble confidence-based prediction filter
├── log_to_optuna_study.py           # Export experiment log to Optuna Study
└── reporting.py                     # Text report formatting helpers
```

## Gotchas / things to know

- `ParamSpace` pre-generates all combinations at construction time and removes them as they are consumed; it is exhausted after `n_permutations` calls to `.generate()`.
- `data_dict_to_numpy` expects the standard `data_dict` schema produced by `split_data_to_prep_output`; keys like `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test` are extracted automatically.
- `confidence_filtering_system` requires a list of trained models and validation data; it calibrates a variance threshold so approximately `target_confidence` fraction of predictions are retained.
- `log_to_optuna_study` requires `optuna` to be installed; it is not a hard dependency of the package.
