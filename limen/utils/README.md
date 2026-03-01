# `limen.utils`

> Miscellaneous utilities shared across the Limen library.

## Responsibilities

Collects small, general-purpose helpers that do not belong to a single domain module: parameter space generation, data-dict conversion, Optuna integration, confidence filtering, and report formatting.

Does **not** own any experiment loop logic, model training, or data fetching.

## Key concepts

- **ParamSpace** – generates one dict of parameter values per experiment round, supporting both random and grid search modes; consumed internally by `UniversalExperimentLoop`
- **data_dict_to_numpy** – converts the structured data dict (train/val/test Polars DataFrames) into NumPy arrays ready for scikit-learn or XGBoost models
- **confidence_filtering_system** – calibrates a prediction-uncertainty threshold on validation data and applies it to test predictions to tag low-confidence bars
- **log_to_optuna_study** – imports `Log` experiment results into an Optuna study for visualisation or further hyperparameter analysis
- **reporting** – lightweight ASCII report formatting helpers (`format_report_header`, `format_report_section`, `format_report_footer`)

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `ParamSpace` | `param_space.py` | Used by UEL to generate `round_params` each iteration |
| `data_dict_to_numpy()` | `data_dict_to_numpy.py` | Call inside an SFD's `model()` function to get `x_train`, `y_train`, etc. |
| `confidence_filtering_system()` | `confidence_filtering_system.py` | Post-training: calibrate on val, apply to test, return confidence-tagged DataFrame |
| `log_to_optuna_study()` | `log_to_optuna_study.py` | Convert a completed `Log` into an Optuna study for analysis |
| `format_report_header()` / `format_report_section()` / `format_report_footer()` | `reporting.py` | Compose a readable plain-text experiment summary |

## Dependencies

- **Internal:** `limen.metrics` (`binary_metrics`, `continuous_metrics`, `safe_ovr_auc`)
- **External:** `numpy`, `polars`, `scikit-learn`, `optuna` (optional, for `log_to_optuna_study`)

## Quick orientation

```text
utils/
├── param_space.py                   # ParamSpace — random/grid parameter sampling
├── data_dict_to_numpy.py            # Convert data dict → NumPy arrays
├── confidence_filtering_system.py   # Validation-calibrated prediction confidence
├── log_to_optuna_study.py           # Export Log results to Optuna
└── reporting.py                     # Plain-text report formatting helpers
```

## Gotchas / things to know

- `ParamSpace` is an internal helper used by UEL; direct instantiation is only needed when building custom experiment drivers
- `data_dict_to_numpy` expects keys `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test` to exist in the data dict; the Manifest pipeline guarantees these
- `confidence_filtering_system` requires an ensemble of models (a list); it measures inter-model prediction variance as the uncertainty proxy — it does not work with a single model
- `log_to_optuna_study` imports `optuna` lazily; if Optuna is not installed the function will raise `ImportError` at call time
