# `limen.sfd.reference_architecture`

> Model-function implementations and templates that own the training logic behind each SFD.

## Canonical docs

- [Reference Architecture](../../../docs/Reference-Architecture.md)

## What this package owns

Owns the reference model classes (and their function forms) that implement fit/predict for each decoder family, without the packaged data pipeline.
Does **not** own the experiment shape or manifest (owned by the matching `limen.sfd.foundational_sfd` module) or experiment execution.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `ReferenceModel` | Base class for reference models | Shared fit/predict contract |
| `LogRegBinary` / `logreg_binary` | Logistic-regression binary model | Class and function forms |
| `DLinearRegressor` / `dlinear_regressor` | Linear time-series regressor | Class and function forms |
| `LightGBMBinary` / `lightgbm_binary` | LightGBM binary classifier | Class and function forms |
| `XGBoostRegressor` / `xgboost_regressor` | XGBoost regressor | Class and function forms |
| `RandomBinary` / `random_binary` | Baseline random classifier | Class and function forms |
| `RuleBasedStrategy` / `rule_based` | Predicate-driven, non-learned strategy | Uses `limen.sfd.rule_based` |
| `TabPFNBinary` / `tabpfn_binary` | TabPFN binary classifier | Optional; requires `tabpfn` |

## Adjacent modules

- `limen.sfd.foundational_sfd` pairs each of these models with an experiment shape and manifest.
- `limen.inference.Sensor` wraps a `ReferenceModel` for live bar-by-bar prediction.
- `limen.metrics` is commonly used inside these model functions to score outputs.

## Quick orientation

```text
reference_architecture/
├── base.py               # ReferenceModel base class
├── logreg_binary.py
├── dlinear_regressor.py
├── lightgbm_binary.py
├── xgboost_regressor.py
├── random_binary.py
├── rule_based.py
└── tabpfn_binary.py      # Optional
```

## Things to know

- Each module exposes both a class (e.g. `LogRegBinary`) and a function form (e.g. `logreg_binary`); pick the class for custom subclassing and the function for manifest wiring.
- These models are the template for custom SFD work: subclass `ReferenceModel` or copy a module to start.
- A reference architecture owns the training logic; the matching foundational SFD owns the manifest and data pipeline.

## Read next

- [Reference Architecture](../../../docs/Reference-Architecture.md)
- [Single-File Decoder](../../../docs/Single-File-Decoder.md)
- [Built-In SFDs](../../../docs/Built-In-SFDs.md)
