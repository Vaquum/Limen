# `limen.sfd.reference_architecture`

> Model-function implementations and templates that own the training logic behind each SFD.

## Canonical docs

- [Reference Architecture](../../../docs/Reference-Architecture.md)

## In the research projects

The [Python project](../../../docs/examples/python-research/README.md) fits its ordinary caller-owned Ridge pipeline. The [Compiled project](../../../docs/examples/compiled-research/README.md) `search` scenario runs the declared CPU model comparisons, retains their metrics and prediction payloads; the selection scenario separately proves CLI reconstruction. The architecture reference describes the prepared-data and output contracts.

## What this package owns

Owns the reference model classes (and their function forms) that implement fit/predict for each decoder family, without the packaged data pipeline.
Does **not** own the experiment shape or manifest (owned by the matching `limen.sfd.foundational_sfd` module) or experiment execution.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `DirectionSizingModel` / `direction_sizing` | Direction and signed exposure model | Native allocation/execution contract |
| `meta_allocation.py` | Fold-qualified allocation labels and policy fitting | Used by allocation manifests; distinct from entry classification |
| `ReferenceModel` | Base class for reference models | Shared fit/predict contract |
| `LogRegBinary` / `logreg_binary` | Logistic-regression binary model | Class and function forms |
| `DLinearRegressor` / `dlinear_regressor` | Linear time-series regressor | Class and function forms |
| `LightGBMBinary` / `lightgbm_binary` | LightGBM binary classifier | Class and function forms |
| `RidgeRegressor` / `ridge_regressor` | sklearn ridge regressor | Class and function forms |
| `XGBoostRegressor` / `xgboost_regressor` | XGBoost regressor | Class and function forms |
| `RandomBinary` / `random_binary` | Baseline random classifier | Class and function forms |
| `RuleBasedStrategy` / `rule_based` | Predicate-driven, non-learned strategy | Uses `limen.sfd.rule_based` |
| `TabPFNBinary` / `tabpfn_binary` | TabPFN binary classifier | Optional; requires `tabpfn` |

## Adjacent modules

- `limen.sfd.foundational_sfd` pairs each of these models with an experiment shape and manifest.
- `limen.inference.Sensor` wraps a `ReferenceModel` for live bar-by-bar prediction.
- `limen.metrics` is commonly used inside these model functions to score outputs.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `base.py` | ReferenceModel base class |
| `tabpfn_binary.py` | Optional |

<!-- reference: package-sfd-reference_architecture-layout source=limen/sfd/reference_architecture/README.md region=layout:package role=layout -->
```text
reference_architecture/
├── README.md
├── __init__.py
├── _backtest_evaluation.py
├── _sequence_binary.py
├── base.py
├── cnn1d_binary.py
├── direction_sizing.py
├── dlinear_regressor.py
├── gaussian_hmm_binary.py
├── lightgbm_binary.py
├── logreg_binary.py
├── lstm_binary.py
├── meta_allocation.py
├── random_binary.py
├── rbf_svm_binary.py
├── ridge_regressor.py
├── rule_based.py
├── tabpfn_binary.py
└── xgboost_regressor.py
```

## Things to know

- Each module exposes both a class (e.g. `LogRegBinary`) and a function form (e.g. `logreg_binary`); pick the class for custom subclassing and the function for manifest wiring.
- These models are the template for custom SFD work: subclass `ReferenceModel` or copy a module to start.
- A reference architecture owns the training logic; the matching foundational SFD owns the manifest and data pipeline.

## Read next

- [Reference Architecture](../../../docs/Reference-Architecture.md)
- [Single-File Decoder](../../../docs/Single-File-Decoder.md)
- [Built-In SFDs](../../../docs/Built-In-SFDs.md)
