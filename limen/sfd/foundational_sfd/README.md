# `limen.sfd.foundational_sfd`

> The catalog of packaged, ready-to-run Single-File Decoders.

## Canonical docs

- [Built-In SFDs](../../../docs/Built-In-SFDs.md)
- [Single-File Decoder](../../../docs/Single-File-Decoder.md)

## In the research projects

The [Python project](../../../docs/examples/python-research/README.md) defines its own SFD functions. The [Compiled project](../../../docs/examples/compiled-research/README.md) selection and `search` scenarios use the declared model families wrapped here. The family reference separates those definitions from the model implementations they call.

## What this package owns

Owns the production-oriented, packaged SFD modules — each defining an experiment shape via `params()` and a `manifest()` that wires the data pipeline together.
Does **not** own model training logic (owned by the matching `limen.sfd.reference_architecture` model) or experiment execution.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `common_strategies` | Declared comparison of the eight named baseline strategies | Rule-based inputs and execution |
| `dollar_bar_crash_reversal` | Crash-reversal strategy on dollar-bar features | Requires the declared dollar-bar input contract |
| `logreg_binary` | Standard binary-classification SFD | Manifest-driven |
| `dlinear_regressor` | Linear time-series regression SFD | Manifest-driven |
| `lightgbm_binary` | Gradient-boosted binary classifier SFD | Manifest-driven |
| `ridge_regressor` | Ridge next-return regression SFD | Manifest-driven |
| `xgboost_regressor` | Regression-style SFD | Manifest-driven |
| `random_binary` | Baseline classifier for comparison | Manifest-driven |
| `rule_based` | Predicate-driven, non-learned SFD | Uses `limen.sfd.rule_based` |
| `tabpfn_binary` | TabPFN-based SFD | Module is lazy; model use requires the `tabpfn` extra |

## Adjacent modules

- `limen.sfd.reference_architecture` supplies the model function each foundational SFD delegates training to.
- `limen.experiment` runs these SFDs, consuming their `params()` and `manifest()`.
- `limen.data`, `limen.indicators`, `limen.features`, `limen.transforms`, and `limen.scalers` are the building blocks the manifests wire together.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `tabpfn_binary.py` | Optional |

<!-- reference: package-sfd-foundational_sfd-layout source=limen/sfd/foundational_sfd/README.md region=layout:package role=layout -->
```text
foundational_sfd/
├── README.md
├── __init__.py
├── _tabpfn_params.py
├── _xgboost_features.py
├── cnn1d_binary.py
├── common_strategies.py
├── dlinear_regressor.py
├── dollar_bar_crash_reversal.py
├── entropy_var_regressor.py
├── gaussian_hmm_binary.py
├── lightgbm_binary.py
├── logreg_binary.py
├── lstm_binary.py
├── random_binary.py
├── rbf_svm_binary.py
├── residual_regressor.py
├── ridge_regressor.py
├── rule_based.py
├── tabpfn_binary.py
├── weight_link_binary.py
└── xgboost_regressor.py
```

## Things to know

- Exports are **lazy**: the package uses module-level `__getattr__`, so importing `foundational_sfd` does not import heavy model dependencies until a specific SFD is accessed. This is why `tabpfn_binary` can be listed without `tabpfn` installed.
- A foundational SFD owns the experiment shape; the matching reference architecture owns the training logic.
- The simplest path to a new SFD is to copy an existing module, adjust `params()`, and modify the manifest chain.

## Read next

- [Built-In SFDs](../../../docs/Built-In-SFDs.md)
- [Reference Architecture](../../../docs/Reference-Architecture.md)
- [Developer: Contributing Foundational SFDs](../../../docs/Developer/Contributing-Foundational-SFDs.md)
