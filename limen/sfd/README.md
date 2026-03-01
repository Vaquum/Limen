# `limen.sfd`

> Ready-to-use Single File Decoder (SFD) implementations for common model types.

## Responsibilities

Provides pre-built SFDs that wrap a complete prep + model pipeline in a single importable object, enabling experiments to be launched with minimal boilerplate.

Does **not** own the experiment loop or data fetching — SFDs are configuration objects consumed by `UniversalExperimentLoop`.

## Key concepts

- **SFD (Single File Decoder)** – an object (or module) with a `params()` method and either a `manifest()` method (manifest-driven) or `prep()` + `model()` callables (legacy custom-functions approach)
- **foundational_sfd** – the primary sub-package; contains battle-tested SFD implementations for standard model types
- **reference_architecture** – example SFDs demonstrating patterns for writing new SFDs; not production models
- **logreg_binary** – logistic regression binary classifier SFD
- **random_binary** – random baseline binary classifier (useful for benchmarking)
- **xgboost_regressor** – XGBoost regression SFD
- **tabpfn_binary** – TabPFN binary classifier SFD (optional; requires `tabpfn` package)

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `limen.sfd.logreg_binary` | `foundational_sfd/logreg_binary.py` | Pass as `sfd=` to UEL for logistic regression experiments |
| `limen.sfd.random_binary` | `foundational_sfd/random_binary.py` | Baseline random model for sanity-checking pipelines |
| `limen.sfd.xgboost_regressor` | `foundational_sfd/xgboost_regressor.py` | XGBoost regression experiments |
| `limen.sfd.tabpfn_binary` | `foundational_sfd/tabpfn_binary.py` | TabPFN classification (import-guarded; `None` if `tabpfn` not installed) |

## Dependencies

- **Internal:** `limen.experiment.Manifest`, `limen.experiment.UniversalExperimentLoop`
- **External:** `scikit-learn`, `xgboost`, `tabpfn` (optional)

## Quick orientation

```text
sfd/
├── foundational_sfd/
│   ├── logreg_binary.py      # Logistic regression binary SFD
│   ├── random_binary.py      # Random baseline SFD
│   ├── xgboost_regressor.py  # XGBoost regression SFD
│   └── tabpfn_binary.py      # TabPFN binary SFD (optional)
└── reference_architecture/
    ├── logreg_binary.py      # Reference pattern for manifest-driven SFDs
    ├── random_binary.py      # Reference pattern for baseline SFDs
    ├── xgboost_regressor.py  # Reference pattern for regression SFDs
    └── tabpfn_binary.py      # Reference pattern for optional-dep SFDs
```

## Gotchas / things to know

- `tabpfn_binary` is `None` when the `tabpfn` package is not installed; guard against this before passing it to UEL
- SFDs in `reference_architecture/` are for reading and copying from — they are not imported into any production pipeline
- Each SFD exposes its hyperparameter search space through `params()`; changing these values is the primary lever for tuning experiment scope
