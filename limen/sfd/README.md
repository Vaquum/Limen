# `limen.sfd`

> Provide ready-to-run Single File Decoders (SFDs) — self-contained experiment configurations that pair a parameter search space with a manifest-driven data/model pipeline.

## Responsibilities

Owns pre-built SFD modules for common model architectures.
Does **not** own the experiment loop, feature engineering, or metrics — SFDs are configurations that wire together the building blocks from `limen.experiment`, `limen.features`, `limen.indicators`, `limen.scalers`, and `limen.transforms`.

## Key concepts

- **SFD (Single File Decoder)** – a Python module that exposes two functions: `params() → dict` (search space) and `manifest() → Manifest` (declarative pipeline config). Passed directly to `UniversalExperimentLoop(sfd=my_sfd)`.
- **foundational_sfd** – production-ready SFDs with real data sources configured and tuned parameter grids.
- **reference_architecture** – model function stubs and templates; used as the `with_model()` target inside `foundational_sfd` manifests, and as a starting point for writing custom SFDs.
- **Model function** – the callable passed to `Manifest.with_model()`; receives `(data: dict, **round_params)` and returns a results dict with metrics and optionally `_preds`, `models`, or `extras`.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `logreg_binary` | `foundational_sfd/logreg_binary.py` | Re-exported from `limen.sfd` for direct use in `UniversalExperimentLoop` |
| `random_binary` | `foundational_sfd/random_binary.py` | Re-exported from `limen.sfd` for baseline benchmarking |
| `xgboost_regressor` | `foundational_sfd/xgboost_regressor.py` | Re-exported from `limen.sfd` for regression experiments |
| `foundational_sfd.tabpfn_binary` | `foundational_sfd/tabpfn_binary.py` | Optional TabPFN SFD available through the subpackage when `tabpfn` is installed |

## Dependencies

- **Internal:** `limen.experiment` (Manifest), `limen.features`, `limen.indicators`, `limen.scalers`, `limen.transforms`, `limen.data`
- **External:** `scikit-learn`, `xgboost` (for XGBoost SFD), `tabpfn` (optional, for TabPFN SFD)

## Quick orientation
```text
sfd/
├── foundational_sfd/          # Production SFDs (params + manifest)
│   ├── logreg_binary.py       # Logistic Regression binary classifier
│   ├── random_binary.py       # Random baseline
│   ├── xgboost_regressor.py   # XGBoost regressor
│   └── tabpfn_binary.py       # TabPFN classifier (optional)
└── reference_architecture/    # Model function templates (no data source)
    ├── logreg_binary.py       # LR model function only
    ├── random_binary.py       # Random model function only
    ├── xgboost_regressor.py   # XGBoost model function only
    └── tabpfn_binary.py       # TabPFN model function only
```

## Gotchas / things to know

- `tabpfn_binary` is imported with a `try/except ImportError` guard; it will be `None` if `tabpfn` is not installed.
- `foundational_sfd` modules call `reference_architecture` model functions via `with_model(logreg_binary)` — the reference architecture contains the sklearn training logic, while the foundational SFD adds the data pipeline.
- To create a new SFD: copy a `foundational_sfd` file, adjust `params()`, and modify the `manifest()` chain. No subclassing required.
- The `manifest()` function is called once per UEL construction; it should be cheap and stateless.
