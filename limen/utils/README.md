# `limen.utils`

> Hold the smaller cross-cutting helpers that support experiments without belonging to one primary domain package.

## Canonical docs

- [Universal Experiment Loop](../../docs/Universal-Experiment-Loop.md)
- [Standard Metrics Library](../../docs/Standard-Metrics-Library.md)
- [Log](../../docs/Log.md)

## What this package owns

Owns general utilities such as legacy parameter sampling, `data_dict` conversion, confidence filtering, report formatting, and Optuna export helpers.
Does **not** own the canonical experiment, feature, metric, or backtest surfaces.

## Key entry points

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `ParamSpace` | You are on the legacy basic `UEL.run()` path and need permutation sampling | Advanced runs use `SearchStrategy` and `ParamDomain` instead |
| `data_dict_to_numpy` | You want numpy arrays from the standard Limen `data_dict` | Common inside sklearn-style model functions |
| `confidence_filtering_system` | You want post-prediction filtering based on model agreement | An optional downstream helper, not part of the main UEL contract |
| `log_to_optuna_study` | You want to export experiment results into Optuna | Requires optional `optuna` |
| Reporting helpers | You want formatted text summaries | Utility surface, not a canonical reporting framework |

## Adjacent modules

- `limen.experiment` uses `ParamSpace` on the legacy run path.
- `limen.metrics` provides the canonical scoring helpers that this package partially re-exports for convenience.
- `limen.sfd.reference_architecture` often calls `data_dict_to_numpy`.

## Quick orientation

```text
utils/
├── param_space.py                 # Legacy permutation sampler
├── data_dict_to_numpy.py          # Convert Limen data_dict to numpy arrays
├── confidence_filtering_system.py # Confidence-based prediction filtering
├── log_to_optuna_study.py         # Optuna export
└── reporting.py                   # Text-formatting helpers
```

## Things to know

- This package is intentionally mixed. If a helper grows into a coherent subsystem, it should usually move out of `utils`.
- `ParamSpace` is the legacy path, not the long-term abstraction for advanced search.
- `data_dict_to_numpy` assumes the standard Limen split schema and is most useful inside model code.
- `optuna` is optional and only required for the study-export helper.

## Read next

- [Universal Experiment Loop](../../docs/Universal-Experiment-Loop.md)
- [Standard Metrics Library](../../docs/Standard-Metrics-Library.md)
- [Single-File Decoder](../../docs/Single-File-Decoder.md)
