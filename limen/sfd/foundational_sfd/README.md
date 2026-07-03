# `limen.sfd.foundational_sfd`

> The catalog of packaged, ready-to-run Single-File Decoders.

## Canonical docs

- [Built-In SFDs](../../../docs/Built-In-SFDs.md)
- [Single-File Decoder](../../../docs/Single-File-Decoder.md)

## What this package owns

Owns the production-oriented, packaged SFD modules — each defining an experiment shape via `params()` and a `manifest()` that wires the data pipeline together.
Does **not** own model training logic (owned by the matching `limen.sfd.reference_architecture` model) or experiment execution.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `logreg_binary` | Standard binary-classification SFD | Manifest-driven |
| `lightgbm_binary` | Gradient-boosted binary classifier SFD | Manifest-driven |
| `xgboost_regressor` | Regression-style SFD | Manifest-driven |
| `random_binary` | Baseline classifier for comparison | Manifest-driven |
| `rule_based` | Predicate-driven, non-learned SFD | Uses `limen.sfd.rule_based` |
| `tabpfn_binary` | TabPFN-based SFD | Optional; unavailable when `tabpfn` is not installed |

## Adjacent modules

- `limen.sfd.reference_architecture` supplies the model function each foundational SFD delegates training to.
- `limen.experiment` runs these SFDs, consuming their `params()` and `manifest()`.
- `limen.data`, `limen.indicators`, `limen.features`, `limen.transforms`, and `limen.scalers` are the building blocks the manifests wire together.

## Quick orientation

```text
foundational_sfd/
├── logreg_binary.py
├── lightgbm_binary.py
├── xgboost_regressor.py
├── random_binary.py
├── rule_based.py
└── tabpfn_binary.py     # Optional
```

## Things to know

- Exports are **lazy**: the package uses module-level `__getattr__`, so importing `foundational_sfd` does not import heavy model dependencies until a specific SFD is accessed. This is why `tabpfn_binary` can be listed without `tabpfn` installed.
- A foundational SFD owns the experiment shape; the matching reference architecture owns the training logic.
- The simplest path to a new SFD is to copy an existing module, adjust `params()`, and modify the manifest chain.

## Read next

- [Built-In SFDs](../../../docs/Built-In-SFDs.md)
- [Reference Architecture](../../../docs/Reference-Architecture.md)
- [Developer: Contributing Foundational SFDs](../../../docs/Developer/Contributing-Foundational-SFDs.md)
