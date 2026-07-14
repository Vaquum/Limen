# `limen.sfd`

> Package ready-to-run Single-File Decoders that plug directly into the Universal Experiment Loop.

## Canonical docs

- [Single-File Decoder](../../docs/Single-File-Decoder.md)
- [Built-In SFDs](../../docs/Built-In-SFDs.md)
- [Reference Architecture](../../docs/Reference-Architecture.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Universal Experiment Loop](../../docs/Universal-Experiment-Loop.md)

## What this package owns

Owns packaged SFD modules and the reference architectures they rely on.
Does **not** own experiment execution, data retrieval, or the lower-level indicator, feature, scaler, and transform libraries it composes.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `logreg_binary` | Standard manifest-driven binary-classification SFD | Exported at the package root |
| `random_binary` | Baseline classifier for comparison | Exported at the package root |
| `xgboost_regressor` | Regression-style SFD | Exported at the package root |
| `foundational_rule_based` | Rule-based foundational SFD | Exported at the package root |
| `foundational_sfd` | Full catalog of packaged SFDs | Subpackage with production-oriented SFD modules |
| `reference_architecture` | Model-function templates without the packaged data pipeline | Starting point for custom SFD work |

## Adjacent modules

- `limen.experiment` runs SFDs and consumes `params()` plus either `manifest()` or custom `prep`/`model` functions.
- `limen.data`, `limen.indicators`, `limen.features`, `limen.transforms`, and `limen.scalers` provide the building blocks that manifest-driven SFDs wire together.
- `limen.metrics` is commonly used inside the reference architecture model functions.

## Quick orientation

```text
sfd/
├── foundational_sfd/          # Packaged SFDs with params + manifest
│   ├── logreg_binary.py
│   ├── dlinear_regressor.py
│   ├── lightgbm_binary.py
│   ├── random_binary.py
│   ├── rule_based.py
│   ├── xgboost_regressor.py
│   └── tabpfn_binary.py       # Optional
└── reference_architecture/    # Model-function implementations and templates
    ├── logreg_binary.py
    ├── dlinear_regressor.py
    ├── lightgbm_binary.py
    ├── random_binary.py
    ├── rule_based.py
    ├── xgboost_regressor.py
    └── tabpfn_binary.py
```

## Things to know

- A foundational SFD owns the experiment shape, while the matching reference architecture owns the model training logic.
- The lazy `tabpfn_binary` module and symbols are importable without the extra; constructing or training the TabPFN model requires `vaquum-limen[tabpfn]`.
- The simplest path to a new SFD is to copy an existing foundational SFD, adjust `params()`, and modify the manifest chain.
- SFD modules are expected to stay cheap and stateless at import and manifest-construction time.

## Read next

- [Single-File Decoder](../../docs/Single-File-Decoder.md)
- [Built-In SFDs](../../docs/Built-In-SFDs.md)
- [Reference Architecture](../../docs/Reference-Architecture.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Developer: Contributing Foundational SFDs](../../docs/Developer/Contributing-Foundational-SFDs.md)
