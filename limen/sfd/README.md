# `limen.sfd`

> Package ready-to-run Single-File Decoders that plug directly into the Universal Experiment Loop.

## Canonical docs

- [Single-File Decoder](../../docs/Single-File-Decoder.md)
- [Built-In SFDs](../../docs/Built-In-SFDs.md)
- [Reference Architecture](../../docs/Reference-Architecture.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Universal Experiment Loop](../../docs/Universal-Experiment-Loop.md)

## In the research projects

The [Python project](../../docs/examples/python-research/README.md) supplies `params`, `prep` and `model` as ordinary caller-owned functions. The [Compiled project](../../docs/examples/compiled-research/README.md) names its pipeline and model in YAML and runs them through the CLI. The Python modules indexed here implement those families. Editing them is optional component development, described in the contributor appendix.

## What this package owns

Owns packaged SFD modules and the reference architectures they rely on.
Does **not** own experiment execution, data retrieval, or the lower-level indicator, feature, scaler, and transform libraries it composes.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `logreg_binary` | Standard manifest-driven binary-classification SFD | Exported at the package root |
| `random_binary` | Baseline classifier for comparison | Exported at the package root |
| `ridge_regressor` | Ridge next-return regression SFD | Under `foundational_sfd` |
| `xgboost_regressor` | Regression-style SFD | Exported at the package root |
| `foundational_rule_based` | Rule-based foundational SFD | Exported at the package root |
| `foundational_sfd` | Full catalog of packaged SFDs | Subpackage with production-oriented SFD modules |
| `reference_architecture` | Model-function templates without the packaged data pipeline | Starting point for custom SFD work |

## Adjacent modules

- `limen.experiment` runs SFDs and consumes `params()` plus either `manifest()` or custom `prep`/`model` functions.
- `limen.data`, `limen.indicators`, `limen.features`, `limen.transforms`, and `limen.scalers` provide the building blocks that manifest-driven SFDs wire together.
- `limen.metrics` is commonly used inside the reference architecture model functions.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `foundational_sfd/` | Packaged SFDs with params + manifest |
| `foundational_sfd/tabpfn_binary.py` | Optional TabPFN definition |
| `reference_architecture/` | Model-function implementations and templates |

<!-- reference: package-sfd-layout source=limen/sfd/README.md region=layout:package role=layout -->
```text
sfd/
├── README.md
├── __init__.py
├── foundational_sfd/
├── reference_architecture/
└── rule_based/
```

## Things to know

- A foundational SFD owns the experiment shape, while the matching reference architecture owns the model training logic.
- The lazy `tabpfn_binary` module and symbols are importable without the extra; constructing or training the TabPFN model requires `vaquum-limen[tabpfn]`.
- Component authors can copy an existing foundational SFD, adjust `params()`, and modify its manifest chain. Compiled Research users select and configure the corresponding YAML template.
- SFD modules are expected to stay cheap and stateless at import and manifest-construction time.

## Read next

- [Single-File Decoder](../../docs/Single-File-Decoder.md)
- [Built-In SFDs](../../docs/Built-In-SFDs.md)
- [Reference Architecture](../../docs/Reference-Architecture.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Developer: Contributing Foundational SFDs](../../docs/Developer/Contributing-Foundational-SFDs.md)
