# Glossary

This glossary defines the standard Limen terms used across public docs, developer docs, package README files, and issue work.

## Core terms

| Term | Meaning |
|---|---|
| `Single-File Decoder` | A self-contained Limen experiment unit that defines a parameter surface, manifest pipeline, and model reference for one research task. Previously called Single File Model. |
| `SFD` | Short form of `Single-File Decoder`. Previously called `SFM`. |
| `Foundational SFD` | A packaged SFD under `limen.sfd.foundational_sfd` that uses one of Limen's reference architectures and usually has a matching YAML template. |
| `Custom SFD` | A user-authored SFD that is not one of the packaged foundational SFDs. It may still use Limen manifests, features, targets, scalers, and reference architectures. |
| `Reference Architecture` | The model implementation contract used by an SFD, such as `LogRegBinary` or `LightGBMBinary`. |
| `Custom Architecture` | A user-authored model implementation that is not one of Limen's reference architectures. |
| `manifest_core` | The underlying manifest implementation in `limen.experiment.manifest_core`; all manifest authoring surfaces compile toward this core. |
| `manifest` | Any wrapper over `manifest_core`: a `def manifest():` function in an SFD, a YAML manifest, or a GUI-emitted Design of Experiment manifest. |
| `experiment_core` | The execution engine in `limen.experiment.experiment_core`; previously called `UniversalExperimentLoop.py`. |
| `experiment` | One full `experiment_core` run for the configured `n_permutations`, including round execution, logs, and post-run artifacts. |

## Naming convention

Core implementation names use the `*_core` form when the file is the substrate beneath multiple authoring surfaces. Current examples include `manifest_core`, `experiment_core`, `cohorts_core`, and `trading_core` when those domains exist in the codebase.

## Read next

- [Single-File Decoder](Single-File-Decoder.md)
- [Built-In SFDs](Built-In-SFDs.md)
- [Experiment Manifest](Experiment-Manifest.md)
- [Universal Experiment Loop](Universal-Experiment-Loop.md)
