# Glossary

This glossary defines the terms used across public docs, developer docs, package READMEs, and issue work.

## Core terms

| Term | Meaning |
| --- | --- |
| Limen | Vaquum's Bitcoin research engine for manifest-driven experiments, logged analytics, backtests, and decoder cohorts. |
| YAML manifest | The default operator-facing experiment definition consumed by the `limen` CLI. |
| `Manifest` | The abstract experiment-pipeline interface. Runnable pipelines use `MLManifest` or `RuleBasedManifest`. |
| `MLManifest` | Manifest implementation for trainable reference architectures. |
| `RuleBasedManifest` | Manifest implementation for predicate-driven, non-learned strategies. |
| Single-File Decoder | A Python experiment unit that exposes a parameter surface and either a manifest or custom prep/model functions. Previously called Single File Model. |
| SFD | Short form of Single-File Decoder. Previously called SFM. |
| Foundational SFD | A packaged SFD under `limen.sfd.foundational_sfd`; most have a matching YAML template. |
| Reference Architecture | A class-based model implementation and function wrapper used by foundational manifests. |
| Universal Experiment Loop | The experiment engine that executes parameter rounds. |
| UEL | Short form of Universal Experiment Loop. |
| round / permutation | One concrete parameter combination executed by UEL. |
| experiment artifact | A persisted run output such as `metadata.json`, `results.csv`, `round_data.jsonl`, `checkpoint.json`, or `audit.jsonl`. |
| `Trainer` | Reconstructs and replays selected artifact-backed rounds, validates metrics, and returns Sensors. |
| `Sensor` | A validated model plus manifest, fitted preprocessing state, round parameters, and inference methods. |
| `Cohort` | A bound set of Sensors with one aggregation mode and cohort identity. |
| selector | A callable that chooses string permutation IDs for a Cohort. |
| `HistoricalData` | Stateful data-access class for Limen's supported market-data sources and files. |
| reducer | A `PruningStrategy` that analyzes completed rounds and returns declarative search interventions. |
| `MSQ` | Mutable Search Queue used by advanced search and feedback dispatch. |
| `Log` | Analysis wrapper over an experiment log and retained round artifacts. |
| benchmark | Confusion-oriented outcome analysis; not an independent public benchmark suite or leaderboard. |
| backtest | Long/flat per-bar research ledger with basis-point outputs; not an execution simulator. |

## Implementation names

`limen.experiment.manifest_core` contains the manifest implementations, and `limen.experiment.experiment_core` contains Universal Experiment Loop. These are module names, not a general `*_core` naming promise for other domains.

## Read next

- [Single-File Decoder](Single-File-Decoder.md)
- [Experiment Manifest](Experiment-Manifest.md)
- [Universal Experiment Loop](Universal-Experiment-Loop.md)
- [Trainer](Trainer.md)
- [Cohort](Cohort.md)
