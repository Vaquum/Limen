# Glossary

These terms apply to the [Python project](examples/python-research/README.md), the [Compiled project](examples/compiled-research/README.md) and the reference appendices. Their distinction is methodological ownership: Compiled Research uses YAML declarations and CLI commands.

## Core terms

| Term | Meaning |
| --- | --- |
| Limen | Vaquum's Bitcoin research engine for Python and Compiled experiments, analysis, backtesting and reusable Sensors and Cohorts. |
| Python Research | Caller-owned data, preparation, models, backtesting and selection in ordinary Python; no Manifest or built-in-model whitelist is required. |
| Compiled Research | A YAML experiment executed through the CLI, with its configured validation, preparation and evidence contracts. |
| YAML manifest | A Compiled experiment definition validated, compiled and executed by the `limen` CLI. |
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
| experiment artifact | A persisted run output such as `metadata.json`, `results.csv`, `round_data.jsonl.gz`, `checkpoint.json`, or `audit.jsonl`. |
| `Trainer` | Reconstructs and replays selected rounds from a saved YAML experiment, validates metrics, and returns Sensors. |
| `Sensor` | A native predictor wrapping fitted state and inference behavior. Python Sensors preserve caller-owned preparation, cutoffs and output meanings; Compiled Sensors carry their reconstructed Manifest and recorded identities. |
| `Cohort` | Compatible Sensors composed with an aggregation rule, timestamp checks and unavailable-result handling. Recorded Compiled membership can additionally bind artifact identities. |
| selector | For artifact-backed Compiled composition, a callable choosing string permutation IDs. Python callers can select their fitted models directly before constructing Sensors. |
| `HistoricalData` | Stateful data-access class for Limen's supported market-data sources and files. |
| reducer | A `PruningStrategy` that analyzes completed rounds and returns declarative search interventions. |
| `MSQ` | Mutable Search Queue inside UEL, used by CLI and Python search-strategy runs to select candidates and apply optional feedback. |
| `Log` | Analysis wrapper over an experiment log and retained round artifacts. |
| benchmark | Outcome analysis of retained research results. Confusion metrics and advisory walk-forward selection reports have different evidence requirements; neither creates statistical acceptance or an independent leaderboard. |
| backtest | Evaluation under an explicit trading policy. Limen offers a snapshot ledger and an event execution engine; Python callers can supply their own backtester. State the execution assumptions and measured units. |
| short/flat snapshot | Explicit binary short-return approximation through `short_flat_strategy`; [Backtest](Backtest.md#explicit-shortflat-snapshots) defines fills, costs and exclusions. |

## Observation and decision boundaries

| Term | Meaning |
| --- | --- |
| fitting cutoff | Latest observation permitted to affect fitted model or preparation state. A later inference call does not extend it. |
| selection cutoff | Latest observation permitted to affect the choice of candidate, threshold or membership. It can be later than fitting. |
| qualification | An explicit check of whether a candidate or result meets declared conditions. Passing does not make reused validation observations untouched. |
| final holdout | Observations excluded before research and used after the candidate is frozen. The native reservation records local exposure history; caller-owned exclusion records only what the caller actually enforces. |
| availability time | Earliest time an observation or prediction can be used. A bar's timestamp alone does not prove that its closing value was already known. |
| reconstruction | A validated refit and replay of recorded native behavior. It is different from retaining the original evaluated Python object and from restoring arbitrary model weights. |

## Implementation names

`limen.experiment.manifest_core` contains the manifest implementations, and `limen.experiment.experiment_core` contains Universal Experiment Loop. These are module names, not a general `*_core` naming promise for other domains.

## Read next

- [Single-File Decoder](Single-File-Decoder.md)
- [Experiment Manifest](Experiment-Manifest.md)
- [Universal Experiment Loop](Universal-Experiment-Loop.md)
- [Trainer](Trainer.md)
- [Cohort](Cohort.md)
