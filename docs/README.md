# Limen docs

This page routes Limen docs by task.

## Limen in one page

Limen is a Bitcoin alpha research engine for turning market data into experiments, logged analytics, backtests, and decoder cohorts. Its default operator path is a YAML manifest run through the CLI; Python APIs remain the extension layer for custom decoder and engine work.

Limen does not perform downstream trade decisioning or execution. In the wider Vaquum architecture, Origo sits upstream as the data layer, while Nexus, Praxis, and Veritas sit downstream for decisioning, execution, and oversight.

## Start here

### New to Limen

1. Read the [product home page](../README.md)
2. Run the [End-to-End Workflow](End-to-End-Workflow.md)
3. Learn the manifest contract in [Experiment Manifest](Experiment-Manifest.md)
4. Learn how data enters Limen in [Historical Data](Historical-Data.md)
5. Review outcomes in [Log](Log.md), [Benchmark](Benchmark.md), and [Backtest](Backtest.md)
6. Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) for the engine beneath CLI
7. Use [Single-File Decoder](Single-File-Decoder.md), [Built-In SFDs](Built-In-SFDs.md), and [Glossary](Glossary.md) for Python extension work

### Author experiments

1. Start with [Experiment Manifest](Experiment-Manifest.md)
2. Run manifests through [End-to-End Workflow](End-to-End-Workflow.md) and [Command Line Interface](Command-Line-Interface.md)
3. Review the shipped decoder patterns in [Built-In SFDs](Built-In-SFDs.md)
4. Use [Single-File Decoder](Single-File-Decoder.md) for custom Python experiment modules
5. Use [Indicators](Indicators.md), [Features](Features.md), [Targets](Targets.md), [Transforms](Transforms.md), [Scalers](Scalers.md), [Calibration](Calibration.md), and [Reference Architecture](Reference-Architecture.md) as the reference layer
6. Strengthen the research surface with [Perturbation Strategies](Perturbation-Strategies.md) for robustness, [Fractional Differentiation](Fractional-Differentiation.md) for stationary features, and [Triple-Barrier Method](Triple-Barrier-Method.md) for path-dependent labels
7. Adaptive search continues in [Advanced Search](Advanced-Search.md) and [Reducers And Feedback](Reducers-And-Feedback.md)
8. Inspect results in [Log](Log.md), [Benchmark](Benchmark.md), and [Backtest](Backtest.md)

### Review finished runs

1. Start with [Log](Log.md)
2. Compare model behavior in [Benchmark](Benchmark.md)
3. Evaluate trading behavior in [Backtest](Backtest.md)
4. Review helper metrics in [Standard Metrics Library](Standard-Metrics-Library.md) and [Reference Architecture](Reference-Architecture.md)
5. Continue to [Trainer](Trainer.md) and [Cohort](Cohort.md) for downstream promotion

### Extend Limen

1. Start with [Reference Architecture](Reference-Architecture.md) and [Built-In SFDs](Built-In-SFDs.md)
2. Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) for direct engine integration
3. Continue to [Advanced Search](Advanced-Search.md) for `SearchStrategy`, `ParamDomain`, `MSQ`, and checkpoints
4. Continue to [Reducers And Feedback](Reducers-And-Feedback.md) for adaptive interventions
5. Use [Utilities](Utilities.md) for the helper layer rather than the main workflow

### Contribute or maintain

1. Start with [Developer Guidelines](Developer/README.md)
2. Read the docs contract in [Documentation System](Developer/Documentation-System.md)
3. Use [Pruning Strategies](Developer/Pruning-Strategies.md) for reducer work
4. Use [Contributing Foundational SFDs](Developer/Contributing-Foundational-SFDs.md) for SFD work
5. Use [Technical Debt](TechnicalDebt.md) and [Audit Closeout](Audit-Closeout.md) when assessing recorded known-risk or audit-closeout items
6. Use [Making Release](Developer/Making-Release.md) and [Semantic Versioning](Semantic-Versioning.md) for maintenance work

## How Limen flows

1. Data enters through [Historical Data](Historical-Data.md) or compatible external OHLC data.
2. Data can be reshaped with [Data Bars](Data-Bars.md) when threshold bars are the right research surface.
3. Indicators, features, transforms, and scalers define the research surface. Targets define supervised labels; calibration adjusts probabilities and thresholds after model output.
4. An experiment is expressed as an [Experiment Manifest](Experiment-Manifest.md) and run through the [Command Line Interface](Command-Line-Interface.md) by default.
5. [Universal Experiment Loop](Universal-Experiment-Loop.md) is the engine beneath CLI execution; [Single-File Decoder](Single-File-Decoder.md), [Built-In SFDs](Built-In-SFDs.md), [Advanced Search](Advanced-Search.md), and [Reducers And Feedback](Reducers-And-Feedback.md) are the Python extension layer.
6. [Log](Log.md), [Benchmark](Benchmark.md), and [Backtest](Backtest.md) explain what happened and why.
7. [Trainer](Trainer.md) turns selected rounds into reusable sensors.
8. [Cohort](Cohort.md) defines selector-driven ensemble inference for multi-member decoder aggregation.
9. Those outputs then move downstream into Nexus and the rest of the Vaquum stack.

## Docs map

- `Overview`: [Product Home](../README.md), [this docs hub](README.md)
- `Guides`: [End-to-End Workflow](End-to-End-Workflow.md), [Command Line Interface](Command-Line-Interface.md), [Experiment Manifest](Experiment-Manifest.md), [Historical Data](Historical-Data.md), [Data Bars](Data-Bars.md), [Single-File Decoder](Single-File-Decoder.md), [Built-In SFDs](Built-In-SFDs.md), [Universal Experiment Loop](Universal-Experiment-Loop.md), [Advanced Search](Advanced-Search.md), [Reducers And Feedback](Reducers-And-Feedback.md), [Perturbation Strategies](Perturbation-Strategies.md), [Fractional Differentiation](Fractional-Differentiation.md), [Triple-Barrier Method](Triple-Barrier-Method.md), [Log](Log.md), [Benchmark](Benchmark.md), [Backtest](Backtest.md), [Trainer](Trainer.md), [Cohort](Cohort.md), [Conserved Flux Renormalization](Conserved-Flux-Renormalization.md)
- `Reference`: [Glossary](Glossary.md), [Indicators](Indicators.md), [Features](Features.md), [Targets](Targets.md), [Transforms](Transforms.md), [Scalers](Scalers.md), [Calibration](Calibration.md), [Standard Metrics Library](Standard-Metrics-Library.md), [Reference Architecture](Reference-Architecture.md), [Utilities](Utilities.md)
- `Developer`: [Developer Guidelines](Developer/README.md), [Documentation System](Developer/Documentation-System.md), [Pruning Strategies](Developer/Pruning-Strategies.md), [Writing Docstrings](Developer/Writing-Docstrings.md), [Contributing Foundational SFDs](Developer/Contributing-Foundational-SFDs.md), [Making Release](Developer/Making-Release.md), [Semantic Versioning](Semantic-Versioning.md), [Technical Debt](TechnicalDebt.md), [Audit Closeout](Audit-Closeout.md)
- `Packages`: package `README`s under `/limen` for `data`, `experiment`, `sfd`, `indicators`, `features`, `transforms`, `scalers`, `metrics`, `log`, `cohort`, `backtest`, `utils`, `calibration`, `cli`, `targets`, and `yaml`

## Product boundary

### Limen owns

- experiment-oriented data access
- indicator, feature, transform, and scaler composition
- target construction, calibration, and CLI-driven YAML experiment execution
- manifest-driven and custom SFD-based research units
- parameter sweep and experiment logging
- benchmark-style analytics and backtesting
- retraining and cohort construction

### Limen does not own

- upstream source-of-truth market data infrastructure
- downstream trade decisioning
- execution and exchange operations
- system-wide oversight and audit

## Read next

- For a first real run, continue to [End-to-End Workflow](End-to-End-Workflow.md), then [Command Line Interface](Command-Line-Interface.md)
- For architecture and system boundaries, continue to [Trainer](Trainer.md) and [Cohort](Cohort.md)
- For the extension layer, continue to [Built-In SFDs](Built-In-SFDs.md), [Reference Architecture](Reference-Architecture.md), and [Advanced Search](Advanced-Search.md)
- For contributor work, continue to [Developer Guidelines](Developer/README.md)
