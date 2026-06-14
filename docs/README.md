# Limen Docs

This page is the routing hub for the Limen docs. Use it to choose the right path based on what you are trying to do.

## Limen In One Page

Limen is a Bitcoin alpha research engine for turning market data into experiments, logged analytics, backtests, and decoder cohorts. It keeps the research loop inside one Python system: data preparation, indicators, features, targets, scaling, parameter search, and post-run evaluation.

Limen does not perform downstream trade decisioning or execution. In the wider Vaquum architecture, Origo sits upstream as the data layer, while Nexus, Praxis, and Veritas sit downstream for decisioning, execution, and oversight.

## Start Here

### If You Are New To Limen

1. Read the [product home page](../README.md)
2. Learn how data enters Limen in [Historical Data](Historical-Data.md)
3. Learn how experiments are packaged in [Single-File Decoder](Single-File-Decoder.md)
4. Review the shipped patterns in [Built-In SFDs](Built-In-SFDs.md)
5. Learn the standard declarative path in [Experiment Manifest](Experiment-Manifest.md)
6. Run experiments in [Universal Experiment Loop](Universal-Experiment-Loop.md)
7. Review outcomes in [Log](Log.md)

### If You Want To Author Experiments

1. Start with [Single-File Decoder](Single-File-Decoder.md)
2. Review the shipped patterns in [Built-In SFDs](Built-In-SFDs.md)
3. Continue to [Experiment Manifest](Experiment-Manifest.md)
4. Use [Indicators](Indicators.md), [Features](Features.md), [Targets](Targets.md), [Transforms](Transforms.md), [Scalers](Scalers.md), [Calibration](Calibration.md), and [Reference Architecture](Reference-Architecture.md) as your reference layer
5. Run the search in [Universal Experiment Loop](Universal-Experiment-Loop.md)
6. If you need adaptive search, continue to [Advanced Search](Advanced-Search.md) and [Reducers And Feedback](Reducers-And-Feedback.md)
7. Inspect results in [Log](Log.md), [Benchmark](Benchmark.md), and [Backtest](Backtest.md)

### If You Want To Review Finished Runs

1. Start with [Log](Log.md)
2. Compare model behavior in [Benchmark](Benchmark.md)
3. Evaluate trading behavior in [Backtest](Backtest.md)
4. Review helper metrics in [Standard Metrics Library](Standard-Metrics-Library.md) and [Reference Architecture](Reference-Architecture.md)
5. Continue to [Trainer](Trainer.md) and [Cohort](Cohort.md) if you are promoting outputs downstream

### If You Want To Extend Limen

1. Start with [Reference Architecture](Reference-Architecture.md) and [Built-In SFDs](Built-In-SFDs.md)
2. Continue to [Advanced Search](Advanced-Search.md) for `SearchStrategy`, `ParamDomain`, `MSQ`, and checkpoints
3. Continue to [Reducers And Feedback](Reducers-And-Feedback.md) for adaptive interventions
4. Use [Command Line Interface](Command-Line-Interface.md) when the extension is YAML-first or shell-driven
5. Use [Utilities](Utilities.md) when you need the helper layer rather than the main workflow

### If You Want To Contribute Or Maintain

1. Start with [Developer Guidelines](Developer/README.md)
2. Read the docs contract in [Documentation System](Developer/Documentation-System.md)
3. Use [Pruning Strategies](Developer/Pruning-Strategies.md) for reducer work
4. Use [Contributing Foundational SFDs](Developer/Contributing-Foundational-SFDs.md) for SFD work
5. Use [Technical Debt](TechnicalDebt.md) when assessing recorded known-risk items
6. Use [Making Release](Developer/Making-Release.md) and [Semantic Versioning](Semantic-Versioning.md) for maintenance work

## How Limen Flows

1. Data enters through [Historical Data](Historical-Data.md) or compatible external OHLC data.
2. Data can be reshaped with [Data Bars](Data-Bars.md) when threshold bars are the right research surface.
3. Indicators, features, transforms, and scalers define the research surface. Targets define supervised labels; calibration adjusts probabilities and thresholds after model output.
4. An experiment is packaged in an [SFD](Single-File-Decoder.md), often starting from [Built-In SFDs](Built-In-SFDs.md) and usually expressed through an [Experiment Manifest](Experiment-Manifest.md).
5. [Universal Experiment Loop](Universal-Experiment-Loop.md) executes the search, with [Advanced Search](Advanced-Search.md) and [Reducers And Feedback](Reducers-And-Feedback.md) extending the artifact-rich path.
6. [Log](Log.md), [Benchmark](Benchmark.md), and [Backtest](Backtest.md) explain what happened and why.
7. [Trainer](Trainer.md) turns selected rounds into reusable sensors.
8. [Cohort](Cohort.md) defines selector-driven ensemble inference for multi-member decoder aggregation.
9. Those outputs then move downstream into Nexus and the rest of the Vaquum stack.

## Docs Map

- `Overview`: [Product Home](../README.md), [this docs hub](README.md)
- `Guides`: [Historical Data](Historical-Data.md), [Data Bars](Data-Bars.md), [Single-File Decoder](Single-File-Decoder.md), [Built-In SFDs](Built-In-SFDs.md), [Experiment Manifest](Experiment-Manifest.md), [Universal Experiment Loop](Universal-Experiment-Loop.md), [Advanced Search](Advanced-Search.md), [Reducers And Feedback](Reducers-And-Feedback.md), [Log](Log.md), [Benchmark](Benchmark.md), [Backtest](Backtest.md), [Trainer](Trainer.md), [Cohort](Cohort.md), [Conserved Flux Renormalization](Conserved-Flux-Renormalization.md)
- `Reference`: [Indicators](Indicators.md), [Features](Features.md), [Targets](Targets.md), [Transforms](Transforms.md), [Scalers](Scalers.md), [Calibration](Calibration.md), [Standard Metrics Library](Standard-Metrics-Library.md), [Reference Architecture](Reference-Architecture.md), [Utilities](Utilities.md), [Command Line Interface](Command-Line-Interface.md)
- `Developer`: [Developer Guidelines](Developer/README.md), [Documentation System](Developer/Documentation-System.md), [Pruning Strategies](Developer/Pruning-Strategies.md), [Writing Docstrings](Developer/Writing-Docstrings.md), [Contributing Foundational SFDs](Developer/Contributing-Foundational-SFDs.md), [Making Release](Developer/Making-Release.md), [Semantic Versioning](Semantic-Versioning.md), [Technical Debt](TechnicalDebt.md)
- `Packages`: package `README`s under `/limen` for `data`, `experiment`, `sfd`, `indicators`, `features`, `transforms`, `scalers`, `metrics`, `log`, `cohort`, `backtest`, `utils`, `calibration`, `cli`, `targets`, and `yaml`

## Product Boundary

### Limen Owns

- experiment-oriented data access
- indicator, feature, transform, and scaler composition
- target construction, calibration, and CLI-driven YAML experiment execution
- manifest-driven and custom SFD-based research units
- parameter sweep and experiment logging
- benchmark-style analytics and backtesting
- retraining and cohort construction

### Limen Does Not Own

- upstream source-of-truth market data infrastructure
- downstream trade decisioning
- execution and exchange operations
- system-wide oversight and audit

## Read Next

- For a first real run, continue to [Historical Data](Historical-Data.md), then [Single-File Decoder](Single-File-Decoder.md), then [Universal Experiment Loop](Universal-Experiment-Loop.md)
- For architecture and system boundaries, continue to [Trainer](Trainer.md) and [Cohort](Cohort.md)
- For the extension layer, continue to [Built-In SFDs](Built-In-SFDs.md), [Reference Architecture](Reference-Architecture.md), and [Advanced Search](Advanced-Search.md)
- For contributor work, continue to [Developer Guidelines](Developer/README.md)
