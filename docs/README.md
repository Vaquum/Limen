# Limen docs

This page routes Limen docs by task.

## Limen in one page

Limen is a research engine for Bitcoin traders. Python Research gives you control of data, preparation, models, backtesting and selection in ordinary Python. Compiled Research adopts the Manifest contract through the CLI or Python, with the configured pipeline checks and recorded evidence. Both use Universal Experiment Loop (UEL) with Mutable Search Queue (MSQ). Choose the interface by who should own the research methodology.

Limen does not perform downstream trade decisioning or execution. In the wider Vaquum architecture, Origo sits upstream as the data layer, while Nexus, Praxis, and Veritas sit downstream for decisioning, execution, and oversight.

## Start here

### New to Limen

1. Read the [product home page](../README.md) and choose Python Research or Compiled Research.
2. For Python Research, start with the [worked Python workflow](Universal-Experiment-Loop.md#direct-python-run) and [custom SFDs](Single-File-Decoder.md#custom-sfds).
3. For Compiled Research, run the [End-to-End Workflow](End-to-End-Workflow.md) and inspect the [Manifest contract](Experiment-Manifest.md).
4. Learn how data enters Limen in [Historical Data](Historical-Data.md).
5. Review outcomes in [Log](Log.md), [Benchmark](Benchmark.md) and [Backtest](Backtest.md).
6. Continue to [Trainer and Sensors](Trainer.md) and [Cohort](Cohort.md) for the selected interface's inference handoff.
7. Use the [Glossary](Glossary.md) and component references for detail.

### Python Research

1. Start with the [Python/MSQ run](Universal-Experiment-Loop.md#direct-python-run) using a built-in search strategy.
2. Define a [custom SFD](Single-File-Decoder.md#custom-sfds) with ordinary `params`, `prep` and `model` functions; your preparation, models and backtesting require no Manifest or new research language.
3. Choose grid or random search in [Advanced Search](Advanced-Search.md), and enable persistence when you need checkpoints or saved round artifacts.
4. Add [Reducers And Feedback](Reducers-And-Feedback.md) only when the experiment calls for adaptive search.
5. Inspect the live run through [Log](Log.md), or read the saved result artifacts.

### Compiled Research

1. Start with [Experiment Manifest](Experiment-Manifest.md)
2. Run manifests through [End-to-End Workflow](End-to-End-Workflow.md) and [Command-Line Interface](Command-Line-Interface.md)
3. Review the shipped decoder patterns in [Built-In SFDs](Built-In-SFDs.md)
4. Read [Single-File Decoder](Single-File-Decoder.md) for the compiled SFD boundary; invoking a Manifest from Python remains Compiled Research
5. Use [Indicators](Indicators.md), [Features](Features.md), [Targets](Targets.md), [Transforms](Transforms.md), [Scalers](Scalers.md), [Calibration](Calibration.md), and [Reference Architecture](Reference-Architecture.md) as the reference layer
6. Strengthen the research surface with [Perturbation Strategies](Perturbation-Strategies.md) for robustness, [Fractional Differentiation](Fractional-Differentiation.md) for stationary features, and [Triple-Barrier Method](Triple-Barrier-Method.md) for path-dependent labels
7. Choose search in [Advanced Search](Advanced-Search.md); adaptive controls are optional in [Reducers And Feedback](Reducers-And-Feedback.md)
8. Inspect results in [Log](Log.md), [Benchmark](Benchmark.md), and [Backtest](Backtest.md)

### Review finished runs

1. Start with [Log](Log.md)
2. Compare model behavior in [Benchmark](Benchmark.md)
3. Evaluate trading behavior in [Backtest](Backtest.md)
4. Review helper metrics in [Standard Metrics Library](Standard-Metrics-Library.md) and [Reference Architecture](Reference-Architecture.md)
5. Reuse selected Python fitted state through [Sensors](Trainer.md#python-construction), or reconstruct recorded Compiled rounds through [Trainer](Trainer.md); continue to [Cohort](Cohort.md) for composition

### Extend Limen

1. Start with [Reference Architecture](Reference-Architecture.md) and [Built-In SFDs](Built-In-SFDs.md)
2. Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) for direct engine integration
3. Continue to [Advanced Search](Advanced-Search.md) for custom `SearchStrategy` implementations, `ParamDomain`, MSQ, and checkpoints
4. Continue to [Reducers And Feedback](Reducers-And-Feedback.md) for adaptive interventions
5. Use [Utilities](Utilities.md) for the helper layer rather than the main workflow

### Contribute or maintain

1. Start with [Developer Guidelines](Developer/README.md)
2. Read the docs contract in [Documentation System](Developer/Documentation-System.md)
3. Use [Pruning Strategies](Developer/Pruning-Strategies.md) for reducer work
4. Use [Contributing Foundational SFDs](Developer/Contributing-Foundational-SFDs.md) for SFD work
5. Use [Technical Debt](TechnicalDebt.md) when assessing accepted known risk
6. Use [Making Release](Developer/Making-Release.md) and [Semantic Versioning](Semantic-Versioning.md) for maintenance work
7. Use [Roadmap](Roadmap.md) for planned direction and [Security Assurance Case](Developer/Security-Assurance-Case.md) for the security posture

## How Limen flows

1. Data enters through [Historical Data](Historical-Data.md) or compatible external OHLC data.
2. Data can be reshaped with [Data Bars](Data-Bars.md) when threshold bars are the right research surface.
3. Indicators, features, transforms, and scalers define the research surface. Targets define supervised labels; calibration adjusts probabilities and thresholds after model output.
4. Define Python Research with caller-owned [SFD functions](Single-File-Decoder.md#custom-sfds), or adopt the [Manifest contract](Experiment-Manifest.md) for Compiled Research through the CLI or Python.
5. Both routes use [Universal Experiment Loop](Universal-Experiment-Loop.md) with MSQ. [Advanced Search](Advanced-Search.md) controls candidate generation; [Reducers And Feedback](Reducers-And-Feedback.md) provides optional adaptive control.
6. [Log](Log.md), [Benchmark](Benchmark.md), and [Backtest](Backtest.md) explain what happened and why.
7. Python Research wraps selected evaluated fitted state in a [Sensor](Trainer.md#python-construction). Compiled Research uses [Trainer](Trainer.md) to reconstruct selected recorded rounds or qualified folds.
8. [Cohort](Cohort.md) combines selected Sensors through caller-owned Python composition or compatible recorded-member binding.
9. Those outputs then move downstream into Nexus and the rest of the Vaquum stack.

## Docs map

- `Overview`: [Product Home](../README.md), [this docs hub](README.md), [Roadmap](Roadmap.md)
- `Guides`: [End-to-End Workflow](End-to-End-Workflow.md), [Command-Line Interface](Command-Line-Interface.md), [Experiment Manifest](Experiment-Manifest.md), [Historical Data](Historical-Data.md), [Data Bars](Data-Bars.md), [Single-File Decoder](Single-File-Decoder.md), [Built-In SFDs](Built-In-SFDs.md), [Universal Experiment Loop](Universal-Experiment-Loop.md), [Advanced Search](Advanced-Search.md), [Reducers And Feedback](Reducers-And-Feedback.md), [Perturbation Strategies](Perturbation-Strategies.md), [Fractional Differentiation](Fractional-Differentiation.md), [Triple-Barrier Method](Triple-Barrier-Method.md), [Log](Log.md), [Benchmark](Benchmark.md), [Backtest](Backtest.md), [Trainer](Trainer.md), [Cohort](Cohort.md), [Conserved Flux Renormalization](Conserved-Flux-Renormalization.md)
- `Reference`: [Glossary](Glossary.md), [Indicators](Indicators.md), [Features](Features.md), [Targets](Targets.md), [Transforms](Transforms.md), [Scalers](Scalers.md), [Calibration](Calibration.md), [Standard Metrics Library](Standard-Metrics-Library.md), [Reference Architecture](Reference-Architecture.md), [Utilities](Utilities.md)
- `Developer`: [Developer Guidelines](Developer/README.md), [Documentation System](Developer/Documentation-System.md), [Pruning Strategies](Developer/Pruning-Strategies.md), [Writing Docstrings](Developer/Writing-Docstrings.md), [Contributing Foundational SFDs](Developer/Contributing-Foundational-SFDs.md), [Making Release](Developer/Making-Release.md), [Packaging](Developer/Packaging.md), [Release Policy](Developer/Release-Policy.md), [Security Assurance Case](Developer/Security-Assurance-Case.md), [Semantic Versioning](Semantic-Versioning.md), and [Technical Debt](TechnicalDebt.md)
- `Packages`: package `README`s under `/limen` for `data`, `experiment`, `inference`, `sfd`, `indicators`, `features`, `transforms`, `scalers`, `metrics`, `log`, `cohort`, `backtest`, `utils`, `calibration`, `cli`, `targets`, and `yaml`, plus their nested package references

The documentation is being developed as a book around two complete executable research projects. Reference and Developer remain available for drill-down and direct lookup. The sections above route the current published pages; the [documentation system contract](Developer/Documentation-System.md) defines the book and the evidence required for its examples.

## Product boundary

### Limen owns

- experiment-oriented data access
- indicator, feature, transform, and scaler composition
- target construction, calibration, and CLI-driven YAML experiment execution
- manifest-driven and custom SFD-based research units
- parameter sweep and experiment logging
- benchmark-style analytics and backtesting
- round reconstruction and cohort construction

### Limen does not own

- upstream source-of-truth market data infrastructure
- downstream trade decisioning
- execution and exchange operations
- system-wide oversight and audit

## Read next

- For a first real run, continue to [End-to-End Workflow](End-to-End-Workflow.md), then [Command-Line Interface](Command-Line-Interface.md)
- For architecture and system boundaries, continue to [Trainer](Trainer.md) and [Cohort](Cohort.md)
- For Python experiments, continue to the [Python/MSQ run](Universal-Experiment-Loop.md#direct-python-run), [Single-File Decoder](Single-File-Decoder.md), and [Advanced Search](Advanced-Search.md)
- For contributor work, continue to [Developer Guidelines](Developer/README.md)
