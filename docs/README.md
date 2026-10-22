# Limen docs

Choose one research interface and follow its question from recorded observations to reusable predictions. The book develops both projects; Reference and Developer provide the contracts and maintenance procedures behind them.

## Start here

[From a research question to a Sensor](Research-Book.md) introduces the decision. Choose [Python Research](examples/python-research/README.md) for caller-owned data, preparation, fitting, backtesting and selection, or [Compiled Research](examples/compiled-research/README.md) for a YAML definition executed through the CLI. Either project can be followed independently.

Obtain the matching wheel and recorded inputs from the project page before running an excerpt. The package installation does not install the example bundle. If final observations must remain excluded, choose the protected branch before research; [Historical Data](Historical-Data.md) explains that first decision.

## Read the book

| Decision | Chapter |
|---|---|
| Choose who owns the methodology | [Begin the research](Research-Book.md) |
| Establish the observations and their clocks | [Historical data](Historical-Data.md) |
| Change the bar clock | [Data bars](Data-Bars.md) |
| Define preparation, fitting and the objective | [Define the experiment](Compiled-Research.md) |
| Choose the candidates to evaluate | [Search](Advanced-Search.md) |
| Change a running search deliberately | [Reducers and feedback](Reducers-And-Feedback.md) |
| Retain, inspect and continue a run | [Run and recover](Run-And-Recover.md) |
| Evaluate successive periods | [Walk-forward](Walk-Forward.md) |
| Interpret predictions as trades | [Trading behavior](Trading-Behavior.md) |
| Carry selected fitted behavior forward | [Selected behavior](Selected-Behavior.md) |
| Assess the observations reserved earlier | [Final assessment](Final-Holdout.md) |
| Inspect the complete chain | [End-to-end workflow](End-to-End-Workflow.md) |

## Reference

The [public API index](Public-API.md) maps every declared export and public class member to its exact source contract. Use topic pages for the decision and semantics; this index closes aliases, methods and ownership.

Use [Manifest fields](Experiment-Manifest.md) and the [CLI](Command-Line-Interface.md) for Compiled authoring. Use [ordinary Python SFD functions](Single-File-Decoder.md#custom-sfds) and the [UEL API](Universal-Experiment-Loop.md) for caller-owned execution. Shared preparation components have their own [indicator](Indicators.md), [feature](Features.md), [target](Targets.md), [transform](Transforms.md), [scaler](Scalers.md) and [calibration](Calibration.md) contracts.

[Log](Log.md), [Benchmark](Benchmark.md), [execution](Backtest.md), [Trainer and Sensors](Trainer.md), [Cohort](Cohort.md) and [guarded holdout](Guarded-Holdout-Reference.md) preserve the exact outputs, evidence requirements and failure behavior. Optional native component integration is appendix material; it is not a Python step in the Compiled journey. Package READMEs identify their public entry points and adjacent ownership. The [glossary](Glossary.md) fixes the terminology.

## Developer

Start with [Developer home](Developer/README.md) to change, test, review, package or release Limen. The [documentation contract](Developer/Documentation-System.md) maintains the connection between the projects, displayed examples, tests, merged source and released wheel. [Roadmap](Roadmap.md) describes future direction; current behavior belongs in the book and reference.
