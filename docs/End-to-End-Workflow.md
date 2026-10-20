# Carry the experiment through to inference

A complete run must preserve the connection between the question, observations, selection rule and fitted state used for later decisions. The [Python project](examples/python-research/README.md) owns those connections in its functions. The [Compiled project](examples/compiled-research/README.md) adopts the declared pipeline and its recorded reconstruction contract.

The opening [research chapter](Research-Book.md) develops selection. This chapter brings the later scenarios together. Run either project independently; their models and evaluation populations differ, so their raw scores are not a head-to-head model comparison.

## Prerequisites

Use the matching Limen wheel, optional dependencies and recorded input bundle described on each project page. Run the following commands from the bundle root. From a repository checkout, work in `docs/examples/` and follow the project pages' input/output path adjustments.

## Run the Python research

<!-- example: workflow-python-all source=docs/examples/commands.json region=python-all scenario=selection interface=python role=executed -->
```bash
python python-research/research.py --scenario all --data-dir inputs --output-dir results/python-all
```

The unmodified entry point executes selection, search, recovery, walk-forward, holdout assessment, calibration, event execution, bar formation and observed-cost stress comparisons. It checks recorded input digests before research and writes a `summary.json` alongside native outputs.

Selection fits four Ridge candidates and retains their fitted scaler/model pipelines in memory. It chooses the best validation result per feature window and builds a Cohort from the selected Sensors. No fit consumes the final assessment partition. `holdout` then evaluates next-bar decisions on that excluded partition; exclusion remains caller-owned and supplies no project reservation or earlier-exposure guarantee.

The base selection's CSV and summary do not serialize arbitrary fitted models. To continue inference in the same process, import `run_research` and use the returned Cohort and retained state. Recovery is a separate scenario that persists queue/results and compares resumed and uninterrupted runs.

## Run the Compiled research

<!-- example: workflow-compiled-all source=docs/examples/commands.json region=compiled-all scenario=selection interface=compiled role=executed -->
```bash
python compiled-research/research.py --scenario all --data-dir inputs --output-dir results/compiled-all
```

This entry point executes selection, search, recovery, walk-forward, final holdout, calibration, event execution, bar formation, batch allocation and command-line project operations. It validates declarations, preserves native artifacts and reconstructs selected recorded rounds through `Trainer`.

The selection declaration applies chronological splits before preparation, fits target/scaler state on training and ranks two logistic candidates by validation event-ledger return. Trainer reconstructs the selected recorded round using the original splits and checks replayed metrics. Reconstruction is a refit under the recorded contract, rather than restoration of arbitrary model weights.

Its `command_line` scenario validates, profiles, compiles, commits and executes the same recorded configuration through installed `limen` commands. It also verifies project lineage, completed-run resume and a local Git backup/restore round trip. See the [command reference](Command-Line-Interface.md) for the exact lifecycle.

## Review the evidence by decision

| Decision | Python project | Compiled project | Contract |
|---|---|---|---|
| Choose trials | Caller-owned SFD and seeded strategies; budget-only and mixed feedback runs | Declared strategies, admitted factorized comparison and model families; shared feedback | [Search](Advanced-Search.md), [feedback](Reducers-And-Feedback.md) |
| Continue after interruption | Reconstruct queue/results; compare to uninterrupted research | Reconstruct declaration, queue/results and a selected Sensor; compare complete artifacts | [Recovery](Advanced-Search.md#resume-flow) |
| Evaluate across time | Own fold fitting, validation selection and later Sensor decisions | Declared folds, purge/embargo and qualified fold reconstruction | [Walk-forward](Walk-Forward.md) |
| Calibrate decisions | Store fitted probability model and threshold in caller-owned state | Four configured modes and recorded Sensor reconstruction | [Calibration](Calibration.md) |
| Assess a selected result | Exclude the final partition from fitting and selection in code | Reserve, freeze and assess with native project records; earlier exposure remains unknown for local files | [Final holdout](Final-Holdout.md) |
| Measure execution | Direct event inputs, policy, source clocks, costs and recorded funding | Declared event policy, ledgers, source bindings and funded comparison | [Backtest](Backtest.md) |
| Form alternate bars | Call the shared bar helper over recorded observations | Form bars separately after each chronological split | [Data bars](Data-Bars.md) |
| Reuse fitted behavior | Wrap the evaluated models and compose compatible native Sensors | Reconstruct recorded rounds or qualified folds, then bind compatible members | [Sensors](Trainer.md), [Cohorts](Cohort.md) |

## Review outputs

| Artifact | Meaning |
|---|---|
| `summary.json` | Project report of the scenarios requested by the entry point. |
| Python selection CSV | Candidate validation scores; fitted models remain in memory. |
| Compiled `metadata.json` and copied definition | Declared configuration and provenance used for reconstruction. |
| `results.csv` | Completed parameter rounds for a persisted experiment. |
| `round_data.jsonl.gz` and signal payloads | Round identities and the payloads selected by the [recording policy](Universal-Experiment-Loop.md#recording-policy). |
| `checkpoint.json` | Persisted queue/domain/reducer state when checkpointing ran. |
| `audit.jsonl` | Applied feedback, suggestions, source errors and queue state when feedback ran. |
| Scenario Parquet ledgers/reports | Recorded execution or assessment details emitted by that scenario. |

Keep the definition, recorded input identities, package/environment identity and applicable artifacts together. A summary alone cannot reconstruct caller-owned state or establish that unrecorded payloads existed.

## Choose parallel execution deliberately

For parallel CPU classification rounds, set every model `n_jobs` value to `1`
and supply `--workers 4` to `limen run` for a compatible declaration. YAML `uel.workers` supplies the
default; the CLI overrides it. Runs created with full
[recording evidence](Universal-Experiment-Loop.md#recording-policy) can resume with
`limen run --resume RESULTS_DIR`, reusing the saved worker policy. See [parallel rounds](Experiment-Manifest.md#parallel-rounds)
for admission, feedback ordering and stop/recovery limits. The canonical projects use one worker to keep their proof population explicit; a larger search is needed to occupy several workers.

## Interpret the result

A successful run establishes the behavior asserted for its recorded inputs and configuration. Selection, calibration, search adaptation and repeated research consume evidence; their best score is not an untouched estimate. Trading assumptions, source clocks, unavailable decisions and exposure history remain part of the result.

Continue to [Log](Log.md), [Benchmark](Benchmark.md) and [Backtest](Backtest.md) for analysis. The Reference and Developer appendices supply API detail and maintenance procedures without changing who owns the research methodology.
