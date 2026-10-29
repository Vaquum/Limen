# Carry the experiment through to inference

A complete run must preserve the connection between the question, observations, selection rule and fitted state used for later decisions. The [Python project](examples/python-research/README.md) owns those connections in its functions. The [Compiled project](examples/compiled-research/README.md) adopts the declared pipeline and its recorded reconstruction contract.

The opening [research chapter](Research-Book.md) develops selection. This chapter brings the later scenarios together. Run either project independently; their models and evaluation populations differ, so their raw scores are not a head-to-head model comparison.

## Prerequisites

Use the matching Limen wheel, optional dependencies and recorded input bundle described on each project page. Run the following commands from the bundle root. From a repository checkout, work in `docs/examples/` and follow the project pages' input/output path adjustments.

## Run the Python research

<!-- example: workflow-python-selection source=docs/examples/commands.json region=python-selection scenario=selection interface=python role=executed -->
```bash
python python-research/research.py --scenario selection --data-dir inputs --output-dir results/python
```

<!-- example: workflow-python-all source=docs/examples/commands.json region=python-all scenario=selection interface=python role=executed -->
```bash
python python-research/research.py --scenario all --data-dir inputs --output-dir results/python-all
```

The first command runs selection; the second executes search, recovery, walk-forward, holdout assessment, calibration, event execution, bar formation and observed-cost stress comparisons. It checks recorded input digests before research and writes a `summary.json` alongside native outputs.

Selection fits four Ridge candidates and retains their fitted scaler/model pipelines in memory. It chooses the best validation result per feature window and builds a Cohort from the selected Sensors. No fit consumes the final assessment partition. `holdout` then evaluates next-bar decisions on that excluded partition; exclusion remains caller-owned and supplies no project reservation or earlier-exposure guarantee.

The base selection's CSV and summary do not serialize arbitrary fitted models. To continue inference in the same process, import `run_research` and use the returned Cohort and retained state. Recovery is a separate scenario that persists queue/results and compares resumed and uninterrupted runs.

## Run the Compiled research

<!-- example: workflow-compiled-selection source=docs/examples/commands.json region=compiled-selection scenario=selection interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/selection.yaml
```

<!-- example: workflow-compiled-all source=docs/examples/commands.json region=compiled-all scenario=selection interface=compiled role=executed -->
```bash
sh compiled-research/run.sh
```

The first command executes the small selection YAML. The shell script runs the complete collection of unchanged YAML declarations, CLI assessment and project operations. `limen assess` reconstructs validation-selected Sensors and their Cohort; the guarded declaration reserves, freezes and assesses its final window through the CLI.

The selection declaration applies chronological splits before preparation, fits target/scaler state on training and ranks two logistic candidates by validation event-ledger return. Trainer reconstructs the selected recorded round using the original splits and checks replayed metrics. Reconstruction is a refit under the recorded contract, rather than restoration of arbitrary model weights.

Its `project.sh` validates, profiles, compiles, commits and executes the recorded production declaration in `project.yaml` through installed `limen` commands. It also verifies project lineage, completed-run resume and a local Git backup/restore round trip. See the [command reference](Command-Line-Interface.md) for the exact lifecycle.

## Project lineage preserves successive definitions

The Compiled lifecycle stores canonical YAML by content hash, records parent lineage, and creates explicit working forks. A stored definition and its Git commit are separate recorded outcomes. `limen ls`, `limen lineage`, and `limen reindex` inspect or rebuild the manifest index from those definitions.

The complete `project.sh` case commits a parent definition, derives and commits a child, executes a content-addressed manifest, backs up the Git project, and restores it from the local backup remote. Its assertions check definition identity, lineage, result completion, and restored source. The [CLI lifecycle](Command-Line-Interface.md#first-yaml-run) gives the exact executable command sequence.

Git backup follows ignore rules. Runtime artifacts and private protected-history evidence therefore need their own declared preservation plan. A restored repository supplies the committed definitions; continuing a guarded project also requires its matching marker, history, and evidence.

## Follow the retained state

The Python `holdout` variant contains a complete caller-owned handoff in one process. It reads the pinned quarter-hour file, calls `run_research` on the first 1,440 observations, retains the evaluated fits, selects one per window and builds their Cohort. That returned Cohort predicts the following 480 observations with preceding feature context. Its first 479 decisions receive a following realized bar; the final decision has no next bar in the declared population. The assessment Parquet retains those decisions, positions and net returns. This variant performs its own selection; it does not restore weights from an earlier command's CSV.

The native final branch begins in its own project, with the reservation chosen before research. Its `holdout.yaml` records the source, research partitions, selection and excluded final dates. The run writes the selected-round evidence, and the reservation-bound assessment consumes that same result directory. `final-report.json` retains the frozen candidate identity, final observations and local exposure record. The separate event branch freezes its declared economics and writes a schema-2 ledger report. Neither branch is a continuation of the introductory example's different calendar population.

The small selection, comprehensive preparation and later variants therefore share the project family and explicit question, while each named variant carries its own dates and policy. Carry an object or artifact forward only within the variant that created it; changing the population or selection creates a new research case.

## Review the evidence by decision

| Decision | Python project | Compiled project | Contract |
|---|---|---|---|
| Choose trials | Caller-owned SFD and seeded strategies; budget-only and mixed feedback runs | Declared strategies, admitted factorized comparison, model families and YAML reducers | [Search](Advanced-Search.md), [feedback](Reducers-And-Feedback.md) |
| Continue after interruption | Reconstruct queue/results; compare to uninterrupted research | Resume the declared queue and results through the CLI; reject changed pinned source bytes and compare complete artifacts | [Recovery](Search-Reference.md#resume-flow) |
| Evaluate across time | Own fold fitting, validation selection and later Sensor decisions | Declared folds, purge/embargo and qualified fold reconstruction | [Walk-forward](Walk-Forward.md) |
| Calibrate decisions | Store fitted probability model and threshold in caller-owned state | Four YAML-configured modes; calibrated selection through CLI assessment | [Calibration](Calibration.md) |
| Assess a selected result | Exclude the final partition from fitting and selection in code | Reserve, freeze and assess with native project records; earlier exposure remains unknown for local files | [Final holdout](Final-Holdout.md) |
| Measure execution | Direct event inputs, policy, source clocks, costs and recorded funding | Declared event policy, ledgers, source bindings and funded comparison | [Backtest](Backtest.md) |
| Form alternate bars | Call the shared bar helper over recorded observations | Form bars separately after each chronological split | [Data bars](Data-Bars.md) |
| Reuse fitted behavior | Wrap the evaluated models and compose compatible native Sensors | Reconstruct recorded rounds or qualified folds, then bind compatible members | [Sensors](Trainer.md), [Cohorts](Cohort.md) |

## The complete chain preserves the research decision

The reading sequence develops [features](Feature-Design.md), [targets](Target-Design.md), [models](Model-Selection.md), and [rules](Rule-Research.md) before execution. [Controlled perturbations](Robustness-Research.md) identify the alternatives. Search and feedback determine which alternatives execute, while retained artifacts identify what completed.

[Trading behavior](Trading-Behavior.md) supplies execution meaning. [Linked models](Linked-Models.md) extends the learning mechanism under explicit chronology. [Research evidence](Research-Evidence.md) connects measured outcomes, parameter effects, and selection reports. Selected Sensors and Cohorts then carry the admitted fitted behavior into inference.

Each change to population, target, representation, model, or economics remains identifiable as a research choice. The [workflow table](Experiment-Manifest.md#supported-research-workflows) states the recovery, worker, inference, and final-assessment requirements that apply to that choice.

## Review outputs

| Artifact | Meaning |
|---|---|
| Python `summary.json` | Caller-owned project report of the requested scenarios. |
| Compiled `assessment.json` and `predictions.parquet` | Validation selection, recorded membership, decisions and native unavailable/availability values. |
| Compiled `final-report.json` | Guarded final assessment, retained rows and local exposure evidence. |
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
default; the CLI overrides it. Runs created with the applicable
[recording evidence](Universal-Experiment-Loop.md#recording-policy) can resume with
`limen run --resume RESULTS_DIR`, reusing the saved worker policy. See [parallel rounds](Experiment-Manifest.md#parallel-rounds)
for admission, feedback ordering and stop/recovery limits. The base project uses one worker. The Compiled `parallel.yaml` variant evaluates the same four-candidate domain with two isolated workers and one native thread per worker; its ordered committed results must agree with the serial candidates. Its completed checkpoint also resumes through the CLI without repeating rows. Python function campaigns do not enter this native worker path.

<!-- example: workflow-compiled-parallel source=docs/examples/compiled-research/run.sh region=parallel scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar --workers 2 compiled-research/parallel.yaml
limen run --no-progress-bar --resume results/dev/compiled-parallel --workers 2
```

<!-- result: workflow-compiled-workers interface=compiled scenario=search fields=parallel.workers,parallel.rounds,parallel.ordered_commit_indices -->

## Interpret the result

A successful run establishes the behavior asserted for its recorded inputs and configuration. Selection, calibration, search adaptation and repeated research consume evidence; their best score is not an untouched estimate. Trading assumptions, source clocks, unavailable decisions and exposure history remain part of the result.

Continue to [Log](Log.md), [Benchmark](Benchmark.md) and [Backtest](Backtest.md) for analysis. The Reference and Developer appendices supply API detail and maintenance procedures without changing who owns the research methodology.

Binary rule and ML entries can use [independent manifest exit rules](Experiment-Manifest.md#independent-exit-rules). Keep the exit definition with the experiment; retain alignment evidence when replay is required.
