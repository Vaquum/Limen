# Retain the run and continue its work

A completed score is useful only while its parameters, observations and research definition remain identifiable. Recording decides which later questions can be answered; a checkpoint decides which unfinished work can continue.

Keep the saved definition and completed candidate identities from the [Python project](examples/python-research/README.md) or [Compiled project](examples/compiled-research/README.md). Decide which evidence later analysis will need before launching a larger search.

## Prerequisites

Keep the acquired project and its matching wheel. A persisted recovery run needs its original source, result directory and checkpoint together; the small Python selection CSV alone does not contain them.

## Persistence preserves four different kinds of state

| Persistence operation | State preserved or recreated | Appropriate continuation |
|---|---|---|
| Result recording | Scalar results, identities, metadata, and explicitly enabled payloads | Analysis supported by those recorded fields |
| Checkpoint recovery | Search domain, queue, reducer state, and the validated committed execution prefix | Continue unfinished candidate work under the same admitted contract |
| Captured `.uel` snapshot | The object graph actually retained by the completed loop | Inspect or use that captured state in the matching trusted environment |
| Trainer reconstruction | A refit and replay of selected recorded YAML rounds or qualified folds | Construct native Sensors from admitted experiment artifacts |

The operation follows the state needed by the next decision. A scalar table supports ranking. Fitted inference needs evaluated state or an admitted reconstruction. Continuing a search needs its checkpoint and committed history.

## Save the captured experiment object

After `run` returns, `uel.save(filepath=None)` writes the full captured object to
`<experiment_name>.uel`, or to an explicit path. Restore it with
`UniversalExperimentLoop.load(filepath)`. Loading restores the retained models,
scalers, observations, parameters and analysis outputs without running the
experiment again. Local Python functions are retained too. State not retained
by the original run is not created by saving; use `post_processing=True` when
the captured models and post-processing outputs are needed later.

Snapshots contain executable Python serialization and require trusted provenance.
Keep the same Python version, Limen source and dependency versions. The reader
checks compatibility and payload integrity before deserialization. A snapshot
does not copy external files or change a run's checkpoint/identity protocol.
Use the original experiment directory and checkpoint to continue unfinished
work; loading a snapshot does not continue it automatically.

Saving replaces its destination atomically. Unsupported state, such as an open
external resource, raises rather than silently omitting the object. A failed
save preserves the previous destination.

## The Python snapshot case retains fitted objects explicitly

The recovery scenario also runs a separate captured-state case. It evaluates the four introductory Ridge candidates, places their caller-owned fitted pipelines in the loop's retained extras, and saves that object. Loading then uses those same captured fits for validation predictions:

<!-- example: retention-python-snapshot source=docs/examples/python-research/snapshot_cases.py region=captured_state scenario=recovery interface=python role=executed -->
```python
def captured_state(bars, output):
    output.mkdir(parents=True)
    loop, retained = project.run_trials(bars[:1440], str(output / 'captured-rounds'))
    loop.extras.append({'fitted_models': retained})
    path = loop.save(output / 'captured.uel')
    restored = UniversalExperimentLoop.load(path)
    assert isinstance(restored, UniversalExperimentLoop)
    if not project.completed_log(restored).equals(project.completed_log(loop)):
        raise ValueError('Captured result log changed during restoration')
    validation = project.prepare_split(bars[960:1440], {'window': 4})
    forecasts = []
    for (window, alpha), fitted in restored.extras[-1]['fitted_models'].items():
        validation = project.prepare_split(bars[960:1440], {'window': window})
        x = validation.select('momentum')
        values = fitted.predict(x)
        if not np.array_equal(values, retained[(window, alpha)].predict(x)):
            raise ValueError('Captured fitted predictions changed during restoration')
        forecasts.append({'window': window, 'alpha': alpha, 'predictions': values.tolist()})
    return {'rounds': project.completed_log(restored).height, 'models': len(forecasts),
            'forecast_rows': validation.height, 'forecasts': forecasts,
            'snapshot_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'restored_fitted_state': True}
```

<!-- result: retention-python-captured interface=python scenario=recovery fields=captured.rounds,captured.models,captured.forecast_rows,captured.restored_fitted_state -->

The project compares restored predictions exactly with the pre-save fits. Independent checks reconstruct the Ridge forecasts from recorded prices and training statistics. The snapshot reports restored fitted state; the checkpoint recovery result below reports restoration of search and results. Both results retain their distinct purpose.

## Python Research

The small selection run writes a CSV and a summary. Its evaluated models remain in memory, returned by `run_research`; the CSV does not restore them. Read the completed frame when choosing candidates:

<!-- example: retention-python-log source=docs/examples/python-research/research.py region=completed_log scenario=selection interface=python role=executed -->
```python
def completed_log(loop: UniversalExperimentLoop) -> pl.DataFrame:
    log = loop.experiment_log
    if log is None:
        raise ValueError("The experiment has no completed result log")
    return log
```

For a search that must survive interruption, the recovery variant records each completed round and its queue state. It requests interruption after two rounds, then compares the resumed four-round result with an uninterrupted run. This restores search and results; it does not serialize arbitrary fitted Python objects.

<!-- result: retention-python-result interface=python scenario=recovery fields=interrupted_rounds,rounds,round_indices,fitted_state_restored,uninterrupted_equivalence -->

## Compiled Research

Run the recovery declaration, then resume its result directory:

<!-- example: retention-compiled-recovery source=docs/examples/compiled-research/run.sh region=recovery scenario=recovery interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/recovery.yaml
limen run --no-progress-bar --resume results/dev/compiled-recovery
```

The script demonstrates a completed-run resume. The project's interruption test separately stops a live CLI process after checkpointed work and requires the same completed artifacts after resumption. In an interactive campaign, interrupt the running command with Ctrl-C, retain its result directory and use `--resume` on that directory. Do not edit the recorded source or replace its definition to make a resume succeed.

<!-- result: retention-compiled-result interface=compiled scenario=recovery fields=rounds,unique_ids -->

## Decide what to retain before execution

`results.csv` identifies completed rounds; `metadata.json` preserves their declaration. `round_data.jsonl.gz` carries round parameters and opted-in evidence. Predictions, alignment, trade ledgers and fold details are separate recording choices. A missing payload is an unavailable analysis or reconstruction boundary, not something a later reader can infer from a score.

The `recording_lean.yaml`, `recording_full.yaml` and `recording_outputs.yaml` variants run identical candidates with different payload choices. The proof compares their recorded scores exactly, checks which row arrays are present, and decodes stored probabilities and execution tracks. Lean output cannot answer a row-level question by inferring missing arrays from a summary score.

<!-- example: retention-compiled-recording source=docs/examples/compiled-research/run.sh region=recording scenario=search interface=compiled role=executed -->
```bash
limen run --no-progress-bar \
  compiled-research/recording_lean.yaml \
  compiled-research/recording_full.yaml \
  compiled-research/recording_outputs.yaml
```

The [recording reference](Universal-Experiment-Loop.md#recording-policy) defines those switches and output schemas. [Log](Log.md) distinguishes captured aggregates from row-level evidence; parameter importance needs at least twenty distinct configurations, which the introductory selection runs do not supply. Use a larger declared analysis campaign for that question.

## Recording choices determine later analysis

The four optional payload groups default off for new persisted runs: predictions, alignment, trade ledger, and fold details. Enabling the payloads required by the next workflow before execution preserves that evidence. A later CSV reader cannot reconstruct omitted row arrays or fitted objects from scalar metrics.

Post-processing retains live round objects for serial analysis. Persisted recording and post-processing are separate choices. The [recording reference](Universal-Experiment-Loop.md#recording-policy) defines artifact groups, while [Log](Log.md) distinguishes aggregate analysis from retained row-level inspection.

## Candidate failures remain part of the campaign

The default error policy aborts on a candidate error. Ordinary unprotected serial MSQ runs can instead declare `on_round_error="record"` with a positive `max_round_errors` budget. The first candidate must succeed. Later failed candidates retain their identities and exception details without receiving scientific metrics or successful round payloads.

The error budget carries across resume. Exceeding it preserves the failed prefix and checkpoint, then raises. Parallel, factorized, walk-forward, protected, and legacy runs have separate admission rules and exclude this record-and-continue mode. [Failed-candidate contracts](Universal-Experiment-Loop.md#failed-candidates) define those boundaries.

## Continue the same question

Use one worker for arbitrary caller-owned Python research. Native concurrent rounds require the admitted recorded execution contract; `uel.workers` and `limen run --workers` state the worker count, including resume rules. See the [worker contract](Universal-Experiment-Loop.md) before changing that execution mode.

Continue to [successive research windows](Walk-Forward.md). The saved definition and completed candidate identities now provide the context needed to interpret a fold result.
