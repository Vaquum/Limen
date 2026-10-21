# Reserve and assess final observations

A final window is assessed after preparation, model fitting and selection have stopped changing. The [Python project](examples/python-research/README.md) owns that exclusion in its research code. The [Compiled project](examples/compiled-research/README.md) uses a guarded local reservation and a frozen selected Cohort.

Reserve a guarded project's final time window before running research. Supported experiments, Trainer refits and Log reconstruction then admit only native bars ending at or before the earliest active reservation. The exclusion applies before selectors, splitting and fitted preparation, including caller-supplied or reassigned frames.

This records local workflow protection. It cannot establish that a person, another project or an external process has never read these observations. Physical source materialization can exceed the admitted research prefix; provenance records both quantities.

## Prerequisites

Use the matching wheel and recorded input bundle for both projects. The guarded Compiled path additionally requires a Git project containing `limen.toml`, supported native OHLC data and serial built-in research. Preserve its local history as described below before cloning or restoring the project.

## Assess caller-owned final observations

The Python scenario selects on the first 1,440 quarter-hour rows, then assesses 479 next-bar decisions from the remaining observations. It reuses the evaluated fitted state. Its exclusion is a property of this code and its recorded input; it does not create reservation or exposure-history evidence.

<!-- example: holdout-python-assessment source=docs/examples/python-research/scenarios.py region=holdout scenario=holdout interface=python role=executed -->
```python
def holdout(data_dir, output):
    bars = quarter_hour(data_dir)
    cohort, _, selected, retained = project.run_research(bars, str(output / 'selection'))
    tape = cohort.predict_all(bars[1428:1920])[12:]
    # A decision at the current completed close trades the next open-to-close.
    predictions = []
    for point in tape[:-1]:
        if point.prediction is None:
            raise ValueError('A final assessment decision is unavailable: ' + str(point.reason))
        predictions.append(point.prediction)
    positions = np.sign(np.asarray(predictions, dtype=np.float64))
    next_returns = bars['close'].to_numpy()[1441:1920] / bars['open'].to_numpy()[1441:1920] - 1
    net = positions * next_returns - abs(positions) * .0002
    pl.DataFrame({'decision': [point.datetime for point in tape[:-1]], 'position': positions, 'net_return': net}).write_parquet(output / 'assessment.parquet')
    return {'selected': selected, 'retained_models': len(retained), 'assessment_rows': len(net), 'mean_net_return': float(net.mean()), 'fitting_cutoff': bars['datetime'][960].isoformat(), 'selection_cutoff': bars['datetime'][1440].isoformat(), 'protection': 'caller-owned exclusion; no project reservation or exposure-history guarantee'}
```

## Reserve and freeze a Compiled selection

The Compiled example uses its own Git project. `limen reserve` records the YAML's final interval before research, `limen run` enforces that reservation, and `limen assess --reservation` reconstructs the selected Cohort, freezes it and writes `final-report.json`. The local source keeps earlier exposure unknown.

<!-- example: holdout-compiled-assessment source=docs/examples/compiled-research/final/run.sh region=final_assessment scenario=holdout interface=compiled role=executed -->
```bash
git init --quiet compiled-research/final
reservation=$(limen reserve compiled-research/final/holdout.yaml)
limen run --no-progress-bar compiled-research/final/holdout.yaml
limen assess compiled-research/final/holdout.yaml --results results/dev/compiled-holdout --reservation "$reservation"
```

## Read the final evidence

The CLI writes `final-report.json` beside the experiment results. It records the reserved interval, selected membership, the frozen assessment specification, scenario outcomes, retained-row identity and exposure history. Selection uses the explicitly declared validation threshold score; the guarded case keeps the supported native snapshot policy. Classification uses the fixed next-bar target declared in the YAML; the last observation has no following label.

Local recorded files leave earlier external exposure unknown. Keep that classification beside any score: reservation protects this project's workflow, and cannot prove that the observations were never read elsewhere.

## Preserve the project

Keep `compiled-research/final/limen.toml`, its native holdout project marker, reservation history and assessment artifacts together. Copying only the result CSV does not preserve reservation or exposure evidence. Git backup follows ignore rules; preserve excluded holdout history separately.

The [guarded holdout reference](Guarded-Holdout-Reference.md) gives the supported modes, reservation lifecycle, failure conditions, retained evidence and optional component APIs. Continue to the [complete workflow](End-to-End-Workflow.md) to review these results alongside the rest of the research.
