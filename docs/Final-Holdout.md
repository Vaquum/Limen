# Reserve and assess final observations

A final window is assessed after preparation, model fitting and selection have stopped changing. The [Python project](examples/python-research/README.md) owns that exclusion in its research code. The [Compiled project](examples/compiled-research/README.md) uses a guarded local reservation and a frozen selected Cohort.

Reserve a guarded project's final time window before running research. Supported experiments, Trainer refits and Log reconstruction then admit only native bars ending at or before the earliest active reservation. The exclusion applies before selectors, splitting and fitted preparation, including caller-supplied or reassigned frames.

This records local workflow protection. It cannot establish that a person, another project or an external process has never read these observations. Physical source materialization can exceed the admitted research prefix; provenance records both quantities.

## Prerequisites

Use the matching wheel and recorded input bundle for both projects. The guarded Compiled path additionally requires a Git project containing `limen.toml` and supported native OHLC data. Preserve its local history as described below before cloning or restoring the project.

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

The same exclusion and revision lease cover profiling, saved resume, factorized reuse, spawned workers and selected walk-forward folds under their ordinary recording requirements. Profiling records its purpose and failed samples in project history. Source formation, declared execution/funding inputs and compatible feature extensions retain their own ordered input evidence; search hooks cannot widen admitted observations. These research modes do not widen final-assessment eligibility. See the [guarded holdout reference](Guarded-Holdout-Reference.md) for restrictions and retained lineage.

<!-- example: holdout-compiled-assessment source=docs/examples/compiled-research/final/run.sh region=final_assessment scenario=holdout interface=compiled role=executed -->
```bash
git init --quiet compiled-research/final
reservation=$(limen reserve compiled-research/final/holdout.yaml)
limen run --no-progress-bar compiled-research/final/holdout.yaml
limen assess compiled-research/final/holdout.yaml --results results/dev/compiled-holdout --reservation "$reservation"
```

## Research support and final eligibility remain separate

| Stage | Admitted scope |
|---|---|
| Reservation-bound research | Independently valid source preparation, profiling, saved resume, factorization, native workers, and walk-forward, with original mode exclusions preserved |
| Snapshot final assessment | Frozen ordinary YAML-backed binary LogReg or LightGBM Sensors/Cohorts with genuine completed Trainer provenance and already fitted state |
| Event final assessment | One ordinary selected Sensor with a recorded round locator, frozen economics, and empty scenario list |

Final assessment excludes caller-owned Python selections, nonordinary splits, formed bars, and unsupported model/output types. The protected research stage can use capabilities that do not qualify for that final route. The [canonical workflow table](Experiment-Manifest.md#supported-research-workflows) and [guarded reference](Guarded-Holdout-Reference.md#final-assessment) define their exact relationship.

## Preserve event economics

The separate `final-event/holdout.yaml` project selects one Sensor and retains the event policy used during research. Its source clock, product, costs and exposure remain fixed through final assessment.

<!-- example: holdout-event-policy source=docs/examples/compiled-research/final-event/holdout.yaml region=event_policy scenario=holdout interface=compiled role=executed -->
```yaml
    backtest:
      fee_bps: 7.0
      slip_bps: 3.0
      notional_rate: 0.25
      product:
        kind: cash_spot
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1.0e-9
        min_notional: 0.0
```

Set `holdout.scenarios: []` to assess that frozen contract once. Event assessment requires one ordinary selected Sensor; a contract-bearing Cohort is refused. The ordinary scenario project above keeps its separate snapshot policy.

<!-- example: holdout-event-command source=docs/examples/compiled-research/final-event/run.sh region=final_assessment scenario=holdout interface=compiled role=executed -->
```bash
git init --quiet compiled-research/final-event
reservation=$(limen reserve compiled-research/final-event/holdout.yaml)
limen run --no-progress-bar compiled-research/final-event/holdout.yaml
limen assess compiled-research/final-event/holdout.yaml --results results/dev/compiled-holdout_event --reservation "$reservation"
```

This variant writes a schema-2 final report with the research and assessment contract digests, policy, initial equity, prediction tape, executed signals and complete ledger frames. Its terminal rule marks remaining exposure without forcing a close. Preserve this project's reservation history and evidence together.

## Read the final evidence

The CLI writes `final-report.json` beside the experiment results. It records the reserved interval, selected membership, the frozen assessment specification, scenario outcomes, retained-row identity and exposure history. Selection uses the explicitly declared validation threshold score; the scenario case keeps the supported native snapshot policy, and the event case retains its frozen contract. Classification uses the fixed next-bar target declared in the YAML; the last observation has no following label.

Local recorded files leave earlier external exposure unknown. Keep that classification beside any score: reservation protects this project's workflow, and cannot prove that the observations were never read elsewhere.

## Read the current final result

The Python result reports its 479 later next-bar decisions under the caller’s exclusion. The Compiled result reports its reserved rows and exposure classification. An `unknown` exposure history remains unknown even when local reservation enforcement passed.

<!-- result: final-python-result interface=python scenario=holdout fields=assessment_rows,mean_net_return,fitting_cutoff,selection_cutoff -->

<!-- result: final-compiled-result interface=compiled scenario=holdout fields=final_rows,exposure,event.schema_version -->

## Preserve the project

Keep each guarded project's `limen.toml`, native holdout project marker, reservation history and assessment artifacts together. Copying only the result CSV does not preserve reservation or exposure evidence. Git backup follows ignore rules; preserve excluded holdout history separately.

The [guarded holdout reference](Guarded-Holdout-Reference.md) gives the supported modes, reservation lifecycle, failure conditions, retained evidence and optional component APIs. Continue to the [complete workflow](End-to-End-Workflow.md) to review these results alongside the rest of the research.
