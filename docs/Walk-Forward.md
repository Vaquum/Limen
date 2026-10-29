# Research across successive windows

A chronological split answers a question about one fitting window and what followed it. Walk-forward research repeats that decision as the available history changes. Each fold needs its own fitted state, validation decision and later assessment window.

## Prerequisites

Use the matching Limen wheel and recorded-input bundle from the [Python](examples/python-research/README.md) and [Compiled](examples/compiled-research/README.md) project pages. Their environment and entry-point instructions also apply to this chapter's `walk_forward` scenario.

The two projects use the first 384 recorded hourly Bitcoin observations. The Compiled YAML names their explicit bundled subset, `spot_1h_20240101_20240116.parquet`; Python selects those rows from the full recorded file. They retain different research decisions. The Python project owns its folds and selection; the Compiled project records a pipeline whose selected folds can be reconstructed. Their results are not interchangeable estimates of the same model.

## Choose the information available to each fit

An anchored window retains its original beginning as observations accumulate. A rolling window advances that beginning. In both cases, the assessment observations follow the fitting and validation observations. Purging and embargo change which rows can take part; they do not prove that an arbitrary feature or label is causal.

The Python scenario creates two folds for each window choice. Each assessment window contains 48 hourly observations. Four bars are purged and three are embargoed. Within each fitting population, the caller reserves its final fifth for validation, fits two Ridge candidates and chooses the validation result before using the later observations.

## Own the folds in Python

This is the `walk_forward` scenario of the [Python project](examples/python-research/README.md). Its preparation and inference functions are in the complete project; the excerpt is not a separate program. Recorded inputs and the entry-point commands are described there.

<!-- example: book-python-walk-forward source=docs/examples/python-research/scenarios.py region=walk_forward scenario=walk_forward interface=python role=executed -->
```python
def walk_forward(data_dir, output):
    bars = hourly(data_dir, 384)
    results = []
    for anchored in (True, False):
        for index, (train, test) in enumerate(split_walk_forward(bars, n_folds=2, test_bars=48, purge_bars=4, embargo_bars=3, anchored=anchored)):
            fitting = int(train.height * .8)
            candidates = []
            for alpha in (.1, 1.0):
                params = {'window': 4, 'alpha': alpha}
                fitted = project.fit(project.prepare_split(train[:fitting], params), params)
                validation = project.prepare_split(train[fitting:], params)
                score = project.backtest(validation, fitted.predict(validation.select('momentum')))
                candidates.append((score, alpha, fitted))
            score, alpha, fitted = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
            sensor = Sensor.from_model(fitted, prepare=prepare_hourly_live, params={'window': 4, 'alpha': alpha}, fitting_cutoff=train['datetime'][fitting], selection_cutoff=test['datetime'][0], output_id='next_bar_return')
            history = bars.filter(pl.col('datetime') <= test['datetime'][-1])
            predictions = [point for point in sensor.predict_all(history) if point.datetime >= test['datetime'][0]]
            results.append({'anchored': anchored, 'fold': index, 'train_rows': train.height, 'test_rows': test.height, 'train_end': train['datetime'][-1].isoformat(), 'test_start': test['datetime'][0].isoformat(), 'alpha': alpha, 'validation_net_return': score, 'predictions': [point.prediction for point in predictions], 'available_at_ns': [point.available_at_ns for point in predictions], 'test_open': test['open'].to_list(), 'test_close': test['close'].to_list()})
    return {'folds': results, 'ownership': 'caller-owned folds and selection; no recorded compiled fold promotion'}
```

Each selected Sensor wraps the evaluated fitted object. Its fitting cutoff describes the fit, while its selection cutoff prevents a decision before the assessment window. The inference preparation declares when the completed hourly observation becomes available. Those clocks answer different questions: a row may belong to the assessment population before its complete observation is available for action.

The independent assertions rebuild the Ridge solution and validation return from the recorded prices, then compare the fold's predictions and availability clocks. A successful run establishes those stated decisions on those observations. It does not establish that a different caller function respects the same boundaries.

## Reconstruct recorded Compiled folds

The [Compiled project](examples/compiled-research/README.md) runs its recorded `walk_forward` definition over the same bounded source. The Manifest supplies fold preparation, fitted targets and transforms, calibration and the configured validation objective. Each selected fold has its own fitted and calibration state.

<!-- example: book-compiled-walk-forward source=docs/examples/compiled-research/run.sh region=walk_forward scenario=walk_forward interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/walk_forward.yaml
limen assess compiled-research/walk_forward.yaml --results results/dev/compiled-walk_forward
limen run --no-progress-bar compiled-research/acceptance.yaml
```

`Trainer` receives both the recorded round and an explicit zero-based fold. It refits and validates that fold against its retained witnesses. This is reconstruction under a recorded contract, not restoration of arbitrary saved model weights. The project retains the compressed round records and their digest so the qualified result can be inspected.

The resulting Cohort binds the recorded fold identities. Its common selection cutoff covers the campaign's available source and selection decisions, including competitors. It must not be read as each fold's final fitting timestamp. The [Trainer reference](Trainer.md#recorded-walk-forward-folds) describes the required witnesses and refusal conditions; [Cohort](Cohort.md#recorded-walk-forward-members) describes compatible composition.

## Read the actual windows

The Python rows show the four assessed folds and their selected alpha. The Compiled result names its two reconstructed fold members. These counts follow the respective definitions above; they are not a comparison of forecast quality.

<!-- result: windows-python-geometry interface=python scenario=walk_forward fields=folds[].anchored,folds[].fold,folds[].train_rows,folds[].test_rows,folds[].alpha -->

<!-- result: windows-compiled-geometry interface=compiled scenario=walk_forward fields=folds,members -->

## Acceptance reports retain the full trial comparison

The Compiled `acceptance.yaml` case records two trial return tracks across its two folds. Its report retains 96 synchronous assessment observations, the supplied trial count, deflated Sharpe probability, PBO, and the declared threshold verdicts:

<!-- result: windows-compiled-acceptance interface=compiled scenario=walk_forward fields=acceptance.n_trials,acceptance.n_bars,acceptance.n_blocks,acceptance.deflated_sharpe_probability,acceptance.pbo,acceptance.verdicts -->

Threshold failures remain advisory report outcomes. They do not erase a completed run or its retained alternatives. [Research evidence](Research-Evidence.md) develops their statistical interpretation. The [family-report contract](Benchmark.md#comparing-a-recorded-experiment-family) adds explicit supplied-experiment comparisons, compatible groups, and missing-track accounting.

## Keep the assessment separate from selection

Repeated windows provide several later assessment populations. Using their outcomes to choose the research definition makes those outcomes part of selection. A collection of folds therefore does not create an untouched final sample.

Inspect the fold geometry, fitting and selection cutoffs, calibration population, validation objective and retained predictions before comparing results. Python supplies those decisions through the caller's functions and assertions. Compiled Research supplies the declared pipeline and recorded promotion evidence. Continue to [final observations](Final-Holdout.md) when the question requires an assessment reserved from research, or to [execution](Backtest.md) when predictions must be interpreted as trades.

Continue to [trading behavior](Trading-Behavior.md) to interpret those later decisions under their declared execution policy.
