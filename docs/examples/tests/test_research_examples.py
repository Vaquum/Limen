from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_python_research(python_project):
    module, bars, (cohort, loop, selected, retained) = python_project
    assert loop.experiment_log.height == len(retained) == 4
    assert loop.data.equals(bars[:1440])
    assert len(cohort.members) == len(selected) == 2
    assert [item['window'] for item in selected] == [4, 12]
    close, opening = bars['close'].to_numpy(), bars['open'].to_numpy()
    forecasts = []
    for sensor, params in zip(cohort.members, selected, strict=True):
        key = params['window'], params['alpha']
        assert sensor.fitted_state is retained[key]
        assert sensor.fitting_cutoff == bars['datetime'][960]
        assert sensor.selection_cutoff == bars['datetime'][1440]
        prepared = module.prep(bars[:1440], params)
        assert [prepared['train'].height, prepared['validation'].height] == [947, 467]
        inputs = pl.DataFrame({'momentum': close[972:1439] / close[972-params['window']:1439-params['window']] - 1})
        predictions = sensor.fitted_state.predict(inputs)
        positions = np.sign(predictions)
        oracle = np.mean(positions * (close[973:1440] / opening[973:1440] - 1) - abs(positions) * 0.0002)
        row = loop.experiment_log.filter((pl.col('window') == params['window']) & (pl.col('alpha') == params['alpha']))
        assert row['validation_net_return'][0] == pytest.approx(oracle, rel=1e-12, abs=1e-12)
        best = loop.experiment_log.filter(pl.col('window') == params['window']).sort(['validation_net_return', 'alpha'], descending=[True, False])
        assert best['alpha'][0] == params['alpha']
        live = pl.DataFrame({'momentum': close[1440:1920] / close[1440-params['window']:1920-params['window']] - 1})
        forecasts.append(sensor.fitted_state.predict(live))
    actual = cohort.predict_all(bars[1428:1920])[12:]
    assert len(actual) == 480
    assert all(point.reason is None for point in actual)
    np.testing.assert_allclose([point.prediction for point in actual], np.mean(forecasts, axis=0), rtol=1e-12, atol=1e-12)
    assert [point.available_at_ns for point in actual] == (bars[1440:1920]['datetime'].dt.epoch('ns') + 900_000_000_000).to_list()
    before = [(sensor.fitted_state.named_steps['ridge'].coef_.copy(), sensor.fitted_state.named_steps['standardscaler'].mean_.copy()) for sensor in cohort.members]
    for stop in (13, 253, 492):
        prefix = cohort.predict_all(bars[1428:1428+stop])
        np.testing.assert_allclose([point.prediction for point in prefix[12:]], [point.prediction for point in actual[:stop-12]], rtol=1e-12, atol=1e-12)
    for sensor, (coefficients, mean) in zip(cohort.members, before, strict=True):
        np.testing.assert_array_equal(sensor.fitted_state.named_steps['ridge'].coef_, coefficients)
        np.testing.assert_array_equal(sensor.fitted_state.named_steps['standardscaler'].mean_, mean)


def test_compiled_research(compiled_project):
    module, output, (loop, sensors, selected) = compiled_project
    assert loop.experiment_log.height == 2
    assert len(sensors) == len(selected) == 1
    assert sensors[0].permutation_id == selected[0]
    assert selected[0] == loop.experiment_log.sort('val_backtest_total_return', descending=True)['id'][0]
    assert (output / 'metadata.json').is_file()
    assert (output / 'results.csv').is_file()
    from limen import Trainer
    replay = Trainer(output).train(selected)[0]
    import polars as pl
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_15m_20250101_20250531.parquet')
    first, second = sensors[0].predict_all(bars), replay.predict_all(bars)
    assert len(first) == len(second) > 0
    assert [(p.datetime, p.prediction, p.reason) for p in first] == [(p.datetime, p.prediction, p.reason) for p in second]
