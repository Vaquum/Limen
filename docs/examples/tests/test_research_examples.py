from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[3]


def assert_python_research(python_project):
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
    from unittest.mock import patch
    from sklearn.pipeline import Pipeline
    calls = []
    original_fit = Pipeline.fit
    def observe(model, *args, **kwargs):
        calls.append(model)
        return original_fit(model, *args, **kwargs)
    with patch.object(Pipeline, 'fit', new=observe):
        for stop in range(13, 493):
            prefix = cohort.predict_all(bars[1428:1428+stop])
            np.testing.assert_allclose([point.prediction for point in prefix[12:]], [point.prediction for point in actual[:stop-12]], rtol=1e-12, atol=1e-12)
    assert calls == []
    for sensor, (coefficients, mean) in zip(cohort.members, before, strict=True):
        np.testing.assert_array_equal(sensor.fitted_state.named_steps['ridge'].coef_, coefficients)
        np.testing.assert_array_equal(sensor.fitted_state.named_steps['standardscaler'].mean_, mean)


def assert_compiled_research(compiled_project):
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


def test_python_research(research_proof):
    assert research_proof['python']['rounds'] == 4


def test_compiled_research(research_proof):
    assert research_proof['compiled']['rounds'] == 2


def test_observed_costs_preserve_caller_fits(cost_model, tmp_path, monkeypatch):
    import json
    from zipfile import ZipFile
    from limen.backtest.observed_costs import CostCalibration
    fixture = ROOT / 'tests/fixtures/observed_costs'
    history = tmp_path / 'history.json'
    with ZipFile(fixture / 'history.zip') as archive:
        history.write_bytes(archive.read('admitted.json'))
    calibration = CostCalibration.from_history(history, cutoff_ns=1785810791000000000)
    artifact, specification = tmp_path / 'calibration.json', tmp_path / 'specification.json'
    calibration.write(artifact)
    specification.write_text(json.dumps({'calibration_digest': calibration.digest}))
    fits, predictions = [], []
    original_fit, original_predict = cost_model.CallerModel.fit, cost_model.CallerModel.predict

    def forbidden(*args, **kwargs):
        raise AssertionError('A frozen caller fit or pinned calibration was re-estimated')

    def fit(model, training):
        fits.append(model)
        return original_fit(model, training)

    def predict(model, evaluation, mode, scale):
        values = original_predict(model, evaluation, mode, scale)
        predictions.append((values, values.to_numpy().tobytes()))
        monkeypatch.setattr(model, 'fit', forbidden)
        return values

    monkeypatch.setattr(CostCalibration, 'from_history', forbidden)
    monkeypatch.setattr(cost_model.CallerModel, 'fit', fit)
    monkeypatch.setattr(cost_model.CallerModel, 'predict', predict)
    for mode in ('binary', 'target_exposure'):
        for hour in (1, 2):
            experiment, comparison = cost_model.run_cost_experiment(artifact, specification, fixture / 'benchmark.json', tmp_path / f'{mode}-{hour}', mode=mode, hour=hour)
            assert experiment.manifest is None and experiment.experiment_log.height == 1
            assert comparison.report()['calibration_digest'] == calibration.digest
    assert len(fits) == len({id(model) for model in fits}) == len(predictions) == 4
    assert all(values.to_numpy().tobytes() == before for values, before in predictions)


CATALOGUE = __import__('json').loads((ROOT / 'docs/examples/catalogue.json').read_text())


@pytest.mark.parametrize(('interface', 'scenario'), [(interface, name) for name, interfaces in CATALOGUE['required_scenarios'].items() for interface in interfaces])
def test_complete_research_scenario(research_proof, interface, scenario):
    item = research_proof['scenarios'][(interface, scenario)]
    assert item['output_sha256'] and item['command']
    assert item['summary']
    if (interface, scenario) == ('compiled', 'recovery'):
        assert 0 < item['summary']['interrupted_rounds'] < item['summary']['resumed_rounds'] == 4
        assert item['summary']['uninterrupted_equivalence'] is True


@pytest.mark.parametrize('interface', ['python', 'compiled'])
def test_mixed_feedback_order_injected_results_and_checkpoint(interface, request, tmp_path):
    from scenario_oracles import verify_feedback
    scenarios = request.getfixturevalue(interface + '_scenarios')
    inputs = ROOT / 'tests/fixtures'
    result = scenarios.mixed_feedback(inputs, tmp_path / interface)
    verify_feedback(interface, result, inputs)


def test_cli_selection_matches_reconstruction_api(compiled_project, tmp_path):
    from wheel_proof import compare_selection_logs
    source = ROOT / 'docs/examples/compiled-research/selection.yaml'
    (tmp_path / 'compiled-research').mkdir()
    (tmp_path / 'inputs').mkdir()
    shutil.copyfile(source, tmp_path / 'compiled-research/selection.yaml')
    shutil.copyfile(ROOT / 'tests/fixtures/spot_15m_20250101_20250531.parquet',
                    tmp_path / 'inputs/spot_15m_20250101_20250531.parquet')
    command = json.loads((ROOT / 'docs/examples/commands.json').read_text())['compiled-yaml']
    environment = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    subprocess.run(shlex.split(command), cwd=tmp_path, env=environment, check=True, timeout=30)
    compare_selection_logs(compiled_project[1], tmp_path / 'results/dev/compiled-cli')


def test_preparation_preserves_training_fits_and_target_geometry(python_scenarios, compiled_scenarios, tmp_path):
    from scenario_oracles import verify_preparation
    inputs = ROOT / 'tests/fixtures'
    for interface, scenarios in [('python', python_scenarios), ('compiled', compiled_scenarios)]:
        output = tmp_path / interface
        result = scenarios.preparation(inputs, output)
        recorded = json.loads(json.dumps(result, allow_nan=False))
        verify_preparation(interface, recorded, output, inputs)


def test_acceptance_report_matches_recorded_tracks(compiled_scenarios, tmp_path):
    from scenario_oracles import verify_acceptance
    inputs = ROOT / 'tests/fixtures'
    result = compiled_scenarios.acceptance(inputs, tmp_path)
    verify_acceptance(json.loads(json.dumps(result, allow_nan=False)), tmp_path, inputs)


def test_common_baselines_preserve_signals_and_declared_economics(python_scenarios, compiled_scenarios, tmp_path):
    from scenario_oracles import verify_common_baselines
    inputs = ROOT / 'tests/fixtures'
    for interface, scenarios in [('python', python_scenarios), ('compiled', compiled_scenarios)]:
        output = tmp_path / interface
        result = scenarios.common_baselines(inputs, output)
        verify_common_baselines(interface, result, output, inputs)
