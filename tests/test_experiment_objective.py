import copy
import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from limen.backtest.execution_events import with_predictions
from limen.backtest.trade_execution import trade_execution
from limen.calibration import grid_threshold_optimizer
from limen.experiment import UniversalExperimentLoop
from limen.experiment.manifest_core import DataSourceConfig
from limen.experiment.reducer import BudgetReducer, FocusReducer
from limen.inference import Trainer
from limen.yaml import CompiledSFD, build_search_strategy
from limen.yaml.compiler import build_pruning_strategies
from tests.stubs.stubs import make_msq
from tests.test_record_model_outputs import _config as _binary_config
from tests.test_record_model_outputs import recorded_source

COLUMN = 'val_backtest_total_return'


def objective_config(direction='maximize', *, calibrated=True):
    config = _binary_config(calibrated=calibrated)
    manifest = config['sfd']['manifest']
    manifest['split_dates'] = {
        'train_start': '2024-01-01', 'train_end': '2024-01-16',
        'val_start': '2024-01-17', 'val_end': '2024-01-23',
        'test_start': '2024-01-24', 'test_end': '2024-01-30',
    }
    manifest['objective'] = {'metric': 'backtest_total_return', 'direction': direction}
    manifest['backtest'] = {
        'product': {'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC',
                    'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0.0},
        'fee_bps': 5.0, 'slip_bps': 5.0,
    }
    if calibrated:
        del manifest['calibration']['threshold_function']['params']['metric']
    config['sfd']['params']['C'] = [0.1, 1.0]
    return config


def _prepared(config, bars=None, *, execution_source=None):
    manifest = CompiledSFD(config).manifest()
    if execution_source is not None:
        manifest.backtest_config = replace(manifest.backtest_config, execution_data_source=DataSourceConfig(execution_source, {'kline_size': 3600}))
    params = {key: values[0] for key, values in config['sfd']['params'].items()}
    data = manifest.prepare_data(recorded_source() if bars is None else bars, params)
    return manifest, params, data


def _assert_ledger(actual, expected):
    assert actual.contract_digest == expected.contract_digest
    assert actual.metrics == expected.metrics
    for field in ('states', 'intents', 'fills', 'episodes', 'funding'):
        assert getattr(actual, field).equals(getattr(expected, field))


@pytest.mark.parametrize('direction', ('maximize', 'minimize'))
def test_recorded_objective_selects_return(direction):
    config = objective_config(direction)
    manifest, params, data = _prepared(config)
    result = manifest.run_model(data, params)
    context = data['_trade_context']
    prediction = result['_model'].predict({'x_test': data['x_val']})
    probabilities = prediction['_probs']
    candidates = [2.0, 0.6, 0.55, 0.5, 0.45, 0.4]
    scores = [trade_execution(with_predictions(context.partitions[1], (probabilities >= threshold).astype(np.int8)), context.policy).metrics['total_return'] for threshold in candidates]
    best = max(scores) if direction == 'maximize' else min(scores)
    assert result[COLUMN] == pytest.approx(best)
    assert result['val_score'] == pytest.approx(result[COLUMN])
    assert result['optimal_threshold'] == pytest.approx(candidates[scores.index(best)])
    if direction == 'maximize':
        assert result['optimal_threshold'] == 2.0
        assert min(scores) < result[COLUMN]
        assert grid_threshold_optimizer(data['y_val'], probabilities, threshold_min=0.4, threshold_max=0.6, threshold_step=0.05)[0] != result['optimal_threshold']
        assert np.count_nonzero(result['_preds']) == 0
    oracle = trade_execution(with_predictions(context.partitions[2], result['_preds']), context.policy)
    _assert_ledger(data['_trade_ledger'], oracle)
    for key, value in oracle.metrics.items():
        assert result[f'backtest_{key}'] == pytest.approx(value)
    native = copy.deepcopy(config)
    del native['sfd']['manifest']['objective']
    native_manifest, native_params, native_data = _prepared(native)
    native_manifest.set_objective(direction=direction)
    native_result = native_manifest.run_model(native_data, native_params)
    assert native_result[COLUMN] == pytest.approx(result[COLUMN])
    assert native_result['optimal_threshold'] == result['optimal_threshold']


@pytest.mark.parametrize('use_calibration,use_threshold', ((True, True), (False, True), (True, False), (False, False), (None, None)))
def test_objective_scores_the_fitted_decision_rule(use_calibration, use_threshold):
    config = objective_config(calibrated=use_calibration is not None)
    manifest, params, data = _prepared(config)
    if use_calibration is not None:
        params.update(use_calibration=use_calibration, use_threshold=use_threshold)
    result = manifest.run_model(data, params)
    predictions = result['_model'].predict({'x_test': data['x_val']})['_preds']
    context = data['_trade_context']
    oracle = trade_execution(with_predictions(context.partitions[1], predictions), context.policy)
    assert result[COLUMN] == pytest.approx(oracle.metrics['total_return'])
    if not use_threshold:
        assert result['val_score'] is None


def test_validation_execution_isolates_first_test_bar_and_candidate_state():
    config = objective_config('minimize')
    config['sfd']['manifest']['split_dates']['val_end'] = config['sfd']['manifest']['split_dates']['test_start']
    bars = recorded_source().with_columns(pl.lit(3600).alias('base_interval'))
    original, params, first = _prepared(config, bars, execution_source=recorded_source)
    baseline = original.run_model(first, params)
    context = first['_trade_context']
    validation = context.partitions[1]
    # The separate source includes a held-out open at the inclusive endpoint.
    assert validation.observations.filter(pl.col('start_ns') == validation.partition_end_ns).height
    from limen.experiment._objective import prepare_objective
    scorer = prepare_objective(first, original.architecture_function, original.prediction_calibration_config)
    changed_observations = validation.observations.with_columns(*[
        pl.when(pl.col('start_ns') >= validation.partition_end_ns).then(pl.col(key) * 1.05).otherwise(pl.col(key)).alias(key)
        for key in ('open', 'high', 'low', 'close')
    ])
    changed_partition = replace(validation, observations=changed_observations)
    second = dict(first)
    second['_alignment'] = copy.deepcopy(first['_alignment'])
    second['_trade_context'] = replace(context, partitions=(context.partitions[0], changed_partition, context.partitions[2]))
    second.pop('_trade_ledger')
    changed_scorer = prepare_objective(second, original.architecture_function, original.prediction_calibration_config)
    assert scorer.inputs.observations.equals(changed_scorer.inputs.observations)
    assert scorer.inputs.sources == changed_scorer.inputs.sources
    before = json.dumps({key: value for key, value in first['_alignment'].items() if not key.startswith('_')}, default=str, sort_keys=True)
    private_inputs = first['_alignment']['_trade_inputs']
    probs = baseline['_model'].predict({'x_test': first['x_val']})['_probs']
    _ = grid_threshold_optimizer(first['y_val'], probs, metric=scorer, _objective_maximize=False)
    assert json.dumps({key: value for key, value in first['_alignment'].items() if not key.startswith('_')}, default=str, sort_keys=True) == before
    assert first['_alignment']['_trade_inputs'] is private_inputs
    changed_result = original.run_model(second, params)
    assert changed_result['optimal_threshold'] == baseline['optimal_threshold']
    assert changed_result[COLUMN] == baseline[COLUMN]
    oracle = trade_execution(with_predictions(context.partitions[2], changed_result['_preds']), context.policy)
    _assert_ledger(second['_trade_ledger'], oracle)


@pytest.mark.parametrize('direction', ('maximize', 'minimize'))
def test_objective_reconstruction(direction, tmp_path):
    config = objective_config(direction)
    config['sfd']['manifest']['split_dates']['test_predict_guard'] = False
    bars = recorded_source()
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=bars, experiment_dir=tmp_path,
                                   yaml_reference=config, search_strategy=build_search_strategy(config), checkpoint_interval=1)
    loop.run('return_objective', n_permutations=1, prep_each_round=True, progress_bar=False,
             record_execution=True, record_model_outputs=True)
    record = json.loads((tmp_path / 'round_data.jsonl').read_text().splitlines()[0])
    pid = loop.experiment_log['_id'][0]
    sensor = Trainer(tmp_path, data=bars).train([pid])[0]
    predictions = sensor.predict_all(bars)
    by_time = {str(row.datetime): row.prediction for row in predictions if row.prediction is not None}
    _, _, data = _prepared(config, bars)
    context = data['_trade_context']
    times = context.partitions[2].signals.select('row_id').join(context.model_rows[2], on='row_id', how='left', maintain_order='left')['datetime'].to_list()
    assert len(times) == len(record['preds'])
    for dt, expected in zip(times, record['preds'], strict=True):
        assert by_time[str(dt)] == expected
    if direction == 'maximize':
        assert record['optimal_threshold'] == 2.0
        assert record['threshold_rule'] == '>='
        assert set(record['preds']) == {0}


@pytest.mark.parametrize('probability', (float('nan'), float('inf'), -0.1, 1.1))
def test_objective_rejects_invalid_recorded_probabilities(probability):
    manifest, params, data = _prepared(objective_config())
    result = manifest.run_model(data, params)
    probs = result['_model'].predict({'x_test': data['x_val']})['_probs'].copy()
    probs[0] = probability
    with pytest.raises(ValueError, match='finite'):
        grid_threshold_optimizer(data['y_val'], probs, metric=lambda labels, preds: 0.0, _objective_maximize=True)


def test_objective_propagates_candidate_execution_errors_and_ties():
    manifest, params, data = _prepared(objective_config())
    result = manifest.run_model(data, params)
    probs = result['_model'].predict({'x_test': data['x_val']})['_probs']
    def failed(labels, preds):
        raise ValueError('Recorded candidate cannot execute')
    with pytest.raises(ValueError, match='cannot execute'):
        grid_threshold_optimizer(data['y_val'], probs, metric=failed, _objective_maximize=True)
    for direction in (True, False):
        assert grid_threshold_optimizer(data['y_val'], probs, metric=lambda labels, preds: 0.0, _objective_maximize=direction) == (2.0, 0.0)
    with pytest.raises(ValueError, match='finite'):
        grid_threshold_optimizer(data['y_val'], probs, metric=lambda labels, preds: float('nan'), _objective_maximize=True)


@pytest.mark.parametrize('direction', ('maximize', 'minimize'))
def test_existing_objective_selection_ignores_test_metric(direction):
    config = objective_config(direction)
    manifest, params, data = _prepared(config)
    rows = []
    for value in (0.1, 1.0, 10.0):
        round_params = {**params, 'C': value, 'use_threshold': False}
        result = manifest.run_model(data, round_params)
        rows.append({'C': value, COLUMN: result[COLUMN], 'backtest_total_return': result['backtest_total_return']})
    log = pl.DataFrame(rows)
    assert log[COLUMN].n_unique() > 1
    # The probe reorders observed return values; it creates no market fixture.
    changed = log.with_columns(pl.col('backtest_total_return').reverse())
    assert not changed['backtest_total_return'].equals(log['backtest_total_return'])
    maximize = direction == 'maximize'
    expected = log.sort(COLUMN, descending=maximize)['C'][0]
    worst = log.sort(COLUMN, descending=not maximize)['C'][0]
    config['uel']['pruning_strategies'] = [
        {'type': 'focus', 'params': {'breakthrough_threshold': 0.0}},
        {'type': 'budget', 'params': {'trim_strategy': 'worst_first', 'max_permutations': 1, 'check_after_pct': 0.0}},
    ]
    focus, budget = build_pruning_strategies(config)
    assert isinstance(focus, FocusReducer) and isinstance(budget, BudgetReducer)
    for frame in (log, changed):
        assert focus._find_best_row(frame, ['C'])[1]['C'] == expected
        msq, _, _ = make_msq(params={'C': [0.1, 1.0, 10.0]})
        interventions = budget._trim_worst_first(frame, msq, 1)
        assert interventions and interventions[0]['value'] == worst
