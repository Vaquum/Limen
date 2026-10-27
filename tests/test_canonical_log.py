from copy import deepcopy
from functools import partial
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests.utils.assert_table_equal import assert_table_equal as assert_frame_equal

from limen.backtest.backtest_snapshot import BACKTEST_SNAPSHOT_COLUMNS, backtest_snapshot
from limen.backtest.execution_events import with_predictions
from limen.backtest.trade_execution import trade_execution
from limen.experiment import MLManifest, UniversalExperimentLoop
from limen.experiment.errors import StrictModeError
from limen.log.log import Log
from limen.sfd.reference_architecture import LogRegBinary
from limen.yaml import CompiledSFD, build_search_strategy, parse
from tests.test_walk_forward_uel import _config as _fold_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'
CONFUSION = ['tp', 'fp', 'tn', 'fn', 'precision', 'recall',
             'tp_mean_return_pct', 'fp_mean_return_pct', 'tn_mean_return_pct', 'fn_mean_return_pct']


def _config(*, event=False, random=False):
    config, errors = parse((ROOT / 'limen/yaml/templates/ridge_regressor.yaml').read_text())
    assert not errors
    manifest = config['sfd']['manifest']
    manifest['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file',
                               'params': {'file_path_or_url': str(FIXTURE)}}
    manifest['split_dates'] = {
        'train_start': '2024-01-01', 'train_end': '2024-01-04',
        'val_start': '2024-01-04', 'val_end': '2024-01-05',
        'test_start': '2024-01-05', 'test_end': '2024-01-07',
        'test_predict_guard': False,
    }
    manifest['features'][0]['params']['end'] = 2
    manifest['target'] = {'name': 'quantile_flag', 'class': 'limen.targets.QuantileBinaryTarget',
                          'fit_params': {'source_column': 'ret_1', 'quantile': 0.5},
                          'transform_params': {'shift': -1}}
    manifest['reference_architecture'] = f'limen.sfd.reference_architecture.{"random_binary" if random else "logreg_binary"}'
    manifest['backtest'] = {'fee_bps': 7.0, 'slip_bps': 3.0}
    if event:
        manifest['backtest']['product'] = {
            'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC',
            'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0.0,
        }
    else:
        manifest['backtest']['notional_rate'] = 0.5
    config['sfd']['params'] = {'random_weights': [0.0, 1.0]} if random else {
        'C': [0.1, 1.0], 'max_iter': [100], 'random_state': [42], 'n_jobs': [1],
    }
    config['uel'].update(n_permutations=2, search_strategy={'type': 'grid'}, checkpoint_interval=1)
    return config


def _loop(config, bars, path, *, search=True, callback=None):
    return UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=bars, experiment_dir=path, yaml_reference=config,
        search_strategy=build_search_strategy(config) if search else None,
        checkpoint_interval=1, feedback_interval=1, intra_callback=callback,
    )


def _run(loop, *, retained=False, workers=None, count=2):
    loop.run('captured', n_permutations=count, prep_each_round=True, random_search=False,
             progress_bar=False, post_processing=retained, workers=workers,
             context_params={'backtest_label': 'recorded', 'confusion_tag': 'captured'})


def _projection(frame, names, prefix):
    return frame[[prefix + name for name in names]].rename(
        columns={prefix + name: name for name in names}).reset_index(drop=True)


def _confusion_oracle(data, predictions):
    predictions = np.asarray(predictions, dtype=int)
    actuals = np.asarray(data['y_test'], dtype=int)
    masks = {'tp': (predictions == 1) & (actuals == 1),
             'fp': (predictions == 1) & (actuals == 0),
             'tn': (predictions == 0) & (actuals == 0),
             'fn': (predictions == 0) & (actuals == 1)}
    result: dict[str, float] = {key: int(mask.sum()) for key, mask in masks.items()}
    result['precision'] = result['tp'] / (result['tp'] + result['fp']) if result['tp'] + result['fp'] else 0.0
    result['recall'] = result['tp'] / (result['tp'] + result['fn']) if result['tp'] + result['fn'] else 0.0
    prices = data['price_data_for_backtest']
    opens, closes = prices['open'].to_numpy(), prices['close'].to_numpy()
    returns = (closes[1:] - opens[1:]) / opens[1:] * 100
    for key, mask in masks.items():
        selected = returns[mask[:-1] & np.isfinite(returns)]
        result[f'{key}_mean_return_pct'] = float(selected.mean()) if selected.size else np.nan
    return result


def _backtest_oracle(data, predictions):
    if '_trade_policy' in data:
        return trade_execution(with_predictions(data['_trade_inputs'], predictions), data['_trade_policy']).metrics
    prices = data['price_data_for_backtest']
    columns = {name: prices[name].to_numpy() for name in ('open', 'high', 'low', 'close')}
    columns.update(predictions=predictions, price_change=columns['close'] - columns['open'])
    return backtest_snapshot(columns, execution_lag_bars=1, fee_bps=7.0, slip_bps=3.0, notional_rate=0.5)


@pytest.fixture(scope='module')
def captured_runs(tmp_path_factory):
    bars = pl.read_parquet(FIXTURE, n_rows=192).with_columns(pl.lit(3600).alias('base_interval'))
    runs = []
    for mode in ('standard', 'msq', 'event', 'parallel', 'retained'):
        config = _config(event=mode == 'event')
        path = tmp_path_factory.mktemp(f'captured_{mode}')
        loop = _loop(config, bars, path, search=mode not in ('standard', 'retained'))
        _run(loop, retained=mode == 'retained', workers=2 if mode == 'parallel' else None)
        captured = loop.experiment_log.to_pandas()
        confusion, backtests = [], []
        for row in captured.to_dict('records'):
            params = {key: row[key] for key in config['sfd']['params']}
            prepared = loop.manifest.prepare_data(bars, params)
            prediction = loop.manifest.run_model(prepared, params)['_preds']
            confusion.append(_confusion_oracle(prepared, prediction))
            backtests.append(_backtest_oracle(prepared, prediction))
        csv_path = path / ('captured.csv' if mode in ('standard', 'retained') else 'results.csv')
        runs.append((mode, loop, captured, pd.DataFrame(confusion), pd.DataFrame(backtests), csv_path))
    return runs


def _forbidden(*args, **kwargs):
    raise AssertionError('Captured aggregate access attempted live preparation or execution')


def test_captured_aggregate_views(captured_runs, monkeypatch):
    monkeypatch.setattr(MLManifest, 'prepare_data', _forbidden)
    monkeypatch.setattr(MLManifest, 'run_model', _forbidden)
    monkeypatch.setattr(LogRegBinary, 'predict', _forbidden)
    monkeypatch.setattr(Log, '_permutation_prediction_performance', _forbidden)
    monkeypatch.setattr(import_module('limen.backtest.trade_execution'), 'trade_execution', _forbidden)
    monkeypatch.setattr('limen.log._snapshot_backtest_round.backtest_snapshot', _forbidden)
    for mode, loop, captured, confusion, backtests, csv_path in captured_runs:
        if mode != 'retained':
            assert loop.round_params == loop.preds == loop.scalers == loop._alignment == []
        monkeypatch.setattr(loop, 'prep', _forbidden)
        monkeypatch.setattr(loop, 'model', _forbidden)
        monkeypatch.setattr(loop, 'data', loop.data.reverse())
        log = Log(uel_object=loop)
        pure = Log(uel_object=SimpleNamespace(experiment_log=loop.experiment_log))
        encoded = Log(uel_object=loop, cols_to_multilabel=['backtest_pnl_bps_p50'] if mode != 'event' else ['backtest_total_return'])
        saved = Log(file_path=str(csv_path))
        assert 'backtest_label' in log.experiment_log and 'confusion_tag' in log.experiment_log
        metric = 'backtest_total_return' if mode == 'event' else 'backtest_pnl_bps_p50'
        assert metric not in encoded.experiment_log
        names = list(backtests.columns)
        if mode != 'event':
            assert names == BACKTEST_SNAPSHOT_COLUMNS
        for view in (log, pure, encoded, saved):
            assert_frame_equal(view.experiment_backtest_results(), _projection(captured, names, 'backtest_'), check_dtype=False)
            assert_frame_equal(view.experiment_confusion_metrics(), _projection(captured, CONFUSION, 'confusion_'), check_dtype=False)
            assert_frame_equal(view.experiment_backtest_results(), backtests, check_dtype=False)
            assert_frame_equal(view.experiment_confusion_metrics(), confusion, check_dtype=False)


def test_row_analysis_contract(captured_runs):
    for mode, loop, captured, _, _, csv_path in captured_runs:
        log = Log(uel_object=loop)
        if mode == 'retained':
            rows = []
            for ordinal, parameters in enumerate(loop.round_params):
                performance = log.permutation_prediction_performance(ordinal)
                np.testing.assert_array_equal(performance['predictions'], loop.preds[ordinal])
                rows.append(log.permutation_confusion_metrics('price_change', ordinal, id_cols=parameters))
                assert_frame_equal(performance, loop._log.permutation_prediction_performance(ordinal))
            expected = pl.concat(rows)
            assert_frame_equal(log.experiment_confusion_metrics('price_change'), expected)
            assert_frame_equal(loop.experiment_confusion_metrics, expected)
            assert_frame_equal(log.permutation_prediction_performance(-1),
                               log.permutation_prediction_performance(len(loop.round_params) - 1))
            for evidence in ('preds', '_alignment'):
                incomplete = Log(uel_object=loop)
                setattr(incomplete, evidence, getattr(incomplete, evidence)[:-1])
                with pytest.raises(ValueError, match=r'retained serial evidence.*post_processing=True'):
                    incomplete.permutation_prediction_performance(0)
            incomplete_scalers = Log(uel_object=loop, inverse_scaler=_forbidden)
            incomplete_scalers.scalers = incomplete_scalers.scalers[:-1]
            with pytest.raises(ValueError, match='fitted scaler'):
                incomplete_scalers.permutation_prediction_performance(0)
            assert {'n_kept', 'tp_x_median', 'fp_x_median', 'precision_pct', 'recall_pct', 'tp_fp_cohen_d', 'tp_fp_ks'}.issubset(expected.columns)
            assert log.experiment_confusion_metrics().columns == CONFUSION
            assert_frame_equal(log.experiment_confusion_metrics(), _projection(captured, CONFUSION, 'confusion_'), check_dtype=False)
        else:
            for view in (log, Log(file_path=str(csv_path))):
                with pytest.raises(ValueError, match=r'retained serial evidence.*post_processing=True'):
                    view.permutation_prediction_performance(0)
                with pytest.raises(ValueError, match=r'retained serial evidence.*post_processing=True'):
                    view.permutation_confusion_metrics('price_change', 0)
                with pytest.raises(ValueError, match=r'retained serial evidence.*post_processing=True'):
                    view.experiment_confusion_metrics('price_change')


def test_captured_fold_contract(tmp_path, monkeypatch):
    bars = pl.read_parquet(FIXTURE, n_rows=96)
    config = _fold_config()
    config['sfd']['manifest']['split_walk_forward'].update(test_bars=16, purge_bars=2, embargo_bars=1)
    loop = _loop(config, bars, tmp_path)
    _run(loop)
    folds = loop.fold_results
    expected = folds.select('id', 'fold', *[f'{name}_test' for name in BACKTEST_SNAPSHOT_COLUMNS]).rename(
        {f'{name}_test': name for name in BACKTEST_SNAPSHOT_COLUMNS}).to_pandas()
    assert folds.height == 4 and loop.experiment_log.height == 2
    assert folds['fold'].to_list() == [0, 1, 0, 1]
    assert folds['pnl_per_bar_bps_test'].n_unique() > 1
    for trial in loop.experiment_log.iter_rows(named=True):
        selected = folds.filter(pl.col('id') == trial['id'])
        assert trial['pnl_per_bar_bps_test'] == pytest.approx(selected['pnl_per_bar_bps_test'].mean())
    monkeypatch.setattr(type(loop.manifest), 'prepare_data', _forbidden)
    monkeypatch.setattr(type(loop.manifest), 'run_model', _forbidden)
    pure = Log(uel_object=SimpleNamespace(experiment_log=loop.experiment_log, fold_results=folds))
    assert_frame_equal(pure.experiment_backtest_results(), expected)
    with pytest.raises(ValueError, match='fold-specific'):
        pure.permutation_prediction_performance(0)
    with pytest.raises(ValueError, match=r'[Cc]aptured.*missing|[Cc]aptured.*incomplete'):
        Log(file_path=str(tmp_path / 'results.csv')).experiment_backtest_results()


def test_capture_errors_and_occurrences(tmp_path):
    bars = pl.read_parquet(FIXTURE, n_rows=192)
    config = _config(random=True)
    def repeat_after_failure(frame, msq):
        if frame.height == 2:
            msq.inject({'random_weights': 0.0}, prioritize=True)
    loop = _loop(config, bars, tmp_path, callback=repeat_after_failure)
    original = loop.prep
    prepared = 0
    def fail_second(data, round_params):
        nonlocal prepared
        prepared += 1
        if prepared == 2:
            raise StrictModeError('Recorded strict preparation failure')
        return original(data, round_params=round_params)
    loop.prep = fail_second
    _run(loop, retained=True, count=3)
    assert loop.experiment_log['_round_index'].to_list() == [0, 1, 2]
    assert loop.experiment_log['strict_mode_error'][1] == 'Recorded strict preparation failure'
    assert loop.experiment_log['_id'][0] == loop.experiment_log['_id'][2]
    assert len(loop.round_params) == len(loop.preds) == 2
    successful = loop.experiment_log.filter(pl.col('strict_mode_error').is_null()).to_pandas()
    log = Log(uel_object=loop)
    for view in (log, Log(file_path=str(tmp_path / 'results.csv'))):
        backtest = view.experiment_backtest_results()
        confusion = view.experiment_confusion_metrics()
        assert_frame_equal(backtest, _projection(successful, BACKTEST_SNAPSHOT_COLUMNS, 'backtest_'), check_dtype=False)
        assert_frame_equal(confusion, _projection(successful, CONFUSION, 'confusion_'), check_dtype=False)
        assert backtest['inventory_per_bar'].to_list() == [0.0, 0.0]
        assert backtest['cost_per_bar_bps'].to_list() == [0.0, 0.0]
        assert backtest['avg_win_bps'].is_nan().all() and backtest['avg_loss_bps'].is_nan().all()
        assert confusion['tp_mean_return_pct'].is_nan().all() and confusion['fp_mean_return_pct'].is_nan().all()
    np.testing.assert_array_equal(log.permutation_prediction_performance(1)['predictions'], loop.preds[1])
    for prefix, names, method in (('backtest_', BACKTEST_SNAPSHOT_COLUMNS, 'experiment_backtest_results'),
                                   ('confusion_', CONFUSION, 'experiment_confusion_metrics')):
        for removed in ([prefix + names[0]], [prefix + name for name in names]):
            incomplete = Log(uel_object=SimpleNamespace(experiment_log=loop.experiment_log.drop(removed)))
            with pytest.raises(ValueError, match=r'[Cc]aptured.*missing|[Cc]aptured.*incomplete'):
                getattr(incomplete, method)()
    empty = Log(uel_object=SimpleNamespace(experiment_log=loop.experiment_log.head(0)))
    assert empty.experiment_backtest_results().is_empty()
    with pytest.raises(ValueError, match='No objects to concatenate'):
        empty.experiment_confusion_metrics()


@pytest.fixture(scope='module')
def retained_keyword_run(tmp_path_factory):
    bars = pl.read_parquet(FIXTURE, n_rows=192).with_columns(pl.lit(3600).alias('base_interval'))
    loop = _loop(_config(), bars, tmp_path_factory.mktemp('keyword_replay'), search=False)
    original = loop.prep

    def prep(data, *, round_params):
        return original(data, round_params=round_params)

    loop.prep = prep
    _run(loop, retained=True, count=1)
    assert loop.experiment_log.height == len(loop.round_params) == len(loop.preds) == 1
    return loop


@pytest.mark.parametrize('shape', ('optional', 'kwargs', 'keyword_only', 'partial'))
def test_prediction_replay_uses_retained_keyword_parameters(retained_keyword_run, shape):
    log = Log(uel_object=retained_keyword_run)
    original = log.prep
    parameters = log.round_params[0]
    saved_parameters = deepcopy(log.round_params)
    saved_predictions = np.array(log.preds[0], copy=True)
    expected = original(log.data, round_params=parameters)
    prices = log._get_test_data_with_all_cols(0)
    calls = []

    def optional(data, round_params=None):
        calls.append(round_params)
        return original(data, round_params=round_params)

    def keywords(data, **kwargs):
        return optional(data, **kwargs)

    def keyword_only(data, *, round_params):
        return optional(data, round_params=round_params)

    log.prep = {'optional': optional, 'kwargs': keywords, 'keyword_only': keyword_only,
                'partial': partial(optional, round_params={'C': 99.0})}[shape]
    performance = log.permutation_prediction_performance(0)
    assert len(calls) == 1 and calls[0] is parameters
    assert log.round_params == saved_parameters
    np.testing.assert_array_equal(log.preds[0], saved_predictions)
    np.testing.assert_array_equal(performance['predictions'], saved_predictions)
    np.testing.assert_array_equal(performance['actuals'], expected['y_test'])
    assert len(performance) == prices.height == len(saved_predictions)
    for column in ('datetime', 'open', 'close'):
        np.testing.assert_array_equal(performance[column].to_numpy(), prices[column].to_numpy())
    np.testing.assert_array_equal(performance['price_change'], prices['close'] - prices['open'])


def test_prediction_replay_propagates_body_typeerror_once(retained_keyword_run):
    log = Log(uel_object=retained_keyword_run)
    original = log.prep
    saved_parameters = deepcopy(log.round_params)
    saved_predictions = np.array(log.preds[0], copy=True)
    failure = TypeError('Recorded preparation body failure')
    calls = []

    def prep(data, round_params=None):
        calls.append('explicit_params' if round_params is not None else 'default_params')
        if round_params is not None:
            raise failure
        return original(data, round_params={})

    log.prep = prep
    with pytest.raises(TypeError) as raised:
        log.permutation_prediction_performance(0)
    assert raised.value is failure and raised.value.__context__ is None
    assert calls == ['explicit_params']
    assert log.round_params == saved_parameters
    np.testing.assert_array_equal(log.preds[0], saved_predictions)


def test_post_processing_propagates_replay_typeerror_after_recording(tmp_path):
    bars = pl.read_parquet(FIXTURE, n_rows=192).with_columns(pl.lit(3600).alias('base_interval'))
    loop = _loop(_config(), bars, tmp_path, search=False)
    original_prep, original_model = loop.prep, loop.model
    completed = False
    calls = []
    failure = TypeError('Recorded finalization preparation failure')

    def prep(data, round_params=None):
        calls.append('explicit_params' if round_params is not None else 'default_params')
        if completed and round_params is not None:
            raise failure
        return original_prep(data, round_params=round_params or {})

    def model(data, round_params):
        nonlocal completed
        result = original_model(data=data, round_params=round_params)
        completed = True
        return result

    loop.prep, loop.model = prep, model
    with pytest.raises(TypeError) as raised:
        _run(loop, retained=True, count=1)
    assert raised.value is failure and raised.value.__context__ is None
    assert completed and calls == ['explicit_params', 'explicit_params']
    assert loop.experiment_log.height == len(loop.round_params) == len(loop.preds) == 1
    assert loop.experiment_log['strict_mode_error'].is_null().all()
    assert pl.read_csv(tmp_path / 'captured.csv').height == 1
    assert loop.experiment_confusion_metrics is None
