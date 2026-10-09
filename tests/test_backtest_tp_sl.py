import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from ruamel.yaml import YAML

from limen.backtest.backtest_snapshot import BACKTEST_SNAPSHOT_COLUMNS, backtest_snapshot, _snapshot_with_execution
from limen.backtest.long_flat_strategy import long_flat_strategy
from limen.cli.commands.run import run_experiment
from limen.data import HistoricalData
from limen.experiment import MLManifest, RuleBasedManifest, UniversalExperimentLoop
from limen.experiment._backtest_provenance import SOURCE_ROW, replay_prices, restore_source_rows, source_splits, validate_witness
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.experiment.param_search.random_strategy import RandomStrategy
from limen.features import dollar_bar_crash_reversal
from limen.inference import Trainer
from limen.log._experiment_backtest_results import experiment_backtest_results
from limen.sfd import reference_architecture
from limen.sfd.foundational_sfd import ridge_regressor as ridge_sfd
from limen.sfd.reference_architecture.ridge_regressor import ridge_regressor
from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy, _compounded_trade_pnl_summary, rule_based
from limen.scalers import RobustScaler
from limen.targets import IdentityTarget
from limen.yaml import parse, validate
from limen.yaml.compiler import build_manifest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/dollar_bar_crash_reversal_15m.parquet'
TEMPLATE = ROOT / 'limen/yaml/templates/ridge_regressor.yaml'


@pytest.fixture(scope='module')
def market():
    bars = pl.read_parquet(FIXTURE)
    signals = {}
    for flow in (-0.5, 0.5):
        featured = dollar_bar_crash_reversal(bars, momentum_threshold_bps=-525.0, flow_z_threshold=flow, hold_minutes=30)
        signals[flow] = featured['dollar_bar_crash_reversal_position'].to_numpy()
    return bars, signals


def _columns(market, start=2897, end=2925, flow=-0.5):
    bars, signals = market
    columns = {col: bars[col].slice(start, end - start).to_numpy() for col in ('open', 'high', 'low', 'close')}
    columns['predictions'] = signals[flow][start:end]
    columns['price_change'] = columns['close'] - columns['open']
    return columns


def _equity(entry, exit_price, fee=10.0, slip=5.0):
    f, s = fee / 10000, slip / 10000
    return (exit_price / entry) * (1 - f) * (1 - s) / (1 + s) - f


def _same_metrics(left, right):
    assert list(left) == list(right)
    for key in left:
        np.testing.assert_equal(left[key], right[key], err_msg=key)


def _rule_manifest():
    return (RuleBasedManifest().set_split_config(1, 1, 2)
            .with_strategy([{'id': 'entry', 'type': 'relative', 'column': 'close', 'operator': '>', 'other_column': 'open'}], entry='entry')
            .with_reference_architecture(rule_based))


def _ml_manifest():
    return (MLManifest().set_split_config(1, 1, 2)
            .with_target_label('close', IdentityTarget).with_reference_architecture(ridge_regressor))


def _fixture_source(**_kwargs):
    return pl.read_parquet(FIXTURE)


def test_manifest_tp_sl_literal_reference_and_null(market):
    manifest = MLManifest().set_backtest_config(fee_bps='fee', slip_bps='slip', notional_rate='size')
    costs = manifest.backtest_config
    manifest.set_backtest_config(fee_bps=costs.fee_bps, slip_bps=costs.slip_bps,
                                 notional_rate=costs.notional_rate, take_profit_bps='{tp}', stop_loss_bps='sl')
    expected = {'fee_bps': 10.0, 'slip_bps': 5.0, 'notional_rate': 0.5, 'take_profit_bps': 100.0, 'stop_loss_bps': None}
    assert manifest.resolve_backtest_config({'fee': 10, 'slip': 5, 'size': 0.5, 'tp': 100, 'sl': None}) == expected
    for reference in ('missing', '{missing}'):
        manifest.set_backtest_config(take_profit_bps=reference)
        with pytest.raises(ValueError, match=r'take_profit_bps.*missing'):
            manifest.resolve_backtest_config({})
    manifest.set_backtest_config(take_profit_bps=75.0)
    assert manifest.backtest_config.fee_bps == 5.0
    assert manifest.resolve_backtest_config({})['stop_loss_bps'] is None
    assert MLManifest().resolve_backtest_config({}) == {}
    for strategy in (GridStrategy, RandomStrategy):
        choices = strategy(ParamDomain({'tp': [None, 50.0]}), seed=42)
        assert {next(choices)['tp'] for _ in range(16 if strategy is RandomStrategy else 2)} == {None, 50.0}
    for name in reference_architecture.__all__:
        signature = inspect.signature(getattr(reference_architecture, name))
        assert not {'take_profit_bps', 'stop_loss_bps'} & signature.parameters.keys()
    bars = market[0].head(1200)
    baseline = ridge_sfd.manifest().prepare_data(bars, {})
    for factory in (_ml_manifest, _rule_manifest):
        for config in (None, {}, {'fee_bps': 20.0}, {'take_profit_bps': None, 'stop_loss_bps': None}):
            ordinary = factory()
            if config is not None:
                ordinary.set_backtest_config(**config)
            data = ordinary.prepare_data(bars, {})
            assert '_backtest_provenance' not in data
            assert '_backtest_provenance' not in data['_alignment']
    assert '_backtest_provenance' not in baseline['_alignment']
    configured = ridge_sfd.manifest().set_backtest_config(take_profit_bps='tp')
    prepared = configured.prepare_data(bars, {'tp': None})
    assert all(isinstance(split.retained, range) for split in prepared['_backtest_provenance'].splits)
    assert SOURCE_ROW not in prepared['_feature_names']
    for key in ('x_train', 'x_val', 'x_test'):
        assert_frame_equal(prepared[key], baseline[key])
    for key in ('y_train', 'y_val', 'y_test'):
        assert prepared[key].equals(baseline[key])

    class CheckedTarget(IdentityTarget):
        def __init__(self, train_data, target_name):
            assert SOURCE_ROW not in train_data.columns
            super().__init__(train_data, target_name)

        def transform(self, frame):
            assert SOURCE_ROW not in frame.columns
            return super().transform(frame)

    class CheckedScaler(RobustScaler):
        def __init__(self, x_train):
            assert SOURCE_ROW not in x_train.columns
            super().__init__(x_train)

        def transform(self, frame):
            assert SOURCE_ROW not in frame.columns
            return super().transform(frame)

    compressed = (_ml_manifest().with_target_label('close', CheckedTarget)
                  .set_scaler(CheckedScaler).set_pca_compression()
                  .set_backtest_config(take_profit_bps='tp'))
    compressed_data = compressed.prepare_data(bars, {'tp': None, 'auto_pca': True, 'pca_k': 2})
    assert SOURCE_ROW not in compressed_data['_pca_input_feature_names']
    assert SOURCE_ROW not in compressed_data['_feature_names']


def test_yaml_tp_sl_compile_and_validate():
    config, errors = parse(TEMPLATE.read_text())
    assert not errors
    config['sfd']['params']['tp'] = [None, 50.0]
    config['sfd']['params']['sl'] = [None, 25.0]
    config['sfd']['manifest']['backtest'] = {'take_profit_bps': ' {tp} ', 'stop_loss_bps': ' {sl} '}
    assert validate(config).valid
    manifest = build_manifest(config)
    assert manifest.resolve_backtest_config({'tp': None, 'sl': 25.0})['stop_loss_bps'] == 25.0
    for field, values in [('take_profit_bps', [0, True, float('inf'), float('nan')]), ('stop_loss_bps', [0, 10000, True])]:
        for value in values:
            invalid = copy.deepcopy(config)
            key = 'tp' if field == 'take_profit_bps' else 'sl'
            invalid['sfd']['params'][key] = [None, value]
            result = validate(invalid)
            assert not result.valid
            assert any(field in error.path for error in result.errors)
            with pytest.raises(ValueError, match=field):
                manifest.resolve_backtest_config({'tp': value if key == 'tp' else None, 'sl': value if key == 'sl' else None})
            invalid['sfd']['manifest']['backtest'][field] = value
            assert not validate(invalid).valid
    invalid = copy.deepcopy(config)
    invalid['sfd']['manifest']['backtest']['take_profit_bps'] = '{missing}'
    assert not validate(invalid).valid


def test_tp_sl_disabled_matches_existing_ledger(market):
    columns = _columns(market)
    original = long_flat_strategy(columns['predictions'], columns['open'], columns['close'], columns['price_change'])
    metrics, actual = _snapshot_with_execution(columns)
    assert list(metrics) == BACKTEST_SNAPSHOT_COLUMNS
    for before, after in zip(original, actual, strict=True):
        np.testing.assert_array_equal(before, after)
    _same_metrics(metrics, backtest_snapshot(columns, take_profit_bps=None, stop_loss_bps=None))
    calls = []
    def callback(predictions, open_px, close_px, price_change, **kwargs):
        calls.append(kwargs)
        return long_flat_strategy(predictions, open_px, close_px, price_change, **kwargs)
    _same_metrics(metrics, backtest_snapshot(columns, strategy=callback))
    assert calls == [{'execution_lag_bars': 1, 'fee_bps': 5.0, 'slip_bps': 5.0}]
    backtest_snapshot(columns, execution_lag_bars=0)
    malformed = dict(columns, high=columns['high'][:-1])
    _same_metrics(metrics, backtest_snapshot(malformed))


def test_tp_sl_enabled_unreachable_matches_legacy_execution(market):
    columns = _columns(market, end=3480)
    assert columns['high'].max() < columns['close'].min() * (1 + 1e9 / 10000)
    assert columns['low'].min() > columns['close'].max() * (1 - 9999.99 / 10000)
    _, actual = _snapshot_with_execution(columns, take_profit_bps=1e9, stop_loss_bps=9999.99, fee_bps=10.0)
    expected = long_flat_strategy(columns['predictions'], columns['open'], columns['close'], columns['price_change'], fee_bps=10.0)
    np.testing.assert_array_equal(actual.pos, expected.pos)
    np.testing.assert_allclose(actual.gross, expected.gross, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual.net, expected.net, atol=1e-12, rtol=0)
    flat = _columns(market, start=0, end=50)
    assert not flat['predictions'].any()
    for lag in (1, len(flat['close'])):
        ledger, result = _snapshot_with_execution(flat, take_profit_bps=50.0, execution_lag_bars=lag)
        assert not result.pos.any()
        assert ledger['trades_per_bar'] == 0
        assert np.isnan(ledger['avg_win_bps']) and np.isnan(ledger['avg_loss_bps'])


def test_tp_sl_real_fixture_entry_relative_thresholds(market):
    columns = _columns(market)
    entry = columns['close'][0]
    assert entry == 76190.52
    for options, exit_row, price in [({'take_profit_bps': 150.49}, 7, entry * (1 + 150.49 / 10000)),
                                     ({'stop_loss_bps': 30.0}, 3, entry * (1 - 30 / 10000))]:
        _, result = _snapshot_with_execution(columns, **options)
        np.testing.assert_array_equal(np.flatnonzero(result.pos), np.arange(1, exit_row + 1))
        assert result.gross[exit_row] == pytest.approx(price / columns['close'][exit_row - 1] - 1, abs=1e-14)
        for fee, slip, size in [(0.0, 0.0, 1.0), (20.0, 3.0, 0.1)]:
            _, changed = _snapshot_with_execution(columns, fee_bps=fee, slip_bps=slip, notional_rate=size, **options)
            np.testing.assert_array_equal(changed.pos, result.pos)
            np.testing.assert_array_equal(changed.gross, result.gross)
    genuine = _columns(market, start=4013, end=4047, flow=0.5)
    E, touch = genuine['close'][0], genuine['high'][1]
    bps = (touch / E - 1) * 10000
    assert E == 66359.6 and touch == 66500.0
    assert E * (1 + bps / 10000) == touch
    _, touched = _snapshot_with_execution(genuine, take_profit_bps=bps)
    assert np.flatnonzero(touched.pos).tolist() == [1]
    assert touched.gross[1] == pytest.approx(touch / E - 1)


def test_tp_sl_real_fixture_opening_gaps(market):
    stop = _columns(market)
    E = stop['close'][0]
    assert stop['open'][1] == 76180.71 < E * (1 - 0.8 / 10000)
    _, sl = _snapshot_with_execution(stop, stop_loss_bps=0.8)
    assert np.flatnonzero(sl.pos).tolist() == [1]
    assert sl.gross[1] == pytest.approx(76180.71 / 76190.52 - 1)
    assert stop['high'][1] >= E * (1 + 10 / 10000)
    _, prioritized = _snapshot_with_execution(stop, stop_loss_bps=0.8, take_profit_bps=10.0)
    np.testing.assert_equal(prioritized.gross, sl.gross)
    target = _columns(market, start=4208, end=4235, flow=0.5)
    E = target['close'][0]
    TP = E * (1 + 150.49 / 10000)
    assert target['high'][1:11].max() < TP
    assert target['close'][10] < TP <= target['open'][11]
    assert target['open'][11] == 61317.05
    _, tp = _snapshot_with_execution(target, take_profit_bps=150.49)
    assert np.flatnonzero(tp.pos).tolist() == list(range(1, 12))
    assert tp.gross[11] == pytest.approx(TP / target['close'][10] - 1)
    assert np.prod(1 + tp.net) == pytest.approx(_equity(E, TP, fee=5), abs=1e-12)


def test_tp_sl_real_fixture_dual_touch_and_exit_precedence(market):
    for end in (2899, 2925):
        columns = _columns(market, end=end)
        E = columns['close'][0]
        SL, TP = E * (1 - 10 / 10000), E * (1 + 10 / 10000)
        assert SL < columns['open'][1] < TP
        assert columns['low'][1] < SL < TP < columns['high'][1]
        _, result = _snapshot_with_execution(columns, take_profit_bps=10.0, stop_loss_bps=10.0)
        assert np.flatnonzero(result.pos).tolist() == [1]
        assert result.gross[1] == pytest.approx(SL / E - 1)
        assert np.prod(1 + result.net) == pytest.approx(_equity(E, SL, fee=5), abs=1e-12)
    last_held = _columns(market, start=2919, end=2923)
    assert last_held['predictions'][0] == 1 and last_held['predictions'][1] == 0
    E = last_held['close'][0]
    _, result = _snapshot_with_execution(last_held, stop_loss_bps=20.0)
    assert last_held['low'][1] < E * (1 - 20 / 10000) < last_held['open'][1]
    assert result.gross[1] == pytest.approx(-20 / 10000)


def test_tp_sl_real_fixture_reentry_and_terminal_exit(market):
    columns = _columns(market, end=3480)
    _, result = _snapshot_with_execution(columns, stop_loss_bps=10.0)
    assert result.pos[1] == 1
    assert not result.pos[2:3432-2897].any()
    assert result.pos[3432-2897] == 1
    assert columns['predictions'][2921-2897] == 0
    prefix = _columns(market, start=2905, end=2910)
    assert prefix['predictions'].all()
    _, terminal = _snapshot_with_execution(prefix, take_profit_bps=1e9)
    assert terminal.pos.tolist() == [0, 1, 1, 1, 1]
    assert np.prod(1 + terminal.net) == pytest.approx(_equity(prefix['close'][0], prefix['close'][-1], fee=5), abs=1e-12)
    _, ordinary = _snapshot_with_execution(_columns(market), take_profit_bps=1e9)
    assert np.flatnonzero(ordinary.pos).tolist() == list(range(1, 24))
    _, lagged = _snapshot_with_execution(prefix, take_profit_bps=1e9, execution_lag_bars=2)
    assert lagged.pos.tolist() == [0, 0, 1, 1, 1]
    assert np.prod(1 + lagged.net) == pytest.approx(_equity(prefix['close'][1], prefix['close'][-1], fee=5), abs=1e-12)


def test_tp_sl_real_fixture_fill_costs_and_sizing(market):
    columns = _columns(market)
    E = columns['close'][0]
    for options, exit_row, X in [({'stop_loss_bps': 0.8}, 1, columns['open'][1]),
                                 ({'take_profit_bps': 30.0}, 1, E * 1.003),
                                 ({'stop_loss_bps': 30.0}, 3, E * 0.997),
                                 ({'take_profit_bps': 150.49}, 7, E * (1 + 150.49 / 10000))]:
        ledger, result = _snapshot_with_execution(columns, fee_bps=10, slip_bps=5, **options)
        equity = np.cumprod(1 + result.net)
        assert equity[exit_row] == pytest.approx(_equity(E, X), abs=1e-12)
        for row in range(1, exit_row):
            assert equity[row] == pytest.approx(columns['close'][row] / (E * 1.0005) - 0.001, abs=1e-12)
        assert result.gross[exit_row] == pytest.approx(X / columns['close'][exit_row-1] - 1)
        scaled, unscaled_result = _snapshot_with_execution(columns, fee_bps=10, slip_bps=5, notional_rate=0.5, **options)
        np.testing.assert_equal(unscaled_result.net, result.net)
        assert scaled['inventory_per_bar'] == pytest.approx(result.pos.mean() * 0.5)
        assert scaled['pnl_per_bar_bps'] == pytest.approx(result.net.mean() * 0.5 * 10000)
        assert scaled['trades_per_bar'] == ledger['trades_per_bar']


def test_tp_sl_rejects_invalid_or_censored_prices(market):
    columns = _columns(market)
    for key, values in [('take_profit_bps', [0, -1, True, float('inf'), 1e308]), ('stop_loss_bps', [0, -1, True, 10000, float('nan')])]:
        for value in values:
            with pytest.raises(ValueError, match=key):
                backtest_snapshot(columns, **{key: value})
    for lag in (0, True, 1.5):
        with pytest.raises(ValueError, match='execution_lag_bars'):
            backtest_snapshot(columns, take_profit_bps=50.0, execution_lag_bars=lag)
    for col in ('high', 'low'):
        absent = {key: value for key, value in columns.items() if key != col}
        with pytest.raises(ValueError, match=col):
            backtest_snapshot(absent, take_profit_bps=50.0)
    with pytest.raises(ValueError, match='equal lengths'):
        backtest_snapshot(dict(columns, high=columns['high'][:-1]), take_profit_bps=50.0)
    with pytest.raises(ValueError, match='1D'):
        backtest_snapshot(dict(columns, high=columns['high'].reshape(-1, 1)), take_profit_bps=50.0)
    with pytest.raises(ValueError, match=r'OHLC.*bounds'):
        backtest_snapshot(dict(columns, high=columns['low']), take_profit_bps=50.0)
    with pytest.raises(ValueError, match=r'OHLC.*positive'):
        backtest_snapshot(dict(columns, low=columns['price_change']), take_profit_bps=50.0)
    with pytest.raises(ValueError, match='price_change'):
        backtest_snapshot(dict(columns, price_change=columns['close']), take_profit_bps=50.0)
    with pytest.raises(ValueError, match=r'predictions.*0 or 1'):
        backtest_snapshot(dict(columns, predictions=columns['price_change']), take_profit_bps=50.0)
    with pytest.raises(ValueError, match='custom strategy'):
        backtest_snapshot(columns, take_profit_bps=50.0, strategy=lambda *_args, **_kwargs: None)
    bars = market[0].slice(2897, 28)
    observed = bars.select('datetime', 'high').filter(pl.col('datetime') != bars['datetime'][0])
    sparse_high = bars.select('datetime').join(observed, on='datetime', how='left', maintain_order='left')['high'].to_numpy()
    assert np.isnan(sparse_high[0]) and columns['predictions'][0] == 1
    with pytest.raises(ValueError, match=r'OHLC.*finite'):
        backtest_snapshot(dict(columns, high=sparse_high), take_profit_bps=50.0)
    backtest_snapshot(dict(columns, high=sparse_high))
    m = ridge_sfd.manifest().set_backtest_config(take_profit_bps=50.0)
    data = m.prepare_data(market[0].head(1200), {})
    data.pop('price_data_for_backtest')
    with pytest.raises(ValueError, match='price_data_for_backtest'):
        m.run_model(data, {})
    with pytest.raises(ValueError, match='open/close'):
        RuleBasedStrategy()._backtest_split(bars.drop('open'), columns['predictions'], {'take_profit_bps': 50.0})


def test_tp_sl_real_fixture_provenance_boundaries(market):
    bars = market[0]
    small = bars.head(16)
    timestamp = small['datetime'][12]
    assert timestamp.isoformat() == '2026-01-01T08:13:29+00:00'
    def remove_row(frame):
        assert SOURCE_ROW not in (frame.collect_schema().names() if isinstance(frame, pl.LazyFrame) else frame.columns)
        return frame.filter(pl.col('datetime') != timestamp)
    for factory in (_ml_manifest, _rule_manifest):
        m = factory().add_indicator(remove_row).set_backtest_config(take_profit_bps='tp')
        if isinstance(m, MLManifest):
            m.set_strict_mode(True)
        with pytest.raises(ValueError, match='censored interior'):
            m.prepare_data(small, {'tp': None})
        selected = factory().set_pre_split_data_selector(remove_row).set_backtest_config(take_profit_bps=50.0)
        prepared = selected.prepare_data(small, {})
        direct_sparse = factory().set_backtest_config(take_profit_bps=50.0).prepare_data(remove_row(small), {})
        assert prepared['_backtest_provenance'] == direct_sparse['_backtest_provenance']
    duplicate_window = bars.slice(2825, 35)
    assert bars['datetime'][2843] == bars['datetime'][2844]
    for filtered in (False, True):
        m = _ml_manifest().set_backtest_config(take_profit_bps='tp')
        if filtered:
            m.add_indicator(lambda frame: frame.filter(pl.col('close') != 79800.0))
        with pytest.raises(ValueError, match='ambiguous alignment'):
            m.prepare_data(duplicate_window, {'tp': None})
    identified = source_splits([duplicate_window])[0]
    assert identified.select('datetime', 'open').is_duplicated().any()
    suffix = duplicate_window.slice(20).select('datetime', 'open')
    restored = restore_source_rows(identified, suffix)
    assert restored[SOURCE_ROW].to_list() == list(range(20, len(duplicate_window)))
    assert_frame_equal(restored.drop(SOURCE_ROW), suffix)
    ambiguous = restore_source_rows(identified, duplicate_window.slice(18).select('datetime', 'open'))
    assert SOURCE_ROW not in ambiguous.columns
    rule = _rule_manifest().set_backtest_config(take_profit_bps=50.0)
    data = rule.prepare_data(duplicate_window, {})
    rule.run_model(data, {})
    assert sum(len(data[key]) for key in ('train', 'val', 'test')) == len(duplicate_window)
    normal = _ml_manifest().set_backtest_config(take_profit_bps=50.0).prepare_data(small, {})
    witness = normal['_backtest_provenance']
    source = small.slice(8)
    assert replay_prices(source, witness, 8).height == 8
    with pytest.raises(ValueError, match='prediction length'):
        replay_prices(source, witness, 7)
    with pytest.raises(ValueError, match='source identity'):
        replay_prices(source.reverse(), witness, 8)
    with pytest.raises(ValueError, match='datetime'):
        replay_prices(source.drop('datetime'), witness, 8)
    with pytest.raises(ValueError, match='provenance'):
        validate_witness(None)
    with pytest.raises(ValueError, match='source identity order'):
        _ml_manifest().set_backtest_config(take_profit_bps=50.0).prepare_data(small.reverse(), {})
    altered = replace(witness, splits=(*witness.splits[:2], replace(witness.splits[2], error='backtest source identity count mismatch')))
    with pytest.raises(ValueError, match='source identity count'):
        validate_witness(altered)
    missing = _ml_manifest().set_backtest_config(take_profit_bps='tp')
    with pytest.raises(ValueError, match='price_data_for_backtest'):
        missing.prepare_data(small.drop('high'), {'tp': None})


def test_tp_sl_cached_preparation_null_enabled_null(market, tmp_path):
    m = ridge_sfd.manifest().set_backtest_config(take_profit_bps='tp')
    seen = []
    cached = []
    def prep(data, round_params=None):
        if not cached:
            cached.append(m.prepare_data(data, round_params or {}))
        return cached[0]
    def model(data, round_params):
        result = m.run_model(data, round_params)
        seen.append((id(data), data['backtest_take_profit_bps'], data['backtest_stop_loss_bps'], data['_backtest_provenance'], dict(result)))
        return result
    params = {'tp': [None, 50.0], 'zz_trial': [0, 1]}
    adapter = SimpleNamespace(__name__=__name__, params=lambda: params, prep=prep, model=model)
    loop = UniversalExperimentLoop(sfd=adapter, data=market[0].head(1200), search_strategy=GridStrategy(ParamDomain(params)), experiment_dir=tmp_path)
    loop.run('cached', n_permutations=3, prep_each_round=False, progress_bar=False)
    assert [item[1] for item in seen] == [None, 50.0, None]
    assert len({item[0] for item in seen}) == 1
    assert all(item[2] is None and item[3] == seen[0][3] for item in seen)
    np.testing.assert_equal(seen[0][4]['_preds'], seen[2][4]['_preds'])
    _same_metrics({k:v for k,v in seen[0][4].items() if k.startswith('backtest_')}, {k:v for k,v in seen[2][4].items() if k.startswith('backtest_')})
    broken = dict(cached[0])
    broken.pop('price_data_for_backtest')
    def never_train(data, **_params):
        pytest.fail('configured null candidate reached the architecture before preflight')
    guarded = _ml_manifest().with_reference_architecture(never_train).set_backtest_config(take_profit_bps='tp')
    with pytest.raises(ValueError, match='price_data_for_backtest'):
        guarded.run_model(broken, {'tp': None})
    m.backtest_config = None
    data = prep(market[0].head(1200))
    data.update(backtest_take_profit_bps=50.0, backtest_stop_loss_bps=25.0)
    m._apply_backtest_cost(data, {})
    assert 'backtest_take_profit_bps' not in data and 'backtest_stop_loss_bps' not in data
    configured = SimpleNamespace(params=lambda: params, manifest=lambda: ridge_sfd.manifest())
    with pytest.raises(ValueError, match='prep_each_round must be True'):
        UniversalExperimentLoop(sfd=configured, data=market[0]).run('invalid', n_permutations=1, progress_bar=False)


def test_tp_sl_real_fixture_inline_post_run_parity(market, tmp_path):
    bars = market[0]
    manifest = ridge_sfd.manifest().set_backtest_config(fee_bps='fee', slip_bps=3.0, notional_rate=0.5, take_profit_bps='tp', stop_loss_bps='sl')
    params = {'fee': [20.0], 'tp': [None, 75.0], 'sl': [None, 50.0], 'solver': ['svd']}
    sfd = SimpleNamespace(params=lambda: params, manifest=lambda: manifest)
    loop = UniversalExperimentLoop(sfd=sfd, data=bars, experiment_dir=tmp_path)
    loop.run('parity', n_permutations=4, prep_each_round=True, random_search=False, post_processing=True, progress_bar=False)
    assert loop.experiment_backtest_results.shape == (4, 20)
    for i in range(4):
        for key, value in loop.experiment_backtest_results.iloc[i].items():
            np.testing.assert_equal(loop.experiment_log[f'backtest_{key}'][i], value)
        fresh = manifest.prepare_data(bars, loop.round_params[i])
        assert fresh['_backtest_provenance'] == loop._alignment[i]['_backtest_provenance']
        np.testing.assert_equal(manifest.run_model(fresh, loop.round_params[i])['_preds'], loop.preds[i])
    legacy = SimpleNamespace(round_params=loop.round_params, permutation_prediction_performance=loop._log.permutation_prediction_performance)
    old = experiment_backtest_results(legacy)
    assert old['inventory_per_bar'].iloc[0] != loop.experiment_backtest_results['inventory_per_bar'].iloc[0]
    baseline_inventory = float(old['inventory_per_bar'].iloc[0])
    corrected_inventory = float(loop.experiment_backtest_results['inventory_per_bar'].iloc[0])
    assert corrected_inventory == pytest.approx(baseline_inventory / 2, abs=0.0001)
    assert 'high' not in loop._log.permutation_prediction_performance(0).columns
    missing_witness = loop._log._alignment[0].pop('_backtest_provenance')
    with pytest.raises(ValueError, match='provenance'):
        loop._log.experiment_backtest_results()
    loop._log._alignment[0]['_backtest_provenance'] = missing_witness
    original_data = loop._log.data
    loop._log.data = bars.reverse()
    with pytest.raises(ValueError, match='source identity'):
        loop._log.experiment_backtest_results()
    loop._log.data = original_data


def test_tp_sl_real_fixture_rule_based_entry_evaluation(market, tmp_path):
    m = _rule_manifest().set_backtest_config(fee_bps=10.0, slip_bps=5.0, notional_rate=0.5, take_profit_bps=20.0, stop_loss_bps=10.0)
    bars = market[0].slice(2890, 130)
    data = m.prepare_data(bars, {})
    results = m.run_model(data, {})
    model = RuleBasedStrategy()
    for split in ('train', 'val', 'test'):
        frame = data[split]
        positions = model._apply_logic(frame, data['strategy']).cast(pl.Int8).to_numpy()
        cols = {col:frame[col].to_numpy() for col in ('open', 'high', 'low', 'close')}
        cols.update(predictions=positions, price_change=cols['close']-cols['open'])
        metrics, execution = _snapshot_with_execution(cols, fee_bps=10.0, slip_bps=5.0, notional_rate=0.5, take_profit_bps=20.0, stop_loss_bps=10.0)
        expected = model._backtest_split(frame, positions, m.resolve_backtest_config({}))
        pnl, count = _compounded_trade_pnl_summary(execution, 0.5)
        held = execution.pos > 0
        changes = np.diff(np.r_[False, held, False].astype(int))
        trades = [np.prod(1 + execution.net[a:b] * 0.5) - 1
                  for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True)]
        assert count == len(trades)
        assert pnl == pytest.approx(round(float(np.mean(trades)) * 10000, 1) if trades else float('nan'), nan_ok=True)
        assert expected['pnl_per_trade_bps'] == pytest.approx(pnl, nan_ok=True)
        assert expected['num_executed_trades'] == count
        for key, value in metrics.items():
            np.testing.assert_equal(expected[key], value)
        assert SOURCE_ROW not in frame.columns
    assert '_preds' in results
    sfd = SimpleNamespace(params=lambda:{'take_profit': [20.0]}, manifest=lambda:m)
    loop = UniversalExperimentLoop(sfd=sfd, data=bars, experiment_dir=tmp_path)
    loop.run('rule', n_permutations=1, prep_each_round=True, post_processing=True, progress_bar=False)
    assert loop.experiment_backtest_results is None


def test_tp_sl_real_fixture_trainer_reconstruction(market, tmp_path, monkeypatch):
    config, errors = parse(TEMPLATE.read_text())
    assert not errors
    config['sfd']['manifest']['split_dates'] = {
        'train_start': '2026-01-01', 'train_end': '2026-01-23',
        'val_start': '2026-01-24', 'val_end': '2026-01-31',
        'test_start': '2026-02-01', 'test_end': '2026-02-06',
    }
    config['sfd']['manifest']['backtest'] = {'take_profit_bps': 75.0, 'stop_loss_bps': 50.0, 'fee_bps': 10.0}
    config['sfd']['params'] = {key:[values[0]] for key,values in ridge_sfd.params().items()}
    config['uel'].update(n_permutations=1, output_path=str(tmp_path/'experiment'))
    yaml_path = tmp_path/'tp-sl.yaml'
    with yaml_path.open('w') as handle:
        YAML().dump(config, handle)
    monkeypatch.setattr(HistoricalData, 'get_spot_klines', staticmethod(_fixture_source))
    assert run_experiment(yaml_path, progress_bar=False)
    path = tmp_path/'experiment'
    record = json.loads((path/'round_data.jsonl').read_text().splitlines()[0])
    trainer = Trainer(path, data=market[0])
    sensors = trainer.train([str(record['round_id'])])
    assert len(sensors) == 1
    assert sensors[0].predict(market[0]).reason is None
    results = pl.read_csv(path/'results.csv')
    assert results.select(pl.selectors.starts_with('backtest_')).width == 20


def test_tp_sl_real_fixture_prefix_causality(market):
    columns = _columns(market, end=3480)
    _, full = _snapshot_with_execution(columns, take_profit_bps=150.49, stop_loss_bps=30.0)
    for count in (2, 4, 8, 24, 400, len(columns['close'])-1):
        prefix = {key:value[:count] for key,value in columns.items()}
        _, shorter = _snapshot_with_execution(prefix, take_profit_bps=150.49, stop_loss_bps=30.0)
        np.testing.assert_array_equal(shorter.pos[:-1], full.pos[:count-1])
        np.testing.assert_allclose(shorter.gross[:-1], full.gross[:count-1], atol=1e-12, rtol=0)
        np.testing.assert_allclose(shorter.net[:-1], full.net[:count-1], atol=1e-12, rtol=0)
