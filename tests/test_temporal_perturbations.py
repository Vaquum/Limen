from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from limen.data import HistoricalData
from limen.data.utils import lookback_window, sample_regime_blocks
from limen.features.classify_market_regime import RegimeConfig, classify_market_regime

SOURCE = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'


@pytest.fixture(scope='module')
def market():
    return pl.read_parquet(SOURCE, n_rows=2500)


def test_lookback_preserves_recorded_rows_and_explicit_source_end(market):
    original = market.clone()
    expected = market.filter(pl.col('datetime') >= market['datetime'][-1] - timedelta(days=2))
    assert_frame_equal(lookback_window(market, 2), expected)
    with patch('limen.data.historical_data._read_any_file', return_value=market):
        source = HistoricalData().get_spot_klines(kline_size=900, lookback_days=2,
            end_date_limit=market['datetime'][-1].strftime('%Y-%m-%dT%H:%M:%S'))
    assert_frame_equal(source, expected)
    assert_frame_equal(market, original)


@pytest.mark.parametrize('value', [True, 0, -1, float('nan'), float('inf')])
def test_invalid_lookback_refuses_before_acquisition(market, value):
    with pytest.raises(ValueError, match='lookback_days'):
        lookback_window(market, value)
    with patch('limen.data.historical_data._read_any_file') as transport:
        with pytest.raises(ValueError, match='lookback_days'):
            HistoricalData().get_spot_klines(lookback_days=value, end_date_limit='2025-01-03')
        transport.assert_not_called()


def test_lookback_requires_explicit_nonconflicting_source_end():
    for options in ({}, {'end_date_limit': '2025-01-03', 'start_date_limit': '2025-01-01'}):
        with patch('limen.data.historical_data._read_any_file') as transport:
            with pytest.raises(ValueError, match='explicit end_date_limit'):
                HistoricalData().get_spot_klines(lookback_days=2, **options)
            transport.assert_not_called()


@pytest.mark.parametrize('method,options', [('get_spot_klines', {'kline_size': 900}), ('get_spot_dollar_klines', {})])
def test_fractional_lookback_does_not_round_the_start_down(market, method, options):
    with patch('limen.data.historical_data._read_any_file', return_value=market):
        result = getattr(HistoricalData(), method)(lookback_days=.020826, end_date_limit='2025-01-03T00:30:00', **options)
    expected = market.filter(pl.col('datetime').is_between(
        pl.datetime(2025, 1, 3, 0, 0, 0, 633600, time_zone='UTC'),
        pl.datetime(2025, 1, 3, 0, 30, time_zone='UTC')))
    assert result.height == expected.height == 2
    assert result['datetime'].equals(expected['datetime'])


def test_recorded_lookback_precision_control_preserves_nanosecond_endpoint(market):
    clock = pl.col('datetime').cast(pl.Datetime('ns', 'UTC'))
    precision_control = market.head(193).with_columns(pl.when(pl.int_range(pl.len()) == 192)
        .then(clock + pl.duration(nanoseconds=1)).otherwise(clock).alias('datetime'))
    assert_frame_equal(lookback_window(precision_control, 2), precision_control.tail(192))


def test_regime_sampling_retains_one_original_contiguous_block(market):
    states = classify_market_regime(market.with_columns(pl.lit('input').alias('series_id')),
                                   config=RegimeConfig(version='label3-test'))['market_regime'].to_list()
    blocks = []
    start = 0
    for end in range(1, len(states) + 1):
        if end == len(states) or states[end] != states[start]:
            if states[start] != 'INITIALIZING':
                blocks.append((end - start, states[start]))
            start = end
    length, regime = max(blocks)
    assert length >= 10
    chosen = sample_regime_blocks(market, regime, 10, seed=42, regime_version='label3-test')
    assert_frame_equal(chosen, sample_regime_blocks(market, regime, 10, seed=42, regime_version='label3-test'))
    selected = set(chosen['datetime'].dt.epoch('ns').to_list())
    positions = [index for index, clock in enumerate(market['datetime'].dt.epoch('ns')) if clock in selected]
    assert positions == list(range(positions[0], positions[-1] + 1))
    assert_frame_equal(chosen, market.slice(positions[0], len(positions)))
    assert set(states[positions[0]:positions[-1] + 1]) == {regime}
    with pytest.raises(ValueError, match='No contiguous'):
        sample_regime_blocks(market, regime, market.height + 1, seed=42, regime_version='label3-test')


@pytest.mark.parametrize('options', [
    {'target_regime': 'INITIALIZING'}, {'target_regime': 'unknown'},
    {'min_block_size': True}, {'min_block_size': 0}, {'seed': True}, {'seed': -1},
])
def test_invalid_regime_sampling_refuses(market, options):
    settings = {'target_regime': 'TREND_UP', 'min_block_size': 10, 'seed': 42, 'regime_version': 'label3-test'}
    settings.update(options)
    with pytest.raises(ValueError):
        sample_regime_blocks(market, **settings)


@pytest.mark.parametrize('selector', ['lookback_window', 'sample_regime_blocks'])
def test_native_yaml_selectors_complete_logged_sweeps(market, tmp_path, selector):
    from limen.experiment import UniversalExperimentLoop
    from limen.yaml import CompiledSFD, build_search_strategy, validate
    from tests.test_factorized_preparation import _config
    reference = _config()
    manifest = reference['sfd']['manifest']
    manifest.pop('split_dates')
    manifest['split_walk_forward'] = {'n_folds': 2, 'test_bars': 10, 'purge_bars': 1,
                                      'embargo_bars': 1, 'anchored': True}
    manifest['reference_architecture'] = 'limen.sfd.reference_architecture.dlinear_regressor'
    manifest['target'] = {'name': 'up_next', 'class': 'limen.targets.NextReturnTarget'}
    params = {'lookback_days': '{lookback_days}'} if selector == 'lookback_window' else {
        'target_regime': 'RANGE_LOW_VOL', 'min_block_size': 80, 'seed': '{seed}', 'regime_version': 'label3-test'}
    manifest['pre_split_data_selector'] = {'func': 'limen.data.utils.' + selector, 'params': params}
    controls = {'lookback_days': [2, 3]} if selector == 'lookback_window' else {'seed': [42, 43]}
    reference['sfd']['params'] = {'lookback': [1], 'fee': [4.], **controls}
    reference['uel']['n_permutations'] = 2
    assert validate(reference).valid
    experiment = UniversalExperimentLoop(sfd=CompiledSFD(reference), data=market,
        yaml_reference=reference, search_strategy=build_search_strategy(reference), experiment_dir=tmp_path)
    experiment.run('selector', n_permutations=2, prep_each_round=True, progress_bar=False)
    log = pl.read_csv(tmp_path / 'results.csv')
    key = next(iter(controls))
    assert log[key].to_list() == controls[key]
    assert log['strict_mode_error'].null_count() == 2


def test_yaml_anchored_lookback_and_sampler_controls(market):
    from ruamel.yaml import YAML
    days = YAML().load('value: &days 2')['value']
    assert_frame_equal(lookback_window(market, days), lookback_window(market, 2))
    false = YAML().load('value: &flag false')['value']
    with pytest.raises(ValueError, match='lookback_days'):
        lookback_window(market, false)
    with pytest.raises(ValueError, match='min_block_size'):
        sample_regime_blocks(market, 'RANGE_LOW_VOL', false, seed=42, regime_version='label3-test')
