from pathlib import Path
from zipfile import ZipFile

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from limen.data.bars import imbalance_bars, run_bars
from limen.data.utils import compute_data_bars
from limen.experiment._prepare_trade_context import admit_formation, sensor_decisions
from limen.experiment import UniversalExperimentLoop
from limen.yaml import CompiledSFD, build_search_strategy, validate
from tests.test_factorized_preparation import _config


@pytest.fixture(scope='module')
def trades():
    source = Path(__file__).parent / 'fixtures/binance_trades_btcusdt_2025-05-23_90s.zip'
    with ZipFile(source) as archive, archive.open(archive.namelist()[0]) as stream:
        recorded = pl.read_csv(stream, has_header=False)
    points = recorded.select(
        pl.from_epoch('column_5', time_unit='us').dt.replace_time_zone('UTC').alias('datetime'),
        *[pl.col('column_2').alias(name) for name in ('open', 'high', 'low', 'close', 'mean')],
        pl.col('column_3').alias('volume'), pl.lit(1).alias('no_of_trades'),
        pl.col('column_4').alias('liquidity_sum'), pl.col('column_6').cast(pl.Float64).alias('maker_ratio'),
        (pl.col('column_3') * pl.col('column_6')).alias('maker_volume'),
        (pl.col('column_4') * pl.col('column_6')).alias('maker_liquidity'))
    points = points.unique(subset='datetime', keep='first', maintain_order=True)
    return points.with_columns([pl.col('datetime').dt.epoch('ns').alias(name)
        for name in ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns')])


@pytest.mark.parametrize('function,kind', [(imbalance_bars, 'imbalance'), (run_bars, 'run')])
def test_recorded_adaptive_bars_conserve_membership_and_clocks(trades, function, kind):
    original = trades.clone()
    bars = function(trades, expected_num_ticks=20, ema_span=10)
    assert bars.height > 2
    assert_frame_equal(bars, compute_data_bars(trades, bar_type=kind, expected_num_ticks=20, ema_span=10))
    admit_formation(function)
    decisions = sensor_decisions(trades, bars)
    signed = trades.select((pl.col('close').diff().sign().replace(0, None).forward_fill().fill_null(0)
                           * pl.col('volume')).alias('signed'))['signed']
    positive, negative = signed.clip(lower_bound=0), (-signed).clip(lower_bound=0)
    mean_signed = signed.ewm_mean(span=10, adjust=False).to_numpy()
    mean_run = np.maximum(positive.ewm_mean(span=10, adjust=False).to_numpy(), negative.ewm_mean(span=10, adjust=False).to_numpy())
    expected_length = 20.
    offset = 0
    for index, bar in enumerate(bars.iter_rows(named=True)):
        source = trades.slice(offset, bar['bar_count'])
        assert source.height == bar['bar_count']
        assert bar['open'] == source['open'][0] and bar['close'] == source['close'][-1]
        assert bar['high'] == source['high'].max() and bar['low'] == source['low'].min()
        for column in ('volume', 'liquidity_sum', 'maker_volume', 'maker_liquidity'):
            np.testing.assert_allclose(bar[column], source[column].sum(), rtol=1e-12)
        assert bar['datetime'] == source['datetime'][0]
        assert decisions['__trade_available_at_ns__'][index] == source['available_at_ns'][-1]
        end = offset + bar['bar_count']
        evidence = (np.maximum(positive.slice(offset, bar['bar_count']).cum_sum(), negative.slice(offset, bar['bar_count']).cum_sum())
                    if kind == 'run' else np.abs(signed.slice(offset, bar['bar_count']).cum_sum()))
        mean = mean_run if kind == 'run' else np.abs(mean_signed)
        threshold = expected_length * (mean[offset:end] if offset == 0 else mean[offset - 1])
        evidence = np.asarray(evidence)
        crosses = (np.arange(offset, end) + 1 >= 20) & (evidence > 0) & (evidence >= threshold - 1e-12)
        assert np.flatnonzero(crosses).tolist() == [bar['bar_count'] - 1]
        expected_length += (2 / 11) * (bar['bar_count'] - expected_length)
        offset += bar['bar_count']
    np.testing.assert_allclose(bars['volume'].sum(), trades['volume'].head(offset).sum(), rtol=1e-12)
    prefix = trades.head(trades.height // 2)
    earlier = function(prefix, expected_num_ticks=20, ema_span=10)
    assert earlier.height < bars.height
    assert_frame_equal(earlier, bars.head(earlier.height))
    assert_frame_equal(trades, original)


@pytest.mark.parametrize('function', [imbalance_bars, run_bars])
def test_adaptive_bars_refuse_candles_and_bad_controls(trades, function):
    candles = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet', n_rows=32)
    with pytest.raises(ValueError, match='individual trade points'):
        function(candles)
    for settings in ({'expected_num_ticks': True}, {'expected_num_ticks': 0}, {'ema_span': 0}):
        with pytest.raises(ValueError, match='positive integer'):
            function(trades, **settings)
    assert function(trades.head(1), expected_num_ticks=20).is_empty()


def test_native_yaml_sweep_runs_both_adaptive_bar_types(trades, tmp_path):
    source = tmp_path / 'recorded-trade-points.parquet'
    trades.write_parquet(source)
    reference = _config()
    manifest = reference['sfd']['manifest']
    manifest.pop('split_dates')
    manifest['split_walk_forward'] = {'n_folds': 2, 'test_bars': 225,
        'purge_bars': 1, 'embargo_bars': 1, 'anchored': True}
    manifest['data_source']['params'] = {'file_path_or_url': str(source)}
    manifest['bar_formation'] = {'func': 'limen.data.utils.compute_data_bars', 'params': {
        'bar_type': '{bar_type}', 'expected_num_ticks': '{expected_num_ticks}', 'ema_span': '{ema_span}'}}
    manifest['reference_architecture'] = 'limen.sfd.reference_architecture.dlinear_regressor'
    manifest['target'] = {'name': 'up_next', 'class': 'limen.targets.NextReturnTarget'}
    manifest['backtest'] = {'product': {'kind': 'linear_perpetual', 'instrument': 'BTCUSDT', 'base_currency': 'BTC',
        'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0.}, 'fee_bps': '{fee}', 'slip_bps': 0., 'notional_rate': .5}
    reference['sfd']['params'] = {'bar_type': ['imbalance', 'run'], 'expected_num_ticks': [5], 'ema_span': [10],
                                 'lookback': [2], 'fee': [4.]}
    reference['uel']['n_permutations'] = 2
    assert validate(reference).valid
    experiment = UniversalExperimentLoop(sfd=CompiledSFD(reference), data=trades, yaml_reference=reference,
        search_strategy=build_search_strategy(reference), experiment_dir=tmp_path / 'experiment')
    experiment.run('adaptive', n_permutations=2, prep_each_round=True, progress_bar=False)
    log = pl.read_csv(tmp_path / 'experiment/results.csv')
    assert log['bar_type'].to_list() == ['imbalance', 'run']
    assert log['strict_mode_error'].null_count() == 2
