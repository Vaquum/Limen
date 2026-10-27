import inspect
from pathlib import Path
from typing import get_type_hints

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen import features, indicators
from limen.features.volatility_barriers import volatility_barriers

FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
LINE_DATA = {'long_lines', 'short_lines', 'long_lines_q', 'short_lines_q'}
REQUIRED = {
    'cols': ['close'], 'col': 'close', 'column': 'close', 'strategy': 'sma_cross',
    'start': 0, 'end': 0, 'lag': 0, 'window': 10, 'max_duration_hours': 24,
    'min_height_pct': .01, 'lookback_hours': 24, 'base_stop_loss': .01,
    'stop_volatility_multiplier': 1., 'base_min_breakout': .01,
    'target_volatility_multiplier': 1., 'momentum_threshold_bps': -100.,
    'flow_z_threshold': 2., 'hold_minutes': 1,
}


def _callables():
    return [getattr(module, name) for module in (features, indicators)
            for name in module.__all__] + [volatility_barriers]


@pytest.mark.parametrize('function', _callables(), ids=lambda function: function.__name__)
def test_every_public_configuration_parameter_rejects_wrong_types(function):
    frame = pl.read_parquet(FIXTURE, n_rows=1).head(0)
    parameters = list(inspect.signature(function).parameters.values())[1:]
    arguments = {parameter.name: REQUIRED[parameter.name] for parameter in parameters
                 if parameter.default is inspect.Parameter.empty and parameter.name not in LINE_DATA}
    arguments.update({name: [] for name in LINE_DATA if name in inspect.signature(function).parameters})
    if function is features.fractional_diff:
        arguments['cols'] = ['close']
    hints = get_type_hints(function)
    for parameter in parameters:
        if parameter.name in LINE_DATA:
            continue
        values = [object()]
        if hints.get(parameter.name) is int or parameter.name in ('start', 'end'):
            values += [True, 1.5]
        elif hints.get(parameter.name) is float:
            values += [True, float('nan'), float('inf'), -float('inf')]
        for value in values:
            with pytest.raises((ValueError, TypeError), match=parameter.name):
                function(frame, **dict(arguments, **{parameter.name: value}))


def test_valid_causal_zero_controls_and_period_iterators():
    frame = pl.read_parquet(FIXTURE, n_rows=100)
    for function in (indicators.price_change_pct, indicators.window_return, features.micro_momentum):
        result = function(frame, period=0)
        assert result.height == frame.height
        assert result[result.columns[-1]].drop_nulls().eq(0).all()
    assert_frame_equal(features.momentum_periods(frame, periods=iter([0, 2])),
                       features.momentum_periods(frame, periods=[0, 2]))
    assert_frame_equal(features.sma_ratios(frame, periods=iter([2, 5])),
                       features.sma_ratios(frame, periods=[2, 5]))


@pytest.mark.parametrize('function,arguments,parameter', [
    (features.price_vs_band_regime, {'band': 'unknown'}, 'band'),
    (features.breakout_percentile_regime, {'p_lo': .8, 'p_hi': .2}, 'p_lo'),
    (features.breakout_percentile_regime, {'p_hi': 1.1}, 'p_hi'),
    (features.dynamic_target, {'base_min_breakout': .01, 'target_volatility_multiplier': 1.,
                               'clip_lower_mult': 2., 'clip_upper_mult': 1.}, 'clip_lower_mult'),
    (features.momentum_periods, {'periods': [2, -1]}, 'periods'),
    (features.momentum_periods, {'periods': [2, True]}, 'periods'),
    (features.is_funding_hour, {'hours': [0, 24]}, 'hours'),
    (features.time_to_funding, {'offset_hour': 24}, 'offset_hour'),
    (features.entry_score_microstructure, {'entry_volume_spike_normalizer': 0.}, 'entry_volume_spike_normalizer'),
    (features.conserved_flux_renormalization, {'kline_interval': '0h'}, 'kline_interval'),
    (features.fractional_diff, {'cols': ['close', 2]}, 'cols'),
])
def test_invalid_domains_fail_before_data_computation(function, arguments, parameter):
    frame = pl.read_parquet(FIXTURE, n_rows=1).head(0)
    with pytest.raises(ValueError, match=parameter):
        function(frame, **arguments)
