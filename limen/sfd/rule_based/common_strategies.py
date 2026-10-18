from collections.abc import Callable, Mapping
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.indicators.bollinger_bands import bollinger_bands
from limen.indicators.ema import ema_from_values
from limen.indicators.macd import macd
from limen.indicators.rsi import rsi_from_values
from limen.indicators.sma import sma_from_values


__all__ = ['STRATEGY_NAMES', 'STRATEGY_PARAMETERS', 'strategy_signals']

STRATEGY_PARAMETERS: Mapping[str, Mapping[str, int | float]] = MappingProxyType({
    'buy_and_hold': MappingProxyType({}),
    'sma_cross': MappingProxyType({'fast_period': 50, 'slow_period': 200}),
    'ema_cross': MappingProxyType({'fast_period': 12, 'slow_period': 26}),
    'macd': MappingProxyType({
        'fast_period': 12, 'slow_period': 26, 'signal_period': 9,
    }),
    'rsi_mean_reversion': MappingProxyType({
        'period': 14, 'entry_threshold': 30.0, 'exit_threshold': 70.0,
    }),
    'bollinger_mean_reversion': MappingProxyType({
        'period': 20, 'standard_deviations': 2.0, 'ddof': 0,
    }),
    'donchian_breakout': MappingProxyType({
        'entry_period': 20, 'exit_period': 10,
    }),
    'time_series_momentum': MappingProxyType({'period': 20}),
})
STRATEGY_NAMES = tuple(STRATEGY_PARAMETERS)


def _validate_prices(data: pl.DataFrame, strategy: str) -> None:
    required = ('close', 'high', 'low') if strategy == 'donchian_breakout' else ('close',)
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'strategy_signals missing required price columns: {missing}')
    for column in required:
        series = data[column]
        if not series.dtype.is_numeric():
            raise TypeError(f'strategy_signals {column} must be numeric')
        values = series.to_numpy().astype(np.float64, copy=False)
        if series.null_count() or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f'strategy_signals {column} must contain finite positive prices')
    if strategy == 'donchian_breakout' and data.select(
        ((pl.col('high') < pl.col('close')) | (pl.col('low') > pl.col('close'))).any()
    ).item():
        raise ValueError('strategy_signals requires low <= close <= high')
    if 'datetime' in data.columns:
        datetimes = data['datetime']
        if datetimes.null_count() or not datetimes.is_sorted() or datetimes.n_unique() != data.height:
            raise ValueError('strategy_signals datetime must be strictly increasing without nulls')


def _hold_until_exit(
    entries: npt.NDArray[np.bool_],
    exits: npt.NDArray[np.bool_],
) -> npt.NDArray[np.int8]:
    '''Retain the latest entry/exit decision; simultaneous exits take precedence.'''

    decisions = np.zeros(entries.size + 1, dtype=np.int8)
    decisions[1:] = entries & ~exits
    decision_rows = np.where(entries | exits, np.arange(1, entries.size + 1), 0)
    return decisions[np.maximum.accumulate(decision_rows)]


def _buy_and_hold(data: pl.DataFrame) -> npt.NDArray[np.int8]:
    return np.ones(data.height, dtype=np.int8)


def _sma_cross(data: pl.DataFrame) -> npt.NDArray[np.int8]:
    prices = data['close'].to_numpy().astype(np.float64, copy=False)
    parameters = STRATEGY_PARAMETERS['sma_cross']
    fast = sma_from_values(prices, int(parameters['fast_period']))
    slow = sma_from_values(prices, int(parameters['slow_period']))
    return (fast > slow).astype(np.int8)


def _ema_cross(data: pl.DataFrame) -> npt.NDArray[np.int8]:
    prices = data['close'].to_numpy().astype(np.float64, copy=False)
    parameters = STRATEGY_PARAMETERS['ema_cross']
    fast = ema_from_values(prices, int(parameters['fast_period']))
    slow = ema_from_values(prices, int(parameters['slow_period']))
    return (fast > slow).astype(np.int8)


def _macd(data: pl.DataFrame) -> npt.NDArray[np.int8]:
    parameters = STRATEGY_PARAMETERS['macd']
    indicators = macd(
        data.select('close'),
        fast_period=int(parameters['fast_period']),
        slow_period=int(parameters['slow_period']),
        signal_period=int(parameters['signal_period']),
    )
    line = indicators['macd'].to_numpy()
    signal = indicators['macd_signal'].to_numpy()
    return (line > signal).astype(np.int8)


def _rsi_mean_reversion(data: pl.DataFrame) -> npt.NDArray[np.int8]:
    parameters = STRATEGY_PARAMETERS['rsi_mean_reversion']
    prices = data['close'].to_numpy().astype(np.float64, copy=False)
    rsi = rsi_from_values(prices, int(parameters['period']))
    return _hold_until_exit(
        rsi <= parameters['entry_threshold'], rsi >= parameters['exit_threshold'],
    )


def _bollinger_mean_reversion(data: pl.DataFrame) -> npt.NDArray[np.int8]:
    parameters = STRATEGY_PARAMETERS['bollinger_mean_reversion']
    bands = bollinger_bands(
        data.select('close'), window=int(parameters['period']),
        num_std=float(parameters['standard_deviations']),
    )
    prices = bands['close'].to_numpy()
    return _hold_until_exit(
        prices <= bands['bb_lower'].to_numpy(),
        prices >= bands['bb_middle'].to_numpy(),
    )


def _donchian_breakout(data: pl.DataFrame) -> npt.NDArray[np.int8]:
    parameters = STRATEGY_PARAMETERS['donchian_breakout']
    prior_high = data['high'].shift(1).rolling_max(int(parameters['entry_period']))
    prior_low = data['low'].shift(1).rolling_min(int(parameters['exit_period']))
    prices = data['close'].to_numpy()
    return _hold_until_exit(prices > prior_high.to_numpy(), prices < prior_low.to_numpy())


def _time_series_momentum(data: pl.DataFrame) -> npt.NDArray[np.int8]:
    period = int(STRATEGY_PARAMETERS['time_series_momentum']['period'])
    prices = data['close'].to_numpy()
    prior_prices = data['close'].shift(period).to_numpy()
    return (prices > prior_prices).astype(np.int8)


_SIGNAL_FUNCTIONS: Mapping[str, Callable[[pl.DataFrame], npt.NDArray[np.int8]]] = {
    'buy_and_hold': _buy_and_hold,
    'sma_cross': _sma_cross,
    'ema_cross': _ema_cross,
    'macd': _macd,
    'rsi_mean_reversion': _rsi_mean_reversion,
    'bollinger_mean_reversion': _bollinger_mean_reversion,
    'donchian_breakout': _donchian_breakout,
    'time_series_momentum': _time_series_momentum,
}


def strategy_signals(data: pl.DataFrame, strategy: str) -> npt.NDArray[np.int8]:
    '''Compute one fixed common-strategy baseline as completed-bar long/flat intent.

    Rows retain input order. Prices must be finite and positive; when supplied,
    datetime must be strictly increasing. Donchian also requires high and low.
    Each call starts flat with no indicator history. Warm-up rows remain flat,
    and stateful strategies hold until their own exit rule occurs. The returned
    array has one 0/1 value per input row, including warm-up and the final row.
    Execution lag, fills, fees and final liquidation belong to the backtester.
    See docs/Common-Strategies.md for the fixed parameters and exact rules.
    '''

    if strategy not in _SIGNAL_FUNCTIONS:
        raise ValueError(f'Unknown strategy {strategy!r}; choose from {STRATEGY_NAMES}')
    _validate_prices(data, strategy)
    return _SIGNAL_FUNCTIONS[strategy](data)
