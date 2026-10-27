from typing import Any
import polars as pl

from limen.data.bars import volume_bars, trade_bars, liquidity_bars, imbalance_bars, run_bars


def compute_data_bars(data: pl.DataFrame, **params: Any) -> pl.DataFrame:

    '''
    Compute adaptive bar formation based on bar_type parameter.

    Args:
        data (pl.DataFrame): Klines dataset with 'datetime', 'open', 'high', 'low', 'close', 'volume', 'no_of_trades', 'liquidity_sum', 'maker_ratio', 'maker_volume', 'maker_liquidity' columns
        **params: Bar formation parameters including bar_type and threshold parameters

    Returns:
        pl.DataFrame: Data bars with columns 'datetime', 'open', 'high', 'low', 'close', 'volume', 'no_of_trades', 'liquidity_sum', 'maker_ratio', 'maker_volume', 'maker_liquidity', 'mean', 'bar_count', 'base_interval'
    '''

    bar_type = params.get('bar_type', 'base')

    if bar_type == 'base':
        return data

    if bar_type in ('imbalance', 'run'):
        function = imbalance_bars if bar_type == 'imbalance' else run_bars
        return function(data, expected_num_ticks=params.get('expected_num_ticks', 100), ema_span=params.get('ema_span', 20))

    standard = {'trade': (trade_bars, 'trade_threshold'), 'volume': (volume_bars, 'volume_threshold'),
                'liquidity': (liquidity_bars, 'liquidity_threshold')}
    if bar_type in standard:
        standard_function, threshold = standard[bar_type]
        if threshold not in params:
            raise ValueError(f'compute_data_bars {threshold} parameter is required for {bar_type} bars')
        return standard_function(data, params[threshold])

    raise ValueError(
        "compute_data_bars bar_type must be one of: base, trade, volume, liquidity, imbalance, run"
    )
