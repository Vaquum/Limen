import polars as pl

from limen.data.bars import volume_bars, trade_bars, liquidity_bars


def compute_data_bars(data: pl.DataFrame, **params) -> pl.DataFrame:

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

    elif bar_type == 'trade':
        if 'trade_threshold' not in params:
            raise ValueError('trade_threshold parameter is required for trade bars')
        return trade_bars(data, params['trade_threshold'])

    elif bar_type == 'volume':
        if 'volume_threshold' not in params:
            raise ValueError('volume_threshold parameter is required for volume bars')
        return volume_bars(data, params['volume_threshold'])

    elif bar_type == 'liquidity':
        if 'liquidity_threshold' not in params:
            raise ValueError('liquidity_threshold parameter is required for liquidity bars')
        return liquidity_bars(data, params['liquidity_threshold'])

    else:
        return data
