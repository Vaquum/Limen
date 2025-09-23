import polars as pl

from loop.data.bars import volume_bars, trade_bars, liquidity_bars


def adaptive_bar_formation(data: pl.DataFrame, **kwargs) -> pl.DataFrame:
    
    '''
    Adaptive bar formation based on bar_type parameter.

    Args:
        data (pl.DataFrame): Input klines data
        **kwargs: Bar formation parameters including bar_type

    Returns:
        pl.DataFrame: Processed bars based on bar_type
    '''

    bar_type = kwargs.get('bar_type', 'base')

    if bar_type == 'base':
        return data
    elif bar_type == 'trade':
        if 'trade_threshold' not in kwargs:
            raise ValueError('trade_threshold parameter is required for trade bars')
        return trade_bars(data, kwargs['trade_threshold'])
    elif bar_type == 'volume':
        if 'volume_threshold' not in kwargs:
            raise ValueError('volume_threshold parameter is required for volume bars')
        return volume_bars(data, kwargs['volume_threshold'])
    elif bar_type == 'liquidity':
        if 'liquidity_threshold' not in kwargs:
            raise ValueError('liquidity_threshold parameter is required for liquidity bars')
        return liquidity_bars(data, kwargs['liquidity_threshold'])
    else:
        return data