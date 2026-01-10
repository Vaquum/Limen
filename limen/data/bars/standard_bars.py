import polars as pl
from typing import Union



def _standard_bars(data: pl.DataFrame, 
                   threshold: Union[int, float],
                   column_name: str) -> pl.DataFrame:

    '''
    Compute standard bars from klines data using fixed threshold on
    column value accumulation.

    Args:
        data (pl.DataFrame): Klines data
        threshold (Union[int, float]): Threshold value for bar creation
        column_name (str): Column name to accumulate for threshold detection

    Returns:
        pl.DataFrame:  Data bars with columns 'datetime', 'open', 'high',
                      'low', 'close', 'volume', 'no_of_trades', 'liquidity_sum',
                      'maker_ratio', 'maker_volume', 'maker_liquidity', 'mean'
                      'bar_count', 'base_interval'

    '''

    base_interval = (
          (data['datetime'][1] - data['datetime'][0]).total_seconds()
          if len(data) >= 2
          else 0
      )
        
    df_with_cumsum = data.with_row_index().with_columns([
        pl.col(column_name).alias('threshold_col')
    ])
    
    def create_bar_groups(df: pl.DataFrame) -> pl.DataFrame:
        cumsum = 0.0
        bar_group = 0
        groups = []
        
        for row in df.iter_rows(named=True):
            cumsum += row['threshold_col']
            groups.append(bar_group)
            
            if cumsum >= threshold:
                bar_group += 1
                cumsum = 0.0
        
        return df.with_columns(pl.Series('bar_group', groups))
    
    df_with_groups = create_bar_groups(df_with_cumsum)
    
    max_bar_group = df_with_groups['bar_group'].max()
    last_bar_sum = (
        df_with_groups
        .filter(pl.col('bar_group') == max_bar_group)[column_name]
        .sum())
    
    if last_bar_sum < threshold:
        df_with_groups = (
            df_with_groups
            .filter(pl.col('bar_group') < max_bar_group)
        )
    
    result = (
        df_with_groups
        .drop(['index', 'threshold_col'])
        .group_by('bar_group', maintain_order=True)
        .agg([
            pl.col('datetime').first().alias('datetime'),
            pl.col('open').first().alias('open'),
            pl.col('high').max().alias('high'),
            pl.col('low').min().alias('low'),
            pl.col('close').last().alias('close'),
            pl.col('volume').sum().alias('volume'),
            pl.col('no_of_trades').sum().alias('no_of_trades'),
            pl.col('liquidity_sum').sum().alias('liquidity_sum'),
            pl.col('maker_volume').sum().alias('maker_volume'),
            pl.col('maker_liquidity').sum().alias('maker_liquidity'),
            ((pl.col('maker_ratio') * pl.col('no_of_trades')).sum()
             / pl.col('no_of_trades').sum()).alias('maker_ratio'),
            ((pl.col('mean') * pl.col('no_of_trades')).sum()
             / pl.col('no_of_trades').sum()).alias('mean'),
            pl.len().alias('bar_count'),
            pl.lit(base_interval).alias('base_interval')
        ])
        .sort('bar_group')
        .select([
            'datetime', 'open', 'high', 'low', 'close', 'volume', 
            'no_of_trades', 'liquidity_sum', 'maker_ratio', 'maker_volume', 
            'maker_liquidity', 'mean', 'bar_count', 'base_interval'
        ])
    )
    
    return result



def volume_bars(data: pl.DataFrame, volume_threshold: float) -> pl.DataFrame:

    '''
    Compute volume bars with fixed volume size sampling.

    Args:
        data (pl.DataFrame): Klines data
        volume_threshold (float): Volume threshold per bar

    Returns:
        pl.DataFrame: Standard volume data bars
    '''

    return _standard_bars(data, volume_threshold, 'volume')


def trade_bars(data: pl.DataFrame, trade_threshold: int) -> pl.DataFrame:

    '''
    Compute trade bars with fixed trade count sampling.

    Args:
        data (pl.DataFrame): Klines data
        trade_threshold (int): Number of trades per bar

    Returns:
        pl.DataFrame: Standard trade count data bars
    '''

    return _standard_bars(data, trade_threshold, 'no_of_trades')


def liquidity_bars(data: pl.DataFrame,
                   liquidity_threshold: float) -> pl.DataFrame:

    '''
    Compute liquidity bars with fixed liquidity sampling.

    Args:
        data (pl.DataFrame): Klines data
        liquidity_threshold (float): Liquidity threshold per bar

    Returns:
        pl.DataFrame: Standard liquidity data bars
    '''

    return _standard_bars(data, liquidity_threshold, 'liquidity_sum')