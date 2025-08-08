import polars as pl

def vwap(data: pl.DataFrame,
         price_col: str = 'close',
         volume_col: str = 'volume') -> pl.DataFrame:
    
    '''
    Compute Volume Weighted Average Price (VWAP) for each kline over its trading day.

    Args:
        data (pl.DataFrame): Klines dataset with price and volume columns
        price_col (str): Name of the price column
        volume_col (str): Name of the volume column

    Returns:
        pl.DataFrame: The input data with a new column 'vwap'
    '''

    return (
        data
        .sort('datetime')
        .with_columns([
            pl.col('datetime').dt.date().alias('__date')
        ])
        .with_columns([
            (pl.col(price_col) * pl.col(volume_col))
                .cum_sum()
                .over('__date')
                .alias('__cum_pv'),
            pl.col(volume_col)
                .cum_sum()
                .over('__date')
                .alias('__cum_vol')
        ])
        .with_columns([
            (pl.col('__cum_pv') / pl.col('__cum_vol'))
                .alias('vwap')
        ])
        .drop(['__date', '__cum_pv', '__cum_vol'])
    )