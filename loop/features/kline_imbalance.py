import polars as pl


def kline_imbalance(data: pl.DataFrame, window: int = 14) -> pl.DataFrame:

    '''
    Compute rolling buyer/seller imbalance over klines instead of raw trades.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'maker_ratio' and 'no_of_trades' columns
        window (int): Number of periods for rolling window calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'imbalance'
    '''

    return (
        data
        .sort('datetime')
        .with_columns([
            ((1 - 2 * pl.col('maker_ratio')) * pl.col('no_of_trades'))
                .alias('imbalance_raw')
        ])
        .with_columns([
            pl.col('imbalance_raw')
              .rolling_sum(window_size=window, min_samples=1)
              .alias('imbalance')
        ])
        .drop('imbalance_raw')
    )