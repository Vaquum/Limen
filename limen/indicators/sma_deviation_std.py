import polars as pl


def sma_deviation_std(data: pl.DataFrame,
                      period: int = 30,
                      window: int = 30) -> pl.DataFrame:

    '''
    Compute rolling standard deviation of the detrended price relative to SMA(period).

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Period for the SMA
        window (int): Rolling window for std of (close - SMA(period))

    Returns:
        pl.DataFrame: The input data with a new column named using the pattern 'sma{period}_dev_std'
    '''

    sma_col = f'sma_{period}'
    dev_col = f'sma{period}_dev'
    out_col = f'sma{period}_dev_std'

    return (
        data
        .with_columns([
            pl.col('close').rolling_mean(window_size=period).alias(sma_col)
        ])
        .with_columns([
            (pl.col('close') - pl.col(sma_col)).alias(dev_col)
        ])
        .with_columns([
            pl.col(dev_col).rolling_std(window_size=window).alias(out_col)
        ])
        .drop([sma_col, dev_col])
    )
