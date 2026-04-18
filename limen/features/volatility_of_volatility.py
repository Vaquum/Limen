import polars as pl


def volatility_of_volatility(
    data: pl.DataFrame,
    volatility_window: int = 12,
    window: int = 48,
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute how unstable recent return volatility is over a second rolling horizon.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        volatility_window (int): Number of periods used for the first-stage return volatility
        window (int): Number of periods used for the second-stage volatility-of-volatility estimate
        close_col (str): Column name used for close-to-close returns

    Returns:
        pl.DataFrame: The input data with a new column 'volatility_of_volatility'
    '''

    return (
        data
        .with_columns(pl.col(close_col).pct_change().alias('_vol_of_vol_returns'))
        .with_columns(
            pl.col('_vol_of_vol_returns').rolling_std(window_size=volatility_window).alias('_vol_of_vol_base')
        )
        .with_columns(
            pl.col('_vol_of_vol_base').rolling_std(window_size=window).alias('volatility_of_volatility')
        )
        .drop(['_vol_of_vol_returns', '_vol_of_vol_base'])
    )
