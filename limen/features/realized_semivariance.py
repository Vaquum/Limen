import polars as pl


def realized_semivariance(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute rolling upside and downside semivariance from close-to-close returns.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        window (int): Number of periods used for the rolling semivariance estimates
        close_col (str): Column name used for close-to-close returns

    Returns:
        pl.DataFrame: The input data with new columns 'upside_semivariance' and 'downside_semivariance'
    '''

    returns = pl.col(close_col).pct_change()

    return (
        data
        .with_columns(returns.alias('_semivariance_returns'))
        .with_columns([
            (
                pl.when(pl.col('_semivariance_returns') > 0.0)
                .then(pl.col('_semivariance_returns') ** 2)
                .otherwise(0.0)
                .rolling_mean(window_size=window)
            ).alias('upside_semivariance'),
            (
                pl.when(pl.col('_semivariance_returns') < 0.0)
                .then(pl.col('_semivariance_returns') ** 2)
                .otherwise(0.0)
                .rolling_mean(window_size=window)
            ).alias('downside_semivariance'),
        ])
        .drop('_semivariance_returns')
    )
