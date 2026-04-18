import polars as pl


def realized_kurtosis(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute rolling standardized kurtosis of close-to-close returns.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        window (int): Number of periods used for the rolling kurtosis estimate
        close_col (str): Column name used for close-to-close returns

    Returns:
        pl.DataFrame: The input data with a new column 'realized_kurtosis'
    '''

    return (
        data
        .with_columns(pl.col(close_col).pct_change().alias('_kurtosis_returns'))
        .with_columns([
            pl.col('_kurtosis_returns').rolling_mean(window_size=window).alias('_kurtosis_mean'),
            (pl.col('_kurtosis_returns') ** 2).rolling_mean(window_size=window).alias('_kurtosis_second_moment'),
            (pl.col('_kurtosis_returns') ** 3).rolling_mean(window_size=window).alias('_kurtosis_third_moment'),
            (pl.col('_kurtosis_returns') ** 4).rolling_mean(window_size=window).alias('_kurtosis_fourth_moment'),
        ])
        .with_columns(
            (pl.col('_kurtosis_second_moment') - (pl.col('_kurtosis_mean') ** 2)).alias('_kurtosis_variance')
        )
        .with_columns(
            (
                pl.col('_kurtosis_fourth_moment')
                - (4.0 * pl.col('_kurtosis_mean') * pl.col('_kurtosis_third_moment'))
                + (6.0 * (pl.col('_kurtosis_mean') ** 2) * pl.col('_kurtosis_second_moment'))
                - (3.0 * (pl.col('_kurtosis_mean') ** 4))
            ).alias('_kurtosis_fourth_central')
        )
        .with_columns(
            pl.when(pl.col('_kurtosis_variance') > 0.0)
            .then(pl.col('_kurtosis_fourth_central') / (pl.col('_kurtosis_variance') ** 2))
            .otherwise(None)
            .alias('realized_kurtosis')
        )
        .drop([
            '_kurtosis_returns',
            '_kurtosis_mean',
            '_kurtosis_second_moment',
            '_kurtosis_third_moment',
            '_kurtosis_fourth_moment',
            '_kurtosis_variance',
            '_kurtosis_fourth_central',
        ])
    )
