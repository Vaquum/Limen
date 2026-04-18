import polars as pl


def realized_skewness(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute rolling standardized skewness of close-to-close returns.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        window (int): Number of periods used for the rolling skewness estimate
        close_col (str): Column name used for close-to-close returns

    Returns:
        pl.DataFrame: The input data with a new column 'realized_skewness'
    '''

    return (
        data
        .with_columns(pl.col(close_col).pct_change().alias('_skew_returns'))
        .with_columns([
            pl.col('_skew_returns').rolling_mean(window_size=window).alias('_skew_mean'),
            (pl.col('_skew_returns') ** 2).rolling_mean(window_size=window).alias('_skew_second_moment'),
            (pl.col('_skew_returns') ** 3).rolling_mean(window_size=window).alias('_skew_third_moment'),
        ])
        .with_columns(
            (pl.col('_skew_second_moment') - (pl.col('_skew_mean') ** 2)).alias('_skew_variance')
        )
        .with_columns(
            (
                pl.col('_skew_third_moment')
                - (3.0 * pl.col('_skew_mean') * pl.col('_skew_second_moment'))
                + (2.0 * (pl.col('_skew_mean') ** 3))
            ).alias('_skew_third_central')
        )
        .with_columns(
            pl.when(pl.col('_skew_variance') > 0.0)
            .then(pl.col('_skew_third_central') / (pl.col('_skew_variance').sqrt() ** 3))
            .otherwise(None)
            .alias('realized_skewness')
        )
        .drop([
            '_skew_returns',
            '_skew_mean',
            '_skew_second_moment',
            '_skew_third_moment',
            '_skew_variance',
            '_skew_third_central',
        ])
    )
