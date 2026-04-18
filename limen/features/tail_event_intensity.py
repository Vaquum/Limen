import polars as pl


def tail_event_intensity(
    data: pl.DataFrame,
    window: int = 20,
    z_threshold: float = 2.0,
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute the trailing share of return shocks that exceed a volatility-scaled threshold.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        window (int): Number of periods used for both the volatility baseline and event share
        z_threshold (float): Multiple of trailing return volatility required to count as a tail event
        close_col (str): Column name used for close-to-close returns

    Returns:
        pl.DataFrame: The input data with a new column 'tail_event_intensity'
    '''

    return (
        data
        .with_columns(pl.col(close_col).pct_change().alias('_tail_returns'))
        .with_columns(pl.col('_tail_returns').rolling_std(window_size=window).shift(1).alias('_tail_baseline_std'))
        .with_columns(
            (
                pl.when(pl.col('_tail_baseline_std') > 0.0)
                .then((pl.col('_tail_returns').abs() > (z_threshold * pl.col('_tail_baseline_std'))).cast(pl.Float64))
                .otherwise(None)
                .rolling_mean(window_size=window)
            ).alias('tail_event_intensity')
        )
        .drop(['_tail_returns', '_tail_baseline_std'])
    )
