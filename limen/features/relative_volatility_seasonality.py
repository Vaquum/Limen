import polars as pl

EPSILON = 1e-10


def relative_volatility_seasonality(
    data: pl.DataFrame,
    datetime_col: str = 'datetime',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute absolute return magnitude relative to the trailing mean for the same hour of the week.

    Args:
        data (pl.DataFrame): Klines dataset with datetime and close columns
        datetime_col (str): Column name for datetimes used to derive hour-of-week buckets
        close_col (str): Column name used for close-to-close returns

    Returns:
        pl.DataFrame: The input data with a new column 'relative_volatility_seasonality'
    '''

    volatility_proxy = pl.col(close_col).pct_change().abs().alias('_seasonality_value')
    hour_of_week = (((pl.col(datetime_col).dt.weekday() - 1) * 24) + pl.col(datetime_col).dt.hour()).alias('_seasonality_hour_of_week')

    return (
        data
        .with_columns([hour_of_week, volatility_proxy])
        .with_columns([
            pl.col('_seasonality_value').cum_sum().over('_seasonality_hour_of_week').alias('_seasonality_cum_sum'),
            pl.col('_seasonality_value').cum_count().over('_seasonality_hour_of_week').alias('_seasonality_cum_count'),
        ])
        .with_columns([
            (pl.col('_seasonality_cum_sum') - pl.col('_seasonality_value')).alias('_seasonality_prev_sum'),
            (pl.col('_seasonality_cum_count') - 1).alias('_seasonality_prev_count'),
        ])
        .with_columns(
            pl.when(pl.col('_seasonality_prev_count') > 0)
            .then(
                pl.col('_seasonality_value')
                / ((pl.col('_seasonality_prev_sum') / pl.col('_seasonality_prev_count')) + EPSILON)
            )
            .otherwise(None)
            .alias('relative_volatility_seasonality')
        )
        .drop([
            '_seasonality_hour_of_week',
            '_seasonality_value',
            '_seasonality_cum_sum',
            '_seasonality_cum_count',
            '_seasonality_prev_sum',
            '_seasonality_prev_count',
        ])
    )
