import polars as pl

EPSILON = 1e-10


def relative_range_seasonality(
    data: pl.DataFrame,
    datetime_col: str = 'datetime',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute normalized bar range relative to the trailing mean for the same hour of the week.

    Args:
        data (pl.DataFrame): Klines dataset with datetime, high, low, and close columns
        datetime_col (str): Column name for datetimes used to derive hour-of-week buckets
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'relative_range_seasonality'
    '''

    range_pct = ((pl.col(high_col) - pl.col(low_col)) / (pl.col(close_col) + EPSILON)).alias('_seasonality_value')
    hour_of_week = (((pl.col(datetime_col).dt.weekday() - 1) * 24) + pl.col(datetime_col).dt.hour()).alias('_seasonality_hour_of_week')

    return (
        data
        .with_columns([hour_of_week, range_pct])
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
            .alias('relative_range_seasonality')
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
