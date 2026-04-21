import polars as pl

EPSILON = 1e-10


def relative_volume_seasonality(
    data: pl.DataFrame,
    datetime_col: str = 'datetime',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Compute current volume relative to the trailing mean for the same hour of the week.

    Args:
        data (pl.DataFrame): Klines dataset with datetime and volume columns
        datetime_col (str): Column name for datetimes used to derive hour-of-week buckets
        volume_col (str): Column name for traded volume

    Returns:
        pl.DataFrame: The input data with a new column 'relative_volume_seasonality'
    '''

    hour_of_week = (((pl.col(datetime_col).dt.weekday() - 1) * 24) + pl.col(datetime_col).dt.hour()).alias('_seasonality_hour_of_week')

    return (
        data
        .with_columns(hour_of_week)
        .with_columns([
            pl.col(volume_col).cum_sum().over('_seasonality_hour_of_week').alias('_seasonality_cum_sum'),
            pl.col(volume_col).cum_count().over('_seasonality_hour_of_week').alias('_seasonality_cum_count'),
        ])
        .with_columns([
            (pl.col('_seasonality_cum_sum') - pl.col(volume_col)).alias('_seasonality_prev_sum'),
            (pl.col('_seasonality_cum_count') - 1).alias('_seasonality_prev_count'),
        ])
        .with_columns(
            pl.when(pl.col('_seasonality_prev_count') > 0)
            .then(
                pl.col(volume_col)
                / ((pl.col('_seasonality_prev_sum') / pl.col('_seasonality_prev_count')) + EPSILON)
            )
            .otherwise(None)
            .alias('relative_volume_seasonality')
        )
        .drop([
            '_seasonality_hour_of_week',
            '_seasonality_cum_sum',
            '_seasonality_cum_count',
            '_seasonality_prev_sum',
            '_seasonality_prev_count',
        ])
    )
