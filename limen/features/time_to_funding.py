import polars as pl


HOURS_PER_DAY = 24


def time_to_funding(
    data: pl.DataFrame,
    interval_hours: int = 8,
    offset_hour: int = 0,
    datetime_col: str = 'datetime',
    output_col: str = 'hours_to_funding',
) -> pl.DataFrame:

    '''
    Compute the number of hours until the next funding settlement.

    Funding settles every `interval_hours` from `offset_hour` UTC (the default
    cadence is 0, 8, and 16 hours). The output is the continuous distance in hours
    from each timestamp to the next settlement, and is zero at a settlement bar.
    The cadence is anchored to UTC midnight, so `interval_hours` must divide 24.
    Timezone-aware timestamps are converted to UTC; naive timestamps are assumed
    to already be in UTC.

    Args:
        data (pl.DataFrame): Dataset with a datetime column
        interval_hours (int): Spacing between funding settlements in hours; must divide 24
        offset_hour (int): Hour of the first daily settlement in UTC
        datetime_col (str): Column name for timestamps
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with a new 'hours_to_funding' column
    '''

    if interval_hours <= 0:
        raise ValueError('time_to_funding interval_hours must be positive')
    if HOURS_PER_DAY % interval_hours != 0:
        raise ValueError('time_to_funding interval_hours must divide 24')

    timestamp = pl.col(datetime_col)
    if data.schema[datetime_col].time_zone is not None:
        timestamp = timestamp.dt.convert_time_zone('UTC')

    hour_of_day = (
        timestamp.dt.hour().cast(pl.Float64)
        + timestamp.dt.minute().cast(pl.Float64) / 60.0
        + timestamp.dt.second().cast(pl.Float64) / 3600.0
    )
    hours_since = (hour_of_day - offset_hour) % interval_hours
    hours_to_next = (interval_hours - hours_since) % interval_hours

    return data.with_columns(hours_to_next.alias(output_col))
