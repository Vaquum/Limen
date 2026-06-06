import polars as pl


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

    Args:
        data (pl.DataFrame): Dataset with a datetime column
        interval_hours (int): Spacing between funding settlements in hours
        offset_hour (int): Hour of the first daily settlement in UTC
        datetime_col (str): Column name for timestamps
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with a new 'hours_to_funding' column
    '''

    if interval_hours <= 0:
        raise ValueError('time_to_funding interval_hours must be positive')

    hour_of_day = (
        pl.col(datetime_col).dt.hour().cast(pl.Float64)
        + pl.col(datetime_col).dt.minute().cast(pl.Float64) / 60.0
        + pl.col(datetime_col).dt.second().cast(pl.Float64) / 3600.0
    )
    hours_since = (hour_of_day - offset_hour) % interval_hours
    hours_to_next = (interval_hours - hours_since) % interval_hours

    return data.with_columns(hours_to_next.alias(output_col))
