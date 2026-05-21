import polars as pl


def is_us_open_hour(
    data: pl.DataFrame,
    hour: int = 14,
    datetime_col: str = 'datetime',
    output_col: str = 'is_us_open_hour',
) -> pl.DataFrame:

    '''
    Mark rows whose hour matches the configured US open hour.

    Args:
        data (pl.DataFrame): Dataset with a datetime-like column
        hour (int): Hour to mark
        datetime_col (str): Column name for timestamps
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with an Int8 US-open-hour flag appended
    '''

    return data.with_columns(
        (pl.col(datetime_col).dt.hour() == hour).cast(pl.Int8).alias(output_col)
    )
