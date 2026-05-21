from collections.abc import Sequence

import polars as pl


def is_funding_hour(
    data: pl.DataFrame,
    hours: Sequence[int] = (0, 8, 16),
    datetime_col: str = 'datetime',
    output_col: str = 'is_funding_hour',
) -> pl.DataFrame:

    '''
    Mark rows whose hour matches a configured funding cadence.

    Args:
        data (pl.DataFrame): Dataset with a datetime-like column
        hours (Sequence[int]): Funding hours to mark
        datetime_col (str): Column name for timestamps
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with an Int8 funding-hour flag appended
    '''

    return data.with_columns(
        pl.col(datetime_col).dt.hour().is_in(list(hours)).cast(pl.Int8).alias(output_col)
    )
