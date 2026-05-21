from collections.abc import Sequence

import polars as pl


def is_funding_hour(
    data: pl.DataFrame,
    hours: Sequence[int] = (0, 8, 16),
    datetime_col: str = 'datetime',
    output_col: str = 'is_funding_hour',
) -> pl.DataFrame:

    '''Mark rows whose hour matches a configured funding cadence.'''

    return data.with_columns(
        pl.col(datetime_col).dt.hour().is_in(list(hours)).cast(pl.Int8).alias(output_col)
    )


def is_us_open_hour(
    data: pl.DataFrame,
    hour: int = 14,
    datetime_col: str = 'datetime',
    output_col: str = 'is_us_open_hour',
) -> pl.DataFrame:

    '''Mark rows whose hour matches the configured US open hour.'''

    return data.with_columns(
        (pl.col(datetime_col).dt.hour() == hour).cast(pl.Int8).alias(output_col)
    )


__all__: Sequence[str] = ['is_funding_hour', 'is_us_open_hour']
