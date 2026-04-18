import math

import polars as pl


HOUR_PERIOD = 24.0
MINUTE_PERIOD = 60.0
ISO_WEEKDAY_PERIOD = 7.0
DAY_OF_MONTH_PERIOD = 31.0
DAY_OF_YEAR_PERIOD = 366.0
ISO_WEEK_OF_YEAR_PERIOD = 53.0
MONTH_PERIOD = 12.0
QUARTER_PERIOD = 4.0
ISO_WEEKDAY_OFFSET = 1.0
CYCLICAL_TURN = 2.0 * math.pi
HOUR_ANGLE = CYCLICAL_TURN / HOUR_PERIOD
MINUTE_ANGLE = CYCLICAL_TURN / MINUTE_PERIOD
ISO_WEEKDAY_ANGLE = CYCLICAL_TURN / ISO_WEEKDAY_PERIOD
DAY_OF_MONTH_ANGLE = CYCLICAL_TURN / DAY_OF_MONTH_PERIOD
DAY_OF_YEAR_ANGLE = CYCLICAL_TURN / DAY_OF_YEAR_PERIOD
ISO_WEEK_OF_YEAR_ANGLE = CYCLICAL_TURN / ISO_WEEK_OF_YEAR_PERIOD
MONTH_ANGLE = CYCLICAL_TURN / MONTH_PERIOD
QUARTER_ANGLE = CYCLICAL_TURN / QUARTER_PERIOD


def cyclical_time_features(df: pl.DataFrame) -> pl.DataFrame:

    '''
    Add sine and cosine encodings for cyclical datetime fields.

    Args:
        df (pl.DataFrame): Input frame with a datetime column named `datetime`

    Returns:
        pl.DataFrame: The input frame with cyclical encodings for hour,
            minute, weekday, day_of_month, day_of_year, week_of_year, month,
            and quarter appended as `*_sin` and `*_cos` columns. Weekday and
            week-of-year use ISO numbering, and weekday cycles are phase-
            aligned with `weekday - 1` before sine/cosine encoding
    '''

    weekday = pl.col('datetime').dt.weekday()

    return df.with_columns([
        ((pl.col('datetime').dt.hour() * HOUR_ANGLE).sin()).alias('hour_sin'),
        ((pl.col('datetime').dt.hour() * HOUR_ANGLE).cos()).alias('hour_cos'),
        ((pl.col('datetime').dt.minute() * MINUTE_ANGLE).sin()).alias('minute_sin'),
        ((pl.col('datetime').dt.minute() * MINUTE_ANGLE).cos()).alias('minute_cos'),
        (((weekday - ISO_WEEKDAY_OFFSET) * ISO_WEEKDAY_ANGLE).sin()).alias('weekday_sin'),
        (((weekday - ISO_WEEKDAY_OFFSET) * ISO_WEEKDAY_ANGLE).cos()).alias('weekday_cos'),
        (((pl.col('datetime').dt.day().cast(pl.Float64) - ISO_WEEKDAY_OFFSET) * DAY_OF_MONTH_ANGLE).sin()).alias('day_of_month_sin'),
        (((pl.col('datetime').dt.day().cast(pl.Float64) - ISO_WEEKDAY_OFFSET) * DAY_OF_MONTH_ANGLE).cos()).alias('day_of_month_cos'),
        (((pl.col('datetime').dt.ordinal_day().cast(pl.Float64) - ISO_WEEKDAY_OFFSET) * DAY_OF_YEAR_ANGLE).sin()).alias('day_of_year_sin'),
        (((pl.col('datetime').dt.ordinal_day().cast(pl.Float64) - ISO_WEEKDAY_OFFSET) * DAY_OF_YEAR_ANGLE).cos()).alias('day_of_year_cos'),
        (((pl.col('datetime').dt.week().cast(pl.Float64) - ISO_WEEKDAY_OFFSET) * ISO_WEEK_OF_YEAR_ANGLE).sin()).alias('week_of_year_sin'),
        (((pl.col('datetime').dt.week().cast(pl.Float64) - ISO_WEEKDAY_OFFSET) * ISO_WEEK_OF_YEAR_ANGLE).cos()).alias('week_of_year_cos'),
        (((pl.col('datetime').dt.month().cast(pl.Float64) - ISO_WEEKDAY_OFFSET) * MONTH_ANGLE).sin()).alias('month_sin'),
        (((pl.col('datetime').dt.month().cast(pl.Float64) - ISO_WEEKDAY_OFFSET) * MONTH_ANGLE).cos()).alias('month_cos'),
        (((pl.col('datetime').dt.quarter().cast(pl.Float64) - ISO_WEEKDAY_OFFSET) * QUARTER_ANGLE).sin()).alias('quarter_sin'),
        (((pl.col('datetime').dt.quarter().cast(pl.Float64) - ISO_WEEKDAY_OFFSET) * QUARTER_ANGLE).cos()).alias('quarter_cos'),
    ])
