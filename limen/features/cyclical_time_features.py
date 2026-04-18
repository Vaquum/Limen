import math

import polars as pl


def cyclical_time_features(df: pl.DataFrame) -> pl.DataFrame:

    '''
    Add sine and cosine encodings for cyclical datetime fields.

    Args:
        df (pl.DataFrame): Input frame with a datetime column named `datetime`

    Returns:
        pl.DataFrame: The input frame with cyclical encodings for hour,
            minute, weekday, day_of_month, day_of_year, week_of_year, month,
            and quarter appended as `*_sin` and `*_cos` columns
    '''

    weekday = pl.col('datetime').dt.weekday()

    return df.with_columns([
        ((pl.col('datetime').dt.hour() * (2.0 * math.pi / 24.0)).sin()).alias('hour_sin'),
        ((pl.col('datetime').dt.hour() * (2.0 * math.pi / 24.0)).cos()).alias('hour_cos'),
        ((pl.col('datetime').dt.minute() * (2.0 * math.pi / 60.0)).sin()).alias('minute_sin'),
        ((pl.col('datetime').dt.minute() * (2.0 * math.pi / 60.0)).cos()).alias('minute_cos'),
        ((((weekday - 1) * (2.0 * math.pi / 7.0))).sin()).alias('weekday_sin'),
        ((((weekday - 1) * (2.0 * math.pi / 7.0))).cos()).alias('weekday_cos'),
        (((pl.col('datetime').dt.day().cast(pl.Float64) - 1.0) * (2.0 * math.pi / 31.0)).sin()).alias('day_of_month_sin'),
        (((pl.col('datetime').dt.day().cast(pl.Float64) - 1.0) * (2.0 * math.pi / 31.0)).cos()).alias('day_of_month_cos'),
        (((pl.col('datetime').dt.ordinal_day().cast(pl.Float64) - 1.0) * (2.0 * math.pi / 366.0)).sin()).alias('day_of_year_sin'),
        (((pl.col('datetime').dt.ordinal_day().cast(pl.Float64) - 1.0) * (2.0 * math.pi / 366.0)).cos()).alias('day_of_year_cos'),
        (((pl.col('datetime').dt.week().cast(pl.Float64) - 1.0) * (2.0 * math.pi / 53.0)).sin()).alias('week_of_year_sin'),
        (((pl.col('datetime').dt.week().cast(pl.Float64) - 1.0) * (2.0 * math.pi / 53.0)).cos()).alias('week_of_year_cos'),
        (((pl.col('datetime').dt.month().cast(pl.Float64) - 1.0) * (2.0 * math.pi / 12.0)).sin()).alias('month_sin'),
        (((pl.col('datetime').dt.month().cast(pl.Float64) - 1.0) * (2.0 * math.pi / 12.0)).cos()).alias('month_cos'),
        (((pl.col('datetime').dt.quarter().cast(pl.Float64) - 1.0) * (2.0 * math.pi / 4.0)).sin()).alias('quarter_sin'),
        (((pl.col('datetime').dt.quarter().cast(pl.Float64) - 1.0) * (2.0 * math.pi / 4.0)).cos()).alias('quarter_cos'),
    ])
