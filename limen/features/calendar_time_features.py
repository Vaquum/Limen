import polars as pl


WEEKEND_START_WEEKDAY = 6


def calendar_time_features(df: pl.DataFrame) -> pl.DataFrame:

    '''
    Add discrete calendar time features from the datetime column.

    Args:
        df (pl.DataFrame): Input frame with a datetime column named `datetime`

    Returns:
        pl.DataFrame: The input frame with `hour`, `minute`, `weekday`,
            `day_of_month`, `day_of_year`, `week_of_year`, `month`,
            `quarter`, and `is_weekend` appended. `weekday` uses ISO
            numbering with `Monday=1` through `Sunday=7`, and
            `week_of_year` follows ISO week numbering
    '''

    weekday = pl.col('datetime').dt.weekday()

    return df.with_columns([
        pl.col('datetime').dt.hour().alias('hour'),
        pl.col('datetime').dt.minute().alias('minute'),
        weekday.alias('weekday'),
        pl.col('datetime').dt.day().alias('day_of_month'),
        pl.col('datetime').dt.ordinal_day().alias('day_of_year'),
        pl.col('datetime').dt.week().alias('week_of_year'),
        pl.col('datetime').dt.month().alias('month'),
        pl.col('datetime').dt.quarter().alias('quarter'),
        (weekday >= WEEKEND_START_WEEKDAY).cast(pl.Int8).alias('is_weekend'),
    ])
