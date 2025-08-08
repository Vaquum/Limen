import polars as pl
from datetime import timedelta
from typing import List


def to_average_price_klines(df: pl.DataFrame, interval_sec: int) -> pl.DataFrame:
    
    '''
    Create average price klines from trade-by-trade liquidity data.
    
    Args:
        df (pl.DataFrame): Trades dataset with 'datetime', 'volume', 'liquidity_sum' columns
        interval_sec (int): Bucket size in seconds for aggregation
        
    Returns:
        pl.DataFrame: The input data aggregated with 'datetime' and 'average_price' columns
    '''

    return (
        df
        .with_columns(
            pl.col('datetime')
              .dt.truncate(every=f"{interval_sec}s")
              .alias('bucket')
        )
        .group_by('bucket')
        .agg(
            (
                pl.col('liquidity_sum').sum()
                / pl.col('volume').sum()
            ).alias('average_price')
        )
        .sort('bucket')
        .rename({'bucket': 'datetime'})
    )


def compute_htf_features(df: pl.DataFrame,
                         *,
                         datetime_col: str,
                         target_col: str,
                         lookahead: timedelta,
                         ema_span: int) -> pl.DataFrame:
    
    '''
    Compute higher timeframe features with EMA and future price extremes.
    
    Args:
        df (pl.DataFrame): Klines dataset with datetime and price columns
        datetime_col (str): Column name for datetime values
        target_col (str): Column name for price values
        lookahead (timedelta): Time period to look ahead for future extremes
        ema_span (int): Number of periods for EMA calculation
        
    Returns:
        pl.DataFrame: The input data with new columns 'ema_2h', 'future_max', 'future_min'
    '''

    # 1) sort data by datetime
    df = df.sort(datetime_col)

    #2) Compute EMA on the 2H average price directly
    df = df.with_columns(pl.col(target_col).ewm_mean(span=ema_span, adjust=False).alias('ema_2h'))

    # 3) compute future_max and future_min
    if df.shape[0] < 2:
        raise ValueError("Dataframe must have at least two rows to compute intervals.")
    interval_diffs = df[datetime_col].diff().drop_nulls()
    if interval_diffs.is_empty():
        raise ValueError("No valid intervals found in datetime column.")
    secs = int(interval_diffs[0].total_seconds())
    window = int(lookahead.total_seconds() // secs)

    prices = df[target_col].to_list()
    rev = prices[::-1]

    fut_max = (
        pl.Series(rev)
          .rolling_max(window+1)
          .shift(-1)
          .to_list()[::-1]
    )
    fut_min = (
        pl.Series(rev)
          .rolling_min(window+1)
          .shift(-1)
          .to_list()[::-1]
    )

    return df.with_columns([
        pl.Series('future_max', fut_max),
        pl.Series('future_min', fut_min),
    ])


def build_breakout_flags(df_feats: pl.DataFrame,
                         deltas: List[float],
                         *,
                         datetime_col: str = 'datetime') -> pl.DataFrame:
    
    '''
    Compute breakout flags based on future price extremes versus EMA thresholds.
    
    Args:
        df_feats (pl.DataFrame): Features dataset with 'ema_2h', 'future_max', 'future_min' columns
        deltas (List[float]): List of delta values for breakout threshold calculations
        datetime_col (str): Column name for datetime values
        
    Returns:
        pl.DataFrame: The input data with new columns for long and short breakout flags
    '''
    
    exprs = []
    for delta in deltas:
        exprs.append(
            (pl.col('future_max') >= pl.col('ema_2h') * (1 + delta))
            .cast(pl.Int8)
            .fill_null(0)
            .alias(f"long_{delta:.2f}".replace('.', '_'))
        )
        exprs.append(
            (pl.col('future_min') <= pl.col('ema_2h') * (1 - delta))
            .cast(pl.Int8)
            .fill_null(0)
            .alias(f"short_{delta:.2f}".replace('.', '_'))
        )

    return df_feats.select([datetime_col] + exprs).sort(datetime_col)
