import loop
import polars as pl
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List


def to_average_price_klines(df: pl.DataFrame, interval_sec: int) -> pl.DataFrame:
    '''
    Aggregate trade-by-trade liquidity into K-second 'klines'.

    Args:
        df (pl.DataFrame): Must contain columns
            - datetime (datetime, UTC ms)
            - volume (float)
            - liquidity_sum (float)
        interval_sec (int): Bucket size in seconds – e.g. 60 for 1 m, 900 for 15 m.

    Returns:
        pl.DataFrame: datetime · average_price
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


def compute_htf_features(
    df: pl.DataFrame,
    *,
    datetime_col: str,
    target_col: str,
    lookahead: timedelta,
    ema_span: int,
) -> pl.DataFrame:
    """
    From a 2 h-bar DataFrame, compute and attach:
      • `_bucket`     : each datetime truncated to the HTF grid
      • `ema_htf`     : EMA(span=ema_span) of the last price_col in each bucket
      • `future_max`  : max price_col in (t, t+lookahead]
      • `future_min`  : min price_col in (t, t+lookahead]

    Args:
        df (pl.DataFrame): Input DataFrame
        datetime_col (str): Name of datetime column
        target_col (str): Name of target/price column
        lookahead (timedelta): Lookahead period
        ema_span (int): EMA span parameter

    Returns:
        pl.DataFrame: DataFrame with columns
            [ datetime_col, target_col, _bucket, ema_htf, future_max, future_min ]
            sorted by datetime_col.
    """
    # 1) bucket into HTF
    df_htf = (
        df.sort(datetime_col)
          .with_columns(
             pl.col(datetime_col)
               .dt.truncate(every=lookahead)
               .alias('_bucket')
          )
    )

    # 2) extract one close per bucket → daily closes
    htf = (
        df_htf
        .group_by('_bucket', maintain_order=True)
        .agg(pl.col(target_col).last().alias('close'))
        .sort('_bucket')
    )

    # 3) compute EMA over those closes using polars
    htf = htf.with_columns(
        pl.col('close').ewm_mean(span=ema_span, adjust=False).alias('ema_htf')
    )

    # 4) join EMA back
    df_htf = df_htf.join(
        htf.select(['_bucket', 'ema_htf']),
        on='_bucket', how='left'
    )

    # 5) compute future_max and future_min
    # infer the lower‐TF bar size in seconds
    secs = int((df_htf[datetime_col][1] - df_htf[datetime_col][0]).total_seconds())
    window = int(lookahead.total_seconds() // secs)

    prices = df_htf[target_col].to_list()
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

    return df_htf.with_columns([
        pl.Series('future_max', fut_max),
        pl.Series('future_min', fut_min),
    ])


def build_breakout_flags(
    df_feats: pl.DataFrame,
    deltas: List[float],
    *,
    datetime_col: str = 'datetime'
) -> pl.DataFrame:
    """
    From a DataFrame with columns
      [datetime_col, ema_htf, future_max, future_min],
    build and return a DataFrame:
      [ datetime_col,
        long_{Δ:.2f} for each Δ,
        short_{Δ:.2f} for each Δ ]
    where
      long = future_max >= ema_htf*(1+Δ)
      short = future_min <= ema_htf*(1−Δ)

    Args:
        df_feats (pl.DataFrame): Input DataFrame with required columns
        deltas (List[float]): List of delta values for breakout calculations
        datetime_col (str): Name of datetime column, defaults to 'datetime'

    Returns:
        pl.DataFrame: DataFrame with breakout flags
    """
    exprs = []
    for delta in deltas:
        exprs.append(
            (pl.col('future_max') >= pl.col('ema_htf') * (1 + delta))
            .cast(pl.Int8)
            .fill_null(0)
            .alias(f"long_{delta:.2f}")
        )
        exprs.append(
            (pl.col('future_min') <= pl.col('ema_htf') * (1 - delta))
            .cast(pl.Int8)
            .fill_null(0)
            .alias(f"short_{delta:.2f}")
        )

    return df_feats.select([datetime_col] + exprs).sort(datetime_col)
