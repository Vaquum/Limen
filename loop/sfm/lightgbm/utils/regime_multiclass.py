import polars as pl
from datetime import timedelta
from typing import List


from loop.utils.random_slice import random_slice
from loop.features.lag_column import lag_column
from loop.features.lag_range import lag_range

from loop.utils.breakout_labeling import to_average_price_klines, compute_htf_features, build_breakout_flags

def build_sample_dataset_for_regime_multiclass(
    df: pl.DataFrame,
    *,
    datetime_col: str,
    target_col: str,
    interval_sec: int,
    lookahead: timedelta,
    ema_span: int,
    deltas: List[float],
    long_col: str,
    short_col: str,
    leakage_shift_bars: int,
    random_slice_size: int,
    random_slice_min_pct: float = 0.25,
    random_slice_max_pct: float = 0.75,
) -> pl.DataFrame:
    '''
    Build sample dataset with average price klines and breakout features for regime multiclass model.
    
    Args:
        df (pl.DataFrame): Raw trade data containing columns
            - datetime_col (datetime, UTC ms)
            - volume (float)
            - liquidity_sum (float)
        datetime_col (str): Name of the datetime column
        target_col (str): Name of the target/price column
        interval_sec (int): Bucket size in seconds for kline aggregation (e.g., 60 for 1m, 900 for 15m)
        lookahead (timedelta): Lookahead period for computing future price extremes
        ema_span (int): EMA span parameter for trend calculation
        deltas (List[float]): List of delta values for breakout calculations
        long_col (str): Name of the long breakout column
        short_col (str): Name of the short breakout column
        leakage_shift_bars (int): Number of bars to shift labels to prevent data leakage
        random_slice_size (int): Size of the random sequential slice to return
        random_slice_min_pct (float): Minimum percentage for random slice range (default: 0.25)
        random_slice_max_pct (float): Maximum percentage for random slice range (default: 0.75)
    
    Returns:
        pl.DataFrame: Processed dataset with columns
            - datetime_col: Original datetime
            - long_base: Leakage-shifted long breakout flags
            - short_base: Leakage-shifted short breakout flags
            - Additional breakout columns for each delta value
    '''
    df_avg_price = to_average_price_klines(df, interval_sec)
    df_feat = compute_htf_features(
        df_avg_price,
        datetime_col=datetime_col,
        target_col=target_col,
        lookahead=lookahead,
        ema_span=ema_span
    )
    df_label = build_breakout_flags(df_feat, deltas)
    
    # BASE-SHIFT (prevent leakage by shifting labels)
    df_label = lag_column(df_label, long_col, leakage_shift_bars, 'long_base')
    df_label = lag_column(df_label, short_col, leakage_shift_bars, 'short_base')

    # Select random sequential dataset with configurable range
    df_random = random_slice(df_label, random_slice_size, 
                            safe_range_low=random_slice_min_pct, 
                            safe_range_high=random_slice_max_pct)
    return df_random


def add_features_to_regime_multiclass_dataset(
    df: pl.DataFrame,
    *,
    lookback_bars: int,
    long_col: str,
    short_col: str,
) -> pl.DataFrame:
    '''
    Add lagged features, counts, and regime labels for breakout classification.
    
    Creates features from historical breakout flags to predict future regime:
    - Lagged breakout flags (1 to LOOKBACK_BARS periods back)
    - Count of breakouts in lookback window (long/short)
    - Net bias (long count - short count)
    - Age since last breakout event
    - Final regime label (0=flat, 1=bullish, 2=bearish)
    
    Args:
        df (pl.DataFrame): Input DataFrame containing columns
            - long_base: Leakage-shifted long breakout flags (0/1)
            - short_base: Leakage-shifted short breakout flags (0/1)
            - long_col: Current long breakout column name
            - short_col: Current short breakout column name
        lookback_bars (int): Number of historical bars to look back for feature creation
        long_col (str): Name of the current long breakout column
        short_col (str): Name of the current short breakout column
    
    Returns:
        pl.DataFrame: Enhanced dataset with additional columns:
            - long_base_lag_1 to long_base_lag_{lookback_bars}: Historical long breakout flags
            - short_base_lag_1 to short_base_lag_{lookback_bars}: Historical short breakout flags
            - long_cnt_{lookback_bars}: Count of long breakouts in lookback window
            - short_cnt_{lookback_bars}: Count of short breakouts in lookback window
            - net_bias_{lookback_bars}: Net bias (long count - short count)
            - last_long_age: Bars since last long breakout
            - last_short_age: Bars since last short breakout
            - regime: Target label (0=flat, 1=bullish, 2=bearish)
    '''
    # 1) Create lagged breakout flags (1 to LOOKBACK_BARS periods back)
    df = lag_range(df, 'long_base', 1, lookback_bars)
    df = lag_range(df, 'short_base', 1, lookback_bars)
    
    # 2) Count breakouts in lookback window and calculate net bias
    lag_long = [f'long_base_lag_{l}' for l in range(1, lookback_bars + 1)]
    lag_short = [f'short_base_lag_{l}' for l in range(1, lookback_bars + 1)]
    
    df = df.with_columns([
        pl.sum_horizontal(*[pl.col(c) for c in lag_long]).alias(f'long_cnt_{lookback_bars}'),
        pl.sum_horizontal(*[pl.col(c) for c in lag_short]).alias(f'short_cnt_{lookback_bars}'),
    ])
    
    # Net bias (positive = more long breakouts, negative = more short breakouts)
    df = df.with_columns(
        (pl.col(f'long_cnt_{lookback_bars}') - pl.col(f'short_cnt_{lookback_bars}')).alias(f'net_bias_{lookback_bars}')
    )
    
    # 3) Calculate age since last breakout (time-based features)
    df = df.with_columns([
        (pl.when(pl.col('long_base') == 1).then(0).otherwise(None).alias('_long_reset')),
        (pl.when(pl.col('short_base') == 1).then(0).otherwise(None).alias('_short_reset')),
    ])
    
    df = df.with_columns([
        pl.col('_long_reset').forward_fill().cum_count().alias('last_long_age'),
        pl.col('_short_reset').forward_fill().cum_count().alias('last_short_age'),
    ]).drop(['_long_reset', '_short_reset'])
    
    # 4) Clean data (drop rows with NaNs from initial lagging)
    df = df.drop_nulls()
    
    # 5) Create target label using current breakout flags (already leak-safe)
    # Regime classification logic:
    # - Class 0 (flat): No breakout or both long and short breakouts (conflicting signals)
    # - Class 1 (bullish): Only long breakout detected
    # - Class 2 (bearish): Only short breakout detected
    # Note: If both long and short breakouts occur simultaneously, we classify as flat
    # to avoid conflicting signals and focus on clear directional moves
    df = df.with_columns(
        pl.when((pl.col(long_col) == 1) & (pl.col(short_col) == 0)).then(1)   # bullish breakout
          .when((pl.col(short_col) == 1) & (pl.col(long_col) == 0)).then(2)   # bearish breakout
          .otherwise(0)                                                        # flat/no breakout
          .alias('regime')
    )
    
    return df
