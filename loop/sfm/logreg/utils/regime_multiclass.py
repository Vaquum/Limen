import polars as pl

from datetime import timedelta

from loop.features.lagged_features import lag_column
from loop.features.lagged_features import lag_range
from loop.utils.breakout_labeling import to_average_price_klines
from loop.utils.breakout_labeling import compute_htf_features
from loop.utils.breakout_labeling import build_breakout_flags


def build_regime_base_features(
    df: pl.DataFrame,
    datetime_col: str,
    target_col: str,
    interval_sec: int,
    lookahead: timedelta,
    ema_span: int,
    deltas: list[float],
    long_col: str,
    short_col: str,
    leakage_shift_bars: int,
) -> pl.DataFrame:

    '''
    Compute base features for regime classification including average price klines, HTF features, breakout flags, and leakage-shifted labels.

    Args:
        df (pl.DataFrame): Klines dataset with 'datetime', 'volume', 'liquidity_sum' columns
        datetime_col (str): Column name for datetime
        target_col (str): Column name for target price
        interval_sec (int): Bucket size in seconds for kline aggregation
        lookahead (timedelta): Lookahead period for computing future price extremes
        ema_span (int): EMA span parameter for trend calculation
        deltas (list[float]): List of delta values for breakout calculations
        long_col (str): Column name for long breakout flags
        short_col (str): Column name for short breakout flags
        leakage_shift_bars (int): Number of bars to shift labels backward to prevent data leakage

    Returns:
        pl.DataFrame: The input data with average price klines, HTF features, breakout flags, and leakage-shifted 'long_base' and 'short_base' columns
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

    df_label = lag_column(df_label, long_col, leakage_shift_bars, 'long_base')
    df_label = lag_column(df_label, short_col, leakage_shift_bars, 'short_base')

    return df_label


def add_regime_lag_features(
    df: pl.DataFrame,
    lookback_bars: int,
) -> pl.DataFrame:

    '''
    Compute lagged features, counts, and age metrics for regime classification.

    Args:
        df (pl.DataFrame): Klines dataset with 'long_base', 'short_base' columns
        lookback_bars (int): Number of historical bars to look back for feature creation

    Returns:
        pl.DataFrame: The input data with lagged breakout flags, count features, net bias, and age metrics
    '''

    df = lag_range(df, 'long_base', 1, lookback_bars)
    df = lag_range(df, 'short_base', 1, lookback_bars)

    lag_long = [f'long_base_lag_{l}' for l in range(1, lookback_bars + 1)]
    lag_short = [f'short_base_lag_{l}' for l in range(1, lookback_bars + 1)]

    df = df.with_columns([
        pl.sum_horizontal(*[pl.col(c) for c in lag_long]).alias(f'long_cnt_{lookback_bars}'),
        pl.sum_horizontal(*[pl.col(c) for c in lag_short]).alias(f'short_cnt_{lookback_bars}'),
    ])

    df = df.with_columns(
        (pl.col(f'long_cnt_{lookback_bars}') - pl.col(f'short_cnt_{lookback_bars}')).alias(f'net_bias_{lookback_bars}')
    )

    df = df.with_columns([
        (pl.when(pl.col('long_base') == 1).then(0).otherwise(None).alias('_long_reset')),
        (pl.when(pl.col('short_base') == 1).then(0).otherwise(None).alias('_short_reset')),
    ])

    df = df.with_columns([
        pl.col('_long_reset').forward_fill().cum_count().alias('last_long_age'),
        pl.col('_short_reset').forward_fill().cum_count().alias('last_short_age'),
    ]).drop(['_long_reset', '_short_reset'])

    df = df.drop_nulls()

    return df