import polars as pl
from datetime import timedelta
from typing import List

from loop.utils.random_slice import random_slice
from loop.utils.breakout_labeling import to_average_price_klines, compute_htf_features, build_breakout_flags


def build_sample_dataset_for_breakout_regressor(
    df: pl.DataFrame,
    *,
    datetime_col: str,
    target_col: str,
    interval_sec: int,
    lookahead: timedelta,
    ema_span: int,
    deltas: List[float],
    long_col_prefix: str,
    short_col_prefix: str,
    shift_bars: int,
    random_slice_size: int,
    long_target_col: str,
    short_target_col: str,
) -> pl.DataFrame:
    '''
    Compute sample dataset with average price klines and breakout features for breakout regressor model.

    Args:
        df (pl.DataFrame): Trades dataset with 'datetime', 'volume', 'liquidity_sum' columns
        datetime_col (str): Name of the datetime column
        target_col (str): Name of the target/price column
        interval_sec (int): Bucket size in seconds for kline aggregation (e.g., 60 for 1m, 900 for 15m)
        lookahead (timedelta): Lookahead period for computing future price extremes
        ema_span (int): EMA span parameter for trend calculation
        deltas (List[float]): List of delta values for breakout calculations (e.g., [0.01, 0.02, 0.05])
        long_col_prefix (str): Prefix for long breakout columns (e.g., 'long_0_')
        short_col_prefix (str): Prefix for short breakout columns (e.g., 'short_0_')
        shift_bars (int): Number of bars to shift targets (negative for future prediction)
        random_slice_size (int): Size of the random sequential slice to return
        long_target_col (str): Name of the long breakout target column (e.g., 'breakout_long')
        short_target_col (str): Name of the short breakout target column (e.g., 'breakout_short')

    Returns:
        pl.DataFrame: The input data with new columns 'breakout_long', 'breakout_short'
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

    long_cols = [c for c in df_label.columns if c.startswith(long_col_prefix)]
    short_cols = [c for c in df_label.columns if c.startswith(short_col_prefix)]

    df_label = df_label.with_columns([
        (pl.max_horizontal([pl.when(pl.col(c) == 1).then(float(c.split('_')[-1])).otherwise(0) for c in long_cols])
         .shift(shift_bars)
         .alias(long_target_col)),

        (pl.max_horizontal([pl.when(pl.col(c) == 1).then(float(c.split('_')[-1])).otherwise(0) for c in short_cols])
         .shift(shift_bars)
         .alias(short_target_col))
    ])

    df_label = df_label.drop(long_cols + short_cols)
    df_label = df_label.drop_nulls(subset=[long_target_col, short_target_col])

    df_random = random_slice(
        df_label,
        random_slice_size,
        safe_range_low=0.05,
        safe_range_high=0.95)
    return df_random


def extract_xy(df: pl.DataFrame, target: str, horizon: int, lookback: int) -> tuple:
    '''
    Compute feature matrix X and target vector y from a DataFrame for breakout regression.

    Args:
        df (pl.DataFrame): Klines dataset with 'breakout_long', 'breakout_short' columns
        target (str): Name of the target column (e.g., 'breakout_long' or 'breakout_short')
        horizon (int): Number of periods to look ahead (prediction horizon)
        lookback (int): Number of historical periods to include as features

    Returns:
        tuple: A tuple containing:
            - x (np.ndarray): Feature matrix with shape (n_samples, n_features)
            - y (np.ndarray): Target vector with shape (n_samples,)
    '''
    x_df, y_series = extract_xy_polars(df, target, horizon, lookback)
    return x_df.to_numpy(), y_series.to_numpy()


def extract_xy_polars(df: pl.DataFrame, target: str, horizon: int, lookback: int) -> tuple:
    '''
    Compute feature matrix X and target vector y from a DataFrame for breakout regression.

    Args:
        df (pl.DataFrame): Klines dataset with 'breakout_long', 'breakout_short' columns
        target (str): Name of the target column (e.g., 'breakout_long' or 'breakout_short')
        horizon (int): Number of periods to look ahead (prediction horizon)
        lookback (int): Number of historical periods to include as features

    Returns:
        tuple: A tuple containing:
            - x (pl.DataFrame): Feature matrix as Polars DataFrame
            - y (pl.Series): Target vector as Polars Series
    '''
    lag_indices = range(horizon, horizon + lookback)

    lag_cols = [f"long_t-{i}" for i in lag_indices] + \
               [f"short_t-{i}" for i in lag_indices]

    extra_cols = df.columns
    feat_cols = list(set(lag_cols + extra_cols))

    x = df.select(feat_cols)
    y = df[target]

    return x, y
