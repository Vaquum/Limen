import polars as pl
from datetime import timedelta

from loop.utils.random_slice import random_slice


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
    long_target_col:str,
    short_target_col:str,
) -> pl.DataFrame:
    '''Build sample dataset with average price klines and breakout features for breakout regressor model'''
    df_avg_price = to_average_price_klines(df, interval_sec)
    df_feat = compute_htf_features(
        df_avg_price,
        datetime_col=datetime_col,
        target_col=target_col,
        lookahead=lookahead,
        ema_span=ema_span
    )
    df_label = build_breakout_flags(df_feat, deltas)
    
    # Find all long_* and short_* columns
    long_cols = [c for c in df_label.columns if c.startswith(long_col_prefix)]
    short_cols = [c for c in df_label.columns if c.startswith(short_col_prefix)]
    
    # Add breakout_long / breakout_short as max % breakout hit
    df_label = df_label.with_columns([
        (pl.max_horizontal([pl.when(pl.col(c) == 1).then(float(c.split('_')[-1])).otherwise(0) for c in long_cols])
         .shift(shift_bars)
         .alias(long_target_col)),
    
        (pl.max_horizontal([pl.when(pl.col(c) == 1).then(float(c.split('_')[-1])).otherwise(0) for c in short_cols])
         .shift(shift_bars)
         .alias(short_target_col))
    ])

    # Drop the original flag columns
    df_label = df_label.drop(long_cols + short_cols)
    df_label = df_label.drop_nulls(subset=[long_target_col, short_target_col])


    # Select random sequential dataset
    df_random = random_slice(
        df_label,
        random_slice_size,
        safe_range_low=0.05,
        safe_range_high=0.95)
    return df_random

# Extract x/y
def extract_xy(df: pl.DataFrame, target: str, horizon: int, lookback: int):
    lag_indices = range(horizon, horizon + lookback)

    # lagged flag features
    lag_cols = [f"long_t-{i}" for i in lag_indices] + \
               [f"short_t-{i}" for i in lag_indices]

    # additional features from build_lagged_flags
    extra_cols = df.columns
    feat_cols = list(set(lag_cols + extra_cols))
    x = df.select(feat_cols).to_numpy()
    y = df[target].to_numpy()

    return x, y
