import polars as pl

from datetime import timedelta

from loop.utils.breakout_labeling import to_average_price_klines
from loop.utils.breakout_labeling import compute_htf_features
from loop.utils.breakout_labeling import build_breakout_flags

def build_breakout_regressor_base_features(
    df: pl.DataFrame,
    datetime_col: str,
    target_col: str,
    interval_sec: int,
    lookahead: timedelta,
    ema_span: int,
    deltas: list[float],
    long_col_prefix: str,
    short_col_prefix: str,
    shift_bars: int,
    long_target_col: str,
    short_target_col: str,
) -> pl.DataFrame:

    '''
    Compute base features for breakout regressor including average price klines, HTF features, breakout flags, and target columns.

    Args:
        df (pl.DataFrame): Trades dataset with 'datetime', 'volume', 'liquidity_sum' columns
        datetime_col (str): Column name for datetime in `{df}`
        target_col (str): Column name for target price in `{df}`
        interval_sec (int): Bucket size in seconds for kline aggregation
        lookahead (timedelta): Lookahead period for computing future price extremes
        ema_span (int): EMA span parameter for trend calculation
        deltas (list[float]): List of delta values for breakout calculations
        long_col_prefix (str): Prefix for long breakout columns
        short_col_prefix (str): Prefix for short breakout columns
        shift_bars (int): Number of bars to shift targets (negative for future prediction)
        long_target_col (str): Name of the long breakout target column
        short_target_col (str): Name of the short breakout target column

    Returns:
        pl.DataFrame: The input data with average price klines, HTF features, breakout flags, and target columns `{long_target_col}` and `{short_target_col}`
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

    long_cols = [c for c in df_label.collect_schema().names() if c.startswith(long_col_prefix)]
    short_cols = [c for c in df_label.collect_schema().names() if c.startswith(short_col_prefix)]

    df_label = df_label.with_columns([
        (pl.max_horizontal([pl.when(pl.col(c) == 1).then(float(c.split('_')[-1])).otherwise(0) for c in long_cols])
         .shift(shift_bars)
         .alias(long_target_col)),

        (pl.max_horizontal([pl.when(pl.col(c) == 1).then(float(c.split('_')[-1])).otherwise(0) for c in short_cols])
         .shift(shift_bars)
         .alias(short_target_col))
    ])

    df_label = df_label.drop(long_cols + short_cols)

    return df_label
