'''
Helper functions for Tradeline Directional Conditional model.
'''
import numpy as np
import polars as pl


def create_quad_labels(df: pl.DataFrame, threshold: float, lookahead_hours: int) -> pl.DataFrame:
    '''
    Compute 4 binary labels for directional conditional probability model.

    Args:
        df (pl.DataFrame): Klines dataset with 'close' column
        threshold (float): Breakout threshold as decimal (e.g., 0.01 for 1%)
        lookahead_hours (int): Number of candle periods to look ahead

    Returns:
        pl.DataFrame: The input data with four columns: 'label_long', 'label_short', 'label_movement', 'label_both'

    NOTE: Labels are forward-looking (no lookahead bias in features)
    NOTE: label_both captures whipsaw/volatile periods
    '''
    closes = df['close'].to_numpy()
    n = len(closes)

    labels_long = np.zeros(n, dtype=np.int32)
    labels_short = np.zeros(n, dtype=np.int32)
    labels_movement = np.zeros(n, dtype=np.int32)
    labels_both = np.zeros(n, dtype=np.int32)

    for i in range(n - lookahead_hours):
        current_price = closes[i]
        future_prices = closes[i+1:i+1+lookahead_hours]

        if len(future_prices) == 0:
            continue

        max_return = (np.max(future_prices) - current_price) / current_price

        min_return = (np.min(future_prices) - current_price) / current_price

        max_abs_return = max(abs(max_return), abs(min_return))

        labels_long[i] = 1 if max_return >= threshold else 0
        labels_short[i] = 1 if min_return <= -threshold else 0
        labels_movement[i] = 1 if max_abs_return >= threshold else 0
        labels_both[i] = 1 if (labels_long[i] == 1 and labels_short[i] == 1) else 0

    df = df.with_columns([
        pl.Series('label_long', labels_long),
        pl.Series('label_short', labels_short),
        pl.Series('label_movement', labels_movement),
        pl.Series('label_both', labels_both)
    ])

    return df
