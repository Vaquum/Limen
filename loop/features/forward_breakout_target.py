import polars as pl


def compute_forward_breakout_threshold(data: pl.DataFrame,
                                      forward_periods: int = 24,
                                      threshold_pct: float = 0.02) -> float:
    '''
    Compute threshold for forward breakout detection.
    This is a fitted parameter, computed on training data only.

    Args:
        data: Training data (unused, kept for fitted transform interface)
        forward_periods: Number of periods to look ahead
        threshold_pct: Percentage threshold (0.02 = 2%)

    Returns:
        The threshold value (percentage)
    '''
    del data, forward_periods
    return threshold_pct


def forward_breakout_target(data: pl.DataFrame,
                            forward_periods: int = 24,
                            threshold: float = 0.02,
                            shift: int = -1) -> pl.DataFrame:
    '''
    Create binary target for forward price breakouts.

    Target = 1 if price increases >= threshold in next forward_periods
    Target = 0 otherwise

    Args:
        data: DataFrame with 'close' column
        forward_periods: How many periods ahead to check
        threshold: Percentage threshold (0.02 = 2%)
        shift: Additional shift to apply (negative for forward-looking)

    Returns:
        Data with 'forward_breakout' column added
    '''
    # Calculate forward return
    future_price = pl.col('close').shift(-forward_periods)
    forward_return = (future_price - pl.col('close')) / pl.col('close')

    # Create binary flag
    target = (forward_return >= threshold).cast(pl.UInt8).alias('forward_breakout')

    result = data.with_columns([target])

    # Apply additional shift if needed
    if shift != 0:
        result = result.with_columns([
            pl.col('forward_breakout').shift(shift).alias('forward_breakout')
        ])

    return result
