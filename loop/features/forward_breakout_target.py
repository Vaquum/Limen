import polars as pl


def forward_breakout_target(data: pl.DataFrame,
                            forward_periods: int = 24,
                            threshold: float = 0.02,
                            shift: int = -1) -> pl.DataFrame:

    '''
    Compute binary target for forward price breakouts.

    Target = 1 if price increases >= threshold in next forward_periods
    Target = 0 otherwise

    Args:
        data (pl.DataFrame): DataFrame with 'close' column
        forward_periods (int): How many periods ahead to check
        threshold (float): Percentage threshold (0.02 = 2%)
        shift (int): Additional shift to apply (negative for forward-looking)

    Returns:
        pl.DataFrame: The input data with a new column 'forward_breakout'
    '''

    future_price = pl.col('close').shift(-forward_periods)
    forward_return = (future_price - pl.col('close')) / pl.col('close')

    target = (forward_return >= threshold).cast(pl.UInt8).alias('forward_breakout')

    result = data.with_columns([target])

    if shift != 0:
        result = result.with_columns([
            pl.col('forward_breakout').shift(shift).alias('forward_breakout')
        ])

    return result
