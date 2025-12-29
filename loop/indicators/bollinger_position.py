import polars as pl


BANDWIDTH_EPSILON = 1e-6


def bollinger_position(data: pl.DataFrame) -> pl.DataFrame:

    '''
    Compute price position within Bollinger Bands as percentage.
    Returns 0 when price is at lower band, 1 when at upper band, 0.5 at middle.

    Args:
        data (pl.DataFrame): Klines dataset with 'close', 'bb_upper', 'bb_lower' columns

    Returns:
        pl.DataFrame: The input data with a new column 'bollinger_position'
    '''

    bandwidth = pl.col('bb_upper') - pl.col('bb_lower')
    position = (
        pl.when(bandwidth < BANDWIDTH_EPSILON)
        .then(0.5)
        .otherwise((pl.col('close') - pl.col('bb_lower')) / bandwidth)
        .clip(0.0, 1.0)
    )

    return data.with_columns([position.alias('bollinger_position')])
