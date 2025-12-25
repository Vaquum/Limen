import polars as pl


def bollinger_position(data: pl.DataFrame) -> pl.DataFrame:

    '''
    Compute price position within Bollinger Bands as percentage.
    Returns 0 when price is at lower band, 1 when at upper band, 0.5 at middle.

    Args:
        data (pl.DataFrame): Klines dataset with 'close', 'bb_upper', 'bb_lower' columns

    Returns:
        pl.DataFrame: The input data with a new column 'bollinger_position'
    '''

    return data.with_columns([
        ((pl.col('close') - pl.col('bb_lower')) /
         (pl.col('bb_upper') - pl.col('bb_lower') + 1e-8))
        .alias('bollinger_position')
    ])
