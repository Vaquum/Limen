import polars as pl


EPSILON = 1e-10


def close_position(
    data: pl.DataFrame,
    window: int = 1,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    output_col: str = 'close_position',
) -> pl.DataFrame:

    '''
    Compute close position within the high-low range as a unit interval fraction.

    Args:
        data (pl.DataFrame): Klines dataset with high, low, and close columns
        window (int): Rolling mean window; `1` preserves the single-bar behavior
        high_col (str): High price column
        low_col (str): Low price column
        close_col (str): Close price column
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with a new close-position column
    '''

    price_range = pl.col(high_col) - pl.col(low_col)
    position = (
        pl.when(price_range.abs() <= EPSILON)
        .then(None)
        .otherwise((pl.col(close_col) - pl.col(low_col)) / price_range)
    )
    if window == 1:
        return data.with_columns(position.alias(output_col))
    if window <= 0:
        raise ValueError('window must be positive')
    return data.with_columns(position.rolling_mean(window_size=window).alias(output_col))
