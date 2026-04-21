import polars as pl

EPSILON = 1e-10


def range_per_dollar_volume(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Compute normalized bar range per traded dollar.

    Args:
        data (pl.DataFrame): Klines dataset with high, low, close, and volume columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        volume_col (str): Column name for traded volume

    Returns:
        pl.DataFrame: The input data with a new column 'range_per_dollar_volume'
    '''

    range_pct = (pl.col(high_col) - pl.col(low_col)) / (pl.col(close_col) + EPSILON)
    dollar_volume_expr = pl.col(close_col) * pl.col(volume_col)

    return data.with_columns(
        (range_pct / (dollar_volume_expr + EPSILON)).alias('range_per_dollar_volume')
    )
