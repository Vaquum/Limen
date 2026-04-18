import polars as pl


def dollar_volume(
    data: pl.DataFrame,
    price_col: str = 'close',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Compute traded dollar volume from price and volume columns.

    Args:
        data (pl.DataFrame): Klines dataset with price and volume columns
        price_col (str): Column name used as the price proxy for traded value
        volume_col (str): Column name for traded volume

    Returns:
        pl.DataFrame: The input data with a new column 'dollar_volume'
    '''

    return data.with_columns((pl.col(price_col) * pl.col(volume_col)).alias('dollar_volume'))
