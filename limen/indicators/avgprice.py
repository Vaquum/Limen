import polars as pl


def avgprice(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Average Price.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'avgprice'
    '''

    avg_price = (
        pl.col(open_col).cast(pl.Float64)
        + pl.col(high_col).cast(pl.Float64)
        + pl.col(low_col).cast(pl.Float64)
        + pl.col(close_col).cast(pl.Float64)
    ) / 4.0

    return data.with_columns(avg_price.alias('avgprice'))
