import polars as pl


<<<<<<< HEAD
def medprice(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
) -> pl.DataFrame:

    '''
    Compute Median Price.

    Args:
        data (pl.DataFrame): Dataset with high and low columns
=======
def medprice(data: pl.DataFrame,
             high_col: str = 'high',
             low_col: str = 'low') -> pl.DataFrame:

    '''
    Compute Median Price (MEDPRICE) indicator.

    Equivalent to TA-Lib MEDPRICE: (high + low) / 2.

    Args:
        data (pl.DataFrame): Klines dataset with 'high' and 'low' columns
>>>>>>> origin/main
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices

    Returns:
        pl.DataFrame: The input data with a new column 'medprice'
    '''

<<<<<<< HEAD
    med_price = (
        pl.col(high_col).cast(pl.Float64)
        + pl.col(low_col).cast(pl.Float64)
    ) / 2.0

    return data.with_columns(med_price.alias('medprice'))
=======
    return data.with_columns([
        ((pl.col(high_col) + pl.col(low_col)) / 2).alias('medprice')
    ])
>>>>>>> origin/main
