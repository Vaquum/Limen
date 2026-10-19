from limen._parallel_contract import attach as _parallel
import polars as pl


def medprice(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
) -> pl.DataFrame:

    '''
    Compute Median Price.

    Args:
        low_col (str): Column name for low prices

    Returns:
        pl.DataFrame: The input data with a new column 'medprice'
    '''

    med_price = (
        pl.col(high_col).cast(pl.Float64)
        + pl.col(low_col).cast(pl.Float64)
    ) / 2.0

    return data.with_columns(med_price.alias('medprice'))

_parallel(medprice, backend_imports=('polars',))
