import polars as pl

def quantile_flag(data: pl.DataFrame,
                  col: str = "close",
                  q: float = 0.1) -> pl.DataFrame:
    '''
    Mark rows where `col` exceeds the (1 - q) quantile.

    Args:
        data (pl.DataFrame): The input data
        col (str): The column name on which to compute the quantile
        q (float): A value in [0,1]; if q = 0.1, use the 90% quantile

    Returns:
        pl.DataFrame: The input data with a new UInt8 column "quantile_flag"
                       that is 1 when `col` > cutoff, else 0
    '''
    
    cutoff = data.select(
        pl.col(col).quantile(1.0 - q)
    ).item()
    
    return data.with_columns([
        (pl.col(col) > cutoff)
            .cast(pl.UInt8)
            .alias("quantile_flag")
    ])
