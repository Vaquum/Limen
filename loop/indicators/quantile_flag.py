import polars as pl


def quantile_flag(data: pl.DataFrame,
                  col: str = "close",
                  q: float = 0.1,
                  cutoff: float = None,
                  return_cutoff: bool = False) -> pl.DataFrame:
    
    '''
    Mark rows where `col` exceeds the (1 - q) quantile.

    Args:
        data (pl.DataFrame): The input data.
        col (str): The column name on which to compute the quantile.
        q (float): A value in [0,1]; if q = 0.1, use the 90% quantile.
        cutoff (float): Optional pre-calculated cutoff value. If provided,
                       this value is used instead of calculating from data.
                       This prevents data leakage when applying training
                       thresholds to test data.
        return_cutoff (bool): If True, returns a tuple (data, cutoff) instead
                             of just the data. Useful for applying the same
                             cutoff to multiple datasets.

    Returns:
        pl.DataFrame: The input data with a new UInt8 column "quantile_flag"
                       that is 1 when `col` > cutoff, else 0
        OR
        tuple: (pl.DataFrame, float) if return_cutoff is True
    '''

    if cutoff is None:
        cutoff = data.select(
            pl.col(col).quantile(1.0 - q)
        ).item()
    
    result = data.with_columns([
        (pl.col(col) > cutoff)
            .cast(pl.UInt8)
            .alias("quantile_flag")
    ])
    
    if return_cutoff:
        return result, cutoff
    return result
