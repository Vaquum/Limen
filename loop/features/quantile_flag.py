import polars as pl


def compute_quantile_cutoff(data: pl.DataFrame, col: str, q: float) -> float:
    '''
    Compute quantile cutoff value for binary flag creation.

    Args:
        data (pl.DataFrame): Dataset with specified column
        col (str): Column name to compute quantile on
        q (float): Quantile parameter (0.1 = 90th percentile)

    Returns:
        float: Cutoff value at (1-q) quantile
    '''
    return data.select(pl.col(col).quantile(1.0 - q)).item()


def quantile_flag(data: pl.DataFrame, col: str, cutoff: float) -> pl.DataFrame:
    '''
    Apply binary flag based on pre-computed cutoff value.

    Args:
        data (pl.DataFrame): Dataset with specified column
        col (str): Column name to apply flag on
        cutoff (float): Pre-computed cutoff value

    Returns:
        pl.DataFrame: Data with quantile_flag column added
    '''
    return data.with_columns([
        (pl.col(col) > cutoff)
            .cast(pl.UInt8)
            .alias('quantile_flag')
    ])