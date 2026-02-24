import polars as pl


def ad(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Compute Chaikin Accumulation/Distribution (A/D) line.
    
    Args:
        data (pl.DataFrame): Klines dataset with high/low/close/volume columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        volume_col (str): Column name for volume

    Returns:
        pl.DataFrame: The input data with a new column 'ad'
    '''

    hl_range = pl.col(high_col) - pl.col(low_col)
    money_flow_volume = (
        pl.when(hl_range > 0.0)
        .then(
            (
                ((pl.col(close_col) - pl.col(low_col)) - (pl.col(high_col) - pl.col(close_col)))
                / hl_range
            ) * pl.col(volume_col).cast(pl.Float64)
        )
        .otherwise(0.0)
    )

    return data.with_columns(money_flow_volume.cum_sum().alias('ad'))
