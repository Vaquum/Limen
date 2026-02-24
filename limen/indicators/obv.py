import polars as pl


def obv(
    data: pl.DataFrame,
    price_col: str = 'close',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Compute On-Balance Volume (OBV).

    Args:
        data (pl.DataFrame): Dataset with price and volume columns
        price_col (str): Column name for price series
        volume_col (str): Column name for volume series

    Returns:
        pl.DataFrame: The input data with a new column 'obv'
    '''

    signed_volume = (
        pl.when(pl.col(price_col).diff(1) > 0.0)
        .then(pl.col(volume_col).cast(pl.Float64))
        .when(pl.col(price_col).diff(1) < 0.0)
        .then(-pl.col(volume_col).cast(pl.Float64))
        .otherwise(0.0)
    )

    return data.with_columns(
        (pl.col(volume_col).cast(pl.Float64).first() + signed_volume.cum_sum()).alias('obv')
    )
