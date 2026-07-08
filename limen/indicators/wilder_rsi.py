import polars as pl

from limen.indicators.rsi import rsi_from_values


CMP_N_100000 = 100000
CMP_N_2 = 2


def wilder_rsi(data: pl.DataFrame,
               period: int = 14) -> pl.DataFrame:

    '''
    Compute Wilder's RSI using canonical Wilder smoothing.

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Number of periods for RSI calculation

    Returns:
        pl.DataFrame: The input data with a new column 'wilder_rsi_{period}'
    '''

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('wilder_rsi period must be between 2 and 100000')

    out_col = f'wilder_rsi_{period}'
    return data.with_columns(
        pl.col('close').map_batches(
            lambda s: pl.Series(
                rsi_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        ).alias(out_col)
    )
