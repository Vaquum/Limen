import numpy as np
import polars as pl

from limen.indicators.rsi import _rsi_from_values
from limen.indicators.stochf import _stochf_from_arrays


CMP_N_100000 = 100000
CMP_N_2 = 2
CMP_N_8 = 8

def _stochrsi_from_values(
    values: np.ndarray,
    period: int,
    fastk_period: int,
    fastd_period: int,
    fastd_ma_type: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    out_fastk = np.full(n, np.nan, dtype=float)
    out_fastd = np.full(n, np.nan, dtype=float)

    rsi_values = _rsi_from_values(values, period)
    lookback_rsi = period
    if n <= lookback_rsi:
        return out_fastk, out_fastd

    rsi_buffer = rsi_values[lookback_rsi:]
    stochf_fastk, stochf_fastd = _stochf_from_arrays(
        rsi_buffer,
        rsi_buffer,
        rsi_buffer,
        fastk_period,
        fastd_period,
        fastd_ma_type,
    )

    end = lookback_rsi + len(rsi_buffer)
    out_fastk[lookback_rsi:end] = stochf_fastk
    out_fastd[lookback_rsi:end] = stochf_fastd
    return out_fastk, out_fastd


def _stochrsi_component(
    values: np.ndarray,
    period: int,
    fastk_period: int,
    fastd_period: int,
    fastd_ma_type: int,
    component_index: int,
) -> np.ndarray:
    return _stochrsi_from_values(
        values,
        period,
        fastk_period,
        fastd_period,
        fastd_ma_type,
    )[component_index]


def stochrsi(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_ma_type: int = 0,
) -> pl.DataFrame:

    '''
    Compute Stochastic RSI (TA_STOCHRSI): fast %K and fast %D on RSI values.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): RSI period (2..100000)
        fastk_period (int): Time period for Fast-K (1..100000)
        fastd_period (int): Smoothing period for Fast-D (1..100000)
        fastd_ma_type (int): MA type for Fast-D (0..8)

    Returns:
        pl.DataFrame: The input data with 'stochrsi_fastk' and 'stochrsi_fastd'
    '''

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('period must be between 2 and 100000')
    if fastk_period < 1 or fastk_period > CMP_N_100000:
        raise ValueError('fastk_period must be between 1 and 100000')
    if fastd_period < 1 or fastd_period > CMP_N_100000:
        raise ValueError('fastd_period must be between 1 and 100000')
    if fastd_ma_type < 0 or fastd_ma_type > CMP_N_8:
        raise ValueError('fastd_ma_type must be between 0 and 8')

    frame = data
    return frame.with_columns(
        [
            pl.col(price_col).map_batches(
                lambda s: pl.Series(
                    _stochrsi_component(
                        s.to_numpy().astype(float, copy=False),
                        period,
                        fastk_period,
                        fastd_period,
                        fastd_ma_type,
                        0,
                    )
                ),
                return_dtype=pl.Float64,
            ).alias('stochrsi_fastk'),
            pl.col(price_col).map_batches(
                lambda s: pl.Series(
                    _stochrsi_component(
                        s.to_numpy().astype(float, copy=False),
                        period,
                        fastk_period,
                        fastd_period,
                        fastd_ma_type,
                        1,
                    )
                ),
                return_dtype=pl.Float64,
            ).alias('stochrsi_fastd'),
        ]
    )
