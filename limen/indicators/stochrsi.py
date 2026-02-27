import numpy as np
import polars as pl

from limen.indicators.rsi import rsi
from limen.indicators.stochf import stochf


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

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')
    if fastk_period < 1 or fastk_period > 100000:
        raise ValueError('fastk_period must be between 1 and 100000')
    if fastd_period < 1 or fastd_period > 100000:
        raise ValueError('fastd_period must be between 1 and 100000')
    if fastd_ma_type < 0 or fastd_ma_type > 8:
        raise ValueError('fastd_ma_type must be between 0 and 8')

    n = len(data)
    out_fastk = np.full(n, np.nan, dtype=float)
    out_fastd = np.full(n, np.nan, dtype=float)

    rsi_col = f'rsi_{period}'
    rsi_values = rsi(
        data,
        price_col=price_col,
        period=period,
    )[rsi_col].to_numpy().astype(float, copy=False)

    lookback_rsi = period
    if n <= lookback_rsi:
        return data.with_columns(
            [
                pl.Series(name='stochrsi_fastk', values=out_fastk),
                pl.Series(name='stochrsi_fastd', values=out_fastd),
            ]
        )

    rsi_buffer = rsi_values[lookback_rsi:]
    rsi_df = pl.DataFrame({'_rsi': rsi_buffer})

    stochf_result = stochf(
        rsi_df,
        high_col='_rsi',
        low_col='_rsi',
        close_col='_rsi',
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_ma_type=fastd_ma_type,
    )

    stochf_fastk = stochf_result['stochf_fastk'].to_numpy().astype(float, copy=False)
    stochf_fastd = stochf_result['stochf_fastd'].to_numpy().astype(float, copy=False)

    end = lookback_rsi + len(rsi_buffer)
    out_fastk[lookback_rsi:end] = stochf_fastk
    out_fastd[lookback_rsi:end] = stochf_fastd

    return data.with_columns(
        [
            pl.Series(name='stochrsi_fastk', values=out_fastk),
            pl.Series(name='stochrsi_fastd', values=out_fastd),
        ]
    )
