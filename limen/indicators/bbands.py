import numpy as np
import polars as pl

from limen.indicators._bbands import _stddev_from_var, _stddev_using_precalc_ma
from limen.indicators.ma import ma


def bbands(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 5,
    nb_dev_up: float = 2.0,
    nb_dev_dn: float = 2.0,
    ma_type: int = 0,
) -> pl.DataFrame:

    '''
    Compute Bollinger Bands (upper/middle/lower).

    Args:
        data (pl.DataFrame): Dataset with price column
        price_col (str): Column name for input price
        period (int): Rolling window length
        nb_dev_up (float): Upper-band deviation multiplier
        nb_dev_dn (float): Lower-band deviation multiplier
        ma_type (int): TA-Lib MA type

    Returns:
        pl.DataFrame: Input data with 'bbands_upper', 'bbands_middle', 'bbands_lower'
    '''

    if period < 2:
        raise ValueError('period must be >= 2')
    if ma_type < 0 or ma_type > 8:
        raise ValueError('ma_type must be between 0 and 8')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    upper = np.full(n, np.nan, dtype=float)
    middle = np.full(n, np.nan, dtype=float)
    lower = np.full(n, np.nan, dtype=float)
    if n == 0:
        return data.with_columns(
            [
                pl.Series(name='bbands_upper', values=upper),
                pl.Series(name='bbands_middle', values=middle),
                pl.Series(name='bbands_lower', values=lower),
            ]
        )

    middle_col = f'ma_{period}_{ma_type}'
    middle_values = ma(data, price_col=price_col, period=period, ma_type=ma_type)[middle_col].to_numpy().astype(float, copy=False)
    middle[:] = middle_values

    valid_idx = np.flatnonzero(~np.isnan(middle_values))
    if valid_idx.size == 0:
        return data.with_columns(
            [
                pl.Series(name='bbands_upper', values=upper),
                pl.Series(name='bbands_middle', values=middle),
                pl.Series(name='bbands_lower', values=lower),
            ]
        )
    out_beg_idx = int(valid_idx[0])
    out_nb_element = n - out_beg_idx

    if ma_type == 0:
        std_values = _stddev_using_precalc_ma(
            values,
            middle_values,
            out_beg_idx,
            out_nb_element,
            period,
        )
    else:
        _, std_values = _stddev_from_var(values, out_beg_idx, n - 1, period)

    if nb_dev_up == nb_dev_dn:
        if nb_dev_up == 1.0:
            upper[out_beg_idx:] = middle_values[out_beg_idx:] + std_values
            lower[out_beg_idx:] = middle_values[out_beg_idx:] - std_values
        else:
            scaled = std_values * nb_dev_up
            upper[out_beg_idx:] = middle_values[out_beg_idx:] + scaled
            lower[out_beg_idx:] = middle_values[out_beg_idx:] - scaled
    elif nb_dev_up == 1.0:
        upper[out_beg_idx:] = middle_values[out_beg_idx:] + std_values
        lower[out_beg_idx:] = middle_values[out_beg_idx:] - (std_values * nb_dev_dn)
    elif nb_dev_dn == 1.0:
        lower[out_beg_idx:] = middle_values[out_beg_idx:] - std_values
        upper[out_beg_idx:] = middle_values[out_beg_idx:] + (std_values * nb_dev_up)
    else:
        upper[out_beg_idx:] = middle_values[out_beg_idx:] + (std_values * nb_dev_up)
        lower[out_beg_idx:] = middle_values[out_beg_idx:] - (std_values * nb_dev_dn)

    return data.with_columns(
        [
            pl.Series(name='bbands_upper', values=upper),
            pl.Series(name='bbands_middle', values=middle),
            pl.Series(name='bbands_lower', values=lower),
        ]
    )
