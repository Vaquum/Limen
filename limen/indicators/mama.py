import math

import numpy as np
import polars as pl

from limen.indicators._hilbert import _do_hilbert_transform, _init_hilbert_state


def mama(
    data: pl.DataFrame,
    price_col: str = 'close',
    fast_limit: float = 0.5,
    slow_limit: float = 0.05,
) -> pl.DataFrame:

    '''
    Compute MESA Adaptive Moving Average (MAMA) and Following Adaptive MA (FAMA).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        fast_limit (float): Upper adaptive limit
        slow_limit (float): Lower adaptive limit

    Returns:
        pl.DataFrame: The input data with new columns 'mama' and 'fama'
    '''

    if fast_limit < 0.01 or fast_limit > 0.99:
        raise ValueError('fast_limit must be between 0.01 and 0.99')
    if slow_limit < 0.01 or slow_limit > 0.99:
        raise ValueError('slow_limit must be between 0.01 and 0.99')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_mama = np.full(n, np.nan, dtype=float)
    out_fama = np.full(n, np.nan, dtype=float)

    lookback_total = 32
    if n <= lookback_total:
        return data.with_columns([
            pl.Series(name='mama', values=out_mama),
            pl.Series(name='fama', values=out_fama),
        ])

    start_idx = lookback_total
    end_idx = n - 1

    trailing_wma_idx = start_idx - lookback_total
    today = trailing_wma_idx

    temp_real = values[today]
    today += 1
    period_wma_sub = temp_real
    period_wma_sum = temp_real

    temp_real = values[today]
    today += 1
    period_wma_sub += temp_real
    period_wma_sum += temp_real * 2.0

    temp_real = values[today]
    today += 1
    period_wma_sub += temp_real
    period_wma_sum += temp_real * 3.0

    trailing_wma_value = 0.0

    def do_price_wma(new_price: float) -> float:
        nonlocal period_wma_sub, period_wma_sum, trailing_wma_value, trailing_wma_idx

        period_wma_sub += new_price
        period_wma_sub -= trailing_wma_value
        period_wma_sum += new_price * 4.0
        trailing_wma_value = values[trailing_wma_idx]
        trailing_wma_idx += 1
        smoothed = period_wma_sum * 0.1
        period_wma_sum -= period_wma_sub
        return smoothed

    for _ in range(9):
        temp_real = values[today]
        today += 1
        smoothed_value = do_price_wma(temp_real)

    hilbert_idx = 0
    detrender_state = _init_hilbert_state()
    q1_state = _init_hilbert_state()
    ji_state = _init_hilbert_state()
    jq_state = _init_hilbert_state()

    period = 0.0
    out_idx = 0
    prev_i2 = 0.0
    prev_q2 = 0.0
    re = 0.0
    im = 0.0

    mama_value = 0.0
    fama_value = 0.0
    prev_phase = 0.0

    i1_for_odd_prev2 = 0.0
    i1_for_odd_prev3 = 0.0
    i1_for_even_prev2 = 0.0
    i1_for_even_prev3 = 0.0

    rad2deg = 180.0 / (4.0 * math.atan(1.0))

    while today <= end_idx:
        adjusted_prev_period = (0.075 * period) + 0.54

        today_value = values[today]
        smoothed_value = do_price_wma(today_value)

        if (today % 2) == 0:
            detrender = _do_hilbert_transform(
                detrender_state,
                smoothed_value,
                adjusted_prev_period,
                hilbert_idx,
                True,
            )
            q1 = _do_hilbert_transform(
                q1_state,
                detrender,
                adjusted_prev_period,
                hilbert_idx,
                True,
            )
            ji = _do_hilbert_transform(
                ji_state,
                i1_for_even_prev3,
                adjusted_prev_period,
                hilbert_idx,
                True,
            )
            jq = _do_hilbert_transform(
                jq_state,
                q1,
                adjusted_prev_period,
                hilbert_idx,
                True,
            )

            hilbert_idx += 1
            if hilbert_idx == 3:
                hilbert_idx = 0

            q2 = (0.2 * (q1 + ji)) + (0.8 * prev_q2)
            i2 = (0.2 * (i1_for_even_prev3 - jq)) + (0.8 * prev_i2)

            i1_for_odd_prev3 = i1_for_odd_prev2
            i1_for_odd_prev2 = detrender

            if i1_for_even_prev3 != 0.0:
                temp_real2 = math.atan(q1 / i1_for_even_prev3) * rad2deg
            else:
                temp_real2 = 0.0
        else:
            detrender = _do_hilbert_transform(
                detrender_state,
                smoothed_value,
                adjusted_prev_period,
                hilbert_idx,
                False,
            )
            q1 = _do_hilbert_transform(
                q1_state,
                detrender,
                adjusted_prev_period,
                hilbert_idx,
                False,
            )
            ji = _do_hilbert_transform(
                ji_state,
                i1_for_odd_prev3,
                adjusted_prev_period,
                hilbert_idx,
                False,
            )
            jq = _do_hilbert_transform(
                jq_state,
                q1,
                adjusted_prev_period,
                hilbert_idx,
                False,
            )

            q2 = (0.2 * (q1 + ji)) + (0.8 * prev_q2)
            i2 = (0.2 * (i1_for_odd_prev3 - jq)) + (0.8 * prev_i2)

            i1_for_even_prev3 = i1_for_even_prev2
            i1_for_even_prev2 = detrender

            if i1_for_odd_prev3 != 0.0:
                temp_real2 = math.atan(q1 / i1_for_odd_prev3) * rad2deg
            else:
                temp_real2 = 0.0

        temp_real = prev_phase - temp_real2
        prev_phase = temp_real2
        if temp_real < 1.0:
            temp_real = 1.0

        if temp_real > 1.0:
            temp_real = fast_limit / temp_real
            if temp_real < slow_limit:
                temp_real = slow_limit
        else:
            temp_real = fast_limit

        mama_value = (temp_real * today_value) + ((1.0 - temp_real) * mama_value)
        temp_real *= 0.5
        fama_value = (temp_real * mama_value) + ((1.0 - temp_real) * fama_value)

        if today >= start_idx:
            out_mama[start_idx + out_idx] = mama_value
            out_fama[start_idx + out_idx] = fama_value
            out_idx += 1

        re = (0.2 * ((i2 * prev_i2) + (q2 * prev_q2))) + (0.8 * re)
        im = (0.2 * ((i2 * prev_q2) - (q2 * prev_i2))) + (0.8 * im)
        prev_q2 = q2
        prev_i2 = i2

        temp_real = period
        if (im != 0.0) and (re != 0.0):
            period = 360.0 / (math.atan(im / re) * rad2deg)

        temp_real2 = 1.5 * temp_real
        if period > temp_real2:
            period = temp_real2
        temp_real2 = 0.67 * temp_real
        if period < temp_real2:
            period = temp_real2
        if period < 6.0:
            period = 6.0
        elif period > 50.0:
            period = 50.0
        period = (0.2 * period) + (0.8 * temp_real)

        today += 1

    return data.with_columns([
        pl.Series(name='mama', values=out_mama),
        pl.Series(name='fama', values=out_fama),
    ])
