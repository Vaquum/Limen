import math
import numpy as np
import polars as pl

from limen.indicators._hilbert import do_hilbert_transform, init_hilbert_state

CMP_N_0_01 = 0.01
CMP_N_0_015 = 0.015
CMP_N_3 = 3
CMP_N_315_0 = 315.0
CMP_N_50_0 = 50.0
CMP_N_6_0 = 6.0

HT_TRENDMODE_PERIOD = 0.0
HT_TRENDMODE_SMOOTH_PERIOD = 0.0


_SMOOTH_PRICE_SIZE = 50


def _ht_trendmode_from_values(values: np.ndarray) -> np.ndarray:
    n = len(values)

    out = np.zeros(n, dtype=float)

    lookback_total = 63
    if n <= lookback_total:
        return out

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

    for _ in range(34):
        temp_real = values[today]
        today += 1
        smoothed_value = do_price_wma(temp_real)

    hilbert_idx = 0
    detrender_state = init_hilbert_state()
    q1_state = init_hilbert_state()
    ji_state = init_hilbert_state()
    jq_state = init_hilbert_state()

    period = HT_TRENDMODE_PERIOD
    out_idx = 0
    prev_i2 = 0.0
    prev_q2 = 0.0
    re = 0.0
    im = 0.0

    i1_for_odd_prev2 = 0.0
    i1_for_odd_prev3 = 0.0
    i1_for_even_prev2 = 0.0
    i1_for_even_prev3 = 0.0
    smooth_period = HT_TRENDMODE_SMOOTH_PERIOD

    i_trend1 = 0.0
    i_trend2 = 0.0
    i_trend3 = 0.0
    days_in_trend = 0
    prev_dc_phase = 0.0
    dc_phase = 0.0
    prev_sine = 0.0
    sine = 0.0
    prev_lead_sine = 0.0
    lead_sine = 0.0

    smooth_price = [0.0] * _SMOOTH_PRICE_SIZE
    smooth_price_idx = 0

    temp_real = math.atan(1.0)
    rad2deg = 45.0 / temp_real
    deg2rad = 1.0 / rad2deg
    const_deg2rad_by_360 = temp_real * 8.0

    while today <= end_idx:
        adjusted_prev_period = (0.075 * period) + 0.54

        today_value = values[today]
        smoothed_value = do_price_wma(today_value)
        smooth_price[smooth_price_idx] = smoothed_value

        if (today % 2) == 0:
            detrender = do_hilbert_transform(
                detrender_state,
                smoothed_value,
                adjusted_prev_period,
                hilbert_idx,
                True,
            )
            q1 = do_hilbert_transform(
                q1_state,
                detrender,
                adjusted_prev_period,
                hilbert_idx,
                True,
            )
            ji = do_hilbert_transform(
                ji_state,
                i1_for_even_prev3,
                adjusted_prev_period,
                hilbert_idx,
                True,
            )
            jq = do_hilbert_transform(
                jq_state,
                q1,
                adjusted_prev_period,
                hilbert_idx,
                True,
            )

            hilbert_idx += 1
            if hilbert_idx == CMP_N_3:
                hilbert_idx = 0

            q2 = (0.2 * (q1 + ji)) + (0.8 * prev_q2)
            i2 = (0.2 * (i1_for_even_prev3 - jq)) + (0.8 * prev_i2)

            i1_for_odd_prev3 = i1_for_odd_prev2
            i1_for_odd_prev2 = detrender
        else:
            detrender = do_hilbert_transform(
                detrender_state,
                smoothed_value,
                adjusted_prev_period,
                hilbert_idx,
                False,
            )
            q1 = do_hilbert_transform(
                q1_state,
                detrender,
                adjusted_prev_period,
                hilbert_idx,
                False,
            )
            ji = do_hilbert_transform(
                ji_state,
                i1_for_odd_prev3,
                adjusted_prev_period,
                hilbert_idx,
                False,
            )
            jq = do_hilbert_transform(
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

        re = (0.2 * ((i2 * prev_i2) + (q2 * prev_q2))) + (0.8 * re)
        im = (0.2 * ((i2 * prev_q2) - (q2 * prev_i2))) + (0.8 * im)
        prev_q2 = q2
        prev_i2 = i2

        temp_real = period
        if (im != 0.0) and (re != 0.0):
            period = 360.0 / (math.atan(im / re) * rad2deg)

        temp_real2 = 1.5 * temp_real
        period = min(period, temp_real2)
        temp_real2 = 0.67 * temp_real
        period = max(period, temp_real2)
        if period < CMP_N_6_0:
            period = 6.0
        elif period > CMP_N_50_0:
            period = 50.0
        period = (0.2 * period) + (0.8 * temp_real)

        smooth_period = (0.33 * period) + (0.67 * smooth_period)

        prev_dc_phase = dc_phase
        dc_period = smooth_period + 0.5
        dc_period_int = int(dc_period)
        real_part = 0.0
        imag_part = 0.0

        idx = smooth_price_idx
        for i in range(dc_period_int):
            temp_real = (float(i) * const_deg2rad_by_360) / float(dc_period_int)
            temp_real2 = smooth_price[idx]
            real_part += math.sin(temp_real) * temp_real2
            imag_part += math.cos(temp_real) * temp_real2
            if idx == 0:
                idx = _SMOOTH_PRICE_SIZE - 1
            else:
                idx -= 1

        temp_real = abs(imag_part)
        if temp_real > 0.0:
            dc_phase = math.atan(real_part / imag_part) * rad2deg
        elif temp_real <= CMP_N_0_01:
            if real_part < 0.0:
                dc_phase -= 90.0
            elif real_part > 0.0:
                dc_phase += 90.0

        dc_phase += 90.0
        dc_phase += 360.0 / smooth_period
        if imag_part < 0.0:
            dc_phase += 180.0
        if dc_phase > CMP_N_315_0:
            dc_phase -= 360.0

        prev_sine = sine
        prev_lead_sine = lead_sine
        sine = math.sin(dc_phase * deg2rad)
        lead_sine = math.sin((dc_phase + 45.0) * deg2rad)

        dc_period = smooth_period + 0.5
        dc_period_int = int(dc_period)
        idx = today
        temp_real = 0.0
        for _ in range(dc_period_int):
            if idx < 0:
                break
            temp_real += values[idx]
            idx -= 1

        if dc_period_int > 0:
            temp_real = temp_real / float(dc_period_int)

        trendline = (4.0 * temp_real + 3.0 * i_trend1 + 2.0 * i_trend2 + i_trend3) / 10.0
        i_trend3 = i_trend2
        i_trend2 = i_trend1
        i_trend1 = temp_real

        trend = 1
        if ((sine > lead_sine) and (prev_sine <= prev_lead_sine)) or (
            (sine < lead_sine) and (prev_sine >= prev_lead_sine)
        ):
            days_in_trend = 0
            trend = 0

        days_in_trend += 1

        if days_in_trend < (0.5 * smooth_period):
            trend = 0

        temp_real = dc_phase - prev_dc_phase
        if (smooth_period != 0.0) and (
            (temp_real > (0.67 * 360.0 / smooth_period))
            and (temp_real < (1.5 * 360.0 / smooth_period))
        ):
            trend = 0

        temp_real = smooth_price[smooth_price_idx]
        if (trendline != 0.0) and (abs((temp_real - trendline) / trendline) >= CMP_N_0_015):
            trend = 1

        if today >= start_idx:
            out[start_idx + out_idx] = float(trend)
            out_idx += 1

        smooth_price_idx += 1
        if smooth_price_idx == _SMOOTH_PRICE_SIZE:
            smooth_price_idx = 0

        today += 1

    return out


def ht_trendmode(
    data: pl.DataFrame,
    price_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Hilbert Transform - Trend vs Cycle Mode.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price

    Returns:
        pl.DataFrame: The input data with a new column 'ht_trendmode'
    '''

    return data.with_columns(
        pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _ht_trendmode_from_values(
                    s.to_numpy().astype(float, copy=False)
                )
            ),
            return_dtype=pl.Float64,
        ).alias('ht_trendmode')
    )
