import numpy as np
import polars as pl


CMP_N_100000 = 100000
CMP_N_2 = 2

_TA_EPSILON = 1e-14


def _kama_from_values(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period
    if n <= lookback_total:
        return out

    start_idx = lookback_total
    end_idx = n - 1

    const_max = 2.0 / (30.0 + 1.0)
    const_diff = (2.0 / (2.0 + 1.0)) - const_max

    sum_roc1 = 0.0
    today = start_idx - lookback_total
    trailing_idx = today
    i = period
    while i > 0:
        temp_real = values[today]
        today += 1
        temp_real -= values[today]
        sum_roc1 += abs(temp_real)
        i -= 1

    prev_kama = values[today - 1]

    temp_real = values[today]
    temp_real2 = values[trailing_idx]
    trailing_idx += 1
    period_roc = temp_real - temp_real2
    trailing_value = temp_real2

    if (sum_roc1 <= period_roc) or (-_TA_EPSILON < sum_roc1 < _TA_EPSILON):
        temp_real = 1.0
    else:
        temp_real = abs(period_roc / sum_roc1)

    temp_real = (temp_real * const_diff) + const_max
    temp_real *= temp_real
    prev_kama = ((values[today] - prev_kama) * temp_real) + prev_kama
    today += 1

    while today <= start_idx:
        temp_real = values[today]
        temp_real2 = values[trailing_idx]
        trailing_idx += 1
        period_roc = temp_real - temp_real2

        sum_roc1 -= abs(trailing_value - temp_real2)
        sum_roc1 += abs(temp_real - values[today - 1])
        trailing_value = temp_real2

        if (sum_roc1 <= period_roc) or (-_TA_EPSILON < sum_roc1 < _TA_EPSILON):
            temp_real = 1.0
        else:
            temp_real = abs(period_roc / sum_roc1)

        temp_real = (temp_real * const_diff) + const_max
        temp_real *= temp_real
        prev_kama = ((values[today] - prev_kama) * temp_real) + prev_kama
        today += 1

    out_beg_idx = today - 1
    out[out_beg_idx] = prev_kama

    while today <= end_idx:
        temp_real = values[today]
        temp_real2 = values[trailing_idx]
        trailing_idx += 1
        period_roc = temp_real - temp_real2

        sum_roc1 -= abs(trailing_value - temp_real2)
        sum_roc1 += abs(temp_real - values[today - 1])
        trailing_value = temp_real2

        if (sum_roc1 <= period_roc) or (-_TA_EPSILON < sum_roc1 < _TA_EPSILON):
            temp_real = 1.0
        else:
            temp_real = abs(period_roc / sum_roc1)

        temp_real = (temp_real * const_diff) + const_max
        temp_real *= temp_real
        prev_kama = ((values[today] - prev_kama) * temp_real) + prev_kama
        out[today] = prev_kama
        today += 1

    return out


def kama(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 30,
) -> pl.DataFrame:

    '''
    Compute Kaufman Adaptive Moving Average (KAMA).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'kama_{period}'
    '''

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('period must be between 2 and 100000')

    out_col = f'kama_{period}'
    frame = data
    kama_expr = pl.col(price_col).map_batches(
        lambda s: pl.Series(
            _kama_from_values(
                s.to_numpy().astype(float, copy=False),
                period,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)
    return frame.with_columns(kama_expr)
