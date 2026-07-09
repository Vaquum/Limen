import numpy as np
import numpy.typing as npt
import polars as pl


CMP_N_100000 = 100000

TA_EPSILON = 1e-14


def _calc_terms(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    day: int,
) -> tuple[float, float]:
    temp_lt = low[day]
    temp_ht = high[day]
    temp_cy = close[day - 1]

    true_low = min(temp_lt, temp_cy)
    close_minus_true_low = close[day] - true_low

    true_range = temp_ht - temp_lt
    temp_double = abs(temp_cy - temp_ht)
    true_range = max(true_range, temp_double)
    temp_double = abs(temp_cy - temp_lt)
    true_range = max(true_range, temp_double)

    return close_minus_true_low, true_range


def _ultosc_from_arrays(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    period1: int,
    period2: int,
    period3: int,
) -> npt.NDArray[np.float64]:
    n = len(close)
    out = np.full(n, np.nan, dtype=float)

    sorted_periods = sorted([period1, period2, period3])
    p1, p2, p3 = sorted_periods[0], sorted_periods[1], sorted_periods[2]

    lookback_total = p3
    if n <= lookback_total:
        return out

    a1_total = 0.0
    b1_total = 0.0
    a2_total = 0.0
    b2_total = 0.0
    a3_total = 0.0
    b3_total = 0.0

    start_idx = lookback_total
    for i in range(start_idx - p1 + 1, start_idx):
        cm_tl, tr = _calc_terms(high, low, close, i)
        a1_total += cm_tl
        b1_total += tr
    for i in range(start_idx - p2 + 1, start_idx):
        cm_tl, tr = _calc_terms(high, low, close, i)
        a2_total += cm_tl
        b2_total += tr
    for i in range(start_idx - p3 + 1, start_idx):
        cm_tl, tr = _calc_terms(high, low, close, i)
        a3_total += cm_tl
        b3_total += tr

    today = start_idx
    trailing_idx1 = today - p1 + 1
    trailing_idx2 = today - p2 + 1
    trailing_idx3 = today - p3 + 1

    while today < n:
        cm_tl, tr = _calc_terms(high, low, close, today)
        a1_total += cm_tl
        a2_total += cm_tl
        a3_total += cm_tl
        b1_total += tr
        b2_total += tr
        b3_total += tr

        output = 0.0
        if abs(b1_total) >= TA_EPSILON:
            output += 4.0 * (a1_total / b1_total)
        if abs(b2_total) >= TA_EPSILON:
            output += 2.0 * (a2_total / b2_total)
        if abs(b3_total) >= TA_EPSILON:
            output += a3_total / b3_total

        cm_tl, tr = _calc_terms(high, low, close, trailing_idx1)
        a1_total -= cm_tl
        b1_total -= tr

        cm_tl, tr = _calc_terms(high, low, close, trailing_idx2)
        a2_total -= cm_tl
        b2_total -= tr

        cm_tl, tr = _calc_terms(high, low, close, trailing_idx3)
        a3_total -= cm_tl
        b3_total -= tr

        out[today] = 100.0 * (output / 7.0)

        today += 1
        trailing_idx1 += 1
        trailing_idx2 += 1
        trailing_idx3 += 1

    return out


def ultosc(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
) -> pl.DataFrame:

    '''
    Compute Ultimate Oscillator (ULTOSC).

    Args:
        data (pl.DataFrame): Dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        period1 (int): Number of bars for 1st period (1..100000)
        period2 (int): Number of bars for 2nd period (1..100000)
        period3 (int): Number of bars for 3rd period (1..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'ultosc_{period1}_{period2}_{period3}'
    '''

    if period1 < 1 or period1 > CMP_N_100000:
        raise ValueError('ultosc period1 must be between 1 and 100000')
    if period2 < 1 or period2 > CMP_N_100000:
        raise ValueError('ultosc period2 must be between 1 and 100000')
    if period3 < 1 or period3 > CMP_N_100000:
        raise ValueError('ultosc period3 must be between 1 and 100000')

    out_col = f"ultosc_{period1}_{period2}_{period3}"
    frame = data
    ultosc_expr = pl.struct([high_col, low_col, close_col]).map_batches(
        lambda s: pl.Series(
            _ultosc_from_arrays(
                s.struct.field(high_col).to_numpy().astype(float, copy=False),
                s.struct.field(low_col).to_numpy().astype(float, copy=False),
                s.struct.field(close_col).to_numpy().astype(float, copy=False),
                period1,
                period2,
                period3,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)

    return frame.with_columns(ultosc_expr)
