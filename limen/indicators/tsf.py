import numpy as np
import polars as pl


def _tsf_from_values(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return out

    start_idx = lookback_total
    end_idx = n - 1

    sum_x = period * (period - 1) * 0.5
    sum_x_sqr = period * (period - 1) * (2 * period - 1) / 6.0
    divisor = (sum_x * sum_x) - (period * sum_x_sqr)

    today = start_idx
    while today <= end_idx:
        sum_xy = 0.0
        sum_y = 0.0

        i = period - 1
        while i >= 0:
            temp_value = values[today - i]
            sum_y += temp_value
            sum_xy += i * temp_value
            i -= 1

        m = ((period * sum_xy) - (sum_x * sum_y)) / divisor
        b = (sum_y - (m * sum_x)) / period
        out[today] = b + (m * period)
        today += 1

    return out


def tsf(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Time Series Forecast (TSF).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'tsf_{period}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    out_col = f'tsf_{period}'
    frame = data
    tsf_expr = pl.col(price_col).map_batches(
        lambda s: pl.Series(
            _tsf_from_values(
                s.to_numpy().astype(float, copy=False),
                period,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)
    return frame.with_columns(tsf_expr)
