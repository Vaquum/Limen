import numpy as np
import polars as pl


CMP_N_100000 = 100000
CMP_N_2 = 2

def _trima_from_values(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return out

    start_idx = lookback_total
    end_idx = n - 1

    if (period % 2) == 1:
        i = period >> 1
        factor = 1.0 / ((i + 1) * (i + 1))

        trailing_idx = start_idx - lookback_total
        middle_idx = trailing_idx + i
        today_idx = middle_idx + i
        numerator = 0.0

        numerator_sub = 0.0
        j = middle_idx
        while j >= trailing_idx:
            temp_real = values[j]
            numerator_sub += temp_real
            numerator += numerator_sub
            j -= 1

        numerator_add = 0.0
        middle_idx += 1
        j = middle_idx
        while j <= today_idx:
            temp_real = values[j]
            numerator_add += temp_real
            numerator += numerator_add
            j += 1

        out_idx = start_idx
        temp_real = values[trailing_idx]
        trailing_idx += 1
        out[out_idx] = numerator * factor
        out_idx += 1
        today_idx += 1

        while today_idx <= end_idx:
            numerator -= numerator_sub
            numerator_sub -= temp_real
            temp_real = values[middle_idx]
            middle_idx += 1
            numerator_sub += temp_real

            numerator += numerator_add
            numerator_add -= temp_real
            temp_real = values[today_idx]
            today_idx += 1
            numerator_add += temp_real

            numerator += temp_real

            temp_real = values[trailing_idx]
            trailing_idx += 1
            out[out_idx] = numerator * factor
            out_idx += 1
    else:
        i = period >> 1
        factor = 1.0 / (i * (i + 1))

        trailing_idx = start_idx - lookback_total
        middle_idx = trailing_idx + i - 1
        today_idx = middle_idx + i
        numerator = 0.0

        numerator_sub = 0.0
        j = middle_idx
        while j >= trailing_idx:
            temp_real = values[j]
            numerator_sub += temp_real
            numerator += numerator_sub
            j -= 1

        numerator_add = 0.0
        middle_idx += 1
        j = middle_idx
        while j <= today_idx:
            temp_real = values[j]
            numerator_add += temp_real
            numerator += numerator_add
            j += 1

        out_idx = start_idx
        temp_real = values[trailing_idx]
        trailing_idx += 1
        out[out_idx] = numerator * factor
        out_idx += 1
        today_idx += 1

        while today_idx <= end_idx:
            numerator -= numerator_sub
            numerator_sub -= temp_real
            temp_real = values[middle_idx]
            middle_idx += 1
            numerator_sub += temp_real

            numerator_add -= temp_real
            numerator += numerator_add
            temp_real = values[today_idx]
            today_idx += 1
            numerator_add += temp_real

            numerator += temp_real

            temp_real = values[trailing_idx]
            trailing_idx += 1
            out[out_idx] = numerator * factor
            out_idx += 1

    return out


def trima(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 30,
) -> pl.DataFrame:

    '''
    Compute Triangular Moving Average (TRIMA).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'trima_{period}'
    '''

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('period must be between 2 and 100000')

    out_col = f'trima_{period}'
    frame = data
    trima_expr = pl.col(price_col).map_batches(
        lambda s: pl.Series(
            _trima_from_values(
                s.to_numpy().astype(float, copy=False),
                period,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)

    return frame.with_columns(trima_expr)
