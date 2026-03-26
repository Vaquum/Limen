import numpy as np
import polars as pl

CDLBELTHOLD_BODY_LONG_AVG_PERIOD = 10
CDLBELTHOLD_BODY_LONG_PERIOD_TOTAL = 0.0
CDLBELTHOLD_SHADOW_VS_AVG_PERIOD = 10
CDLBELTHOLD_SHADOW_VS_FACTOR = 0.1
CDLBELTHOLD_SHADOW_VS_PERIOD_TOTAL = 0.0


def _cdlbelthold_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Belt-hold candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlbelthold'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_long_avg_period = CDLBELTHOLD_BODY_LONG_AVG_PERIOD
    shadow_vs_avg_period = CDLBELTHOLD_SHADOW_VS_AVG_PERIOD
    shadow_vs_factor = CDLBELTHOLD_SHADOW_VS_FACTOR
    lookback_total = max(body_long_avg_period, shadow_vs_avg_period)

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlbelthold', values=out))

    body_long_period_total = CDLBELTHOLD_BODY_LONG_PERIOD_TOTAL
    shadow_vs_period_total = CDLBELTHOLD_SHADOW_VS_PERIOD_TOTAL
    body_long_trailing_idx = lookback_total - body_long_avg_period
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period

    i = body_long_trailing_idx
    while i < lookback_total:
        body_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total += high_values[i] - low_values[i]
        i += 1

    i = lookback_total
    while i < n:
        real_body = abs(close_values[i] - open_values[i])
        color = 1 if close_values[i] >= open_values[i] else -1
        lower_shadow = min(open_values[i], close_values[i]) - low_values[i]
        upper_shadow = high_values[i] - max(open_values[i], close_values[i])

        body_long_avg = body_long_period_total / body_long_avg_period
        shadow_vs_avg = shadow_vs_factor * (shadow_vs_period_total / shadow_vs_avg_period)

        if real_body > body_long_avg and (
            (color == 1 and lower_shadow < shadow_vs_avg)
            or (color == -1 and upper_shadow < shadow_vs_avg)
        ):
            out[i] = color * 100

        body_long_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_long_trailing_idx] - open_values[body_long_trailing_idx])
        )
        shadow_vs_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[shadow_vs_trailing_idx] - low_values[shadow_vs_trailing_idx])
        )

        i += 1
        body_long_trailing_idx += 1
        shadow_vs_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlbelthold', values=out))


def cdlbelthold(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdlbelthold'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdlbelthold_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
