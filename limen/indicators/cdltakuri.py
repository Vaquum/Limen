import numpy as np
import polars as pl

CDLTAKURI_BODY_DOJI_AVG_PERIOD = 10
CDLTAKURI_BODY_DOJI_FACTOR = 0.1
CDLTAKURI_BODY_DOJI_PERIOD_TOTAL = 0.0
CDLTAKURI_SHADOW_VL_AVG_PERIOD = 0
CDLTAKURI_SHADOW_VL_FACTOR = 2.0
CDLTAKURI_SHADOW_VL_PERIOD_TOTAL = 0.0
CDLTAKURI_SHADOW_VS_AVG_PERIOD = 10
CDLTAKURI_SHADOW_VS_FACTOR = 0.1
CDLTAKURI_SHADOW_VS_PERIOD_TOTAL = 0.0


def _cdltakuri_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Takuri (Dragonfly Doji with very long lower shadow) candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdltakuri'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_doji_avg_period = CDLTAKURI_BODY_DOJI_AVG_PERIOD
    body_doji_factor = CDLTAKURI_BODY_DOJI_FACTOR
    shadow_vs_avg_period = CDLTAKURI_SHADOW_VS_AVG_PERIOD
    shadow_vs_factor = CDLTAKURI_SHADOW_VS_FACTOR
    shadow_vl_avg_period = CDLTAKURI_SHADOW_VL_AVG_PERIOD
    shadow_vl_factor = CDLTAKURI_SHADOW_VL_FACTOR
    lookback_total = max(body_doji_avg_period, shadow_vs_avg_period, shadow_vl_avg_period)

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdltakuri', values=out))

    body_doji_period_total = CDLTAKURI_BODY_DOJI_PERIOD_TOTAL
    shadow_vs_period_total = CDLTAKURI_SHADOW_VS_PERIOD_TOTAL
    shadow_vl_period_total = CDLTAKURI_SHADOW_VL_PERIOD_TOTAL
    body_doji_trailing_idx = lookback_total - body_doji_avg_period
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period
    shadow_vl_trailing_idx = lookback_total - shadow_vl_avg_period

    i = body_doji_trailing_idx
    while i < lookback_total:
        body_doji_period_total += high_values[i] - low_values[i]
        i += 1

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total += high_values[i] - low_values[i]
        i += 1

    i = shadow_vl_trailing_idx
    while i < lookback_total:
        shadow_vl_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = lookback_total
    while i < n:
        real_body = abs(close_values[i] - open_values[i])
        upper_shadow = high_values[i] - max(open_values[i], close_values[i])
        lower_shadow = min(open_values[i], close_values[i]) - low_values[i]

        body_doji_avg = body_doji_factor * (body_doji_period_total / body_doji_avg_period)
        shadow_vs_avg = shadow_vs_factor * (shadow_vs_period_total / shadow_vs_avg_period)
        if shadow_vl_avg_period == 0:
            shadow_vl_avg = shadow_vl_factor * real_body
        else:
            shadow_vl_avg = shadow_vl_factor * (shadow_vl_period_total / shadow_vl_avg_period)

        if (
            real_body <= body_doji_avg
            and upper_shadow < shadow_vs_avg
            and lower_shadow > shadow_vl_avg
        ):
            out[i] = 100

        body_doji_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[body_doji_trailing_idx] - low_values[body_doji_trailing_idx])
        )
        shadow_vs_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[shadow_vs_trailing_idx] - low_values[shadow_vs_trailing_idx])
        )
        shadow_vl_period_total += (
            real_body
            - abs(close_values[shadow_vl_trailing_idx] - open_values[shadow_vl_trailing_idx])
        )

        i += 1
        body_doji_trailing_idx += 1
        shadow_vs_trailing_idx += 1
        shadow_vl_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdltakuri', values=out))


def cdltakuri(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdltakuri'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdltakuri_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
