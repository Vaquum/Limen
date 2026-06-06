import polars as pl

from limen.indicators._atr import _atr_from_true_range_expr

ATR_PERIOD_MIN = 1
ATR_PERIOD_MAX = 100000
ATR_OUT_COL_PREFIX = 'atr_'
ATR_TR_COL = '__atr_tr'


def atr(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Average True Range (ATR).

    Args:
        data (pl.DataFrame): Klines dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        period (int): Number of periods for ATR calculation

    Returns:
        pl.DataFrame: The input data with a new column 'atr_{period}'
    '''

    if period < ATR_PERIOD_MIN or period > ATR_PERIOD_MAX:
        raise ValueError(f"atr period must be between {ATR_PERIOD_MIN} and {ATR_PERIOD_MAX}")

    out_col = f"{ATR_OUT_COL_PREFIX}{period}"
    prev_close = pl.col(close_col).shift(1)
    true_range_expr = (
        pl.max_horizontal(pl.col(high_col), prev_close)
        - pl.min_horizontal(pl.col(low_col), prev_close)
    ).alias(ATR_TR_COL)

    frame = data.with_columns(true_range_expr)

    return frame.with_columns(
        _atr_from_true_range_expr(ATR_TR_COL, period).alias(out_col)
    ).drop(ATR_TR_COL)
