import polars as pl

from limen.indicators._atr import _atr_from_true_range_expr

NATR_PERIOD_MIN = 1
NATR_PERIOD_MAX = 100000
NATR_SCALE = 100.0
NATR_OUT_COL_PREFIX = 'natr_'
NATR_TR_COL = '__natr_tr'
NATR_ATR_COL = '__natr_atr'


def natr(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Normalized Average True Range (NATR).

    Args:
        data (pl.DataFrame): Klines dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        period (int): Number of periods for NATR calculation

    Returns:
        pl.DataFrame: The input data with a new column 'natr_{period}'
    '''

    if period < NATR_PERIOD_MIN or period > NATR_PERIOD_MAX:
        raise ValueError(f'period must be between {NATR_PERIOD_MIN} and {NATR_PERIOD_MAX}')

    out_col = f'{NATR_OUT_COL_PREFIX}{period}'
    prev_close = pl.col(close_col).shift(1)
    true_range_expr = (
        pl.max_horizontal(pl.col(high_col), prev_close)
        - pl.min_horizontal(pl.col(low_col), prev_close)
    ).alias(NATR_TR_COL)

    frame = data.with_columns(true_range_expr).with_columns(
        _atr_from_true_range_expr(NATR_TR_COL, period).alias(NATR_ATR_COL)
    )

    if period <= 1:
        return frame.with_columns(
            pl.when(pl.int_range(0, pl.len()) < 1)
            .then(None)
            .otherwise(pl.col(NATR_TR_COL))
            .alias(out_col)
        ).drop([NATR_TR_COL, NATR_ATR_COL])

    natr_expr = (
        pl.when(pl.col(NATR_ATR_COL).is_null())
        .then(None)
        .when(pl.col(close_col) == 0.0)
        .then(0.0)
        .otherwise((pl.col(NATR_ATR_COL) / pl.col(close_col)) * NATR_SCALE)
        .alias(out_col)
    )

    return frame.with_columns(natr_expr).drop([NATR_TR_COL, NATR_ATR_COL])
