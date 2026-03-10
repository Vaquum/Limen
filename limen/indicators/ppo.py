import polars as pl
from limen.indicators.ma import ma

TA_EPSILON = 1e-14


def ppo(
    data: pl.DataFrame,
    price_col: str = 'close',
    fast_period: int = 12,
    slow_period: int = 26,
    ma_type: int = 0,
) -> pl.DataFrame:

    '''
    Compute Percentage Price Oscillator (PPO).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        fast_period (int): Number of periods for fast MA (2..100000)
        slow_period (int): Number of periods for slow MA (2..100000)
        ma_type (int): TA-Lib MA type (0..8)

    Returns:
        pl.DataFrame: The input data with a new column 'ppo_{fast_period}_{slow_period}_{ma_type}'
    '''

    if fast_period < 2 or fast_period > 100000:
        raise ValueError('fast_period must be between 2 and 100000')
    if slow_period < 2 or slow_period > 100000:
        raise ValueError('slow_period must be between 2 and 100000')
    if ma_type < 0 or ma_type > 8:
        raise ValueError('ma_type must be between 0 and 8')

    effective_fast = fast_period
    effective_slow = slow_period
    if effective_slow < effective_fast:
        effective_fast, effective_slow = effective_slow, effective_fast

    out_col = f'ppo_{fast_period}_{slow_period}_{ma_type}'
    frame = data

    fast_col = f'ma_{effective_fast}_{ma_type}'
    slow_col = f'ma_{effective_slow}_{ma_type}'

    frame = ma(
        frame,
        price_col=price_col,
        period=effective_fast,
        ma_type=ma_type,
    )
    frame = ma(
        frame,
        price_col=price_col,
        period=effective_slow,
        ma_type=ma_type,
    )

    ppo_expr = (
        pl.when(pl.col(fast_col).is_null() | pl.col(slow_col).is_null())
        .then(None)
        .when(pl.col(slow_col).abs() >= TA_EPSILON)
        .then(((pl.col(fast_col) - pl.col(slow_col)) / pl.col(slow_col)) * 100.0)
        .otherwise(0.0)
        .alias(out_col)
    )

    return frame.with_columns(ppo_expr).drop([fast_col, slow_col])
