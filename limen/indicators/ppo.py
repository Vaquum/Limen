import polars as pl
from limen.indicators.ma import ma

CMP_N_100000 = 100000
CMP_N_2 = 2
CMP_N_8 = 8

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
        slow_period (int): Number of periods for slow MA (2..100000, must be greater than fast_period)
        ma_type (int): TA-Lib MA type (0..8)

    Returns:
        pl.DataFrame: The input data with a new column 'ppo_{fast_period}_{slow_period}_{ma_type}'
    '''

    if fast_period < CMP_N_2 or fast_period > CMP_N_100000:
        raise ValueError('ppo fast_period must be between 2 and 100000')
    if slow_period < CMP_N_2 or slow_period > CMP_N_100000:
        raise ValueError('ppo slow_period must be between 2 and 100000')
    if ma_type < 0 or ma_type > CMP_N_8:
        raise ValueError('ppo ma_type must be between 0 and 8')
    if slow_period <= fast_period:
        raise ValueError('ppo slow_period must be greater than fast_period')

    out_col = f"ppo_{fast_period}_{slow_period}_{ma_type}"
    frame = data

    fast_col = f"ma_{fast_period}_{ma_type}"
    slow_col = f"ma_{slow_period}_{ma_type}"

    frame = ma(
        frame,
        price_col=price_col,
        period=fast_period,
        ma_type=ma_type,
    )
    frame = ma(
        frame,
        price_col=price_col,
        period=slow_period,
        ma_type=ma_type,
    )

    missing_fast = pl.col(fast_col).is_null() | pl.col(fast_col).is_nan()
    missing_slow = pl.col(slow_col).is_null() | pl.col(slow_col).is_nan()
    ppo_expr = (
        pl.when(missing_fast | missing_slow)
        .then(None)
        .when(pl.col(slow_col).abs() >= TA_EPSILON)
        .then(((pl.col(fast_col) - pl.col(slow_col)) / pl.col(slow_col)) * 100.0)
        .otherwise(0.0)
        .alias(out_col)
    )

    return frame.with_columns(ppo_expr).drop([fast_col, slow_col])
