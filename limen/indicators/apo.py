import polars as pl

from limen.indicators.ma import ma


CMP_N_100000 = 100000
CMP_N_2 = 2
CMP_N_8 = 8

def apo(
    data: pl.DataFrame,
    price_col: str = 'close',
    fast_period: int = 12,
    slow_period: int = 26,
    ma_type: int = 0,
) -> pl.DataFrame:

    '''
    Compute Absolute Price Oscillator (APO).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        fast_period (int): Number of periods for fast MA (2..100000)
        slow_period (int): Number of periods for slow MA (2..100000)
        ma_type (int): TA-Lib MA type (0..8)

    Returns:
        pl.DataFrame: The input data with a new column 'apo_{fast_period}_{slow_period}_{ma_type}'
    '''

    if fast_period < CMP_N_2 or fast_period > CMP_N_100000:
        raise ValueError('fast_period must be between 2 and 100000')
    if slow_period < CMP_N_2 or slow_period > CMP_N_100000:
        raise ValueError('slow_period must be between 2 and 100000')
    if ma_type < 0 or ma_type > CMP_N_8:
        raise ValueError('ma_type must be between 0 and 8')

    effective_fast = fast_period
    effective_slow = slow_period
    if effective_slow < effective_fast:
        effective_fast, effective_slow = effective_slow, effective_fast

    out_col = f'apo_{fast_period}_{slow_period}_{ma_type}'
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

    return frame.with_columns(
        (pl.col(fast_col) - pl.col(slow_col)).alias(out_col)
    ).drop([fast_col, slow_col])
