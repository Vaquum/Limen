import polars as pl


def adosc(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: str = 'volume',
    fast_period: int = 3,
    slow_period: int = 10,
) -> pl.DataFrame:

    '''
    Compute Chaikin A/D Oscillator (ADOSC).

    Args:
        data (pl.DataFrame): Klines dataset with high/low/close/volume columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        volume_col (str): Column name for volume
        fast_period (int): Fast EMA period (>= 2)
        slow_period (int): Slow EMA period (>= 2)

    Returns:
        pl.DataFrame: The input data with a new column 'adosc_{fast_period}_{slow_period}'
    '''

    if fast_period < 2 or slow_period < 2:
        raise ValueError('fast_period and slow_period must be >= 2')

    out_col = f'adosc_{fast_period}_{slow_period}'
    lookback = max(fast_period, slow_period) - 1
    alpha_fast = 2.0 / (fast_period + 1.0)
    alpha_slow = 2.0 / (slow_period + 1.0)

    hl_range = pl.col(high_col) - pl.col(low_col)
    money_flow_volume = (
        pl.when(hl_range > 0.0)
        .then(
            (
                ((pl.col(close_col) - pl.col(low_col)) - (pl.col(high_col) - pl.col(close_col)))
                / hl_range
            ) * pl.col(volume_col).cast(pl.Float64)
        )
        .otherwise(0.0)
        .alias('__ad_mfv')
    )

    return (
        data
        .with_columns([money_flow_volume])
        .with_columns([pl.col('__ad_mfv').cum_sum().alias('__ad')])
        .with_columns([
            pl.col('__ad').ewm_mean(alpha=alpha_fast, adjust=False).alias('__ad_ema_fast'),
            pl.col('__ad').ewm_mean(alpha=alpha_slow, adjust=False).alias('__ad_ema_slow'),
        ])
        .with_columns([
            pl.when(pl.int_range(0, pl.len()) < lookback)
            .then(None)
            .otherwise(pl.col('__ad_ema_fast') - pl.col('__ad_ema_slow'))
            .alias(out_col)
        ])
        .drop(['__ad_mfv', '__ad', '__ad_ema_fast', '__ad_ema_slow'])
    )
