import polars as pl


def bbands(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 5,
    nb_dev_up: float = 2.0,
    nb_dev_dn: float = 2.0,
    ma_type: int = 0,
) -> pl.DataFrame:

    '''
    Compute Bollinger Bands (upper/middle/lower).

    Args:
        data (pl.DataFrame): Dataset with price column
        price_col (str): Column name for input price
        period (int): Rolling window length (>= 2)
        nb_dev_up (float): Upper-band deviation multiplier
        nb_dev_dn (float): Lower-band deviation multiplier
        ma_type (int): TA-Lib MA type (0 = SMA)

    Returns:
        pl.DataFrame: Input data with 'bbands_upper', 'bbands_middle', 'bbands_lower'
    '''

    if period < 2:
        raise ValueError('period must be >= 2')
    if ma_type != 0:
        raise NotImplementedError('Only ma_type=0 (SMA) is currently supported')

    middle = pl.col(price_col).rolling_mean(window_size=period)
    std = pl.col(price_col).rolling_std(window_size=period, ddof=0)

    return data.with_columns([
        (middle + (std * nb_dev_up)).alias('bbands_upper'),
        middle.alias('bbands_middle'),
        (middle - (std * nb_dev_dn)).alias('bbands_lower'),
    ])
