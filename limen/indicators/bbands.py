import polars as pl

from limen.indicators.ma import ma

BBANDS_PERIOD_MIN = 2
BBANDS_PERIOD_MAX = 100000
BBANDS_NBDEV_MIN = -3e37
BBANDS_NBDEV_MAX = 3e37
BBANDS_MATYPE_MIN = 0
BBANDS_MATYPE_MAX = 8
BBANDS_STDDEV_FLOOR = 0.0


def _bbands_stddev_expr(
    price_col: str,
    middle_col: str,
    period: int,
) -> pl.Expr:
    mean_sq = (pl.col(price_col) * pl.col(price_col)).rolling_mean(window_size=period)
    variance = mean_sq - (pl.col(middle_col) * pl.col(middle_col))

    return (
        pl.when(pl.col(middle_col).is_null())
        .then(None)
        .otherwise(
            pl.when(variance > BBANDS_STDDEV_FLOOR)
            .then(variance.sqrt())
            .otherwise(BBANDS_STDDEV_FLOOR)
        )
    )


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
        period (int): Rolling window length
        nb_dev_up (float): Upper-band deviation multiplier
        nb_dev_dn (float): Lower-band deviation multiplier
        ma_type (int): TA-Lib MA type

    Returns:
        pl.DataFrame: Input data with 'bbands_upper', 'bbands_middle', 'bbands_lower'
    '''

    if period < BBANDS_PERIOD_MIN or period > BBANDS_PERIOD_MAX:
        raise ValueError(f"period must be between {BBANDS_PERIOD_MIN} and {BBANDS_PERIOD_MAX}")
    if nb_dev_up < BBANDS_NBDEV_MIN or nb_dev_up > BBANDS_NBDEV_MAX:
        raise ValueError(f"nb_dev_up must be between {BBANDS_NBDEV_MIN} and {BBANDS_NBDEV_MAX}")
    if nb_dev_dn < BBANDS_NBDEV_MIN or nb_dev_dn > BBANDS_NBDEV_MAX:
        raise ValueError(f"nb_dev_dn must be between {BBANDS_NBDEV_MIN} and {BBANDS_NBDEV_MAX}")
    if ma_type < BBANDS_MATYPE_MIN or ma_type > BBANDS_MATYPE_MAX:
        raise ValueError(f"ma_type must be between {BBANDS_MATYPE_MIN} and {BBANDS_MATYPE_MAX}")

    middle_col = f"ma_{period}_{ma_type}"
    frame = ma(
        data,
        price_col=price_col,
        period=period,
        ma_type=ma_type,
    )

    stddev_expr = _bbands_stddev_expr(
        price_col=price_col,
        middle_col=middle_col,
        period=period,
    ).alias('__bbands_stddev')
    frame = frame.with_columns(stddev_expr)

    if nb_dev_up == nb_dev_dn:
        if nb_dev_up == 1.0:
            upper_expr = pl.col(middle_col) + pl.col('__bbands_stddev')
            lower_expr = pl.col(middle_col) - pl.col('__bbands_stddev')
        else:
            scaled_stddev = pl.col('__bbands_stddev') * nb_dev_up
            upper_expr = pl.col(middle_col) + scaled_stddev
            lower_expr = pl.col(middle_col) - scaled_stddev
    elif nb_dev_up == 1.0:
        upper_expr = pl.col(middle_col) + pl.col('__bbands_stddev')
        lower_expr = pl.col(middle_col) - (pl.col('__bbands_stddev') * nb_dev_dn)
    elif nb_dev_dn == 1.0:
        upper_expr = pl.col(middle_col) + (pl.col('__bbands_stddev') * nb_dev_up)
        lower_expr = pl.col(middle_col) - pl.col('__bbands_stddev')
    else:
        upper_expr = pl.col(middle_col) + (pl.col('__bbands_stddev') * nb_dev_up)
        lower_expr = pl.col(middle_col) - (pl.col('__bbands_stddev') * nb_dev_dn)

    return frame.with_columns(
        [
            upper_expr.alias('bbands_upper'),
            pl.col(middle_col).alias('bbands_middle'),
            lower_expr.alias('bbands_lower'),
        ]
    ).drop([middle_col, '__bbands_stddev'])
