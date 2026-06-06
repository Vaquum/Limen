import polars as pl

from limen.indicators.dema import _dema_from_values
from limen.indicators.ema import _ema_from_values
from limen.indicators.kama import _kama_from_values
from limen.indicators.mama import _mama_from_values
from limen.indicators.sma import _sma_from_values
from limen.indicators.t3 import _t3_from_values
from limen.indicators.tema import _tema_from_values
from limen.indicators.trima import _trima_from_values
from limen.indicators.wma import _wma_from_values


CMP_N_100000 = 100000
CMP_N_2 = 2
CMP_N_3 = 3
CMP_N_4 = 4
CMP_N_5 = 5
CMP_N_6 = 6
CMP_N_7 = 7
CMP_N_8 = 8

def ma(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 30,
    ma_type: int = 0,
) -> pl.DataFrame:

    '''
    Compute Moving Average with selectable MA type (TA-Lib MA).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods
        ma_type (int): TA-Lib MA type

    Returns:
        pl.DataFrame: The input data with a new column 'ma_{period}_{ma_type}'
    '''

    if period < 1 or period > CMP_N_100000:
        raise ValueError('ma period must be between 1 and 100000')
    if ma_type < 0 or ma_type > CMP_N_8:
        raise ValueError('ma ma_type must be between 0 and 8')

    out_col = f"ma_{period}_{ma_type}"
    frame = data
    if period == 1:
        return frame.with_columns(pl.col(price_col).alias(out_col))

    if ma_type == 0:
        expr = pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _sma_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        )
    elif ma_type == 1:
        expr = pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _ema_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        )
    elif ma_type == CMP_N_2:
        expr = pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _wma_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        )
    elif ma_type == CMP_N_3:
        expr = pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _dema_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        )
    elif ma_type == CMP_N_4:
        expr = pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _tema_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        )
    elif ma_type == CMP_N_5:
        expr = pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _trima_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        )
    elif ma_type == CMP_N_6:
        expr = pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _kama_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        )
    elif ma_type == CMP_N_7:
        expr = pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _mama_from_values(
                    s.to_numpy().astype(float, copy=False),
                    0.5,
                    0.05,
                )[0]
            ),
            return_dtype=pl.Float64,
        )
    elif ma_type == CMP_N_8:
        expr = pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _t3_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                    0.7,
                )
            ),
            return_dtype=pl.Float64,
        )

    return frame.with_columns(expr.alias(out_col))
