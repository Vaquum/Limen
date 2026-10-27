import polars as pl
import limen._feature_constraints as _check

DEFAULT_MOMENTUM_PERIODS = [12, 24, 48]


def momentum_periods(data: pl.DataFrame, periods: list[int] | None = None, price_col: str = 'close') -> pl.DataFrame:

    '''
    Compute momentum over multiple time periods.

    Args:
        data (pl.DataFrame): Dataset with price column
        periods (list): Non-negative periods for momentum calculation
        price_col (str): Name of the price column (default: 'close')

    Returns:
        pl.DataFrame: The input data with new columns 'momentum_{period}' for each period
    '''

    _check.names(price_col=price_col)

    periods = _check.integer_sequence('periods', DEFAULT_MOMENTUM_PERIODS if periods is None else periods, minimum=0)
    momentum_expressions = [
        pl.col(price_col).pct_change(period).alias(f'momentum_{period}') for period in periods
    ]

    return data.with_columns(momentum_expressions)
