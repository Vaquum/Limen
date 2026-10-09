import polars as pl

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

    if periods is None:
        periods = DEFAULT_MOMENTUM_PERIODS
    if any(period < 0 for period in periods):
        raise ValueError('periods must be non-negative to avoid future prices')
    momentum_expressions = [
        pl.col(price_col).pct_change(period).alias(f'momentum_{period}') for period in periods
    ]

    return data.with_columns(momentum_expressions)
