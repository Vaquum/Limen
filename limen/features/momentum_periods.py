import polars as pl

DEFAULT_MOMENTUM_PERIODS = [12, 24, 48]


def momentum_periods(data: pl.DataFrame, periods: list | None = None, price_col: str = 'close') -> pl.DataFrame:

    '''
    Compute momentum over multiple time periods.

    Args:
        data (pl.DataFrame): Dataset with price column
        periods (list): List of periods for momentum calculation
        price_col (str): Name of the price column (default: 'close')

    Returns:
        pl.DataFrame: The input data with new columns 'momentum_{period}' for each period
    '''

    if periods is None:
        periods = DEFAULT_MOMENTUM_PERIODS
    momentum_expressions = []
    for period in periods:
        momentum_expressions.append(
            pl.col(price_col).pct_change(period).alias(f'momentum_{period}')
        )

    return data.with_columns(momentum_expressions)
