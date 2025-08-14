import polars as pl


def rolling_volatility(data: pl.DataFrame,
                       column: str = 'close',
                       window: int = 12) -> pl.DataFrame:
    '''
    Compute rolling volatility (standard deviation) over a specified period.

    Args:
        data (pl.DataFrame): Klines dataset with price/returns column
        column (str): Column name to calculate volatility on (typically returns)
        window (int): Number of periods for rolling window calculation

    Returns:
        pl.DataFrame: The input data with a new column '{column}_volatility_{window}'
    '''

    return data.with_columns([
        pl.col(column).rolling_std(window_size=window).alias(
            f"{column}_volatility_{window}")
    ])
