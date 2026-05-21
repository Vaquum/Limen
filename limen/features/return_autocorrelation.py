import polars as pl


def return_autocorrelation(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
    output_col: str = 'return_autocorrelation',
) -> pl.DataFrame:

    '''
    Compute rolling autocorrelation between returns and one-bar lagged returns.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        window (int): Number of periods in the rolling correlation
        close_col (str): Column name for close prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the return-autocorrelation column appended
    '''

    returns = pl.col(close_col).pct_change()

    return data.with_columns(
        pl.rolling_corr(returns, returns.shift(1), window_size=window).alias(output_col)
    )
