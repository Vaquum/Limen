import polars as pl


def rogers_satchell_volatility(
    data: pl.DataFrame,
    window: int = 20,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Rogers-Satchell range-based volatility over a rolling window.

    Args:
        data (pl.DataFrame): Klines dataset with open, high, low, and close price columns
        window (int): Number of periods used for the rolling estimator
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'rogers_satchell_volatility'
    '''

    high_open = pl.col(high_col).log() - pl.col(open_col).log()
    high_close = pl.col(high_col).log() - pl.col(close_col).log()
    low_open = pl.col(low_col).log() - pl.col(open_col).log()
    low_close = pl.col(low_col).log() - pl.col(close_col).log()

    return (
        data
        .with_columns(((high_open * high_close) + (low_open * low_close)).alias('_rogers_satchell_variance'))
        .with_columns(
            (
                pl.col('_rogers_satchell_variance')
                .rolling_mean(window_size=window)
                .clip(lower_bound=0.0)
                .sqrt()
            ).alias('rogers_satchell_volatility')
        )
        .drop('_rogers_satchell_variance')
    )
