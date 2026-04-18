import polars as pl


def yang_zhang_volatility(
    data: pl.DataFrame,
    window: int = 20,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Yang-Zhang volatility over a rolling window.

    Args:
        data (pl.DataFrame): Klines dataset with open, high, low, and close price columns
        window (int): Number of periods used for the rolling estimator; must be greater than 1
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'yang_zhang_volatility'
    '''

    if window <= 1:
        raise ValueError('window must be greater than 1 for yang_zhang_volatility')

    overnight_return = pl.col(open_col).log() - pl.col(close_col).shift(1).log()
    open_close_return = pl.col(close_col).log() - pl.col(open_col).log()
    high_open = pl.col(high_col).log() - pl.col(open_col).log()
    high_close = pl.col(high_col).log() - pl.col(close_col).log()
    low_open = pl.col(low_col).log() - pl.col(open_col).log()
    low_close = pl.col(low_col).log() - pl.col(close_col).log()
    rs_variance = (high_open * high_close) + (low_open * low_close)

    k = 0.34 / (1.34 + ((window + 1.0) / (window - 1.0)))

    return (
        data
        .with_columns([
            overnight_return.alias('_yz_overnight_return'),
            open_close_return.alias('_yz_open_close_return'),
            rs_variance.alias('_yz_rs_variance'),
        ])
        .with_columns(
            (
                pl.col('_yz_overnight_return').rolling_var(window_size=window)
                + (k * pl.col('_yz_open_close_return').rolling_var(window_size=window))
                + ((1.0 - k) * pl.col('_yz_rs_variance').rolling_mean(window_size=window))
            )
            .clip(lower_bound=0.0)
            .sqrt()
            .alias('yang_zhang_volatility')
        )
        .drop([
            '_yz_overnight_return',
            '_yz_open_close_return',
            '_yz_rs_variance',
        ])
    )
