import polars as pl


def add_breakout_ema(data: pl.DataFrame,
                     target_col: str,
                     ema_span: int = 30,
                     breakout_delta: float = 0.2,
                     breakout_horizon: int = 3) -> pl.DataFrame:

    '''
    Compute breakout labels based on EMA crossover patterns.

    Args:
        data (pl.DataFrame): Klines dataset with specified target column
        target_col (str): Column name to compute EMA breakout labels for
        ema_span (int): Number of periods for EMA calculation
        breakout_delta (float): Percentage threshold for breakout detection
        breakout_horizon (int): Number of periods to look ahead for breakout confirmation

    Returns:
        pl.DataFrame: The input data with a new column 'breakout_ema'
    '''

    return (data
        .with_columns([
            pl.col(target_col).ewm_mean(span=ema_span, adjust=False).alias('_ema'),
            pl.col(target_col).shift(-breakout_horizon).alias('_future_val')
        ])
        .with_columns([
            (pl.col('_future_val') > pl.col('_ema') * (1 + breakout_delta))
            .cast(pl.Int32)
            .fill_null(0)
            .alias('breakout_ema')
        ])
        .drop(['_ema', '_future_val'])
        .slice(0, data.height - breakout_horizon)
    )
