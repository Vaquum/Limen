import polars as pl


def ema_breakout(data: pl.DataFrame,
                 target_col: str,
                 ema_span: int = 30,
                 breakout_delta: float = 0.2,
                 breakout_horizon: int = 3) -> pl.DataFrame:

    '''
    Compute EMA breakout indicator based on price deviation from EMA.

    Args:
        data (pl.DataFrame): Klines dataset with price columns
        target_col (str): Column name to analyze for breakouts
        ema_span (int): Period for EMA calculation
        breakout_delta (float): Threshold for breakout detection
        breakout_horizon (int): Lookback period for breakout validation
        
    Returns:
        pl.DataFrame: The input data with a new column 'breakout_ema'
    '''

    alpha = 2.0 / (ema_span + 1)

    label_expr = (
        pl.col(target_col).shift(-breakout_horizon)
        > pl.col(target_col).ewm_mean(alpha=alpha, adjust=False) * (1 + breakout_delta)
    ).cast(pl.UInt8)

    return (
        data
        .with_columns(label_expr.alias('breakout_ema'))
        .filter(pl.col('breakout_ema').is_not_null())
    )