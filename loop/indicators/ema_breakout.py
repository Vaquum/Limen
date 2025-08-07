import polars as pl


def ema_breakout(data: pl.DataFrame,
                 target_col: str,
                 ema_span: int = 30,
                 breakout_delta: float = 0.2,
                 breakout_horizon: int = 3) -> pl.DataFrame:

    '''
    Calculate the EMA breakout indicator.

    Args:
        data (pl.DataFrame): The input data.
        target_col (str): The target column.
        ema_span (int): The EMA span.
        breakout_delta (float): The breakout delta.
        breakout_horizon (int): The breakout horizon.
        
    Returns:
        pl.DataFrame: The input data with the EMA breakout indicator.
    '''

    alpha = 2.0 / (ema_span + 1)

    label_expr = (
        pl.col(target_col).shift(-breakout_horizon)
        > pl.col(target_col).ewm_mean(alpha=alpha, adjust=False) * (1 + breakout_delta)
    ).cast(pl.UInt8)

    return (
        data
        .with_columns(label_expr.alias("breakout_ema"))
        .filter(pl.col("breakout_ema").is_not_null())
    )
