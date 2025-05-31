import polars as pl

def ema_breakout(
    data: pl.DataFrame,
    target_col: str,
    ema_span: int = 30,
    breakout_delta: float = 0.2,
    breakout_horizon: int = 3
) -> pl.DataFrame:
    """
    Append a 0/1 “breakout_ema” column to `data`:
      – compute EMA(span=ema_span) of `target_col`
      – look ahead `breakout_horizon` rows for a price > EMA * (1 + breakout_delta)
      – label = 1 if true, else 0
      – drop the final `breakout_horizon` rows (which become null after shift).
    """
    alpha = 2.0 / (ema_span + 1)
    # build the “breakout_ema” expression
    label_expr = (
        pl.col(target_col).shift(-breakout_horizon)
        > pl.col(target_col).ewm_mean(alpha=alpha, adjust=False) * (1 + breakout_delta)
    ).cast(pl.UInt8)

    return (
        data
        .with_columns(label_expr.alias("breakout_ema"))
        .filter(pl.col("breakout_ema").is_not_null())
    )