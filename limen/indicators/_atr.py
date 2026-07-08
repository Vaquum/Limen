import polars as pl


def _wilder_atr_from_tr_values(tr_values: list[float | None], period: int) -> list[float | None]:
    length = len(tr_values)
    out: list[float | None] = [None] * length

    if length <= 1:
        return out

    if period <= 1:
        for idx in range(1, length):
            out[idx] = tr_values[idx]
        return out

    if length <= period:
        return out

    seed_sum = 0.0
    for idx in range(1, period + 1):
        value = tr_values[idx]
        if value is None:
            return out
        seed_sum += value

    prev_atr = seed_sum / period
    out[period] = prev_atr

    for idx in range(period + 1, length):
        value = tr_values[idx]
        if value is None:
            out[idx] = None
            continue
        prev_atr = ((prev_atr * (period - 1)) + value) / period
        out[idx] = prev_atr

    return out


def atr_from_true_range_expr(tr_col: str, period: int) -> pl.Expr:
    return pl.col(tr_col).map_batches(
        lambda s: pl.Series(
            _wilder_atr_from_tr_values(
                s.to_list(),
                period,
            )
        ),
        return_dtype=pl.Float64,
    )
