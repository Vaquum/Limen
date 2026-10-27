import polars as pl
import limen._feature_constraints as _check


def momentum_weight(data: pl.DataFrame, period: int = 12, weight_multiplier: float = 0.5, base_weight: float = 0.5) -> pl.DataFrame:

    '''
    Compute momentum-based weighting factor from price change direction.

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Period for momentum calculation
        weight_multiplier (float): Multiplier for positive momentum
        base_weight (float): Base weight added to all observations

    Returns:
        pl.DataFrame: The input data with a new column 'momentum_weight'
    '''

    _check.integers(lower=1, period=period)
    _check.numbers(weight_multiplier=weight_multiplier, base_weight=base_weight)

    return data.with_columns([
        ((pl.col('close').pct_change(period) > 0).cast(pl.Float64) * weight_multiplier + base_weight)
        .alias('momentum_weight')
    ])
