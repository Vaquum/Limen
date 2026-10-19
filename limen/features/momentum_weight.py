from limen._parallel_contract import attach as _parallel
import polars as pl


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

    return data.with_columns([
        ((pl.col('close').pct_change(period) > 0).cast(pl.Float64) * weight_multiplier + base_weight)
        .alias('momentum_weight')
    ])

_parallel(momentum_weight, backend_imports=('polars',))
