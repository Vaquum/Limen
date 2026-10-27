import polars as pl
import limen._feature_constraints as _check


def dynamic_target(data: pl.DataFrame,
                  base_min_breakout: float,
                  target_volatility_multiplier: float,
                  clip_lower_mult: float = 0.6,
                  clip_upper_mult: float = 1.4) -> pl.DataFrame:

    '''
    Compute dynamic target levels based on volatility conditions and regime.

    Args:
        data (pl.DataFrame): Klines dataset with 'volatility_measure', 'regime_multiplier' columns
        base_min_breakout (float): Base minimum breakout threshold
        target_volatility_multiplier (float): Multiplier for volatility-adjusted targets
        clip_lower_mult (float): Lower bound multiplier for clipping
        clip_upper_mult (float): Upper bound multiplier for clipping

    Returns:
        pl.DataFrame: The input data with a new column 'dynamic_target'
    '''

    _check.numbers(
        base_min_breakout=base_min_breakout,
        target_volatility_multiplier=target_volatility_multiplier,
        clip_lower_mult=clip_lower_mult,
        clip_upper_mult=clip_upper_mult,
    )
    _check.ordered('clip_lower_mult', clip_lower_mult, 'clip_upper_mult', clip_upper_mult)

    return data.with_columns([
        (pl.col('volatility_measure') * target_volatility_multiplier * pl.col('regime_multiplier'))
        .clip(base_min_breakout * clip_lower_mult, base_min_breakout * clip_upper_mult)
        .alias('dynamic_target')
    ])
