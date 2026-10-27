import polars as pl
import limen._feature_constraints as _check


def dynamic_stop_loss(data: pl.DataFrame,
                     base_stop_loss: float,
                     stop_volatility_multiplier: float,
                     clip_lower_mult: float = 0.7,
                     clip_upper_mult: float = 1.4) -> pl.DataFrame:

    '''
    Compute dynamic stop loss levels based on volatility conditions and regime.

    Args:
        data (pl.DataFrame): Klines dataset with 'volatility_measure', 'regime_multiplier' columns
        base_stop_loss (float): Base stop loss threshold
        stop_volatility_multiplier (float): Multiplier for volatility-adjusted stops
        clip_lower_mult (float): Lower bound multiplier for clipping
        clip_upper_mult (float): Upper bound multiplier for clipping

    Returns:
        pl.DataFrame: The input data with a new column 'dynamic_stop_loss'
    '''

    _check.numbers(
        base_stop_loss=base_stop_loss,
        stop_volatility_multiplier=stop_volatility_multiplier,
        clip_lower_mult=clip_lower_mult,
        clip_upper_mult=clip_upper_mult,
    )
    _check.ordered('clip_lower_mult', clip_lower_mult, 'clip_upper_mult', clip_upper_mult)

    return data.with_columns([
        (pl.col('volatility_measure') * stop_volatility_multiplier * pl.col('regime_multiplier'))
        .clip(base_stop_loss * clip_lower_mult, base_stop_loss * clip_upper_mult)
        .alias('dynamic_stop_loss')
    ])
