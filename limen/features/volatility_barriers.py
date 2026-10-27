import math

import polars as pl
import limen._feature_constraints as _check

__all__ = ['volatility_barriers']
_MIN_OBSERVATIONS = 2
_BPS = 10000.0


def volatility_barriers(data: pl.DataFrame, *, span: int = 24, regime_span: int = 168,
                        min_periods: int = 24, take_profit_multiple: float = 2.0,
                        stop_loss_multiple: float = 1.0, minimum_bps: float = 10.0,
                        maximum_bps: float = 2000.0, low_regime_ratio: float = 0.75,
                        high_regime_ratio: float = 1.25, low_multiplier: float = 0.8,
                        normal_multiplier: float = 1.0, high_multiplier: float = 1.2) -> pl.DataFrame:
    """Produce entry distances from strictly prior simple-return EWMA volatility."""
    _check.integers(lower=1, span=span, regime_span=regime_span, min_periods=min_periods)
    _check.numbers(take_profit_multiple=take_profit_multiple, stop_loss_multiple=stop_loss_multiple, minimum_bps=minimum_bps, maximum_bps=maximum_bps, low_regime_ratio=low_regime_ratio, high_regime_ratio=high_regime_ratio, low_multiplier=low_multiplier, normal_multiplier=normal_multiplier, high_multiplier=high_multiplier)

    if any(type(value) is not int for value in (span, regime_span, min_periods)) or not _MIN_OBSERVATIONS <= min_periods <= span <= regime_span:
        raise ValueError('Require integer 2 <= min_periods <= span <= regime_span')
    positive = (take_profit_multiple, stop_loss_multiple, minimum_bps, maximum_bps,
                low_regime_ratio, high_regime_ratio, low_multiplier, normal_multiplier, high_multiplier)
    if not all(_positive(value) for value in positive):
        raise ValueError('Barrier coefficients must be finite and positive')
    if not minimum_bps <= maximum_bps < _BPS or low_regime_ratio >= high_regime_ratio:
        raise ValueError('Require ordered barrier clips below 10000 bps and regime thresholds')
    closes = data['close']
    if closes.null_count() or not closes.is_finite().all() or (closes <= 0).any():
        raise ValueError('Volatility barriers require finite positive recorded closes')
    prior = (closes / closes.shift(1) - 1).shift(1)
    fast = prior.ewm_std(span=span, adjust=True, bias=False, min_samples=min_periods)
    slow = prior.ewm_std(span=regime_span, adjust=True, bias=False, min_samples=min_periods)
    ratio = pl.when((pl.col('_fast') == 0) & (pl.col('_slow') == 0)).then(1.0).otherwise(pl.col('_fast') / pl.col('_slow'))
    temporary = pl.DataFrame({'_fast': fast, '_slow': slow}).with_columns(ratio.alias('_ratio'))
    multiplier = pl.when(pl.col('_ratio') < low_regime_ratio).then(low_multiplier).when(pl.col('_ratio') > high_regime_ratio).then(high_multiplier).otherwise(normal_multiplier)
    distances = temporary.select(
        (_BPS * pl.col('_fast') * multiplier * take_profit_multiple).clip(minimum_bps, maximum_bps).alias('volatility_take_profit_bps'),
        (_BPS * pl.col('_fast') * multiplier * stop_loss_multiple).clip(minimum_bps, maximum_bps).alias('volatility_stop_loss_bps'),
    )
    ready = pl.Series('__barrier_ready__', [index >= min_periods + 1 for index in range(data.height)])
    return data.hstack(distances).with_columns(ready)


def _positive(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
