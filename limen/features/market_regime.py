import polars as pl
from limen.indicators.sma import sma
from limen.features.trend_strength import trend_strength
from limen.features.volume_regime import volume_regime
import limen._feature_constraints as _check


def market_regime(df: pl.DataFrame, lookback: int = 48, short_sma: int = 20, long_sma: int = 50) -> pl.DataFrame:

    '''
    Add close-average, volume-regime and market-favorability features.

    sma_20 and sma_50 bind the selected short/long close averages, including equal
    or swapped periods; volume_sma binds the selected volume average. Fixed aliases
    take precedence over close canonicals, which take precedence over the volume
    canonical. Nonconflicting sma_<period> outputs retain their existing positions.
    Incoming fixed aliases raise DuplicateError. SMA warmup remains NaN; derived
    trend, volatility, volume-regime and favorability warmup remains null.
    '''

    _check.integers(lower=1, lookback=lookback, short_sma=short_sma, long_sma=long_sma)

    conflicts = {'sma_20', 'sma_50', 'volume_sma'} & set(df.collect_schema().names())
    if conflicts:
        raise pl.exceptions.DuplicateError(f'market_regime output columns already exist: {sorted(conflicts)}')
    df = sma(df, 'close', short_sma)
    df = sma(df, 'close', long_sma)
    df = trend_strength(df, short_sma, long_sma)

    df = df.with_columns([
        pl.col('close').pct_change().alias('returns_temp')
    ])

    df = df.with_columns([
        (pl.col('returns_temp').rolling_std(window_size=12) /
         pl.col('returns_temp').rolling_std(window_size=lookback)).alias('volatility_ratio')
    ])

    df = sma(df, 'volume', lookback)
    df = volume_regime(df, lookback)

    TREND_STRENGTH_THRESHOLD = -0.001
    VOLATILITY_RATIO_THRESHOLD = 2.0
    VOLUME_REGIME_THRESHOLD = 0.7
    SCORE_NORMALIZER = 3.0

    df = df.with_columns([
        (((pl.col('trend_strength') > TREND_STRENGTH_THRESHOLD).cast(pl.Int32) +
          (pl.col('volatility_ratio') < VOLATILITY_RATIO_THRESHOLD).cast(pl.Int32) +
          (pl.col('volume_regime') > VOLUME_REGIME_THRESHOLD).cast(pl.Int32)) / SCORE_NORMALIZER)
        .alias('market_favorable')
    ])

    df = df.drop('returns_temp')
    aliases = {'sma_20': f'close_sma_{short_sma}', 'sma_50': f'close_sma_{long_sma}',
               'volume_sma': f'volume_sma_{lookback}'}
    sources = {f'sma_{period}': f'close_sma_{period}' for period in (short_sma, long_sma)} | aliases
    names = dict.fromkeys(
        target for column in df.collect_schema().names()
        for target in ([alias for alias, source in aliases.items() if source == column] or [column])
    )
    return df.select(pl.col(sources.get(name, name)).alias(name) for name in names)
