import math
import numbers

import polars as pl


OUTPUT_COLUMN = 'dollar_bar_crash_reversal_position'

_REQUIRED_COLUMNS = (
    'datetime',
    'open',
    'close',
    'liquidity_sum',
    'maker_liquidity',
)
_REFERENCE_LOOKBACK_HOURS = 4
_ROBUST_WINDOW = '30d'
_ROBUST_MIN_SAMPLES = 100
_ROBUST_SCALE = 1.4826
_ROBUST_EPSILON = 1e-9
_BPS_PER_UNIT = 10_000.0

_REFERENCE_TIME_COLUMN = '_dcr_reference_time'
_REFERENCE_DATETIME_COLUMN = '_dcr_reference_datetime'
_REFERENCE_OPEN_COLUMN = '_dcr_reference_open'
_STRUCTURAL_CORE_COLUMN = '_dcr_structural_core'
_MAKER_FLOW_COLUMN = '_dcr_maker_flow'
_FLOW_MEDIAN_COLUMN = '_dcr_flow_median'
_FLOW_DEVIATION_COLUMN = '_dcr_flow_deviation'
_FLOW_MAD_COLUMN = '_dcr_flow_mad'
_FLOW_Z_COLUMN = '_dcr_flow_z'
_MOMENTUM_COLUMN = '_dcr_momentum_bps'
_TRIGGER_COLUMN = '_dcr_trigger'

_INTERNAL_COLUMNS = (
    _REFERENCE_TIME_COLUMN,
    _REFERENCE_DATETIME_COLUMN,
    _REFERENCE_OPEN_COLUMN,
    _STRUCTURAL_CORE_COLUMN,
    _MAKER_FLOW_COLUMN,
    _FLOW_MEDIAN_COLUMN,
    _FLOW_DEVIATION_COLUMN,
    _FLOW_MAD_COLUMN,
    _FLOW_Z_COLUMN,
    _MOMENTUM_COLUMN,
    _TRIGGER_COLUMN,
)


def _validate_threshold(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f'dollar_bar_crash_reversal {name} must be a finite real number'
        )
    if not math.isfinite(value):
        raise ValueError(
            f'dollar_bar_crash_reversal {name} must be a finite real number'
        )
    return float(value)


def _validate_hold_minutes(hold_minutes: int) -> int:
    if isinstance(hold_minutes, bool) or not isinstance(
        hold_minutes, numbers.Integral
    ):
        raise TypeError(
            'dollar_bar_crash_reversal hold_minutes must be a positive integer'
        )
    if hold_minutes <= 0:
        raise ValueError(
            'dollar_bar_crash_reversal hold_minutes must be a positive integer'
        )
    return int(hold_minutes)


def _validate_data(data: object) -> pl.DataFrame | pl.LazyFrame:
    if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(
            'dollar_bar_crash_reversal data must be a DataFrame or LazyFrame'
        )
    return data


def dollar_bar_crash_reversal(
    data: pl.DataFrame | pl.LazyFrame,
    momentum_threshold_bps: float,
    flow_z_threshold: float,
    hold_minutes: int,
) -> pl.DataFrame | pl.LazyFrame:
    '''Build the 15M-dollar-bar crash-reversal long/flat position signal.

    Momentum compares each close with the latest open at or before four hours
    earlier. Maker flow is standardized against prior-only 30-day rolling
    medians and median absolute deviations. A trigger remains active for the
    requested wall-clock hold window.
    '''

    validated_data = _validate_data(data)
    schema = validated_data.collect_schema()
    missing = [column for column in _REQUIRED_COLUMNS if column not in schema]
    if missing:
        raise ValueError(
            'dollar_bar_crash_reversal missing required columns: '
            + ', '.join(missing)
        )
    if not isinstance(schema['datetime'], pl.Datetime):
        raise TypeError(
            'dollar_bar_crash_reversal datetime must be a Datetime column'
        )

    momentum_threshold = _validate_threshold(
        momentum_threshold_bps,
        'momentum_threshold_bps',
    )
    flow_threshold = _validate_threshold(flow_z_threshold, 'flow_z_threshold')
    hold = _validate_hold_minutes(hold_minutes)

    eager = isinstance(validated_data, pl.DataFrame)
    lazy = validated_data.lazy() if eager else validated_data
    reference = lazy.select(
        pl.col('datetime').alias(_REFERENCE_DATETIME_COLUMN),
        pl.col('open').alias(_REFERENCE_OPEN_COLUMN),
    )

    result = (
        lazy
        .with_columns(
            (
                pl.col('datetime')
                - pl.duration(hours=_REFERENCE_LOOKBACK_HOURS)
            ).alias(_REFERENCE_TIME_COLUMN)
        )
        .join_asof(
            reference,
            left_on=_REFERENCE_TIME_COLUMN,
            right_on=_REFERENCE_DATETIME_COLUMN,
            strategy='backward',
        )
        .with_columns(
            (
                pl.col('datetime').dt.date()
                == pl.col('datetime').shift(-1).dt.date()
            ).fill_null(False).alias(_STRUCTURAL_CORE_COLUMN)
        )
        .with_columns(
            pl.when(
                pl.col(_STRUCTURAL_CORE_COLUMN)
                & pl.col('liquidity_sum').is_finite()
                & (pl.col('liquidity_sum') > 0.0)
                & pl.col('maker_liquidity').is_finite()
            )
            .then(
                1.0
                - 2.0
                * pl.col('maker_liquidity')
                / pl.col('liquidity_sum')
            )
            .otherwise(None)
            .alias(_MAKER_FLOW_COLUMN)
        )
        .with_columns(
            pl.col(_MAKER_FLOW_COLUMN)
            .rolling_median_by(
                'datetime',
                _ROBUST_WINDOW,
                closed='left',
                min_samples=_ROBUST_MIN_SAMPLES,
            )
            .alias(_FLOW_MEDIAN_COLUMN)
        )
        .with_columns(
            (
                pl.col(_MAKER_FLOW_COLUMN) - pl.col(_FLOW_MEDIAN_COLUMN)
            ).abs().alias(_FLOW_DEVIATION_COLUMN)
        )
        .with_columns(
            pl.col(_FLOW_DEVIATION_COLUMN)
            .rolling_median_by(
                'datetime',
                _ROBUST_WINDOW,
                closed='left',
                min_samples=_ROBUST_MIN_SAMPLES,
            )
            .alias(_FLOW_MAD_COLUMN)
        )
        .with_columns(
            (
                (pl.col(_MAKER_FLOW_COLUMN) - pl.col(_FLOW_MEDIAN_COLUMN))
                / (
                    _ROBUST_SCALE * pl.col(_FLOW_MAD_COLUMN)
                    + _ROBUST_EPSILON
                )
            ).alias(_FLOW_Z_COLUMN),
            (
                (pl.col('close') / pl.col(_REFERENCE_OPEN_COLUMN)).log()
                * _BPS_PER_UNIT
            ).alias(_MOMENTUM_COLUMN),
        )
        .with_columns(
            (
                pl.col(_STRUCTURAL_CORE_COLUMN)
                & (pl.col(_MOMENTUM_COLUMN) <= momentum_threshold)
                & (pl.col(_FLOW_Z_COLUMN) > flow_threshold)
            )
            .fill_null(False)
            .cast(pl.Int8)
            .alias(_TRIGGER_COLUMN)
        )
        .with_columns(
            pl.col(_TRIGGER_COLUMN)
            .rolling_max_by(
                'datetime',
                f'{hold}m',
                closed='right',
                min_samples=1,
            )
            .cast(pl.Int8)
            .alias(OUTPUT_COLUMN)
        )
        .drop(_INTERNAL_COLUMNS)
    )

    return result.collect() if eager else result
