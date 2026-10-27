import math
from typing import cast
from numbers import Integral
from ruamel.yaml.scalarbool import ScalarBoolean

import polars as pl

from limen.data.bars._bar_aggregation import aggregate_bars

__all__ = ['imbalance_bars', 'run_bars']


def _adaptive(data: pl.DataFrame, expected_num_ticks: int, ema_span: int, *, runs: bool) -> pl.DataFrame:
    for name, value in (('expected_num_ticks', expected_num_ticks), ('ema_span', ema_span)):
        if not isinstance(cast(object, value), Integral) or isinstance(value, (bool, ScalarBoolean)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    required = ('datetime', 'open', 'high', 'low', 'close', 'volume', 'no_of_trades',
                'liquidity_sum', 'maker_ratio', 'maker_volume', 'maker_liquidity', 'mean')
    if any(name not in data.columns for name in required):
        raise ValueError('Adaptive bars require the complete native trade-point schema')
    if not isinstance(data.schema['datetime'], pl.Datetime):
        raise ValueError('Trade points require datetime observations')
    if any(data[name].null_count() for name in required) or not data['datetime'].is_sorted():
        raise ValueError('Trade points must be non-null and chronological')
    if any(not data[name].dtype.is_numeric() or not data[name].is_finite().all() for name in required[1:]):
        raise ValueError('Trade-point quantities must be finite numeric values')
    invalid = (pl.col('no_of_trades') != 1) | (pl.col('volume') <= 0) | (pl.col('close') <= 0)
    invalid |= pl.any_horizontal([pl.col(name) != pl.col('close') for name in ('open', 'high', 'low')])
    if data.select(invalid.any()).item():
        raise ValueError('Adaptive bars require individual trade points, not aggregated candles')
    alpha = 2. / (ema_span + 1)
    expected_length = float(expected_num_ticks)
    mean_signed = mean_positive = mean_negative = 0.
    threshold = 0.
    signed = positive = negative = 0.
    direction, length, group, last_end = 0, 0, 0, 0
    previous: float | None = None
    groups: list[int] = []
    for index, (price, volume) in enumerate(cast(list[tuple[float, float]], data.select('close', 'volume').rows())):
        if previous is not None and price != previous:
            direction = 1 if price > previous else -1
        previous = price
        flow = direction * volume
        mean_signed += alpha * (flow - mean_signed)
        mean_positive += alpha * (max(flow, 0.) - mean_positive)
        mean_negative += alpha * (max(-flow, 0.) - mean_negative)
        signed += flow
        positive += max(flow, 0.)
        negative += max(-flow, 0.)
        length += 1
        groups.append(group)
        if group == 0:
            threshold = expected_length * (max(mean_positive, mean_negative) if runs else abs(mean_signed))
        evidence = max(positive, negative) if runs else abs(signed)
        if index + 1 >= expected_num_ticks and evidence > 0 and evidence >= threshold:
            last_end = index + 1
            expected_length += alpha * (length - expected_length)
            threshold = expected_length * (max(mean_positive, mean_negative) if runs else abs(mean_signed))
            if not math.isfinite(threshold):
                raise ValueError('Adaptive bar threshold is nonfinite')
            signed = positive = negative = 0.
            length, group = 0, group + 1
    return aggregate_bars(data.head(last_end).with_columns(pl.Series('bar_group', groups[:last_end], dtype=pl.Int64)), 0.)


def imbalance_bars(data: pl.DataFrame, expected_num_ticks: int = 100, ema_span: int = 20) -> pl.DataFrame:
    """Form causal EMA volume-imbalance bars from individual native trade points."""
    return _adaptive(data, expected_num_ticks, ema_span, runs=False)


def run_bars(data: pl.DataFrame, expected_num_ticks: int = 100, ema_span: int = 20) -> pl.DataFrame:
    """Form causal EMA directional-volume run bars from individual native trade points."""
    return _adaptive(data, expected_num_ticks, ema_span, runs=True)
