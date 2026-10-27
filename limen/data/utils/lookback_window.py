from datetime import datetime
from decimal import Decimal
import math
from numbers import Real
from typing import cast
from ruamel.yaml.scalarbool import ScalarBoolean

import polars as pl

__all__ = ['lookback_window']
_NANOSECONDS_PER_DAY = 86_400_000_000_000


def window_width(days: float) -> int:
    if isinstance(days, (bool, ScalarBoolean)) or not isinstance(cast(object, days), Real) or not math.isfinite(days) or days <= 0:
        raise ValueError('lookback_days must be a positive finite number')
    width = int(Decimal(str(days)) * _NANOSECONDS_PER_DAY)
    if width > 2**63 - 1:
        raise ValueError('lookback_days exceeds the nanosecond interval range')
    return width


def source_cutoff(start: str | None, end: str | None, days: float | None, rows: int | None) -> int | None:
    if days is None:
        return None
    if start is not None or end is None:
        raise ValueError('lookback_days requires an explicit end_date_limit and no start_date_limit')
    if rows is not None:
        raise ValueError('lookback_days cannot be combined with row limits')
    end_time = datetime.strptime(end, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1)
    return end_time.days * _NANOSECONDS_PER_DAY + end_time.seconds * 1_000_000_000 - window_width(days)


def clip_window(data: pl.DataFrame, cutoff: int | None) -> pl.DataFrame:
    return data if cutoff is None else data.filter(pl.col('datetime').dt.epoch('ns') >= cutoff)


def lookback_window(data: pl.DataFrame, lookback_days: float) -> pl.DataFrame:
    """Select the trailing inclusive temporal window ending at the last supplied row."""
    if data.is_empty() or 'datetime' not in data.columns:
        raise ValueError('lookback_window requires nonempty datetime observations')
    clock = data['datetime']
    if not isinstance(clock.dtype, pl.Datetime) or clock.null_count() or not clock.is_sorted() or clock.n_unique() != data.height:
        raise ValueError('lookback_window requires unique increasing datetime observations')
    cutoff = int(clock.dt.epoch('ns')[-1]) - window_width(lookback_days)
    return clip_window(data, cutoff)
