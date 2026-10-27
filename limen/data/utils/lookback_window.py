from datetime import datetime, timedelta
import math

import polars as pl

__all__ = ['lookback_window']


def window_start(end: datetime, days: float) -> datetime:
    if isinstance(days, bool) or not isinstance(days, (int, float)) or not math.isfinite(days) or days <= 0:
        raise ValueError('lookback_days must be a positive finite number')
    return end - timedelta(days=days)


def start_limit(start: str | None, end: str | None, days: float | None) -> str | None:
    if days is None:
        return start
    if start is not None or end is None:
        raise ValueError('lookback_days requires an explicit end_date_limit and no start_date_limit')
    return window_start(datetime.strptime(end, '%Y-%m-%d %H:%M:%S'), days).strftime('%Y-%m-%d %H:%M:%S')


def lookback_window(data: pl.DataFrame, lookback_days: float) -> pl.DataFrame:
    """Select the trailing inclusive temporal window ending at the last supplied row."""
    if data.is_empty() or 'datetime' not in data.columns:
        raise ValueError('lookback_window requires nonempty datetime observations')
    clock = data['datetime']
    if not isinstance(clock.dtype, pl.Datetime) or clock.null_count() or not clock.is_sorted() or clock.n_unique() != data.height:
        raise ValueError('lookback_window requires unique increasing datetime observations')
    end = clock[-1]
    if not isinstance(end, datetime):
        raise ValueError('lookback_window requires datetime observations')
    return data.filter(pl.col('datetime') >= window_start(end, lookback_days))
