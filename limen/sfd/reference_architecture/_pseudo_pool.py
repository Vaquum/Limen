from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return

__all__: list[str] = []
_MINIMUM_HISTORY = 2


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('PseudoLabelBinary requires timezone-aware pool bounds and training cutoff')
    return parsed.astimezone(timezone.utc)


def load_pool(path: str, sha256: str, start: str, end: str, cutoff: str,
              interval: int, columns: list[str], rows: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, datetime]:
    source = Path(path)
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError('PseudoLabelBinary pool SHA-256 differs from the supplied identity')
    first, last, training_cutoff = utc(start), utc(end), utc(cutoff)
    step = timedelta(seconds=interval)
    train_times = rows['train']['datetime'].to_list()
    evaluation = [time for name in ('val', 'test') for time in rows[name]['datetime'].to_list()]
    if not train_times or not evaluation or not first < last <= training_cutoff <= min(evaluation) + step:
        raise ValueError('PseudoLabelBinary pool must precede observed evaluation and the training cutoff')
    if max(train_times) + 2 * step > first + step:
        raise ValueError('PseudoLabelBinary pool must follow the observed upstream fitting boundary')
    raw = pl.read_parquet(BytesIO(payload)).select('datetime', 'close')
    if not isinstance(raw.schema['datetime'], pl.Datetime) or cast(pl.Datetime, raw.schema['datetime']).time_zone is None:
        raise ValueError('PseudoLabelBinary pool requires timezone-aware recorded row times')
    raw = raw.filter(pl.col('datetime') + step <= training_cutoff)
    times = raw['datetime'].dt.epoch('ns').to_numpy()
    if raw['datetime'].null_count() or len(times) < _MINIMUM_HISTORY or not np.all(np.diff(times) == interval * 1_000_000_000):
        raise ValueError('PseudoLabelBinary pool rows must be unique, chronological and follow the declared interval')
    prices = raw['close'].to_numpy()
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError('PseudoLabelBinary pool requires finite positive recorded closes')
    pool = lag_range(window_return(raw, period=1), col='ret_1', start=0, end=len(columns) - 1)
    pool = pool.filter(pl.col('datetime').is_between(first, last, closed='left')).select('datetime', *columns)
    if pool.is_empty() or pool.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError('PseudoLabelBinary requires nonempty pool windows with complete recorded history')
    observed = {time for frame in rows.values() for time in frame['datetime'].to_list()}
    if observed.intersection(pool['datetime'].to_list()):
        raise ValueError('PseudoLabelBinary pool overlaps observed training or evaluation rows')
    return pool, training_cutoff
