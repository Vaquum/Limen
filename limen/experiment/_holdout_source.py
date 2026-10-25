from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast
from collections.abc import Callable
from itertools import pairwise
from contextvars import ContextVar
import limen.data.historical_data as _historical

import numpy as np
import polars as pl
from polars.exceptions import ComputeError

from limen.data.historical_data import HistoricalData
from limen.experiment._holdout_history import HoldoutEvidenceError, ProjectBinding, utc_day
class SourceConfig(Protocol):
    @property
    def method(self) -> Callable[..., object]: ...
    @property
    def params(self) -> Mapping[str, object]: ...

class SourceManifest(Protocol):
    @property
    def data_source_config(self) -> SourceConfig | None: ...

_DEFAULT_SPOT_DATASET_REPO = cast(str, vars(_historical)['_DEFAULT_SPOT_DATASET_REPO'])
_normalize_datetime_column = cast(Callable[[pl.DataFrame], pl.DataFrame], vars(_historical)['_normalize_datetime_column'])
_normalize_timestamp_column = cast(Callable[[pl.DataFrame], pl.DataFrame], vars(_historical)['_normalize_timestamp_column'])
_read_any_file = cast(Callable[..., pl.DataFrame], vars(_historical)['_read_any_file'])
_spot_dataset_repo_for_kline_size = cast(Callable[[int, str], str], vars(_historical)['_spot_dataset_repo_for_kline_size'])

_NATIVE = frozenset((60, 900, 1800, 3600, 7200, 14400))
_GET_FILE, _GET_SPOT = HistoricalData.get_any_file, HistoricalData.get_spot_klines
source_workflow_active: ContextVar[bool] = ContextVar('holdout_source_workflow', default=False)


class Bounds(TypedDict):
    first_open_ns: int | None
    last_close_ns: int | None
    rows: int
    gaps: list[list[int]]


class SourceEvidence(TypedDict):
    kind: Literal['hosted_native', 'local_file', 'caller_frame']
    method: str
    identity: dict[str, object]
    interval_seconds: int
    native_witness: Literal['default-hosted-contract', 'caller-declared']
    materialized: Bounds
    admitted: Bounds
    admitted_sha256: str
    physical_scan: Literal['unknown']


def bounds(frame: pl.DataFrame, interval: int) -> Bounds:
    times = frame['datetime'].dt.epoch('ns').to_numpy()
    step = interval * 1_000_000_000
    return {'first_open_ns': int(times[0]) if times.size else None,
            'last_close_ns': int(times[-1]) + step if times.size else None,
            'rows': frame.height,
            'gaps': [[int(left) + step, int(right)] for left, right in pairwise(times) if right - left > step]}


def _normal(frame: pl.DataFrame) -> pl.DataFrame:
    dtype = frame.schema.get('datetime')
    if isinstance(dtype, pl.Datetime) and dtype.time_zone not in (None, 'UTC'):
        raise HoldoutEvidenceError('Holdout source datetime timezone must be UTC')
    try:
        frame = _normalize_datetime_column(_normalize_timestamp_column(frame))
    except ComputeError as exc:
        raise HoldoutEvidenceError('Holdout source requires recorded nonnull UTC datetimes') from exc
    dtype = frame.schema.get('datetime')
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone != 'UTC' or frame['datetime'].null_count():
        raise HoldoutEvidenceError('Holdout source requires recorded nonnull UTC datetimes')
    values = frame['datetime'].dt.epoch('ns').to_numpy()
    if values.size > 1 and not np.all(np.diff(values) > 0):
        raise HoldoutEvidenceError('Holdout source datetimes must be strictly increasing and unique')
    return frame


def admit_frame(frame: pl.DataFrame, *, interval: int, lower_ns: int | None, upper_ns: int,
                complete: bool = False) -> pl.DataFrame:
    frame = _normal(frame)
    step = interval * 1_000_000_000
    clock = pl.col('datetime').dt.epoch('ns')
    timing = {'start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns'}
    if timing & set(frame.columns):
        if not timing <= set(frame.columns) or any(frame[name].dtype != pl.Int64 or frame[name].null_count() for name in timing):
            raise HoldoutEvidenceError('Holdout native source requires all four nonnull nanosecond clocks')
        if frame.select(((pl.col('start_ns') != clock) | (pl.col('end_ns') != clock + step)
                         | (pl.col('available_at_ns') < pl.col('end_ns'))
                         | (pl.col('open_available_at_ns') < pl.col('start_ns'))
                         | (pl.col('open_available_at_ns') > pl.col('available_at_ns'))).any()).item():
            raise HoldoutEvidenceError('Holdout native source clocks disagree with complete causal support')
        if frame.select(((clock + step <= upper_ns) & (pl.col('available_at_ns') > upper_ns)).any()).item():
            raise HoldoutEvidenceError('Holdout native source availability crosses the reserved cutoff')
    frame = frame.filter((clock + step <= upper_ns) & (clock >= lower_ns if lower_ns is not None else pl.lit(True)))
    if frame.is_empty():
        raise HoldoutEvidenceError('Holdout admission has no complete native bars')
    times = frame['datetime'].dt.epoch('ns').to_numpy()
    if np.any(times % step) or np.any(np.diff(times) % step):
        raise HoldoutEvidenceError('Holdout source does not match its declared native interval/grid; no resampling is allowed')
    if complete and (lower_ns is None or int(times[0]) != lower_ns or int(times[-1]) + step != upper_ns or np.any(np.diff(times) != step)):
        raise HoldoutEvidenceError('Holdout final context/window lacks complete native support')
    if not {'open', 'high', 'low', 'close'} <= set(frame.columns):
        raise HoldoutEvidenceError('Holdout native bars require open/high/low/close')
    ohlc = frame.select('open', 'high', 'low', 'close')
    if any(not dtype.is_numeric() for dtype in ohlc.schema.values()):
        raise HoldoutEvidenceError('Holdout OHLC must be numeric')
    prices = ohlc.to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any() or np.any(prices[:, 2] > np.minimum(prices[:, 0], prices[:, 3])) or np.any(prices[:, 1] < np.maximum(prices[:, 0], prices[:, 3])):
        raise HoldoutEvidenceError('Holdout OHLC must be finite, positive and internally consistent')
    return frame


def source_spec(manifest: SourceManifest, binding: ProjectBinding) -> tuple[str, dict[str, object], int]:
    config = manifest.data_source_config
    if config is None or binding.interval_seconds is None:
        raise HoldoutEvidenceError('Holdout research requires a declared native source/interval')
    params = dict(config.params)
    method = getattr(config.method, '__func__', config.method)
    owner = getattr(config.method, '__self__', None)
    if owner is not None and not isinstance(owner, HistoricalData):
        raise HoldoutEvidenceError('Holdout source overrides/subclasses are unsupported')
    if any(params.get(name) is not None for name in ('row_count_limit', 'n_rows')):
        raise HoldoutEvidenceError('Holdout research does not support source row limits')
    if method is _GET_FILE:
        if set(params) - {'file_path_or_url', 'has_header', 'columns', 'row_count_limit', 'n_rows'}:
            raise HoldoutEvidenceError('Holdout file source has unsupported parameters')
        if owner is not None:
            mro = type(owner).__mro__
            defining = next((klass for klass in mro if '_store' in vars(klass)), None)
            if defining is not HistoricalData or '_store' in vars(owner):
                raise HoldoutEvidenceError('Holdout source overrides/subclasses are unsupported')
            if any('__getattribute__' in vars(klass) or '__getattr__' in vars(klass) for klass in mro if klass is not object):
                raise HoldoutEvidenceError('Holdout source overrides/subclasses are unsupported')
        if owner is not None and type(owner) is not HistoricalData and set(vars(owner)) - {'data', 'data_columns'}:
            raise HoldoutEvidenceError('Holdout source overrides/subclasses are unsupported')
        if type(params.get('has_header', True)) is not bool:
            raise HoldoutEvidenceError('Holdout file source has_header must be a boolean')
        value = params.get('file_path_or_url')
        if not isinstance(value, str) or '://' in value or '{' in value or Path(value).suffix.lower() not in ('.parquet', '.csv'):
            raise HoldoutEvidenceError('Holdout file source must be a literal local Parquet or CSV path')
        params['file_path_or_url'] = str(Path(value).resolve())
        return 'local_file', params, binding.interval_seconds
    if method is not _GET_SPOT or set(params) - {'kline_size', 'start_date_limit', 'end_date_limit', 'row_count_limit', 'n_rows'}:
        raise HoldoutEvidenceError('Holdout source must be native default spot klines or a local Parquet or CSV file')
    if owner is not None and type(owner) is not HistoricalData:
        raise HoldoutEvidenceError('Holdout source overrides/subclasses are unsupported')
    interval = params.get('kline_size', 60)
    if type(interval) is not int or interval not in _NATIVE or interval != binding.interval_seconds:
        raise HoldoutEvidenceError('Holdout hosted interval must match a supported native dataset')
    if HistoricalData.DEFAULT_SPOT_KLINES_DATASET_REPO != _DEFAULT_SPOT_DATASET_REPO or (owner is not None and owner.DEFAULT_SPOT_KLINES_DATASET_REPO != _DEFAULT_SPOT_DATASET_REPO):
        raise HoldoutEvidenceError('Holdout does not support overridden hosted dataset identity')
    params['dataset_repo'] = _spot_dataset_repo_for_kline_size(interval, _DEFAULT_SPOT_DATASET_REPO)
    return 'hosted_native', params, interval


def _date_limit(value: object, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HoldoutEvidenceError(f'Holdout {name} must be a literal UTC date')
    return utc_day(value)


def load_admitted_source(manifest: SourceManifest, binding: ProjectBinding, *, lower_ns: int | None,
                         upper_ns: int, complete: bool) -> tuple[pl.DataFrame, SourceEvidence]:
    kind, params, interval = source_spec(manifest, binding)
    identity = {key: value for key, value in params.items() if key not in ('start_date_limit', 'end_date_limit', 'row_count_limit', 'n_rows')}
    if kind == 'local_file':
        path = cast(str, params['file_path_or_url'])
        columns = params.get('columns')
        if columns is not None and (not isinstance(columns, list) or not all(isinstance(item, str) for item in cast(list[object], columns))):
            raise HoldoutEvidenceError('Holdout source columns must be literal names')
        raw = _read_any_file(path, columns=cast(list[str] | None, columns), has_header=cast(bool, params.get('has_header', True)))
    else:
        if not complete:
            declared_start = _date_limit(params.get('start_date_limit'), 'start_date_limit')
            declared_end = _date_limit(params.get('end_date_limit'), 'end_date_limit')
            lower_ns = declared_start if lower_ns is None else lower_ns
            if declared_end is not None:
                upper_ns = min(upper_ns, declared_end + 86400 * 1_000_000_000)
        # Existing native reader resolves/cache-maps the default hosted snapshot.
        # Half-open row admission below replaces its inclusive end-date filter.
        raw = _read_any_file(cast(str, params['dataset_repo']))
    raw = _normal(raw)
    admitted = admit_frame(raw, interval=interval, lower_ns=lower_ns, upper_ns=upper_ns, complete=complete)
    buf = admitted.write_ipc(None, compression='uncompressed')
    evidence: SourceEvidence = {'kind': cast(Literal['hosted_native', 'local_file'], kind), 'method': f'{manifest.data_source_config.method.__module__}.{manifest.data_source_config.method.__qualname__}' if manifest.data_source_config is not None else '',
        'identity': identity, 'interval_seconds': interval, 'native_witness': 'default-hosted-contract' if kind == 'hosted_native' else 'caller-declared',
        'materialized': bounds(raw, interval), 'admitted': bounds(admitted, interval), 'admitted_sha256': sha256(buf.getvalue()).hexdigest(), 'physical_scan': 'unknown'}
    return admitted, evidence


def frame_evidence(frame: pl.DataFrame, binding: ProjectBinding) -> tuple[pl.DataFrame, SourceEvidence]:
    if binding.cutoff_ns is None or binding.interval_seconds is None:
        raise HoldoutEvidenceError('Holdout caller frame requires an active reservation')
    raw = _normal(frame)
    admitted = admit_frame(raw, interval=binding.interval_seconds, lower_ns=None, upper_ns=binding.cutoff_ns)
    buf = admitted.write_ipc(None, compression='uncompressed')
    return admitted, {'kind': 'caller_frame', 'method': 'caller.data', 'identity': {}, 'interval_seconds': binding.interval_seconds,
        'native_witness': 'caller-declared', 'materialized': bounds(raw, binding.interval_seconds), 'admitted': bounds(admitted, binding.interval_seconds), 'admitted_sha256': sha256(buf.getvalue()).hexdigest(), 'physical_scan': 'unknown'}


def replay_witness(alignment: object, index: int, actual: SourceEvidence, *, unchanged: bool) -> Mapping[str, object]:
    if not isinstance(alignment, list):
        raise HoldoutEvidenceError('Holdout replay requires retained preparation evidence')
    row = cast(list[object], alignment)[index]
    if not isinstance(row, Mapping):
        raise HoldoutEvidenceError('Holdout replay requires retained preparation evidence')
    recorded = cast(Mapping[str, object], row).get('holdout')
    if not isinstance(recorded, Mapping):
        raise HoldoutEvidenceError('Holdout replay lacks a protected source witness')
    witness = cast(Mapping[str, object], recorded)
    source = witness.get('source')
    if not unchanged or not isinstance(source, Mapping) or cast(Mapping[str, object], source).get('admitted_sha256') != actual['admitted_sha256']:
        raise HoldoutEvidenceError('Holdout replay admitted source changed; use the retained research input')
    return witness


def record_geometry(provenance: dict[str, object], witness: Mapping[str, object] | None,
                    raw: pl.DataFrame, splits: Mapping[str, pl.DataFrame], interval: int) -> None:
    selected_bounds = bounds(raw, interval)
    measured = {name: bounds(frame, interval) for name, frame in splits.items()}
    if witness is not None:
        expected = witness.get('splits')
        if (selected_bounds != witness.get('selected') or not isinstance(expected, Mapping)
                or any(cast(Mapping[str, object], expected).get(name) != value for name, value in measured.items())):
            raise HoldoutEvidenceError('Holdout replay selection/split geometry changed; use the retained preparation')
    provenance['selected'] = selected_bounds
    cast(dict[str, Bounds], provenance['splits']).update(measured)
