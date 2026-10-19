from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeVar, cast

import polars as pl
import numpy as np

from limen.experiment._walk_forward_split import WalkForwardConfig
from limen.experiment._fold_fingerprint import FoldEvidenceError, clock, membership, schema, segments, validate_digest, validate_schema

__all__: list[str] = []
_NAMES = ('fit', 'validation', 'test')
_T = TypeVar('_T')
_SEGMENT_FIELDS = 2
_MATRIX_DIMENSIONS = 2


def validate_partition(value: object, limit: int, *, raw_rows: bool) -> None:
    if not isinstance(value, dict):
        raise ValueError('fold witness partition must be an object')
    item = cast(dict[str, object], value)
    if set(item) != {'rows', 'segments', 'row_digest', 'schema'} or type(item['rows']) is not int or item['rows'] < 0 or not isinstance(item['segments'], list):
        raise ValueError('fold witness partition fields are invalid')
    validate_schema(item['schema'])
    validate_digest(item['row_digest'])
    positions: list[int] = []
    previous = -1
    for segment in cast(list[object], item['segments']):
        if not isinstance(segment, list) or len(cast(list[object], segment)) != _SEGMENT_FIELDS:
            raise ValueError('fold witness partition segment is incomplete')
        pair = cast(list[object], segment)
        if any(type(index) is not int for index in pair):
            raise ValueError('fold witness partition indices must be literal integers')
        first, last = cast(list[int], pair)
        if not 0 <= first < last <= limit or first <= previous:
            raise ValueError('fold witness partition segments overlap or escape source rows')
        positions.extend(range(first, last))
        previous = last
    if (not positions) != (item['rows'] == 0) or item['rows'] > len(positions):
        raise ValueError('fold witness partition row count is inconsistent')
    if raw_rows and (item['rows'] != len(positions) or item['row_digest'] != membership([[index] for index in positions])):
        raise ValueError('fold witness raw membership digest/count disagrees with segments')


def partition(frame: pl.DataFrame, groups: Sequence[Sequence[int]]) -> dict[str, object]:
    return {'rows': len(groups), 'segments': segments(index for group in groups for index in group),
            'row_digest': membership(groups), 'schema': schema(frame.drop('__backtest_source_row__', strict=False))}


def _same_inputs(actual: object, expected: pl.DataFrame | pl.Series) -> bool:
    if isinstance(actual, pl.DataFrame) and isinstance(expected, pl.DataFrame):
        return actual.equals(expected)
    if isinstance(actual, pl.Series) and isinstance(expected, pl.Series):
        return actual.equals(expected)
    try:
        left, right = np.asarray(actual), np.asarray(expected)
        if isinstance(expected, pl.Series) and left.ndim == _MATRIX_DIMENSIONS and left.shape[1] == 1:
            left = left[:, 0]
        numeric = np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number)
        return bool(np.array_equal(left, right, equal_nan=numeric))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise FoldEvidenceError('row_membership', f'Cannot recover native model input rows: {exc}') from exc


def raw(manifest: object, selected: pl.DataFrame, splits: list[pl.DataFrame]) -> None:
    context = cast(dict[str, object] | None, getattr(manifest, '_fold_capture', None))
    if context is None:
        return
    try:
        positions = {value: index for index, value in enumerate(clock(selected))}
        if len(positions) != selected.height or not selected['datetime'].is_sorted():
            raise FoldEvidenceError('row_membership', 'Selected fold source requires unique, increasing timestamps')
        groups = [[[positions[value]] for value in clock(frame)] if frame.height else [] for frame in splits]
        witnesses = {name: partition(frame, rows) for name, frame, rows in zip(_NAMES, splits, groups, strict=True)}
        context.update(raw=witnesses, positions=positions, bar_groups={}, raw_splits=splits)
        expected = cast(Mapping[str, object] | None, context.get('expected'))
        if expected is not None and witnesses != expected['raw']:
            raise ValueError('raw partitions changed before preprocessing fit')
    except FoldEvidenceError as exc:
        if context.get('expected') is not None:
            raise ValueError(f'raw fold witness unavailable before preprocessing fit: {exc}') from exc
        context['unavailable'] = exc.record()


def bars(manifest: object, before: pl.DataFrame, formed: pl.DataFrame, datetimes: _T) -> tuple[_T, pl.DataFrame]:
    context = cast(dict[str, object] | None, getattr(manifest, '_fold_capture', None))
    if context is not None and 'unavailable' not in context:
        try:
            positions = cast(dict[int, int], context['positions'])
            native = clock(before) if before.height else []
            offsets = {value: index for index, value in enumerate(native)}
            groups = cast(dict[int, list[int]], context['bar_groups'])
            if context.get('formed') and 'bar_count' not in formed.columns:
                raise FoldEvidenceError('row_membership', 'Formed rows require their actual constituent count carrier')
            counts = formed['bar_count'].to_list() if context.get('formed') else [1] * formed.height
            for instant, count in zip(clock(formed) if formed.height else [], counts, strict=True):
                start = offsets[instant]
                if type(count) is not int or count <= 0 or start + count > len(native):
                    raise FoldEvidenceError('row_membership', 'Formed row has no recoverable constituent count')
                groups[instant] = [positions[value] for value in native[start:start + count]]
        except (KeyError, FoldEvidenceError) as exc:
            unavailable = exc if isinstance(exc, FoldEvidenceError) else FoldEvidenceError('row_membership', f'Unrecoverable formed source row: {exc}')
            context['unavailable'] = unavailable.record()
    return datetimes, formed


def retained(manifest: object, splits: list[pl.DataFrame], data: dict[str, object]) -> dict[str, object]:
    context = cast(dict[str, object] | None, getattr(manifest, '_fold_capture', None))
    if context is None:
        return data
    data.update(_fold_trial=context['trial'], _fold_n_folds=cast(WalkForwardConfig, getattr(manifest, 'split_walk_forward', None)).n_folds,
                _fold_selected_times=list(cast(dict[int, int], context.get('positions', {}))))
    if 'unavailable' in context:
        data['_fold_partitions'] = context['unavailable']
        return data
    try:
        groups = cast(dict[int, list[int]], context['bar_groups'])
        raw_witnesses = cast(Mapping[str, Mapping[str, object]], context['raw'])
        retained_witnesses: dict[str, object] = {}
        for name, suffix, frame in zip(_NAMES, ('train', 'val', 'test'), splits, strict=True):
            rows = [groups[value] for value in clock(frame)] if frame.height else []
            if len(cast(Sequence[object], data['x_' + suffix])) != len(rows):
                raise FoldEvidenceError('row_membership', f'{name} extension changed model rows without recoverable carriers')
            columns = [column for column in frame.columns if column not in ('datetime', '__backtest_source_row__')]
            target = getattr(manifest, 'target_column', None)
            label = target if isinstance(target, str) and target in columns else columns[-1]
            if not _same_inputs(data['x_' + suffix], frame.select([column for column in columns if column != label])) or not _same_inputs(data['y_' + suffix], frame[label]):
                raise FoldEvidenceError('row_membership', f'{name} model rows differ from their retained source carriers')
            allowed = {index for first, last in cast(list[list[int]], raw_witnesses[name]['segments']) for index in range(first, last)}
            if any(index not in allowed for group in rows for index in group):
                raise FoldEvidenceError('row_membership', f'{name} retained support escaped its raw partition')
            retained_witnesses[name] = partition(frame, rows)
        result = {'version': 1, 'raw': context['raw'], 'retained': retained_witnesses}
        expected = cast(Mapping[str, object] | None, context.get('expected'))
        if expected is not None and retained_witnesses != expected['retained']:
            raise ValueError('retained membership/schema changed before model fit')
        data['_fold_partitions'] = result
    except (KeyError, FoldEvidenceError) as exc:
        unavailable = exc if isinstance(exc, FoldEvidenceError) else FoldEvidenceError('row_membership', f'Missing retained carrier: {exc}')
        data['_fold_partitions'] = unavailable.record()
    return data


def capture_clock(manifest: object, frame: pl.DataFrame) -> None:
    callback = getattr(manifest, '_fold_decision_clock', None)
    if callback is not None:
        callback(frame)


class DecisionMaker(Protocol):
    def __call__(self, raw: pl.DataFrame, bars: pl.DataFrame, *, interval_seconds: object = None) -> pl.DataFrame: ...


def decisions(manifest: object, raw: pl.DataFrame, bars: pl.DataFrame, *, interval_seconds: object, native: DecisionMaker) -> pl.DataFrame:
    step = cast(int | None, getattr(manifest, '_fold_native_interval', None))
    if step is not None:
        timing = {'start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns'}
        if timing & set(raw.columns) and not timing <= set(raw.columns):
            raise ValueError('Fold Sensor input has incomplete explicit native timing')
        if not timing <= set(raw.columns):
            start = raw['datetime'].dt.epoch('ns')
            raw = raw.with_columns(start.alias('start_ns'), (start + step).alias('end_ns'),
                                   start.alias('open_available_at_ns'), (start + step).alias('available_at_ns'))
        if not getattr(manifest, '_fold_formed', False):
            bars = bars.drop('bar_count', strict=False)
    return native(raw, bars, interval_seconds=interval_seconds)
