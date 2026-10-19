from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar, cast

import polars as pl

from limen.experiment._fold_fingerprint import Unavailable, clock, membership, schema, segments

__all__: list[str] = []
_NAMES = ('fit', 'validation', 'test')
_T = TypeVar('_T')


def partition(frame: pl.DataFrame, groups: Sequence[Sequence[int]]) -> dict[str, object]:
    return {'rows': len(groups), 'segments': segments(index for group in groups for index in group),
            'row_digest': membership(groups), 'schema': schema(frame.drop('__backtest_source_row__', strict=False))}


def raw(manifest: object, selected: pl.DataFrame, splits: list[pl.DataFrame]) -> None:
    context = cast(dict[str, object] | None, getattr(manifest, '_fold_capture', None))
    if context is None:
        return
    try:
        positions = {value: index for index, value in enumerate(clock(selected))}
        groups = [[[positions[value]] for value in clock(frame)] if frame.height else [] for frame in splits]
        witnesses = {name: partition(frame, rows) for name, frame, rows in zip(_NAMES, splits, groups, strict=True)}
        context.update(raw=witnesses, positions=positions, bar_groups={}, raw_splits=splits)
        expected = cast(Mapping[str, object] | None, context.get('expected'))
        if expected is not None and witnesses != expected['raw']:
            raise ValueError('raw partitions changed before preprocessing fit')
    except Unavailable as exc:
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
                raise Unavailable('row_membership', 'Formed rows require their actual constituent count carrier')
            counts = formed['bar_count'].to_list() if 'bar_count' in formed.columns else [1] * formed.height
            for instant, count in zip(clock(formed) if formed.height else [], counts, strict=True):
                start = offsets[instant]
                if type(count) is not int or count <= 0 or start + count > len(native):
                    raise Unavailable('row_membership', 'Formed row has no recoverable constituent count')
                groups[instant] = [positions[value] for value in native[start:start + count]]
        except (KeyError, Unavailable) as exc:
            unavailable = exc if isinstance(exc, Unavailable) else Unavailable('row_membership', f'Unrecoverable formed source row: {exc}')
            context['unavailable'] = unavailable.record()
    return datetimes, formed


def retained(manifest: object, splits: list[pl.DataFrame], data: dict[str, object]) -> dict[str, object]:
    context = cast(dict[str, object] | None, getattr(manifest, '_fold_capture', None))
    if context is None:
        return data
    data.update(_fold_trial=context['trial'], _fold_n_folds=getattr(manifest, 'split_walk_forward').n_folds,
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
                raise Unavailable('row_membership', f'{name} extension changed model rows without recoverable carriers')
            allowed = {index for first, last in cast(list[list[int]], raw_witnesses[name]['segments']) for index in range(first, last)}
            if any(index not in allowed for group in rows for index in group):
                raise Unavailable('row_membership', f'{name} retained support escaped its raw partition')
            retained_witnesses[name] = partition(frame, rows)
        result = {'version': 1, 'raw': context['raw'], 'retained': retained_witnesses}
        expected = cast(Mapping[str, object] | None, context.get('expected'))
        if expected is not None and retained_witnesses != expected['retained']:
            raise ValueError('retained membership/schema changed before model fit')
        data['_fold_partitions'] = result
    except (KeyError, Unavailable) as exc:
        unavailable = exc if isinstance(exc, Unavailable) else Unavailable('row_membership', f'Missing retained carrier: {exc}')
        data['_fold_partitions'] = unavailable.record()
    return data


def capture_clock(manifest: object, frame: pl.DataFrame) -> None:
    callback = getattr(manifest, '_fold_decision_clock', None)
    if callback is not None:
        callback(frame)
