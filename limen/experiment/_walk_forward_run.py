import copy
import json
import math
from collections.abc import Callable, Mapping
from datetime import date
from itertools import islice
from numbers import Integral, Real
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import polars as pl
import limen.experiment.manifest_core as _manifest_core

from limen.backtest.trade_contract import TradeInputs, TradeLedger
from limen.experiment.manifest_core import Manifest, MLManifest
from limen.experiment._walk_forward_split import WalkForwardConfig, read_walk_forward_config
from limen.log._permutation_returns import TrialReturnsWriter

__all__ = ['WalkForwardRun', 'validate_walk_forward_resume']


class _SplitResolver(Protocol):
    def __call__(self, manifest: Manifest, raw_data: pl.DataFrame, *, require_validation: bool = False) -> list[pl.DataFrame]: ...


_resolve_split = cast(_SplitResolver, vars(_manifest_core)['_resolve_split'])
_resolve_params = cast(Callable[[Mapping[str, object], Mapping[str, object]], dict[str, object]], vars(_manifest_core)['_resolve_params'])
_requires_fold_validation = cast(Callable[[Manifest, Mapping[str, object]], bool], vars(_manifest_core)['_requires_fold_validation'])


def validate_walk_forward_resume(manifest: Manifest | None, metadata: Mapping[str, object]) -> None:
    config = None if manifest is None else manifest.split_walk_forward
    declaration = None if config is None else config.as_dict()
    saved = metadata.get('split_walk_forward')
    if saved is not None:
        saved = read_walk_forward_config(saved).as_dict()
    if saved != declaration:
        raise ValueError('Cannot resume with a different split_walk_forward declaration')
    if config is not None and manifest is not None and metadata.get('walk_forward_validation_ratio') != list(manifest.split_config[:2]):
        raise ValueError('Cannot resume with a different split_walk_forward validation ratio')


def _return_track(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError('split_walk_forward requires a non-empty execution return track')
    values = cast(list[object], value)
    if not all(not isinstance(item, bool) and isinstance(item, Real) and math.isfinite(float(item)) for item in values):
        raise ValueError('split_walk_forward requires finite numeric execution returns')
    return [float(cast(Real, item)) for item in values]


def _net_returns(data: Mapping[str, object]) -> list[float]:
    alignment = cast(Mapping[str, object], data['_alignment'])
    execution = alignment.get('execution')
    if isinstance(execution, dict):
        return _return_track(cast(Mapping[str, object], execution).get('net'))
    ledger, inputs = data.get('_trade_ledger'), data.get('_trade_inputs')
    if not isinstance(ledger, TradeLedger) or not isinstance(inputs, TradeInputs):
        raise ValueError('split_walk_forward requires evaluated snapshot or event execution')
    endpoints = inputs.signals.select(pl.col('available_at_ns').alias('time_ns'))
    sampled = endpoints.join_asof(ledger.states.select('time_ns', 'equity'), on='time_ns', strategy='backward')
    equity = [float(value) for value in sampled['equity']]
    if not equity or not all(math.isfinite(value) and value > 0 for value in equity):
        raise ValueError('split_walk_forward lacks recorded equity at test-bar endpoints')
    equity[-1] = float(ledger.states['equity'][-1])
    previous = [inputs.initial_equity, *equity[:-1]]
    return _return_track([ending / starting - 1.0 for ending, starting in zip(equity, previous, strict=True)])


def _iso_date(value: object) -> str:
    if not isinstance(value, date):
        raise ValueError('split_walk_forward alignment requires recorded datetimes')
    return value.isoformat()


def _fold_record(fold: int, data: Mapping[str, object], result: Mapping[str, object], *, record_execution: bool, record_outputs: bool) -> dict[str, object]:
    alignment = cast(Mapping[str, object], data['_alignment'])
    scalars = {
        key: value if isinstance(value, bool) else int(value) if isinstance(value, Integral) else float(value) if isinstance(value, Real) else value
        for key, value in result.items() if not key.startswith('_') and key not in ('models', 'extras')
    }
    missing = cast(list[object], alignment.get('missing_datetimes', []))
    record: dict[str, object] = {
        'fold': fold, 'results': scalars, 'preds': np.asarray(result['_preds'], dtype=object).tolist(),
        'net_returns': _net_returns(data), 'optimal_threshold': scalars.get('optimal_threshold'),
        'alignment': {
            'missing_datetimes': [_iso_date(value) for value in missing],
            'test_datetimes': [_iso_date(value) for value in cast(list[object], alignment['test_datetimes'])],
            'first_test_datetime': _iso_date(alignment['first_test_datetime']),
            'last_test_datetime': _iso_date(alignment['last_test_datetime']),
        },
        **{key: alignment[key] for key in ('trade_contract', 'trade_contract_digest', 'trade_ledger', 'learning_binding') if key in alignment},
    }
    if record_execution:
        record.update(execution=alignment.get('execution'), market=alignment.get('market'))
    if record_outputs:
        record.update(cast(Mapping[str, object], alignment.get('model_outputs', {'probs': None})))
    return record


def _average(records: list[dict[str, object]]) -> dict[str, object]:
    rows = [cast(Mapping[str, object], record['results']) for record in records]
    result: dict[str, object] = {}
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if key == 'optimal_threshold':
            result[key] = None
        elif all(not isinstance(value, bool) and isinstance(value, Real) for value in values):
            finite = [float(cast(Real, value)) for value in values if math.isfinite(float(cast(Real, value)))]
            result[key] = float(np.mean(finite)) if finite else float('nan')
        elif all(value == values[0] for value in values):
            result[key] = values[0]
        else:
            result[key] = None
    return result


class WalkForwardRun:
    def __init__(self, manifest: Manifest, raw: pl.DataFrame, directory: Path) -> None:
        super().__init__()
        if not isinstance(manifest.split_walk_forward, WalkForwardConfig):
            raise ValueError('split_walk_forward requires its manifest configuration')
        self.manifest, self.raw, self.config = manifest, raw, manifest.split_walk_forward
        if manifest.pre_split_data_selector is None:
            self._preflight(raw, require_validation=isinstance(manifest, MLManifest) and manifest.objective is not None)
        self.writer = TrialReturnsWriter(directory)
        self.rows: list[dict[str, object]] = []

    def _preflight(self, raw: pl.DataFrame, *, require_validation: bool) -> None:
        for fold in range(self.config.n_folds):
            manifest = copy.deepcopy(self.manifest)
            vars(manifest)['_walk_forward_fold'] = fold
            _ = _resolve_split(manifest, raw, require_validation=require_validation)

    @property
    def results(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows, infer_schema_length=None) if self.rows else pl.DataFrame()

    def prepare(self, raw: pl.DataFrame, round_params: Mapping[str, object] | None = None) -> dict[str, object]:
        params = dict(round_params or {})
        if self.manifest.pre_split_data_selector is not None:
            func, base_params = self.manifest.pre_split_data_selector
            raw = func(raw, **_resolve_params(base_params, params))
        self._preflight(raw, require_validation=_requires_fold_validation(self.manifest, params))
        return {'_walk_forward_params': params, '_walk_forward_raw': raw}

    def evaluate(self, data: dict[str, object], round_params: dict[str, object]) -> dict[str, object]:
        raw = data.get('_walk_forward_raw')
        if not isinstance(raw, pl.DataFrame):
            raise ValueError('split_walk_forward requires current prepared source rows')
        records: list[dict[str, object]] = []
        for fold in range(self.config.n_folds):
            manifest = copy.deepcopy(self.manifest)
            vars(manifest)['_walk_forward_fold'] = fold
            manifest.pre_split_data_selector = None
            prepared = manifest.prepare_data(raw, dict(round_params))
            prepared['_record_execution'] = True
            prepared['_record_model_outputs'] = bool(data.get('_record_model_outputs'))
            result = manifest.run_model(prepared, dict(round_params))
            records.append(_fold_record(fold, prepared, result,
                                       record_execution=bool(data.get('_record_execution')),
                                       record_outputs=bool(data.get('_record_model_outputs'))))
        data['_alignment'] = {'folds': records}
        return _average(records)

    def accept(self, trial: object, data: Mapping[str, object]) -> None:
        alignment = cast(Mapping[str, object], data['_alignment'])
        records = cast(list[dict[str, object]], alignment['folds'])
        self._accept_records(trial, records)

    def _accept_records(self, trial: object, records: list[dict[str, object]]) -> None:
        self.writer.append(str(trial), [_return_track(record['net_returns']) for record in records])
        self.rows.extend({'id': trial, 'fold': record['fold'], **cast(Mapping[str, object], record['results'])} for record in records)

    def complete_failed_header(self, path: Path, columns: list[str], keys: list[str]) -> list[str]:
        if len(self.rows) == self.config.n_folds and set(keys) - set(columns):
            header = list(dict.fromkeys([*keys, *columns]))
            previous = pl.read_csv(path)
            previous.select(pl.col(key) if key in columns else pl.lit(None).alias(key) for key in header).write_csv(path)
            return header
        return columns

    def restore(self, path: Path, up_to_round: int | None, expected: list[tuple[int, str]] | None = None) -> int:
        entries: list[dict[str, object]] = []
        with path.open() as stream:
            for line in islice(stream, None if expected is None else len(expected)):
                entry = cast(dict[str, object], json.loads(line))
                index = entry.get('_round_index')
                if not isinstance(index, int) or isinstance(index, bool):
                    raise ValueError('Cannot resume split_walk_forward without recorded round identity')
                if up_to_round is not None and index >= up_to_round:
                    break
                records = entry.get('folds')
                if not isinstance(records, list) or len(cast(list[object], records)) != self.config.n_folds:
                    raise ValueError('Cannot resume split_walk_forward without complete recorded folds')
                for fold, raw_record in enumerate(cast(list[object], records)):
                    if not isinstance(raw_record, dict):
                        raise ValueError('Cannot resume split_walk_forward with invalid fold evidence')
                    record = cast(dict[str, object], raw_record)
                    if type(record.get('fold')) is not int or record.get('fold') != fold or not isinstance(record.get('results'), dict) or not record.get('results'):
                        raise ValueError('Cannot resume split_walk_forward with incomplete fold evidence')
                    values = _return_track(record.get('net_returns'))
                    predictions = record.get('preds')
                    if not isinstance(predictions, list) or len(cast(list[object], predictions)) != len(values) or not isinstance(record.get('alignment'), dict):
                        raise ValueError('Cannot resume split_walk_forward with incomplete aligned predictions')
                if not isinstance(entry.get('round_id'), str):
                    raise ValueError('Cannot resume split_walk_forward without recorded trial identity')
                entries.append(entry)
        if expected is not None and [(entry['_round_index'], entry['round_id']) for entry in entries] != expected:
            raise ValueError('Cannot resume split_walk_forward: recorded folds differ from successful trial results')
        for entry in entries:
            self._accept_records(entry['round_id'], cast(list[dict[str, object]], entry['folds']))
        return len(entries)

    def finish(self) -> None:
        self.writer.finish()
