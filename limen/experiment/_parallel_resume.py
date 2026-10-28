from __future__ import annotations

import limen.experiment._round_storage as _storage
import limen.experiment._holdout_checkpoint as _checkpoint_proof
import limen.experiment._holdout_research as _holdout

import csv
import json
import math
from ast import literal_eval
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

import polars as pl

from limen.backtest.trade_contract import JsonValue, contract_digest, json_value
from limen.experiment._objective_run import finalize_objective_result, restored_log, run_objective, validate_objective_resume
from limen.experiment._exit_rules import verify_record
from limen.experiment._parallel_commit import context_parameter_keys, failed_parameters
from limen.experiment._walk_forward_run import WalkForwardRun, validate_walk_forward_resume
from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.manifest_core import AblationConfig, Manifest
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.search_strategy import SearchStrategy
from limen.experiment.reducer.pruning_strategy import PruningStrategy



__all__ = ['restore_parallel']


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f'Cannot resume parallel run without valid {name}')
    return cast(dict[str, object], value)


def _parameter_matches(cell: str | None, value: object) -> bool:
    if cell is None or value is None:
        return cell == ''
    if type(value) is bool:
        return cell.lower() == str(value).lower()
    try:
        if type(value) is int:
            return Decimal(cell) == value
        if type(value) is float:
            return float(cell) == value
        if isinstance(value, (list, dict)):
            return json.loads(json.dumps(literal_eval(cell))) == value
    except (ValueError, SyntaxError, InvalidOperation) as exc:
        raise ValueError('Cannot resume parallel run with malformed CSV parameter evidence') from exc
    return cell == str(value)


def _values(value: object, name: str, length: int | None = None, *, nullable: bool = False) -> list[object]:
    if not isinstance(value, list) or (length is not None and len(cast(list[object], value)) != length):
        raise ValueError(f'Cannot resume parallel run without aligned {name}')
    values = cast(list[object], value)
    for item in values:
        if not (nullable and item is None) and (type(item) not in (int, float) or not math.isfinite(float(cast(float, item)))):
            raise ValueError(f'Cannot resume parallel run with invalid {name}')
    return values


def _evidence(record: Mapping[str, object], *, event: bool, snapshot: bool, execution: bool, outputs: bool) -> None:
    verify_record(record)
    predictions = _values(record.get('preds'), 'predictions')
    if not predictions:
        raise ValueError('Cannot resume parallel run without predictions')
    alignment = _mapping(record.get('alignment'), 'alignment')
    for name in ('first_test_datetime', 'last_test_datetime'):
        value = alignment.get(name)
        if not isinstance(value, str):
            raise ValueError('Cannot resume parallel run without dated alignment')
        _ = datetime.fromisoformat(value)
    missing = alignment.get('missing_datetimes')
    if not isinstance(missing, list):
        raise ValueError('Cannot resume parallel run without missing-row alignment')
    for value in cast(list[object], missing):
        if not isinstance(value, str):
            raise ValueError('Cannot resume parallel run with malformed missing datetimes')
        _ = datetime.fromisoformat(value)
    if event and 'trade_contract' not in record:
        raise ValueError('Cannot resume parallel event execution without its trade contract/ledger evidence')
    if 'trade_contract' in record:
        contract = _mapping(record['trade_contract'], 'trade contract')
        if contract_digest(cast(dict[str, JsonValue], json_value(contract))) != record.get('trade_contract_digest'):
            raise ValueError('Cannot resume parallel run with mismatched trade contract evidence')
        ledger = _mapping(record.get('trade_ledger'), 'trade ledger')
        for key in ('states', 'intents', 'fills', 'episodes', 'funding'):
            rows = ledger.get(key)
            if not isinstance(rows, list) or (key == 'states' and not rows):
                raise ValueError('Cannot resume parallel run without complete trade ledger evidence')
            for row in cast(list[object], rows):
                _ = _mapping(row, f'trade ledger {key} row')
    if execution:
        if 'execution' not in record or 'market' not in record:
            raise ValueError('Cannot resume parallel run without recorded execution/market evidence')
        if snapshot and record['execution'] is None:
            raise ValueError('Cannot resume configured snapshot execution without its recorded ledger/market evidence')
        if record['execution'] is not None:
            ledger = _mapping(record['execution'], 'execution')
            for key in ('pos', 'gross', 'net'):
                _ = _values(ledger.get(key), f'execution {key}', len(predictions))
            market = _mapping(record['market'], 'market')
            _ = _values(market.get('ret'), 'market returns', len(predictions), nullable=True)
    if outputs and 'probs' not in record:
        raise ValueError('Cannot resume parallel run without recorded model outputs')
    if outputs and record['probs'] is not None:
        probabilities = _values(record['probs'], 'probabilities', len(predictions))
        threshold, rule = record.get('optimal_threshold'), record.get('threshold_rule')
        if any(float(cast(float, value)) < 0 or float(cast(float, value)) > 1 for value in probabilities) or type(threshold) not in (int, float) or not math.isfinite(float(cast(float, threshold))) or rule not in ('>', '>='):
            raise ValueError('Cannot resume parallel run with invalid threshold/probability evidence')
        selected = [int(float(cast(float, value)) >= float(cast(float, threshold))) if rule == '>=' else int(float(cast(float, value)) > float(cast(float, threshold))) for value in probabilities]
        if predictions != selected:
            raise ValueError('Cannot resume parallel run with inconsistent recorded predictions')


def _audit_prefix(path: Path, frontier: int, count: int, controller: FeedbackController) -> list[str]:
    lines: list[str] = []
    previous = -1
    if count and not path.exists():
        raise ValueError('Cannot resume parallel run without committed feedback audit evidence')
    if path.exists():
        with path.open() as stream:
            for number in range(count):
                line = stream.readline()
                if not line:
                    raise ValueError('Cannot resume parallel run without complete feedback audit evidence')
                entry = _mapping(json.loads(line), 'feedback audit')
                index = entry.get('round')
                if type(index) is not int or not previous < index <= frontier or not controller.should_trigger(index) or type(entry.get('trigger_number')) is not int or entry['trigger_number'] != number:
                    raise ValueError('Parallel feedback audit differs from its committed checkpoint prefix')
                previous = index
                lines.append(line.rstrip('\n'))
    return lines


def restore_parallel(host: object, msq: MSQ, domain: ParamDomain,
                     feedback_controller: FeedbackController, checkpoint_manager: CheckpointManager,
                     *, content_hash: str, strategy_type: str, csv_path: Path, round_data_path: Path,
                     failed_round_params: list[dict[str, object]], n_permutations: int,
                     context_params: Mapping[str, object] | None = None) -> int:
    attributes = cast(dict[str, object], vars(host))
    manifest = cast(Manifest | None, attributes['manifest'])
    directory = attributes.get('_experiment_dir')
    strategy = attributes.get('_search_strategy')
    if not isinstance(directory, Path) or not isinstance(strategy, SearchStrategy):
        raise ValueError('Parallel checkpoint restore requires experiment_dir and search_strategy')
    metadata = _mapping(json.loads((directory / 'metadata.json').read_text()), 'metadata')
    if metadata.get('workers', 1) != attributes['_workers'] or metadata.get('worker_native_threads', 1) != 1:
        raise ValueError('Cannot resume with a different worker/thread policy')
    for key in ('record_execution', 'record_model_outputs', 'factorize'):
        if metadata.get(key, False) != attributes[f'_{key}']:
            raise ValueError(f'Cannot resume with a different {key} setting')
    validate_objective_resume(manifest, metadata, csv_path)
    validate_walk_forward_resume(manifest, metadata)
    checkpoint = cast(dict[str, object], checkpoint_manager.validate(directory, content_hash=content_hash, strategy_type=strategy_type))
    _checkpoint_proof.validate_checkpoint(directory, checkpoint, metadata, _holdout.binding_of(manifest), getattr(manifest, '_holdout_source', None), manifest, cast(pl.DataFrame, vars(host)['data']))
    frontier = _mapping(checkpoint['metadata'], 'checkpoint metadata')['experiment_round']
    if type(frontier) is not int or frontier < -1:
        raise ValueError('Cannot resume parallel run with invalid terminal frontier')
    start = frontier + 1
    if set(context_keys := context_parameter_keys(metadata)) != {key for key in (context_params or {}) if not key.startswith('_')} and start < n_permutations:
        raise ValueError('Cannot resume unfinished parallel run with a different context parameter declaration')
    msq_state = _mapping(checkpoint['msq_state'], 'MSQ state')
    if type(msq_state.get('yielded_count')) is not int or msq_state['yielded_count'] != start:
        raise ValueError('Parallel checkpoint draw state differs from its terminal frontier')
    feedback_state = _mapping(checkpoint.get('feedback_controller_state'), 'feedback state')
    trigger_count = feedback_state.get('trigger_count')
    if type(trigger_count) is not int or trigger_count < 0:
        raise ValueError('Cannot resume parallel run with invalid feedback trigger count')
    audit_path = directory / 'audit.jsonl'
    audit_lines = _audit_prefix(audit_path, frontier, trigger_count, feedback_controller)
    with csv_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        if header is None or len(set(header)) != len(header) or not {'_round_index', 'id', 'strict_mode_error'} <= set(header):
            raise ValueError('Cannot resume parallel run without a complete result header')
        rows = [row for _, row in zip(range(start), reader, strict=False)]
    if len(rows) != start or any(row.get('_round_index') != str(index) or not row.get('id') or None in row or any(value is None for value in row.values()) for index, row in enumerate(rows)):
        raise ValueError('Cannot resume parallel run without a contiguous terminal CSV prefix')
    if any(row.get('_id', row['id']) != row['id'] for row in rows):
        raise ValueError('Cannot resume parallel run with mismatched CSV occurrence identity')
    ablation = cast(AblationConfig | None, getattr(manifest, 'ablation_config', None))
    if ablation is not None and '_dropped_features' not in header:
        raise ValueError('Cannot resume ablation results without _dropped_features')
    expected = [(index, str(row['id'])) for index, row in enumerate(rows) if not row['strict_mode_error']]
    committed: list[dict[str, object]] = []
    _storage.validate_prefix(round_data_path, expected)
    if round_data_path.exists():
        with _storage.open_text(round_data_path, limit=len(expected)) as stream:
            for line in stream:
                committed.append(_mapping(json.loads(line), 'round data'))
    forward = cast(WalkForwardRun | None, attributes['_walk_forward'])
    failed = failed_parameters(checkpoint, rows)
    parameter_keys = {key for key in (*_mapping(attributes.get('params'), 'search parameters'), *(context_params or {}), *context_keys) if not key.startswith('_')}
    parameter_keys.update(key for entry in [*committed, *failed] for key in _mapping(entry.get('round_params'), 'round parameters') if not key.startswith('_'))
    recorded_parameters: dict[int, dict[str, object]] = {}
    for entry in [*committed, *failed]:
        parameters = _mapping(entry.get('round_params'), 'round parameters')
        if not parameter_keys <= parameters.keys():
            raise ValueError('Cannot resume parallel run without all recorded parameters')
        row = rows[cast(int, entry['_round_index'])]
        if any(not _parameter_matches(row.get(key), value) for key, value in parameters.items() if not key.startswith('_')):
            raise ValueError('Cannot resume parallel run with mismatched CSV/JSONL round parameters')
        recorded_parameters[cast(int, entry['_round_index'])] = parameters
        if row['strict_mode_error']:
            continue
        if ablation is not None and forward is None:
            count = 0 if (raw_count := parameters.get(ablation.drop_count_key)) is None else raw_count
            dropped = parameters.get('_dropped_features', [])
            if type(count) is not int or count < 0 or not isinstance(dropped, list) or len(cast(list[object], dropped)) != count or any(not isinstance(value, str) for value in cast(list[object], dropped)) or json.loads(rows[cast(int, entry['_round_index'])]['_dropped_features']) != dropped:
                raise ValueError('Cannot resume parallel ablation without matching dropped-feature evidence')
        event = manifest is not None and manifest.resolve_trade_policy(parameters) is not None
        snapshot = not event and any(name.startswith('backtest_') for name in header)
        if forward is None:
            _evidence(entry, event=event, snapshot=snapshot, execution=bool(attributes['_record_execution']), outputs=bool(attributes['_record_model_outputs']))
        else:
            folds = entry.get('folds')
            if not isinstance(folds, list) or len(cast(list[object], folds)) != forward.config.n_folds:
                raise ValueError('Cannot resume parallel walk-forward without complete recorded folds')
            for number, value in enumerate(cast(list[object], folds)):
                fold = _mapping(value, 'fold')
                if type(fold.get('fold')) is not int or fold['fold'] != number or not _mapping(fold.get('results'), 'fold results'):
                    raise ValueError('Cannot resume parallel walk-forward with invalid fold identity/results')
                returns = _values(fold.get('net_returns'), 'fold returns')
                _ = _values(fold.get('preds'), 'fold predictions', len(returns))
                _evidence(fold, event=event, snapshot=snapshot, execution=bool(attributes['_record_execution']), outputs=bool(attributes['_record_model_outputs']))
    overrides: dict[str, type[pl.DataType]] = {'_convergence_warning': pl.Boolean}
    overrides.update(dict.fromkeys((key for parameters in recorded_parameters.values() for key in parameters if not key.startswith('_')), pl.String))
    overrides.update({objective.column: pl.Float64} if (objective := run_objective(manifest)) is not None else {})
    log = restored_log(csv_path, start, overrides, parameter_keys)
    log = log.with_columns(pl.DataFrame([{key: value for key, value in recorded_parameters[index].items() if not key.startswith('_')} for index in range(start)], infer_schema_length=None, strict=False).get_columns())
    if '_convergence_warning' not in log.columns:
        log = log.with_columns(pl.lit(None, dtype=pl.Boolean).alias('_convergence_warning'))
    for index, row in enumerate(log.iter_rows(named=True)):
        finalize_objective_result(manifest, row, not rows[index]['strict_mode_error'])
    domain.set_state(_mapping(checkpoint['domain_state'], 'domain state'))
    msq.set_state(msq_state)
    feedback_controller.set_state(feedback_state)
    pruning = cast(list[PruningStrategy], attributes['_pruning_strategies'])
    states = checkpoint.get('pruning_strategy_states', [])
    if not isinstance(states, list) or len(cast(list[object], states)) != len(pruning):
        raise ValueError('Cannot resume parallel run with a different pruning strategy count')
    for reducer, state in zip(pruning, cast(list[object], states), strict=True):
        reducer.set_state(_mapping(state, 'pruning state'))
    strategy.rebuild_seen_from_log(str(row['id']) for row in rows)
    if forward is not None:
        if expected:
            _ = forward.restore(round_data_path, start, expected)
        else:
            forward.writer.clear()
    attributes['experiment_log'] = log if start else None
    failed_round_params.extend(failed)
    if round_data_path.exists():
        _storage.truncate_rounds(round_data_path, start)
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, header)
        writer.writeheader()
        writer.writerows(rows)
    if audit_path.exists():
        _ = audit_path.write_text(''.join(line + '\n' for line in audit_lines))
    return start
