from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl

from limen.backtest.trade_contract import JsonValue, contract_digest, json_value
from limen.experiment._objective_run import finalize_objective_result, validate_objective_resume
from limen.experiment._walk_forward_run import WalkForwardRun, validate_walk_forward_resume
from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.search_strategy import SearchStrategy
from limen.experiment.reducer.pruning_strategy import PruningStrategy

if TYPE_CHECKING:
    from limen.experiment.experiment_core import UniversalExperimentLoop

__all__ = ['restore_parallel']


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f'Cannot resume parallel run without valid {name}')
    return cast(dict[str, object], value)


def _values(value: object, name: str, length: int | None = None, *, nullable: bool = False) -> list[object]:
    if not isinstance(value, list) or (length is not None and len(cast(list[object], value)) != length):
        raise ValueError(f'Cannot resume parallel run without aligned {name}')
    values = cast(list[object], value)
    for item in values:
        if not (nullable and item is None) and (type(item) not in (int, float) or not math.isfinite(float(cast(float, item)))):
            raise ValueError(f'Cannot resume parallel run with invalid {name}')
    return values


def _evidence(record: Mapping[str, object], *, execution: bool, outputs: bool) -> None:
    predictions = _values(record.get('preds'), 'predictions')
    if not predictions or any(value not in (0, 1) for value in predictions):
        raise ValueError('Cannot resume parallel run without binary predictions')
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
    if 'trade_contract' in record:
        contract = _mapping(record['trade_contract'], 'trade contract')
        if contract_digest(cast(dict[str, JsonValue], json_value(contract))) != record.get('trade_contract_digest'):
            raise ValueError('Cannot resume parallel run with mismatched trade contract evidence')
        ledger = _mapping(record.get('trade_ledger'), 'trade ledger')
        if any(not isinstance(ledger.get(key), list) for key in ('states', 'intents', 'fills', 'episodes', 'funding')) or not ledger['states']:
            raise ValueError('Cannot resume parallel run without complete trade ledger evidence')
    if execution:
        if 'execution' not in record or 'market' not in record:
            raise ValueError('Cannot resume parallel run without recorded execution/market evidence')
        if record['execution'] is not None:
            ledger = _mapping(record['execution'], 'execution')
            for key in ('pos', 'gross', 'net'):
                _ = _values(ledger.get(key), f'execution {key}', len(predictions))
            market = _mapping(record['market'], 'market')
            _ = _values(market.get('ret'), 'market returns', len(predictions), nullable=True)
    if outputs:
        probabilities = _values(record.get('probs'), 'probabilities', len(predictions))
        threshold, rule = record.get('optimal_threshold'), record.get('threshold_rule')
        if any(float(cast(float, value)) < 0 or float(cast(float, value)) > 1 for value in probabilities) or type(threshold) not in (int, float) or not math.isfinite(float(cast(float, threshold))) or rule not in ('>', '>='):
            raise ValueError('Cannot resume parallel run with invalid threshold/probability evidence')
        selected = [int(float(cast(float, value)) >= float(cast(float, threshold))) if rule == '>=' else int(float(cast(float, value)) > float(cast(float, threshold))) for value in probabilities]
        if predictions != selected:
            raise ValueError('Cannot resume parallel run with inconsistent recorded predictions')


def restore_parallel(host: UniversalExperimentLoop, msq: MSQ, domain: ParamDomain,
                     feedback_controller: FeedbackController, checkpoint_manager: CheckpointManager,
                     *, content_hash: str, strategy_type: str, csv_path: Path, round_data_path: Path) -> int:
    attributes = cast(dict[str, object], vars(host))
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
    validate_objective_resume(host.manifest, metadata, csv_path)
    validate_walk_forward_resume(host.manifest, metadata)
    checkpoint = cast(dict[str, object], checkpoint_manager.validate(directory, content_hash=content_hash, strategy_type=strategy_type))
    saved = _mapping(checkpoint['metadata'], 'checkpoint metadata')
    frontier = saved['experiment_round']
    if type(frontier) is not int or cast(int, frontier) < -1:
        raise ValueError('Cannot resume parallel run with invalid terminal frontier')
    start = cast(int, frontier) + 1
    msq_state = _mapping(checkpoint['msq_state'], 'MSQ state')
    if type(msq_state.get('yielded_count')) is not int or msq_state['yielded_count'] != start:
        raise ValueError('Parallel checkpoint draw state differs from its terminal frontier')
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
    if getattr(host.manifest, 'ablation_config', None) is not None and '_dropped_features' not in header:
        raise ValueError('Cannot resume ablation results without _dropped_features')
    expected = [(index, str(row['id'])) for index, row in enumerate(rows) if not row['strict_mode_error']]
    committed: list[dict[str, object]] = []
    lines: list[str] = []
    if expected and not round_data_path.exists():
        raise ValueError('Cannot resume parallel run without successful round data')
    if round_data_path.exists():
        with round_data_path.open() as stream:
            for index, identity in expected:
                line = stream.readline()
                if not line:
                    raise ValueError('Cannot resume parallel run without all successful round evidence')
                entry = _mapping(json.loads(line), 'round data')
                if type(entry.get('_round_index')) is not int or entry['_round_index'] != index or entry.get('round_id') != identity:
                    raise ValueError('Parallel JSONL evidence differs from successful terminal CSV occurrences')
                committed.append(entry)
                lines.append(line.rstrip('\n'))
    forward = cast(WalkForwardRun | None, attributes['_walk_forward'])
    for entry in committed:
        _ = _mapping(entry.get('round_params'), 'round parameters')
        if forward is None:
            _evidence(entry, execution=bool(attributes['_record_execution']), outputs=bool(attributes['_record_model_outputs']))
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
                _evidence(fold, execution=bool(attributes['_record_execution']), outputs=bool(attributes['_record_model_outputs']))
    log = pl.read_csv(csv_path, n_rows=start, schema_overrides={'_convergence_warning': pl.Boolean}).head(start)
    if '_convergence_warning' not in log.columns:
        log = log.with_columns(pl.lit(None, dtype=pl.Boolean).alias('_convergence_warning'))
    for index, row in enumerate(log.iter_rows(named=True)):
        finalize_objective_result(host.manifest, row, not rows[index]['strict_mode_error'])
    domain.set_state(_mapping(checkpoint['domain_state'], 'domain state'))
    msq.set_state(msq_state)
    if 'feedback_controller_state' in checkpoint:
        feedback_controller.set_state(_mapping(checkpoint['feedback_controller_state'], 'feedback state'))
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
    host.experiment_log = log
    _ = round_data_path.write_text(''.join(line + '\n' for line in lines))
    log.write_csv(csv_path)
    return start
