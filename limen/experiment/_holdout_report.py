from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import json
import inspect
import numbers
from dataclasses import fields, is_dataclass
from datetime import date, datetime
import math
import os
from pathlib import Path
import sqlite3
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn import metrics as _metrics

from limen.backtest.trade_contract import BPS, JsonValue, contract_digest
from limen.experiment import _holdout_history as history

roc_auc_score = cast(Callable[[Sequence[int], Sequence[float]], float], vars(_metrics)['roc_auc_score'])
_BINARY_CLASSES = 2
_ATTEMPT_VERSION = 2

def normalized(value: object) -> object:
    result: object
    if value is None or type(value) in (str, bool, int):
        result = value
    elif isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Holdout specification must be finite')
    elif isinstance(value, (date, datetime)):
        result = value.isoformat()
    elif isinstance(value, Mapping):
        items = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in items):
            raise ValueError('Holdout specification requires string keys')
        result = {str(key): normalized(item) for key, item in items.items()}
    elif isinstance(value, (list, tuple)):
        result = [normalized(item) for item in cast(list[object] | tuple[object, ...], value)]
    elif inspect.ismethod(value):
        result = normalized(value.__func__)
    elif inspect.isfunction(value) or isinstance(value, type):
        if not value.__module__.startswith(('limen.', 'sklearn.')):
            raise ValueError('Holdout specification refuses custom callables')
        name = f'{value.__module__}.{value.__qualname__}'
        if inspect.isfunction(value) and value.__closure__:
            result = {'callable': name, 'defaults': normalized(value.__defaults__),
                    'closure': {key: normalized(cell.cell_contents) for key, cell in zip(value.__code__.co_freevars, value.__closure__, strict=True)}}
        else:
            result = name
    elif is_dataclass(value) and not isinstance(value, type) and type(value).__module__.startswith('limen.'):
        result = {field.name: normalized(getattr(value, field.name)) for field in fields(value)}
    else:
        raise ValueError(f'Unsupported holdout specification value: {type(value).__name__}')
    return result


def encoded(value: object) -> bytes:
    return json.dumps(normalized(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value: object) -> str:
    return sha256(encoded(value)).hexdigest()


class Metric(TypedDict):
    value: float | None
    reason: str | None
    numerator: float | None
    denominator: int | None
    support_rows: int


def metric(value: float | None, support: int, *, reason: str | None = None,
           numerator: float | None = None, denominator: int | None = None) -> Metric:
    return {'value': value, 'reason': reason, 'numerator': numerator, 'denominator': denominator, 'support_rows': support}


def classification(labels: list[int | None], predictions: list[int], probabilities: list[float | None]) -> dict[str, Metric]:
    valid = [(label, pred) for label, pred in zip(labels, predictions, strict=True) if label is not None]
    n = len(valid)
    tp = sum(label == pred == 1 for label, pred in valid)
    populations = {'accuracy': (sum(label == pred for label, pred in valid), n),
                   'precision': (tp, sum(pred == 1 for _, pred in valid)),
                   'recall': (tp, sum(label == 1 for label, _ in valid)),
                   'fpr': (sum(label == 0 and pred == 1 for label, pred in valid), sum(label == 0 for label, _ in valid))}
    result = {key: metric(num / den if den else None, n, reason=None if den else 'zero-denominator' if n else 'no-labels', numerator=float(num), denominator=den) for key, (num, den) in populations.items()}
    pairs = [(label, prob) for label, prob in zip(labels, probabilities, strict=True) if label is not None and prob is not None]
    reason = 'no-labels' if not n else 'no-probability' if not pairs else 'one-class' if len({label for label, _ in pairs}) < _BINARY_CLASSES else None
    result['auc'] = metric(float(roc_auc_score([label for label, _ in pairs], [prob for _, prob in pairs])) if reason is None else None, len(pairs), reason=reason)
    return result


def return_metrics(values: Mapping[str, float], net: npt.NDArray[np.float64], pos: npt.NDArray[np.float64]) -> dict[str, Metric]:
    result: dict[str, Metric] = {}
    for key, value in values.items():
        support = int((net > 0).sum()) if key == 'avg_win_bps' else int((net < 0).sum()) if key == 'avg_loss_bps' else len(net)
        numerator = float((net > 0).sum()) if key == 'wins_per_bar' else float(((pos > 0) & ~np.concatenate(([False], pos[:-1] > 0))).sum()) if key == 'trades_per_bar' else None
        result[key] = metric(float(value) if math.isfinite(value) else None, support,
                             reason=None if math.isfinite(value) else 'undefined-return-statistic',
                             numerator=numerator, denominator=len(net) if numerator is not None else None)
    return result


_MEMBER = pl.List(pl.Struct({'index': pl.Int64, 'id': pl.String, 'prediction': pl.Int64, 'probability': pl.Float64, 'reason': pl.String}))
_SCENARIO = pl.List(pl.Struct({'index': pl.Int64, **dict.fromkeys(('raw_position', 'raw_gross', 'raw_net', 'raw_cost', 'deployed_position', 'deployed_gross', 'deployed_net', 'deployed_cost'), pl.Float64)}))
_COLUMNS = {'open_ns': pl.Int64, 'close_ns': pl.Int64, 'role': pl.String, 'source_row_sha256': pl.String,
            'members': _MEMBER, 'aggregate_prediction': pl.Int64, 'aggregate_probability': pl.Float64,
            'aggregate_reason': pl.String, 'label': pl.Int64, 'label_reason': pl.String, 'scenarios': _SCENARIO}
_REPORT_KEYS: set[str] = {'schema_version', 'attempt_id', 'assessment_index', 'state', 'created_ns', 'finished_ns', 'project_id', 'reservation', 'spec', 'spec_sha256', 'state_sha256', 'scenarios_sha256', 'source', 'source_column_map', 'source_schema', 'planned', 'prior', 'exposure', 'rows_path', 'rows_sha256', 'classification', 'scenarios'}
_SCHEMA_EXECUTION = 2
_EXECUTION_KEYS: set[str] = {'research_digest', 'assessment_digest', 'policy', 'initial_equity', 'metrics', 'ledger', 'contract', 'tape', 'signals'}
_LEDGER_KEYS: set[str] = {'states', 'intents', 'fills', 'episodes', 'funding'}


def retained_rows(raw: pl.DataFrame, step: int, start: int, tape: list[dict[str, object]],
                  labels: list[int | None], evaluated: bool, scenarios: list[list[dict[str, object]]]) -> tuple[pl.DataFrame, dict[str, str]]:
    times = raw['datetime'].dt.epoch('ns').to_list()
    offset = sum(value < start for value in times)
    rows: list[dict[str, object]] = []
    mapping: dict[str, str] = {}
    used = set(_COLUMNS)
    for name in raw.columns:
        retained = name if name in ('open', 'high', 'low', 'close', 'volume') else f'source_{name}'
        while retained in used:
            retained = f'source_{retained}'
        mapping[name] = retained
        used.add(retained)
    for index, timestamp in enumerate(times):
        final = index >= offset
        point = tape[index - offset] if final else {}
        source = raw.slice(index, 1)
        row: dict[str, object] = {'open_ns': timestamp, 'close_ns': timestamp + step, 'role': 'final' if final else 'context',
            'source_row_sha256': sha256(source.write_ipc(None, compression='uncompressed').getvalue()).hexdigest(),
            'members': point.get('members'), 'aggregate_prediction': point.get('prediction'), 'aggregate_probability': point.get('probability'), 'aggregate_reason': point.get('reason'),
            'label': labels[index - offset] if final else None,
            'label_reason': ('terminal-label-unavailable' if index == len(times) - 1 else None) if evaluated and final else 'no-labels' if final else None,
            'scenarios': scenarios[index - offset] if final else None}
        row.update({mapping[name]: float(value) if name in ('open', 'high', 'low', 'close', 'volume') and value is not None else value for name, value in raw.row(index, named=True).items()})
        rows.append(row)
    schema = {**_COLUMNS, **{mapping[name]: pl.Float64 if name in ('open', 'high', 'low', 'close', 'volume') else dtype for name, dtype in raw.schema.items()}}
    return pl.DataFrame(rows, schema=schema), mapping


def exposure(reservation: Mapping[str, object], prior: Mapping[str, object], provenances: list[dict[str, object]], start: int, end: int, *, coverage: dict[str, list[str]]) -> dict[str, object]:
    attempts = cast(list[dict[str, object]], prior['attempts'])
    if reservation['state'] == 'released':
        classification, reasons = 'post-release', ['reservation-released']
    elif any(int(cast(int, row['context_start_ns'])) < end and start < int(cast(int, row['end_ns'])) for row in attempts):
        classification, reasons = 'subsequent', ['prior-planned-support-intersects-final']
    elif coverage['overlapping_ids']:
        classification, reasons = 'subsequent', [f'prior-recorded-research:{value}' for value in coverage['overlapping_ids']]
    elif any(cast(dict[str, object], prior['overlaps'])[key] for key in ('prior_reserved_context_ids', 'current_reserved_context_ids')):
        classification, reasons = 'subsequent', ['planned-context-overlaps-recorded-reservation']
    elif any(cast(dict[str, object], proof.get('source', {})).get('kind') != 'hosted_native' for proof in provenances):
        classification, reasons = 'unknown', ['non-hosted-or-caller-training-source']
    elif coverage['unknown_ids']:
        classification, reasons = 'unknown', [f'prior-research-coverage-unavailable:{value}' for value in coverage['unknown_ids']]
    else:
        classification, reasons = 'initial-recorded', ['conditional-on-local-completed-research-coverage']
    return {'class': classification, 'reasons': reasons, 'unknown_external_exposure': True}


def _path(root: Path, value: object, attempt_id: str, filename: str) -> Path:
    expected = f'manifests/holdout/{attempt_id}/{filename}'
    if value != expected:
        raise ValueError(f'Holdout attempt {attempt_id} has an invalid artifact path')
    target = (root / expected).resolve()
    if not target.is_relative_to(root) or not target.is_relative_to((root / 'manifests/holdout').resolve()):
        raise ValueError(f'Holdout attempt {attempt_id} artifact escapes evidence directory')
    return target


def _sync_replace(temp: Path, target: Path) -> None:
    with temp.open('rb') as stream:
        os.fsync(stream.fileno())
    _ = temp.replace(target)
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def artifacts(root: Path, report: dict[str, object], rows: pl.DataFrame) -> dict[str, object]:
    attempt_id = cast(str, report['attempt_id'])
    rows_relative = f'manifests/holdout/{attempt_id}/rows.parquet'
    report_relative = f'manifests/holdout/{attempt_id}/report.json'
    target = _path(root, rows_relative, attempt_id, 'rows.parquet')
    target.parent.mkdir(parents=True, exist_ok=False)
    temp = target.with_suffix('.parquet.tmp')
    rows.write_parquet(temp)
    _sync_replace(temp, target)
    rows_hash = sha256(target.read_bytes()).hexdigest()
    report.update(rows_path=rows_relative, rows_sha256=rows_hash)
    target = _path(root, report_relative, attempt_id, 'report.json')
    temp = target.with_suffix('.json.tmp')
    _ = temp.write_bytes(encoded(report))
    _sync_replace(temp, target)
    return {'rows_path': rows_relative, 'rows_sha256': rows_hash, 'report_path': report_relative, 'report_sha256': sha256(target.read_bytes()).hexdigest()}


def read_holdout_report(project_root: str | Path, attempt_id: str) -> dict[str, object]:
    """Read completed evidence, verifying hashes and schemas without evaluating."""
    try:
        root = history.resolve_root(project_root)
        record = _completed(root, attempt_id)
        report_file = _path(root, record['report_path'], attempt_id, 'report.json')
        rows_file = _path(root, record['rows_path'], attempt_id, 'rows.parquet')
        if sha256(report_file.read_bytes()).hexdigest() != record['report_sha256'] or sha256(rows_file.read_bytes()).hexdigest() != record['rows_sha256']:
            raise ValueError('Artifact hash mismatch')
        report = cast(dict[str, object], json.loads(report_file.read_bytes()))
        _ = encoded(report)
        expected: set[str] = _REPORT_KEYS | ({'execution'} if report.get('schema_version') == _SCHEMA_EXECUTION else cast(set[str], set()))
        if set(report) != expected or report.get('schema_version') not in (1, _SCHEMA_EXECUTION) or report['state'] != 'completed' or report['attempt_id'] != attempt_id:
            raise ValueError('Invalid report schema or identity')
        for key in ('project_id', 'assessment_index', 'created_ns', 'finished_ns', 'spec_sha256', 'state_sha256', 'scenarios_sha256', 'rows_path', 'rows_sha256'):
            if report[key] != record[key]:
                raise ValueError(f'Report/history {key} mismatch')
        rows = pl.read_parquet(rows_file)
        if any(rows.schema.get(name) != dtype for name, dtype in _COLUMNS.items()):
            raise ValueError('Invalid retained rows schema')
        mapping = cast(dict[str, str], report['source_column_map'])
        if set(rows.columns) != set(_COLUMNS) | set(mapping.values()) or rows.is_empty() or set(rows['role'].to_list()) != {'context', 'final'}:
            raise ValueError('Invalid retained source/role columns')
        _verify(report, record, rows)
        return report
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, sqlite3.Error, pl.exceptions.PolarsError) as exc:
        raise ValueError(f'Holdout attempt {attempt_id}: {exc}') from exc


def _completed(root: Path, attempt_id: str) -> dict[str, object]:
    with history.admission(root):
        binding = history.read_binding(root)
        with history.connection(root / 'manifests/holdout.sqlite3') as db:
            if db.execute('PRAGMA user_version').fetchone()[0] != _ATTEMPT_VERSION:
                raise ValueError(f'Holdout attempt {attempt_id} is unknown')
            row = db.execute("SELECT * FROM attempts WHERE id=? AND project_id=? AND state='completed'", (history.identifier(attempt_id), binding.project_id)).fetchone()
            if row is None:
                raise ValueError(f'Holdout attempt {attempt_id} is unknown or not completed')
            return cast(dict[str, object], dict(row))


def _verify(report: dict[str, object], record: dict[str, object], rows: pl.DataFrame) -> None:
    planned = cast(dict[str, int], report['planned'])
    if planned != {key: record[key] for key in ('context_start_ns', 'start_ns', 'end_ns')} or report['prior'] != json.loads(cast(str, record['prior_json'])):
        raise ValueError('Report/history planned support or prior mismatch')
    support = cast(dict[str, object], json.loads(cast(str, record['support_json'])))
    if report['source'] != support['actual']:
        raise ValueError('Report/history source mismatch')
    spec = cast(dict[str, object], report['spec'])
    if digest(spec) != report['spec_sha256'] or digest(spec['scenarios']) != report['scenarios_sha256']:
        raise ValueError('Report specification digest mismatch')
    reservation = cast(dict[str, object], report['reservation'])
    if reservation['id'] != record['reservation_id'] or reservation['revision'] != record['revision'] or reservation['state'] != record['reservation_state']:
        raise ValueError('Report/history reservation mismatch')
    step = cast(int, reservation['interval_seconds']) * 1_000_000_000
    opens = rows['open_ns'].to_numpy()
    if int(opens[0]) != planned['context_start_ns'] or int(opens[-1]) + step != planned['end_ns'] or np.any(np.diff(opens) != step) or not (rows['close_ns'] == rows['open_ns'] + step).all():
        raise ValueError('Invalid retained observation clock')
    final = rows.filter(pl.col('open_ns') >= planned['start_ns'])
    context = rows.filter(pl.col('open_ns') < planned['start_ns'])
    if set(final['role'].to_list()) != {'final'} or set(context['role'].to_list()) != {'context'} or context['members'].null_count() != context.height or final['members'].null_count():
        raise ValueError('Invalid retained participation roles')
    mapping = cast(dict[str, str], report['source_column_map'])
    source_schema = cast(list[dict[str, str]], report['source_schema'])
    if set(mapping) != {field['name'] for field in source_schema} or any(str(rows.schema[mapping[field['name']]]) != ('Float64' if field['name'] in ('open', 'high', 'low', 'close', 'volume') else field['dtype']) for field in source_schema):
        raise ValueError('Invalid retained original source schema')
    classification_metrics = cast(dict[str, object], report['classification'])
    if set(classification_metrics) != {'accuracy', 'precision', 'recall', 'fpr', 'auc'}:
        raise ValueError('Invalid classification metric schema')
    populations = [classification_metrics]
    scenarios = cast(list[dict[str, object]], report['scenarios'])
    declarations = cast(list[dict[str, object]], spec['scenarios'])
    if len(scenarios) != len(declarations):
        raise ValueError('Invalid scenario population')
    for index, (scenario, declaration) in enumerate(zip(scenarios, declarations, strict=True)):
        if set(scenario) != {'index', 'declaration', 'metrics'} or scenario['index'] != index or scenario['declaration'] != declaration:
            raise ValueError('Invalid retained scenario identity')
        populations.append(cast(dict[str, object], scenario['metrics']))
    if cast(int, report['schema_version']) == _SCHEMA_EXECUTION:
        populations.append(_verify_execution(cast(dict[str, object], report['execution']), final, spec, cast(dict[str, int], report['planned']), float(final[mapping['close']][-1])))
    elif cast(list[dict[str, object]], spec['members'])[0].get('execution') is not None:
        raise ValueError('Invalid schema for pinned execution evidence')
    _metric_schema(populations)


def _verify_execution(execution: dict[str, object], final: pl.DataFrame, spec: Mapping[str, object], planned: Mapping[str, int], mark: float) -> dict[str, object]:
    members = cast(list[dict[str, object]], spec['members'])
    pin = cast(dict[str, object], members[0].get('execution', {})) if len(members) == 1 else {}
    ledger, contract = cast(dict[str, list[dict[str, object]]], execution['ledger']), cast(dict[str, object], execution['contract'])
    initial, product = cast(dict[str, object], contract['initial_conditions']), cast(dict[str, object], cast(dict[str, object], execution['policy'])['product'])
    partition, tape, signals = cast(list[int], contract['partition']), cast(list[dict[str, object]], execution['tape']), cast(list[dict[str, object]], execution['signals'])
    rows, opens = cast(list[dict[str, object]], contract['signal_rows']), final['open_ns'].to_list()
    grid, lo, hi = final['open_ns'].to_list(), final['low'].to_list(), final['high'].to_list()
    cash, pinned = {cast(int, row['time_ns']): cast(float, row['cash']) for row in ledger['states']}, cast(dict[str, object], execution['policy'])
    requested_by, quantities, rate = {cast(str, row['intent_id']): cast(int, row['requested_at_ns']) for row in ledger['intents']}, [(cast(int, row['time_ns']), cast(float, row['quantity'])) for row in ledger['states']], (cast(float, pinned['slip_bps']) / BPS, cast(float, pinned['fee_bps']) / BPS, cast(float, product['quantity_step']))
    intents, episodes, targets, ep_ids, fill_ids, fill_eps = {row['intent_id'] for row in ledger['intents']}, {row['episode_id'] for row in ledger['episodes']}, {cast(str, row['intent_id']): cast(float, row['target']) for row in ledger['intents']}, [row['episode_id'] for row in ledger['episodes']], [row['fill_id'] for row in ledger['fills']], {row['episode_id'] for row in ledger['fills']}
    times, fills, requested, signal_moments = [cast(int, row['time_ns']) for row in ledger['states']], [cast(int, row['time_ns']) for row in ledger['fills']], [cast(int, row['requested_at_ns']) for row in ledger['intents']], {cast(int, point['available_at_ns']) for point in signals}
    equities = [cast(float, row['equity']) for row in ledger['states']]
    ending, initial_equity = equities[-1], cast(float, execution['initial_equity'])
    drawdown = float(pl.DataFrame({'equity': equities}).select((pl.col('equity') / pl.max_horizontal(pl.col('equity').cum_max(), pl.lit(initial_equity)) - 1).min()).item())
    fees, slip, funding = (sum(cast(float, row[key]) for row in ledger[frame]) for frame, key in (('fills', 'fee'), ('fills', 'slippage'), ('funding', 'recognized_delta')))
    closed = [row for row in ledger['episodes'] if row['closed_at_ns'] is not None]
    exposures = [cast(float, row['absolute_exposure']) for row in ledger['states'] if row['signal_sample']]
    replayed = {'total_return': ending / initial_equity - 1, 'ending_equity': ending, 'max_drawdown': drawdown, 'net_pnl': ending - initial_equity, 'gross_pnl': ending - initial_equity + fees + slip - funding, 'fees': fees, 'slippage': slip, 'funding_pnl': funding, 'completed_trades': float(len(closed)), 'open_trades': float(len(ledger['episodes']) - len(closed)), 'episode_win_rate': sum(cast(float, row['net_pnl']) > 0 for row in closed) / len(closed) if closed else 0.0, 'avg_absolute_exposure': sum(exposures) / len(exposures) if exposures else 0.0}
    values = {name: field['value'] for name, field in cast(dict[str, dict[str, object]], execution['metrics']).items()}
    mismatch = set(values) != set(replayed) or any(values[name] != (value if math.isfinite(value) else None) for name, value in replayed.items())
    aligned = len(tape) == len(opens) == len(signals) == len(rows)
    chained = aligned and not any(point['open_ns'] != open_ns or point['trade_contract_digest'] != execution['research_digest'] or point['available_at_ns'] != signal['available_at_ns'] or signal['target'] != point['prediction'] or (signal['row_id'], signal['available_at_ns']) != (row['row_id'], row['available_at_ns']) for point, signal, row, open_ns in zip(tape, signals, rows, opens, strict=True))
    checks = ((set(execution) != _EXECUTION_KEYS or set(ledger) != _LEDGER_KEYS or len(ep_ids) != len(set(ep_ids)) or len(fill_ids) != len(set(fill_ids)) or any(row not in fill_eps for row in ep_ids), 'Invalid execution evidence schema'), (execution['research_digest'] != pin.get('research_digest') or execution['policy'] != pin.get('policy') or execution['initial_equity'] != pin.get('initial_equity'), 'Invalid execution evidence pin'), (contract_digest(cast(Mapping[str, JsonValue], contract)) != execution['assessment_digest'], 'Invalid execution contract digest'), (cast(dict[str, object], contract['rules']).get('terminal') != 'mark_without_close', 'Invalid execution terminal semantics'), (initial.get('equity') != pin.get('initial_equity') or initial.get('quantity') != 0 or initial.get('currency') != product.get('quote_currency'), 'Invalid execution economics'),
        (ledger['funding'] != [] or contract.get('calibration') is not None or cast(dict[str, object], execution['policy']).get('funding') is not None or any(cast(float, row['accrued_funding']) != 0 for row in ledger['states']), 'Invalid execution funding absence'), (partition != [planned['context_start_ns'], planned['end_ns']], 'Invalid execution partition bounds'), (not aligned, 'Invalid execution tape identity'), (not chained, 'Invalid execution prediction chain'), (any(row['intent_id'] not in intents for row in ledger['fills']), 'Invalid execution intent linkage'), (any(row['episode_id'] not in episodes for row in (*ledger['intents'], *ledger['fills']) if row['episode_id'] is not None), 'Invalid execution episode linkage'),
        (any(signal['target'] != 0 for signal in signals) and not any(row['reason'] == 'signal' for row in ledger['intents']), 'Invalid execution ledger activity'), (any(cast(float, row['fee']) != abs(cast(float, row['quantity_delta'])) * cast(float, row['fill_price']) * cast(float, pinned['fee_bps']) / BPS or cast(float, row['slippage']) != cast(float, row['quantity_delta']) * (cast(float, row['fill_price']) - cast(float, row['reference_price'])) for row in ledger['fills']), 'Invalid execution fill costs'),
        (times != sorted(times) or fills != sorted(fills) or requested != sorted(requested) or len(times) != len(set(times)), 'Invalid execution time order'), (any(moment < partition[0] or moment > partition[1] for moment in (*times, *fills, *requested)), 'Invalid execution time bounds'),
        (not times or times[0] != partition[0] or ledger['states'][0]['equity'] != pin.get('initial_equity') or ledger['states'][0]['quantity'] != 0 or ledger['states'][0]['cash'] != pin.get('initial_equity'), 'Invalid execution accounting anchor'), (any(all((row['requested_at_ns'], row['target']) != (signal['available_at_ns'], cast(float, signal['target']) * cast(float, cast(dict[str, object], execution['policy'])['notional_rate'])) for signal in signals) for row in ledger['intents'] if row['reason'] == 'signal'), 'Invalid execution signal linkage'), (not times or times[-1] != partition[1] or ending != cast(float, ledger['states'][-1]['cash']) + cast(float, ledger['states'][-1]['quantity']) * (mark if product.get('kind') == 'cash_spot' else mark - cast(float, ledger['states'][-1]['basis'])) + cast(float, ledger['states'][-1]['accrued_funding']), 'Invalid execution terminal valuation'), (any(sum(1 for fill in ledger['fills'] if fill['time_ns'] == quantities[i][0]) == 1 and quantities[i][1] - quantities[i - 1][1] != next(cast(float, fill['quantity_delta']) for fill in ledger['fills'] if fill['time_ns'] == quantities[i][0]) for i in range(1, len(quantities))), 'Invalid execution position transition'), (any((position := bisect_right(grid, cast(int, row['time_ns']))) <= 1 or not (lo[position - 1] <= cast(float, row['reference_price']) <= hi[position - 1] or lo[position - 2] <= cast(float, row['reference_price']) <= hi[position - 2]) for row in ledger['fills']), 'Invalid execution fill reference'), (any(cast(float, row['fill_price']) != cast(float, row['reference_price']) * (1 + math.copysign(rate[0], cast(float, row['quantity_delta']))) for row in ledger['fills']), 'Invalid execution fill price'), (any(cast(int, row['time_ns']) < requested_by[cast(str, row['intent_id'])] for row in ledger['fills']), 'Invalid execution fill request order'), (any(delta != sized - prior and not (fraction * prior < 0 and delta == -prior) for row in ledger['fills'] for moment in (cast(int, row['time_ns']),) if fills.count(moment) == 1 and times.count(moment) == 1 and cast(str, row['intent_id']) in targets for position in (times.index(moment),) for before in (ledger['states'][position - 1] if position else {'cash': initial_equity, 'quantity': 0.0, 'basis': 0.0, 'accrued_funding': 0.0},) for prior in (cast(float, before['quantity']),) for price in (cast(float, row['reference_price']),) for fraction in (targets[cast(str, row['intent_id'])],) for equity in (cast(float, before['cash']) + prior * (price if product.get('kind') == 'cash_spot' else price - cast(float, before['basis'])) + cast(float, before['accrued_funding']),) for cands in ([fraction * (equity + unit * action * prior) / (price + fraction * unit * action) for action in (1, -1) for unit in (price * (rate[0] + (1 + action * rate[0]) * rate[1]),) if price + fraction * unit * action > 0 and (fraction * (equity + unit * action * prior) / (price + fraction * unit * action) - prior) * action >= 0],) for sized in (0.0 if fraction == 0 else (math.copysign(math.floor(math.nextafter(abs(cands[0]) / rate[2], math.inf)) * rate[2], cands[0]) if cands else float('nan')),) for delta in (cast(float, row['quantity_delta']),)), 'Invalid execution position sizing'), (any(row['time_ns'] not in [moment for moment, _ in quantities] for row in ledger['fills']), 'Invalid execution fill state coverage'), (any(cast(float, ledger['states'][i]['cash']) != cast(float, ledger['states'][i - 1]['cash']) for i in range(1, len(times)) if fills.count(times[i]) == 0 and times[i - 1] < times[i]) or any(cast(float, now['cash']) != cast(float, before['cash']) - (delta * fill_price + fee) if product.get('kind') == 'cash_spot' else cast(float, now['cash']) != cast(float, before['cash']) + (realized - fee) for row in ledger['fills'] for moment in (cast(int, row['time_ns']),) if fills.count(moment) == 1 and times.count(moment) == 1 for position in (times.index(moment),) if position > 0 for before in (ledger['states'][position - 1],) for now in (ledger['states'][position],) for delta in (cast(float, row['quantity_delta']),) for fill_price in (cast(float, row['fill_price']),) for fee in (cast(float, row['fee']),) for prior in (cast(float, before['quantity']),) for realized in (min(abs(delta), abs(prior)) * math.copysign(1.0, prior) * (fill_price - cast(float, before['basis'])) if prior != 0 and delta * prior < 0 else 0.0,)), 'Invalid execution cash reconciliation'), (any(cast(float, ledger['states'][i]['basis']) != cast(float, ledger['states'][i - 1]['basis']) for i in range(1, len(times)) if fills.count(times[i]) == 0 and times[i - 1] < times[i]) or any(cast(float, now['basis']) != (0.0 if quantity == 0 else ((abs(prior) * base + abs(delta) * fill_price) / abs(quantity) if prior == 0 or delta * prior > 0 else base)) for row in ledger['fills'] for moment in (cast(int, row['time_ns']),) if fills.count(moment) == 1 and times.count(moment) == 1 for position in (times.index(moment),) if position > 0 for before in (ledger['states'][position - 1],) for now in (ledger['states'][position],) for delta in (cast(float, row['quantity_delta']),) for fill_price in (cast(float, row['fill_price']),) for prior in (cast(float, before['quantity']),) for quantity in (cast(float, now['quantity']),) for base in (cast(float, before['basis']),)), 'Invalid execution basis reconciliation'), (any((cast(float, row['quantity']) == 0) != (cast(float, row['absolute_exposure']) == 0) for row in ledger['states']) or abs(cast(float, ledger['states'][-1]['quantity'])) * mark / ending != cast(float, ledger['states'][-1]['absolute_exposure']), 'Invalid execution exposure evidence'), (any(cast(float, row['quantity']) == 0 and cast(float, row['equity']) != cast(float, row['cash']) + cast(float, row['accrued_funding']) for row in ledger['states']), 'Invalid execution flat-state valuation'), ((pinned.get('take_profit_bps_column') is None and pinned.get('stop_loss_bps_column') is None) and any(cast(float, now['equity']) != cast(float, now['cash']) + cast(float, now['quantity']) * (price if product.get('kind') == 'cash_spot' else price - cast(float, now['basis'])) + cast(float, now['accrued_funding']) for row in ledger['fills'] for moment in (cast(int, row['time_ns']),) if fills.count(moment) == 1 and times.count(moment) == 1 for now in (ledger['states'][times.index(moment)],) for price in (cast(float, row['reference_price']),)), 'Invalid execution fill-state valuation'), (any(cast(float, row['entry_equity']) != cast(float, before['cash']) + cast(float, before['accrued_funding']) for row in ledger['episodes'] for moment in (cast(int, row['first_fill_ns']),) if moment in times and times.count(moment) == 1 for position in (times.index(moment),) for before in (ledger['states'][position - 1] if position else {'cash': initial_equity, 'accrued_funding': 0.0},)), 'Invalid execution episode entry'), (any(row['closed_at_ns'] != next((fill['time_ns'] for fill in reversed(ledger['fills']) if fill['episode_id'] == row['episode_id']), None) for row in ledger['episodes'] if row['closed_at_ns'] is not None), 'Invalid execution episode close'), (any(quantities[i][1] != quantities[i - 1][1] for i in range(1, len(quantities)) if sum(1 for fill in ledger['fills'] if fill['time_ns'] == quantities[i][0]) == 0 and quantities[i][0] != quantities[i - 1][0]), 'Invalid execution position continuity'),
        ([point['prediction'] for point in tape] != final['aggregate_prediction'].to_list(), 'Invalid execution retained predictions'), (contract['policy'] != execution['policy'], 'Invalid execution contract policy'), (mismatch, 'Invalid execution metrics evidence'),
        (any(cast(int, point['available_at_ns']) < close_ns for point, close_ns in zip(tape, final['close_ns'].to_list(), strict=True)), 'Invalid execution availability floor'), (any(cast(int, point['available_at_ns']) not in times and partition[0] <= cast(int, point['available_at_ns']) <= partition[1] for point in signals) or any(bool(row['signal_sample']) != (cast(int, row['time_ns']) in signal_moments) for row in ledger['states']), 'Invalid execution signal sampling'), (any(row['closed_at_ns'] is None and cast(float, row['net_pnl']) != ending - cast(float, row['entry_equity']) for row in ledger['episodes']), 'Invalid execution open-episode valuation'), (any(sum(1 for fill in ledger['fills'] if fill['time_ns'] == row['closed_at_ns']) == 1 and cast(float, row['net_pnl']) != cash.get(cast(int, row['closed_at_ns']), float('nan')) - cast(float, row['entry_equity']) for row in ledger['episodes'] if row['closed_at_ns'] is not None), 'Invalid execution closed-episode valuation'))
    for failed, message in checks:
        if failed:
            raise ValueError(message)
    return cast(dict[str, object], execution['metrics'])


def _metric_schema(populations: list[dict[str, object]]) -> None:
    for population in populations:
        for value in population.values():
            field = cast(dict[str, object], value)
            if set(field) != {'value', 'reason', 'numerator', 'denominator', 'support_rows'} or type(field['support_rows']) is not int or field['support_rows'] < 0:
                raise ValueError('Invalid metric support/schema')
            if (field['value'] is None) != (isinstance(field['reason'], str)) or (field['value'] is not None and type(field['value']) not in (int, float)):
                raise ValueError('Invalid metric availability/schema')
