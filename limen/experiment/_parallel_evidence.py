from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Protocol, cast

from limen._parallel_contract import ParallelRoundContract
from limen.backtest.trade_contract import JsonValue, contract_digest, json_value
from limen.experiment._parallel_encoding import VERSION, decode, digest, encode, parameter_projection


class _ArrayValues(Protocol):
    def tolist(self) -> object: ...

def mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f'Cannot resume parallel run without valid {name}')
    return cast(dict[str, object], value)


def values(value: object, name: str, length: int | None = None, *, nullable: bool = False) -> list[object]:
    if not isinstance(value, list) or (length is not None and len(cast(list[object], value)) != length):
        raise ValueError(f'Cannot resume parallel run without aligned {name}')
    values = cast(list[object], value)
    for item in values:
        if not (nullable and item is None) and (type(item) not in (int, float) or not math.isfinite(float(cast(float, item)))):
            raise ValueError(f'Cannot resume parallel run with invalid {name}')
    return values


def validate_evidence(record: Mapping[str, object], *, event: bool, snapshot: bool, execution: bool, outputs: bool,
                      kind: str = 'binary', probabilities: bool = True) -> None:
    predictions = values(record.get('preds'), 'predictions')
    if not predictions or (kind == 'binary' and any(value not in (0, 1) for value in predictions)):
        raise ValueError('Cannot resume parallel run without binary predictions')
    if kind == 'target_exposure' and any(not -1 <= cast(float, value) <= 1 for value in predictions):
        raise ValueError('Cannot resume parallel run with out-of-bounds exposure')
    alignment = mapping(record.get('alignment'), 'alignment')
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
        contract = mapping(record['trade_contract'], 'trade contract')
        if contract_digest(cast(dict[str, JsonValue], json_value(contract))) != record.get('trade_contract_digest'):
            raise ValueError('Cannot resume parallel run with mismatched trade contract evidence')
        ledger = mapping(record.get('trade_ledger'), 'trade ledger')
        for key in ('states', 'intents', 'fills', 'episodes', 'funding'):
            rows = ledger.get(key)
            if not isinstance(rows, list) or (key == 'states' and not rows):
                raise ValueError('Cannot resume parallel run without complete trade ledger evidence')
            for row in cast(list[object], rows):
                _ = mapping(row, f'trade ledger {key} row')
    if execution:
        if 'execution' not in record or 'market' not in record:
            raise ValueError('Cannot resume parallel run without recorded execution/market evidence')
        if snapshot and record['execution'] is None:
            raise ValueError('Cannot resume configured snapshot execution without its recorded ledger/market evidence')
        if record['execution'] is not None:
            ledger = mapping(record['execution'], 'execution')
            for key in ('pos', 'gross', 'net'):
                _ = values(ledger.get(key), f'execution {key}', len(predictions))
            market = mapping(record['market'], 'market')
            _ = values(market.get('ret'), 'market returns', len(predictions), nullable=True)
    if outputs and not probabilities and record.get('probs') is not None:
        raise ValueError('Cannot resume parallel run with inapplicable probabilities')
    if outputs and probabilities:
        probability_values = values(record.get('probs'), 'probabilities', len(predictions))
        threshold, rule = record.get('optimal_threshold'), record.get('threshold_rule')
        if any(float(cast(float, value)) < 0 or float(cast(float, value)) > 1 for value in probability_values) or type(threshold) not in (int, float) or not math.isfinite(float(cast(float, threshold))) or rule not in ('>', '>='):
            raise ValueError('Cannot resume parallel run with invalid threshold/probability evidence')
        selected = [int(float(cast(float, value)) >= float(cast(float, threshold))) if rule == '>=' else int(float(cast(float, value)) > float(cast(float, threshold))) for value in probability_values]
        if predictions != selected:
            raise ValueError('Cannot resume parallel run with inconsistent recorded predictions')


def _wire(value: object, *, metrics: bool = False, hashing: bool = False) -> object:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {key: _wire(item, metrics=metrics or key == 'results', hashing=hashing) for key, item in cast(Mapping[str, object], value).items()}
    if isinstance(value, (list, tuple)):
        return [_wire(item, metrics=metrics, hashing=hashing) for item in cast(list[object] | tuple[object, ...], value)]
    if hasattr(value, 'tolist'):
        return _wire(cast(_ArrayValues, value).tolist(), metrics=metrics, hashing=hashing)
    if metrics and type(value) is float and not math.isfinite(value):
        return ['undefined_metric', value.hex()] if hashing else value
    return json_value(value)


def stamp(record: dict[str, object], context: Mapping[str, object], contract: ParallelRoundContract,
          count: int) -> None:
    supplied = all(mapping(fold['parallel_output'], 'fold output')['has_probabilities'] is True
                   for fold in cast(list[Mapping[str, object]], record['folds'])) if 'folds' in record else record.get('probs') is not None
    evidence = {'version': VERSION, 'kind': contract.prediction_kind, 'count': count,
                'has_probabilities': supplied}
    record['parallel_output'] = evidence | {'digest': digest((context, evidence, _wire(record, hashing=True)))}


def validate_record(record: Mapping[str, object], context: Mapping[str, object], contract: ParallelRoundContract,
                    *, execution: bool, outputs: bool, event: bool = False, snapshot: bool = False) -> None:
    marker = mapping(record.get('parallel_output'), 'parallel output evidence')
    count = marker.get('count')
    if (type(marker.get('version')) is not int or marker.get('version') != VERSION or type(count) is not int or count < 1
            or marker.get('kind') != contract.prediction_kind or marker.get('has_probabilities') is not (outputs and contract.probabilities)):
        raise ValueError('Cannot resume: missing or inconsistent parallel output evidence')
    if not contract.probabilities and any(record.get(key) is not None for key in ('probs', 'optimal_threshold', 'threshold_rule')):
        raise ValueError('Cannot resume parallel run with inapplicable probabilities/threshold')
    if record.get('best_iteration') is not None and (type(record['best_iteration']) is not int or cast(int, record['best_iteration']) < 0):
        raise ValueError('Cannot resume parallel run with invalid iteration evidence')
    payload = {key: value for key, value in record.items() if key not in ('parallel_output', 'parallel_parameters')}
    evidence = {key: value for key, value in marker.items() if key != 'digest'}
    if marker.get('digest') != digest((context, evidence, _wire(payload, hashing=True))):
        raise ValueError('Cannot resume parallel run with mismatched output evidence digest')
    if 'folds' in record:
        folds = record['folds']
        if not isinstance(folds, list) or not folds:
            raise ValueError('Cannot resume parallel run without complete recorded folds')
        total = 0
        for fold in cast(list[Mapping[str, object]], folds):
            validate_record(fold, context, contract, execution=execution, outputs=outputs, event=event, snapshot=snapshot)
            total += cast(int, mapping(fold['parallel_output'], 'fold output')['count'])
        if count != total:
            raise ValueError('Cannot resume parallel run with mismatched fold output count')
    else:
        alignment = mapping(record.get('alignment'), 'alignment')
        dates = alignment.get('test_datetimes')
        if not isinstance(dates, list) or len(cast(list[object], dates)) != count or len(set(cast(list[str], dates))) != count:
            raise ValueError('Cannot resume parallel run without independently aligned test rows')
        _ = values(record.get('preds'), 'predictions', count)
        validate_evidence(record, event=event, snapshot=snapshot, execution=execution, outputs=outputs,
                          kind=cast(str, contract.prediction_kind), probabilities=contract.probabilities)


def round_record(round_params: Mapping[str, object], params: Mapping[str, object], results: Mapping[str, object],
                 alignment: Mapping[str, object], identity: Mapping[str, object], contract: ParallelRoundContract,
                 *, count: int, execution: bool, outputs: bool) -> dict[str, object]:
    record: dict[str, object] = {'round_id': round_params['_id'], '_round_index': round_params['_round_index'],
                                'round_params': dict(params), 'parameter_encoding': VERSION, 'parallel_plan': dict(identity)}
    context = {key: record[key] for key in ('round_id', '_round_index', 'round_params', 'parallel_plan')}
    if 'folds' in alignment:
        record.update(cast(dict[str, object], _wire(alignment)))
        count = 0
        for fold in cast(list[dict[str, object]], record['folds']):
            size = len(cast(list[object], mapping(fold['alignment'], 'fold alignment')['test_datetimes']))
            stamp(fold, context, contract, size)
            count += size
    else:
        record.update(preds=_wire(results['_preds']), alignment=_wire({key: value for key, value in alignment.items()
                       if key not in ('execution', 'market', 'model_outputs', 'trade_contract', 'trade_contract_digest', 'trade_ledger', 'learning_binding')}))
        record.update({key: alignment[key] for key in ('trade_contract', 'trade_contract_digest', 'trade_ledger', 'learning_binding') if key in alignment})
        if execution:
            record.update(execution=alignment.get('execution'), market=alignment.get('market'))
        if outputs:
            record.update(cast(Mapping[str, object], alignment.get('model_outputs', {'probs': None})))
            if 'best_iteration' in results:
                record['best_iteration'] = results['best_iteration']
    stamp(record, context, contract, count)
    validate_record(record, context, contract, execution=execution, outputs=outputs)
    return record


def record_context(record: Mapping[str, object]) -> dict[str, object]:
    return {key: record[key] for key in ('round_id', '_round_index', 'round_params', 'parallel_plan')}


def wire_record(record: Mapping[str, object], *, legacy: bool = False, restore: bool = False) -> dict[str, object]:
    result = dict(record)
    if legacy:
        for name in ('parallel_output', 'parallel_plan', 'parameter_encoding', 'parallel_parameters'):
            _ = result.pop(name, None)
        if 'folds' in result:
            result['folds'] = [wire_record(fold, legacy=True) for fold in cast(list[Mapping[str, object]], result['folds'])]
        return result
    if type(result.get('parameter_encoding')) is not int or result.get('parameter_encoding') != VERSION:
        raise ValueError('Cannot resume: missing or inconsistent parallel evidence version')
    if restore:
        node = result.get('parallel_parameters')
        if not isinstance(node, list):
            raise ValueError('Cannot resume: missing or inconsistent parallel evidence version')
        native = decode(cast(list[object], node))
        if encode(parameter_projection(encode(native))) != encode(result.get('round_params')):
            raise ValueError('Cannot resume parallel run with mismatched typed parameter evidence')
        result['round_params'] = native
    else:
        node = encode(result['round_params'])
        result.update(parallel_parameters=node, round_params=parameter_projection(node))
    return result


def project_record(record: Mapping[str, object], policy: Mapping[str, bool] | None) -> dict[str, object]:
    if policy is None or all(policy.values()):
        return dict(record)
    excluded = {field for setting, field in (
        ('predictions', 'preds'), ('trade_ledger', 'trade_ledger'), ('fold_details', 'folds'), ('fold_details', 'fold_promotion'),
    ) if not policy[setting]}
    projected = {key: value for key, value in record.items() if key not in excluded}
    for name, fields in (('alignment', ('missing_datetimes', 'test_datetimes')), ('trade_contract', ('signal_rows',))):
        if not policy['alignment'] and name in projected:
            projected[name] = {key: value for key, value in cast(Mapping[str, object], projected[name]).items() if key not in fields}
    if 'folds' in projected:
        projected['folds'] = [project_record(fold, policy) for fold in cast(list[Mapping[str, object]], projected['folds'])]
    return projected


