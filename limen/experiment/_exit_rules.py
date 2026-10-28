from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Protocol, TypeVar, cast

import polars as pl

from limen.backtest._exit_evidence import EXIT_SEMANTICS, RULE_EXIT_VERSION, evidence, validate_definition
from limen.backtest.trade_contract import JsonValue, TradeInputs, TradePolicy, contract_digest, json_value
from limen.experiment._backtest_provenance import SOURCE_ROW
from limen.experiment._prepare_trade_context import PreparedTradeContext
from limen.sfd.rule_based._conditions import resolve_condition
from limen.sfd.rule_based._temporal import validate_temporal_conditions
from limen.sfd.rule_based.config import RuleBasedConfig
from limen.sfd.rule_based.predicates import build_predicate

class ExitManifest(Protocol):
    exit_rules_config: RuleBasedConfig | None
    target_column: str | None

_TManifest = TypeVar('_TManifest', bound=ExitManifest)
_ARGUMENTS = {'column', 'other_column', 'value', 'lookback', 'persistence_n', 'recency_n', 'direction'}


def with_exit_rules(self: _TManifest, conditions: Sequence[Mapping[str, object]], exit_root: str) -> _TManifest:
    '''Configure independent causal research exits for binary entry decisions.'''
    rows = deepcopy([dict(row) for row in conditions])
    validate_definition(cast(Mapping[str, JsonValue], {'conditions': rows, 'exit': exit_root}))
    self.exit_rules_config = RuleBasedConfig(rows, exit_root)
    return self


def declaration(manifest: object) -> dict[str, JsonValue] | None:
    config = getattr(manifest, 'exit_rules_config', None)
    if not isinstance(config, RuleBasedConfig):
        return None
    return {'version': RULE_EXIT_VERSION, 'signal_semantics': EXIT_SEMANTICS,
            'conditions': json_value(deepcopy(config.conditions)), 'exit': config.entry}


def resolved(manifest: object, params: Mapping[str, object]) -> dict[str, JsonValue] | None:
    definition = declaration(manifest)
    if definition is None:
        return None
    target = getattr(manifest, 'target_class_config', None)
    architecture = getattr(manifest, 'architecture_function', None)
    if (target is not None and getattr(target.target_class, 'requires_trade_context', False)) or getattr(manifest, 'allocation_config', None) is not None:
        raise ValueError('Explicit exits reject trade-outcome targets and allocation')
    output = getattr(architecture, 'inference_output', None)
    if output is not None and output[0] != 'binary':
        raise ValueError('Explicit exits require binary entry architecture output')
    conditions = cast(list[dict[str, object]], definition['conditions'])
    for condition in conditions:
        for key in _ARGUMENTS & condition.keys():
            value = condition[key]
            if isinstance(value, str):
                exact = value.startswith('{') and value.endswith('}') and value.count('{') == 1
                condition[key] = params[value[1:-1]] if exact else value.format(**params)
    validate_temporal_conditions(conditions)
    return definition


def decorate(policy: TradePolicy | None, manifest: object, params: Mapping[str, object]) -> TradePolicy | None:
    definition = resolved(manifest, params)
    return replace(policy, exit_rules=definition) if policy is not None and definition is not None else policy


class ExitPreparation:
    def __init__(self, manifest: ExitManifest, params: Mapping[str, object], trade: PreparedTradeContext | None) -> None:
        super().__init__()
        self.definition = resolved(manifest, params)
        self.target = manifest.target_column
        self.trade = trade
        self.frames: list[pl.DataFrame] = []
        self.selected: list[dict[str, JsonValue]] = []

    def capture(self, data: pl.DataFrame) -> None:
        if self.definition is None:
            return
        conditions = cast(list[dict[str, object]], self.definition['conditions'])
        legal = data.drop([key for key in data.columns if key == self.target or key.startswith('__')])
        leaves = legal.select([build_predicate(row, {}).fill_null(False).alias(cast(str, row['id'])) for row in conditions if 'type' in row])
        index = {cast(str, row['id']): row for row in conditions}
        bits = resolve_condition(index[cast(str, self.definition['exit'])], index, leaves)
        clock = pl.col('available_at_ns') if 'available_at_ns' in data.columns else pl.lit(None, dtype=pl.Int64).alias('available_at_ns')
        self.frames.append(data.select(SOURCE_ROW, 'datetime', clock).with_columns(bits.rename('exit')))

    def select(self, splits: list[pl.DataFrame]) -> PreparedTradeContext | None:
        if self.definition is None:
            return self.trade
        partitions: list[TradeInputs] = []
        for index, (rows, frame) in enumerate(zip(splits, self.frames, strict=True)):
            if SOURCE_ROW not in rows.columns or rows[SOURCE_ROW].null_count() or rows[SOURCE_ROW].is_duplicated().any():
                raise ValueError('Explicit exits require unique retained source identity')
            selected = rows.select(SOURCE_ROW).join(frame, on=SOURCE_ROW, how='left', maintain_order='left')
            if selected.height != rows.height or selected['exit'].null_count():
                raise ValueError('Explicit exit evidence does not match retained source rows')
            if self.trade is not None:
                original = self.trade.partitions[index]
                selected = selected.drop('available_at_ns').join(self.trade.model_rows[index].select('datetime', 'row_id', 'available_at_ns'), on='datetime', how='left', maintain_order='left')
                signals = selected.select('row_id', 'exit').join(original.signals, on='row_id', how='left', maintain_order='left')
                partitions.append(replace(original, signals=signals))
            else:
                selected = selected.with_columns(pl.col(SOURCE_ROW).cast(pl.String).alias('row_id'))
            self.selected.append(evidence(selected))
        return replace(self.trade, partitions=tuple(partitions)) if self.trade is not None else None

    def attach(self, data: dict[str, object]) -> None:
        if self.definition is None:
            return
        data.update(_exit_rules=self.definition, _exit_evidence=dict(zip(('train', 'val', 'test'), self.selected, strict=True)))
        alignment = cast(dict[str, object], data['_alignment'])
        alignment.update(exit_evidence_digest=contract_digest(self.selected[2]), exit_evidence=self.selected[2])
        if self.trade is None:
            alignment['exit_rules'] = self.definition


def snapshot_exits(data: Mapping[str, object], split: str = 'test') -> list[bool]:
    witnesses = cast(Mapping[str, Mapping[str, JsonValue]], data['_exit_evidence'])
    rows = cast(list[Mapping[str, JsonValue]], witnesses[split]['rows'])
    return [cast(bool, row['exit']) for row in rows]


__all__: list[str] = []


def finish_snapshot(data: Mapping[str, object], result: dict[str, object]) -> dict[str, object]:
    from collections.abc import Callable
    from importlib import import_module
    compute_backtest = cast(Callable[[object, Mapping[str, object]], dict[str, float]], vars(import_module('limen.sfd.reference_architecture._backtest_evaluation'))['compute_backtest'])
    if result.get('_prediction_mode', getattr(result.get('_model'), 'prediction_mode', 'binary')) != 'binary' or result.get('_preds') is None:
        raise ValueError('Explicit exits require causal binary model predictions')
    metrics = compute_backtest(result['_preds'], data)
    return {**result, **metrics}


def admit_run(manifest: object, metadata: Mapping[str, object], *, resume: bool, factorize: bool) -> None:
    current = declaration(manifest)
    if factorize and current is not None:
        raise ValueError('factorize does not support explicit exits')
    if resume and metadata.get('exit_rules_declaration') != current:
        raise ValueError('Cannot resume with a changed exit_rules_declaration')


def verify_record(record: Mapping[str, object]) -> None:
    contract = record.get('trade_contract')
    owner = cast(Mapping[str, object], contract) if isinstance(contract, Mapping) else record
    digest = owner.get('exit_evidence_digest')
    if digest is None:
        if 'exit_rules' in record or 'exit_evidence' in record:
            raise ValueError('Configured exit recording is missing its evidence digest')
        return
    payload = record.get('exit_evidence')
    if not isinstance(payload, Mapping) or contract_digest(cast(Mapping[str, JsonValue], payload)) != digest:
        raise ValueError('Exit evidence is missing or changed')


def verify_replay(record: Mapping[str, object], fresh: Mapping[str, object]) -> None:
    verify_record(record)
    alignment = cast(Mapping[str, object], fresh.get('_alignment', {}))
    verify_record(alignment)
    for key in ('exit_rules', 'exit_evidence_digest', 'exit_evidence'):
        if record.get(key) != alignment.get(key):
            raise ValueError('Exit rules or evidence changed during replay')


def validate_cache(manifest: object, data: Mapping[str, object], params: Mapping[str, object]) -> None:
    if resolved(manifest, params) != data.get('_exit_rules'):
        raise ValueError('Cached exit preparation does not match this round; refresh preparation')
