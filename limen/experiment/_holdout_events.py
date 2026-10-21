'''Protected event-execution checks shared by freeze and assessment.'''

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from limen.backtest.trade_contract import JsonValue, TradePolicy, contract_digest, json_value
from limen.cohort import Cohort
from limen.experiment._holdout_history import utc_day
from limen.experiment._recording import project_record
from limen.experiment._resolve_trade_policy import BacktestConfig, holdout_event_policy, resolve_number, resolve_trade_policy
from limen.inference._trainer_records import load_round_data
from limen.inference.sensor import Sensor
from limen.yaml.compiler import CompiledSFD

_BODY_KEYS = ('rule_version', 'policy', 'rules', 'initial_conditions', 'partition', 'sources', 'calibration')


def _member_manifest(sensor: object) -> object:
    '''Return the cached manifest or rebuild it from the declared reference.'''
    manifest = vars(sensor).get('_manifest')
    if manifest is None:
        manifest = CompiledSFD(cast(dict[str, object], vars(sensor)['_yaml_reference'])).manifest()
    return manifest


def freeze_kind(selection: object, locator: object) -> str:
    '''Classify a selection; refuse mismatches and locator-less contracts.'''
    if type(selection) is Sensor:
        items = [selection]
    elif type(selection) is Cohort:
        items = list(cast(list[object], vars(selection)['_members']))
    else:
        raise ValueError('Holdout accepts exact Sensor or Cohort only')
    kinds: set[tuple[bool, bool]] = set()
    for item in items:
        if type(item) is not Sensor:
            raise ValueError('Holdout accepts exact Sensor or Cohort only')
        params = cast(dict[str, object], vars(item)['_round_params'])
        manifest = _member_manifest(item)
        policy = resolve_trade_policy(cast(BacktestConfig | None, getattr(manifest, 'backtest_config', None)), params)
        kinds.add((vars(item).get('_trade_contract') is not None, policy is not None))
    if kinds == {(False, False)}:
        return 'snapshot'
    if kinds == {(True, True)}:
        if type(selection) is not Sensor:
            raise ValueError('Holdout refuses contract-bearing Cohorts')
        if locator is None:
            raise ValueError('Holdout contract freeze requires a round locator')
        return 'contract'
    raise ValueError('Holdout contract/policy mismatch refuses freezing')


def execution_key(sensor: object, locator: Mapping[str, object] | None) -> dict[str, object]:
    '''Return the member execution pin, or nothing for snapshot/copies.'''
    if vars(sensor).get('_trade_contract') is None:
        return {}
    if locator is None:
        if vars(sensor).get('_trade_digest') != contract_digest(cast(Mapping[str, JsonValue], vars(sensor)['_trade_contract'])):
            raise ValueError('Holdout contract digest mismatch after copy')
        return {}
    return {'execution': contract_execution(sensor, locator)}


def contract_execution(sensor: object, locator: Mapping[str, object]) -> dict[str, object]:
    '''Validate a contract member and pin its frozen economics.'''
    contract = cast(dict[str, object], vars(sensor)['_trade_contract'])
    params = cast(dict[str, object], vars(sensor)['_round_params'])
    manifest = _member_manifest(sensor)
    config = getattr(manifest, 'backtest_config', None)
    policy = holdout_event_policy(cast(BacktestConfig | None, config), params)
    if policy is None:
        raise ValueError('Holdout contract/policy mismatch refuses freezing')
    rules = cast(dict[str, object], contract['rules'])
    if rules.get('terminal') != 'mark_without_close':
        raise ValueError('Holdout contract requires mark_without_close terminal semantics')
    initial = cast(dict[str, object], contract['initial_conditions'])
    typed = cast(BacktestConfig, config)
    equity = resolve_number(typed.initial_equity, params, 'initial equity')
    if equity != initial['equity'] or initial['quantity'] != 0 or initial['currency'] != policy.product.quote_currency:
        raise ValueError('Holdout contract initial conditions disagree with resolved configuration')
    digest = contract_digest(cast(Mapping[str, JsonValue], contract))
    if vars(sensor).get('_trade_digest') != digest:
        raise ValueError('Holdout contract digest disagrees with the Sensor record')
    partition = cast(list[object], contract['partition'])
    split_dates = getattr(manifest, 'split_dates', None)
    if split_dates is None or cast(int, partition[1]) > utc_day(split_dates[-1].isoformat()) + 86_400_000_000_000:
        raise ValueError('Holdout recorded contract crosses the test window')
    _check_round(sensor, locator, digest)
    return {'policy': json_value(policy), 'research_digest': digest, 'initial_equity': equity}


def check_frozen_backtest(config: object, params: Mapping[str, object]) -> None:
    '''Refuse frozen event configs outside the allowlist.'''
    _ = holdout_event_policy(cast(BacktestConfig | None, config), params)


def preallocate_contract(sensor: object, execution: Mapping[str, object]) -> TradePolicy:
    '''Re-resolve and compare frozen economics before allocation.'''
    manifest = _member_manifest(sensor)
    params = cast(dict[str, object], vars(sensor)['_round_params'])
    config = cast(BacktestConfig | None, getattr(manifest, 'backtest_config', None))
    policy = holdout_event_policy(config, params)
    if policy is None or json_value(policy) != execution['policy']:
        raise ValueError('Holdout frozen policy disagrees with the fresh manifest')
    if config is None or resolve_number(config.initial_equity, params, 'initial equity') != execution['initial_equity']:
        raise ValueError('Holdout frozen equity disagrees with the fresh manifest')
    if vars(sensor).get('_trade_digest') != execution['research_digest']:
        raise ValueError('Holdout frozen digest disagrees with the Sensor record')
    return policy


def _check_round(sensor: object, locator: Mapping[str, object], digest: str) -> None:
    '''Bind a contract Sensor to its recorded round file entry.'''
    rounds = load_round_data(Path(str(locator['experiment_dir'])))
    record = rounds.get(str(locator['round_id']))
    if record is None:
        raise ValueError('Holdout contract round is missing from recorded rounds')
    if record.get('trade_contract_digest') != digest:
        raise ValueError('Holdout contract digest disagrees with the recorded round')
    body = project_record(cast(Mapping[str, object], vars(sensor)['_trade_contract']), cast(dict[str, bool] | None, locator.get('recording_policy')))
    recorded = cast(dict[str, object], record.get('trade_contract', {}))
    if any(body.get(key) != recorded.get(key) for key in _BODY_KEYS):
        raise ValueError('Holdout contract body disagrees with the recorded round')
