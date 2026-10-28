from collections.abc import Mapping
from typing import cast

from limen.backtest._exit_evidence import validate_definition
from limen.backtest.trade_contract import JsonValue
from limen.sfd.rule_based._temporal import validate_temporal_conditions
from limen.yaml.errors import YAMLError
from limen.yaml.rules import ConditionsList


def apply_exit_rules(manifest: object, settings: Mapping[str, object]) -> None:
    from limen.experiment.manifest_core import Manifest
    if 'exit_rules' in settings:
        config = cast(Mapping[str, object], settings['exit_rules'])
        if not isinstance(manifest, Manifest):
            raise ValueError('Exit rules require a Manifest')
        _ = manifest.with_exit_rules(cast(list[Mapping[str, object]], config['conditions']), cast(str, config['exit']))


class ExitRulesSpec:
    def check(self, document: dict[str, object], errors: list[YAMLError], warnings: list[YAMLError]) -> None:
        sfd = document.get('sfd')
        if not isinstance(sfd, Mapping):
            return
        sfd = cast(Mapping[str, object], sfd)
        manifest = sfd.get('manifest')
        if not isinstance(manifest, Mapping) or 'exit_rules' not in manifest:
            return
        manifest = cast(Mapping[str, object], manifest)
        value = manifest['exit_rules']
        path = 'sfd.manifest.exit_rules'
        if not isinstance(value, Mapping) or set(cast(Mapping[str, object], value)) != {'conditions', 'exit'}:
            errors.append(YAMLError('exit_rules requires exactly conditions and exit', path))
            return
        definition = cast(Mapping[str, JsonValue], value)
        try:
            validate_definition(definition)
            validate_temporal_conditions(cast(list[Mapping[str, object]], definition['conditions']), cast(Mapping[str, object], sfd.get('params', {})))
        except ValueError as exc:
            errors.append(YAMLError(str(exc), path))
            return
        temporary: list[YAMLError] = []
        ConditionsList().check({'sfd': {'manifest': {'strategy': {'conditions': definition['conditions'], 'entry': definition['exit']}}}}, temporary, warnings)
        errors.extend(YAMLError(error.message, error.path.replace('strategy', 'exit_rules')) for error in temporary)
        uel = document.get('uel')
        if isinstance(uel, Mapping) and cast(Mapping[str, object], uel).get('factorize'):
            errors.append(YAMLError('factorize does not support explicit exits', 'uel.factorize'))


__all__: list[str] = []
