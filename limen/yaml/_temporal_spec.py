from collections.abc import Mapping
from typing import cast

from limen.sfd.rule_based._temporal import validate_temporal_conditions
from limen.yaml.errors import YAMLError


class TemporalSpec:
    def check(self, yaml_dict: dict[str, object], errors: list[YAMLError], _warnings: list[YAMLError]) -> None:
        sfd = yaml_dict.get('sfd')
        if not isinstance(sfd, Mapping):
            return
        sfd = cast(Mapping[str, object], sfd)
        manifest = sfd.get('manifest')
        if not isinstance(manifest, Mapping):
            return
        manifest = cast(Mapping[str, object], manifest)
        if manifest.get('type') != 'rule_based':
            return
        strategy = manifest.get('strategy')
        if not isinstance(strategy, Mapping):
            return
        conditions = cast(Mapping[str, object], strategy).get('conditions')
        domains = sfd.get('params')
        if not isinstance(conditions, list) or not isinstance(domains, Mapping):
            return
        for index, condition in enumerate(cast(list[object], conditions)):
            if not isinstance(condition, Mapping):
                continue
            condition = cast(Mapping[str, object], condition)
            path = f'sfd.manifest.strategy.conditions[{index}]'
            try:
                validate_temporal_conditions([condition], cast(Mapping[str, object], domains))
            except ValueError as exc:
                field = next((field for field in ('persistence_n', 'recency_n') if field in condition), None)
                errors.append(YAMLError(message=str(exc), path=f'{path}.{field}' if field else path))
