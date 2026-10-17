from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
import re
from typing import cast


_FIELDS = ('persistence_n', 'recency_n')
_REFERENCE = re.compile(r'\{([A-Za-z_]\w*)\}')


def positive_temporal_window(value: object, *, condition_id: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(
            f'Condition {condition_id!r} {field} must be a positive integer, got {value!r}'
        )
    return int(value)


def _reference(value: object, condition_id: object, field: str) -> str | None:
    if isinstance(value, str):
        matched = _REFERENCE.fullmatch(value)
        if matched is None:
            raise ValueError(
                f'Condition {condition_id!r} {field} must be a positive integer or an exact {{param}} reference, got {value!r}'
            )
        return matched.group(1)
    _ = positive_temporal_window(value, condition_id=condition_id, field=field)
    return None


def validate_temporal_conditions(
    conditions: Iterable[Mapping[str, object]],
    parameter_domains: Mapping[str, object] | None = None,
) -> None:
    for condition in conditions:
        condition_id = condition.get('id', '<anonymous>')
        fields = [field for field in _FIELDS if field in condition]
        if len(fields) > 1:
            raise ValueError(
                f'Condition {condition_id!r} cannot specify both persistence_n and recency_n'
            )
        if fields and 'type' not in condition:
            raise ValueError(
                f'Compound condition {condition_id!r} cannot specify {fields[0]}'
            )
        for field in fields:
            reference = _reference(condition[field], condition_id, field)
            if reference is None or parameter_domains is None:
                continue
            if reference not in parameter_domains:
                raise ValueError(
                    f'Condition {condition_id!r} {field} references missing parameter {reference!r}'
                )
            values = parameter_domains[reference]
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
                raise ValueError(
                    f'Condition {condition_id!r} {field} parameter {reference!r} requires a nonempty domain'
                )
            for index, value in enumerate(cast(Sequence[object], values)):
                _ = positive_temporal_window(value, condition_id=condition_id, field=f'{field} parameter {reference!r}[{index}]')


def resolve_temporal_window(
    condition: Mapping[str, object], field: str, round_params: Mapping[str, object],
) -> int:
    condition_id = condition.get('id', '<anonymous>')
    value = condition[field]
    reference = _reference(value, condition_id, field)
    if reference is not None:
        if reference not in round_params:
            raise ValueError(
                f'Condition {condition_id!r} {field} references missing parameter {reference!r}'
            )
        value = round_params[reference]
    return positive_temporal_window(value, condition_id=condition_id, field=field)
