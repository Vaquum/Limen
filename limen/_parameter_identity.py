from collections.abc import Mapping, Sequence
import hashlib
import json
from numbers import Real
from typing import TypeVar

__all__: list[str] = []
_Params = TypeVar('_Params', bound=Mapping[str, Sequence[object]])


def execution_hash(combo: Mapping[str, object]) -> str:
    clean = {key: value for key, value in combo.items() if not key.startswith('_')}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()


def choice_issue(param: str, values: Sequence[object]) -> tuple[int, str] | None:
    seen: dict[str, int] = {}
    for index, value in enumerate(values):
        try:
            identity = execution_hash({param: value})
        except (TypeError, ValueError) as error:
            return index, f"Parameter '{param}' choice {index} has no execution identity: {error}"
        if identity in seen:
            return index, f"Parameter '{param}' choices {seen[identity]} and {index} collide under the execution identity"
        seen[identity] = index
    return None


def admit_choices(params: _Params) -> _Params:
    for param, values in params.items():
        if issue := choice_issue(param, values):
            raise ValueError(issue[1])
    return params


def logged_choices(param: str, value: object, choices: Sequence[object]) -> tuple[object, ...]:
    identity = execution_hash({param: value})
    matches = tuple(choice for choice in choices if execution_hash({param: choice}) == identity or
                    (isinstance(value, float) and isinstance(choice, Real) and choice == value))
    if len(matches) > 1:
        raise ValueError(f"Pruning log value {value!r} for '{param}' ambiguously represents distinct execution identities")
    return matches
