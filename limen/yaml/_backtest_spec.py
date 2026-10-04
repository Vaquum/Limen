import re
from collections.abc import Mapping
from typing import cast

from limen.experiment._resolve_backtest_config import BACKTEST_KEYS, _validate_backtest_value
from limen.yaml.errors import YAMLError

_PARAM_REF_RE = re.compile(r'\{(\w+)\}')


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _check_backtest_spec(yaml_dict: Mapping[str, object], errors: list[YAMLError]) -> None:
    sfd = _mapping(yaml_dict.get('sfd'))
    backtest = _mapping(_mapping(sfd.get('manifest')).get('backtest'))
    params = _mapping(sfd.get('params'))
    for key in BACKTEST_KEYS:
        if key in backtest:
            value = backtest[key]
            path = f'sfd.manifest.backtest.{key}'
            if isinstance(value, str):
                ref = _PARAM_REF_RE.fullmatch(value.strip())
                if ref is None or ref.group(1) not in params:
                    errors.append(YAMLError(message=f"backtest.{key} requires a number or a {{param}} reference in sfd.params (got '{value}')", path=path))
                else:
                    candidates = params[ref.group(1)]
                    if isinstance(candidates, list):
                        for index, candidate in enumerate(cast(list[object], candidates)):
                            _check_value(key, candidate, f'{path} / sfd.params.{ref.group(1)}[{index}]', errors)
            else:
                _check_value(key, value, path, errors)


def _check_value(key: str, value: object, path: str, errors: list[YAMLError]) -> None:
    try:
        _ = _validate_backtest_value(key, value)
    except ValueError as exc:
        errors.append(YAMLError(message=str(exc), path=path))
