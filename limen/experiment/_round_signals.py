import math
import re
from collections.abc import Mapping, Sequence
from typing import cast

__all__: list[str] = []


def count(value: object) -> int | None:
    if not isinstance(value, list):
        return None
    values = cast(list[object], value)
    if any(item is not None and type(item) not in (bool, int, float) or
           type(item) is float and not math.isfinite(cast(float, item)) for item in values):
        return None
    return len(values)


def reference(value: object, *, signal: bool) -> tuple[str, int | None] | None:
    if not signal:
        return (value, None) if isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None else None
    if not isinstance(value, dict) or set(cast(dict[str, object], value)) != {'sha256', 'count'}:
        return None
    fields = cast(dict[str, object], value)
    digest, length = fields['sha256'], fields['count']
    if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None or type(length) is not int or length < 0:
        return None
    return digest, length


def aligned(record: Mapping[str, object], fields: Sequence[str], *, eligible: bool = False) -> bool:
    values = [record.get(field) for field in fields]
    if eligible:
        values = [value for value in values if count(value) is not None]
    if not values:
        return True
    if isinstance(execution := record.get('execution'), Mapping):
        values.append(cast(Mapping[str, object], execution).get('pos'))
    lengths = {len(cast(list[object], value)) for value in values if isinstance(value, list)}
    return len(lengths) <= 1
