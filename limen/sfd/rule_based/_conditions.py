from collections.abc import Mapping, Sequence
from typing import cast

import polars as pl


def resolve_condition(condition: Mapping[str, object], index: Mapping[str, Mapping[str, object]], frame: pl.DataFrame) -> pl.Series:
    if 'type' in condition:
        return frame[cast(str, condition['id'])]
    operator = condition['operator']
    if operator not in ('and', 'or', 'not'):
        raise ValueError(f'Unknown logical operator: {operator!r}')
    operands = [resolve_condition(index[key], index, frame) for key in cast(Sequence[str], condition.get('operands', []))]
    if not operands:
        raise ValueError(f'Compound condition {condition.get("id")!r} has no operands')
    if operator == 'not':
        if len(operands) != 1:
            raise ValueError(f'NOT operator requires exactly 1 operand, got {len(operands)}')
        return ~operands[0]
    result = operands[0]
    for value in operands[1:]:
        result = result & value if operator == 'and' else result | value
    return result


__all__: list[str] = []
