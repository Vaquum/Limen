from dataclasses import dataclass


_VALID_OPERATORS = ('and', 'or', 'not')

_LEAF_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    'threshold':   ('column', 'operator', 'value'),
    'relative':    ('column', 'operator', 'other_column'),
    'crossover':   ('column', 'other_column'),
    'slope':       ('column',),
    'polars_expr': ('expr',),
}


def _validate_leaf(cond: dict) -> None:
    ptype = cond['type']
    if ptype not in _LEAF_REQUIRED_FIELDS:
        raise ValueError(f'Condition {cond["id"]!r} has unknown type {ptype!r}')
    for field in _LEAF_REQUIRED_FIELDS[ptype]:
        if field not in cond:
            raise ValueError(
                f'Condition {cond["id"]!r} of type {ptype!r} is missing required field {field!r}'
            )
    if 'persistence_n' in cond and 'recency_n' in cond:
        raise ValueError(f'Condition {cond["id"]!r} cannot specify both persistence_n and recency_n')


def _validate_compound(cond: dict, known_ids: set[str]) -> None:
    operator = cond.get('operator')
    if operator not in _VALID_OPERATORS:
        raise ValueError(
            f'Condition {cond["id"]!r} has unknown operator {operator!r} — must be one of {_VALID_OPERATORS}'
        )
    operands = cond.get('operands', [])
    if not operands:
        raise ValueError(f'Compound condition {cond["id"]!r} has no operands')
    if operator == 'not' and len(operands) != 1:
        raise ValueError(
            f'NOT condition {cond["id"]!r} must have exactly 1 operand, got {len(operands)}'
        )
    for op_id in operands:
        if op_id not in known_ids:
            raise ValueError(f'Operand {op_id!r} in condition {cond["id"]!r} references unknown id')


@dataclass
class RuleBasedConfig:

    '''Rule-based strategy configuration: boolean predicate conditions and entry signal id.'''

    conditions: list[dict]
    entry: str

    def __post_init__(self) -> None:

        self.conditions = list(self.conditions)

        known_ids: set[str] = set()
        for cond in self.conditions:
            if 'id' not in cond:
                raise ValueError(f'Condition missing required "id" field: {cond!r}')
            if cond['id'] in known_ids:
                raise ValueError(f'Duplicate condition id: {cond["id"]!r}')
            known_ids.add(cond['id'])

        if self.entry not in known_ids:
            raise ValueError(f'Entry id {self.entry!r} not found in conditions')

        for cond in self.conditions:
            if 'type' in cond:
                _validate_leaf(cond)
            else:
                _validate_compound(cond, known_ids)
