from __future__ import annotations

from typing import TYPE_CHECKING, cast
from collections.abc import Mapping

import polars as pl

if TYPE_CHECKING:
    from limen.backtest.trade_contract import JsonValue, TradeInputs, TradePolicy

RULE_EXIT_VERSION = 'limen-rule-exit-v1'
EXIT_SEMANTICS = 'entry_until_exit'


def validate_definition(value: Mapping[str, JsonValue]) -> None:
    from limen.sfd.rule_based.config import RuleBasedConfig
    conditions, root = value.get('conditions'), value.get('exit')
    if not isinstance(conditions, list) or not all(isinstance(item, dict) for item in conditions) or not isinstance(root, str):
        raise ValueError('Explicit exit definition requires conditions and an exit root')
    rows = cast(list[dict[str, object]], conditions)
    if any(condition.get('type') == 'sql_expr' for condition in rows):
        raise ValueError('Explicit exit rules do not support sql_expr')
    _ = RuleBasedConfig(rows, root)


def validate_exits(inputs: TradeInputs, policy: TradePolicy) -> None:
    enabled = policy.exit_rules is not None
    present = 'exit' in inputs.signals.columns
    if enabled != present:
        raise ValueError('Explicit exits require both a resolved rule definition and aligned exit values')
    if enabled and (inputs.signals['exit'].dtype != pl.Boolean or inputs.signals['exit'].null_count()):
        raise ValueError('Explicit exit values must be nonmissing booleans')


def evidence(signals: pl.DataFrame) -> dict[str, JsonValue]:
    if signals['exit'].dtype != pl.Boolean or signals['exit'].null_count():
        raise ValueError('Exit evidence must contain nonmissing boolean decisions')
    selected = signals.select('row_id', 'available_at_ns', 'exit')
    if selected['row_id'].dtype != pl.String or selected['available_at_ns'].dtype not in (pl.Int64, pl.Null):
        raise ValueError('Exit evidence requires string identities and declared integer availability or null')
    if selected['row_id'].null_count() or selected['row_id'].is_duplicated().any():
        raise ValueError('Exit evidence requires unique nonmissing source identities')
    return {'version': 'limen-rule-exit-evidence-v1', 'rows': cast('list[JsonValue]', selected.to_dicts())}


__all__: list[str] = []
