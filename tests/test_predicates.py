import polars as pl

from limen.sfd.rule_based.config import RuleBasedConfig
from limen.sfd.rule_based.predicates import build_predicate
from limen.sfd.rule_based.predicates import crossover
from limen.sfd.rule_based.predicates import sql_expr
from limen.sfd.rule_based.predicates import relative
from limen.sfd.rule_based.predicates import slope
from limen.sfd.rule_based.predicates import threshold
from limen.sfd.rule_based.predicates import with_persistence
from limen.sfd.rule_based.predicates import with_recency


def _eval(expr: pl.Expr, df: pl.DataFrame) -> list:
    return df.select(expr.alias('result'))['result'].to_list()


def test_threshold() -> None:
    df = pl.DataFrame({'rsi': [25.0, 35.0, 29.0]})
    assert _eval(threshold('rsi', '<', 30.0), df) == [True, False, True]


def test_relative() -> None:
    df = pl.DataFrame({'close': [100.0, 200.0, 150.0], 'sma': [150.0, 150.0, 150.0]})
    assert _eval(relative('close', '>', 'sma'), df) == [False, True, False]


def test_crossover_above() -> None:
    df = pl.DataFrame({'fast': [1.0, 2.0, 3.0], 'slow': [2.0, 2.0, 2.0]})
    result = _eval(crossover('fast', 'slow', direction='above'), df)
    assert result[2] is True
    assert result[1] is False


def test_crossover_below() -> None:
    df = pl.DataFrame({'fast': [3.0, 2.0, 1.0], 'slow': [2.0, 2.0, 2.0]})
    result = _eval(crossover('fast', 'slow', direction='below'), df)
    assert result[2] is True
    assert result[1] is False


def test_slope_rising() -> None:
    df = pl.DataFrame({'price': [1.0, 2.0, 1.5]})
    result = _eval(slope('price', direction='rising', lookback=1), df)
    assert result[1] is True
    assert result[2] is False


def test_slope_falling() -> None:
    df = pl.DataFrame({'price': [3.0, 2.0, 2.5]})
    result = _eval(slope('price', direction='falling', lookback=1), df)
    assert result[1] is True
    assert result[2] is False


def test_with_persistence() -> None:
    df = pl.DataFrame({'val': [True, True, True, False, True]})
    result = _eval(with_persistence(pl.col('val'), 3), df)
    assert result[2] is True
    assert result[3] is False


def test_with_recency() -> None:
    df = pl.DataFrame({'val': [False, True, False, False]})
    result = _eval(with_recency(pl.col('val'), 2), df)
    assert result[1] is True
    assert result[2] is True
    assert result[3] is False


def test_sql_expr_escape_hatch() -> None:
    df = pl.DataFrame({'volume': [100.0, 300.0], 'avg_vol': [200.0, 200.0]})
    expr = sql_expr('volume > avg_vol * {multiplier}', {'multiplier': 1.2})
    assert _eval(expr, df) == [False, True]


def test_build_predicate_threshold_with_param_substitution() -> None:
    df = pl.DataFrame({'rsi_14': [20.0, 40.0]})
    condition = {'type': 'threshold', 'column': 'rsi_{period}', 'operator': '<', 'value': 30}
    assert _eval(build_predicate(condition, {'period': 14}), df) == [True, False]


def test_build_predicate_unknown_type_raises() -> None:
    try:
        build_predicate({'type': 'magic'}, {})
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'Unknown predicate type' in str(e)


def test_threshold_unknown_operator_raises() -> None:
    try:
        threshold('col', '??', 1.0)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'Unknown operator' in str(e)


def test_sql_expr_missing_param_raises() -> None:
    try:
        sql_expr('volume > {threshold}', {})
        assert False, 'Expected ValueError for missing param'
    except ValueError as e:
        assert 'missing template parameter' in str(e)


def test_rule_based_config_cycle_detection() -> None:
    conditions = [
        {'id': 'a', 'type': 'threshold', 'column': 'x', 'operator': '>', 'value': 0},
        {'id': 'b', 'operator': 'and', 'operands': ['a', 'c']},
        {'id': 'c', 'operator': 'and', 'operands': ['b']},
    ]
    try:
        RuleBasedConfig(conditions=conditions, entry='b')
        assert False, 'Expected ValueError for cyclic reference'
    except ValueError as e:
        assert 'Cyclic' in str(e) or 'cycle' in str(e).lower()


def test_rule_based_config_operands_must_be_list() -> None:
    conditions = [
        {'id': 'a', 'type': 'threshold', 'column': 'x', 'operator': '>', 'value': 0},
        {'id': 'b', 'operator': 'and', 'operands': 'a'},
    ]
    try:
        RuleBasedConfig(conditions=conditions, entry='b')
        assert False, 'Expected ValueError for non-list operands'
    except ValueError as e:
        assert 'list' in str(e).lower()


def test_rule_based_config_operand_elements_must_be_strings() -> None:
    conditions = [
        {'id': 'a', 'type': 'threshold', 'column': 'x', 'operator': '>', 'value': 0},
        {'id': 'b', 'operator': 'and', 'operands': ['a', 42]},
    ]
    try:
        RuleBasedConfig(conditions=conditions, entry='b')
        assert False, 'Expected ValueError for non-string operand'
    except ValueError as e:
        assert 'string' in str(e).lower() or 'str' in str(e).lower()
