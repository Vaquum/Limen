from typing import Any

import polars as pl


_OPS = {
    '<':  lambda a, b: a < b,
    '<=': lambda a, b: a <= b,
    '>':  lambda a, b: a > b,
    '>=': lambda a, b: a >= b,
    '==': lambda a, b: a == b,
    '!=': lambda a, b: a != b,
}


def _fmt(val: Any, params: dict) -> Any:
    if not isinstance(val, str):
        return val
    try:
        return val.format(**params)
    except KeyError as e:
        raise ValueError(
            f'Template placeholder {e} not found in round_params — '
            f'available keys: {sorted(params)}'
        ) from e


def _coerce_value(val: Any) -> Any:
    if not isinstance(val, str):
        return val
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _op_expr(left: pl.Expr, operator: str, right: Any) -> pl.Expr:
    if operator not in _OPS:
        raise ValueError(f'Unknown operator: {operator!r}')
    return _OPS[operator](left, right)


def threshold(column: str, operator: str, value: Any) -> pl.Expr:

    '''
    Compare a column against a constant value.

    Args:
        column (str): Column name
        operator (str): Comparison operator — one of <, <=, >, >=, ==, !=
        value (Any): Constant to compare against

    Returns:
        pl.Expr: Boolean expression
    '''

    return _op_expr(pl.col(column), operator, value)


def relative(column: str, operator: str, other_column: str) -> pl.Expr:

    '''
    Compare a column against another column.

    Args:
        column (str): Column name
        operator (str): Comparison operator — one of <, <=, >, >=, ==, !=
        other_column (str): Column name to compare against

    Returns:
        pl.Expr: Boolean expression
    '''

    return _op_expr(pl.col(column), operator, pl.col(other_column))


def crossover(column: str, other_column: str, direction: str = 'above') -> pl.Expr:

    '''
    Detect when a column crosses another column.

    Args:
        column (str): Column name
        other_column (str): Column name to cross
        direction (str): 'above' — column crosses above other_column;
            'below' — column crosses below other_column

    Returns:
        pl.Expr: Boolean expression, True on the bar of the crossover
    '''

    col = pl.col(column)
    other = pl.col(other_column)

    if direction == 'above':
        return ((col > other) & (col.shift(1) <= other.shift(1))).fill_null(False)
    if direction == 'below':
        return ((col < other) & (col.shift(1) >= other.shift(1))).fill_null(False)
    raise ValueError(f'Unknown crossover direction: {direction!r}')


def slope(column: str, direction: str = 'rising', lookback: int = 1) -> pl.Expr:

    '''
    Detect whether a column is rising or falling over a lookback window.

    Args:
        column (str): Column name
        direction (str): 'rising' or 'falling'
        lookback (int): Number of bars to look back

    Returns:
        pl.Expr: Boolean expression
    '''

    if lookback <= 0:
        raise ValueError(f'lookback must be a positive integer, got {lookback}')

    col = pl.col(column)

    if direction == 'rising':
        return (col > col.shift(lookback)).fill_null(False)
    if direction == 'falling':
        return (col < col.shift(lookback)).fill_null(False)
    raise ValueError(f'Unknown slope direction: {direction!r}')


def with_persistence(expr: pl.Expr, n: int) -> pl.Expr:

    '''
    Wrap a predicate expression to require it to be True for n consecutive bars.

    Args:
        expr (pl.Expr): Boolean predicate expression
        n (int): Number of consecutive bars required

    Returns:
        pl.Expr: Boolean expression
    '''

    if n <= 0:
        raise ValueError(f'n must be a positive integer, got {n}')

    return (expr.cast(pl.Int8).rolling_sum(n, min_samples=n) == n).fill_null(False)


def with_recency(expr: pl.Expr, n: int) -> pl.Expr:

    '''
    Wrap a predicate expression to require it to have been True within the last n bars.

    Args:
        expr (pl.Expr): Boolean predicate expression
        n (int): Lookback window in bars

    Returns:
        pl.Expr: Boolean expression
    '''

    if n <= 0:
        raise ValueError(f'n must be a positive integer, got {n}')

    return (expr.cast(pl.Int8).rolling_sum(n, min_samples=1) >= 1).fill_null(False)


def sql_expr(expr_string: str, params: dict) -> pl.Expr:

    '''
    Parse a SQL expression string with parameter substitution into a polars expression.

    Column names are referenced directly without wrappers — e.g.
    'volume > avg_volume_20 * 2.0' rather than 'pl.col(...)'.
    Parameter placeholders use Python format syntax: 'rsi_14 < {rsi_threshold}'.

    Args:
        expr_string (str): SQL expression string with optional {param} placeholders
        params (dict): Parameter values for substitution

    Returns:
        pl.Expr: Polars expression
    '''

    try:
        resolved = expr_string.format(**params)
    except KeyError as e:
        raise ValueError(
            f'sql_expr missing template parameter {e} — '
            f'available keys: {sorted(params)}'
        ) from e
    return pl.sql_expr(resolved)


def build_predicate(condition: dict, round_params: dict) -> pl.Expr:

    '''
    Route a condition config dict to the appropriate predicate function.

    Args:
        condition (dict): Condition config with 'type' key and type-specific fields.
            Optional 'persistence_n' or 'recency_n' fields wrap the result with
            the corresponding temporal modifier.
        round_params (dict): Parameter values for template substitution

    Returns:
        pl.Expr: Boolean predicate expression
    '''

    ptype = condition['type']

    if ptype == 'threshold':
        expr = threshold(
            column=_fmt(condition['column'], round_params),
            operator=condition['operator'],
            value=_coerce_value(_fmt(condition['value'], round_params)),
        )

    elif ptype == 'relative':
        expr = relative(
            column=_fmt(condition['column'], round_params),
            operator=condition['operator'],
            other_column=_fmt(condition['other_column'], round_params),
        )

    elif ptype == 'crossover':
        expr = crossover(
            column=_fmt(condition['column'], round_params),
            other_column=_fmt(condition['other_column'], round_params),
            direction=condition.get('direction', 'above'),
        )

    elif ptype == 'slope':
        expr = slope(
            column=_fmt(condition['column'], round_params),
            direction=condition.get('direction', 'rising'),
            lookback=int(_fmt(condition.get('lookback', 1), round_params)),
        )

    elif ptype == 'sql_expr':
        expr = sql_expr(condition['expr'], round_params)

    else:
        raise ValueError(f'Unknown predicate type: {ptype!r}')

    if 'persistence_n' in condition and 'recency_n' in condition:
        raise ValueError(f'Condition {condition.get("id")!r} cannot specify both persistence_n and recency_n')
    if 'persistence_n' in condition:
        expr = with_persistence(expr, int(condition['persistence_n']))
    elif 'recency_n' in condition:
        expr = with_recency(expr, int(condition['recency_n']))

    return expr
