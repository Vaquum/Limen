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
    return val.format(**params) if isinstance(val, str) else val


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
        return (col > other) & (col.shift(1) <= other.shift(1))
    if direction == 'below':
        return (col < other) & (col.shift(1) >= other.shift(1))
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

    col = pl.col(column)

    if direction == 'rising':
        return col > col.shift(lookback)
    if direction == 'falling':
        return col < col.shift(lookback)
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

    return expr.cast(pl.Int8).rolling_sum(n) == n


def with_recency(expr: pl.Expr, n: int) -> pl.Expr:

    '''
    Wrap a predicate expression to require it to have been True within the last n bars.

    Args:
        expr (pl.Expr): Boolean predicate expression
        n (int): Lookback window in bars

    Returns:
        pl.Expr: Boolean expression
    '''

    return expr.cast(pl.Int8).rolling_sum(n) >= 1


def polars_expr(expr_string: str, params: dict) -> pl.Expr:

    '''
    Evaluate a raw polars expression string with parameter substitution.

    Args:
        expr_string (str): Polars expression string with optional {param} placeholders
        params (dict): Parameter values for substitution

    Returns:
        pl.Expr: Evaluated polars expression

    NOTE: Uses eval() with restricted globals — __builtins__ is empty, only pl is available.
    '''

    resolved = expr_string.format(**params)
    return eval(resolved, {'pl': pl, '__builtins__': {}}, {})  # noqa: S307


def build_predicate(condition: dict, round_params: dict) -> pl.Expr:

    '''
    Route a condition config dict to the appropriate predicate function.

    Args:
        condition (dict): Condition config with 'type' key and type-specific fields
        round_params (dict): Parameter values for template substitution

    Returns:
        pl.Expr: Boolean predicate expression
    '''

    ptype = condition['type']

    if ptype == 'threshold':
        return threshold(
            column=_fmt(condition['column'], round_params),
            operator=condition['operator'],
            value=float(_fmt(condition['value'], round_params)),
        )

    if ptype == 'relative':
        return relative(
            column=_fmt(condition['column'], round_params),
            operator=condition['operator'],
            other_column=_fmt(condition['other_column'], round_params),
        )

    if ptype == 'crossover':
        return crossover(
            column=_fmt(condition['column'], round_params),
            other_column=_fmt(condition['other_column'], round_params),
            direction=condition.get('direction', 'above'),
        )

    if ptype == 'slope':
        return slope(
            column=_fmt(condition['column'], round_params),
            direction=condition.get('direction', 'rising'),
            lookback=int(_fmt(condition.get('lookback', 1), round_params)),
        )

    if ptype == 'polars_expr':
        return polars_expr(condition['expr'], round_params)

    raise ValueError(f'Unknown predicate type: {ptype!r}')
