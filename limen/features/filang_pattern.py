import numpy as np
import polars as pl

import limen._feature_constraints as _check
from limen.features._filang_engine import PRICE_COLUMNS, Answers, pattern_answers


def filang_pattern(
    data: pl.DataFrame | pl.LazyFrame,
    definition: str,
    hold_bars: int = 1,
    interval_seconds: int = 900,
    prefix: str = 'filang',
) -> pl.DataFrame | pl.LazyFrame:

    '''
    Compute where a Filang candle pattern holds, with a fixed-length position after each match.

    The pattern is answered by the Filang engine from the optional `filang` package. A pattern
    that reads volume, trades, delta, size or imbalance also needs the 'liquidity_sum',
    'maker_liquidity' and 'no_of_trades' columns of Limen's spot klines.

    Args:
        data (pl.DataFrame | pl.LazyFrame): Klines dataset with 'datetime', 'open', 'high', 'low', 'close' and 'volume' columns
        definition (str): Filang pattern: an event, optionally followed by `where <condition>` clauses
        hold_bars (int): Number of rows a position is held from each matching close
        interval_seconds (int): Bar length of the dataset in seconds, a whole number of minutes
        prefix (str): Prefix of the three output column names

    Returns:
        pl.DataFrame | pl.LazyFrame: The input data with new columns '<prefix>_event', '<prefix>_unknown' and '<prefix>_position'
    '''

    _pattern(definition)
    _check.integers(lower=1, hold_bars=hold_bars)
    _check.integers(lower=60, interval_seconds=interval_seconds)
    if interval_seconds % 60:
        raise ValueError('interval_seconds must be a whole number of minutes')
    _check.names(prefix=prefix)

    names = (f'{prefix}_event', f'{prefix}_unknown', f'{prefix}_position')
    schema = data.collect_schema()
    missing = [column for column in ('datetime', *PRICE_COLUMNS) if column not in schema]
    if missing:
        raise ValueError(f'filang_pattern data is missing columns: {missing}')
    taken = [column for column in names if column in schema]
    if taken:
        raise ValueError(f'filang_pattern prefix {prefix!r} gives columns that already exist: {taken}')

    def append(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.height == 0:
            return frame.with_columns(*[pl.Series(name, [], dtype=pl.Int8) for name in names])
        answers = pattern_answers(frame, definition, interval_seconds)
        return frame.with_columns(
            pl.Series(names[0], (answers == 1).astype(np.int8)),
            pl.Series(names[1], (answers == -1).astype(np.int8)),
            pl.Series(names[2], _hold(answers, hold_bars)),
        )

    if isinstance(data, pl.DataFrame):
        return append(data)
    for column in names:
        schema[column] = pl.Int8
    return data.map_batches(
        append, schema=schema, predicate_pushdown=False, projection_pushdown=False,
        slice_pushdown=False, streamable=False,
    )


def _pattern(definition: object) -> None:
    if not isinstance(definition, str) or not definition.strip():
        raise ValueError('definition must be a non-empty Filang pattern')


def _hold(answers: Answers, hold_bars: int) -> Answers:
    position = np.zeros(answers.shape[0], dtype=np.int8)
    for start in (answers == 1).nonzero()[0].tolist():
        position[start:start + hold_bars] = 1
    return position
