from limen._parallel_contract import attach as _parallel
import polars as pl

from limen.sfd.rule_based.common_strategies import strategy_signals


__all__ = ['POSITION_COLUMN', 'common_strategy_positions']

POSITION_COLUMN = 'common_strategy_position'


def common_strategy_positions(
    data: pl.DataFrame | pl.LazyFrame,
    strategy: str,
) -> pl.DataFrame | pl.LazyFrame:
    '''Append one common-strategy position column without dropping warm-up rows.

    Lazy execution evaluates the complete input frame once, so optimizations
    cannot move a filter or slice ahead of the strategy's indicator history.
    '''

    schema = data.collect_schema()
    if POSITION_COLUMN in schema:
        raise ValueError(f'common_strategy_positions output column already exists: {POSITION_COLUMN}')

    def append_positions(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.Series(POSITION_COLUMN, strategy_signals(frame, strategy)))

    if isinstance(data, pl.DataFrame):
        return append_positions(data)
    schema[POSITION_COLUMN] = pl.Int8
    return data.map_batches(
        append_positions,
        schema=schema,
        predicate_pushdown=False,
        projection_pushdown=False,
        slice_pushdown=False,
        streamable=False,
    )

_parallel(common_strategy_positions)
