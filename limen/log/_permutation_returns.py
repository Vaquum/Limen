import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import polars as pl

__all__ = ['TrialReturnsWriter']


class _ParquetWriter(Protocol):
    def write_table(self, table: object, *, row_group_size: int) -> None: ...
    def close(self) -> None: ...


class _ArrowTable(Protocol):
    schema: object


class _ArrowFrame(Protocol):
    def to_arrow(self) -> _ArrowTable: ...


class TrialReturnsWriter:
    def __init__(self, directory: Path) -> None:
        self.path, self.temporary = directory / 'trial_returns.parquet', directory / 'trial_returns.parquet.tmp'
        self.writer: _ParquetWriter | None = None

    def append(self, trial: str, tracks: list[list[float]]) -> None:
        import pyarrow.parquet as pq

        values = [value for track in tracks for value in track]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError('split_walk_forward requires a non-empty finite execution return track')
        frame = pl.DataFrame(
            {'trial': [trial] * len(values), 'bar': list(range(len(values))), 'net_return': values},
            schema={'trial': pl.String, 'bar': pl.Int64, 'net_return': pl.Float64},
        )
        table = cast(_ArrowFrame, frame).to_arrow()
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            create_writer = cast(Callable[[Path, object], _ParquetWriter], pq.ParquetWriter)
            self.writer = create_writer(self.temporary, table.schema)
        self.writer.write_table(table, row_group_size=len(values))

    def finish(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            _ = self.temporary.replace(self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
        self.temporary.unlink(missing_ok=True)
