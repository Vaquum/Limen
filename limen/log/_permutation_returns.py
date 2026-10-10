import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ['TrialReturnsWriter']

_SCHEMA = pa.schema([('trial', pa.string()), ('bar', pa.int64()), ('net_return', pa.float64())])


class TrialReturnsWriter:
    def __init__(self, directory: Path) -> None:
        self.path = directory / 'trial_returns.parquet'
        self.temporary = directory / 'trial_returns.parquet.tmp'
        self.writer: pq.ParquetWriter | None = None

    def append(self, trial: str, tracks: list[list[float]]) -> None:
        values = [value for track in tracks for value in track]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError('split_walk_forward requires a non-empty finite execution return track')
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.temporary, _SCHEMA)
        table = pa.Table.from_pydict(
            {'trial': [trial] * len(values), 'bar': list(range(len(values))), 'net_return': values},
            schema=_SCHEMA,
        )
        self.writer.write_table(table, row_group_size=len(values))

    def finish(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.temporary.replace(self.path)
