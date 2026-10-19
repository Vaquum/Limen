from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl


@dataclass(frozen=True)
class FrameSnapshot:
    path: str
    checksum: str

    def read(self) -> pl.DataFrame:
        payload = Path(self.path).read_bytes()
        if sha256(payload).hexdigest() != self.checksum:
            raise ValueError('Frozen parallel source snapshot changed')
        return pl.DataFrame.deserialize(BytesIO(payload))


@dataclass(frozen=True)
class SourceSnapshot:
    learning: FrameSnapshot
    execution: FrameSnapshot | None
    funding: FrameSnapshot | None


def _write(frame: pl.DataFrame, directory: Path, name: str) -> FrameSnapshot:
    buffer = BytesIO()
    frame.serialize(buffer)
    payload = buffer.getvalue()
    path = directory / name
    _ = path.write_bytes(payload)
    path.chmod(0o400)
    return FrameSnapshot(str(path), sha256(payload).hexdigest())


@contextmanager
def snapshot_sources(learning: pl.DataFrame, execution: pl.DataFrame | None,
                     funding: pl.DataFrame | None, directory: Path) -> Iterator[SourceSnapshot]:
    with TemporaryDirectory(prefix='.limen-parallel-', dir=directory) as temporary:
        root = Path(temporary).resolve()
        yield SourceSnapshot(_write(learning, root, 'learning'),
                             _write(execution, root, 'execution') if execution is not None else None,
                             _write(funding, root, 'funding') if funding is not None else None)


__all__: list[str] = []
