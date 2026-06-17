from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import gzip
import os
import tarfile
import tempfile

from setuptools import build_meta as _setuptools_build_meta


build_wheel = _setuptools_build_meta.build_wheel
build_editable = _setuptools_build_meta.build_editable
prepare_metadata_for_build_wheel = _setuptools_build_meta.prepare_metadata_for_build_wheel
get_requires_for_build_wheel = _setuptools_build_meta.get_requires_for_build_wheel
get_requires_for_build_sdist = _setuptools_build_meta.get_requires_for_build_sdist


def build_sdist(
    sdist_directory: str,
    config_settings: Mapping[str, Sequence[str] | str] | None = None,
) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        built_name = _setuptools_build_meta.build_sdist(tmpdir, config_settings)
        source = Path(tmpdir, built_name)
        target = Path(sdist_directory, built_name)
        _normalize_tar_gz(source, target)
    return target.name


def _normalize_tar_gz(source: Path, target: Path) -> None:
    epoch = int(os.environ.get('SOURCE_DATE_EPOCH', '0'))
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []

    with tarfile.open(source, 'r:gz') as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            info = tarfile.TarInfo(member.name)
            info.type = member.type
            info.linkname = member.linkname
            info.size = member.size if member.isfile() else 0
            info.mtime = epoch
            info.uid = 0
            info.gid = 0
            info.uname = ''
            info.gname = ''
            info.mode = 0o755 if member.isdir() else 0o644
            data = None
            if member.isfile():
                extracted = archive.extractfile(member)
                data = extracted.read() if extracted is not None else b''
            members.append((info, data))

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=epoch) as gz:
            with tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as archive:
                for info, data in members:
                    if data is None:
                        archive.addfile(info)
                    else:
                        archive.addfile(info, _BytesReader(data))


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._data) - self._offset
        start = self._offset
        end = min(len(self._data), start + size)
        self._offset = end
        return self._data[start:end]
