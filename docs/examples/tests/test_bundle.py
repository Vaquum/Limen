import io
from pathlib import Path
import tarfile

import pytest

from build_bundle import build
from wheel_proof import archive_bytes, unpack

ROOT = Path(__file__).resolve().parents[3]


def test_archive_rejects_missing_project_schema_and_input(tmp_path):
    source = tmp_path / 'complete.tar.gz'
    build(ROOT, source)
    with tarfile.open(source) as archive:
        entries = {item.name: archive_bytes(archive, item.name) for item in archive.getmembers()}
    for missing in ('compiled-research/allocation.yaml', 'evidence.schema.json', 'retained/selection.json', 'inputs/btcusdt_funding_20240101_20240108.json'):
        changed = tmp_path / 'incomplete.tar.gz'
        with tarfile.open(changed, 'w:gz') as archive:
            for name, content in entries.items():
                if name != missing:
                    member = tarfile.TarInfo(name);member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
        with pytest.raises(ValueError, match='incomplete|inventory differs'):
            unpack(changed, tmp_path / 'unpacked', ROOT)
