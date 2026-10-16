import json
from pathlib import Path

import polars as pl
import pytest

from limen.experiment import _round_storage as storage
from tests.test_record_execution import _config, _loop, _run


@pytest.fixture(scope='module')
def recorded_rounds(tmp_path_factory):
    root = Path(__file__).parent / 'fixtures'
    bars = pl.read_parquet(root / 'spot_1h_20240101_20241231.parquet').head(1200)
    captured = {}
    for kind in ('rule', 'binary', 'ridge'):
        directory = tmp_path_factory.mktemp('storage-' + kind)
        loop = _loop(_config(kind), bars, directory)
        _run(loop, record_execution=True, record_model_outputs=True)
        with storage.open_text(storage.round_path(directory)) as stream:
            captured[kind] = [json.loads(line) for line in stream]
    return captured


def test_compressed_equals_uncompressed(tmp_path, recorded_rounds):
    for kind, records in recorded_rounds.items():
        legacy, compact = tmp_path / (kind + '-legacy'), tmp_path / (kind + '-compact')
        legacy.mkdir()
        compact.mkdir()
        (compact / 'metadata.json').write_text(json.dumps({'round_storage': storage.FORMAT}))
        original, compressed = storage.round_path(legacy), storage.round_path(compact)
        for record in records:
            storage.append_record(original, record)
            storage.append_record(compressed, record)
        with storage.open_text(original) as left, storage.open_text(compressed) as right:
            assert list(left) == list(right)
        assert original.read_text() == ''.join(json.dumps(record) + '\n' for record in records)
