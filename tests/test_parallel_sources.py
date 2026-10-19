import pickle
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen.experiment._parallel_sources import snapshot_sources
from tests.test_fixed_trade_sources import _funding_frame
from tests.test_parallel_rounds import recorded_source


def test_recorded_sources_round_trip_with_small_startup_reference(tmp_path):
    learning = recorded_source()
    funding = _funding_frame()
    with snapshot_sources(learning, learning, funding, tmp_path) as snapshot:
        assert len(pickle.dumps(snapshot)) < 2048
        for frame, saved in ((learning, snapshot.learning), (learning, snapshot.execution), (funding, snapshot.funding)):
            assert saved is not None
            assert_frame_equal(saved.read(), frame, check_exact=True)
            assert Path(saved.path).parent.parent == tmp_path
            assert Path(saved.path).stat().st_mode & 0o222 == 0
        restored = snapshot.learning.read()
        restored.replace_column(0, restored.to_series(0).reverse())
        assert_frame_equal(snapshot.learning.read(), learning, check_exact=True)
        path = Path(snapshot.learning.path)
    assert not path.parent.exists()


def test_snapshot_rejects_changed_payload_and_cleans_after_failure(tmp_path):
    path = None
    with pytest.raises(ValueError, match='snapshot changed'), snapshot_sources(recorded_source(), None, None, tmp_path) as snapshot:
        assert snapshot.execution is snapshot.funding is None
        path = Path(snapshot.learning.path)
        path.chmod(0o600)
        _ = path.write_bytes(path.read_bytes()[:-1])
        snapshot.learning.read()
    assert path is not None and not path.parent.exists()


def test_snapshot_preserves_source_schema(tmp_path):
    data = recorded_source()
    data = pl.concat([data.head(400), data.slice(400)], rechunk=False)
    with snapshot_sources(data, None, None, tmp_path) as snapshot:
        assert_frame_equal(snapshot.learning.read(), data, check_exact=True)
