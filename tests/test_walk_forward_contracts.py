from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from limen.data.utils import split_walk_forward


_RECORDED = pl.read_parquet(
    Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
).head(128)
_Geometry = tuple[int, int, int, int, int, bool]


@st.composite
def _geometries(draw: st.DrawFn) -> _Geometry:
    n_folds = draw(st.integers(1, 5))
    test_bars = draw(st.integers(1, 8))
    purge_bars = draw(st.integers(0, 6))
    embargo_bars = draw(st.integers(0, 16))
    span = draw(st.integers(1, 40))
    anchored = draw(st.booleans())
    total = n_folds * test_bars + purge_bars + span
    return total, n_folds, test_bars, purge_bars, embargo_bars, anchored


def _split(source: pl.DataFrame, geometry: _Geometry) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    _, n_folds, test_bars, purge_bars, embargo_bars, anchored = geometry
    return split_walk_forward(
        source, n_folds=n_folds, test_bars=test_bars,
        purge_bars=purge_bars, embargo_bars=embargo_bars, anchored=anchored,
    )


@settings(max_examples=50, derandomize=True, deadline=None)
@given(geometry=_geometries())
def test_fold_partition_laws(geometry: _Geometry) -> None:
    total, n_folds, test_bars, purge_bars, embargo_bars, anchored = geometry
    source = _RECORDED.head(total)
    original = source.clone()
    first_test = total - n_folds * test_bars
    expected_train = []
    for fold in range(n_folds):
        test_start = first_test + fold * test_bars
        embargoed = {
            row
            for earlier in range(fold)
            for row in range(
                first_test + (earlier + 1) * test_bars,
                first_test + (earlier + 1) * test_bars + embargo_bars,
            )
        }
        start = 0 if anchored else fold * test_bars
        expected_train.append([
            row for row in range(start, test_start - purge_bars)
            if row not in embargoed
        ])

    if any(not rows for rows in expected_train):
        with pytest.raises(ValueError, match=r'^split_walk_forward'):
            _split(source, geometry)
        assert source.equals(original)
        return

    folds = _split(source, geometry)
    assert len(folds) == n_folds
    for fold, (train, test) in enumerate(folds):
        test_start = first_test + fold * test_bars
        assert train.schema == source.schema == test.schema
        assert train.rows() == [source.row(row) for row in expected_train[fold]]
        assert test.equals(source.slice(test_start, test_bars))
    assert pl.concat([test for _, test in folds]).equals(source.tail(n_folds * test_bars))
    assert source.equals(original)


@pytest.mark.parametrize('anchored', [True, False])
def test_purge_gap_holds(anchored: bool) -> None:
    source = _RECORDED.head(24)
    folds = split_walk_forward(
        source, n_folds=3, test_bars=4, purge_bars=3, embargo_bars=0,
        anchored=anchored,
    )
    for fold, (train, _) in enumerate(folds):
        start = 0 if anchored else fold * 4
        end = 12 + fold * 4 - 3
        assert train.equals(source.slice(start, end - start))


@pytest.mark.parametrize('anchored', [True, False])
@pytest.mark.parametrize('embargo_bars', [2, 7])
def test_embargo_holds(anchored: bool, embargo_bars: int) -> None:
    source = _RECORDED.head(28)
    folds = split_walk_forward(
        source, n_folds=4, test_bars=4, purge_bars=0,
        embargo_bars=embargo_bars, anchored=anchored,
    )
    start = 0 if anchored else 12
    kept = list(range(start, 16)) + ([18, 19, 22, 23] if embargo_bars == 2 else [])
    assert folds[-1][0].rows() == [source.row(row) for row in kept]


@pytest.mark.parametrize(
    ('geometry', 'message'),
    [
        ((12, 0, 1, 0, 0, True), 'n_folds'),
        ((12, 1, 0, 0, 0, True), 'test_bars'),
        ((12, 1, 1, -1, 0, True), 'purge_bars'),
        ((12, 1, 1, 0, -1, True), 'embargo_bars'),
        ((0, 1, 1, 0, 0, True), 'geometry'),
        ((12, 3, 4, 0, 0, True), 'geometry'),
        ((12, 2, 4, 4, 0, False), 'geometry'),
        ((8, 3, 2, 0, 10, False), 'fold 2 train window is empty'),
    ],
)
def test_degenerate_geometry_raises(geometry: _Geometry, message: str) -> None:
    with pytest.raises(ValueError, match=f'^split_walk_forward {message}'):
        _split(_RECORDED.head(geometry[0]), geometry)
