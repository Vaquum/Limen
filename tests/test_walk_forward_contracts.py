'''Property-based tests for the purged and embargoed walk-forward splitter.

Hypothesis generates frame lengths and fold geometries against the
no-leakage laws of ``split_walk_forward``: every train row strictly
precedes its fold's test start minus the purge gap, no train row falls
inside the embargo window after any earlier test window, test windows
partition the tail contiguously without overlap, returned rows are the
source rows unchanged, and degenerate geometries fail loud. The
registered profile is deterministic so the required CI gate cannot
flake.
'''

import polars as pl
import pytest
from hypothesis import HealthCheck
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from limen.data.utils import split_walk_forward


MAX_EXAMPLES = 50

settings.register_profile(
    'ci',
    settings(
        derandomize=True,
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    ),
)
settings.load_profile('ci')


@st.composite
def _geometries(draw: st.DrawFn) -> tuple[int, int, int, int, int, bool]:
    n_folds = draw(st.integers(min_value=1, max_value=5))
    test_bars = draw(st.integers(min_value=1, max_value=8))
    purge_bars = draw(st.integers(min_value=0, max_value=6))
    embargo_bars = draw(st.integers(min_value=0, max_value=6))
    anchored = draw(st.booleans())
    floor = n_folds * test_bars + purge_bars + 1
    total = draw(st.integers(min_value=floor, max_value=floor + 40))
    return total, n_folds, test_bars, purge_bars, embargo_bars, anchored


def _frame(total: int) -> pl.DataFrame:
    return pl.DataFrame({'row_id': list(range(total)), 'value': [float(i) for i in range(total)]})


def _folds_or_reject(geometry: tuple[int, int, int, int, int, bool]) -> list[tuple[pl.DataFrame, pl.DataFrame]]:

    '''
    Split the generated frame, rejecting examples that fail loud on
    an empty rolling train window (contract-conformant per the spec:
    embargoed rolling geometries may legitimately raise rather than
    return degenerate folds).

    Args:
        geometry (tuple): Generated (total, n_folds, test_bars, purge_bars, embargo_bars, anchored)

    Returns:
        list[tuple[pl.DataFrame, pl.DataFrame]]: Ordered (train, test) pairs
    '''

    total, n_folds, test_bars, purge_bars, embargo_bars, anchored = geometry
    try:
        return split_walk_forward(
            _frame(total),
            n_folds=n_folds,
            test_bars=test_bars,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            anchored=anchored,
        )
    except ValueError as exc:
        assert str(exc).startswith('split_walk_forward')
        assert 'train window is empty' in str(exc)
        assert not anchored
        assume(False)
        raise


def _fold_rows(geometry: tuple[int, int, int, int, int, bool]) -> list[tuple[list[int], list[int]]]:
    folds = _folds_or_reject(geometry)
    return [(train['row_id'].to_list(), test['row_id'].to_list()) for train, test in folds]


@given(geometry=_geometries())
def test_purge_gap_holds(geometry: tuple[int, int, int, int, int, bool]) -> None:
    total, n_folds, test_bars, purge_bars, _, _ = geometry
    first_test_start = total - n_folds * test_bars

    for fold, (train_rows, _) in enumerate(_fold_rows(geometry)):
        test_start = first_test_start + fold * test_bars
        assert train_rows
        assert max(train_rows) < test_start - purge_bars


@given(geometry=_geometries())
def test_embargo_holds(geometry: tuple[int, int, int, int, int, bool]) -> None:
    total, n_folds, test_bars, _, embargo_bars, _ = geometry
    first_test_start = total - n_folds * test_bars

    for fold, (train_rows, _) in enumerate(_fold_rows(geometry)):
        embargoed = {
            row
            for earlier in range(fold)
            for row in range(
                first_test_start + (earlier + 1) * test_bars,
                first_test_start + (earlier + 1) * test_bars + embargo_bars,
            )
        }
        assert embargoed.isdisjoint(train_rows)


@given(geometry=_geometries())
def test_fold_partition_laws(geometry: tuple[int, int, int, int, int, bool]) -> None:
    total, n_folds, test_bars, _, _, anchored = geometry
    first_test_start = total - n_folds * test_bars
    fold_rows = _fold_rows(geometry)

    assert len(fold_rows) == n_folds

    for fold, (train_rows, test_rows) in enumerate(fold_rows):
        test_start = first_test_start + fold * test_bars
        assert test_rows == list(range(test_start, test_start + test_bars))
        assert set(train_rows).isdisjoint(test_rows)
        assert train_rows == sorted(train_rows)
        if anchored:
            assert train_rows[0] == 0
        else:
            assert len(train_rows) <= first_test_start


@given(geometry=_geometries())
def test_rows_pass_through_unchanged(geometry: tuple[int, int, int, int, int, bool]) -> None:
    source = _frame(geometry[0])
    folds = _folds_or_reject(geometry)

    for train, test in folds:
        for part in (train, test):
            expected = source.filter(pl.col('row_id').is_in(part['row_id'].to_list()))
            assert part.equals(expected)


def test_degenerate_geometry_raises() -> None:
    frame = _frame(12)

    with pytest.raises(ValueError, match=r'^split_walk_forward n_folds'):
        split_walk_forward(frame, n_folds=0, test_bars=1, purge_bars=0, embargo_bars=0, anchored=True)

    with pytest.raises(ValueError, match=r'^split_walk_forward test_bars'):
        split_walk_forward(frame, n_folds=1, test_bars=0, purge_bars=0, embargo_bars=0, anchored=True)

    with pytest.raises(ValueError, match=r'^split_walk_forward purge_bars'):
        split_walk_forward(frame, n_folds=1, test_bars=1, purge_bars=-1, embargo_bars=0, anchored=True)

    with pytest.raises(ValueError, match=r'^split_walk_forward embargo_bars'):
        split_walk_forward(frame, n_folds=1, test_bars=1, purge_bars=0, embargo_bars=-1, anchored=True)

    with pytest.raises(ValueError, match=r'^split_walk_forward geometry leaves no train rows'):
        split_walk_forward(frame, n_folds=3, test_bars=4, purge_bars=0, embargo_bars=0, anchored=True)

    with pytest.raises(ValueError, match=r'^split_walk_forward geometry leaves no train rows'):
        split_walk_forward(frame, n_folds=2, test_bars=4, purge_bars=4, embargo_bars=0, anchored=False)

    with pytest.raises(ValueError, match=r'^split_walk_forward fold 2 train window is empty'):
        split_walk_forward(_frame(8), n_folds=3, test_bars=2, purge_bars=0, embargo_bars=10, anchored=False)


@given(geometry=_geometries())
def test_degenerate_short_frame_raises(geometry: tuple[int, int, int, int, int, bool]) -> None:
    _, n_folds, test_bars, purge_bars, embargo_bars, anchored = geometry
    short_total = n_folds * test_bars + purge_bars

    with pytest.raises(ValueError, match=r'^split_walk_forward geometry leaves no train rows'):
        split_walk_forward(
            _frame(short_total),
            n_folds=n_folds,
            test_bars=test_bars,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            anchored=anchored,
        )
