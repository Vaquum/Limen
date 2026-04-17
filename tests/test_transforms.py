import numpy as np
import polars as pl
import pytest

from limen.transforms.mad_transform import mad_transform
from limen.transforms.optimize_binary_threshold import optimize_binary_threshold
from limen.transforms.quantile_trim_transform import quantile_trim_transform
from limen.transforms.winsorize_transform import winsorize_transform
from limen.transforms.zscore_transform import zscore_transform


def _make_transform_frame() -> pl.DataFrame:
    return pl.DataFrame({
        'datetime': [1, 2, 3, 4],
        'price': [10.0, 12.0, 14.0, 16.0],
        'constant': [5.0, 5.0, 5.0, 5.0],
        'label': ['a', 'b', 'c', 'd'],
    })


def test_mad_transform_scales_around_column_median_and_preserves_context_columns() -> None:
    result = mad_transform(_make_transform_frame())

    assert result.columns == ['datetime', 'label', 'price', 'constant']
    assert result['datetime'].to_list() == [1, 2, 3, 4]
    assert result['label'].to_list() == ['a', 'b', 'c', 'd']
    assert result['price'].to_list() == pytest.approx([-1.5, -0.5, 0.5, 1.5])
    assert result['constant'].to_list() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_zscore_transform_centers_numeric_values_and_guards_constant_columns() -> None:
    result = zscore_transform(_make_transform_frame())

    assert result['price'].to_list() == pytest.approx(
        [-1.161895, -0.387298, 0.387298, 1.161895],
        rel=1e-6,
    )
    assert result['constant'].to_list() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_winsorize_transform_clips_extreme_tail_values() -> None:
    data = pl.DataFrame({
        'datetime': list(range(101)),
        'value': [float(i) for i in range(100)] + [1000.0],
        'other': [float(i) for i in range(100, 0, -1)] + [-1000.0],
    })

    result = winsorize_transform(data)

    assert result.height == data.height
    assert result['datetime'].to_list() == data['datetime'].to_list()
    assert result['value'].max() == pytest.approx(99.0)
    assert result['other'].min() == pytest.approx(1.0)


def test_quantile_trim_transform_removes_tail_rows_and_preserves_nulls() -> None:
    data = pl.DataFrame({
        'datetime': list(range(202)),
        'value': [float(i) for i in range(200)] + [1000.0, None],
    })

    result = quantile_trim_transform(data)

    datetimes = result['datetime'].to_list()
    assert result.height == 200
    assert 0 not in datetimes
    assert 200 not in datetimes
    assert 201 in datetimes
    assert result['value'].null_count() == 1


def test_numeric_transforms_leave_non_numeric_frames_unchanged() -> None:
    data = pl.DataFrame({
        'datetime': [1, 2],
        'label': ['alpha', 'beta'],
    })

    for transform in (
        mad_transform,
        quantile_trim_transform,
        winsorize_transform,
        zscore_transform,
    ):
        assert transform(data).equals(data)


def test_optimize_binary_threshold_picks_best_balanced_threshold() -> None:
    y_val = np.asarray([0, 1, 1, 0], dtype=np.int8)
    y_val_proba = np.asarray([0.2, 0.8, 0.52, 0.45])

    threshold, score = optimize_binary_threshold(
        y_val,
        y_val_proba,
        threshold_min=0.2,
        threshold_max=0.7,
        threshold_step=0.1,
    )

    assert threshold == pytest.approx(0.5)
    assert score == pytest.approx(np.sqrt(0.5))


def test_optimize_binary_threshold_returns_default_when_all_thresholds_predict_zero() -> None:
    threshold, score = optimize_binary_threshold(
        np.asarray([0, 1, 1, 0], dtype=np.int8),
        np.asarray([0.01, 0.02, 0.03, 0.04]),
        threshold_min=0.2,
        threshold_max=0.3,
        threshold_step=0.1,
        default_threshold=0.35,
    )

    assert threshold == 0.35
    assert score == -1.0
