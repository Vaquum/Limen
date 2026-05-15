import polars as pl

from limen.experiment import MLManifest
from limen.experiment import Manifest
from limen.targets import ForwardBreakoutTarget
from limen.targets import IdentityTarget
from limen.targets import NextBarUpTarget
from limen.targets import NextReturnTarget
from limen.targets import QuantileBinaryTarget
from limen.targets import RandomBinaryTarget
from limen.targets import ThresholdBinaryTarget


def _close_series(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({'close': values})


def _roc_series(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({'roc_1': values})


def test_quantile_binary_fits_cutoff_on_train() -> None:
    train = _roc_series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    t = QuantileBinaryTarget(train, 'flag', source_column='roc_1', quantile=0.3)
    assert abs(t.cutoff - 0.7) < 1e-6


def test_quantile_binary_transform_labels_above_cutoff() -> None:
    train = _roc_series([0.1, 0.2, 0.3, 0.4, 0.5])
    t = QuantileBinaryTarget(train, 'flag', source_column='roc_1', quantile=0.4)
    data = _roc_series([0.1, 0.4, 0.9])
    result = t.transform(data, shift=0)
    labels = result['flag'].to_list()
    assert labels[2] == 1
    assert labels[0] == 0


def test_quantile_binary_shift_applied_when_nonzero() -> None:
    train = _roc_series([0.1, 0.9])
    t = QuantileBinaryTarget(train, 'flag', source_column='roc_1', quantile=0.4)
    data = _roc_series([0.9, 0.1, 0.9])
    unshifted = t.transform(data, shift=0)['flag'].to_list()
    shifted = t.transform(data, shift=-1)['flag'].to_list()
    # shift(-1) moves labels one position back; last entry becomes null → None
    assert unshifted[0] == shifted[1]
    assert shifted[-1] is None


def test_forward_breakout_labels_above_threshold() -> None:
    close = [100.0, 103.0, 100.0, 101.0, 100.0]
    data = _close_series(close)
    t = ForwardBreakoutTarget(data, 'breakout')
    result = t.transform(data, forward_periods=1, threshold=0.02, shift=0)
    assert result['breakout'][0] == 1
    assert result['breakout'][1] == 0


def test_forward_breakout_shift_applied() -> None:
    # Bar 2 produces a breakout (2.5% > 2%). After shift(-1) it should appear at bar 1.
    close = [100.0, 100.0, 100.0, 102.5, 100.0, 100.0]
    data = _close_series(close)
    t = ForwardBreakoutTarget(data, 'breakout')
    unshifted = t.transform(data, forward_periods=1, threshold=0.02, shift=0)
    shifted = t.transform(data, forward_periods=1, threshold=0.02, shift=-1)
    # shift(-1): shifted[i] == unshifted[i+1]
    assert unshifted['breakout'][2] == 1
    assert shifted['breakout'][1] == 1


def test_threshold_binary_labels_above_threshold() -> None:
    data = pl.DataFrame({'rsi': [20.0, 50.0, 80.0]})
    t = ThresholdBinaryTarget(data, 'rsi_high', source_column='rsi', threshold=60.0)
    result = t.transform(data, shift=0)
    assert result['rsi_high'].to_list() == [0, 0, 1]


def test_threshold_binary_threshold_is_fixed() -> None:
    train = pl.DataFrame({'rsi': [10.0, 20.0]})
    val = pl.DataFrame({'rsi': [70.0, 90.0]})
    t = ThresholdBinaryTarget(train, 'rsi_high', source_column='rsi', threshold=60.0)
    result = t.transform(val, shift=0)
    assert result['rsi_high'].to_list() == [1, 1]


def test_threshold_binary_shift_applied() -> None:
    data = pl.DataFrame({'v': [1.0, 5.0, 1.0, 1.0]})
    t = ThresholdBinaryTarget(data, 'flag', source_column='v', threshold=3.0)
    unshifted = t.transform(data, shift=0)
    shifted = t.transform(data, shift=-1)
    # shift(-1): label at bar 1 (v=5.0 > 3.0, label=1) moves to bar 0
    assert unshifted['flag'][1] == shifted['flag'][0]


def test_next_return_computes_percentage_return() -> None:
    data = _close_series([100.0, 110.0, 99.0])
    t = NextReturnTarget(data, 'ret')
    result = t.transform(data, periods=1, scale=100.0)
    assert abs(result['ret'][0] - 10.0) < 1e-6
    assert abs(result['ret'][1] - (-10.0)) < 1e-6


def test_next_return_respects_periods() -> None:
    data = _close_series([100.0, 100.0, 120.0])
    t = NextReturnTarget(data, 'ret')
    result = t.transform(data, periods=2, scale=100.0)
    assert abs(result['ret'][0] - 20.0) < 1e-6


def test_next_return_respects_scale() -> None:
    data = _close_series([100.0, 101.0, 100.0])
    t = NextReturnTarget(data, 'ret')
    result_pct = t.transform(data, periods=1, scale=100.0)
    result_raw = t.transform(data, periods=1, scale=1.0)
    assert abs(result_pct['ret'][0] - result_raw['ret'][0] * 100.0) < 1e-9


def test_next_bar_up_labels_next_close_above_current_close() -> None:
    data = _close_series([100.0, 101.0, 100.0, 100.0])
    t = NextBarUpTarget(data, 'next_bar_up')
    result = t.transform(data)
    assert result['next_bar_up'].to_list() == [1, 0, 0, None]


def test_next_bar_up_manifest_pipeline_adds_target() -> None:
    raw_data = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=pl.datetime(2025, 1, 1, 0, 0, 0),
            end=pl.datetime(2025, 1, 1, 7, 0, 0),
            interval='1h',
            eager=True,
        ),
        'close': [100.0, 101.0, 100.0, 102.0, 100.0, 99.0, 100.0, 101.0],
    })
    manifest = (
        MLManifest()
        .set_split_config(4, 2, 2)
        .with_target_label('next_bar_up', NextBarUpTarget)
    )

    data = manifest.prepare_data(raw_data, {'bar_type': 'base'})

    assert data['y_train'].to_list() == [1, 0, 1]
    assert data['y_val'].to_list() == [0]
    assert data['y_test'].to_list() == [1]


def _make_split_data() -> list[pl.DataFrame]:
    roc = [float(i) / 10 for i in range(20)]
    close = [100.0 + i for i in range(20)]
    df = pl.DataFrame({'datetime': pl.Series(range(20)).cast(pl.Int64).cast(pl.Datetime), 'roc_1': roc, 'close': close})
    return [df[:10], df[10:15], df[15:]]


def test_with_target_label_sets_target_class_config() -> None:
    m = Manifest().with_target_label('qflag', QuantileBinaryTarget,
                                     fit_params={'source_column': 'roc_1', 'quantile': 0.3})
    assert m.target_class_config is not None
    assert m.target_class_config.target_class is QuantileBinaryTarget
    assert m.target_class_config.fit_params['source_column'] == 'roc_1'
    assert m.target_class_config.fit_params['quantile'] == 0.3


def test_with_target_label_applies_transform_to_all_splits() -> None:
    splits = _make_split_data()
    m = Manifest()
    m.with_target_label('qflag', QuantileBinaryTarget,
                         fit_params={'source_column': 'roc_1', 'quantile': 0.3})

    from limen.experiment.manifest_core import _apply_class_based_target
    all_fitted: dict = {}
    for i, split in enumerate(splits):
        result, all_fitted = _apply_class_based_target(m, split, {}, all_fitted, is_training=(i == 0))
        assert 'qflag' in result.columns


def test_with_target_label_fits_only_on_train() -> None:
    splits = _make_split_data()
    m = Manifest()
    m.with_target_label('qflag', QuantileBinaryTarget,
                         fit_params={'source_column': 'roc_1', 'quantile': 0.3})

    from limen.experiment.manifest_core import _apply_class_based_target
    all_fitted: dict = {}
    splits[0], all_fitted = _apply_class_based_target(m, splits[0], {}, all_fitted, is_training=True)
    instance_after_train = all_fitted['_target_cls_qflag']
    cutoff = instance_after_train.cutoff

    splits[1], all_fitted = _apply_class_based_target(m, splits[1], {}, all_fitted, is_training=False)
    # Instance not re-fitted — cutoff must be unchanged
    assert all_fitted['_target_cls_qflag'].cutoff == cutoff


def test_random_binary_target_adds_column_with_binary_values() -> None:
    data = pl.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
    t = RandomBinaryTarget(data, 'noise')
    result = t.transform(data)
    assert 'noise' in result.columns
    assert set(result['noise'].to_list()).issubset({0, 1})
    assert result.height == data.height


def test_identity_target_returns_data_unchanged() -> None:
    data = pl.DataFrame({'feature': [1.0, 2.0, 3.0], 'label': [0, 1, 0]})
    t = IdentityTarget(data, 'label')
    result = t.transform(data)
    assert result.equals(data)


def test_identity_target_raises_when_column_missing_on_init() -> None:
    data = pl.DataFrame({'feature': [1.0, 2.0]})
    try:
        IdentityTarget(data, 'label')
        assert False, 'expected ValueError'
    except ValueError as e:
        assert 'label' in str(e)


def test_identity_target_raises_when_column_missing_on_transform() -> None:
    train = pl.DataFrame({'feature': [1.0, 2.0], 'label': [0, 1]})
    t = IdentityTarget(train, 'label')
    data_without_label = pl.DataFrame({'feature': [3.0, 4.0]})
    try:
        t.transform(data_without_label)
        assert False, 'expected ValueError'
    except ValueError as e:
        assert 'label' in str(e)
