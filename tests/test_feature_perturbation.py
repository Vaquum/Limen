import polars as pl

from limen.experiment import Manifest
from limen.experiment.manifest_core import TransformEntry
from limen.targets import RandomBinaryTarget
from tests.utils.historical_data import get_cached_spot_klines_2h


def _make_manifest_with_groups() -> Manifest:

    return (Manifest()
        .set_test_data_source(
            method=get_cached_spot_klines_2h,
            params={'n_rows': 500}
        )
        .set_split_config(3, 1, 1)
        .add_indicator(
            lambda df: df.with_columns((pl.col('close').pct_change()).alias('roc')),
            group='momentum',
        )
        .add_indicator(
            lambda df: df.with_columns(pl.col('close').rolling_std(5).alias('vol_5')),
            group='volatility',
        )
        .add_indicator(
            lambda df: df.with_columns(pl.col('close').rolling_mean(10).alias('sma_10')),
            group='trend',
        )
        .with_target_label('outcome', RandomBinaryTarget)
    )


def _make_manifest_with_include_if() -> Manifest:

    return (Manifest()
        .set_test_data_source(
            method=get_cached_spot_klines_2h,
            params={'n_rows': 500}
        )
        .set_split_config(3, 1, 1)
        .add_indicator(
            lambda df: df.with_columns((pl.col('close').pct_change()).alias('roc')),
            include_if='include_roc',
        )
        .with_target_label('outcome', RandomBinaryTarget)
    )


def _prepare(manifest, round_params=None):

    raw_data = manifest.fetch_test_data()
    if round_params is None:
        round_params = {}
    return manifest.prepare_data(raw_data, round_params)


def test_add_indicator_creates_transform_entry():

    manifest = Manifest().add_indicator(lambda x: x, group='momentum', period=14)
    assert len(manifest.feature_transforms) == 1
    entry = manifest.feature_transforms[0]
    assert isinstance(entry, TransformEntry)
    assert entry.group == 'momentum'
    assert entry.params == {'period': 14}


def test_feature_group_filtering_includes_selected():

    manifest = _make_manifest_with_groups()
    data = _prepare(manifest, {'feature_groups': 'momentum'})
    columns = list(data['_feature_names'])
    assert 'roc' in columns
    assert 'vol_5' not in columns
    assert 'sma_10' not in columns


def test_feature_group_filtering_multiple_groups():

    manifest = _make_manifest_with_groups()
    data = _prepare(manifest, {'feature_groups': 'momentum|trend'})
    columns = list(data['_feature_names'])
    assert 'roc' in columns
    assert 'sma_10' in columns
    assert 'vol_5' not in columns


def test_feature_group_absent_includes_all():

    manifest = _make_manifest_with_groups()
    data = _prepare(manifest, {})
    columns = list(data['_feature_names'])
    assert 'roc' in columns
    assert 'vol_5' in columns
    assert 'sma_10' in columns


def test_ungrouped_features_always_included():

    manifest = (Manifest()
        .set_test_data_source(
            method=get_cached_spot_klines_2h,
            params={'n_rows': 500}
        )
        .set_split_config(3, 1, 1)
        .add_indicator(
            lambda df: df.with_columns((pl.col('close').pct_change()).alias('roc')),
            group='momentum',
        )
        .add_indicator(
            lambda df: df.with_columns(pl.col('close').rolling_std(5).alias('vol_5')),
        )
        .with_target_label('outcome', RandomBinaryTarget)
    )
    data = _prepare(manifest, {'feature_groups': 'momentum'})
    columns = list(data['_feature_names'])
    assert 'roc' in columns
    assert 'vol_5' in columns


def test_combined_group_and_include_if():

    manifest = (Manifest()
        .set_test_data_source(
            method=get_cached_spot_klines_2h,
            params={'n_rows': 500}
        )
        .set_split_config(3, 1, 1)
        .add_indicator(
            lambda df: df.with_columns((pl.col('close').pct_change()).alias('roc')),
            group='momentum',
            include_if='include_roc',
        )
        .add_indicator(
            lambda df: df.with_columns(pl.col('close').rolling_std(5).alias('vol_5')),
            group='volatility',
        )
        .with_target_label('outcome', RandomBinaryTarget)
    )
    data = _prepare(manifest, {'feature_groups': 'momentum|volatility', 'include_roc': False})
    columns = list(data['_feature_names'])
    assert 'roc' not in columns
    assert 'vol_5' in columns


def test_include_if_true():

    data = _prepare(_make_manifest_with_include_if(), {'include_roc': True})
    assert 'roc' in data['_feature_names']


def test_include_if_false():

    data = _prepare(_make_manifest_with_include_if(), {'include_roc': False})
    assert 'roc' not in data['_feature_names']


def test_include_if_key_missing_includes():

    data = _prepare(_make_manifest_with_include_if(), {})
    assert 'roc' in data['_feature_names']


def test_ablation_drops_correct_count():

    manifest = _make_manifest_with_groups().set_feature_ablation()
    rp = {'feature_drop_count': 1, 'feature_drop_seed': 42}
    _prepare(manifest, rp)
    assert len(rp['_dropped_features']) == 1


def test_ablation_deterministic_with_seed():

    manifest = _make_manifest_with_groups().set_feature_ablation()

    rp1 = {'feature_drop_count': 1, 'feature_drop_seed': 42}
    _prepare(manifest, rp1)
    rp2 = {'feature_drop_count': 1, 'feature_drop_seed': 42}
    _prepare(manifest, rp2)
    assert rp1['_dropped_features'] == rp2['_dropped_features']


def test_ablation_zero_drops_nothing():

    manifest = _make_manifest_with_groups().set_feature_ablation()
    rp = {'feature_drop_count': 0, 'feature_drop_seed': 42}
    _prepare(manifest, rp)
    assert '_dropped_features' not in rp


def test_ablation_preserves_datetime_and_target():

    manifest = _make_manifest_with_groups().set_feature_ablation()
    rp = {'feature_drop_count': 2, 'feature_drop_seed': 42}
    _prepare(manifest, rp)
    assert 'datetime' not in rp['_dropped_features']
    assert 'outcome' not in rp['_dropped_features']


def test_ablation_consistent_across_splits():

    manifest = _make_manifest_with_groups().set_feature_ablation()
    rp = {'feature_drop_count': 1, 'feature_drop_seed': 42}
    data = _prepare(manifest, rp)
    dropped = rp['_dropped_features']
    for key in ('x_train', 'x_val', 'x_test'):
        if hasattr(data[key], 'columns'):
            for col in dropped:
                assert col not in data[key].columns


def test_ablation_not_configured_noop():

    manifest = _make_manifest_with_groups()
    rp = {'feature_drop_count': 2, 'feature_drop_seed': 42}
    _prepare(manifest, rp)
    assert '_dropped_features' not in rp


def test_feature_groups_all_includes_everything():

    manifest = _make_manifest_with_groups()
    data = _prepare(manifest, {'feature_groups': 'all'})
    columns = list(data['_feature_names'])
    assert 'roc' in columns
    assert 'vol_5' in columns
    assert 'sma_10' in columns


def test_ablation_invalid_drop_count_raises():

    manifest = _make_manifest_with_groups().set_feature_ablation()
    rp = {'feature_drop_count': -1, 'feature_drop_seed': 42}
    try:
        _prepare(manifest, rp)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'non-negative int' in str(e)


def test_ablation_drop_count_exceeds_eligible_raises():

    manifest = _make_manifest_with_groups().set_feature_ablation()
    rp = {'feature_drop_count': 9999, 'feature_drop_seed': 42}
    try:
        _prepare(manifest, rp)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'feature_drop_count' in str(e)
        assert 'exceeds' in str(e)
