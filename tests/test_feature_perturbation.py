import numpy as np
import polars as pl

from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.experiment.manifest_core import TransformEntry


def _make_manifest_with_groups() -> Manifest:

    return (Manifest()
        .set_test_data_source(method=HistoricalData._get_data_for_test)
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
        .with_target('outcome')
            .add_transform(lambda data: data.with_columns(
                pl.Series('outcome', np.random.randint(0, 2, size=data.height))
            ))
            .add_transform(lambda data: data[:-1])
            .done()
    )


def _prepare(manifest, round_params=None):

    raw_data = manifest.fetch_test_data()
    if round_params is None:
        round_params = {}
    return manifest.prepare_data(raw_data, round_params)


# --- TransformEntry ---

def test_transform_entry_defaults():

    entry = TransformEntry(func=lambda x: x)
    assert entry.group is None
    assert entry.include_if is None
    assert entry.params == {}


def test_add_indicator_creates_transform_entry():

    manifest = Manifest().add_indicator(lambda x: x, group='momentum', period=14)
    assert len(manifest.feature_transforms) == 1
    entry = manifest.feature_transforms[0]
    assert isinstance(entry, TransformEntry)
    assert entry.group == 'momentum'
    assert entry.params == {'period': 14}


# --- Group filtering ---

def test_feature_group_filtering_includes_selected():

    manifest = _make_manifest_with_groups()
    data = _prepare(manifest, {'feature_groups': ['momentum']})
    columns = list(data['_feature_names'])
    assert 'roc' in columns
    assert 'vol_5' not in columns
    assert 'sma_10' not in columns


def test_feature_group_filtering_multiple_groups():

    manifest = _make_manifest_with_groups()
    data = _prepare(manifest, {'feature_groups': ['momentum', 'trend']})
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


# --- Conditional inclusion ---

def _make_manifest_with_include_if() -> Manifest:

    return (Manifest()
        .set_test_data_source(method=HistoricalData._get_data_for_test)
        .set_split_config(3, 1, 1)
        .add_indicator(
            lambda df: df.with_columns((pl.col('close').pct_change()).alias('roc')),
            include_if='include_roc',
        )
        .with_target('outcome')
            .add_transform(lambda data: data.with_columns(
                pl.Series('outcome', np.random.randint(0, 2, size=data.height))
            ))
            .add_transform(lambda data: data[:-1])
            .done()
    )


def test_include_if_true():

    data = _prepare(_make_manifest_with_include_if(), {'include_roc': True})
    assert 'roc' in data['_feature_names']


def test_include_if_false():

    data = _prepare(_make_manifest_with_include_if(), {'include_roc': False})
    assert 'roc' not in data['_feature_names']


def test_include_if_key_missing_includes():

    # Key not in round_params → default to include
    data = _prepare(_make_manifest_with_include_if(), {})
    assert 'roc' in data['_feature_names']


# --- Feature ablation ---

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
