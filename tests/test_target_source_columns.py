from pathlib import Path

import polars as pl
import pytest

from limen.data.utils import split_sequential
from limen.experiment import MLManifest, RuleBasedManifest
from limen.targets import ExitQualityTarget, RiskRewardRatioTarget


FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
SPLITS = ('train', 'val', 'test')


class _VolumeTarget:
    fit_count = 0

    def __init__(self, train_data, target_name):
        type(self).fit_count += 1
        self.target_name = target_name
        self.cutoff = train_data['volume'].quantile(0.5)
        self.transform_count = 0

    def transform(self, data):
        self.transform_count += 1
        return data.with_columns(
            (pl.col('volume').shift(-1) > self.cutoff).cast(pl.UInt8).alias(self.target_name),
        )


class _DeclaredVolumeTarget(_VolumeTarget):
    source_columns = ['volume']


class _EmptyVolumeTarget(_VolumeTarget):
    source_columns = []


class _LegacyVolumeTarget(_VolumeTarget):
    def transform(self, data):
        return super().transform(data).drop('volume')


class _AlreadyDroppedVolumeTarget(_LegacyVolumeTarget):
    source_columns = ['volume']


@pytest.fixture(scope='module')
def recorded_source_bars():
    return pl.read_parquet(FIXTURE, n_rows=192)


def _manifest(kind, target):
    manifest = MLManifest() if kind == 'ml' else RuleBasedManifest().with_strategy(
        [{'id': 'higher_close', 'type': 'relative', 'column': 'close',
          'operator': '>', 'other_column': 'open'}], entry='higher_close',
    )
    return (manifest.set_split_config(6, 2, 2)
            .with_target_label('next_volume', target)
            .set_backtest_config(take_profit_bps=50.0, stop_loss_bps=50.0))


def _assert_prepared_equal(actual, expected, kind):
    assert set(actual) == set(expected)
    assert actual['_alignment'] == expected['_alignment']
    assert actual['_backtest_provenance'] == expected['_backtest_provenance']
    for split in SPLITS:
        if kind == 'ml':
            assert actual[f'x_{split}'].equals(expected[f'x_{split}'])
            assert actual[f'y_{split}'].equals(expected[f'y_{split}'])
        else:
            assert actual[split].equals(expected[split])
    if kind == 'ml':
        assert actual['_feature_names'] == expected['_feature_names']
        assert actual['price_data_for_backtest'].equals(expected['price_data_for_backtest'])


@pytest.mark.parametrize('kind', ('ml', 'rule_based'))
def test_target_source_columns_match_legacy_drop_on_retained_recorded_rows(recorded_source_bars, kind):
    before = recorded_source_bars.clone()
    actual = _manifest(kind, _DeclaredVolumeTarget).prepare_data(recorded_source_bars, {})
    legacy = _manifest(kind, _LegacyVolumeTarget).prepare_data(recorded_source_bars, {})
    _assert_prepared_equal(actual, legacy, kind)
    raw_splits = split_sequential(recorded_source_bars, (6, 2, 2))
    cutoff = raw_splits[0]['volume'].quantile(0.5)
    assert actual['_alignment']['missing_datetimes'] == sorted(
        split['datetime'][-1] for split in raw_splits
    )
    for split_name, raw in zip(SPLITS, raw_splits, strict=True):
        expected_labels = (raw['volume'].shift(-1) > cutoff).cast(pl.UInt8).head(raw.height - 1)
        expected_features = raw.head(raw.height - 1).drop('datetime', 'volume')
        if kind == 'ml':
            assert actual[f'x_{split_name}'].equals(expected_features)
            assert actual[f'y_{split_name}'].to_list() == expected_labels.to_list()
        else:
            prepared = actual[split_name]
            assert prepared.select(expected_features.columns).equals(expected_features)
            assert prepared['next_volume'].to_list() == expected_labels.to_list()
            assert prepared.columns == [*expected_features.columns, 'next_volume', 'higher_close']
        assert 'volume' not in (actual[f'x_{split_name}'] if kind == 'ml' else actual[split_name]).columns
    witness = actual['_backtest_provenance']
    assert all(split.error is None for split in witness.splits)
    assert tuple(len(split.retained) for split in witness.splits) == tuple(
        raw.height - 1 for raw in raw_splits
    )
    assert recorded_source_bars.equals(before)


def test_target_source_columns_replay_reuses_fitted_instance(recorded_source_bars):
    _DeclaredVolumeTarget.fit_count = 0
    manifest = _manifest('ml', _DeclaredVolumeTarget)
    original = manifest.prepare_data(recorded_source_bars, {})
    fitted = original['_fitted_params']
    target = fitted['_target_cls_next_volume']
    assert _DeclaredVolumeTarget.fit_count == 1
    assert target.transform_count == 3
    replayed = manifest._prepare_data_with_fitted(recorded_source_bars, {}, fitted)
    _assert_prepared_equal(replayed, original, 'ml')
    assert replayed['_fitted_params']['_target_cls_next_volume'] is target
    assert _DeclaredVolumeTarget.fit_count == 1
    assert target.transform_count == 6


@pytest.mark.parametrize('kind', ('ml', 'rule_based'))
def test_absent_and_empty_source_columns_preserve_legacy_features(recorded_source_bars, kind):
    absent = _manifest(kind, _VolumeTarget).prepare_data(recorded_source_bars, {})
    empty = _manifest(kind, _EmptyVolumeTarget).prepare_data(recorded_source_bars, {})
    _assert_prepared_equal(empty, absent, kind)
    for split in SPLITS:
        frame = absent[f'x_{split}'] if kind == 'ml' else absent[split]
        assert 'volume' in frame.columns


@pytest.mark.parametrize('kind', ('ml', 'rule_based'))
def test_source_columns_tolerate_target_internal_drop(recorded_source_bars, kind):
    declared = _manifest(kind, _AlreadyDroppedVolumeTarget).prepare_data(recorded_source_bars, {})
    legacy = _manifest(kind, _LegacyVolumeTarget).prepare_data(recorded_source_bars, {})
    _assert_prepared_equal(declared, legacy, kind)


@pytest.mark.parametrize('declaration', ('volume', ('volume',), ['volume', 1], [None]))
@pytest.mark.parametrize('kind', ('ml', 'rule_based'))
def test_target_source_columns_reject_invalid_declarations(recorded_source_bars, monkeypatch, declaration, kind):
    monkeypatch.setattr(_DeclaredVolumeTarget, 'source_columns', declaration)
    with pytest.raises(TypeError, match='source_columns'):
        _manifest(kind, _DeclaredVolumeTarget).prepare_data(recorded_source_bars, {})


@pytest.mark.parametrize('kind', ('ml', 'rule_based'))
def test_target_source_columns_reject_target_collision(recorded_source_bars, monkeypatch, kind):
    monkeypatch.setattr(_DeclaredVolumeTarget, 'source_columns', ['next_volume'])
    with pytest.raises(ValueError, match='source_columns'):
        _manifest(kind, _DeclaredVolumeTarget).prepare_data(recorded_source_bars, {})


@pytest.mark.parametrize('target_class,source_columns', (
    (ExitQualityTarget, ['exit_reason', 'exit_net_return']),
    (RiskRewardRatioTarget, ['capturable_breakout', 'max_drawdown']),
))
def test_builtin_source_columns_preserve_direct_transform_and_preparation(
    recorded_source_bars, target_class, source_columns,
):
    # One-bar timeout outcomes and high/low excursions use recorded future prices.
    data = recorded_source_bars.with_columns(
        (pl.col('close').shift(-1) / pl.col('close') - 1).alias('exit_net_return'),
        pl.when(pl.col('close').shift(-1).is_not_null())
          .then(pl.lit('timeout')).otherwise(None).alias('exit_reason'),
        (pl.col('high').shift(-1) / pl.col('close') - 1).alias('capturable_breakout'),
        (pl.col('low').shift(-1) / pl.col('close') - 1).alias('max_drawdown'),
    ).drop_nulls().drop([name for name in (
        'exit_reason', 'exit_net_return', 'capturable_breakout', 'max_drawdown',
    ) if name not in source_columns])
    assert target_class.source_columns == source_columns
    direct = target_class(data, 'next_volume').transform(data)
    assert direct.columns == [name for name in data.columns if name not in source_columns] + ['next_volume']
    assert direct.height == data.height
    assert direct['datetime'].equals(data['datetime'])
    if target_class is ExitQualityTarget:
        expected = data.select(
            pl.when(pl.col('exit_net_return') < 0).then(0.2).otherwise(0.5).alias('next_volume'),
        )['next_volume']
    else:
        expected = data.select(
            (pl.col('capturable_breakout') / (pl.col('max_drawdown').abs() + 0.001)).alias('next_volume'),
        )['next_volume']
    assert direct['next_volume'].equals(expected)
    legacy_class = type('LegacyTarget', (target_class,), {'source_columns': []})
    prepared = _manifest('ml', target_class).prepare_data(data, {})
    legacy = _manifest('ml', legacy_class).prepare_data(data, {})
    _assert_prepared_equal(prepared, legacy, 'ml')
    for split in SPLITS:
        assert not set(source_columns) & set(prepared[f'x_{split}'].columns)
