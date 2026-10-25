from datetime import date, datetime, timedelta, timezone
from functools import wraps
import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML

from limen.cli.main import cli
from limen.data import HistoricalData
from limen.data.utils.splits import split_by_dates
from limen.data.utils.splits import split_data_to_prep_output
from limen.data.utils.splits import split_data_to_rule_based_prep_output
from limen.data.utils.splits import split_random
from limen.data.utils.splits import split_sequential
from limen.experiment import UniversalExperimentLoop
from limen.experiment.manifest_core import MLManifest, RuleBasedManifest
from limen.inference import Trainer
from limen.log.log import Log
from limen.sfd.reference_architecture.ridge_regressor import ridge_regressor
from limen.targets import IdentityTarget, RandomBinaryTarget
from limen.yaml import CompiledSFD, build_search_strategy, parse


@pytest.mark.parametrize('seed', [None, 42])
@pytest.mark.parametrize('ratios', [(6, 2, 2), (1, 1, 1), (3, 3, 4)])
def test_split_random_partitions_recorded_rows(seed: int | None, ratios: tuple[int, int, int]) -> None:
    data = pl.read_parquet(Path(__file__).parent / 'fixtures' / 'spot_15m_20250101_20250531.parquet')

    splits = split_random(data, ratios, seed=seed)

    first_end = int(data.height * ratios[0] / sum(ratios))
    second_end = int(data.height * sum(ratios[:2]) / sum(ratios))
    assert [split.height for split in splits] == [first_end, second_end - first_end, data.height - second_end]
    for index, split in enumerate(splits):
        for other in splits[index + 1:]:
            assert set(split['datetime']).isdisjoint(other['datetime'])
    assert pl.concat(splits).sort('datetime').equals(data.sort('datetime'))
    if seed is None:
        second = split_random(data, ratios)
        assert not splits[0]['datetime'].equals(second[0]['datetime'])


@pytest.mark.parametrize('seed', [0, 42])
@pytest.mark.parametrize('ratios', [(6, 2, 2), (3, 3, 4)])
def test_split_random_preserves_seeded_membership(seed: int, ratios: tuple[int, int, int]) -> None:
    data = pl.read_parquet(Path(__file__).parent / 'fixtures' / 'spot_15m_20250101_20250531.parquet')
    first_end = int(data.height * ratios[0] / sum(ratios))
    second_end = int(data.height * sum(ratios[:2]) / sum(ratios))
    bounds = ((0, first_end), (first_end, second_end), (second_end, data.height))
    previous = [
        data.sample(fraction=1.0, seed=seed, shuffle=True).slice(start, end - start)
        for start, end in bounds
    ]

    splits = split_random(data, ratios, seed=seed)

    assert all(actual.equals(expected) for actual, expected in zip(splits, previous, strict=True))


def _make_splits() -> tuple[list[pl.DataFrame], list]:
    df = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=pl.datetime(2025, 1, 1),
            end=pl.datetime(2025, 1, 8),
            interval='1d',
            eager=True,
        ),
        'feature': list(range(8)),
        'target': [1, 0] * 4,
    })
    return split_sequential(df, (6, 1, 1)), df['datetime'].to_list()


def test_prep_output_ml_path() -> None:
    splits, all_datetimes = _make_splits()
    cols = ['datetime', 'feature', 'target']

    result = split_data_to_prep_output(splits, cols, all_datetimes)

    assert set(result.keys()) >= {'x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test', '_alignment'}
    assert 'feature' in result['x_train'].columns
    assert 'target' not in result['x_train'].columns
    assert result['y_test'].name == 'target'
    assert 'missing_datetimes' in result['_alignment']
    assert result['_alignment']['missing_datetimes'] == []


def test_rule_based_prep_output_returns_full_dataframes() -> None:
    splits, all_datetimes = _make_splits()

    result = split_data_to_rule_based_prep_output(splits, all_datetimes)

    assert set(result.keys()) == {'train', 'val', 'test', '_alignment'}
    for split in ('train', 'val', 'test'):
        assert isinstance(result[split], pl.DataFrame)
        assert 'datetime' not in result[split].columns
        assert 'feature' in result[split].columns
        assert 'target' in result[split].columns
    assert 'missing_datetimes' in result['_alignment']
    assert result['_alignment']['missing_datetimes'] == []


def _make_daily_df(start: datetime, end: datetime) -> pl.DataFrame:
    return pl.DataFrame({
        'datetime': pl.datetime_range(start=start, end=end, interval='1d', eager=True),
    }).with_columns(pl.int_range(0, pl.len()).alias('feature'))


def test_split_by_dates_returns_three_dataframes_in_order() -> None:
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))

    splits = split_by_dates(
        df,
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    assert len(splits) == 3
    train, val, test = splits
    assert train['datetime'].min() == datetime(2024, 1, 1)
    assert train['datetime'].max() == datetime(2024, 6, 30)
    assert val['datetime'].min() == datetime(2024, 7, 1)
    assert val['datetime'].max() == datetime(2024, 9, 30)
    assert test['datetime'].min() == datetime(2024, 10, 1)
    assert test['datetime'].max() == datetime(2024, 12, 31)


def test_split_by_dates_no_row_loss_when_windows_are_contiguous() -> None:
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))
    total = df.height

    splits = split_by_dates(
        df,
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    assert sum(s.height for s in splits) == total


def test_split_by_dates_no_row_overlap_at_window_boundary() -> None:
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))

    splits = split_by_dates(
        df,
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    train, val, test = splits
    train_dts = set(train['datetime'].to_list())
    val_dts = set(val['datetime'].to_list())
    test_dts = set(test['datetime'].to_list())
    assert not (train_dts & val_dts)
    assert not (val_dts & test_dts)
    assert not (train_dts & test_dts)


def test_split_by_dates_drops_rows_in_gaps_between_windows() -> None:
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))

    splits = split_by_dates(
        df,
        datetime(2024, 1, 1), datetime(2024, 6, 1),
        datetime(2024, 7, 1), datetime(2024, 9, 1),
        datetime(2024, 10, 1), datetime(2024, 12, 1),
    )

    expected_train = 31 + 29 + 31 + 30 + 31
    expected_val = 31 + 31
    expected_test = 31 + 30
    assert splits[0].height == expected_train
    assert splits[1].height == expected_val
    assert splits[2].height == expected_test


def test_set_split_dates_stores_bounds_in_order() -> None:
    m = MLManifest().set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    assert m.split_dates == (
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )


def test_set_split_dates_rejects_out_of_order_bounds() -> None:
    with pytest.raises(ValueError, match='train_end'):
        MLManifest().set_split_dates(
            datetime(2024, 7, 1), datetime(2024, 1, 1),
            datetime(2024, 7, 1), datetime(2024, 10, 1),
            datetime(2024, 10, 1), datetime(2025, 1, 1),
        )


def test_set_split_dates_rejects_val_before_train_end() -> None:
    with pytest.raises(ValueError, match='val_start'):
        MLManifest().set_split_dates(
            datetime(2024, 1, 1), datetime(2024, 8, 1),
            datetime(2024, 7, 1), datetime(2024, 10, 1),
            datetime(2024, 10, 1), datetime(2025, 1, 1),
        )


def test_set_split_dates_allows_gaps_between_adjacent_windows() -> None:
    MLManifest().set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 6, 1),
        datetime(2024, 7, 1), datetime(2024, 9, 1),
        datetime(2024, 10, 1), datetime(2024, 12, 1),
    )


def test_set_split_dates_does_not_mutate_split_config() -> None:
    m = MLManifest()
    original_split_config = m.split_config
    m.set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )
    assert m.split_config == original_split_config
    assert m.split_dates is not None


def test_manifest_default_split_dates_is_none() -> None:
    assert MLManifest().split_dates is None


def test_compute_test_bars_returns_test_window_on_date_path() -> None:
    '''
    compute_test_bars must return exactly the test-window rows when
    split_dates is set, and must do so without materialising the train
    and val DataFrames (perf-regression guard from the date path).
    '''
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))

    m = MLManifest().set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    # compute_test_bars needs bar_formation to do _process_bars; without
    # it the helper just returns the test slice unchanged. We assert on
    # the slice contents, not the bar shape.
    test_bars = m.compute_test_bars(df, {})

    assert test_bars['datetime'].min() == datetime(2024, 10, 1)
    assert test_bars['datetime'].max() == datetime(2024, 12, 31)
    assert test_bars.height == 31 + 30 + 31


def test_compute_test_bars_still_works_on_ratio_path() -> None:
    '''
    Sanity guard: the ratio path through compute_test_bars must keep
    working unchanged after the date-path optimisation.
    '''
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))
    total = df.height

    m = MLManifest().set_split_config(8, 1, 2)
    test_bars = m.compute_test_bars(df, {})

    expected_train = int(total * 8 / 11)
    expected_val = int(total * 1 / 11)
    expected_test = total - expected_train - expected_val
    assert test_bars.height == expected_test


def _make_prepare_data_manifest() -> MLManifest:
    return (MLManifest()
        .set_data_source(method=lambda: None, params={})
        .with_target_label('outcome', RandomBinaryTarget)
    )


def test_prepare_data_honours_split_dates_at_real_call_site() -> None:
    '''
    Drive `MLManifest.prepare_data` with `split_dates` set and assert the
    per-slice datetime bounds match the configured windows exactly. This
    pins the date-path through `_run_prepare_setup` -> `_resolve_split`
    -> `_finalize_to_data_dict`, not just `split_by_dates` in isolation.
    '''
    np.random.seed(0)
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))

    manifest = _make_prepare_data_manifest().set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    data = manifest.prepare_data(df, {'bar_type': 'base'})

    expected_train = 31 + 29 + 31 + 30 + 31 + 30
    expected_val = 31 + 31 + 30
    expected_test = 31 + 30 + 31

    assert data['x_train'].height == expected_train
    assert data['x_val'].height == expected_val
    assert data['x_test'].height == expected_test
    assert data['y_train'].name == 'outcome'
    assert data['_alignment']['first_test_datetime'] == datetime(2024, 10, 1)
    assert data['_alignment']['last_test_datetime'] == datetime(2024, 12, 31)
    assert data['_alignment']['missing_datetimes'] == []


def test_prepare_data_ratio_path_unchanged_when_split_dates_unset() -> None:
    '''
    Sanity guard: the ratio path through `prepare_data` keeps working
    when `split_dates` is not configured (the default).
    '''
    np.random.seed(0)
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))
    total = df.height

    manifest = _make_prepare_data_manifest().set_split_config(8, 1, 2)
    data = manifest.prepare_data(df, {'bar_type': 'base'})

    expected_train = int(total * 8 / 11)
    expected_val = int(total * 1 / 11)
    expected_test = total - expected_train - expected_val
    assert data['x_train'].height == expected_train
    assert data['x_val'].height == expected_val
    assert data['x_test'].height == expected_test


def test_with_params_override_split_config_clears_split_dates() -> None:
    '''
    `with_params_override(split_config=...)` must clear any
    previously-pinned `split_dates` so the ratio override actually
    takes effect downstream. Without this, `_resolve_split` would
    keep using `split_dates` and a caller-owned ratio override would silently no-op.
    '''
    base = MLManifest().set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    overridden = base.with_params_override(split_config=(1, 0, 0))

    assert overridden.split_config == (1, 0, 0)
    assert overridden.split_dates is None
    # Original manifest is untouched (deep copy)
    assert base.split_dates is not None
    assert base.split_config == (8, 1, 2)


def test_with_params_override_non_split_config_keeps_split_dates() -> None:
    '''
    Overriding a data-source param (anything other than `split_config`)
    must NOT clear `split_dates` - the date pin is still in force.
    '''
    base = (MLManifest()
        .set_data_source(
            method=lambda kline_size=3600: None,
            params={'kline_size': 3600},
        )
        .set_split_dates(
            datetime(2024, 1, 1), datetime(2024, 7, 1),
            datetime(2024, 7, 1), datetime(2024, 10, 1),
            datetime(2024, 10, 1), datetime(2025, 1, 1),
        )
    )

    overridden = base.with_params_override(kline_size=7200)

    assert overridden.split_dates == base.split_dates
    assert overridden.data_source_config.params['kline_size'] == 7200


def test_set_split_dates_rejects_non_date_bounds() -> None:
    '''
    `set_split_dates` accepts only `date`/`datetime` instances. Passing
    a comparable-but-wrong type (string, int, float, None) must raise
    `TypeError` at the API boundary, not fail later inside Polars.
    '''
    bad_values = ['2024-01-01', 1704067200, 1704067200.0, None]
    for bad_value in bad_values:
        with pytest.raises(TypeError, match='train_start'):
            MLManifest().set_split_dates(
                bad_value, datetime(2024, 7, 1),
                datetime(2024, 7, 1), datetime(2024, 10, 1),
                datetime(2024, 10, 1), datetime(2025, 1, 1),
            )


def test_split_by_dates_rejects_non_date_bounds() -> None:
    '''
    `split_by_dates` is exported as a public helper so it has to
    validate its own bounds at the boundary too - mirroring what
    `set_split_dates` does for callers who go through the manifest.
    '''
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))
    bad_values = ['2024-01-01', 1704067200, None]
    for bad_value in bad_values:
        with pytest.raises(TypeError, match='train_start'):
            split_by_dates(
                df,
                bad_value, datetime(2024, 7, 1),
                datetime(2024, 7, 1), datetime(2024, 10, 1),
                datetime(2024, 10, 1), datetime(2025, 1, 1),
            )


_NATIVE_DATE_BOUNDS = (
    date(2025, 1, 1), date(2025, 1, 16),
    date(2025, 1, 17), date(2025, 1, 23),
    date(2025, 1, 24), date(2025, 2, 6),
)
_DATE_FIELDS = ('train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end')


@pytest.fixture(scope='module')
def native_date_frames():
    fixture = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(HistoricalData, 'DEFAULT_SPOT_KLINES_DATASET_REPO', str(fixture))
        yield {
            interval: HistoricalData().get_spot_klines(
                kline_size=interval, start_date_limit='2025-01-01', end_date_limit='2025-02-06',
            )
            for interval in (900, 3600, 25200)
        }


def _native_date_manifest(interval=25200, *, bounds=_NATIVE_DATE_BOUNDS):
    return (MLManifest()
        .set_data_source(HistoricalData.get_spot_klines, {'kline_size': interval})
        .with_target_label('close', IdentityTarget)
        .set_split_dates(*bounds)
    )


@pytest.mark.parametrize(('index', 'end'), [
    (1, date(2025, 1, 15)), (3, date(2025, 1, 22)), (5, date(2025, 2, 5)),
])
def test_native_date_ends_reject_crossing_bars_before_preparation(native_date_frames, index, end):
    bounds = list(_NATIVE_DATE_BOUNDS)
    bounds[index] = end
    manifest = _native_date_manifest(bounds=bounds)
    frame = native_date_frames[25200]
    match = rf'data_source {_DATE_FIELDS[index]}=.*UTC Unix-epoch grid.*kline_size=25200 seconds'

    with pytest.raises(ValueError, match=match):
        manifest.prepare_data(frame, {})
    # Reconstruction must reject the declaration even without a crossing row in its input.
    with pytest.raises(ValueError, match=match):
        manifest.compute_test_bars(frame.head(1), {})

    if index == 1:
        label = datetime(2025, 1, 14, 20, tzinfo=timezone.utc)
        constituents = native_date_frames[900].filter(
            (pl.col('datetime') >= label) & (pl.col('datetime') < label + timedelta(hours=7)),
        )
        assert constituents.height == 28
        assert constituents.filter(pl.col('datetime') >= end).height == 12
        assert frame.filter(pl.col('datetime') == label)['close'][0] == constituents['close'][-1] == 96824.06


@pytest.mark.parametrize('interval', [3600, 25200])
def test_native_date_aligned_ends_preserve_recorded_bars_and_gaps(native_date_frames, interval):
    frame = native_date_frames[interval]
    manifest = _native_date_manifest(interval)
    prepared = manifest.prepare_data(frame, {})
    expected = split_by_dates(frame, *_NATIVE_DATE_BOUNDS)

    for name, split, end in zip(('train', 'val', 'test'), expected, _NATIVE_DATE_BOUNDS[1::2], strict=True):
        assert prepared['y_' + name].to_list() == split['close'].to_list()
        assert split['datetime'][-1] + timedelta(seconds=interval) == datetime.combine(end, datetime.min.time(), timezone.utc)
    assert manifest.compute_test_bars(frame, {}).equals(expected[2])
    assert sum(split.height for split in expected) < frame.height
    assert frame.filter(pl.col('datetime') < _NATIVE_DATE_BOUNDS[0]).height == (interval == 25200)


@pytest.mark.parametrize('kind', ['unbound', 'bound', 'inherited', 'wrapper', 'override'])
def test_native_date_detection_preserves_callable_source_contract(native_date_frames, kind):
    class Inherited(HistoricalData):
        """Use the unchanged native implementation with the recorded repository override."""

    class Overridden(HistoricalData):
        def get_spot_klines(self, kline_size=60):
            return super().get_spot_klines(kline_size=kline_size)

    @wraps(HistoricalData.get_spot_klines)
    def wrapper(kline_size=60):
        return HistoricalData().get_spot_klines(kline_size=kline_size)

    method = {
        'unbound': HistoricalData.get_spot_klines,
        'bound': HistoricalData().get_spot_klines,
        'inherited': Inherited().get_spot_klines,
        'wrapper': wrapper,
        'override': Overridden().get_spot_klines,
    }[kind]
    bounds = list(_NATIVE_DATE_BOUNDS)
    bounds[1] = date(2025, 1, 15)
    manifest = _native_date_manifest(bounds=bounds).set_data_source(method, {'kline_size': 25200})
    frame = manifest.fetch_data()
    if kind in ('unbound', 'bound', 'inherited'):
        with pytest.raises(ValueError, match=r'data_source train_end=.*kline_size=25200 seconds'):
            manifest.prepare_data(frame, {})
    else:
        expected = split_by_dates(frame, *bounds)
        assert manifest.prepare_data(frame, {})['y_train'].to_list() == expected[0]['close'].to_list()


def test_native_date_sources_use_separate_effective_intervals(native_date_frames):
    bounds = list(_NATIVE_DATE_BOUNDS)
    bounds[1] = date(2025, 1, 15)
    manifest = _native_date_manifest(3600, bounds=bounds)
    manifest.set_test_data_source(HistoricalData().get_spot_klines, {'kline_size': 25200})
    with pytest.raises(ValueError, match=r'test_data_source train_end=.*kline_size=25200 seconds'):
        manifest.prepare_data(native_date_frames[3600], {})

    manifest.set_test_data_source(HistoricalData.get_spot_klines, {'kline_size': 3600})
    changed = manifest.with_params_override(kline_size=25200)
    assert changed.test_data_source_config.params['kline_size'] == 3600
    with pytest.raises(ValueError, match=r'data_source train_end=.*kline_size=25200 seconds'):
        changed.prepare_data(native_date_frames[3600], {})
    restored = changed.with_params_override(kline_size=3600)
    assert restored.compute_test_bars(native_date_frames[3600], {}).height > 0


@pytest.mark.parametrize('params', [{}, {'kline_size': None}])
def test_native_date_default_interval_uses_supplied_recorded_frame(native_date_frames, params):
    bounds = [datetime.combine(value, datetime.min.time(), timezone.utc) for value in _NATIVE_DATE_BOUNDS]
    bounds[1] += timedelta(minutes=1)
    manifest = _native_date_manifest(bounds=bounds).set_data_source(HistoricalData.get_spot_klines, params)
    # This is declaration admission of caller-supplied observations, not minute-bar acquisition.
    assert manifest.compute_test_bars(native_date_frames[900], {}).height > 0
    bounds[1] += timedelta(seconds=30)
    manifest.set_split_dates(*bounds)
    with pytest.raises(ValueError, match=r'data_source train_end=.*kline_size=60 seconds'):
        manifest.prepare_data(native_date_frames[900], {})


@pytest.mark.parametrize('offset', [timedelta(minutes=5), timedelta(microseconds=1)])
def test_native_date_nonaggregated_aware_cutoff_uses_utc_instant(native_date_frames, offset):
    zone = timezone(timedelta(hours=5, minutes=30))
    bounds = [datetime.combine(value, datetime.min.time(), timezone.utc).astimezone(zone) for value in _NATIVE_DATE_BOUNDS]
    manifest = _native_date_manifest(900, bounds=bounds)
    assert manifest.compute_test_bars(native_date_frames[900], {}).equals(
        split_by_dates(native_date_frames[900], *_NATIVE_DATE_BOUNDS)[2],
    )
    bounds[1] += offset
    manifest.set_split_dates(*bounds)
    with pytest.raises(ValueError, match=r'data_source train_end=.*kline_size=900 seconds'):
        manifest.prepare_data(native_date_frames[900], {})


def test_native_date_naive_bounds_and_rule_preparation_refuse(native_date_frames):
    naive = [datetime.combine(value, datetime.min.time()) for value in _NATIVE_DATE_BOUNDS]
    with pytest.raises(TypeError, match=r'data_source train_end=.*timezone-aware'):
        _native_date_manifest(3600, bounds=naive).prepare_data(native_date_frames[3600], {})

    bounds = list(_NATIVE_DATE_BOUNDS)
    bounds[1] = date(2025, 1, 15)
    rule = (RuleBasedManifest()
        .set_data_source(HistoricalData.get_spot_klines, {'kline_size': 25200})
        .set_split_dates(*bounds)
        .with_strategy(conditions=[{'id': 'up', 'type': 'relative', 'column': 'close', 'operator': '>', 'other_column': 'open'}], entry='up')
    )
    with pytest.raises(ValueError, match=r'data_source train_end=.*kline_size=25200 seconds'):
        rule.prepare_data(native_date_frames[25200], {})


def _native_date_yaml():
    template = Path(__file__).resolve().parents[1] / 'limen/yaml/templates/ridge_regressor.yaml'
    config, errors = parse(template.read_text())
    assert not errors
    manifest = config['sfd']['manifest']
    manifest['data_source']['params']['kline_size'] = 25200
    manifest['split_dates'] = dict(zip(_DATE_FIELDS, (value.isoformat() for value in _NATIVE_DATE_BOUNDS), strict=True))
    config['sfd']['params'] = {key: [values[0]] for key, values in config['sfd']['params'].items()}
    config['sfd']['params']['solver'] = ['svd']
    config['uel'].update(n_permutations=1, search_strategy={'type': 'grid'}, checkpoint_interval=1,
                         recording={'predictions': True, 'alignment': True})
    return config


@pytest.mark.parametrize(('test_mode', 'learning_interval', 'test_interval', 'source'), [
    (False, 3600, 25200, 'test_data_source'), (True, 25200, 3600, 'data_source'),
])
def test_native_date_unused_source_refusal_is_fatal_in_python_uel(
    native_date_frames, tmp_path, test_mode, learning_interval, test_interval, source,
):
    bounds = list(_NATIVE_DATE_BOUNDS)
    bounds[1] = date(2025, 1, 15)
    manifest = (_native_date_manifest(learning_interval, bounds=bounds)
        .set_test_data_source(HistoricalData.get_spot_klines, {'kline_size': test_interval})
        .with_reference_architecture(ridge_regressor)
    )
    sfd = SimpleNamespace(params=lambda: {'alpha': [1.0]}, manifest=lambda: manifest)
    loop = UniversalExperimentLoop(sfd=sfd, test_mode=test_mode, experiment_dir=tmp_path)
    assert loop.data['datetime'][1] - loop.data['datetime'][0] == timedelta(hours=1)
    with pytest.raises(ValueError, match=rf'Manifest {source} train_end=.*kline_size=25200 seconds'):
        loop.run('native-date-ends', n_permutations=1, prep_each_round=True, progress_bar=False)


def test_native_date_refusal_is_fatal_in_cli(native_date_frames, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _native_date_yaml()
    config['sfd']['manifest']['data_source']['params']['kline_size'] = 3600
    config['sfd']['manifest']['params_override'] = {'kline_size': 25200}
    config['sfd']['manifest']['split_dates']['train_end'] = '2025-01-15'
    config['uel']['output_path'] = 'native-results'
    yaml_path = tmp_path / 'native.yaml'
    with yaml_path.open('w') as handle:
        YAML().dump(config, handle)
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(yaml_path)])
    assert result.exit_code == 1
    assert 'Experiment failed: Manifest data_source train_end=' in result.output
    assert 'UTC Unix-epoch grid for get_spot_klines kline_size=25200 seconds' in result.output


def test_native_date_refusal_is_fatal_in_spawned_worker(native_date_frames, tmp_path):
    config = _native_date_yaml()
    config['sfd']['manifest']['split_dates']['train_end'] = '2025-01-15'
    loop = UniversalExperimentLoop(
        sfd=CompiledSFD(config), search_strategy=build_search_strategy(config),
        experiment_dir=tmp_path, yaml_reference=config,
    )
    with pytest.raises(RuntimeError, match='Parallel worker failed for round 0') as failure:
        loop.run('native-date-worker', n_permutations=1, prep_each_round=True, progress_bar=False, workers=2)
    assert 'Manifest data_source train_end=' in str(failure.value)
    assert 'UTC Unix-epoch grid for get_spot_klines kline_size=25200 seconds' in str(failure.value)


def test_native_date_recorded_replay_refuses_incompatible_retained_declaration(native_date_frames, tmp_path):
    config = _native_date_yaml()
    loop = UniversalExperimentLoop(
        sfd=CompiledSFD(config), search_strategy=build_search_strategy(config),
        experiment_dir=tmp_path, yaml_reference=config,
    )
    loop.run('native-date-replay', n_permutations=1, prep_each_round=True, progress_bar=False, post_processing=True)
    ids = loop.experiment_log['id'].to_list()
    assert len(ids) == 1
    assert len(Trainer(tmp_path).train(ids)) == 1
    log = Log(uel_object=loop)
    assert len(log.permutation_prediction_performance(0)) == len(loop.preds[0]) > 0

    # The run above is genuine; make its retained declaration incompatible as a replay fault control.
    bounds = list(_NATIVE_DATE_BOUNDS)
    bounds[1] = date(2025, 1, 15)
    loop.manifest.set_split_dates(*bounds)
    with pytest.raises(ValueError, match=r'data_source train_end=.*kline_size=25200 seconds'):
        log.permutation_prediction_performance(0)
    with pytest.raises(ValueError, match=r'data_source train_end=.*kline_size=25200 seconds'):
        log._get_test_data_with_all_cols(0)

    metadata_path = tmp_path / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['yaml_reference']['sfd']['manifest']['split_dates']['train_end'] = '2025-01-15'
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r'data_source train_end=.*kline_size=25200 seconds'):
        Trainer(tmp_path).train(ids)
