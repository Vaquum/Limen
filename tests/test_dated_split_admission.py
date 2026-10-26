from datetime import date, datetime, timezone
import importlib
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML
from sklearn.decomposition import PCA

from limen.cli.main import cli
from limen.data.bars import volume_bars
from limen.experiment import MLManifest, RuleBasedManifest, UniversalExperimentLoop
from limen.experiment.errors import StrictModeError
from limen.experiment._manifest_preparation import admit_dated_split
from limen.indicators import roc
from limen.log.log import Log
from limen.scalers import CausalRollingRobustScaler, RobustScaler
from limen.targets import IdentityTarget, NextReturnTarget
from limen.yaml import CompiledSFD, build_search_strategy, parse

FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
DATES = tuple(date(2024, 1, day) for day in (1, 5, 5, 6, 6, 7))


@pytest.fixture(scope='module')
def bars():
    return pl.read_parquet(FIXTURE, n_rows=336)


def _manifest(dates=DATES, *, rule=False):
    manifest = RuleBasedManifest().with_strategy([
        {'id': 'positive', 'type': 'threshold', 'column': 'close', 'operator': '>', 'value': 0}], entry='positive') if rule else MLManifest().with_target_label('close', IdentityTarget)
    return manifest.set_split_dates(*dates)


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('index,name', [(0, 'train'), (1, 'validation'), (2, 'test')])
def test_each_missing_raw_window_refuses_before_any_fit(bars, monkeypatch, strict, index, name):
    manifest = _manifest().set_strict_mode(strict).set_scaler(RobustScaler).set_pca_compression()
    start, end = DATES[index * 2:index * 2 + 2]
    admitted = bars.filter(~pl.col('datetime').is_between(start, end, closed='left'))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Missing dated window reached fitting')

    monkeypatch.setattr(IdentityTarget, '__init__', forbidden)
    monkeypatch.setattr(RobustScaler, '__init__', forbidden)
    monkeypatch.setattr(PCA, 'fit_transform', forbidden)
    with pytest.raises(ValueError, match=name + ' split') as error:
        manifest.prepare_data(admitted, {'auto_pca': True, 'pca_k': 1})
    assert f'[{start}, {end}) contains no raw rows' in str(error.value)
    assert calls == []


@pytest.mark.parametrize('rule', [False, True])
def test_exact_recorded_96_row_case_and_active_dates_ignore_ratio_and_guards(bars, rule):
    manifest = _manifest(rule=rule).set_split_config(1, 0, 0)
    manifest.val_predict_guard = manifest.test_predict_guard = False
    with pytest.raises(ValueError, match=r'validation split .*contains no raw rows'):
        manifest.prepare_data(bars.head(96), {})


@pytest.mark.parametrize('index', [1, 2])
def test_pre_split_selector_cannot_erase_a_required_window(bars, index):
    start, end = DATES[index * 2:index * 2 + 2]
    manifest = _manifest().set_pre_split_data_selector(
        lambda data: data.filter(~pl.col('datetime').is_between(start, end, closed='left')))
    with pytest.raises(ValueError, match=('validation' if index == 1 else 'test') + ' split'):
        manifest.prepare_data(bars, {})


def test_zero_width_train_and_positive_subbar_future_are_required(bars):
    with pytest.raises(ValueError, match=r'train split .*contains no raw rows'):
        _manifest((DATES[0], DATES[0], *DATES[2:])).prepare_data(bars, {})
    def stamp(hour, minute=0):
        return datetime(2024, 1, 1, hour, minute, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match=r'validation split .*contains no raw rows'):
        _manifest((stamp(0), stamp(3), stamp(3, 1), stamp(3, 59), stamp(4), stamp(5))).prepare_data(bars, {})


@pytest.mark.parametrize('zero_val,zero_test', [(True, False), (False, True), (True, True)])
def test_intentional_zero_future_windows_and_nonempty_membership_are_preserved(bars, zero_val, zero_test):
    dates = list(DATES)
    if zero_val:
        dates[3] = dates[2]
    if zero_test:
        dates[5] = dates[4]
    manifest = _manifest(dates)
    prepared = manifest.prepare_data(bars, {})
    for index, name in enumerate(('train', 'val', 'test')):
        start, end = dates[index * 2:index * 2 + 2]
        expected = bars.filter(pl.col('datetime').is_between(start, end, closed='left'))
        assert prepared['y_' + name].equals(expected['close'])
    assert manifest.compute_test_bars(bars, {}).equals(bars.filter(pl.col('datetime').is_between(dates[4], dates[5], closed='left')))


def test_zero_future_does_not_waive_its_independently_empty_positive_neighbor(bars):
    dates = (*DATES[:2], DATES[2], DATES[2], *DATES[4:])
    with pytest.raises(ValueError, match=r'test split .*contains no raw rows'):
        _manifest(dates).prepare_data(bars.head(96), {})


@pytest.mark.parametrize('stage', ['raw', 'formed', 'retained'])
@pytest.mark.parametrize('index', [1, 2])
def test_repeated_local_hour_is_positive_at_every_admission_stage(bars, stage, index):
    start = datetime(2024, 11, 3, 1, tzinfo=ZoneInfo('America/New_York'), fold=0)
    end = start.replace(fold=1)
    dates = [start] * 6
    dates[index * 2 + 1] = end
    with pytest.raises(ValueError if stage == 'raw' else StrictModeError, match=('validation' if index == 1 else 'test') + ' split'):
        admit_dated_split(tuple(dates), bars.head(0), index, stage=stage)
    # Equal instants remain intentional zero windows, including different zones.
    dates[index * 2 + 1] = start.astimezone(timezone.utc)
    empty = bars.head(0)
    assert admit_dated_split(tuple(dates), empty, index, stage=stage) is empty


@pytest.mark.parametrize('rule', [False, True])
@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('index', [1, 2])
def test_recorded_fold_hour_cannot_be_erased_from_required_native_split(rule, strict, index):
    start = datetime(2024, 11, 3, 1, tzinfo=ZoneInfo('America/New_York'), fold=0)
    end = start.replace(fold=1)
    before = datetime(2024, 11, 2, tzinfo=timezone.utc)
    after = datetime(2024, 11, 4, tzinfo=timezone.utc)
    dates = (before, start, start, end, end, after) if index == 1 else (before, start, start, start, start, end)
    recorded = pl.read_parquet(FIXTURE).filter(pl.col('datetime').is_between(before, after, closed='left'))
    window = recorded.filter(pl.col('datetime').is_between(start, end, closed='left'))
    assert window.height == 1
    absent = recorded.filter(~pl.col('datetime').is_between(start, end, closed='left'))
    manifest = _manifest(dates, rule=rule).set_strict_mode(strict)
    with pytest.raises(ValueError, match=('validation' if index == 1 else 'test') + ' split'):
        manifest.prepare_data(absent, {})
    if index == 2:
        with pytest.raises(ValueError, match='test split'):
            manifest.compute_test_bars(absent, {})


def test_nonempty_natural_gaps_retain_exact_half_open_membership(bars):
    dates = tuple(date(2024, 1, day) for day in (1, 3, 4, 5, 6, 7))
    manifest = _manifest(dates)
    prepared = manifest.prepare_data(bars, {})
    for index, name in enumerate(('train', 'val', 'test')):
        expected = bars.filter(pl.col('datetime').is_between(*dates[index * 2:index * 2 + 2], closed='left'))
        assert prepared['y_' + name].equals(expected['close'])
    assert [prepared['x_' + name].height for name in ('train', 'val', 'test')] == [48, 24, 24]


def test_test_only_replay_does_not_materialize_or_require_other_windows(bars):
    manifest = _manifest()
    test = bars.filter(pl.col('datetime').is_between(DATES[4], DATES[5], closed='left'))
    assert manifest.compute_test_bars(test, {}).equals(test)
    with pytest.raises(ValueError, match=r'test split .*contains no raw rows'):
        manifest.compute_test_bars(bars.head(96), {})


def test_feature_exhaustion_refuses_before_target_or_scaler_fit(bars, monkeypatch):
    manifest = _manifest().add_feature(lambda data: data.head(0)).set_scaler(RobustScaler)

    def forbidden(*args, **kwargs):
        raise AssertionError('Empty retained training frame reached fitting')

    monkeypatch.setattr(IdentityTarget, '__init__', forbidden)
    monkeypatch.setattr(RobustScaler, '__init__', forbidden)
    with pytest.raises(StrictModeError, match=r'train split .*contains no retained rows'):
        manifest.prepare_data(bars, {})


@pytest.mark.parametrize('stage', ['indicator', 'target', 'formation'])
def test_recorded_processing_exhaustion_is_a_failed_candidate_boundary(bars, monkeypatch, stage):
    manifest = _manifest().set_scaler(RobustScaler)
    if stage == 'indicator':
        manifest.add_indicator(roc, period=1000)
    elif stage == 'target':
        manifest.with_target_label('next_return', NextReturnTarget, transform_params={'periods': 1000})
    else:
        manifest.set_bar_formation(volume_bars, volume_threshold=float(bars['volume'].sum()) * 2)

    def forbidden(*args, **kwargs):
        raise AssertionError('Exhausted dated training frame reached scaler fitting')

    monkeypatch.setattr(RobustScaler, '__init__', forbidden)
    expected = 'formed' if stage == 'formation' else 'retained'
    with pytest.raises(StrictModeError, match='train split .*contains no ' + expected + ' rows'):
        manifest.prepare_data(bars, {'bar_type': 'volume'})
    if stage == 'formation':
        with pytest.raises(StrictModeError, match=r'test split .*contains no formed rows'):
            manifest.compute_test_bars(bars, {'bar_type': 'volume'})


def test_fitted_preparation_reuse_and_explicit_date_clear_preserve_the_boundary(bars, monkeypatch):
    manifest = _manifest()
    prepared = manifest.prepare_data(bars, {})
    fitted = prepared['_fitted_params']

    def forbidden(*args, **kwargs):
        raise AssertionError('Fitted target was constructed again')

    with monkeypatch.context() as guard:
        guard.setattr(IdentityTarget, '__init__', forbidden)
        reused = manifest._prepare_data_with_fitted(bars, {}, fitted)
        for name in ('train', 'val', 'test'):
            assert reused['y_' + name].equals(prepared['y_' + name])
        with pytest.raises(ValueError, match='validation split'):
            manifest._prepare_data_with_fitted(bars.head(96), {}, fitted)
    refit = manifest.with_params_override(split_config=(1, 0, 0))
    result = refit.prepare_data(bars.head(96), {})
    assert result['x_train'].height == 96 and result['x_val'].height == result['x_test'].height == 0


@pytest.mark.parametrize('strict', [False, True])
def test_context_only_future_rows_refuse_after_cco_removal_before_pca(bars, monkeypatch, strict):
    start, end = DATES[2:4]
    manifest = (_manifest().set_strict_mode(strict)
        .add_feature(lambda data: data.filter(~pl.col('datetime').is_between(start, end, closed='left')))
        .set_scaler(CausalRollingRobustScaler, extra_params={'window': 5, 'min_samples': 2})
        .set_pca_compression())
    observed = []
    transform = CausalRollingRobustScaler.transform

    def observe(instance, data):
        observed.append(data.height)
        return transform(instance, data)

    def forbidden(*args, **kwargs):
        raise AssertionError('Context-only future split reached PCA fitting')

    monkeypatch.setattr(CausalRollingRobustScaler, 'transform', observe)
    monkeypatch.setattr(PCA, 'fit_transform', forbidden)
    with pytest.raises(StrictModeError, match=r'validation split .*contains no retained rows'):
        manifest.prepare_data(bars, {'auto_pca': True, 'pca_k': 1})
    assert observed == [96, 5]


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('stage', ['raw', 'retained'])
@pytest.mark.parametrize('family', ['ml', 'rule'])
def test_real_cli_workers_refuse_raw_absence_and_record_processing_failure(tmp_path, monkeypatch, bars, workers, stage, family):
    from tests.test_column_shims import _config

    config = _config()
    if family == 'rule':
        rule, errors = parse(FIXTURE.parents[2] / 'limen/yaml/templates/rule_based.yaml')
        assert not errors
        rule['sfd']['manifest']['split_dates'] = config['sfd']['manifest']['split_dates']
        config = rule
    config['uel']['n_permutations'] = 1
    declaration = config['sfd']['manifest']
    actual = bars
    if stage == 'raw':
        actual = actual.filter(~pl.col('datetime').is_between(date(2024, 1, 8), date(2024, 1, 10), closed='left'))
    elif family == 'ml':
        declaration['target']['transform_params']['periods'] = 1000
    else:
        config['sfd']['params']['rsi_period'] = [1000]
    source = tmp_path / 'admitted.parquet'
    actual.write_parquet(source)
    declaration['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(source)}}
    config['uel']['output_path'] = 'dated_cli'
    submitted = tmp_path / 'dated.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ['run', '--workers', str(workers), '--no-progress-bar', str(submitted)])
    if stage == 'raw':
        assert result.exit_code == 1 and 'validation split' in result.output and 'contains no raw rows' in result.output
    else:
        assert result.exit_code == 0, result.output
        log = pl.read_csv(tmp_path / 'results/dev/dated_cli/results.csv')
        assert log.height == 1
        assert log['strict_mode_error'].to_list() == ['Manifest train split [2024-01-01, 2024-01-08) contains no retained rows']


def test_genuine_saved_round_refuses_short_test_source_before_metric_reconstruction(tmp_path, monkeypatch, bars):
    from tests.test_column_shims import _config

    replay = importlib.import_module('limen.log._snapshot_backtest_round')
    config = _config()
    declaration = config['sfd']['manifest']
    declaration['reference_architecture'] = 'limen.sfd.reference_architecture.logreg_binary'
    declaration['target'] = {'name': 'label', 'class': 'limen.targets.QuantileBinaryTarget',
        'fit_params': {'source_column': 'close', 'quantile': 0.5}, 'transform_params': {'shift': -1}}
    config['sfd']['params'] = {'C': [0.1], 'max_iter': [1000]}
    declaration['backtest'].update(take_profit_bps=100.0, stop_loss_bps=50.0)
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=bars, experiment_dir=tmp_path,
        search_strategy=build_search_strategy(config), yaml_reference=config, checkpoint_interval=1)
    loop.run('dated_replay', n_permutations=1, prep_each_round=True, post_processing=True, progress_bar=False,
        recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    assert loop.experiment_log.height == 1 and loop.experiment_log['strict_mode_error'].null_count() == 1
    log = Log(loop)
    assert replay.snapshot_backtest_round(log, 0, lambda frame: frame)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir() if path.is_file()}
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Short test source reached metric reconstruction')

    monkeypatch.setattr(replay, 'backtest_snapshot', forbidden)
    log.data = bars.head(96)
    with pytest.raises(ValueError, match=r'test split .*contains no raw rows'):
        replay.snapshot_backtest_round(log, 0, lambda frame: frame)
    assert not calls
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir() if path.is_file()} == before
