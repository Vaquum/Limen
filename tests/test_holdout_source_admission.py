import datetime
import json
from types import SimpleNamespace

import polars as pl
import pytest
from ruamel.yaml import YAML

from limen.data.historical_data import HistoricalData
from limen.experiment import UniversalExperimentLoop
from limen.experiment import _holdout_history as history
from limen.experiment._holdout_source import admit_frame, load_admitted_source, source_spec
from limen.experiment.holdout import reserve_holdout
from limen.experiment.manifest_core import DataSourceConfig
from limen.yaml import CompiledSFD
from tests.test_holdout_research import _FIXTURE, _project
from tests.test_trainer import _E2E_LOGREG_YAML


def _manifest(method, params):
    return SimpleNamespace(data_source_config=DataSourceConfig(method, params))


def _binding(root, interval=3600):
    return history.ProjectBinding(root=root, project_id=None, revision=0, cutoff_ns=None, interval_seconds=interval)


def test_holdout_source_spec_matrix(tmp_path):
    manifest = _manifest(HistoricalData.get_any_file, {'file_path_or_url': str(_FIXTURE)})
    kind, params, interval = source_spec(manifest, _binding(tmp_path))
    assert kind == 'local_file'
    assert params == {'file_path_or_url': str(_FIXTURE.resolve())}
    assert interval == 3600
    csv = tmp_path / 'spot.csv'
    csv.write_text('x')
    manifest = _manifest(HistoricalData.get_any_file, {'file_path_or_url': str(csv), 'has_header': False, 'columns': ['a']})
    kind, params, _ = source_spec(manifest, _binding(tmp_path))
    assert kind == 'local_file'
    assert params['has_header'] is False
    assert params['columns'] == ['a']
    bad_params = [{'file_path_or_url': str(tmp_path / 'a.zip')}, {'file_path_or_url': 'https://x/y.parquet'},
        {'file_path_or_url': str(tmp_path / '{name}.parquet')}, {'file_path_or_url': str(_FIXTURE), 'row_count_limit': 10},
        {'file_path_or_url': str(_FIXTURE), 'n_rows': 10}, {'file_path_or_url': str(_FIXTURE), 'bogus': 1},
        {'file_path_or_url': str(csv), 'has_header': 'yes'}]
    for params in bad_params:
        with pytest.raises(ValueError):
            source_spec(_manifest(HistoricalData.get_any_file, params), _binding(tmp_path))
    with pytest.raises(ValueError):
        source_spec(_manifest(HistoricalData.get_spot_dollar_klines, {}), _binding(tmp_path))


class _CleanSubclass(HistoricalData):
    pass


class _StoreOverride(_CleanSubclass):
    def _store(self, data):
        return super()._store(data)


def test_holdout_source_subclass_matrix(tmp_path):
    file_params = {'file_path_or_url': str(_FIXTURE)}
    kind, _, _ = source_spec(_manifest(_CleanSubclass().get_any_file, file_params), _binding(tmp_path))
    assert kind == 'local_file'
    kind, _, _ = source_spec(_manifest(HistoricalData().get_any_file, file_params), _binding(tmp_path))
    assert kind == 'local_file'
    with pytest.raises(ValueError, match='overrides/subclasses'):
        source_spec(_manifest(_StoreOverride().get_any_file, file_params), _binding(tmp_path))
    dirty = _CleanSubclass()
    dirty.extra_state = 1
    with pytest.raises(ValueError, match='overrides/subclasses'):
        source_spec(_manifest(dirty.get_any_file, file_params), _binding(tmp_path))
    shadowed = HistoricalData()
    shadowed._store = lambda data: data
    with pytest.raises(ValueError, match='overrides/subclasses'):
        source_spec(_manifest(shadowed.get_any_file, file_params), _binding(tmp_path))
    tagged = HistoricalData()
    tagged.note = 'benign'
    kind, _, _ = source_spec(_manifest(tagged.get_any_file, file_params), _binding(tmp_path))
    assert kind == 'local_file'
    with pytest.raises(ValueError, match='overrides/subclasses'):
        source_spec(_manifest(_CleanSubclass().get_spot_klines, {'kline_size': 3600}), _binding(tmp_path))


def _csv_copy(source, destination, *, header=True):
    frame = pl.read_parquet(source)
    if header:
        frame.write_csv(destination)
    else:
        frame.write_csv(destination, include_header=False)
    return frame


def _csv_configuration(path, csv_path):
    config = YAML(typ='safe').load(_E2E_LOGREG_YAML)
    manifest = config['sfd']['manifest']
    manifest['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(csv_path)}}
    manifest['split_dates'] = dict(zip(('train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end'),
        ('2024-01-01', '2024-01-03', '2024-01-03', '2024-01-04', '2024-01-04', '2024-01-05'), strict=True))
    config['uel']['n_permutations'] = 1
    config['sfd']['params']['C'] = [1.0]
    (path / 'research.yaml').write_text(json.dumps(config))
    return config


def test_holdout_csv_research_end_to_end(tmp_path):
    root = _project(tmp_path / 'project')
    csv_path = root / 'spot.csv'
    _csv_copy(_FIXTURE, csv_path)
    config = _csv_configuration(root, csv_path)
    reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    compiled = CompiledSFD(config, project_root=root)
    loop = UniversalExperimentLoop(sfd=compiled, project_root=root)
    assert loop.data.height == 96
    assert loop.data['datetime'].dt.epoch('ns').max() < history.utc_day('2024-01-05')
    kind, params, _ = source_spec(compiled.manifest(), history.bind(root))
    assert kind == 'local_file'
    assert params['file_path_or_url'].endswith('spot.csv')


def test_holdout_headerless_csv_matches_headed(tmp_path):
    headed = tmp_path / 'headed.csv'
    headless = tmp_path / 'headless.csv'
    frame = _csv_copy(_FIXTURE, headed)
    frame.write_csv(headless, include_header=False)
    binding = _binding(tmp_path)
    upper = history.utc_day('2024-01-05')
    headed_out, _ = load_admitted_source(
        _manifest(HistoricalData.get_any_file, {'file_path_or_url': str(headed)}), binding,
        lower_ns=history.utc_day('2024-01-01'), upper_ns=upper, complete=False)
    headless_out, _ = load_admitted_source(
        _manifest(HistoricalData.get_any_file, {'file_path_or_url': str(headless), 'has_header': False, 'columns': frame.columns}),
        binding, lower_ns=history.utc_day('2024-01-01'), upper_ns=upper, complete=False)
    assert headless_out.height == headed_out.height == 96
    assert headless_out['datetime'].to_list() == headed_out['datetime'].to_list()


def test_holdout_csv_format_conversion_detected(tmp_path):
    csv_path = tmp_path / 'spot.csv'
    _csv_copy(_FIXTURE, csv_path)
    binding = _binding(tmp_path)
    lower, upper = history.utc_day('2024-01-01'), history.utc_day('2024-01-05')
    _, parquet_evidence = load_admitted_source(
        _manifest(HistoricalData.get_any_file, {'file_path_or_url': str(_FIXTURE)}), binding,
        lower_ns=lower, upper_ns=upper, complete=False)
    _, csv_evidence = load_admitted_source(
        _manifest(HistoricalData.get_any_file, {'file_path_or_url': str(csv_path)}), binding,
        lower_ns=lower, upper_ns=upper, complete=False)
    assert csv_evidence['kind'] == 'local_file'
    assert csv_evidence['native_witness'] == 'caller-declared'
    assert csv_evidence['admitted'] == parquet_evidence['admitted']
    assert csv_evidence['admitted_sha256'] != parquet_evidence['admitted_sha256']


def test_holdout_non_numeric_ohlc_is_value_error():
    frame = pl.DataFrame({'datetime': [datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)],
        'open': [1.0], 'high': ['n/a'], 'low': [1.0], 'close': [1.0]})
    with pytest.raises(ValueError, match='OHLC must be numeric'):
        admit_frame(frame, interval=3600, lower_ns=None, upper_ns=history.utc_day('2024-01-02'))


def test_holdout_csv_datetime_rules(tmp_path):
    frame = pl.read_parquet(_FIXTURE).head(48).with_columns(
        pl.col('datetime').dt.strftime('%Y-%m-%dT%H:%M:%SZ').alias('datetime'))
    zulu = tmp_path / 'zulu.csv'
    frame.write_csv(zulu)
    binding = _binding(tmp_path)
    admitted, _ = load_admitted_source(
        _manifest(HistoricalData.get_any_file, {'file_path_or_url': str(zulu)}), binding,
        lower_ns=history.utc_day('2024-01-01'), upper_ns=history.utc_day('2024-01-03'), complete=False)
    assert admitted.height == 48
    assert admitted.schema['datetime'].time_zone == 'UTC'
    naive = tmp_path / 'naive.csv'
    pl.read_parquet(_FIXTURE).head(48).with_columns(
        pl.col('datetime').dt.strftime('%Y-%m-%d %H:%M:%S').alias('datetime')).write_csv(naive)
    naive_out, _ = load_admitted_source(
        _manifest(HistoricalData.get_any_file, {'file_path_or_url': str(naive)}), binding,
        lower_ns=history.utc_day('2024-01-01'), upper_ns=history.utc_day('2024-01-03'), complete=False)
    assert naive_out['datetime'].to_list() == admitted['datetime'].to_list()
    corrupt = tmp_path / 'corrupt.csv'
    lines = zulu.read_text().splitlines()
    lines[10] = lines[10].split(',', 1)[0][:10] + 'not-a-date' + ',' + lines[10].split(',', 1)[1]
    corrupt.write_text('\n'.join(lines) + '\n')
    with pytest.raises(ValueError):
        load_admitted_source(_manifest(HistoricalData.get_any_file, {'file_path_or_url': str(corrupt)}),
            binding, lower_ns=history.utc_day('2024-01-01'), upper_ns=history.utc_day('2024-01-03'), complete=False)
