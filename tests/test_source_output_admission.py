from pathlib import Path

import polars as pl
import pytest

from limen.experiment import Manifest, MLManifest, RuleBasedManifest, UniversalExperimentLoop
from limen.yaml import CompiledSFD
from tests.test_column_shims import _config


@pytest.fixture(scope='module')
def frame():
    return pl.read_parquet(Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet', n_rows=3)


@pytest.mark.parametrize('manifest_type', [Manifest, MLManifest, RuleBasedManifest])
@pytest.mark.parametrize('kind', ['none', 'list', 'mapping', 'owner', 'lazy', 'series', 'pandas'])
@pytest.mark.parametrize('api', ['data', 'test_data'])
def test_invalid_effective_outputs_refuse_with_source_and_type_exactly_once(frame, kind, api, manifest_type):
    value = {'none': None, 'list': frame.to_dicts(), 'mapping': frame.row(0, named=True),
             'owner': Manifest(), 'lazy': frame.lazy(), 'series': frame['close'], 'pandas': frame.to_pandas()}[kind]
    calls = []

    def recorded_source():
        calls.append(1)
        return value

    manifest = manifest_type()
    getattr(manifest, f'set_{api}_source')(recorded_source)
    with pytest.raises(TypeError, match=r'materialized polars\.DataFrame') as error:
        getattr(manifest, f'fetch_{api}')()
    assert 'recorded_source' in str(error.value) and type(value).__name__ in str(error.value)
    assert calls == [1]


@pytest.mark.parametrize('kind', ['none', 'list', 'mapping', 'lazy'])
def test_bound_owner_data_stays_authoritative_over_a_valid_raw_return(frame, kind):
    value = {'none': None, 'list': frame.to_dicts(), 'mapping': frame.row(0, named=True), 'lazy': frame.lazy()}[kind]

    class Owner:
        def __init__(self):
            self.calls = 0

        def load(self):
            self.calls += 1
            self.data = value
            return frame

    owner = Owner()
    with pytest.raises(TypeError, match=r'materialized polars\.DataFrame') as error:
        Manifest().set_data_source(owner.load).fetch_data()
    assert 'owner.data' in str(error.value)
    assert owner.calls == 1 and owner.data is value


def test_valid_materialized_source_preserves_exact_frame_identity(frame):
    calls = []

    def load():
        calls.append(1)
        return frame

    assert Manifest().set_data_source(load).fetch_data() is frame
    assert calls == [1]


class _RecordedOwner:
    calls = 0
    output = None
    returned = None

    def load(self):
        type(self).calls += 1
        self.data = type(self).output
        return type(self).returned


@pytest.mark.parametrize('kind', ['none', 'list', 'mapping', 'lazy'])
def test_unbound_owner_data_stays_authoritative_over_a_valid_raw_return(frame, kind, monkeypatch):
    value = {'none': None, 'list': frame.to_dicts(), 'mapping': frame.row(0, named=True), 'lazy': frame.lazy()}[kind]
    monkeypatch.setattr(_RecordedOwner, 'calls', 0)
    monkeypatch.setattr(_RecordedOwner, 'output', value)
    monkeypatch.setattr(_RecordedOwner, 'returned', frame)
    with pytest.raises(TypeError, match=r'materialized polars\.DataFrame') as error:
        Manifest().set_data_source(_RecordedOwner.load).fetch_data()
    assert 'owner.data' in str(error.value)
    assert _RecordedOwner.calls == 1


def test_materialized_empty_frame_is_preserved_for_the_preparation_boundary(frame):
    for empty in (frame.head(0), frame.select([])):
        assert Manifest().set_data_source(lambda admitted=empty: admitted).fetch_data() is empty


@pytest.mark.parametrize('kind', ['none', 'list'])
def test_invalid_source_refuses_uel_construction_before_preparation_or_model(frame, tmp_path, monkeypatch, kind):
    compiled = CompiledSFD(_config())
    value = None if kind == 'none' else frame.to_dicts()
    source_calls, downstream = [], []

    def source():
        source_calls.append(1)
        return value

    def forbidden(*args, **kwargs):
        downstream.append(1)
        raise AssertionError('Invalid source reached preparation or fitting')

    compiled.manifest().set_data_source(source)
    monkeypatch.setattr(MLManifest, 'prepare_data', forbidden)
    monkeypatch.setattr(MLManifest, 'run_model', forbidden)
    directory = tmp_path / 'refused'
    with pytest.raises(TypeError, match=r'materialized polars\.DataFrame'):
        UniversalExperimentLoop(sfd=compiled, experiment_dir=directory)
    assert source_calls == [1] and not downstream and not directory.exists()
