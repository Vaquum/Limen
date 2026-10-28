from importlib import import_module
from pathlib import Path

import polars as pl
import pytest
from ruamel.yaml import YAML

from limen.yaml import CompiledSFD, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('family', ['rbf_svm_binary', 'gaussian_hmm_binary'])
def test_dated_model_source_prepares_the_declared_window(family, monkeypatch):
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')
    calls = []

    def recorded_transport(path, **_kwargs):
        calls.append(path)
        return bars

    monkeypatch.setattr(import_module('limen.data.historical_data'), '_read_any_file', recorded_transport)
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates' / (family + '.yaml'))
    assert validate(config).valid
    compiled = CompiledSFD(config)
    manifest = compiled.manifest()
    admitted = manifest.fetch_data()
    assert len(calls) == 1 and admitted.height < 1024
    assert admitted['datetime'].min().isoformat().startswith('2024-01-01')
    params = {key: values[0] for key, values in compiled.params().items()}
    data = manifest.prepare_data(admitted, params)
    assert all(len(data['y_' + split]) > 24 for split in ('train', 'val', 'test'))
    assert data['_alignment']['first_test_datetime'].isoformat().startswith('2024-01-18')
