from pathlib import Path
import json
from datetime import datetime
from decimal import Decimal
from enum import IntEnum
from fractions import Fraction

import numpy as np
import polars as pl
import pytest

from limen.experiment import Manifest, MLManifest, RuleBasedManifest
from limen.experiment.manifest_core import _resolve_split


@pytest.mark.parametrize('manifest_type', (Manifest, MLManifest, RuleBasedManifest))
@pytest.mark.parametrize('ratios', ((0, 1, 1), (0, 0, 1), (0, 1, 0)))
def test_override_rejects_zero_training(manifest_type, ratios):
    manifest = manifest_type().set_split_config(3, 1, 1)
    with pytest.raises(ValueError, match='train split ratio must be positive'):
        manifest.with_params_override(split_config=ratios)
    assert manifest.split_config == (3, 1, 1)


@pytest.mark.parametrize('manifest_type', (Manifest, MLManifest, RuleBasedManifest))
@pytest.mark.parametrize('route', ('setter', 'override'))
@pytest.mark.parametrize('ratios,lengths', (
    ((8, 1, 2), (87, 10, 23)),
    ((1, 0, 0), (120, 0, 0)),
    ((1, 1, 0), (60, 60, 0)),
    ((1, 0, 1), (60, 0, 60)),
    ((3, 1, 1), (72, 24, 24)),
))
def test_positive_overrides_preserve_recorded_rows(manifest_type, route, ratios, lengths):
    source = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(120)
    manifest = manifest_type().set_split_config(8, 1, 2)
    clone = manifest.with_params_override(split_config=ratios) if route == 'override' else manifest_type().set_split_config(*ratios)
    splits = _resolve_split(clone, source)
    assert tuple(split.height for split in splits) == lengths
    assert pl.concat(splits).equals(source)
    assert clone.split_config == ratios
    assert manifest.split_config == (8, 1, 2)


_BAD_ELEMENTS = (True, False, 1.5, 3.0, float('nan'), float('inf'), float('-inf'),
                 Fraction(3, 2), Decimal('1.5'), np.int64(3), np.float64(3), '3', None)


@pytest.mark.parametrize('manifest_type', (Manifest, MLManifest, RuleBasedManifest))
@pytest.mark.parametrize('position', range(3))
@pytest.mark.parametrize('value', _BAD_ELEMENTS)
def test_split_ratio_type_admission_has_identical_diagnostics_before_mutation(manifest_type, position, value):
    ratios = [3, 1, 1]
    ratios[position] = value
    _assert_same_refusal(manifest_type, tuple(ratios), 'must be a 3-tuple of ints')


@pytest.mark.parametrize('manifest_type', (Manifest, MLManifest, RuleBasedManifest))
@pytest.mark.parametrize('ratios,message', (
    ((-1, 1, 1), 'must be non-negative'), ((3, -1, 1), 'must be non-negative'),
    ((3, 1, -1), 'must be non-negative'), ((0, 0, 0), 'must not all be zero'),
    ((0, 1, 1), 'train split ratio must be positive'),
    ((0, 1, 0), 'train split ratio must be positive'), ((0, 0, 1), 'train split ratio must be positive'),
))
def test_split_ratio_range_admission_has_identical_diagnostics_before_mutation(manifest_type, ratios, message):
    _assert_same_refusal(manifest_type, ratios, message)


def _assert_same_refusal(manifest_type, ratios, message):
    manifest = manifest_type().set_split_config(8, 1, 2).set_split_dates(
        datetime(2025, 1, 1), datetime(2025, 1, 3),
        datetime(2025, 1, 3), datetime(2025, 1, 4),
        datetime(2025, 1, 4), datetime(2025, 1, 5),
    )
    original_ratios, original_dates = manifest.split_config, manifest.split_dates
    messages = []
    for operation in (lambda: manifest.set_split_config(*ratios),
                      lambda: manifest.with_params_override(split_config=ratios)):
        with pytest.raises(ValueError, match=message) as failure:
            operation()
        messages.append(str(failure.value))
        assert manifest.split_config is original_ratios
        assert manifest.split_dates is original_dates
    assert messages[0] == messages[1]


class Ratio(IntEnum):
    ONE = 1
    THREE = 3


@pytest.mark.parametrize('manifest_type', (Manifest, MLManifest, RuleBasedManifest))
def test_split_ratios_preserve_existing_integer_subclasses_and_date_policy(manifest_type):
    manifest = manifest_type().set_split_dates(
        datetime(2025, 1, 1), datetime(2025, 1, 3),
        datetime(2025, 1, 3), datetime(2025, 1, 4),
        datetime(2025, 1, 4), datetime(2025, 1, 5),
    )
    dates = manifest.split_dates
    ratios = (Ratio.THREE, Ratio.ONE, Ratio.ONE)
    assert manifest.set_split_config(*ratios) is manifest
    assert manifest.split_dates is dates
    assert manifest.split_config == ratios
    assert all(isinstance(value, Ratio) for value in manifest.split_config)
    clone = manifest.with_params_override(split_config=ratios)
    assert clone.split_dates is None
    assert manifest.split_dates is dates
    assert clone.split_config == ratios
    assert all(isinstance(value, Ratio) for value in clone.split_config)
    source = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(120)
    assert tuple(frame.height for frame in _resolve_split(clone, source)) == (72, 24, 24)


def test_admitted_walk_forward_ratio_records_and_reconstructs(tmp_path):
    from limen.inference import Trainer
    from tests.test_walk_forward_uel import _bars, _config, _loop, _run

    bars = _bars()
    output = tmp_path / 'recorded-fold-ratio'
    loop = _loop(_config(), bars, output, ratios=(3, 1, 0))
    _run(loop, n_permutations=1)
    metadata = json.loads((output / 'metadata.json').read_text())
    assert metadata['walk_forward_validation_ratio'] == [3, 1]
    assert all(type(value) is int for value in metadata['walk_forward_validation_ratio'])
    results = pl.read_csv(output / 'results.csv')
    assert results.height == 1
    sensor = Trainer(output, data=bars).train([str(results['id'][0])], fold=0)[0]
    assert sensor.fold == 0
    assert sensor.permutation_id == str(results['id'][0])
    future = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet').head(480)
    assert any(prediction.reason is None for prediction in sensor.predict_all(future))
    for invalid in (True, 3.0):
        destination = tmp_path / ('refused-' + str(invalid))
        with pytest.raises(ValueError, match='must be a 3-tuple of ints'):
            _loop(_config(), bars, destination, ratios=(invalid, 1, 0))
        assert not destination.exists()


@pytest.mark.parametrize('ratios,message', (
    ((0, 0, 0), 'must not all be zero'),
    ((-1, 1, 1), 'must be non-negative'),
    ((1, -1, 1), 'must be non-negative'),
    ((1, 1, -1), 'must be non-negative'),
    ((True, 1, 1), 'must be a 3-tuple of ints'),
    ((1.0, 1, 1), 'must be a 3-tuple of ints'),
    ([1, 1, 1], 'must be a 3-tuple of ints'),
    ((1, 1), 'must be a 3-tuple of ints'),
))
def test_other_invalid_overrides_keep_their_errors(ratios, message):
    manifest = Manifest().set_split_config(3, 1, 1)
    with pytest.raises(ValueError, match=message):
        manifest.with_params_override(split_config=ratios)
    assert manifest.split_config == (3, 1, 1)
