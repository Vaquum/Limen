from pathlib import Path

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
@pytest.mark.parametrize('ratios,lengths', (
    ((1, 0, 0), (120, 0, 0)),
    ((1, 1, 0), (60, 60, 0)),
    ((1, 0, 1), (60, 0, 60)),
    ((3, 1, 1), (72, 24, 24)),
))
def test_positive_overrides_preserve_recorded_rows(manifest_type, ratios, lengths):
    source = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(120)
    manifest = manifest_type().set_split_config(8, 1, 2)
    clone = manifest.with_params_override(split_config=ratios)
    splits = _resolve_split(clone, source)
    assert tuple(split.height for split in splits) == lengths
    assert pl.concat(splits).equals(source)
    assert clone.split_config == ratios
    assert manifest.split_config == (8, 1, 2)


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
