from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from limen.experiment.errors import StrictModeError
from limen.experiment.manifest_core import MLManifest
from tests.test_parallel_rounds import records
from tests.test_result_field_collisions import (
    MODES,
    _assert_message,
    _configuration,
    _loop,
    _rows,
    _run,
)


def _first_fold_collision(marker):
    original = MLManifest.run_model

    def run_model(manifest, data, round_params):
        affected = round_params['C'] == 0.3
        fold = vars(manifest)['_walk_forward_fold']
        if affected and fold == 1:
            Path(marker).write_text('later fold executed')
            raise StrictModeError('later fold must not hide an earlier collision')
        result = original(manifest, data, round_params)
        if affected and fold == 0:
            result['id'] = 'caller-supplied first-fold identity'
        return result

    return patch.object(MLManifest, 'run_model', run_model)


def _fold_failure_worker(function, parameters, marker):
    with _first_fold_collision(marker):
        return function(parameters)


class _FoldFailurePool(ProcessPoolExecutor):
    marker = ''

    def submit(self, function, parameters):
        return super().submit(_fold_failure_worker, function, parameters, self.marker)


@pytest.mark.parametrize('mode', MODES)
def test_fold_collision_precedes_later_strict_failure(mode, tmp_path, monkeypatch):
    path = tmp_path / 'run'
    marker = tmp_path / 'later-fold-executed'
    loop = _loop(mode, _configuration(walk_forward=True), path)
    monkeypatch.setattr(_FoldFailurePool, 'marker', str(marker))
    monkeypatch.setattr('limen.experiment._parallel_rounds.ProcessPoolExecutor', _FoldFailurePool)

    with _first_fold_collision(marker), pytest.raises(ValueError) as error:
        _run(loop, mode)
    _assert_message(error, 'result bookkeeping', ('id',))
    assert not marker.exists()

    saved = _rows(path)
    assert len(saved) == 1
    assert float(saved[0]['C']) == 0.1
    assert saved[0]['strict_mode_error'] == ''
    identities = {saved[0]['id']}
    assert loop.fold_results.height == 2
    assert set(loop.fold_results['id'].cast(pl.String)) == identities
    assert loop.fold_results['fold'].sort().to_list() == [0, 1]
    tracks = pl.read_parquet(path / 'trial_returns.parquet')
    assert set(tracks['trial']) == identities
    if mode != 'legacy':
        assert [row['round_id'] for row in records(path)] == [saved[0]['id']]
