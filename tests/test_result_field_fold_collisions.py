from functools import wraps

import polars as pl
import pytest

from tests.test_parallel_rounds import records
from tests.test_result_field_collisions import (
    _assert_message,
    _configuration,
    _loop,
    _rows,
    _run,
)


@pytest.mark.parametrize('mode', ('legacy', 'msq'))
@pytest.mark.parametrize('field', ('id', '_warnings'))
def test_later_fold_bookkeeping_is_rejected_before_trial_acceptance(
    mode, field, tmp_path, monkeypatch,
):
    path = tmp_path / 'run'
    loop = _loop(mode, _configuration(walk_forward=True), path)
    assert loop.manifest is not None
    architecture = loop.manifest.architecture_function
    assert architecture is not None
    evaluated = []

    @wraps(architecture)
    def colliding_architecture(data, **parameters):
        result = architecture(data, **parameters)
        evaluated.append(parameters['C'])
        if len(evaluated) == 4:
            result[field] = 'caller-supplied fold bookkeeping'
        return result

    monkeypatch.setattr(loop.manifest, 'architecture_function', colliding_architecture)
    with pytest.raises(ValueError) as error:
        _run(loop, mode)
    _assert_message(error, 'result bookkeeping', (field,))
    assert evaluated == [0.1, 0.1, 0.3, 0.3]

    saved = _rows(path)
    assert len(saved) == 1
    assert float(saved[0]['C']) == 0.1
    identities = {saved[0]['id']}
    assert loop.fold_results.height == 2
    assert set(loop.fold_results['id'].cast(pl.String)) == identities
    assert loop.fold_results['fold'].sort().to_list() == [0, 1]
    tracks = pl.read_parquet(path / 'trial_returns.parquet')
    assert set(tracks['trial']) == identities
    if mode == 'msq':
        assert [row['round_id'] for row in records(path)] == [saved[0]['id']]
