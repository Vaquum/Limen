from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.utils.param_space import ParamSpace


PARAMS = {'a': [0, 1], 'b': [10, 20]}
ORDER = [{'a': 0, 'b': 10}, {'a': 1, 'b': 10}, {'a': 0, 'b': 20}, {'a': 1, 'b': 20}]


def recorded_prep(data, round_params):
    return {'source': data}


def recorded_model(data, round_params):
    return {'mean_close': data['source']['close'].mean()}


@pytest.mark.parametrize('count', (1, 3, 4))
def test_standard_nonrandom_run_enumerates_prefix(count, tmp_path, monkeypatch):
    def reject_sample(*args, **kwargs):
        raise AssertionError('Nonrandom enumeration must not sample the grid')
    monkeypatch.setattr('random.Random.sample', reject_sample)
    source = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(40)
    for repeat in range(2):
        sfd = SimpleNamespace(params=lambda: PARAMS, prep=recorded_prep, model=recorded_model)
        loop = UniversalExperimentLoop(data=source, sfd=sfd)
        loop.run(experiment_name=str(tmp_path / f'run-{repeat}'), n_permutations=count, random_search=False, progress_bar=False)
        assert loop.experiment_log.select('a', 'b').to_dicts() == ORDER[:count]
        assert loop.experiment_log['mean_close'].to_list() == [source['close'].mean()] * count


@pytest.mark.parametrize('count', (0, 1, 3, 4, 10))
def test_helper_enumeration_prefix_and_exhaustion(count):
    space = ParamSpace(PARAMS, count, sample=False)
    expected = ORDER[:count]
    assert [space.generate(random_search=False) for _ in expected] == expected
    assert space.generate(random_search=False) is None


def test_huge_space_enumeration_is_bounded(monkeypatch):
    def reject_bits(*args):
        raise AssertionError('Enumeration must not sample a huge grid')
    monkeypatch.setattr('random.Random.getrandbits', reject_bits)
    params = {f'p{i}': list(range(10)) for i in range(19)}
    space = ParamSpace(params, 3, sample=False)
    assert space.total_space == 10**19
    assert space.df_params.height == 3
    assert [space.generate(random_search=False) for _ in range(3)] == [{**dict.fromkeys(params, 0), 'p0': value} for value in range(3)]


@pytest.mark.parametrize('sample', (True, False))
def test_negative_count_remains_invalid(sample):
    with pytest.raises(ValueError, match='Sample larger than population or is negative'):
        ParamSpace(PARAMS, -1, sample=sample)


def test_default_sampling_mode_is_unchanged():
    default = ParamSpace(PARAMS, 3, seed=123)
    explicit = ParamSpace(PARAMS, 3, seed=123, sample=True)
    assert default.df_params.equals(explicit.df_params)
    assert [default.generate() for _ in range(3)] == [explicit.generate() for _ in range(3)] == [{'a': 1, 'b': 10}, {'a': 1, 'b': 20}, {'a': 0, 'b': 10}]
