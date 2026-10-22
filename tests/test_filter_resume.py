import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from threadpoolctl import threadpool_limits

from limen.experiment.reducer.filter_types import (
    FILTER_BUILDERS, FILTER_EXCLUDE_VALUE, FILTER_SAMPLE, SAMPLE_HASH_RULE,
)
from tests.test_parallel_rounds import _assert_same, make_loop, parallel_config, records


def _configuration(*, periodic=False):
    config = parallel_config()
    config['uel'].pop('pruning_strategies')
    config['uel']['checkpoint_interval'] = 3 if periodic else 100
    return config


def _install_filters(kind):
    def callback(log, msq):
        if log.height != 3:
            return
        if kind in ('custom', 'mixed'):
            msq.remove_custom(lambda params: params['C'] == 3.0)
        if kind in ('named', 'mixed'):
            msq.set_filter('exclude_C', lambda params: params['C'] == 5.0)
        if kind == 'declarative':
            for key, filter_type, parameters in (
                ('exclude_C', FILTER_EXCLUDE_VALUE, {'param': 'C', 'value': 3.0}),
                ('sample_C', FILTER_SAMPLE, {'param': 'C', 'value': 5.0, 'fraction': 0.0}),
            ):
                msq.set_filter(key, FILTER_BUILDERS[filter_type](parameters),
                               filter_type=filter_type, filter_params=parameters)
    return callback


def _run(loop, config, workers, *, resume=False):
    with threadpool_limits(limits=1):
        loop.run(
            config['metadata']['name'], n_permutations=config['uel']['n_permutations'],
            prep_each_round=True, progress_bar=False, workers=workers, resume=resume,
            record_execution=True, record_model_outputs=True,
            recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True),
        )


def _interrupt(config, directory, workers, kind, *, crash=False):
    loop = make_loop(config, directory, callback=_install_filters(kind))
    original_append = loop._append_round_data

    def stop_after_four(*args, **kwargs):
        original_append(*args, **kwargs)
        if kwargs['round_index'] == 3:
            loop._shutdown_requested = True

    if crash:
        original_model = loop.model
        calls = 0

        def fail_fifth(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise RuntimeError('Interrupted after periodic checkpoint')
            return original_model(*args, **kwargs)

        with (patch.object(loop, 'model', side_effect=fail_fifth),
              pytest.raises(RuntimeError, match='Interrupted after periodic checkpoint')):
            _run(loop, config, workers)
    else:
        with patch.object(loop, '_append_round_data', side_effect=stop_after_four):
            _run(loop, config, workers)
    assert len(records(directory)) == 4
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpoint['metadata']['experiment_round'] == 3
    assert checkpoint['msq_state']['yielded_count'] == 4
    return checkpoint['msq_state']


def _artifacts(directory):
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob('*') if path.is_file()
    }


@pytest.mark.parametrize('workers, kind, crash', [
    (1, 'custom', False),
    (1, 'named', True),
    (2, 'mixed', False),
])
def test_unfinished_resume_refuses_lost_filters_before_mutation(tmp_path, workers, kind, crash):
    config = _configuration(periodic=crash)
    directory = tmp_path / 'interrupted'
    state = _interrupt(config, directory, workers, kind, crash=crash)
    assert state['custom_filters_count'] == (1 if kind in ('custom', 'mixed') else 0)
    assert state['named_filter_keys'] == (['exclude_C'] if kind in ('named', 'mixed') else [])
    before = _artifacts(directory)
    resumed = make_loop(config, directory)
    model_guard = patch.object(resumed, 'model', side_effect=AssertionError('Rejected resume ran a model')) if workers == 1 else nullcontext()
    with (model_guard,
          patch.object(ProcessPoolExecutor, '_launch_processes', side_effect=AssertionError('Rejected resume launched workers')),
          pytest.raises(ValueError, match='Cannot resume') as error):
        _run(resumed, config, workers, resume=True)
    if kind in ('custom', 'mixed'):
        assert '1 custom filter' in str(error.value)
    if kind in ('named', 'mixed'):
        assert 'exclude_C' in str(error.value)
    assert _artifacts(directory) == before


@pytest.mark.parametrize('workers', [1, 2])
def test_declarative_filters_survive_graceful_resume(tmp_path, workers):
    config = _configuration()
    interrupted, control = tmp_path / 'interrupted', tmp_path / 'control'
    state = _interrupt(config, interrupted, workers, 'declarative')
    assert state['named_filter_descriptors']['sample_C'][1]['hash_rule'] == SAMPLE_HASH_RULE
    resumed = make_loop(config, interrupted, callback=_install_filters('declarative'))
    _run(resumed, config, workers, resume=True)
    uninterrupted = make_loop(config, control, callback=_install_filters('declarative'))
    _run(uninterrupted, config, workers)
    expected = pl.read_csv(control / 'results.csv')
    actual = pl.read_csv(interrupted / 'results.csv', schema_overrides=expected.schema)
    assert actual['C'].to_list() == [0.1, 0.3, 0.5, 1.0, 10.0, 30.0]
    columns = [name for name in expected.columns if name != 'execution_time']
    assert_frame_equal(actual.select(columns), expected.select(columns),
                       check_exact=False, rel_tol=1e-10, abs_tol=1e-12)
    _assert_same(records(interrupted), records(control))
