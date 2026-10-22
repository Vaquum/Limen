import csv
from types import SimpleNamespace

import polars as pl
import pytest

from limen.experiment import UniversalExperimentLoop
from limen.experiment import _round_storage as _storage
from limen.experiment.errors import StrictModeError
from limen.yaml import CompiledSFD, build_search_strategy
from tests.test_objective_resume import COLUMN, _config as _objective_config
from tests.test_record_execution import _config, _records
from tests.test_record_model_outputs import recorded_source


RECORDING = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)


def _configuration(**params):
    config = _config('binary')
    config['sfd']['params'] = {'C': [0.1], 'max_iter': [1000], **params}
    return config


def _loop(config, path, *, custom=False, callback=None):
    compiled = CompiledSFD(config)
    sfd = compiled
    if custom:
        prepared = UniversalExperimentLoop(sfd=compiled, data=recorded_source())
        sfd = SimpleNamespace(__name__=__name__, params=compiled.params, prep=prepared.prep, model=prepared.model)
    return UniversalExperimentLoop(
        sfd=sfd, data=recorded_source(),
        search_strategy=build_search_strategy(config), experiment_dir=path,
        checkpoint_interval=1, feedback_interval=1, intra_callback=callback,
    )


def _run(loop, count, *, stop_after=None, failures=(), **kwargs):
    model = loop.model
    assert model is not None
    calls = 0

    def run_model(data, round_params):
        nonlocal calls
        current = calls
        calls += 1
        if stop_after is not None and calls == stop_after:
            loop._shutdown_requested = True
        if current in failures:
            raise StrictModeError('Recorded strict-mode failure')
        return model(data, round_params)

    loop.model = run_model
    loop.run('serial_parameters', n_permutations=count, prep_each_round=True,
             progress_bar=False, recording=RECORDING, **kwargs)


def _rows(path):
    with (path / 'results.csv').open(newline='') as handle:
        return list(csv.DictReader(handle))


def _parameters_equal(left, right, columns):
    for column in columns:
        assert left[column].dtype == right[column].dtype
        assert left[column].to_list() == right[column].to_list()


@pytest.mark.parametrize('custom', (False, True))
def test_completed_serial_resume_uses_typed_success_parameters(custom, tmp_path):
    config = _configuration(label=['01', '1'], nullable=[None, ''], flag=[True], count=[7], fraction=[0.125])
    path = tmp_path / 'completed'
    first = _loop(config, path, custom=custom)
    assert (first.manifest is None) == custom
    _run(first, 4, context_params={'extra': '009'})
    expected = first.experiment_log
    before = _rows(path)
    records = _records(path)
    columns = ['label', 'nullable', 'flag', 'count', 'fraction', 'extra']
    assert expected['label'].n_unique() == 2
    assert set(expected['nullable'].to_list()) == {None, ''}
    assert {row['flag'] for row in before} == {'True'}
    for _ in range(2):
        resumed = _loop(config, path, custom=custom)
        _run(resumed, 4, resume=True, context_params={'extra': 9, 'count': 'changed'})
        _parameters_equal(resumed.experiment_log, expected, columns)
        assert _rows(path) == before
        assert _records(path) == records


def test_interrupted_serial_resume_keeps_categories_in_feedback(tmp_path):
    config = _configuration(label=['01', '1', '001'], flag=[True])
    full = _loop(config, tmp_path / 'full')
    _run(full, 3)
    path = tmp_path / 'resumed'
    first = _loop(config, path)
    _run(first, 3, stop_after=1)
    observed = []

    def feedback(log, msq):
        observed.append((log['label'].to_list(), log['label'].dtype))

    for stop_after in (1, None, None):
        before = _rows(path)
        records = _records(path)
        resumed = _loop(config, path, callback=feedback)
        _run(resumed, 3, stop_after=stop_after, resume=True)
        assert _rows(path)[:len(before)] == before
        assert _records(path)[:len(records)] == records
        _parameters_equal(resumed.experiment_log, full.experiment_log.head(resumed.experiment_log.height),
                          ['label', 'flag', 'C', 'max_iter'])
    assert observed
    for values, dtype in observed:
        assert dtype == pl.String
        assert values == full.experiment_log['label'].to_list()[:len(values)]
    _parameters_equal(resumed.experiment_log, full.experiment_log, ['label', 'flag', 'C', 'max_iter'])


@pytest.mark.parametrize('custom', (False, True))
@pytest.mark.parametrize('failures', ((0, 1), (0,)))
def test_serial_failed_parameters_use_effective_scalar_hints(custom, failures, tmp_path):
    config = _configuration(label=['01', '1'], domain_label=['001'], count=[3], category=[''], flag=[True], fraction=[0.125])
    context = {'label': 7, 'count': '007', 'extra': '009', 'empty': '', 'null': None}
    first = _loop(config, tmp_path, custom=custom)
    _run(first, 2, failures=failures, context_params=context)
    expected = first.experiment_log
    before = _rows(tmp_path)
    record_path = _storage.round_path(tmp_path)
    records = _records(tmp_path) if record_path.exists() else []
    assert record_path.exists() == (len(failures) != 2)
    columns = ['label', 'domain_label', 'count', 'category', 'flag', 'fraction', 'extra', 'empty', 'null']
    for _ in range(2):
        resumed = _loop(config, tmp_path, custom=custom)
        _run(resumed, 2, resume=True, context_params=context)
        _parameters_equal(resumed.experiment_log, expected, columns)
        assert _rows(tmp_path) == before
        assert (_records(tmp_path) if record_path.exists() else []) == records


def test_serial_parameter_restore_preserves_objective_numeric_values(tmp_path):
    config = _objective_config()
    config['sfd']['params'].update(label=['001'], flag=[True])
    first = _loop(config, tmp_path)
    _run(first, 2)
    expected = first.experiment_log
    before = _rows(tmp_path)
    resumed = _loop(config, tmp_path)
    _run(resumed, 2, resume=True)
    _parameters_equal(resumed.experiment_log, expected, ['C', 'max_iter', 'label', 'flag', COLUMN])
    assert resumed.experiment_log[COLUMN].dtype == pl.Float64
    assert _rows(tmp_path) == before


def test_serial_strings_are_protected_before_csv_inference(tmp_path, monkeypatch):
    config = _configuration(label=['001', 'later'])
    first = _loop(config, tmp_path)
    _run(first, 2)
    read_csv = pl.read_csv

    def short_inference(*args, **kwargs):
        kwargs.setdefault('infer_schema_length', 1)
        return read_csv(*args, **kwargs)

    monkeypatch.setattr(pl, 'read_csv', short_inference)
    resumed = _loop(config, tmp_path)
    _run(resumed, 2, resume=True)
    assert resumed.experiment_log['label'].to_list() == ['001', 'later']
    assert resumed.experiment_log['label'].dtype == pl.String
