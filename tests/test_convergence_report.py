import json
import logging
import warnings
from types import SimpleNamespace

import polars as pl
import pytest
from sklearn.exceptions import ConvergenceWarning

from limen.data.utils.splits import split_data_to_prep_output, split_sequential
from limen.experiment import UniversalExperimentLoop
from limen.experiment.convergence_report import convergence_report
from limen.experiment.errors import StrictModeError
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.yaml import CompiledSFD, build_search_strategy
from tests.test_record_model_outputs import _config, recorded_source


class _PreparationConvergenceWarning(ConvergenceWarning):
    """A convergence category raised while preparing recorded observations."""


def _run(loop, count, *, resume=False, post_processing=False):
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        loop.run('results', n_permutations=count, prep_each_round=True,
                 random_search=False, progress_bar=False,
                 post_processing=post_processing, resume=resume)
    return emitted


def _diagnostic_loop(path, msq, cases=('ordinary', 'model', 'prep', 'failed')):
    bars = recorded_source().head(64).select('datetime', 'open', 'close').with_columns(
        (pl.col('close') > pl.col('open')).cast(pl.Int8).alias('target'),
    )
    params = {'warning_case': list(cases)}

    def prep(data, round_params):
        if round_params['warning_case'] == 'prep':
            warnings.warn('Preparation did not converge', _PreparationConvergenceWarning, stacklevel=2)
        return split_data_to_prep_output(split_sequential(data, (3, 1, 1)),
                                         data.columns, data['datetime'].to_list())

    def model(data, round_params):
        case = round_params['warning_case']
        if case == 'ordinary':
            warnings.warn('ConvergenceWarning is mentioned here', UserWarning, stacklevel=2)
        elif case in ('model', 'failed'):
            for _ in range(2):
                warnings.warn('Solver reached iteration limit', ConvergenceWarning, stacklevel=2)
        if case == 'failed':
            raise StrictModeError('Recorded diagnostic failure')
        return {'positive_fraction': float(data['y_test'].mean()),
                '_preds': data['y_test'].to_list()}

    sfd = SimpleNamespace(__name__=__name__, params=lambda: params, prep=prep, model=model)
    strategy = GridStrategy(ParamDomain(params)) if msq else None
    return UniversalExperimentLoop(data=bars, sfd=sfd, search_strategy=strategy,
                                   experiment_dir=path, checkpoint_interval=1)


@pytest.fixture(scope='module')
def fitted_runs(tmp_path_factory):
    runs = []
    for msq in (False, True):
        config = _config()
        config['sfd']['params'] = {'C': [1.0], 'max_iter': [1, 1000]}
        path = tmp_path_factory.mktemp(f'convergence_logreg_{msq}')
        loop = UniversalExperimentLoop(
            sfd=CompiledSFD(config), data=recorded_source(), experiment_dir=path,
            search_strategy=build_search_strategy(config) if msq else None,
            checkpoint_interval=1,
        )
        _run(loop, 2, post_processing=True)
        runs.append((loop, path))
    return runs


def test_recorded_logreg_convergence_at_run_conclusion(fitted_runs):
    for loop, path in fitted_runs:
        rows = loop.experiment_log.sort('max_iter')
        assert rows['max_iter'].to_list() == [1, 1000]
        assert rows['_convergence_warning'].to_list() == [True, False]
        report = loop.convergence_report
        assert report['rounds'] == report['observed_rounds'] == 2
        assert report['unavailable_rounds'] == 0
        assert report['convergence_warning_rounds'] == 1
        assert report['convergence_warning_pct'] == 50.0
        assert report['parameter_patterns'] == [
            {'parameter': 'C', 'value': '1.0', 'observed_rounds': 2,
             'convergence_warning_rounds': 1, 'convergence_warning_pct': 50.0},
            {'parameter': 'max_iter', 'value': '1', 'observed_rounds': 1,
             'convergence_warning_rounds': 1, 'convergence_warning_pct': 100.0},
        ]
        assert json.loads((path / 'convergence_report.json').read_text()) == report
        assert pl.read_csv(path / 'results.csv')['_convergence_warning'].to_list() == loop.experiment_log['_convergence_warning'].to_list()


@pytest.mark.parametrize('msq', (False, True))
def test_categories_duplicate_warnings_and_failed_rounds(tmp_path, msq):
    loop = _diagnostic_loop(tmp_path, msq)
    emitted = _run(loop, 4)
    rows = {row['warning_case']: row for row in loop.experiment_log.iter_rows(named=True)}
    assert rows['ordinary']['_convergence_warning'] is False
    assert rows['model']['_convergence_warning'] is True
    assert rows['prep']['_convergence_warning'] is True
    assert rows['failed']['_convergence_warning'] is None
    assert rows['failed']['strict_mode_error'] == 'Recorded diagnostic failure'
    report = loop.convergence_report
    assert report['rounds'] == 4
    assert report['observed_rounds'] == 3
    assert report['unavailable_rounds'] == 1
    assert report['convergence_warning_rounds'] == 2
    assert report['convergence_warning_pct'] == pytest.approx(200 / 3)
    assert report['parameter_patterns'] == [
        {'parameter': 'warning_case', 'value': repr(case), 'observed_rounds': 1,
         'convergence_warning_rounds': 1, 'convergence_warning_pct': 100.0}
        for case in ('model', 'prep')
    ]
    assert json.loads((tmp_path / 'convergence_report.json').read_text()) == report
    assert loop._log is None
    if msq:
        assert emitted == []
        assert json.loads(rows['ordinary']['_warnings']) == ['ConvergenceWarning is mentioned here']
        assert json.loads(rows['model']['_warnings']) == ['Solver reached iteration limit']
        assert json.loads(rows['failed']['_warnings']) == ['Solver reached iteration limit']
    else:
        assert '_warnings' not in loop.experiment_log.columns
        assert any(warning.category is UserWarning for warning in emitted)
        assert sum(issubclass(warning.category, ConvergenceWarning) for warning in emitted) == 5


def test_unavailable_legacy_evidence_and_invalid_flags(fitted_runs):
    rows = fitted_runs[0][0].experiment_log
    legacy = rows.drop('_convergence_warning')
    report = convergence_report(legacy, parameter_columns=['C', 'max_iter'])
    assert report == {'rounds': 2, 'observed_rounds': 0, 'unavailable_rounds': 2,
                      'convergence_warning_rounds': 0, 'convergence_warning_pct': None,
                      'parameter_patterns': []}
    partial = rows.with_columns(pl.when(pl.col('max_iter') == 1000)
                                .then(None).otherwise(pl.col('_convergence_warning'))
                                .alias('_convergence_warning'))
    report = convergence_report(partial, parameter_columns=['max_iter'])
    assert report['observed_rounds'] == report['unavailable_rounds'] == 1
    assert report['convergence_warning_pct'] == 100.0
    all_null = legacy.with_columns(pl.lit(None).alias('_convergence_warning'))
    assert convergence_report(all_null, parameter_columns=['C'])['convergence_warning_pct'] is None
    for expression in (pl.lit('false'), pl.lit(0), pl.lit(0.0)):
        invalid = rows.with_columns(expression.alias('_convergence_warning'))
        with pytest.raises(ValueError, match='Boolean or null'):
            convergence_report(invalid, parameter_columns=['C'])
    with pytest.raises(ValueError, match='declared parameter columns'):
        convergence_report(rows, parameter_columns=['not_a_parameter'])
    empty = convergence_report(rows.head(0), parameter_columns=['C'])
    assert empty['rounds'] == empty['observed_rounds'] == empty['unavailable_rounds'] == 0
    assert empty['convergence_warning_pct'] is None


def test_in_memory_convergence_summary(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    loop = _diagnostic_loop(None, False, ('ordinary', 'model'))
    with caplog.at_level(logging.INFO, logger='limen.experiment.experiment_core'):
        _run(loop, 2)
    assert loop.convergence_report['convergence_warning_pct'] == 50.0
    assert not (tmp_path / 'convergence_report.json').exists()
    assert any('convergence' in record.getMessage().lower() for record in caplog.records)


@pytest.mark.parametrize('evidence_mode', ('current', 'missing', 'null'))
def test_resume_preserves_observed_and_unavailable_evidence(tmp_path, evidence_mode):
    cases = ('ordinary', 'model', 'prep')
    first = _diagnostic_loop(tmp_path, True, cases)
    original = first.model
    completed = []

    def stop_after_two(data, round_params):
        result = original(data, round_params)
        completed.append(round_params['warning_case'])
        if len(completed) == 2:
            first._shutdown_requested = True
        return result

    first.model = stop_after_two
    _run(first, 3)
    saved = pl.read_csv(tmp_path / 'results.csv')
    assert saved.height == 2
    legacy = evidence_mode != 'current'
    if evidence_mode == 'missing':
        saved.drop('_convergence_warning').write_csv(tmp_path / 'results.csv')
    elif evidence_mode == 'null':
        saved.with_columns(pl.lit(None, dtype=pl.Boolean).alias('_convergence_warning')).write_csv(tmp_path / 'results.csv')
    resumed = _diagnostic_loop(tmp_path, True, cases)
    _run(resumed, 3, resume=True)
    rows = resumed.experiment_log
    assert rows.height == 3
    assert rows['id'].n_unique() == 3
    report = resumed.convergence_report
    assert report['rounds'] == 3
    assert report['unavailable_rounds'] == (2 if legacy else 0)
    assert report['observed_rounds'] == (1 if legacy else 3)
    assert report['convergence_warning_rounds'] == (1 if legacy else 2)
    assert report['convergence_warning_pct'] == pytest.approx(100.0 if legacy else 200 / 3)
    expected = [None, None, True] if legacy else [False, True, True]
    assert rows['_convergence_warning'].to_list() == expected
    assert pl.read_csv(tmp_path / 'results.csv')['_convergence_warning'].to_list() == expected
    assert json.loads((tmp_path / 'convergence_report.json').read_text()) == report

@pytest.mark.parametrize('msq', (False, True))
def test_walk_forward_warning_in_one_fold_is_one_round(tmp_path, monkeypatch, msq):
    from limen.experiment import RuleBasedManifest
    from tests.test_walk_forward_uel import _bars, _config as fold_config, _loop

    seen = []
    original = RuleBasedManifest.run_model

    def warning_in_last_fold(self, data, round_params):
        fold = vars(self)['_walk_forward_fold']
        seen.append((round_params['entry_return'], fold))
        if fold == 1 and round_params['entry_return'] == 0.0:
            for _ in range(2):
                warnings.warn('Recorded fold convergence diagnostic', ConvergenceWarning, stacklevel=2)
        return original(self, data, round_params)

    monkeypatch.setattr(RuleBasedManifest, 'run_model', warning_in_last_fold)
    loop = _loop(fold_config(), _bars(), tmp_path, search=msq)
    _run(loop, 2)
    assert sorted(seen) == [(0.0, 0), (0.0, 1), (0.001, 0), (0.001, 1)]
    assert loop.fold_results.height == 4
    rows = loop.experiment_log.sort('entry_return')
    assert rows['_convergence_warning'].to_list() == [True, False]
    report = loop.convergence_report
    assert report['rounds'] == report['observed_rounds'] == 2
    assert report['convergence_warning_rounds'] == 1
    assert report['convergence_warning_pct'] == 50.0
    assert report['parameter_patterns'] == [
        {'parameter': 'entry_return', 'value': '0.0', 'observed_rounds': 1,
         'convergence_warning_rounds': 1, 'convergence_warning_pct': 100.0},
    ]
    assert json.loads((tmp_path / 'convergence_report.json').read_text()) == report
    if msq:
        messages = rows['_warnings'].to_list()
        assert json.loads(messages[0]) == ['Recorded fold convergence diagnostic']
        assert json.loads(messages[1]) == []


def test_legacy_standard_append_preserves_rows_and_unavailable_evidence(tmp_path):
    import csv

    note = 'recorded, "quoted"\nsecond line'
    first = _diagnostic_loop(tmp_path, False, ('ordinary', 'model'))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        first.run('results', n_permutations=2, prep_each_round=True,
                  random_search=False, progress_bar=False, context_params={'note': note})
    path = tmp_path / 'results.csv'
    with path.open(newline='') as source:
        old_rows = list(csv.DictReader(source))
    for row in old_rows:
        del row['_convergence_warning']
    with path.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(old_rows[0]))
        writer.writeheader()
        writer.writerows(old_rows)
    next_run = _diagnostic_loop(tmp_path, False, ('ordinary', 'model'))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        next_run.run('results', n_permutations=2, prep_each_round=True,
                     random_search=False, progress_bar=False, context_params={'note': note})
    with path.open(newline='') as source:
        combined = list(csv.DictReader(source))
    assert len(combined) == 4
    for old, preserved in zip(old_rows, combined[:2], strict=True):
        assert {key: preserved[key] for key in old} == old
        assert preserved['_convergence_warning'] == ''
        assert preserved['note'] == note
    evidence = pl.read_csv(path)['_convergence_warning'].to_list()
    assert evidence == [None, None, False, True]
    assert next_run.convergence_report['rounds'] == next_run.convergence_report['observed_rounds'] == 2
    assert next_run.convergence_report['convergence_warning_pct'] == 50.0
    complete = convergence_report(pl.read_csv(path), parameter_columns=['warning_case'])
    assert complete['rounds'] == 4
    assert complete['observed_rounds'] == complete['unavailable_rounds'] == 2
    assert complete['convergence_warning_pct'] == 50.0


@pytest.mark.parametrize('category,case', ((UserWarning, 'ordinary'), (ConvergenceWarning, 'model')))
def test_standard_warning_error_filter_still_aborts(tmp_path, category, case):
    loop = _diagnostic_loop(tmp_path, False, (case,))
    with warnings.catch_warnings():
        warnings.simplefilter('error', category)
        with pytest.raises(category):
            loop.run('results', n_permutations=1, prep_each_round=True,
                     random_search=False, progress_bar=False)
    assert loop.convergence_report is None
    assert not (tmp_path / 'results.csv').exists()
    assert not (tmp_path / 'convergence_report.json').exists()


def test_standard_warning_remains_visible_before_model_error(tmp_path):
    loop = _diagnostic_loop(tmp_path, False, ('ordinary',))
    original = loop.model

    def fail_after_warning(data, round_params):
        _ = original(data, round_params)
        raise ValueError('Recorded model failure')

    loop.model = fail_after_warning
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match='Recorded model failure'):
            loop.run('results', n_permutations=1, prep_each_round=True,
                     random_search=False, progress_bar=False)
    assert len(emitted) == 1
    assert emitted[0].category is UserWarning
    assert str(emitted[0].message) == 'ConvergenceWarning is mentioned here'
    assert loop.convergence_report is None


def test_standard_ignored_convergence_warning_is_recorded_without_emission(tmp_path):
    loop = _diagnostic_loop(tmp_path, False, ('model',))
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        warnings.simplefilter('ignore', ConvergenceWarning)
        loop.run('results', n_permutations=1, prep_each_round=True,
                 random_search=False, progress_bar=False)
    assert emitted == []
    assert loop.experiment_log['_convergence_warning'].to_list() == [True]
    assert loop.convergence_report['rounds'] == loop.convergence_report['observed_rounds'] == 1
    assert loop.convergence_report['convergence_warning_rounds'] == 1
    assert loop.convergence_report['convergence_warning_pct'] == 100.0
    assert json.loads((tmp_path / 'convergence_report.json').read_text()) == loop.convergence_report

def test_standard_module_ignore_filter_is_preserved(tmp_path):
    loop = _diagnostic_loop(tmp_path, False, ('model',))
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('error')
        warnings.filterwarnings('ignore', category=ConvergenceWarning,
                                module=r'^limen\.experiment\.experiment_core$')
        loop.run('results', n_permutations=1, prep_each_round=True,
                 random_search=False, progress_bar=False)
    assert emitted == []
    assert loop.experiment_log['_convergence_warning'].to_list() == [True]
    assert loop.convergence_report['convergence_warning_pct'] == 100.0
