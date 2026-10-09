import json

import polars as pl
import pytest

from limen.experiment import UniversalExperimentLoop
from limen.experiment.errors import StrictModeError
from limen.experiment.reducer import BudgetReducer, CorrelationReducer, FocusReducer, SanityReducer, SaturationReducer
from limen.yaml import CompiledSFD, build_search_strategy
from tests.test_record_model_outputs import _config as _binary_config
from tests.test_record_model_outputs import recorded_source


COLUMN = 'val_backtest_total_return'


def _config(*, objective=True, direction='maximize'):
    config = _binary_config()
    manifest = config['sfd']['manifest']
    manifest['backtest'] = {
        'product': {'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC',
                    'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0.0},
        'fee_bps': 5.0, 'slip_bps': 5.0,
    }
    if objective:
        manifest['objective'] = {'metric': 'backtest_total_return', 'direction': direction}
    return config


def _loop(config, path, *, reducers=(), search=True):
    return UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=recorded_source(), experiment_dir=path, yaml_reference=config,
        search_strategy=build_search_strategy(config) if search else None,
        checkpoint_interval=1, pruning_strategies=list(reducers), feedback_interval=1,
    )


def _run(loop, **kwargs):
    loop.run('objective_resume', n_permutations=2, prep_each_round=True, progress_bar=False, **kwargs)


def _stop_after_first(loop):
    model = loop.model
    assert model is not None

    def interrupted(data, round_params):
        result = model(data, round_params)
        loop._shutdown_requested = True
        return result

    loop.model = interrupted


def _artifact_bytes(path):
    return {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}


@pytest.mark.parametrize('objective', (False, True))
def test_same_objective_resume_matches_uninterrupted_run(objective, tmp_path):
    config = _config(objective=objective)
    reducers = [SanityReducer(metric=COLUMN)] if objective else []
    full = _loop(config, tmp_path / 'full', reducers=reducers)
    _run(full)
    first = _loop(config, tmp_path / 'resume', reducers=[SanityReducer(metric=COLUMN)] if objective else [])
    _stop_after_first(first)
    _run(first)
    original_round = (tmp_path / 'resume/round_data.jsonl').read_bytes()
    resumed = _loop(config, tmp_path / 'resume', reducers=[SanityReducer(metric=COLUMN)] if objective else [])
    _run(resumed, resume=True)
    assert resumed.experiment_log.drop('execution_time').equals(full.experiment_log.drop('execution_time'))
    assert (tmp_path / 'resume/round_data.jsonl').read_bytes().startswith(original_round)
    for path in (tmp_path / 'full', tmp_path / 'resume'):
        metadata = json.loads((path / 'metadata.json').read_text())
        if objective:
            assert metadata['objective'] == config['sfd']['manifest']['objective']
            assert resumed.experiment_log[COLUMN].dtype == pl.Float64
        else:
            assert 'objective' not in metadata
            assert COLUMN not in resumed.experiment_log.columns
    audits = []
    for path in (tmp_path / 'full/audit.jsonl', tmp_path / 'resume/audit.jsonl'):
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        for entry in entries:
            entry.pop('timestamp')
        audits.append(entries)
    assert audits[0] == audits[1]


@pytest.mark.parametrize('change', ('add', 'remove', 'direction', 'missing_column'))
def test_rejected_objective_resume_does_not_rewrite_artifacts(change, tmp_path):
    saved = _config(objective=change != 'add')
    first = _loop(saved, tmp_path)
    _stop_after_first(first)
    _run(first)
    current = _config(objective=change != 'remove', direction='minimize' if change == 'direction' else 'maximize')
    if change == 'missing_column':
        results_path = tmp_path / 'results.csv'
        pl.read_csv(results_path).drop(COLUMN).write_csv(results_path)
    before = _artifact_bytes(tmp_path)
    resumed = _loop(current, tmp_path)
    domain_before = resumed._search_strategy.domain.get_state()
    with pytest.raises(ValueError, match='objective'):
        _run(resumed, resume=True)
    assert _artifact_bytes(tmp_path) == before
    assert resumed._search_strategy.domain.get_state() == domain_before


@pytest.mark.parametrize('search', (False, True))
def test_objective_column_survives_first_strict_failure(search, tmp_path):
    loop = _loop(_config(), tmp_path, search=search)
    model = loop.model
    assert model is not None
    calls = []

    def first_failure(data, round_params):
        calls.append(round_params)
        if len(calls) == 1:
            raise StrictModeError('Recorded strict-mode failure')
        return model(data, round_params)

    loop.model = first_failure
    _run(loop)
    assert loop.experiment_log[COLUMN].dtype == pl.Float64
    assert loop.experiment_log[COLUMN][0] is None
    assert loop.experiment_log[COLUMN][1] is not None
    output = tmp_path / ('results.csv' if search else 'objective_resume.csv')
    assert pl.read_csv(output)[COLUMN].to_list() == loop.experiment_log[COLUMN].to_list()
    assert (tmp_path / 'metadata.json').exists() == search


@pytest.mark.parametrize('reducer', (
    CorrelationReducer(metric='backtest_total_return'),
    CorrelationReducer(metric=COLUMN, maximize=False),
    FocusReducer(metric=COLUMN, breakthrough_threshold=1.0, maximize=False),
    SanityReducer(metric='accuracy'),
    SaturationReducer(metric='auc'),
    BudgetReducer(max_permutations=1, trim_strategy='worst_first', metric=COLUMN, maximize=False),
))
def test_native_objective_reducer_conflicts_fail_before_artifacts(reducer, tmp_path):
    loop = _loop(_config(), tmp_path, reducers=[reducer])
    with pytest.raises(ValueError, match='metric|direction'):
        _run(loop)
    assert _artifact_bytes(tmp_path) == {}


def test_native_matching_reducers_preserve_zero_as_valid(tmp_path):
    reducers = [
        CorrelationReducer(metric=COLUMN),
        FocusReducer(metric=COLUMN, breakthrough_threshold=1.0),
        SanityReducer(metric=COLUMN),
        SaturationReducer(metric=COLUMN),
        BudgetReducer(max_permutations=10),
        BudgetReducer(max_permutations=10, trim_strategy='worst_first', metric=COLUMN),
    ]
    config = _config()
    config['sfd']['manifest']['calibration'] = {
        'threshold_function': {'func': 'limen.calibration.grid_threshold_optimizer',
                               'params': {'threshold_min': 2.0, 'threshold_max': 2.0, 'threshold_step': 0.1}},
    }
    loop = _loop(config, tmp_path, reducers=reducers)
    _run(loop)
    assert loop.experiment_log.height == 2
    assert loop.experiment_log[COLUMN].to_list() == [0.0, 0.0]
    audit = [json.loads(line) for line in (tmp_path / 'audit.jsonl').read_text().splitlines()]
    assert all(not entry['errors'] for entry in audit)
