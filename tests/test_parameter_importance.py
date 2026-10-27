from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests.utils.assert_table_equal import assert_table_equal as assert_frame_equal
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from limen.experiment import MLManifest, UniversalExperimentLoop
from limen.log.log import Log
from limen.yaml import CompiledSFD, build_search_strategy
from tests.test_factorized_preparation import _bars, _config


METRIC = 'val_backtest_total_return'
PARAMETERS = ['lookback', 'fee', 'n_estimators', 'boosting_type', 'force_row_wise', 'random_state']
COLUMNS = ['parameter', 'importance_mean', 'importance_std', 'baseline_score',
           'n_train', 'n_evaluation', 'n_configurations']


@pytest.fixture(scope='module')
def recorded_importance(tmp_path_factory):
    config = copy.deepcopy(_config())
    manifest = config['sfd']['manifest']
    manifest['objective'] = {'metric': 'backtest_total_return', 'direction': 'maximize'}
    manifest['backtest'] = {
        'product': {'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC',
                    'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0.0},
        'fee_bps': '{fee}', 'slip_bps': 3.0, 'notional_rate': 0.5,
    }
    config['sfd']['params'].update(lookback=[2, 4, 6, 8], fee=[4., 9., 14., 19.],
                                  n_estimators=[4, 8], boosting_type=['gbdt', 'dart'])
    config['uel']['n_permutations'] = 64
    path = tmp_path_factory.mktemp('parameter_importance')
    loop = UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=_bars(), experiment_dir=path, yaml_reference=config,
        search_strategy=build_search_strategy(config), checkpoint_interval=1000,
        feedback_interval=1000,
    )
    loop.run('parameter_importance', n_permutations=64, prep_each_round=True,
             progress_bar=False, post_processing=False, factorize=False,
             record_execution=True, record_model_outputs=True)
    rows = loop.experiment_log.to_pandas()
    assert len(rows) == len(rows[PARAMETERS].drop_duplicates()) == 64
    assert np.isfinite(rows[METRIC]).all() and rows[METRIC].nunique() > 1
    return loop, path, rows


def _log(rows):
    frame = pl.DataFrame({name: pl.Series(name, rows[name].to_list(), dtype=pl.Object)
                         if rows[name].dtype == object else pl.Series(name, rows[name].to_list())
                         for name in rows.columns})
    return Log(uel_object=SimpleNamespace(experiment_log=frame))


def _reference(rows, *, seed=7, repeats=3):
    if isinstance(rows, pl.DataFrame):
        rows = rows.to_pandas()
    frame = rows[PARAMETERS]
    groups = frame.groupby(PARAMETERS, sort=False, observed=True).ngroup().to_numpy()
    train, evaluation = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                               random_state=seed).split(frame, groups=groups))
    transformer = ColumnTransformer(
        [('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['boosting_type'])],
        remainder='passthrough',
    )
    estimator = make_pipeline(transformer, RandomForestRegressor(n_estimators=100,
                                                                n_jobs=1, random_state=seed))
    estimator.fit(frame.iloc[train], rows[METRIC].iloc[train])
    score = estimator.score(frame.iloc[evaluation], rows[METRIC].iloc[evaluation])
    result = permutation_importance(estimator, frame.iloc[evaluation], rows[METRIC].iloc[evaluation],
                                    scoring='r2', n_repeats=repeats, random_state=seed, n_jobs=1)
    expected = pd.DataFrame({
        'parameter': PARAMETERS, 'importance_mean': result.importances_mean,
        'importance_std': result.importances_std, 'baseline_score': score,
        'n_train': len(train), 'n_evaluation': len(evaluation),
        'n_configurations': len(np.unique(groups)),
    }).sort_values('importance_mean', ascending=False, kind='stable').reset_index(drop=True)
    return expected, train, evaluation


def test_recorded_parameter_importance(recorded_importance):
    loop, path, rows = recorded_importance
    expected, train, evaluation = _reference(rows)
    live = Log(uel_object=loop).experiment_parameter_importance(
        METRIC, parameters=PARAMETERS, random_state=7, n_repeats=3)
    saved_log = Log(file_path=str(path / 'results.csv'))
    saved_expected, _, _ = _reference(saved_log._captured_log)
    captured = saved_log.experiment_parameter_importance(
        METRIC, parameters=PARAMETERS, random_state=7, n_repeats=3)
    assert list(live.columns) == COLUMNS
    assert_frame_equal(live, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    assert_frame_equal(captured, saved_expected, check_exact=False, rtol=1e-12, atol=1e-12)
    assert (live['n_train'] == len(train)).all() and (live['n_evaluation'] == len(evaluation)).all()
    constants = live.filter(pl.col('parameter').is_in(['force_row_wise', 'random_state']))
    assert (constants.select('importance_mean', 'importance_std').to_numpy() == 0.0).all()
    zero_names = live.filter(pl.col('importance_mean') == 0.0)['parameter'].to_list()
    assert zero_names == [name for name in PARAMETERS if name in zero_names]
    assert np.isfinite(live[['importance_mean', 'importance_std', 'baseline_score']]).all().all()
    numeric_parameters = ['lookback', 'fee', 'n_estimators']
    numeric = Log(uel_object=loop).experiment_parameter_importance(
        METRIC, parameters=numeric_parameters, n_repeats=2)
    assert set(numeric['parameter']) == set(numeric_parameters)
    assert (numeric['n_configurations'] == 32).all()


def test_invalid_importance_inputs(recorded_importance):
    _, _, rows = recorded_importance
    log = _log(rows)
    numeric_objects = rows.copy()
    numeric_objects[['lookback', METRIC]] = numeric_objects[['lookback', METRIC]].astype(object)
    assert_frame_equal(
        _log(numeric_objects).experiment_parameter_importance(METRIC, parameters=PARAMETERS, n_repeats=2),
        log.experiment_parameter_importance(METRIC, parameters=PARAMETERS, n_repeats=2),
    )
    for kwargs in ({'parameters': []}, {'parameters': 'lookback'},
                   {'parameters': ['lookback', 'lookback']}, {'parameters': ['missing']},
                   {'parameters': [METRIC]}, {'parameters': [True]}):
        with pytest.raises(ValueError):
            log.experiment_parameter_importance(METRIC, **kwargs)
    for metric in ('missing', '', None):
        with pytest.raises(ValueError):
            log.experiment_parameter_importance(metric, parameters=PARAMETERS)
    for name, values in (('n_repeats', (0, 1, True, 2.5)),
                         ('random_state', (-1, True, 1.5, 2**32))):
        for value in values:
            with pytest.raises(ValueError):
                log.experiment_parameter_importance(METRIC, parameters=PARAMETERS, **{name: value})
    for invalid in (None, np.nan, np.inf, -np.inf, [2, 4], {'lookback': 2}, 1 + 2j):
        changed = rows.copy()
        changed['lookback'] = changed['lookback'].astype(object)
        changed.at[0, 'lookback'] = invalid
        with pytest.raises(ValueError):
            _log(changed).experiment_parameter_importance(METRIC, parameters=PARAMETERS)
    for invalid in (None, np.nan, np.inf, -np.inf, 'not a score', 1 + 2j):
        changed = rows.copy()
        changed[METRIC] = changed[METRIC].astype(object)
        changed.at[0, METRIC] = invalid
        with pytest.raises(ValueError):
            _log(changed).experiment_parameter_importance(METRIC, parameters=PARAMETERS)
    repeated = pd.concat([rows.iloc[:19]] * 4, ignore_index=True)
    with pytest.raises(ValueError, match='20 distinct'):
        _log(repeated).experiment_parameter_importance(METRIC, parameters=PARAMETERS)
    constant = rows.copy()
    constant[METRIC] = rows[METRIC].iloc[0]
    with pytest.raises(ValueError, match='Held-out metric must vary'):
        _log(constant).experiment_parameter_importance(METRIC, parameters=PARAMETERS)

    failed = rows.iloc[:1].copy()
    failed['strict_mode_error'] = 'Recorded error injection: this attempt failed'
    failed['lookback'] = None
    failed[METRIC] = np.nan
    with_failure = pd.concat([rows, failed], ignore_index=True)
    expected = log.experiment_parameter_importance(METRIC, parameters=PARAMETERS,
                                                  random_state=7, n_repeats=2)
    actual = _log(with_failure).experiment_parameter_importance(METRIC, parameters=PARAMETERS,
                                                               random_state=7, n_repeats=2)
    assert_frame_equal(actual, expected)


def test_importance_grouping_and_encoding(recorded_importance, monkeypatch):
    _, _, recorded = recorded_importance
    rows = pd.concat([recorded.loc[recorded['boosting_type'].eq('gbdt')],
                      recorded.loc[recorded['boosting_type'].eq('dart')].iloc[:1]], ignore_index=True)
    rows = pd.concat([rows, rows.iloc[:4], rows.iloc[-1:]], ignore_index=True)
    groups = rows.groupby(PARAMETERS, sort=False, observed=True).ngroup().to_numpy()
    for seed in range(100):
        train, evaluation = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                                   random_state=seed).split(rows, groups=groups))
        if 'dart' in set(rows.iloc[evaluation]['boosting_type']):
            break
    else:
        pytest.fail('Recorded rare category did not enter a seeded holdout')
    assert not set(groups[train]).intersection(groups[evaluation])
    assert set(rows.iloc[train]['boosting_type']) == {'gbdt'}
    expected, reference_train, reference_evaluation = _reference(rows, seed=seed, repeats=2)
    pipelines = []
    module = importlib.import_module('sklearn.pipeline')
    original = module.make_pipeline

    def capture_pipeline(*steps):
        pipeline = original(*steps)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(module, 'make_pipeline', capture_pipeline)
    actual = _log(rows).experiment_parameter_importance(METRIC, parameters=PARAMETERS,
                                                       random_state=seed, n_repeats=2)
    assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(train, reference_train)
    np.testing.assert_array_equal(evaluation, reference_evaluation)
    assert (actual['n_configurations'] == 33).all()
    encoder = pipelines[0].named_steps['columntransformer'].named_transformers_['categorical']
    assert encoder.categories_[0].tolist() == ['gbdt']
    estimator = pipelines[0].named_steps['randomforestregressor']
    assert estimator.n_jobs == 1 and estimator.n_estimators == 100 and estimator.random_state == seed


def test_importance_read_only(recorded_importance, monkeypatch):
    loop, path, _ = recorded_importance
    log = Log(uel_object=loop)
    original = log._captured_log.clone()
    artifacts = {file.relative_to(path): file.read_bytes() for file in path.rglob('*') if file.is_file()}
    search = copy.deepcopy(loop._search_strategy.get_state())

    def forbidden(*args, **kwargs):
        raise AssertionError('Recorded importance must not prepare data or replay a model')

    monkeypatch.setattr(MLManifest, 'prepare_data', forbidden)
    monkeypatch.setattr(MLManifest, 'run_model', forbidden)
    log.experiment_log = pl.DataFrame()
    first = log.experiment_parameter_importance(METRIC, parameters=PARAMETERS,
                                               random_state=7, n_repeats=2)
    first = first.with_columns(pl.lit(np.nan).alias('importance_mean'))
    second = log.experiment_parameter_importance(METRIC, parameters=PARAMETERS,
                                                random_state=7, n_repeats=2)
    assert np.isfinite(second['importance_mean']).all()
    assert_frame_equal(log._captured_log, original)
    assert {file.relative_to(path): file.read_bytes() for file in path.rglob('*') if file.is_file()} == artifacts
    assert loop._search_strategy.get_state() == search
