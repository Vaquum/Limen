import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.linear_model import Ridge

from limen.experiment import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.indicators.mom import mom
from limen.indicators.roc import roc
from limen.sfd.foundational_sfd import random_binary as sfd_module
from tests.test_experiment_core_msq import _make_uel


@pytest.fixture(scope='module')
def recorded_bars():
    return pl.read_parquet(
        Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet', n_rows=1440,
    )


def _custom_loop(bars, space, *, declared_space=None, directory=None):
    calls = {'prep': [], 'model': [], 'feedback': []}

    def prep(data, round_params):
        feature, window = round_params['feature'], round_params['window']
        calls['prep'].append(feature)
        function = {'roc': roc, 'mom': mom}[feature] if isinstance(feature, str) else feature

        def prepare_split(frame):
            return function(frame, period=window).select(
                pl.col(f'{function.__name__}_{window}').alias('feature'),
                (pl.col('close').shift(-1) / pl.col('close') - 1).alias('next_return'),
            ).drop_nulls()

        return {'train': prepare_split(data[:960]), 'validation': prepare_split(data[960:])}

    def model(data, round_params):
        calls['model'].append(round_params['feature'])
        train, validation = data['train'], data['validation']
        fitted = Ridge(alpha=1.0).fit(train.select('feature'), train['next_return'])
        predictions = fitted.predict(validation.select('feature'))
        return {'validation_mse': float(np.mean((predictions - validation['next_return'].to_numpy()) ** 2)),
                'validation_rows': validation.height}

    def feedback(log, _msq):
        calls['feedback'].append(log.height)

    sfd = SimpleNamespace(__name__=__name__,
                          params=lambda: space if declared_space is None else declared_space,
                          prep=prep, model=model)
    loop = UniversalExperimentLoop(
        data=bars, sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)),
        experiment_dir=directory, feedback_interval=1, intra_callback=feedback,
    )
    return loop, calls


def test_callable_parameters_complete_like_scalar_names(recorded_bars, tmp_path):
    results = {}
    for name, features in (('callable', [roc, mom]), ('scalar', ['roc', 'mom'])):
        space = {'feature': features, 'window': [4, 12]}
        loop, calls = _custom_loop(recorded_bars, space)
        stem = tmp_path / name
        loop.run(str(stem), n_permutations=4, workers=1, progress_bar=False)
        log = loop.experiment_log
        expected_features = features * 2
        assert calls['prep'] == calls['model'] == expected_features
        assert calls['feedback'] == [2, 3, 4]
        assert log.height == 4
        assert log['window'].to_list() == [4, 4, 12, 12]
        assert log['_round_index'].to_list() == [0, 1, 2, 3]
        assert log['validation_rows'].to_list() == [475, 475, 467, 467]
        assert log['strict_mode_error'].is_null().all()
        assert loop.convergence_report['rounds'] == loop.convergence_report['observed_rounds'] == 4
        if name == 'callable':
            assert log.schema['feature'] == pl.Object
            for actual in (calls['prep'], calls['model'], log['feature'].to_list()):
                assert all(value is expected for value, expected in zip(actual, expected_features, strict=True))
        with stem.with_suffix('.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4
        assert [row['feature'] for row in rows] == [str(value) for value in expected_features]
        assert [row['id'] for row in rows] == log['id'].to_list()
        results[name] = log.select('window', '_round_index', 'validation_rows', 'validation_mse')
    assert_frame_equal(results['callable'], results['scalar'], check_exact=True)
    assert set(tmp_path.iterdir()) == {tmp_path / 'callable.csv', tmp_path / 'scalar.csv'}


def test_unused_callable_in_sfd_params_needs_no_checkpoint(recorded_bars, tmp_path):
    loop, calls = _custom_loop(recorded_bars, {'feature': ['roc'], 'window': [4]},
                               declared_space={'feature': [roc], 'window': [4]})
    loop.run(str(tmp_path / 'unused'), n_permutations=1, workers=1, progress_bar=False)
    assert calls['prep'] == calls['model'] == ['roc']
    assert loop.experiment_log.height == loop.convergence_report['rounds'] == 1
    assert (tmp_path / 'unused.csv').is_file()


def test_manifest_sfd_callable_parameter_needs_no_checkpoint(tmp_path, monkeypatch):
    space = {**sfd_module.params(), 'aux_fn': [roc]}
    monkeypatch.setattr(sfd_module, 'params', lambda: space)
    loop, _, _ = _make_uel(strategy_cls=GridStrategy)
    loop.run(str(tmp_path / 'manifest'), n_permutations=1, workers=1, progress_bar=False)
    assert loop.manifest is not None
    assert loop.experiment_log.height == loop.convergence_report['rounds'] == 1
    assert loop.experiment_log['aux_fn'][0] is roc
    assert loop.experiment_log['strict_mode_error'].is_null().all()
    assert (tmp_path / 'manifest.csv').is_file()


def test_persisted_callable_sfd_params_refuse_before_writes(recorded_bars, tmp_path):
    directory = tmp_path / 'persisted'
    loop, calls = _custom_loop(recorded_bars, {'feature': [roc], 'window': [4]}, directory=directory)
    with pytest.raises(TypeError, match='not JSON serializable'):
        loop.run(str(tmp_path / 'persisted'), n_permutations=1, workers=1, progress_bar=False)
    assert calls == {'prep': [], 'model': [], 'feedback': []}
    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []
