from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from limen.data.historical_data import _aggregate_spot_klines
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy
from limen.log import ConfusionNotApplicableError
from limen.log._permutation_confusion_metrics import confusion_mean_return_pct
from limen.log._permutation_confusion_metrics import _declared_output_modes
from limen.sfd.foundational_sfd import logreg_binary as logreg_sfd
from limen.sfd.foundational_sfd import ridge_regressor as ridge_sfd
from limen.yaml import CompiledSFD, build_search_strategy, parse
from tests.test_direction_sizing import YAML_SPEC
from tests.test_dynamic_barriers import FIXTURE
from tests.test_meta_allocation import _config as _allocation_config

HOURLY = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
QUARTER_HOUR = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'


def _foundational_run(sfd_module, *, bars=1024):
    params = sfd_module.params()
    uel = UniversalExperimentLoop(
        data=_aggregate_spot_klines(pl.read_parquet(HOURLY), 7200).tail(bars),
        sfd=sfd_module,
        search_strategy=RandomStrategy(ParamDomain(params), seed=42),
        test_mode=True,
    )
    with TemporaryDirectory() as tmpdir:
        uel.run(experiment_name=str(Path(tmpdir) / 'test'), n_permutations=1, post_processing=True)
    return uel


def _compiled_run(config, bars, *, count=1):
    with TemporaryDirectory() as tmpdir:
        uel = UniversalExperimentLoop(
            sfd=CompiledSFD(config), data=bars, experiment_dir=tmpdir,
            yaml_reference=config, search_strategy=build_search_strategy(config),
        )
        uel.run('recorded', n_permutations=count, prep_each_round=True,
                random_search=False, progress_bar=False, post_processing=True)
    return uel


def test_recorded_binary_confusion_internally_consistent() -> None:
    uel = _foundational_run(logreg_sfd)
    table = uel.experiment_confusion_metrics
    assert table is not None
    row = table.row(0, named=True)
    assert int(row['tp_count'] + row['fp_count']) == int(row['pred_pos_count'])
    assert int(row['pred_pos_count']) <= int(row['n_kept'])
    expected_precision = 100.0 * row['tp_count'] / row['pred_pos_count'] if row['pred_pos_count'] else np.nan
    if np.isnan(expected_precision):
        assert np.isnan(row['precision_pct'])
    else:
        assert row['precision_pct'] == pytest.approx(expected_precision)
    assert uel.experiment_backtest_results is not None
    assert uel.experiment_parameter_correlation is not None


def test_recorded_regressor_confusion_unavailable() -> None:
    uel = _foundational_run(ridge_sfd)
    assert uel.experiment_confusion_metrics is None
    assert uel.experiment_backtest_results is not None
    assert len(uel.experiment_backtest_results) > 0
    assert uel.experiment_parameter_correlation is not None
    assert uel.convergence_report is not None
    preds = np.asarray(uel.preds[0], dtype=float)
    assert preds.size > 0
    assert np.isfinite(preds).all()
    assert ((preds != 0) & (preds != 1)).any()


def test_recorded_direction_sizing_confusion_unavailable() -> None:
    config, errors = parse(YAML_SPEC)
    assert errors == []
    backtest = config['sfd']['manifest']['backtest']
    backtest['max_holding_seconds'] = 1800
    backtest['funding']['params']['rate'] = 0.0001
    bars = pl.read_parquet(QUARTER_HOUR).filter(pl.col('datetime').dt.date() < pl.date(2025, 1, 5))
    uel = _compiled_run(config, bars)
    assert uel.experiment_confusion_metrics is None
    assert uel.experiment_backtest_results is not None
    assert uel.convergence_report is not None
    preds = np.asarray(uel.preds[0], dtype=float)
    assert preds.size > 0
    assert np.isfinite(preds).all()
    assert (preds < 0).any()
    assert ((preds != 0) & (preds != 1)).any()


def test_target_exposure_declaration_overrides_binary_values() -> None:
    config = _allocation_config('logreg')
    manifest = CompiledSFD(config).manifest()
    assert manifest.backtest_config is not None
    assert manifest.backtest_config.prediction_mode == 'target_exposure'
    uel = _compiled_run(config, pl.read_parquet(FIXTURE, n_rows=480))
    assert uel.experiment_confusion_metrics is None
    assert uel.experiment_backtest_results is not None
    preds = np.asarray(uel.preds[0], dtype=float)
    assert preds.size > 0
    assert np.isfinite(preds).all()


def _caller_owned_run(sfd_module):
    manifest = sfd_module.manifest()
    sfd = SimpleNamespace(params=sfd_module.params, prep=manifest.prepare_data, model=manifest.run_model)
    uel = UniversalExperimentLoop(
        data=_aggregate_spot_klines(pl.read_parquet(HOURLY), 7200).tail(512),
        sfd=sfd,
        search_strategy=RandomStrategy(ParamDomain(sfd_module.params()), seed=42),
        test_mode=True,
    )
    assert uel.manifest is None
    with TemporaryDirectory() as tmpdir:
        uel.run(experiment_name=str(Path(tmpdir) / 'test'), n_permutations=1, post_processing=True)
    return uel


def test_caller_owned_fractional_sfd_confusion_unavailable() -> None:
    uel = _caller_owned_run(ridge_sfd)
    assert uel.experiment_confusion_metrics is None
    assert uel.experiment_backtest_results is not None
    preds = np.asarray(uel.preds[0], dtype=float)
    assert ((preds != 0) & (preds != 1)).any()


def test_caller_owned_binary_sfd_confusion_reported() -> None:
    uel = _caller_owned_run(logreg_sfd)
    assert uel.experiment_confusion_metrics is not None
    assert len(uel.experiment_confusion_metrics) == 1


def test_caller_owned_sizing_confusion_unavailable() -> None:
    config, errors = parse(YAML_SPEC)
    assert errors == []
    config['sfd']['manifest']['backtest']['max_holding_seconds'] = 1800
    config['sfd']['manifest']['backtest']['funding']['params']['rate'] = 0.0001
    manifest = CompiledSFD(config).manifest()
    grid = config['sfd']['params']
    sfd = SimpleNamespace(params=lambda: grid, prep=manifest.prepare_data, model=manifest.run_model)
    bars = pl.read_parquet(QUARTER_HOUR).filter(pl.col('datetime').dt.date() < pl.date(2025, 1, 5))
    uel = UniversalExperimentLoop(
        data=bars, sfd=sfd,
        search_strategy=RandomStrategy(ParamDomain(grid), seed=42),
        test_mode=True,
    )
    assert uel.manifest is None
    with TemporaryDirectory() as tmpdir:
        uel.run(experiment_name=str(Path(tmpdir) / 'test'), n_permutations=1, post_processing=True)
    assert uel.experiment_confusion_metrics is None
    assert _declared_output_modes(uel._log.manifest, uel._log._alignment, 0) == {'target_exposure'}
    assert uel.experiment_backtest_results is not None


def test_confusion_helper_rejects_fractional() -> None:
    opens = pd.Series([100.0, 101.0, 102.0, 103.0])
    changes = pd.Series([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match='requires binary predictions and actuals'):
        confusion_mean_return_pct(pd.Series([0.5, 0.0, 1.0, 0.0]), pd.Series([0, 0, 1, 1]), opens, changes)
    result = confusion_mean_return_pct(
        pd.Series([1.0, 0.0, 1.0, 0.0]), pd.Series([1.0, None, 1.0, 0.0]), opens, changes,
        execution_lag_bars=0)
    assert set(result) == {'tp_mean_return_pct', 'fp_mean_return_pct', 'tn_mean_return_pct', 'fn_mean_return_pct'}
    assert result['tn_mean_return_pct'] == pytest.approx(100.0 / 103.0)


def test_confusion_not_applicable_is_value_error() -> None:
    assert issubclass(ConfusionNotApplicableError, ValueError)
