from pathlib import Path
from tempfile import TemporaryDirectory

import math

from limen.backtest.backtest_snapshot import BACKTEST_SNAPSHOT_COLUMNS
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.sfd.foundational_sfd import logreg_binary as logreg_binary_sfd
from limen.sfd.foundational_sfd import random_binary as random_binary_sfd
from limen.sfd.foundational_sfd import xgboost_regressor as xgboost_sfd
from limen.experiment.param_search import RandomStrategy


def _run_uel(sfd_module=random_binary_sfd,
             n_permutations: int = 3) -> UniversalExperimentLoop:

    params = sfd_module.params()
    domain = ParamDomain(params)
    strategy = RandomStrategy(domain, seed=42)

    uel = UniversalExperimentLoop(
        sfd=sfd_module,
        search_strategy=strategy,
        test_mode=True,
    )

    with TemporaryDirectory() as tmpdir:
        uel.run(
            experiment_name=str(Path(tmpdir) / 'test'),
            n_permutations=n_permutations,
            post_processing=True,
        )

    return uel


def test_inline_and_post_experiment_metrics() -> None:

    uel = _run_uel()
    log_cols = uel.experiment_log.columns

    assert uel.experiment_log.height > 0

    for col in ['confusion_tp', 'confusion_fp', 'confusion_tn', 'confusion_fn',
                'confusion_precision', 'confusion_recall']:
        assert col in log_cols, f"Missing inline confusion column: {col}"
        assert uel.experiment_log[col].null_count() == 0, f"Null values in {col}"

    for col in ['confusion_tp_mean_return_pct', 'confusion_fp_mean_return_pct',
                'confusion_tn_mean_return_pct', 'confusion_fn_mean_return_pct']:
        assert col in log_cols, f"Missing inline confusion return column: {col}"

    for metric_col in BACKTEST_SNAPSHOT_COLUMNS:
        col = f'backtest_{metric_col}'
        assert col in log_cols, f"Missing inline backtest column: {col}"
        assert uel.experiment_log[col].null_count() == 0, f"Null values in {col}"

    assert uel.experiment_confusion_metrics is not None
    assert len(uel.experiment_confusion_metrics) > 0
    for col in ['tp_mean_return_pct', 'fp_mean_return_pct', 'tn_mean_return_pct', 'fn_mean_return_pct']:
        assert col in uel.experiment_confusion_metrics.columns, f"Missing confusion metric column: {col}"
    assert uel.experiment_backtest_results is not None
    assert len(uel.experiment_backtest_results) > 0
    assert uel.experiment_backtest_results.columns.tolist() == BACKTEST_SNAPSHOT_COLUMNS

    assert uel.experiment_log['backtest_trade_pnl_net_bps_p50'].to_list() == \
        uel.experiment_backtest_results['trade_pnl_net_bps_p50'].tolist()


def test_logreg_inline_and_post_confusion_metrics_match() -> None:

    uel = _run_uel(sfd_module=logreg_binary_sfd, n_permutations=1)

    for inline_col, post_col in [
        ('confusion_tp_mean_return_pct', 'tp_mean_return_pct'),
        ('confusion_fp_mean_return_pct', 'fp_mean_return_pct'),
        ('confusion_tn_mean_return_pct', 'tn_mean_return_pct'),
        ('confusion_fn_mean_return_pct', 'fn_mean_return_pct'),
    ]:
        for inline_value, post_value in zip(
            uel.experiment_log[inline_col].to_list(),
            uel.experiment_confusion_metrics[post_col].tolist(),
            strict=True,
        ):
            if math.isnan(inline_value) and math.isnan(post_value):
                continue

            assert inline_value == post_value


def test_xgboost_inline_and_post_backtest_metrics_match() -> None:

    uel = _run_uel(sfd_module=xgboost_sfd, n_permutations=1)

    assert uel.experiment_log['backtest_trade_pnl_net_bps_p50'].to_list() == \
        uel.experiment_backtest_results['trade_pnl_net_bps_p50'].tolist()
