from pathlib import Path
from tempfile import TemporaryDirectory

from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.sfd.foundational_sfd import random_binary as sfd_module
from tests.stubs.stubs import StubStrategy


def _run_uel(n_permutations: int = 3) -> UniversalExperimentLoop:

    params = sfd_module.params()
    domain = ParamDomain(params)
    strategy = StubStrategy(domain)

    uel = UniversalExperimentLoop(
        sfd=sfd_module,
        search_strategy=strategy,
    )

    with TemporaryDirectory() as tmpdir:
        uel.run(
            experiment_name=str(Path(tmpdir) / 'test'),
            n_permutations=n_permutations,
        )

    return uel


def test_inline_and_post_experiment_metrics() -> None:

    uel = _run_uel()
    log_cols = uel.experiment_log.columns

    # inline confusion columns present in experiment_log
    for col in ['confusion_tp', 'confusion_fp', 'confusion_tn', 'confusion_fn',
                'confusion_precision', 'confusion_recall']:
        assert col in log_cols, f"Missing inline confusion column: {col}"

    # inline backtest columns present in experiment_log
    for col in ['backtest_trade_win_rate_pct', 'backtest_max_drawdown_pct',
                'backtest_total_return_net_pct', 'backtest_sharpe_per_bar']:
        assert col in log_cols, f"Missing inline backtest column: {col}"

    # post-experiment metrics still computed
    assert uel.experiment_confusion_metrics is not None
    assert len(uel.experiment_confusion_metrics) > 0
    assert uel.experiment_backtest_results is not None
    assert len(uel.experiment_backtest_results) > 0
