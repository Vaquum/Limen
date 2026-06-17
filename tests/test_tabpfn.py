import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import limen
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy

logger = logging.getLogger(__name__)


pytestmark = pytest.mark.tabpfn_model


def test_tabpfn():
    '''Test TabPFN SFD via MSQ path with RandomStrategy.'''

    if os.getenv('LIMEN_RUN_TABPFN_MODEL_TEST') != '1':
        pytest.skip('Real TabPFN model-download validation is split to #628')
    pytest.importorskip('tabpfn')

    tabpfn_sfds = [
        limen.sfd.foundational_sfd.tabpfn_binary,
    ]

    for sfd_module in tabpfn_sfds:

        with TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / 'experiment'
            domain = ParamDomain(sfd_module.params())
            strategy = RandomStrategy(domain, seed=42)

            uel = limen.UniversalExperimentLoop(
                sfd=sfd_module,
                search_strategy=strategy,
                experiment_dir=experiment_dir,
            )

            uel.run(
                experiment_name=str(experiment_dir / 'test'),
                n_permutations=2,
            )

        logger.info('%s: PASSED', sfd_module.__name__)


if __name__ == '__main__':
    test_tabpfn()
