import os
import sys

import pytest

_ = os.environ.setdefault('POLARS_MAX_THREADS', '1')

if sys.platform == 'darwin':
    _ = os.environ.setdefault('OMP_NUM_THREADS', '1')


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node) -> None:
    node.workerinput['limen_longest_holdout_first'] = node.config.getoption('dist', 'no') == 'load' and node.config.getoption('maxschedchunk', 0) == 1


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if getattr(config, 'workerinput', {}).get('limen_longest_holdout_first', False):
        items.sort(key=lambda item: item.nodeid != 'tests/test_holdout_assessment.py::test_frozen_dependencies_and_invalid_state')
