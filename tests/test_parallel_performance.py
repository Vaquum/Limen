"""Deterministic marker/shutdown interleavings and failure diagnostics."""
from unittest.mock import Mock

import pytest

from tests.test_parallel_rounds import _wait_for


def test_marker_published_during_process_exit(tmp_path):
    marker = tmp_path / '1.failed'
    output = tmp_path / 'process.log'
    output.write_text('controlled failure')

    def exit_after_marker():
        marker.touch()
        return 1

    _wait_for(marker, Mock(poll=exit_after_marker), output)


@pytest.mark.parametrize('returncode', [None, 1])
def test_marker_failure_includes_process_output(tmp_path, returncode):
    output = tmp_path / 'process.log'
    output.write_text('initializer traceback')
    process = Mock()
    process.poll.return_value = returncode
    with pytest.raises(pytest.fail.Exception, match='initializer traceback'):
        _wait_for(tmp_path / '1.failed', process, output, timeout=0)
