import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from wheel_proof import run


def test_command_timeout_terminates_its_descendants(tmp_path):
    script = tmp_path / 'parent.py'
    script.write_text('''import subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path('child.pid').write_text(str(child.pid))
child.wait()
''')
    with pytest.raises(subprocess.TimeoutExpired):
        run([sys.executable, script], cwd=tmp_path, timeout=1)
    child = int((tmp_path / 'child.pid').read_text())
    result = subprocess.run(['ps', '-p', str(child), '-o', 'stat='], capture_output=True, text=True, check=False)
    assert result.returncode in (0, 1)
    state = result.stdout.strip()
    if state and not state.startswith('Z'):
        os.kill(child, signal.SIGKILL)
    assert not state or state.startswith('Z'), 'A timed-out command left its child running'
