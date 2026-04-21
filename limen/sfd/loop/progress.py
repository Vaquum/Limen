'''Progress reporting via UEL's intra_callback hook.

UEL's intra_callback fires during feedback triggers — every feedback_interval
rounds. It receives (log, msq) where log is the polars experiment log and msq
is the mutable search queue. The return value is ignored and exceptions are
caught (see limen/experiment/feedback_controller.py:240-263), so a callback
that only writes to disk is safe.

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands.
'''

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def make_progress_callback(progress_file: Path,
                           total: int) -> Callable[[Any, Any], None]:

    '''
    Create an intra_callback that writes progress to a JSON file.

    The returned callable matches the intra_callback signature expected by
    UniversalExperimentLoop's feedback controller. Each invocation overwrites
    progress_file with the current state.

    Args:
        progress_file (Path): Destination path for the progress JSON file
        total (int): Total number of permutations the experiment will run

    Returns:
        Callable: Function with signature (log, msq) -> None that writes
            {'completed', 'total', 'percent', 'updated_at'} to progress_file

    '''

    progress_file = Path(progress_file)

    def _callback(log: Any, _msq: Any) -> None:

        try:
            completed = len(log) if log is not None else 0
            payload = {
                'completed': completed,
                'total': total,
                'percent': round(100 * completed / total, 2) if total > 0 else 0.0,
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            tmp = progress_file.with_suffix('.tmp')
            tmp.write_text(json.dumps(payload))
            tmp.replace(progress_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning('Progress callback failed: %s', exc)

    return _callback


__all__ = ['make_progress_callback']
