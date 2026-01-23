"""
Test cleanup utilities for managing temporary files and handling signals.
"""

import atexit
import signal
from pathlib import Path
import sys
import contextlib
from typing import Any


def cleanup_csv_files():
    """
    Clean up any CSV files created in the project root during testing.
    This function is designed to be safe and only target the project root,
    not subdirectories like datasets/.
    """
    project_root = Path.cwd()

    csv_files = list(project_root.glob('*.csv'))

    if csv_files:
        for csv_file in csv_files:

            with contextlib.suppress(OSError, PermissionError):
                csv_file.unlink()



def signal_handler(_signum: Any, _frame: Any):
    """Handle interrupt signals and ensure cleanup."""
    cleanup_csv_files()
    sys.exit(1)


def setup_cleanup_handlers():
    """Register cleanup handlers for multiple exit scenarios."""
    atexit.register(cleanup_csv_files)
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
