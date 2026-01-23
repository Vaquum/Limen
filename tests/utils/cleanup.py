"""
Test cleanup utilities for managing temporary files and handling signals.
"""

import atexit
import signal
import glob
import os
from pathlib import Path


def cleanup_csv_files():
    """
    Clean up any CSV files created in the project root during testing.
    This function is designed to be safe and only target the project root,
    not subdirectories like datasets/.
    """
    project_root = Path.cwd()

    csv_pattern = os.path.join(project_root, '*.csv')
    csv_files = glob.glob(csv_pattern)

    if csv_files:
        for csv_file in csv_files:

            try:
                os.remove(csv_file)
                print(f'  ✅ Deleted: {os.path.basename(csv_file)}')

            except (OSError, PermissionError) as e:
                print(f'  ❌ Failed to delete {os.path.basename(csv_file)}: {e}')


def signal_handler(signum, frame):
    """Handle interrupt signals and ensure cleanup."""
    cleanup_csv_files()
    exit(1)


def setup_cleanup_handlers():
    """Register cleanup handlers for multiple exit scenarios."""
    atexit.register(cleanup_csv_files)
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
