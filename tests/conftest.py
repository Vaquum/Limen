import os
import sys

_ = os.environ.setdefault('POLARS_MAX_THREADS', '1')

if sys.platform == 'darwin':
    _ = os.environ.setdefault('OMP_NUM_THREADS', '1')
