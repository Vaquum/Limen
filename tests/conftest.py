import os
import sys

if sys.platform == 'darwin':
    _ = os.environ.setdefault('OMP_NUM_THREADS', '1')
