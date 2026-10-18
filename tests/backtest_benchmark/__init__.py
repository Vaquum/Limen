"""Optional, isolated cross-engine benchmarks on recorded market bars."""

THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'POLARS_MAX_THREADS',
           'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS', 'OMP_THREAD_LIMIT')
