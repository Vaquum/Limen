import logging
import math
import random
import re
import time
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

from typing_extensions import override

from limen.yaml.schema import COMPLEXITY_HIGH_MAX
from limen.yaml.schema import COMPLEXITY_LOW_MAX
from limen.yaml.schema import COMPLEXITY_MEDIUM_MAX

if TYPE_CHECKING:
    from limen.yaml.compiler import CompiledSFD


_COMPLEXITY_THRESHOLDS = [
    (COMPLEXITY_LOW_MAX, 'low'),
    (COMPLEXITY_MEDIUM_MAX, 'medium'),
    (COMPLEXITY_HIGH_MAX, 'high'),
]


@dataclass
class ProfileResult:

    '''Summary of a YAML experiment profile run.'''

    total_permutations: int
    param_cardinalities: dict[str, int]
    complexity_rating: str
    sample_permutations_attempted: int = 0
    sample_permutations_completed: int = 0
    sample_time_seconds_per_permutation: float | None = None
    warnings: list[str] = field(default_factory=list[str])
    errors: list[str] = field(default_factory=list[str])
    data_quality_warnings: list[str] = field(default_factory=list[str])


_MANIFEST_CORE_LOGGER = 'limen.experiment.manifest_core'


class _LogCapture(logging.Handler):

    def __init__(self) -> None:

        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:

        self.records.append(record.getMessage())


def _complexity_rating(total: int) -> str:

    for threshold, label in _COMPLEXITY_THRESHOLDS:
        if total <= threshold:
            return label
    return 'extreme'



def make_covering_array(params: dict[str, list[Any]], seed: int = 42) -> list[dict[str, Any]]:

    '''
    Build a randomised strength-1 covering array for the given parameter space.

    Every value of every parameter appears in at least one permutation.
    N = max cardinality across all params. Each parameter's column is filled
    with round-robin values then shuffled independently, removing systematic
    correlation between parameter assignments.

    Args:
        params (dict): Parameter name to list of values
        seed (int): Random seed for reproducibility

    Returns:
        list[dict]: N permutations, each a dict of param name to sampled value

    '''

    if not params:
        return []

    n = max(len(v) for v in params.values())
    rng = random.Random(seed)
    columns: dict[str, list[Any]] = {}

    for key, values in params.items():
        if not values:
            raise ValueError(f"Parameter '{key}' has an empty values list")
        column = [values[i % len(values)] for i in range(n)]
        rng.shuffle(column)
        columns[key] = column

    return [{key: columns[key][i] for key in params} for i in range(n)]


def _classify_error(exc: Exception) -> str:

    msg = str(exc)
    lower = msg.lower()
    if 'sample' in lower and ('0 ' in lower or 'empty' in lower):
        return f'Data too small for profiling — extend the date range in data_source. ({msg})'
    if 'class' in lower and ('one' in lower or 'least' in lower):
        return f'Not enough class diversity in test data — try larger n_rows. ({msg})'
    if 'nan' in lower or re.search(r'\binf\b', lower):
        return f'Test data contains NaN or Inf values. ({msg})'
    return f'{type(exc).__name__}: {msg}'


def profile(compiled_sfd: 'CompiledSFD') -> ProfileResult:

    '''
    Profile a compiled YAML experiment.

    Computes static fields (permutation count, complexity) always. Fetches
    data via the experiment's own data_source and runs a randomised strength-1
    covering array of permutations to record timing and surface data quality
    warnings.

    Args:
        compiled_sfd (CompiledSFD): Compiled SFD built from a validated yaml_dict

    Returns:
        ProfileResult: Static fields always populated; runtime fields populated
            when data loads successfully and at least one permutation completes

    '''

    params: dict[str, list[Any]] = {}
    cardinalities: dict[str, int] = {}
    for k, v in compiled_sfd.params().items():
        lst = list(v)
        params[k] = lst
        cardinalities[k] = len(lst)
    total = math.prod(cardinalities.values())

    result = ProfileResult(
        total_permutations=total,
        param_cardinalities=cardinalities,
        complexity_rating=_complexity_rating(total),
    )

    manifest = compiled_sfd.manifest()

    try:
        raw_data = manifest.fetch_data()
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f'Failed to load data: {type(exc).__name__}: {exc}')
        return result

    permutations = make_covering_array(params)
    result.sample_permutations_attempted = len(permutations)

    from limen.experiment.manifest_core import MLManifest  # local to avoid circular import

    elapsed_times: list[float] = []
    saved_strict_mode = False
    ml_manifest = manifest if isinstance(manifest, MLManifest) else None
    if ml_manifest is not None:
        saved_strict_mode = ml_manifest.strict_mode
        ml_manifest.strict_mode = False

    log_capture = _LogCapture()
    manifest_logger = logging.getLogger(_MANIFEST_CORE_LOGGER)
    manifest_logger.addHandler(log_capture)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for round_params in permutations:
                try:
                    t0 = time.perf_counter()
                    data_dict = manifest.prepare_data(raw_data, round_params)
                    _: Any = manifest.run_model(data_dict, round_params)
                    elapsed_times.append(time.perf_counter() - t0)
                except (ValueError, RuntimeError, MemoryError, TypeError, ArithmeticError) as exc:
                    result.errors.append(_classify_error(exc))
    finally:
        manifest_logger.removeHandler(log_capture)
        if ml_manifest is not None:
            ml_manifest.strict_mode = saved_strict_mode

    seen: set[str] = set()
    for msg in log_capture.records:
        if 'Unexpected nulls' in msg and msg not in seen:
            seen.add(msg)
            result.data_quality_warnings.append(msg)

    result.sample_permutations_completed = len(elapsed_times)

    if elapsed_times:
        result.sample_time_seconds_per_permutation = sum(elapsed_times) / len(elapsed_times)
    elif not result.errors:
        result.warnings.append('No permutations completed — timing unavailable.')

    if result.sample_permutations_completed < result.sample_permutations_attempted:
        failed = result.sample_permutations_attempted - result.sample_permutations_completed
        result.warnings.append(
            f"{failed} of {result.sample_permutations_attempted} sample permutation(s) failed — timing estimate based on {result.sample_permutations_completed} completed run(s)."
        )

    return result
