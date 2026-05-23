import random
import time
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from limen.yaml.compiler import CompiledSFD


_COMPLEXITY_THRESHOLDS = [
    (100, 'low'),
    (1000, 'medium'),
    (10000, 'high'),
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
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _complexity_rating(total: int) -> str:

    for threshold, label in _COMPLEXITY_THRESHOLDS:
        if total <= threshold:
            return label
    return 'extreme'


def _total_permutations(params: dict[str, list[Any]]) -> int:

    result = 1
    for values in params.values():
        result *= len(values)
    return result


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
        column = [values[i % len(values)] for i in range(n)]
        rng.shuffle(column)
        columns[key] = column

    return [{key: columns[key][i] for key in params} for i in range(n)]


def _classify_error(exc: Exception) -> str:

    msg = str(exc)
    lower = msg.lower()
    if 'sample' in lower and ('0 ' in lower or 'empty' in lower):
        return f'Test data too small — increase n_rows in test_data_source. ({msg})'
    if 'class' in lower and ('one' in lower or '1' in lower or 'least' in lower):
        return f'Not enough class diversity in test data — try larger n_rows. ({msg})'
    if 'nan' in lower or 'inf' in lower:
        return f'Test data contains NaN or Inf values. ({msg})'
    return f'{type(exc).__name__}: {msg}'


def profile(compiled_sfd: 'CompiledSFD') -> ProfileResult:

    '''
    Profile a compiled YAML experiment.

    Computes static fields (permutation count, complexity) always. If
    test_data_source is configured, runs a randomised strength-1 covering
    array of permutations against test data and records timing.

    Args:
        compiled_sfd (CompiledSFD): Compiled SFD built from a validated yaml_dict

    Returns:
        ProfileResult: Static fields always populated; runtime fields populated
            only when test_data_source is present and at least one permutation
            completes successfully

    '''

    params: dict[str, list[Any]] = {
        k: list(v) for k, v in compiled_sfd.params().items()
    }
    cardinalities = {k: len(v) for k, v in params.items()}
    total = _total_permutations(params) if params else 0

    result = ProfileResult(
        total_permutations=total,
        param_cardinalities=cardinalities,
        complexity_rating=_complexity_rating(total),
    )

    manifest = compiled_sfd.manifest()

    if manifest.test_data_source_config is None:
        result.warnings.append(
            'test_data_source not defined — runtime sampling skipped. '
            'Add test_data_source to the manifest for timing estimates.'
        )
        return result

    try:
        raw_data = manifest.fetch_test_data()
    except (OSError, ValueError, RuntimeError, ConnectionError) as exc:
        result.errors.append(f'Failed to load test data: {type(exc).__name__}: {exc}')
        return result

    permutations = make_covering_array(params)
    result.sample_permutations_attempted = len(permutations)

    elapsed_times: list[float] = []

    for round_params in permutations:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data_dict = manifest.prepare_data(raw_data, round_params)
                t0 = time.perf_counter()
                manifest.run_model(data_dict, round_params)
                elapsed_times.append(time.perf_counter() - t0)
        except (ValueError, RuntimeError, MemoryError, TypeError, ArithmeticError) as exc:
            result.errors.append(_classify_error(exc))

    result.sample_permutations_completed = len(elapsed_times)

    if elapsed_times:
        result.sample_time_seconds_per_permutation = sum(elapsed_times) / len(elapsed_times)
    elif not result.errors:
        result.warnings.append('No permutations completed — timing unavailable.')

    if result.sample_permutations_completed < result.sample_permutations_attempted:
        failed = result.sample_permutations_attempted - result.sample_permutations_completed
        result.warnings.append(
            f'{failed} of {result.sample_permutations_attempted} sample '
            f'permutation(s) failed — timing estimate based on '
            f'{result.sample_permutations_completed} completed run(s).'
        )

    return result
