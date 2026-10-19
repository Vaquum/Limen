import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import cast

__all__: list[str] = []

_THRESHOLDS = {'min_deflated_sharpe_probability', 'max_pbo'}
_COUNTS = (2, 4, 6, 8)
_MINIMUM_BLOCK_BARS = 2


def pbo_blocks(settings: Mapping[str, object] | None) -> int:
    value = 2 if settings is None else settings.get('pbo_blocks', 2)
    if type(value) is not int or value not in _COUNTS:
        raise ValueError('acceptance.pbo_blocks must be a literal even integer in {2, 4, 6, 8}')
    return value


def read_acceptance(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError('acceptance must be a mapping of probability thresholds and pbo_blocks')
    declaration = cast(Mapping[object, object], value)
    if not declaration or any(key not in _THRESHOLDS | {'pbo_blocks'} for key in declaration):
        raise ValueError('acceptance allows only min_deflated_sharpe_probability, max_pbo and pbo_blocks')
    result: dict[str, float] = {}
    for key, threshold in declaration.items():
        if key == 'pbo_blocks':
            result['pbo_blocks'] = pbo_blocks({'pbo_blocks': threshold})
        else:
            if isinstance(threshold, bool) or not isinstance(threshold, Real):
                raise ValueError(f'acceptance.{key} must be a finite literal probability in [0, 1]')
            try:
                probability = float(threshold)
            except OverflowError as exc:
                raise ValueError(f'acceptance.{key} must be a finite literal probability in [0, 1]') from exc
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f'acceptance.{key} must be a finite literal probability in [0, 1]')
            result[str(key)] = probability
    return result


def _yaml_acceptance(reference: Mapping[str, object]) -> Mapping[str, object]:
    value: object = reference.get('sfd', {})
    for key in ('manifest', 'acceptance'):
        if not isinstance(value, Mapping):
            raise ValueError('PBO count requires valid YAML acceptance provenance')
        value = cast(Mapping[str, object], value).get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError('PBO count requires valid YAML acceptance provenance')
    return cast(Mapping[str, object], value)


def saved_pbo_blocks(metadata: Mapping[str, object]) -> int:
    count = pbo_blocks(metadata)
    reference = metadata.get('yaml_reference')
    if isinstance(reference, Mapping):
        settings = _yaml_acceptance(cast(Mapping[str, object], reference))
        declared = pbo_blocks(settings)
        if 'pbo_blocks' in metadata and 'pbo_blocks' in settings and count != declared:
            raise ValueError('Saved pbo_blocks conflicts with saved YAML acceptance')
    return count


def _metadata(directory: Path | None) -> Mapping[str, object]:
    path = None if directory is None else directory / 'metadata.json'
    if path is None or not path.exists():
        return {}
    value: object = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError('PBO count requires valid metadata')
    return cast(Mapping[str, object], value)


def validate_pbo_resume(settings: Mapping[str, object] | None, metadata: Mapping[str, object]) -> None:
    if pbo_blocks(settings) != saved_pbo_blocks(metadata):
        raise ValueError('Cannot resume with different acceptance.pbo_blocks')


def prepare_pbo_settings(settings: Mapping[str, object] | None, *, reference: Mapping[str, object] | None,
                         compiled_reference: Mapping[str, object] | None, directory: Path | None,
                         has_search: bool, resume: bool) -> dict[str, object]:
    count = pbo_blocks(settings)
    declarations = [settings or {}, *(_yaml_acceptance(ref) for ref in (reference, compiled_reference) if ref is not None)]
    if any(pbo_blocks(value) != count for value in declarations):
        raise ValueError('Conflicting live, compiled YAML and yaml_reference acceptance.pbo_blocks')
    explicit = any('pbo_blocks' in value for value in declarations)
    if explicit and (not has_search or directory is None):
        raise ValueError('Explicit acceptance.pbo_blocks requires search_strategy and experiment_dir')
    if resume:
        validate_pbo_resume(settings, _metadata(directory))
    return {**(settings or {}), **({'pbo_blocks': count} if explicit else {})}


def report_settings(directory: Path, acceptance: Mapping[str, float] | None) -> tuple[dict[str, float], int]:
    settings = read_acceptance(acceptance) if acceptance is not None else {}
    saved = saved_pbo_blocks(_metadata(directory))
    if 'pbo_blocks' in settings and pbo_blocks(settings) != saved:
        raise ValueError('Requested acceptance.pbo_blocks conflicts with the saved count')
    return settings, saved


def partition_geometry(n_bars: int | None, count: int) -> dict[str, object]:
    combinations = math.comb(count, count // 2)
    valid = n_bars is not None and n_bars % count == 0 and n_bars // count >= _MINIMUM_BLOCK_BARS
    return {'pbo_combinations': combinations, 'pbo_resolution': 1 / combinations,
            'pbo_block_bars': n_bars // count if valid and n_bars is not None else None}


def require_partition(n_bars: int, count: int) -> None:
    if partition_geometry(n_bars, count)['pbo_block_bars'] is None:
        supported = [value for value in _COUNTS if n_bars % value == 0 and n_bars // value >= _MINIMUM_BLOCK_BARS]
        raise ValueError(f'PBO requires equal blocks with at least two bars: n_bars={n_bars}, n_blocks={count}; supported counts: {supported or "none"}')
