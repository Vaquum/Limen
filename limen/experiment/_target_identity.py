from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import cast

from limen.targets._perturbation import NOISE_KEYS

__all__: list[str] = []


@contextmanager
def bind(manifest: object, params: dict[str, object], identity: object) -> Iterator[None]:
    config: object = getattr(manifest, 'target_class_config', None)
    component: object = getattr(config, 'target_class', None)
    settings: object = getattr(config, 'transform_params', None)
    active = (identity is not None and getattr(component, 'requires_permutation_id', False) is True
              and isinstance(settings, Mapping) and bool(set(NOISE_KEYS) & cast(Mapping[str, object], settings).keys()))
    if active:
        if not isinstance(identity, str) or not identity or '_permutation_id' in params:
            raise ValueError('Target permutation identity is a reserved framework operand')
        params['_permutation_id'] = identity
    try:
        yield
    finally:
        if active:
            del params['_permutation_id']
