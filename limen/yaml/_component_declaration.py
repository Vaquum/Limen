from collections.abc import Callable, Mapping
import copy
from dataclasses import fields
from hashlib import sha256
from typing import cast

from limen.experiment._holdout_calls import call_functions
from limen.experiment._holdout_state import fitted_witness
from limen.experiment._holdout_spec import digest, normalized
from limen.experiment.manifest_core import Manifest

__all__ = ['Components', 'bind_components', 'configure_manifest']
Components = Callable[[Manifest], None]


class _Pending(dict[str, object]):
    __slots__ = ()


def _identity(components: Components) -> dict[str, str]:
    bindings = tuple((function, {name: function.__globals__[name] for name in function.__code__.co_names if name in function.__globals__}) for function in call_functions(components))
    return {'name': f'{getattr(components, "__module__", type(components).__module__)}.{getattr(components, "__qualname__", type(components).__qualname__)}', 'sha256': sha256(fitted_witness((components, bindings))).hexdigest()}


def bind_components(reference: Mapping[str, object], components: Components | None, *, record: bool = False) -> dict[str, object]:
    sfd = reference.get('sfd')
    if components is None:
        if isinstance(sfd, Mapping) and 'components' in sfd:
            raise ValueError('Recorded trusted components require an explicit matching components configurator')
        return dict(reference)
    if not isinstance(sfd, Mapping):
        raise ValueError('Trusted components require an sfd mapping')
    if not callable(components):
        raise TypeError('components must be a callable Manifest configurator')
    declared, identity = cast(Mapping[str, object], sfd).get('components'), _identity(components)
    if declared is None and record and 'components' not in sfd:
        marker: dict[str, object] = _Pending(identity)
    elif isinstance(declared, Mapping) and all(cast(Mapping[str, object], declared).get(key) == value for key, value in identity.items()) and (type(cast(object, declared)) is _Pending or isinstance(cast(Mapping[str, object], declared).get('pipeline_sha256'), str)):
        marker = copy.deepcopy(cast(dict[str, object], declared))
    else:
        raise ValueError('Trusted components configurator differs from the recorded declaration')
    result = copy.deepcopy(dict(reference))
    cast(dict[str, object], result['sfd'])['components'] = marker
    return result


def configure_manifest(manifest: Manifest, reference: Mapping[str, object], components: Components | None) -> Manifest:
    _ = bind_components(reference, components)
    if components is not None and components(manifest) is not None:
        raise ValueError('components configurator must modify the Manifest and return None')
    if components is not None:
        sfd = cast(dict[str, object], reference['sfd'])
        marker = cast(dict[str, object], sfd['components'])
        pipeline = digest({field.name: normalized(getattr(manifest, field.name), declaration=field.name not in ('data_source_config', 'test_data_source_config')) for field in fields(manifest)})
        if type(marker) is _Pending:
            sfd['components'] = {**marker, 'pipeline_sha256': pipeline}
        elif marker['pipeline_sha256'] != pipeline:
            raise ValueError('Trusted components configurator differs from the recorded declaration')
    return manifest
