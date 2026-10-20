from collections.abc import Callable, Mapping
import copy
from hashlib import sha256
from typing import cast

from limen.experiment._holdout_state import fitted_witness
from limen.experiment.manifest_core import Manifest

__all__ = ['Components', 'bind_components', 'configure_manifest']

Components = Callable[[Manifest], None]


def bind_components(reference: Mapping[str, object], components: Components | None, *, record: bool = False) -> dict[str, object]:
    sfd = reference.get('sfd')
    if components is None:
        if isinstance(sfd, Mapping) and 'components' in sfd:
            raise ValueError('Recorded trusted components require an explicit matching components configurator')
        return dict(reference)
    if not isinstance(sfd, Mapping):
        raise ValueError('Trusted components require an sfd mapping')
    declared = sfd.get('components')
    if not callable(components):
        raise TypeError('components must be a callable Manifest configurator')
    name = f'{getattr(components, "__module__", type(components).__module__)}.{getattr(components, "__qualname__", type(components).__qualname__)}'
    identity = {'name': name, 'sha256': sha256(fitted_witness(components)).hexdigest()}
    if declared != identity and not (record and 'components' not in sfd):
        raise ValueError('Trusted components configurator differs from the recorded declaration')
    result = copy.deepcopy(dict(reference))
    cast(dict[str, object], result['sfd'])['components'] = identity
    return result


def configure_manifest(manifest: Manifest, reference: Mapping[str, object], components: Components | None) -> Manifest:
    _ = bind_components(reference, components)
    if components is not None and components(manifest) is not None:
        raise ValueError('components configurator must modify the Manifest and return None')
    return manifest
