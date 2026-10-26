from collections.abc import Collection, Mapping, Sequence
from string import Formatter
from typing import cast

from limen.experiment._factorized_contract import declared
from limen.yaml._parameter_refs import json_bindings, lookup_references, native_bindings
from limen.yaml.resolver import is_resolvable, resolve

__all__: list[str] = []


def declared_keys(component: object) -> set[str] | None:
    contract = declared(component)
    if contract is None or contract.validation_keys is None or contract.consumes is None:
        return None
    return {*contract.parameter_keys, *contract.validation_keys, *(name.split('.')[0] for name in contract.consumes)}


def _format_keys(value: object, keys: Collection[str]) -> set[str]:
    if isinstance(value, str):
        if '{' not in value or '}' not in value:
            return set()
        try:
            fields = tuple(Formatter().parse(value))
        except ValueError:
            # Unsupported formatting cannot prove that an axis is unused.
            return set(keys)
        found: set[str] = set()
        for _, field, spec, conversion in fields:
            if field is not None:
                name = field.split('.', 1)[0].split('[', 1)[0]
                if not name or name.isdecimal() or conversion not in (None, 's', 'r', 'a'):
                    return set(keys)
                found.add(name)
            found.update(_format_keys(spec, keys))
        return found
    if isinstance(value, Mapping):
        value = tuple(cast(Mapping[object, object], value).values())
    if isinstance(value, (list, tuple)):
        return set[str]().union(*(_format_keys(item, keys) for item in cast(Sequence[object], value)))
    return set()


def _opaque_json(value: object) -> bool:
    if callable(value):
        return True
    if isinstance(value, Mapping):
        value = tuple(cast(Mapping[object, object], value).values())
    return any(_opaque_json(item) for item in cast(Sequence[object], value)) if isinstance(value, (list, tuple)) else False


def used_references(manifest: Mapping[str, object], keys: Collection[str]) -> set[str]:
    found = set(lookup_references(manifest, keys))
    for path, values in json_bindings(manifest):
        if _opaque_json(values) or (path == 'sfd.manifest.allocation.params' and isinstance(values, Mapping) and any(isinstance(item, str) and item.startswith('limen.') for item in cast(Mapping[object, object], values).values())):
            return set(keys)
    for _, value, bare in native_bindings(manifest):
        if bare:
            if not isinstance(value, str):
                continue
            if value in keys:
                found.add(value)
                continue
            if value.startswith('_'):
                continue
        found.update(_format_keys(value, keys))
    for name, field in (('data_dict_extension', 'func'), ('allocation', 'method')):
        block = manifest.get(name)
        if isinstance(block, Mapping):
            path = cast(Mapping[str, object], block).get(field)
            if not isinstance(path, str) or not is_resolvable(path):
                return set(keys)
            dependencies = declared_keys(resolve(path))
            if dependencies is None:
                return set(keys)
            found.update(dependencies)
    return found
