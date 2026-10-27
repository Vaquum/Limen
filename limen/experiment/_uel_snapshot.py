import hashlib
from collections.abc import Callable
from functools import wraps
from importlib import import_module, metadata
import inspect
import json
from pathlib import Path
import sys
import tempfile
from typing import Concatenate, ParamSpec, Protocol, TypeVar, cast

__all__: list[str] = []
_MAGIC = b'LIMEN-UEL\x00\x01'
_Loop = TypeVar('_Loop')
_P = ParamSpec('_P')
_Result = TypeVar('_Result')


def captured(func: Callable[Concatenate[_Loop, _P], _Result]) -> Callable[Concatenate[_Loop, _P], _Result]:
    @wraps(func)
    def run(host: _Loop, *args: _P.args, **kwargs: _P.kwargs) -> _Result:
        attributes = vars(host)
        attributes['_snapshot_ready'] = False
        name = inspect.signature(func).bind(host, *args, **kwargs).arguments['experiment_name']
        result = func(host, *args, **kwargs)
        attributes.update(_snapshot_name=name, _snapshot_ready=True)
        return result
    return run


class _Serializer(Protocol):
    def dumps(self, value: object, protocol: int) -> bytes: ...
    def loads(self, payload: bytes) -> object: ...


def _environment() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\x00' + path.read_bytes() + b'\x00')
    packages = {dist.metadata['Name'].lower().replace('_', '-'): dist.version for dist in metadata.distributions()}
    return {'python': list(sys.version_info[:3]), 'source_sha256': digest.hexdigest(), 'packages': packages}


def save(host: object, filepath: str | Path | None = None) -> Path:
    """Atomically save the complete captured experiment in a trusted local artifact."""
    attributes = vars(host)
    if not attributes.get('_snapshot_ready'):
        raise ValueError('Save requires a successfully returned experiment run')
    path = Path(filepath) if filepath is not None else Path(str(attributes['_snapshot_name']) + '.uel')
    serializer = cast(_Serializer, import_module('cloudpickle'))
    payload = serializer.dumps(host, protocol=5)
    header = json.dumps({'environment': _environment(), 'sha256': hashlib.sha256(payload).hexdigest()}, sort_keys=True).encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            _ = stream.write(_MAGIC + header + b'\n' + payload)
            stream.flush()
        except BaseException:
            temporary.unlink()
            raise
    try:
        _ = temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load(cls: type[_Loop], filepath: str | Path) -> _Loop:
    """Restore trusted captured state after compatibility and checksum validation."""
    with Path(filepath).open('rb') as stream:
        if stream.read(len(_MAGIC)) != _MAGIC:
            raise ValueError('Unknown experiment snapshot format')
        document = cast(object, json.loads(stream.readline()))
        if not isinstance(document, dict) or set(cast(dict[str, object], document)) != {'environment', 'sha256'}:
            raise ValueError('Invalid experiment snapshot header')
        header = cast(dict[str, object], document)
        environment = header['environment']
        current = _environment()
        if not isinstance(environment, dict):
            raise ValueError('Invalid experiment snapshot environment')
        saved = cast(dict[str, object], environment)
        if saved.get('python') != current['python'] or saved.get('source_sha256') != current['source_sha256']:
            raise ValueError('Experiment snapshot requires its matching Python and Limen source')
        packages = saved.get('packages')
        current_packages = cast(dict[str, str], current['packages'])
        if not isinstance(packages, dict) or any(current_packages.get(name) != version for name, version in cast(dict[str, object], packages).items()):
            raise ValueError('Experiment snapshot requires its matching dependency versions')
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != header['sha256']:
        raise ValueError('Experiment snapshot payload checksum mismatch')
    serializer = cast(_Serializer, import_module('cloudpickle'))
    value = serializer.loads(payload)
    if not isinstance(value, cls):
        raise TypeError('Experiment snapshot does not contain the requested experiment type')
    return value
