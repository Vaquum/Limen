from importlib.util import find_spec
from pathlib import Path

_spec = find_spec('limen.yaml')
if _spec is None or _spec.origin is None:
    raise ImportError(
        "Cannot locate limen.yaml package resources — is limen installed correctly?"
    )
TEMPLATES_DIR = Path(_spec.origin).parent / 'templates'
