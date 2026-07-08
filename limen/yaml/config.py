from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


_LIMEN_TOML_NAME = 'limen.toml'
STORE_RELATIVE = Path('manifests') / 'committed'


def find_project_root(start: Path) -> Path | None:

    '''
    Walk up the directory tree looking for limen.toml.

    Args:
        start (Path): Directory to start searching from

    Returns:
        Path | None: The directory containing limen.toml, or None if not found

    '''

    current = start.resolve()
    while True:
        if (current / _LIMEN_TOML_NAME).exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def read_limen_toml(project_root: Path) -> dict[str, Any]:

    '''
    Read limen.toml from the project root.

    Args:
        project_root (Path): Directory containing limen.toml

    Returns:
        dict: Parsed contents of limen.toml

    '''

    toml_path = project_root / _LIMEN_TOML_NAME
    if not toml_path.exists():
        raise FileNotFoundError(f"limen.toml not found at '{toml_path}'")

    return tomllib.loads(toml_path.read_text(encoding='utf-8'))


def get_store_path(start: Path) -> Path:

    '''
    Locate the manifest store by finding the project root.

    Args:
        start (Path): Directory to start searching from

    Returns:
        Path: Path to manifests/committed/

    Raises:
        FileNotFoundError: If no limen.toml is found in the directory tree

    '''

    root = find_project_root(start)
    if root is None:
        raise FileNotFoundError(
            'No limen.toml found. Run limen new <project-name> to create a project.'
        )
    return root / STORE_RELATIVE
