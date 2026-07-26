"""Docstring conventions gate: the three mechanizable rules fire and pass."""
from __future__ import annotations

import importlib
import types


def _load() -> types.ModuleType:
    # governance/ is on sys.path via governance/tests/conftest.py.
    return importlib.import_module('check_docstrings')


def test_forbidden_title_verb_is_flagged() -> None:
    mod = _load()
    assert mod.check_docstring('Calculate the average.') != []
    assert mod.check_docstring('Compute the average.') == []


def test_default_in_description_is_flagged() -> None:
    mod = _load()
    body = 'Compute x.\n\nArgs:\n    p (int): periods (default: 14)'
    assert mod.check_docstring(body) != []
    assert mod.check_docstring('Compute x.\n\nArgs:\n    p (int): periods') == []


def test_note_casing_is_flagged() -> None:
    mod = _load()
    assert mod.check_docstring('Compute x.\n\nNote: be careful.') != []
    assert mod.check_docstring('Compute x.\n\nNOTE: be careful.') == []


def test_find_violations_locates_the_function() -> None:
    mod = _load()
    src = 'def f():\n    """Calculate it."""\n    return 1\n'
    found = mod.find_violations(src)
    assert len(found) == 1
    assert found[0][1] == 'f'
