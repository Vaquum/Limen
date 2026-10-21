from __future__ import annotations

from collections import deque
from functools import lru_cache
from hashlib import sha256
from importlib.abc import InspectLoader
import marshal
import pickle
import sys
from types import CodeType, EllipsisType, FunctionType
from typing import cast
from typing_extensions import Buffer

__all__ = ['code_aliases', 'code_cache_info', 'code_digest', 'function_module', 'native_codes', 'reader_names', 'reader_types']


_Constant = type[StopIteration] | EllipsisType | bool | int | float | complex | str | bytes | CodeType | Buffer | tuple['_Constant', ...] | frozenset['_Constant'] | None


def code_aliases(code: CodeType, fields: tuple[str, ...]) -> tuple[int, ...]:
    pending, seen, aliases = [cast(object, code)], dict[int, tuple[int, object]](), list[int]()
    while pending:
        repeated = id(item := pending.pop()) in seen
        aliases.append(seen.setdefault(id(item), (len(seen), item))[0])
        if not repeated:
            pending.extend(tuple(getattr(item, name, None) for name in fields) if isinstance(item, CodeType) else cast(tuple[object, ...], item) if isinstance(item, tuple) else sorted(cast(frozenset[object], item), key=lambda constant: marshal.dumps(cast(_Constant, constant), 2)) if isinstance(item, frozenset) else ())
    return tuple(aliases)


@lru_cache(maxsize=2048)
def _code_hash(code: CodeType, _identity: int) -> str:
    return sha256(marshal.dumps(code, 2) + pickle.dumps(code_aliases(code, ('co_code', 'co_consts', 'co_names', 'co_varnames', 'co_freevars', 'co_cellvars', 'co_filename', 'co_name', 'co_qualname', 'co_linetable', 'co_exceptiontable')), protocol=5)).hexdigest()


def code_digest(code: CodeType) -> str:
    return _code_hash(code, id(code))


@lru_cache(maxsize=256)
def native_codes(name: str) -> frozenset[CodeType]:
    loader = getattr(sys.modules[name], '__loader__', None)
    code = loader.get_code(name) if isinstance(loader, InspectLoader) else None
    pending, codes = [code] if code is not None else [], set[CodeType]()
    while pending:
        current = pending.pop()
        codes.add(current)
        pending.extend(part for part in current.co_consts if isinstance(part, CodeType))
    return frozenset(codes)


def function_module(value: FunctionType, libraries: tuple[str, ...]) -> str:
    name = value.__globals__.get('__name__')
    module = sys.modules.get(name) if isinstance(name, str) else None
    if module is None or vars(module) is not value.__globals__:
        return ''
    if (module.__name__.split('.')[0] in sys.stdlib_module_names or module.__name__.split('.')[0] in libraries) and value.__code__ not in native_codes(module.__name__):
        return ''
    return module.__name__


def code_cache_info() -> dict[str, int | None]:
    return _code_hash.cache_info()._asdict()


reader_types: tuple[type[object], ...] = (list, tuple, dict, deque)


def reader_names(value: object, key: str | int) -> tuple[str, ...]:
    size = list[object].__len__(cast(list[object], value)) if isinstance(value, list) else tuple[object, ...].__len__(cast(tuple[object, ...], value)) if isinstance(value, tuple) else deque[object].__len__(cast(deque[object], value)) if isinstance(value, deque) else None
    return (str(key), str(key - size)) if isinstance(key, int) and size is not None else (str(key),)
