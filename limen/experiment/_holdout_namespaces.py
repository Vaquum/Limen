import dis
from functools import lru_cache
from types import CodeType

from limen.experiment._holdout_code import code_aliases, reader_names, reader_types

__all__ = ['code_aliases', 'namespace_calls', 'reader_names', 'reader_types', 'reflective_reads']
_CALLS = frozenset({'CALL', 'CALL_FUNCTION', 'CALL_METHOD'})


@lru_cache(maxsize=2048)
def namespace_calls(code: CodeType) -> tuple[tuple[str, str | None], ...]:
    instructions = tuple(part for part in dis.get_instructions(code) if part.opname != 'PRECALL')
    found: list[tuple[str, str | None]] = []
    aliases: dict[str, set[str]] = {}
    for index, part in enumerate(instructions[:-1]):
        if part.opname not in ('LOAD_GLOBAL', 'LOAD_NAME', 'LOAD_ATTR', 'LOAD_METHOD', 'LOAD_CONST', 'LOAD_FAST', 'LOAD_DEREF') or not isinstance(part.argval, (str, int)):
            continue
        names = aliases.get(str(part.argval), {str(part.argval)}) if part.opname in ('LOAD_FAST', 'LOAD_DEREF') else {str(part.argval)}
        call_index = index + 2 if instructions[index + 1].opname == 'BINARY_SUBSCR' and index + 2 < len(instructions) else index + 1
        call = instructions[call_index]
        if call.opname in ('STORE_FAST', 'STORE_DEREF') and (part.opname != 'LOAD_CONST' or call_index > index + 1):
            aliases.setdefault(str(call.argval), set()).update(names)
        lookup_start: int | None = call_index + 1 if call.opname in _CALLS and call.arg == 0 else None
        if lookup_start is None and part.opname == 'LOAD_CONST':
            for position in range(index + 1, len(instructions) - 1):
                getter, invocation = instructions[position:position + 2]
                if getter.opname in _CALLS and getter.arg in (1, 2, 3):
                    if invocation.opname in ('STORE_FAST', 'STORE_DEREF'):
                        aliases.setdefault(str(invocation.argval), set()).update(names)
                    elif invocation.opname in _CALLS and invocation.arg == 0:
                        lookup_start = position + 2
                        break
        if lookup_start is not None:
            lookup = instructions[lookup_start:lookup_start + 2]
            key = lookup[0].argval if tuple(part.opname for part in lookup) == ('LOAD_CONST', 'BINARY_SUBSCR') and isinstance(lookup[0].argval, str) else None
            found.extend((name, key) for name in names)
    return (*found, *((alias, key) for part in code.co_consts if isinstance(part, CodeType) for name, key in namespace_calls(part) for alias in aliases.get(name, {name})))


@lru_cache(maxsize=2048)
def reflective_reads(code: CodeType) -> tuple[tuple[str, str], ...]:
    instructions = tuple(dis.get_instructions(code))
    found: list[tuple[str, str]] = []
    for index, part in enumerate(instructions):
        indirect = part.opname == 'LOAD_CONST' and part.argval in ('__globals__', '__builtins__') and any(item.opname in _CALLS for item in instructions[index + 1:index + 4])
        if part.argval in ('__globals__', '__builtins__') and (part.opname in ('LOAD_ATTR', 'LOAD_METHOD') or indirect):
            lookup = tuple(item for item in instructions[index + 1:index + 6] if item.opname != 'PRECALL')
            if indirect:
                lookup = lookup[next(offset for offset, item in enumerate(lookup) if item.opname in _CALLS) + 1:]
            if tuple(item.opname for item in lookup[:2]) != ('LOAD_CONST', 'BINARY_SUBSCR') or not isinstance(lookup[0].argval, str):
                raise ValueError('Holdout requires a literal key for reflective factory namespaces')
            found.append((str(part.argval), lookup[0].argval))
    return (*found, *(entry for part in code.co_consts if isinstance(part, CodeType) for entry in reflective_reads(part)))
