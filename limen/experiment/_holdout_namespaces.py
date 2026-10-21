import dis
from functools import lru_cache
from types import CodeType

__all__ = ['namespace_calls']
_CALLS = frozenset({'CALL', 'CALL_FUNCTION', 'CALL_METHOD'})
_LOOKUP_WIDTH = 2


@lru_cache(maxsize=2048)
def namespace_calls(code: CodeType) -> tuple[tuple[str, str | None], ...]:
    instructions = tuple(part for part in dis.get_instructions(code) if part.opname != 'PRECALL')
    found: list[tuple[str, str | None]] = []
    for index, part in enumerate(instructions[:-1]):
        if part.opname not in ('LOAD_ATTR', 'LOAD_METHOD', 'LOAD_CONST') or not isinstance(part.argval, str):
            continue
        call = instructions[index + 1]
        lookup_start: int | None = index + 2 if call.opname in _CALLS and call.arg == 0 else None
        if lookup_start is None and part.opname == 'LOAD_CONST':
            for position in range(index + 1, min(index + 3, len(instructions) - 1)):
                getter, invocation = instructions[position:position + 2]
                if getter.opname in _CALLS and getter.arg in (2, 3) and invocation.opname in _CALLS and invocation.arg == 0:
                    lookup_start = position + 2
                    break
        if lookup_start is not None:
            lookup = instructions[lookup_start:lookup_start + 2]
            key = lookup[0].argval if len(lookup) == _LOOKUP_WIDTH and lookup[0].opname == 'LOAD_CONST' and isinstance(lookup[0].argval, str) and lookup[1].opname == 'BINARY_SUBSCR' else None
            found.append((part.argval, key))
    return (*found, *(entry for part in code.co_consts if isinstance(part, CodeType) for entry in namespace_calls(part)))
