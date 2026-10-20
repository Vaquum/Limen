from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from threading import RLock
from types import FunctionType

__all__ = ['call_codes']

_LOCK = RLock()


@contextmanager
def call_codes(functions: Sequence[FunctionType]) -> Iterator[None]:
    with _LOCK:
        originals = {function: function.__code__ for function in functions}
        try:
            for function, code in originals.items():
                function.__code__ = code.replace()
            yield
        finally:
            for function, code in originals.items():
                function.__code__ = code
