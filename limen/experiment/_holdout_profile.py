from collections.abc import Callable, Iterator
from contextlib import contextmanager
from cProfile import Profile
import sys
from types import FrameType
from threading import local

__all__ = ['profile_active', 'readonly_profile']


class _ProfileState(local):
    failure: list[str] | None = None


_ACTIVE = _ProfileState()


def _audit(event: str, _args: tuple[object, ...]) -> None:
    active = _ACTIVE.failure if event == 'sys.setprofile' else None
    if active is not None:
        active.append('Holdout inference attempted profile replacement')
        raise ValueError(active[-1])


sys.addaudithook(_audit)


def profile_active() -> bool:
    return _ACTIVE.failure is not None


@contextmanager
def readonly_profile(callback: Callable[[FrameType, str, object], None]) -> Iterator[None]:
    previous, setprofile = sys.getprofile(), sys.setprofile
    failure: list[str] = []

    try:
        if isinstance(previous, Profile):
            previous.disable()
        setprofile(callback)
        _ACTIVE.failure = failure
        try:
            yield
        finally:
            if sys.getprofile() is not callback:
                failure.append('Holdout inference profile changed')
            _ACTIVE.failure = None
            if isinstance(previous, Profile):
                previous.enable()
            else:
                setprofile(previous)
    finally:
        if failure:
            raise ValueError(failure[0])
