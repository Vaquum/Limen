from collections.abc import Callable, Iterator
from contextlib import contextmanager
from cProfile import Profile
import sys
from types import FrameType
from threading import local

__all__ = ['readonly_profile']


class _ProfileState(local):
    failure: list[str] | None = None


_ACTIVE = _ProfileState()


def _audit(event: str, _args: tuple[object, ...]) -> None:
    active = _ACTIVE.failure if event == 'sys.setprofile' else None
    if active is not None:
        active.append('Holdout inference attempted profile replacement')
        raise ValueError(active[-1])


sys.addaudithook(_audit)


@contextmanager
def readonly_profile(callback: Callable[[FrameType, str, object], None]) -> Iterator[None]:
    previous, setprofile = sys.getprofile(), sys.setprofile
    failure: list[str] = []

    def observe(frame: FrameType, event: str, arg: object) -> None:
        try:
            if event == 'c_call' and _ACTIVE.failure is failure and isinstance(getattr(arg, '__self__', None), Profile) and getattr(arg, '__name__', None) in ('enable', 'disable'):
                failure.append('Holdout inference attempted profile replacement')
                raise ValueError(failure[-1])
            callback(frame, event, arg)
            if callable(previous):
                _ = previous(frame, event, arg)
        except BaseException:
            failure.append('Holdout inference guard failed')
            raise

    try:
        if isinstance(previous, Profile):
            previous.disable()
        setprofile(observe)
        _ACTIVE.failure = failure
        try:
            yield
        finally:
            if sys.getprofile() is not observe:
                failure.append('Holdout inference profile changed')
            _ACTIVE.failure = None
            if isinstance(previous, Profile):
                previous.enable()
            else:
                setprofile(previous)
    finally:
        if failure:
            raise ValueError(failure[0])
