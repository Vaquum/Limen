from collections.abc import Callable, Iterator
from contextlib import contextmanager
from cProfile import Profile
import sys
from types import FrameType

__all__ = ['readonly_profile']


@contextmanager
def readonly_profile(callback: Callable[[FrameType, str, object], None]) -> Iterator[None]:
    previous, owner, setprofile = sys.getprofile(), sys._getframe(), sys.setprofile
    failure: str | None = None

    def observe(frame: FrameType, event: str, arg: object) -> None:
        nonlocal failure
        try:
            if event == 'c_call' and frame is not owner and (arg is setprofile or (isinstance(getattr(arg, '__self__', None), Profile) and getattr(arg, '__name__', None) in ('enable', 'disable'))):
                failure = 'Holdout inference attempted profile replacement'
                raise ValueError(failure)
            callback(frame, event, arg)
            if callable(previous):
                _ = previous(frame, event, arg)
        except BaseException:
            failure = failure or 'Holdout inference guard failed'
            raise

    try:
        if isinstance(previous, Profile):
            previous.disable()
        setprofile(observe)
        try:
            yield
        finally:
            if sys.getprofile() is not observe:
                failure = failure or 'Holdout inference profile changed'
            if isinstance(previous, Profile):
                previous.enable()
            else:
                setprofile(previous)
    finally:
        if failure is not None:
            raise ValueError(failure)
