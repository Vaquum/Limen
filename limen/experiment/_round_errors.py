from collections.abc import Callable, Mapping
import csv
from functools import wraps
import inspect
import json
import logging
from pathlib import Path
from typing import Concatenate, ParamSpec, TypeVar, cast

from limen.experiment._fold_fingerprint import FoldEvidenceError
from limen.experiment._holdout_history import HoldoutEvidenceError
from limen.experiment._round_storage import RoundStorageError
from limen.experiment.errors import StrictModeError

__all__: list[str] = []
logger = logging.getLogger(__name__)
_P = ParamSpec('_P')
_Host = TypeVar('_Host')
_Result = TypeVar('_Result')


class RoundErrorPolicy:
    def __init__(self, mode: object = 'abort', limit: object = None) -> None:
        if not isinstance(mode, str) or mode not in ('abort', 'record'):
            raise ValueError('on_round_error must be abort or record')
        if mode == 'record' and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ValueError('record requires a positive integer max_round_errors')
        if mode == 'abort' and limit is not None:
            raise ValueError('max_round_errors requires on_round_error=record')
        self.mode, self.limit = mode, cast(int | None, limit)
        self.failures = 0
        self._terminal: Exception | None = None

    @property
    def metadata(self) -> dict[str, object]:
        return {'round_error_policy': {'on_round_error': self.mode, 'max_round_errors': self.limit}} if self.mode == 'record' else {}

    @property
    def exhausted(self) -> bool:
        return self._terminal is not None

    def modes(self, workers: object, *, search: bool, factorize: object = False,
              walk_forward: object = None, protected: object = None) -> None:
        if self.mode == 'record' and (workers != 1 or not search or factorize is True or walk_forward is not None or protected is not None):
            raise ValueError('record round errors requires ordinary unprotected serial MSQ execution')

    def admit(self, host: object, call: Mapping[str, object]) -> None:
        attributes = cast(dict[str, object], vars(host))
        manifest = attributes.get('manifest')
        workers = call.get('workers') if call.get('workers') is not None else attributes.get('_workers') or 1
        self.modes(workers, search=attributes.get('_search_strategy') is not None,
                   factorize=call.get('factorize'), walk_forward=getattr(manifest, 'split_walk_forward', None),
                   protected=attributes.get('_holdout_binding') or getattr(manifest, '_holdout_binding', None))
        self.failures, self._terminal = 0, None
        directory = attributes.get('_experiment_dir')
        if call.get('resume') is not True or not isinstance(directory, Path):
            return
        path = directory / 'metadata.json'
        if not path.is_file():
            return
        document = cast(object, json.loads(path.read_text()))
        if not isinstance(document, dict):
            raise ValueError('Invalid saved experiment metadata')
        saved = cast(dict[str, object], document).get('round_error_policy')
        if saved != self.metadata.get('round_error_policy'):
            raise ValueError('Cannot resume with a changed round error policy or budget')
        if self.mode == 'record':
            with (directory / 'results.csv').open(newline='') as stream:
                self.failures = sum(bool(row.get('strict_mode_error')) for row in csv.DictReader(stream))
            if self.failures > cast(int, self.limit):
                raise ValueError('Cannot resume an exhausted round error budget')

    def failure(self, host: object, error: Exception, index: int) -> dict[str, object]:
        if isinstance(error, (HoldoutEvidenceError, RoundStorageError, FoldEvidenceError)):
            raise error
        if not isinstance(error, StrictModeError) and self.mode == 'abort':
            raise error
        if self.mode == 'record':
            if index == 0:
                raise RuntimeError('Record mode requires a successful first round') from error
            self.failures += 1
            if self.failures > cast(int, self.limit):
                vars(host)['_shutdown_requested'] = True
                self._terminal = error
        logger.error('Round %d failed: %s: %s', index, type(error).__name__, error)
        message = str(error) or type(error).__name__
        return {'strict_mode_error': f'{type(error).__name__}: {message}' if self.mode == 'record' else message}

    def finish(self) -> None:
        if self._terminal is not None:
            raise RuntimeError(f'Round error budget exceeded: {self.failures} > {self.limit}; failed prefix retained') from self._terminal


def admit(host: object, call: Mapping[str, object]) -> None:
    cast(RoundErrorPolicy, vars(host)['_round_error_policy']).admit(host, call)


def configuration(config: Mapping[str, object]) -> None:
    raw = config.get('uel')
    uel = cast(Mapping[str, object], raw) if isinstance(raw, Mapping) else {}
    raw_sfd = config.get('sfd')
    sfd = cast(Mapping[str, object], raw_sfd) if isinstance(raw_sfd, Mapping) else {}
    raw_manifest = sfd.get('manifest')
    manifest = cast(Mapping[str, object], raw_manifest) if isinstance(raw_manifest, Mapping) else {}
    policy = RoundErrorPolicy(uel.get('on_round_error', 'abort'), uel.get('max_round_errors'))
    policy.modes(uel.get('workers', 1), search=True, factorize=uel.get('factorize'), walk_forward=manifest.get('split_walk_forward'))


def guarded(func: Callable[Concatenate[_Host, _P], _Result]) -> Callable[Concatenate[_Host, _P], _Result]:
    @wraps(func)
    def run(host: _Host, *args: _P.args, **kwargs: _P.kwargs) -> _Result:
        admit(host, cast(Mapping[str, object], inspect.signature(func).bind(host, *args, **kwargs).arguments))
        result = func(host, *args, **kwargs)
        cast(RoundErrorPolicy, vars(host)['_round_error_policy']).finish()
        return result
    return run
