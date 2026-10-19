from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Literal, TypedDict, cast

from limen.backtest import _cost_history as _history
from limen.backtest import _cost_identity as _identity
from limen.backtest import _cost_statistics as _statistics
from limen.backtest.trade_contract import JsonValue

CostDisposition = Literal['original', 'retrospective_cost_scenario',
                          'retrospective_low_effective_support', 'unavailable', 'economic_failure']
CostClock = Literal['assumed_exchange_label']
_FORMAT = 'limen.execution_cost.calibration'
_ENVELOPE_KEYS = frozenset(('format', 'version', 'digest', 'payload'))
_PAYLOAD_KEYS = frozenset(('history', 'method', 'cutoff_ns', 'available_at_ns', 'created_at_ns', 'report'))
_REPORT_KEYS = frozenset(('method', 'source', 'clocks', 'estimates', 'support', 'sensitivity', 'exclusions', 'limitations'))


class CostConditions(TypedDict):
    venue: str
    environment: Literal['mainnet']
    product_kind: Literal['linear_perpetual']
    instrument: str
    base_currency: str
    quote_currency: str
    reference_id: str
    reference_convention: Literal['minute_midpoint_label']
    clock: CostClock


class CalibrationReport(TypedDict):
    digest: str
    method: Literal['minute_midpoint_1s_v1']
    source: JsonValue
    clocks: JsonValue
    estimates: JsonValue
    support: JsonValue
    sensitivity: JsonValue
    exclusions: JsonValue
    limitations: JsonValue


class ComparisonReport(TypedDict):
    digest: str
    calibration_digest: str
    candidates: JsonValue
    ranks: JsonValue
    failures: JsonValue
    unavailable: JsonValue
    limitations: JsonValue


class CostCalibration:
    """Immutable recorded-price stress with separately reported account fees.

    Pin ``digest`` in the caller's research specification before later ``read``.
    The method preserves original candidate fees; observed cross-account fees
    never establish the fees a different account will pay.
    """

    __slots__ = ('_payload', '_digest')
    _payload: dict[str, JsonValue]
    _digest: str

    def __init__(self) -> None:
        raise TypeError('Use CostCalibration.from_history() or CostCalibration.read()')

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError('CostCalibration is immutable')

    @classmethod
    def _create(cls, payload: dict[str, JsonValue]) -> CostCalibration:
        instance = object.__new__(cls)
        object.__setattr__(instance, '_payload', copy.deepcopy(payload))
        object.__setattr__(instance, '_digest', _identity.digest(instance._content()))
        return instance

    def _content(self) -> dict[str, JsonValue]:
        return {'format': _FORMAT, 'version': 1, 'payload': self._payload}

    @classmethod
    def from_history(cls, path: str | Path, *, cutoff_ns: int) -> CostCalibration:
        """Validate local normalized facts and freeze their available evidence."""
        cutoff = _history.integer(cutoff_ns, 'cutoff_ns')
        history = _history.validate(_identity.read_json(path))
        report = _statistics.report(history, cutoff)
        clocks = _history.mapping(report['clocks'])
        return cls._create({'history': history, 'method': _history.METHOD, 'cutoff_ns': cutoff,
                            'available_at_ns': clocks['available_at_ns'], 'created_at_ns': time.time_ns(),
                            'report': report})

    @classmethod
    def read(cls, path: str | Path, *, expected_digest: str) -> CostCalibration:
        """Read externally pinned content without re-estimating its rates."""
        envelope = _history.mapping(_identity.read_json(path), _ENVELOPE_KEYS)
        expected = _history.text(expected_digest, 'expected digest')
        if len(expected) != 64 or any(char not in '0123456789abcdef' for char in expected):
            raise ValueError('Malformed expected calibration digest')
        if envelope['format'] != _FORMAT or type(envelope['version']) is not int or envelope['version'] != 1:
            raise ValueError('Unsupported calibration artifact format/version')
        payload = _history.mapping(envelope['payload'], _PAYLOAD_KEYS)
        if payload['method'] != _history.METHOD:
            raise ValueError('Unsupported execution-cost method')
        _ = _history.integer(payload['cutoff_ns'], 'cutoff_ns')
        _ = _history.integer(payload['created_at_ns'], 'created_at_ns')
        if payload['available_at_ns'] is not None:
            _ = _history.integer(payload['available_at_ns'], 'available_at_ns')
        _ = _history.validate(payload['history'])
        report = _history.mapping(payload['report'], _REPORT_KEYS)
        clocks = _history.mapping(report['clocks'])
        if (report['method'] != payload['method'] or clocks.get('available_at_ns') != payload['available_at_ns']
                or clocks.get('cutoff_ns') != payload['cutoff_ns']):
            raise ValueError('Calibration artifact clocks/method contradict its report')
        result = cls._create(payload)
        if envelope['digest'] != expected or result.digest != expected:
            raise ValueError('Calibration content differs from the externally pinned digest')
        encoded = json.dumps(envelope, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
        if Path(path).read_bytes() != encoded:
            raise ValueError('Calibration artifact bytes are not canonical JSON')
        return result

    @property
    def digest(self) -> str:
        """Return the versioned canonical content identity."""
        if _identity.digest(self._content()) != self._digest:
            raise ValueError('Frozen calibration content changed')
        return self._digest

    def report(self) -> CalibrationReport:
        """Return copied estimates, source claims and explicit transfer limits."""
        result = copy.deepcopy(_history.mapping(self._payload['report']))
        result['digest'] = self.digest
        return cast(CalibrationReport, result)

    def write(self, path: str | Path) -> str:
        """Publish once; the same canonical bytes may be written idempotently."""
        identity = self.digest
        _identity.publish(path, {**self._content(), 'digest': identity})
        return identity


__all__ = ['CalibrationReport', 'ComparisonReport', 'CostCalibration', 'CostClock', 'CostConditions', 'CostDisposition']
