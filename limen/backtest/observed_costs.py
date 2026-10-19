from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Literal, TypedDict, cast

import polars as pl

from limen.backtest import _cost_comparison as _comparison
from limen.backtest import _cost_history as _history
from limen.backtest import _cost_identity as _identity
from limen.backtest import _cost_statistics as _statistics
from limen.backtest.trade_contract import JsonValue, TradeInputs, TradeLedger, TradePolicy

CostDisposition = Literal['original', 'retrospective_cost_scenario',
                          'retrospective_low_effective_support', 'unavailable', 'economic_failure']
CostClock = Literal['assumed_exchange_label']
_FORMAT = 'limen.execution_cost.calibration'



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


@dataclass(frozen=True, eq=False)
class CostCandidate:
    inputs: InitVar[TradeInputs]
    policy: InitVar[TradePolicy]
    research_id: InitVar[str]
    conditions: InitVar[CostConditions]
    benchmark: InitVar[pl.DataFrame]
    _snapshot: dict[str, object] = field(init=False)
    _seal: str = field(init=False)

    def __post_init__(self, inputs: TradeInputs, policy: TradePolicy, research_id: str,
                      conditions: CostConditions, benchmark: pl.DataFrame) -> None:
        snapshot = _comparison.freeze(inputs, policy, research_id, conditions, benchmark)
        object.__setattr__(self, '_snapshot', snapshot)
        object.__setattr__(self, '_seal', _identity.digest(snapshot))


@dataclass(frozen=True, slots=True, eq=False)
class CostComparison:
    _payload: dict[str, JsonValue]
    _ledgers: dict[tuple[str, str], TradeLedger]
    _digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, '_payload', copy.deepcopy(self._payload))
        object.__setattr__(self, '_ledgers', copy.deepcopy(self._ledgers))
        object.__setattr__(self, '_digest', _identity.digest(self._content()))

    def _content(self) -> dict[str, JsonValue]:
        return {'format': 'limen.execution_cost.comparison', 'version': 1, 'payload': self._payload}

    @property
    def digest(self) -> str:
        if _identity.digest(self._content()) != self._digest:
            raise ValueError('Frozen comparison content changed')
        return self._digest

    def report(self) -> ComparisonReport:
        return cast(ComparisonReport, {'digest': self.digest, **{key: copy.deepcopy(self._payload[key])
            for key in ('calibration_digest', 'candidates', 'ranks', 'failures', 'unavailable', 'limitations')}})

    def ledger(self, candidate: str, scenario: Literal['original', 'calibrated']) -> TradeLedger:
        if not isinstance(cast(object, candidate), str) or not isinstance(cast(object, scenario), str):
            raise TypeError('Ledger names must be strings')
        _ = self.digest
        if (candidate, scenario) not in self._ledgers:
            raise ValueError('Unknown candidate/scenario or no executed ledger')
        ledger = self._ledgers[candidate, scenario]
        item = _history.mapping(_history.mapping(self._payload['candidates'])[candidate])
        expected = _history.mapping(_history.mapping(item['scenarios'])[scenario])['ledger']
        if _identity.digest(_comparison.wire(ledger)) != _identity.digest(expected):
            raise ValueError('Frozen comparison ledger changed')
        return copy.deepcopy(ledger)

    def write(self, path: str | Path) -> str:
        identity = self.digest
        _identity.publish(path, {**self._content(), 'digest': identity})
        return identity


@dataclass(frozen=True, slots=True, eq=False)
class CostCalibration:
    """Immutable recorded-price stress with separately reported account fees.

    Pin ``digest`` in the caller's research specification before later ``read``.
    The method preserves original candidate fees; observed cross-account fees
    never establish the fees a different account will pay.
    """

    _payload: dict[str, JsonValue]
    _digest: str = field(init=False)

    def __post_init__(self) -> None:
        payload = _history.mapping(self._payload, _history.CALIBRATION_KEYS)
        object.__setattr__(self, '_payload', payload)
        object.__setattr__(self, '_digest', _identity.digest(self._content()))

    @classmethod
    def _create(cls, payload: dict[str, JsonValue]) -> CostCalibration:
        return cls(payload)

    def _content(self) -> dict[str, JsonValue]:
        return {'format': _FORMAT, 'version': 1, 'payload': self._payload}

    @classmethod
    def from_history(cls, path: str | Path, *, cutoff_ns: int) -> CostCalibration:
        """Validate local normalized facts and freeze their available evidence."""
        cutoff = _history.integer(cutoff_ns, 'cutoff_ns')
        history = _history.validate(_identity.read_json(path))
        report = _statistics.report(history, cutoff)
        clocks = _history.mapping(report['clocks'])
        created_at_ns = time.time_ns()
        report['clocks'] = {**clocks, 'created_at_ns': created_at_ns}
        return cls._create({'history': history, 'method': _history.METHOD, 'cutoff_ns': cutoff,
                            'available_at_ns': clocks['available_at_ns'], 'created_at_ns': created_at_ns,
                            'report': report})

    @classmethod
    def read(cls, path: str | Path, *, expected_digest: str) -> CostCalibration:
        """Read externally pinned content without re-estimating its rates."""
        return cls._create(_history.read_calibration(path, expected_digest))

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


    def compare(self, candidates: Mapping[str, CostCandidate]) -> CostComparison:
        if not isinstance(cast(object, candidates), Mapping):
            raise TypeError('Candidates must be a mapping of names to CostCandidate')
        if not candidates:
            raise ValueError('Comparison requires at least one candidate')
        frozen: dict[str, Mapping[str, object]] = {}
        for name, candidate in candidates.items():
            _ = _history.text(name, 'candidate name')
            if not isinstance(cast(object, candidate), CostCandidate):
                raise TypeError('Every candidate must be a CostCandidate')
            state = cast(dict[str, object], vars(candidate))
            snapshot = cast(dict[str, object], state['_snapshot'])
            if _identity.digest(snapshot) != state['_seal']:
                raise ValueError('Frozen candidate content changed')
            frozen[name] = snapshot
        payload, ledgers = _comparison.compare(self._payload, self.digest, frozen)
        return CostComparison(payload, ledgers)


__all__ = ['CalibrationReport', 'ComparisonReport', 'CostCalibration', 'CostCandidate', 'CostClock', 'CostComparison', 'CostConditions', 'CostDisposition']
