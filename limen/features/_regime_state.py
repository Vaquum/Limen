from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
from typing import cast

from limen.features._regime_audit import Bar, admit_bar
from limen.features._regime_config import RegimeConfig
from limen.features._regime_contract import MAX_BARS, RANKS, RANK_WINDOW, SLOW, Regime
from limen.features._regime_ranks import Ranks
from limen.features._regime_rules import BREAKOUTS, Decision

__all__: list[str] = []


@dataclass
class Series:
    last_timestamp: int | None = None
    bars: deque[Bar | None] = field(default_factory=lambda: deque(maxlen=SLOW + 1))
    ranks: Ranks = field(default_factory=Ranks)
    decision: Decision = field(default_factory=Decision)


def dump_state(states: dict[str, Series], config_hash: str, feature_hash: str) -> str:
    payload = {key: {'last_timestamp': state.last_timestamp,
                     'bars': [list(asdict(bar).values()) if bar is not None else None for bar in state.bars],
                     'ranks': [list(values) for values in state.ranks.history],
                     'decision': asdict(state.decision)} for key, state in states.items()}
    return json.dumps({'schema_version': 1, 'config_hash': config_hash,
                       'feature_hash': feature_hash, 'series': payload},
                      sort_keys=True, separators=(',', ':'), allow_nan=False)


def _mapping(value: object, keys: set[str] | None = None) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError('Checkpoint requires a JSON object')
    result = cast(dict[str, object], value)
    if keys is not None and set(result) != keys:
        raise ValueError('Checkpoint object fields differ')
    return result


def _list(value: object, maximum: int) -> list[object]:
    if not isinstance(value, list):
        raise ValueError('Checkpoint history must be an array')
    result = cast(list[object], value)
    if len(result) > maximum:
        raise ValueError('Checkpoint history exceeds its window')
    return result


def _integer(value: object, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError('Checkpoint integer is outside its declared domain')
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Checkpoint observations must be finite numbers')
    return float(value)


def _optional(value: object) -> float | None:
    if value is None:
        return None
    result = _number(value)
    if result < 0:
        raise ValueError('Checkpoint magnitude must be non-negative')
    return result


def _decision(value: object, config: RegimeConfig) -> Decision:
    raw = _mapping(value, {'regime', 'bars', 'pending', 'pending_count', 'boundary', 'previous_up', 'previous_down'})
    regime = Regime(_integer(raw['regime'], 0, len(Regime) - 1))
    pending = None if raw['pending'] is None else Regime(_integer(raw['pending'], 1, len(Regime) - 1))
    count = _integer(raw['pending_count'], 0, config.confirmation_rows - 1)
    boundary = _optional(raw['boundary'])
    if (pending is None) != (count == 0) or pending == regime:
        raise ValueError('Checkpoint pending commitment is inconsistent')
    if (regime in BREAKOUTS) != (boundary is not None):
        raise ValueError('Checkpoint breakout boundary is inconsistent')
    if boundary is not None and boundary <= 0:
        raise ValueError('Checkpoint breakout boundary must be positive')
    if regime in {*BREAKOUTS, Regime.INITIALIZING} and pending is not None:
        raise ValueError('Checkpoint regime cannot hold a pending commitment')
    return Decision(regime=regime, bars=_integer(raw['bars'], 1, MAX_BARS), pending=pending,
                     pending_count=count, boundary=boundary, previous_up=_optional(raw['previous_up']),
                     previous_down=_optional(raw['previous_down']))


def _series(value: object, config: RegimeConfig) -> Series:
    raw = _mapping(value, {'last_timestamp', 'bars', 'ranks', 'decision'})
    state = Series(last_timestamp=_integer(raw['last_timestamp'], -(2**63), 2**63 - 1),
                    decision=_decision(raw['decision'], config))
    bars = _list(raw['bars'], SLOW + 1)
    if not bars:
        raise ValueError('Checkpoint series has no observation history')
    for item in bars:
        if item is None:
            state.bars.append(None)
        else:
            observation, error = admit_bar(tuple(_list(item, 5)))
            if error or observation is None:
                raise ValueError('Checkpoint contains an invalid observed bar')
            state.bars.append(observation)
    histories = _list(raw['ranks'], len(RANKS))
    if len(histories) != len(RANKS):
        raise ValueError('Checkpoint rank histories differ from the feature schema')
    for index, history in enumerate(histories):
        values = [_number(item) for item in _list(history, RANK_WINDOW)]
        state.ranks.history[index] = deque(values)
        state.ranks.sorted[index] = sorted(values)
    return state


def restore_state(snapshot: object, config: RegimeConfig, config_hash: str, feature_hash: str) -> dict[str, Series]:
    if not isinstance(snapshot, str):
        raise TypeError('snapshot must be JSON text')
    decoded: object = json.loads(snapshot)
    payload = _mapping(decoded, {'schema_version', 'config_hash', 'feature_hash', 'series'})
    _ = _integer(payload['schema_version'], 1, 1)
    if payload['config_hash'] != config_hash or payload['feature_hash'] != feature_hash:
        raise ValueError('Checkpoint configuration or feature hash differs')
    return {key: _series(value, config) for key, value in _mapping(payload['series']).items()}
