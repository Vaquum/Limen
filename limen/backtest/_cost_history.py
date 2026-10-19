from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import cast

from limen.backtest.trade_contract import JsonValue, json_value

METHOD = 'minute_midpoint_1s_v1'
MAX_AGE_NS = 1_000_000_000
SHA256_HEX_LENGTH = 64
_WINDOW_BOUNDS = 2
_HISTORY = frozenset(('version', 'source', 'fills', 'source_window_summary', 'source_diagnostics'))
_SOURCE = frozenset(('identity', 'assets', 'venue', 'environment', 'product', 'captured_at_ns',
                     'reference_id', 'reference_convention', 'windows', 'supersedes'))
_ASSET = frozenset(('locator', 'sha256', 'receipt_bound_ns', 'receipt_basis', 'kind'))
_FILL = frozenset(('leg_id', 'account', 'parent_id', 'source_asset_sha256', 'source_row', 'block_number',
                   'intra_block_index', 'block_time_ns', 'node_local_time_ns', 'side', 'role', 'mechanism',
                   'quantity', 'quote_notional', 'time_ns', 'received_at_ns', 'fee', 'fee_currency',
                   'builder_fee', 'twap_id', 'fee_received_at_ns', 'reference_price', 'reference_time_ns',
                   'reference_received_at_ns'))


def mapping(value: object, keys: frozenset[str] | None = None) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValueError('Execution-cost record must be a mapping')
    converted = json_value(cast(Mapping[object, object], value))
    if not isinstance(converted, dict):
        raise ValueError('Execution-cost record must contain JSON values')
    if keys is not None and frozenset(converted) != keys:
        raise ValueError(f'Execution-cost record keys differ: {sorted(frozenset(converted) ^ keys)}')
    return converted


def integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonblank string')
    return value


def decimal(value: object, name: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a decimal economic string')
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f'Malformed {name}') from error
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f'Invalid {name}')
    return parsed


def rows(value: object, name: str) -> list[dict[str, JsonValue]]:
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a list')
    return [mapping(row) for row in cast(list[object], value)]


def digest_id(value: object) -> str:
    result = text(value, 'SHA-256')
    if len(result) != SHA256_HEX_LENGTH or any(char not in '0123456789abcdef' for char in result):
        raise ValueError('Malformed SHA-256')
    return result


def _source(value: object) -> dict[str, JsonValue]:
    source = mapping(value, _SOURCE)
    for name in ('identity', 'venue', 'reference_id'):
        _ = text(source[name], name)
    if source['environment'] != 'mainnet' or source['reference_convention'] != 'minute_midpoint_label':
        raise ValueError('History requires mainnet minute-midpoint evidence')
    product = mapping(source['product'], frozenset(('kind', 'instrument', 'base_currency', 'quote_currency')))
    if product['kind'] != 'linear_perpetual':
        raise ValueError('History requires linear-perpetual economic units')
    for name, item in product.items():
        _ = text(item, name)
    _ = integer(source['captured_at_ns'], 'captured_at_ns')
    if source['supersedes'] is not None:
        _ = digest_id(source['supersedes'])
    assets = rows(source['assets'], 'assets')
    if not assets:
        raise ValueError('History requires pinned source assets')
    seen: set[str] = set()
    for asset in assets:
        _ = mapping(asset, _ASSET)
        _ = text(asset['locator'], 'asset locator')
        sha = digest_id(asset['sha256'])
        if sha in seen:
            raise ValueError('Duplicate source asset')
        seen.add(sha)
        _ = integer(asset['receipt_bound_ns'], 'receipt bound')
        if asset['receipt_basis'] not in ('publisher_commit', 'collector_capture', 'caller_receipt') or asset['kind'] not in ('fills', 'benchmark'):
            raise ValueError('Unknown source receipt basis or asset kind')
    windows = source['windows']
    if not isinstance(windows, list) or not windows:
        raise ValueError('History requires explicit source windows')
    previous = -1
    for item in windows:
        if not isinstance(item, list) or len(item) != _WINDOW_BOUNDS:
            raise ValueError('Source window must be [start_ns, end_ns)')
        start, end = (integer(item[index], 'window time') for index in (0, 1))
        if not previous <= start < end:
            raise ValueError('Source windows overlap or are unordered')
        previous = end
    return source


def selected(fill: Mapping[str, JsonValue]) -> bool:
    reference_time = fill['reference_time_ns']
    return (fill['role'] == 'taker' and (reference_time is None or (type(reference_time) is int
            and 0 <= cast(int, fill['time_ns']) - reference_time <= MAX_AGE_NS)))


def validate(value: object) -> dict[str, JsonValue]:
    history = mapping(value, _HISTORY)
    if type(history['version']) is not int or history['version'] != 1:
        raise ValueError('Unsupported recorded-cost history version')
    source = _source(history['source'])
    assets = {cast(str, item['sha256']) for item in rows(source['assets'], 'assets')}
    fills = rows(history['fills'], 'fills')
    seen: set[tuple[str, str | None, str]] = set()
    previous: tuple[int, int, str, int, str] | None = None
    for fill in fills:
        _ = mapping(fill, _FILL)
        for name in ('leg_id', 'account'):
            _ = text(fill[name], name)
        for name in ('parent_id', 'mechanism', 'twap_id', 'fee_currency'):
            if fill[name] is not None:
                _ = text(fill[name], name)
        key = (cast(str, fill['account']), cast(str | None, fill['parent_id']), cast(str, fill['leg_id']))
        if key in seen:
            raise ValueError('Duplicate or conflicting fill leg identity')
        seen.add(key)
        if fill['side'] not in ('buy', 'sell') or fill['role'] not in ('taker', 'maker', 'unknown'):
            raise ValueError('Unknown fill side or role')
        for name in ('source_row', 'block_number', 'block_time_ns', 'time_ns', 'received_at_ns'):
            _ = integer(fill[name], name)
        for name in ('intra_block_index', 'node_local_time_ns', 'fee_received_at_ns', 'reference_time_ns', 'reference_received_at_ns'):
            if fill[name] is not None:
                _ = integer(fill[name], name)
        for receipt, event in (('received_at_ns', 'time_ns'), ('fee_received_at_ns', 'time_ns'),
                               ('reference_received_at_ns', 'reference_time_ns')):
            if fill[receipt] is not None and fill[event] is not None and cast(int, fill[receipt]) < cast(int, fill[event]):
                raise ValueError(f'{receipt} cannot precede {event}')
        windows = cast(list[list[int]], source['windows'])
        if not any(start <= cast(int, fill['time_ns']) < end for start, end in windows):
            raise ValueError('Fill is outside the declared source windows')
        if fill['reference_time_ns'] is not None and cast(int, fill['reference_time_ns']) > cast(int, fill['time_ns']):
            raise ValueError('Fill reference cannot come from the future')
        sha = digest_id(fill['source_asset_sha256'])
        if sha not in assets:
            raise ValueError('Fill has no pinned source asset')
        order = (cast(int, fill['time_ns']), cast(int, fill['block_number']), sha,
                 cast(int, fill['source_row']), cast(str, fill['leg_id']))
        if previous is not None and order < previous:
            raise ValueError('Fill records are out of canonical source order')
        previous = order
        _ = decimal(fill['quantity'], 'quantity', positive=True)
        _ = decimal(fill['quote_notional'], 'quote notional', positive=True)
        for name in ('fee', 'builder_fee', 'reference_price'):
            if fill[name] is not None:
                _ = decimal(fill[name], name, positive=name == 'reference_price')
        if (fill['role'] == 'taker' and fill['builder_fee'] is not None and fill['fee'] is not None
                and not 0 <= decimal(fill['builder_fee'], 'builder fee') <= decimal(fill['fee'], 'fee')):
            raise ValueError('Known taker builder fee must be included in total fee')
    summary = mapping(history['source_window_summary'])
    expected = integer(summary.get('selected_legs'), 'declared selected count')
    if expected != sum(selected(fill) for fill in fills):
        raise ValueError('Selected fill count differs from the declared normalized population')
    _ = mapping(history['source_diagnostics'])
    return history


__all__: list[str] = []
