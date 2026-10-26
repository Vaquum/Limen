from collections.abc import Collection, Iterator, Mapping, Sequence
import re
from typing import cast

from limen.experiment._resolve_backtest_config import BACKTEST_KEYS
from limen.experiment._resolve_trade_policy import TRADE_NUMBERS

__all__: list[str] = []
_REFERENCE = re.compile(r'\{\{|\}\}|\{(\w+)\}')


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _collect(value: object, path: str, found: dict[str, str], *, embedded: bool) -> None:
    if isinstance(value, str):
        names = (match.group(1) for match in _REFERENCE.finditer(value)) if embedded else (
            (value[1:-1],) if value.startswith('{') and value.endswith('}') else ())
        for name in names:
            if name is not None:
                _ = found.setdefault(name, path)
    elif isinstance(value, Mapping):
        for key, item in cast(Mapping[str, object], value).items():
            _collect(item, f'{path}.{key}', found, embedded=embedded)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(cast(Sequence[object], value)):
            _collect(item, f'{path}[{index}]', found, embedded=embedded)


def native_bindings(manifest: Mapping[str, object]) -> Iterator[tuple[str, object, bool]]:
    '''Yield formatted bindings; the flag identifies native exact-name lookup.'''
    def arguments(values: object, path: str, *, groups: bool = False) -> Iterator[tuple[str, object, bool]]:
        for key, value in _mapping(values).items():
            if not (groups and key == 'group'):
                yield f'{path}.{key}', value, True

    ml = manifest.get('type') == 'ml'
    for name in ('indicators', 'features') if ml else ('indicators',):
        entries = manifest.get(name)
        if isinstance(entries, list):
            for index, entry in enumerate(cast(list[object], entries)):
                yield from arguments(_mapping(entry).get('params'), f'sfd.manifest.{name}[{index}].params', groups=True)
    if ml:
        for name in ('pre_split_data_selector', 'bar_formation', 'scaler'):
            yield from arguments(_mapping(manifest.get(name)).get('params'), f'sfd.manifest.{name}.params')
        for name in ('fit_params', 'transform_params'):
            yield from arguments(_mapping(manifest.get('target')).get(name), f'sfd.manifest.target.{name}')
        for name in ('probability_calibration', 'threshold_function'):
            yield from arguments(_mapping(_mapping(manifest.get('calibration')).get(name)).get('params'), f'sfd.manifest.calibration.{name}.params')
        yield 'sfd.manifest.feature_shims', manifest.get('feature_shims'), False
    else:
        conditions = _mapping(manifest.get('strategy')).get('conditions')
        if isinstance(conditions, list):
            for index, condition in enumerate(cast(list[object], conditions)):
                for name in ('column', 'other_column', 'value', 'lookback', 'expr', 'persistence_n', 'recency_n'):
                    value = _mapping(condition).get(name)
                    if isinstance(value, str):
                        yield f'sfd.manifest.strategy.conditions[{index}].{name}', value, False


def json_bindings(manifest: Mapping[str, object]) -> Iterator[tuple[str, object]]:
    backtest = _mapping(manifest.get('backtest'))
    for name in (*BACKTEST_KEYS, *TRADE_NUMBERS, 'initial_equity'):
        value = backtest.get(name)
        yield f'sfd.manifest.backtest.{name}', value.strip() if name in BACKTEST_KEYS and isinstance(value, str) else value
    for name in ('quantity_step', 'min_notional', 'initial_margin_fraction', 'maintenance_margin_fraction'):
        yield f'sfd.manifest.backtest.product.{name}', _mapping(backtest.get('product')).get(name)
    for path, values in (
        ('allocation.params', _mapping(manifest.get('allocation')).get('params')),
        ('backtest.funding.params', _mapping(backtest.get('funding')).get('params')),
        ('backtest.execution_data_source.params', _mapping(backtest.get('execution_data_source')).get('params')),
        ('backtest.funding.data_source.params', _mapping(_mapping(backtest.get('funding')).get('data_source')).get('params')),
        ('backtest.barrier_features.params', _mapping(backtest.get('barrier_features')).get('params')),
    ):
        yield f'sfd.manifest.{path}', values


def lookup_references(manifest: Mapping[str, object], declared: Collection[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for path, value, bare in native_bindings(manifest):
        if bare and (not isinstance(value, str) or value in declared or value.startswith('_')):
            continue
        _collect(value, path, found, embedded=True)
    for path, value in json_bindings(manifest):
        _collect(value, path, found, embedded=False)
    return found
