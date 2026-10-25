from collections.abc import Collection, Mapping, Sequence
import re
from typing import cast

from limen.experiment._resolve_backtest_config import BACKTEST_KEYS
from limen.experiment._resolve_trade_policy import TRADE_NUMBERS

__all__: list[str] = []
_REFERENCE = re.compile(r'\{(\w+)\}')


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _collect(value: object, path: str, found: dict[str, str], *, embedded: bool) -> None:
    if isinstance(value, str):
        for match in _REFERENCE.finditer(value):
            if embedded or match.group(0) == value:
                _ = found.setdefault(match.group(1), path)
    elif isinstance(value, Mapping):
        for key, item in cast(Mapping[str, object], value).items():
            _collect(item, f'{path}.{key}', found, embedded=embedded)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(cast(Sequence[object], value)):
            _collect(item, f'{path}[{index}]', found, embedded=embedded)


def lookup_references(manifest: Mapping[str, object], declared: Collection[str]) -> dict[str, str]:
    found: dict[str, str] = {}

    def native(values: object, path: str, *, groups: bool = False) -> None:
        for key, value in _mapping(values).items():
            if isinstance(value, str) and value not in declared and not value.startswith('_') and not (groups and key == 'group'):
                _collect(value, path + '.' + key, found, embedded=True)

    ml = manifest.get('type') == 'ml'
    for name in ('indicators', 'features') if ml else ('indicators',):
        entries = manifest.get(name)
        if isinstance(entries, list):
            for index, entry in enumerate(cast(list[object], entries)):
                native(_mapping(entry).get('params'), f'sfd.manifest.{name}[{index}].params', groups=True)
    if ml:
        for name in ('pre_split_data_selector', 'bar_formation', 'scaler'):
            native(_mapping(manifest.get(name)).get('params'), f'sfd.manifest.{name}.params')
        for name in ('fit_params', 'transform_params'):
            native(_mapping(manifest.get('target')).get(name), f'sfd.manifest.target.{name}')
        for name in ('probability_calibration', 'threshold_function'):
            native(_mapping(_mapping(manifest.get('calibration')).get(name)).get('params'), f'sfd.manifest.calibration.{name}.params')
        _collect(manifest.get('feature_shims'), 'sfd.manifest.feature_shims', found, embedded=True)
    else:
        conditions = _mapping(manifest.get('strategy')).get('conditions')
        if isinstance(conditions, list):
            for index, condition in enumerate(cast(list[object], conditions)):
                for name in ('column', 'other_column', 'value', 'lookback', 'expr', 'persistence_n', 'recency_n'):
                    value = _mapping(condition).get(name)
                    if isinstance(value, str):
                        _collect(value, f'sfd.manifest.strategy.conditions[{index}].{name}', found, embedded=True)
    backtest = _mapping(manifest.get('backtest'))
    for name in (*BACKTEST_KEYS, *TRADE_NUMBERS, 'initial_equity'):
        _collect(backtest.get(name), f'sfd.manifest.backtest.{name}', found, embedded=False)
    for name in ('quantity_step', 'min_notional', 'initial_margin_fraction', 'maintenance_margin_fraction'):
        _collect(_mapping(backtest.get('product')).get(name), f'sfd.manifest.backtest.product.{name}', found, embedded=False)
    for path, values in (
        ('allocation.params', _mapping(manifest.get('allocation')).get('params')),
        ('backtest.funding.params', _mapping(backtest.get('funding')).get('params')),
        ('backtest.barrier_features.params', _mapping(backtest.get('barrier_features')).get('params')),
    ):
        _collect(values, f'sfd.manifest.{path}', found, embedded=False)
    return found
