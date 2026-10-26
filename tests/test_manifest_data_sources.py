from functools import partial, wraps
import inspect
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen.data import HistoricalData
from limen.experiment import _holdout_research
from limen.experiment.holdout import release_holdout, reserve_holdout
from limen.experiment.manifest_core import Manifest, MLManifest, RuleBasedManifest, DataSourceResolver
from tests.test_holdout_research import _configuration, _project

_FILE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
_QUARTER = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'
_TYPES = (Manifest, MLManifest, RuleBasedManifest)
_APIS = ('data', 'test_data')


def _recorded(*, rows=3):
    return pl.read_parquet(_FILE).head(rows)


def _counted(method):
    @wraps(method)
    def wrapper(receiver, **params):
        type(receiver).calls += 1
        return method(receiver, **params)
    return wrapper


def _variadic(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)
    return wrapper


def load(receiver, *, rows=3):
    receiver.data = _recorded(rows=rows)
    return _recorded(rows=1)


def named_self(self, *, rows=3):
    return self.head(rows)


class _Source:
    calls = 0
    load = load
    named_self = staticmethod(named_self)

    @_counted
    def decorated(self, *, rows=3):
        self.data = _recorded(rows=rows)

    @_variadic
    def variadic(self, *, args):
        type(self).calls += 1
        self.data = _recorded(rows=args)

    def positional(self, /, **kwargs):
        type(self).calls += 1
        self.data = _recorded(rows=kwargs['self'])

    @staticmethod
    def static(*, rows=3):
        return _recorded(rows=rows)

    def __call__(self, *, rows=3):
        return _recorded(rows=rows)


# Keep real function/staticmethod descriptors with deliberately nonstandard parameters.
_Source.load.__qualname__ = '_Source.load'
_Source.named_self.__qualname__ = '_Source.named_self'


class _EmptySource:
    def load(self):
        return _recorded()


class _NeedsArgument:
    def __init__(self, required):
        self.required = required

    def load(self):
        return _recorded()


def _fetch(manifest, api, method, params):
    getattr(manifest, f'set_{api}_source')(method, params=params)
    return getattr(manifest, f'fetch_{api}')()


def _normalized(frame):
    return frame.with_columns(
        pl.col('datetime').cast(pl.Datetime('ms', 'UTC')),
    ).sort('datetime')


@pytest.mark.parametrize('manifest_type', _TYPES)
@pytest.mark.parametrize('api', _APIS)
@pytest.mark.parametrize('form', ('function', 'partial', 'parquet_partial', 'instance', 'local', 'static',
                                'named_self', 'receiver', 'unbound', 'bound', 'decorated',
                                'variadic', 'variadic_bound', 'positional', 'positional_bound',
                                'selective_wrapper', 'native_file', 'native_file_bound',
                                'native_spot', 'native_spot_bound'))
def test_manifest_callable_sources(manifest_type, api, form):
    raw = pl.read_parquet(_FILE)
    expected = raw.head(3)
    params = {'rows': 3}
    owner = _Source()
    calls = []

    def local(*, rows):
        calls.append(rows)
        return raw.head(rows)

    @wraps(HistoricalData.get_any_file)
    def selective(*, rows):
        calls.append(rows)
        return raw.head(rows)

    methods = {'function': _recorded, 'partial': partial(_recorded, rows=2),
               'parquet_partial': partial(pl.read_parquet, _FILE),
               'instance': owner, 'local': local, 'static': _Source.static,
               'named_self': _Source.named_self, 'receiver': _Source.load,
               'unbound': _Source.load, 'bound': owner.load,
               'decorated': _Source.decorated, 'variadic': _Source.variadic,
               'variadic_bound': owner.variadic, 'positional': _Source.positional,
               'positional_bound': owner.positional, 'selective_wrapper': selective}
    _Source.calls = 0
    if form.startswith('native_file'):
        method = HistoricalData().get_any_file if form.endswith('bound') else HistoricalData.get_any_file
        params = {'file_path_or_url': str(_FILE), 'row_count_limit': 3}
        expected = _normalized(raw).tail(3)
    elif form.startswith('native_spot'):
        method = HistoricalData().get_spot_klines if form.endswith('bound') else HistoricalData.get_spot_klines
        params = {'kline_size': 900, 'row_count_limit': 3}
        quarter = pl.read_parquet(_QUARTER)
        columns = ['datetime', 'open', 'high', 'low', 'close', 'mean', 'std', 'volume',
                   'maker_ratio', 'no_of_trades', 'open_liquidity', 'high_liquidity',
                   'low_liquidity', 'close_liquidity', 'liquidity_sum', 'maker_volume', 'maker_liquidity']
        expected = _normalized(quarter).select(columns).tail(3)
    else:
        method = methods[form]
        if form == 'parquet_partial':
            params = {'n_rows': 3}
        if form == 'receiver':
            params['receiver'] = owner
            expected = raw.head(1)
        elif form == 'named_self':
            params['self'] = raw
        elif form.startswith('variadic'):
            params = {'args': 3}
        elif form.startswith('positional'):
            params = {'self': 3}
    manifest = manifest_type()
    if form.startswith('native_spot'):
        with patch('limen.data.historical_data._read_any_file', return_value=quarter) as transport:
            result = _fetch(manifest, api, method, params)
        transport.assert_called_once_with('vaquum/binance_btcusdt_15m_klines', has_header=True)
    else:
        result = _fetch(manifest, api, method, params)
    assert_frame_equal(result, expected)
    config = getattr(manifest, f'{api}_source_config')
    assert config.method is method
    assert config.params is params
    key = ('row_count_limit' if form.startswith('native') else 'n_rows' if form == 'parquet_partial'
           else 'args' if form.startswith('variadic') else 'self' if form.startswith('positional') else 'rows')
    assert params[key] == 3
    if form in ('local', 'selective_wrapper'):
        assert calls == [3]
        assert result is not raw
    if form in ('decorated', 'variadic', 'variadic_bound', 'positional', 'positional_bound'):
        assert _Source.calls == 1
    if form in ('bound', 'receiver', 'variadic_bound', 'positional_bound'):
        assert_frame_equal(owner.data, raw.head(3))


@pytest.mark.parametrize('manifest_type', _TYPES)
@pytest.mark.parametrize('api', _APIS)
def test_manifest_callable_argument_forwarding(manifest_type, api):
    frame = _recorded()
    marker = object()
    calls = []

    def configured(*, rows, token):
        calls.append((rows, token))
        return frame

    source = partial(configured, rows=1, token=marker)
    params = {'rows': 3}
    assert _fetch(manifest_type(), api, source, params) is frame
    assert calls == [(3, marker)]
    assert source.func is configured
    assert source.args == ()
    assert source.keywords == {'rows': 1, 'token': marker}
    assert params == {'rows': 3}
    failure = RuntimeError('recorded source unavailable')

    @wraps(HistoricalData.get_any_file)
    def raising(**kwargs):
        calls.append(kwargs)
        raise failure

    with pytest.raises(RuntimeError) as caught:
        _fetch(manifest_type(), api, raising, {'token': marker})
    assert caught.value is failure
    assert calls == [(3, marker), {'token': marker}]


@pytest.mark.parametrize('manifest_type', _TYPES)
@pytest.mark.parametrize('form', ('partial', 'instance', 'function'))
def test_manifest_callable_override_validation(manifest_type, form):
    method = {'partial': partial(_recorded, rows=2), 'instance': _Source(), 'function': _recorded}[form]
    manifest = manifest_type().set_data_source(method, {'rows': 2}).set_test_data_source(method, {'rows': 1})
    changed = manifest.with_params_override(rows=3)
    assert_frame_equal(changed.fetch_data(), _recorded(rows=3))
    assert_frame_equal(changed.fetch_test_data(), _recorded(rows=1))
    assert manifest.data_source_config.params == {'rows': 2}
    assert changed.data_source_config.params == {'rows': 3}
    assert changed.data_source_config.params is not manifest.data_source_config.params
    assert changed.test_data_source_config is not manifest.test_data_source_config
    accepted = sorted(set(inspect.signature(method).parameters) - {'self', 'cls'})
    name = getattr(method, '__name__', type(method).__name__)
    with pytest.raises(ValueError) as caught:
        manifest.with_params_override(unknown=3)
    assert str(caught.value) == f"Manifest Unknown data source params: ['unknown']. Accepted by {name}: {accepted}"


@pytest.mark.parametrize('manifest_type', _TYPES)
@pytest.mark.parametrize('api', _APIS)
def test_manifest_source_validation(manifest_type, api, tmp_path):
    with pytest.raises(ValueError, match=f'Manifest No {"test " if api == "test_data" else ""}data source configured'):
        getattr(manifest_type(), f'fetch_{api}')()
    with pytest.raises(FileNotFoundError):
        _fetch(manifest_type(), api, HistoricalData.get_any_file,
               {'file_path_or_url': str(tmp_path / 'missing.parquet')})
    with patch('limen.data.historical_data._read_any_file') as transport:
        for params in ({'kline_size': 0}, {'kline_size': 900, 'row_count_limit': 0},
                       {'kline_size': 900, 'row_count_limit': 3, 'n_rows': 4},
                       {'kline_size': 900, 'row_count_limit': 3,
                        'start_date_limit': '2025-01-01', 'end_date_limit': '2025-01-02'}):
            with pytest.raises(ValueError):
                _fetch(manifest_type(), api, HistoricalData.get_spot_klines, params)
        with pytest.raises(TypeError, match='HistoricalData kline_size must be an int'):
            _fetch(manifest_type(), api, HistoricalData.get_spot_klines, {'kline_size': True})
        transport.assert_not_called()
    failure = OSError('recorded transport failed')
    with patch('limen.data.historical_data._read_any_file', side_effect=failure) as transport:
        with pytest.raises(OSError) as caught:
            _fetch(manifest_type(), api, HistoricalData.get_spot_klines, {'kline_size': 900})
        assert caught.value is failure
        transport.assert_called_once()


@pytest.mark.parametrize('manifest_type', _TYPES)
def test_manifest_source_holdout_rejection(manifest_type, tmp_path):
    root = _project(tmp_path / 'project')
    _configuration(root)
    reservation = reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    try:
        manifest = manifest_type().set_data_source(HistoricalData.get_any_file, {'file_path_or_url': str(_FILE.resolve())})
        _holdout_research.bind_manifest(manifest, root)
        expected = _normalized(pl.read_parquet(_FILE)).head(96)
        assert_frame_equal(manifest.fetch_data(), expected)
        assert expected.height == 96

        @wraps(HistoricalData.get_any_file)
        def copied(**params):
            raise AssertionError('wrapped loader must not run')

        for method in (partial(HistoricalData().get_any_file), copied):
            manifest = manifest_type().set_data_source(method, {'file_path_or_url': str(_FILE.resolve())})
            _holdout_research.bind_manifest(manifest, root)
            with patch.object(DataSourceResolver, 'resolve', side_effect=AssertionError('resolver must not run')) as resolver, \
                    patch('limen.data.historical_data._read_any_file', side_effect=AssertionError('loader must not run')) as loader:
                with pytest.raises(ValueError, match='Holdout source must be native default spot klines or a local Parquet or CSV file'):
                    manifest.fetch_data()
                resolver.assert_not_called()
                loader.assert_not_called()
    finally:
        release_holdout(root, reservation)


@pytest.mark.parametrize('manifest_type', _TYPES)
@pytest.mark.parametrize('api', _APIS)
def test_manifest_callable_boundaries(manifest_type, api):
    for method in (_Source, object(), None):
        with pytest.raises(ValueError) as caught:
            _fetch(manifest_type(), api, method, {})
        assert str(caught.value) == f'DataSourceResolver Unsupported callable type: {type(method)}'

    def receiver_required(receiver, *, rows):
        return receiver.head(rows)

    with pytest.raises(TypeError, match='receiver'):
        _fetch(manifest_type(), api, partial(receiver_required), {'rows': 3})
    assert_frame_equal(_fetch(manifest_type(), api, partial(receiver_required, _recorded()), {'rows': 2}), _recorded(rows=2))
    with pytest.raises(ValueError, match="instance does not have 'data' attribute"):
        _fetch(manifest_type(), api, _EmptySource.load, {})
    with pytest.raises(TypeError, match='required'):
        _fetch(manifest_type(), api, _NeedsArgument.load, {})

    class MissingData:
        def load(self):
            return _recorded()

    assert_frame_equal(_fetch(manifest_type(), api, MissingData().load, {}), _recorded())
    for value in (None, _recorded().to_dicts()):
        calls = Mock(return_value=value)

        @wraps(_Source.load)
        def wrapped(_calls=calls, **params):
            return _calls(**params)

        with pytest.raises(TypeError, match=r'materialized polars\.DataFrame') as caught:
            _fetch(manifest_type(), api, wrapped, {'rows': 3})
        assert type(value).__name__ in str(caught.value)
        calls.assert_called_once_with(rows=3)

    failure = RuntimeError('source cannot be copied')

    class Uncopyable(_Source):
        def __deepcopy__(self, memo):
            raise failure

    manifest = manifest_type().set_data_source(Uncopyable())
    with pytest.raises(RuntimeError) as caught:
        manifest.with_params_override(rows=3)
    assert caught.value is failure

    signature_failure = RuntimeError('source signature unavailable')

    class Malformed(_Source):
        @property
        def __signature__(self):
            raise signature_failure

    manifest = manifest_type().set_data_source(Malformed())
    with pytest.raises(RuntimeError) as caught:
        manifest.with_params_override(rows=3)
    assert caught.value is signature_failure
    malformed = manifest_type().set_data_source(partial(receiver_required, _recorded(), _recorded()))
    with pytest.raises(ValueError, match='incorrect arguments'):
        malformed.with_params_override(rows=3)
