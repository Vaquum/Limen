from __future__ import annotations

import json
import pickle
import multiprocessing
import copy
import os
import gzip
import shutil
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Mapping
from pathlib import Path
from dataclasses import dataclass, fields, is_dataclass
from functools import partial, wraps

import pytest
import polars as pl

from limen._parallel_contract import ParallelRoundContract, declare_parallel, deterministic_stage, seeded_stage
from limen.experiment._parallel_encoding import decode, digest, encode, source_digest
from limen.experiment._parallel_plan import ScalerFactory, clone_manifest, freeze_manifest
from limen.experiment._parallel_resources import sklearn_stage, validate_resources, xgboost_stage, validate_domain
from limen.experiment._parallel_worker import startup_limits
from limen.experiment._parallel_evidence import record_context, validate_record, wire_record, parameter_projection, parameter_matches
from limen.experiment import _round_storage as storage
from limen.scalers.robust_scaler import RobustScaler
from limen.targets.random_binary import RandomBinaryTarget
from limen.yaml import parse
from limen.yaml.compiler import build_manifest
from tests.test_parallel_rounds import CallerRollingBase


def _configuration() -> dict[str, object]:
    config, errors = parse(Path('limen/yaml/templates/logreg_binary.yaml').read_text())
    assert not errors
    return config


def _parameters():
    return {'n_jobs': 1, 'random_state': 42, 'solver': 'liblinear', 'scaler_type': 'robust',
            'roc_period': 1, 'frac_diff_d': 0.0, 'q': 0.5, 'shift': -1,
            'cal_method': 'sigmoid', 'threshold_min': 0.25, 'threshold_max': 0.65,
            'threshold_step': 0.05, 'window': 2, 'range': (0.1, 0.9)}


def _caller_feature(data, window=2):
    return data.with_columns(pl.col('close').rolling_mean(window).alias('caller_mean'))


def _feature_decorator(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)
    return wrapper


@_feature_decorator
def _wrapped_feature(data, window=2):
    return _caller_feature(data, window)


def _kwargs_feature(data, **kwargs):
    return _caller_feature(data, kwargs.get('window', 2))


def _mapping_feature(data, options):
    return _caller_feature(data, options['missing'] + 3)


def _nested_default_feature(data, components):
    return components[0](data)


def _keyword_feature(data, *, window=2):
    return _caller_feature(data, window)


def _mutable_feature(data, windows=None):
    window = windows[0]
    windows[0] += 1
    return _caller_feature(data, window)


_mutable_feature.__defaults__ = ([2],)


def _configured_feature(window):
    def feature(data):
        return _caller_feature(data, window)
    feature.__qualname__ = '_closed_feature'
    return feature


_closed_feature = _configured_feature(2)


def _counter_features():
    window = 2

    def increment(data):
        nonlocal window
        window += 1
        return _caller_feature(data, window)

    def read(data):
        return _caller_feature(data, window)

    increment.__qualname__, read.__qualname__ = '_cell_increment', '_cell_read'
    return increment, read


_cell_increment, _cell_read = _counter_features()



def _caller_validator(params):
    if type(params.get('window')) is not int or params['window'] < 1:
        raise ValueError('positive integer window required')


def _configured_validator(minimum):
    def validate(params, *, maximum=4):
        if not minimum <= params['window'] <= maximum:
            raise ValueError('window outside captured bounds')
    validate.__qualname__ = '_closed_validator'
    return validate


_closed_validator = _configured_validator(2)


@dataclass
class CallerRolling(CallerRollingBase):
    window: int

    @classmethod
    def class_transform(cls, data, window=3):
        return _caller_feature(data, window)

    def __call__(self, data):
        return self.transform(data)


class InheritedCallable(CallerRollingBase):
    def __init__(self, window):
        self.window = window


class AllocatedRolling(CallerRollingBase):
    def __new__(cls, *, seed=None):
        instance = super().__new__(cls)
        instance.window = 3 if seed == 42 else 1
        return instance

    def __call__(self, data):
        return self.transform(data)


class DualSeedRolling(AllocatedRolling):
    def __new__(cls, *, seed=None):
        instance = object.__new__(cls)
        instance.window = 3 if seed == 42 else 1
        return instance

    def __init__(self, *, seed=42):
        self.initializer_seed = seed


class InheritedCallOwnTransform(InheritedCallable):
    def transform(self, data):
        return _caller_feature(data, self.window)


@dataclass
class InheritedDataclass(CallerRollingBase):
    window: int


@dataclass
class NormalizedRolling(CallerRollingBase):
    window: int

    def __post_init__(self):
        self.window += 1


@dataclass
class ShadowedSeedRolling:
    random_state: int | None = 42

    def __call__(self, data, random_state=None):
        return _caller_feature(data, window=random.Random(random_state).randint(1, 3))


def _closed_implementation(windows):
    def implementation(self, data):
        windows[0] += 1
        return _caller_feature(data, windows[0])
    implementation.__qualname__ = 'ClosureRolling.__call__'
    return implementation


class ClosureRolling:
    __call__ = _closed_implementation([2])


@dataclass
class FunctionRolling:
    function: object

    def __call__(self, data):
        return self.function(data)


@dataclass
class RollingCounter:
    window: int = 2


@dataclass
class CallableCounter(RollingCounter):
    def __call__(self):
        return self.window


@dataclass
class DefaultRollingFactory:
    window: int = 3

    def __call__(self, *, n_jobs=1):
        return CallerRolling(self.window)


def _object_default_feature(data, counter=None):
    counter.window += 1
    return _caller_feature(data, counter.window)


def _callable_default_feature(data, counter=None):
    counter.window += 1
    return counter()(data)


@dataclass
class AliasedRolling:
    left: list[int]
    right: list[int]

    def __call__(self, data):
        assert self.left is self.right
        self.left[0] += 1
        return _caller_feature(data, self.right[0])


@dataclass
class KeyedRolling:
    weights: dict[int, list[int]]

    def __call__(self, data):
        assert self.weights[0] is self.weights[1]
        self.weights[0][0] += 1
        return _caller_feature(data, self.weights[1][0])


@dataclass
class DefaultMappingRolling:
    weights: defaultdict[str, int]

    def __call__(self, data):
        self.weights['calls'] += 1
        return _caller_feature(data, self.weights['calls'] + 2)


def _target_constructor(window):
    def implementation(self, train_data, target_name):
        self.target_name, self.window = target_name, window
    implementation.__qualname__ = 'ClosureTarget.__init__'
    return implementation


def _target_transform(window):
    def implementation(self, data):
        return data.with_columns((pl.col('close') > pl.col('close').shift(self.window + window)).cast(pl.Int8).alias(self.target_name))
    implementation.__qualname__ = 'ClosureTarget.transform'
    return implementation


class ClosureTarget:
    __init__ = _target_constructor(1)
    transform = _target_transform(1)


def _target_allocator(window):
    def implementation(cls, train_data, target_name):
        instance = object.__new__(cls)
        instance.window = window
        return instance
    implementation.__qualname__ = 'AllocatedTarget.__new__'
    return implementation


class AllocatedTarget:
    __new__ = staticmethod(_target_allocator(1))

    def __init__(self, train_data, target_name):
        self.target_name = target_name

    def transform(self, data):
        return data.with_columns((pl.col('close') > pl.col('close').shift(self.window)).cast(pl.Int8).alias(self.target_name))


class DualDefaultTarget(AllocatedTarget):
    def __new__(cls, train_data, target_name, window=2):
        instance = object.__new__(cls)
        instance.window = window
        return instance

    def __init__(self, train_data, target_name, window=3):
        self.window += window
        self.target_name = target_name


def _method_only_transform(owner, data):
    return _caller_feature(data, owner.window)


class MethodOnlyRolling:
    def __init__(self, window):
        self.window = window

    transform = _method_only_transform


def _unused_call(secret):
    def implementation(self, data):
        raise AssertionError(f'Unused call must not execute: {secret}')
    return implementation


class OpaqueUnusedCallRolling(MethodOnlyRolling):
    __call__ = _unused_call(object())


class SignatureSeedRolling(ShadowedSeedRolling):
    def __call__(self, data, random_state=42):
        return super().__call__(data, random_state)


def _seed_first_feature(random_state, data):
    return ShadowedSeedRolling()(data, random_state)


class OpaqueFactory(tuple):
    def __call__(self, *, window=3):
        return CallerRolling(self[0] + window)


class MappingRolling(Mapping[str, int]):
    def __init__(self, window):
        self.window = window

    def __getitem__(self, key):
        if key != 'window':
            raise KeyError(key)
        return self.window

    def __iter__(self):
        return iter(('window',))

    def __len__(self):
        return 1

    def __call__(self, data):
        return _caller_feature(data, window=self.window)


class OrderedRolling:
    def __init__(self, window):
        self.window = window
        self.steps = {'offset': 1, 'multiply': 2}

    def __call__(self, data):
        result = _caller_feature(data, self.window)
        expression = pl.col('caller_mean')
        for name, amount in self.steps.items():
            expression = expression + amount if name == 'offset' else expression * amount
        return result.with_columns(expression.alias('caller_mean'))


class DictRolling(dict):
    def __call__(self, data):
        return _caller_feature(data, window=self['window'])


class ListRolling(list):
    def __call__(self, data):
        return _caller_feature(data, window=self[0])


def _dict_validator(params):
    _caller_validator({'window': dict(params['__parallel_native_state__'])['window']})


def _list_validator(params):
    _caller_validator({'window': params['__parallel_native_state__'][0]})


class OpaqueSequenceRolling(tuple):
    def __call__(self, data):
        return _caller_feature(data, window=self[0])


class CallerPartial(partial):
    __slots__ = ('bias',)

    def __call__(self, data):
        return super().__call__(data).with_columns((pl.col('caller_mean') * self.multiplier + self.bias).alias('caller_mean'))


def _partial_stage():
    value = CallerPartial(_caller_feature, window=3)
    value.multiplier, value.bias = 2, 1
    return value


def _partial_validator(params):
    if params.get('multiplier') != 2 or type(params.get('bias')) is not int:
        raise ValueError('partial configuration required')


class SlottedRollingBase:
    __slots__ = ('window',)


class SlottedRolling(SlottedRollingBase):
    __slots__ = ('__dict__', 'offset', 'unused')

    def __init__(self, window, offset=1):
        self.window, self.offset = window, offset
        self.label = 'caller'

    def __call__(self, data):
        return _caller_feature(data, window=self.window + self.offset)


def _auxiliary_factory(*, window=3):
    return CallerRolling(window)


def _factory_architecture(data, auxiliary_factory=_auxiliary_factory, auxiliary_params=None):
    return auxiliary_factory(**(auxiliary_params or {}))(data)


def _optional_factory_architecture(data, auxiliary_factory=None, auxiliary_params=None):
    return data if auxiliary_factory is None else auxiliary_factory(**(auxiliary_params or {}))(data)


def _other_auxiliary_factory(*, window=3):
    return CallerRolling(window)


@dataclass
class CallerExtension:
    random_state: int = 42

    def __call__(self, *, data_dict, split_data, round_params, fitted_params):
        return data_dict | {'caller_close': split_data[0]['close'].to_list(),
                            'caller_seed': self.random_state, 'caller_round_params': round_params}


def _extension_probe(plan, params):
    from tests.test_parallel_rounds import recorded_source
    from limen.experiment._parallel_plan import clone_manifest
    validate_resources(plan, params)
    result = clone_manifest(plan.manifest).prepare_data(recorded_source(), params)
    return {key: result[key] for key in ('caller_close', 'caller_seed', 'caller_round_params')}, os.getpid()


class CallerTarget(RandomBinaryTarget):
    '''Importable caller target using the native seeded noise benchmark.'''


class TransformSeedTarget(CallerTarget):
    def transform(self, data, random_state=None):
        return RandomBinaryTarget(data, self.target_name, random_state).transform(data)


def _caller_scaler(data, n_jobs=2):
    return data.with_columns((pl.col('close') * n_jobs).alias('close'))


@dataclass
class CallerScalerTransform:
    n_jobs: int = 2

    def __call__(self, data):
        return _caller_scaler(data, self.n_jobs)


def _prepared_probe(plan, params):
    from tests.test_parallel_rounds import recorded_source
    from limen.experiment._parallel_plan import clone_manifest
    validate_resources(plan, params)
    data = clone_manifest(plan.manifest).prepare_data(recorded_source(), params)
    return {key: data[key].to_dict(as_series=False) for key in ('x_train', 'x_val', 'x_test')}, os.getpid()


def _target_probe(plan, params):
    from tests.test_parallel_rounds import recorded_source
    validate_resources(plan, params)
    data = clone_manifest(plan.manifest).prepare_data(recorded_source(), params)
    return {key: data[key].to_list() for key in ('y_train', 'y_val', 'y_test')}


def _carried_probe(plan, params):
    validate_resources(plan, params)
    from tests.test_parallel_rounds import recorded_source
    data = recorded_source().head(24).select('close')
    from limen.experiment._parallel_plan import clone_manifest
    entries = clone_manifest(plan.manifest).feature_transforms
    start = -2 if all(getattr(entry.func, '__qualname__', '') == '_mutable_feature' for entry in entries[-2:]) else -1
    for entry in entries[start:]:
        data = entry.func(data, **entry.params)
    return data['caller_mean'].to_list(), os.getpid()


def _repeated_carried_probe(plan, params):
    before = digest(plan)
    first, pid = _carried_probe(plan, params)
    second, _ = _carried_probe(plan, params)
    assert first == second and digest(plan) == before
    return first, pid


def _shared_carried_probe(plan, identity='function'):
    from tests.test_parallel_rounds import recorded_source
    before, rounds = digest(plan), []
    for _ in range(2):
        entries = clone_manifest(plan.manifest).feature_transforms[-2:]
        if identity == 'cell':
            assert entries[0].func.__closure__[0] is entries[1].func.__closure__[0]
        elif identity == 'owner':
            assert entries[0].func.__self__ is entries[1].func
        else:
            assert entries[0].func.function is entries[1].func.function
        data, outputs = recorded_source().head(24).select('close'), []
        for entry in entries:
            data = entry.func(data)
            outputs.append(data['caller_mean'].to_list())
        rounds.append(outputs)
    assert rounds[0] == rounds[1] and digest(plan) == before
    return rounds[0], os.getpid()


def _frozen_probe(plan, params):
    from limen.experiment import _parallel_identity as identities
    from limen.experiment._parallel_plan import plan_digests
    def no_discovery():
        raise AssertionError('worker repeated parent distribution discovery')
    with pytest.MonkeyPatch.context() as patch, identities.distribution_providers(plan.distribution_providers):
        patch.setattr(identities, 'package_distributions', no_discovery)
        assert plan_digests(plan.manifest, plan.stages) == (plan.pipeline_digest, plan.resource_digest)
        version = identities.metadata.version
        patch.setattr(identities.metadata, 'version', lambda name: version(name) + '-changed')
        assert plan_digests(plan.manifest, plan.stages) != (plan.pipeline_digest, plan.resource_digest)
    validate_resources(plan, params)
    from tests.test_parallel_rounds import recorded_source
    data = recorded_source().head(24).select('close')
    computations = plan.manifest.scaler[0]
    from limen.experiment import manifest_core
    resolved = manifest_core.__dict__['_resolve_params'](computations[0][2], params)
    scaler = computations[0][1](data, **resolved)
    transformed = scaler.transform(data)
    caller = next((stage for stage in plan.stages if getattr(stage.component, '__code__', None) is _caller_feature.__code__), None)
    actual = None if caller is None else caller.component(data, window=params['window'])['caller_mean'].to_list()
    return (plan.pipeline_digest, isinstance(computations[0][1], ScalerFactory),
            hasattr(_caller_feature, '__parallel_round_contract__'), os.getpid(),
            Path(__file__).resolve().as_posix(), transformed.to_dict(as_series=False), actual)


def test_frozen_manifest_round_plan() -> None:
    defaults = defaultdict(int)
    transported = decode(encode([defaults, defaults]))
    assert transported[0] is transported[1] and type(transported[0]) is defaultdict
    assert transported[0].default_factory is int and transported[0]['missing'] == 0
    assert not defaults
    owner = InheritedCallOwnTransform(3)
    pair = [owner.transform, owner]
    encoded_pair = encode(pair)
    transported = decode(encoded_pair)
    assert transported[0].__self__ is transported[1]
    from limen.experiment import _parallel_encoding as codec
    original_identity = codec.module_identity
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(codec, 'module_identity', lambda name: [*original_identity(name), 'changed'] if name == 'tests.test_parallel_rounds' else original_identity(name))
        assert encode(pair) != encoded_pair
        with pytest.raises(ValueError, match='callable implementation changed'):
            decode(encoded_pair)
    shared = [2]
    aliased = {'left': (shared,), 'right': (shared,)}
    transported = decode(encode(aliased))
    assert transported['left'][0] is transported['right'][0]
    assert parameter_projection(encode(aliased)) == {'left': [[2]], 'right': [[2]]}
    for mapping in ({1: shared, 0: shared}, {(1, 'key'): shared, (0, 'key'): shared}):
        transported = decode(encode(mapping))
        values = list(transported.values())
        assert values[0] is values[1]
        assert digest(mapping) == digest(dict(reversed(list(mapping.items()))))
    for invalid in (['ref', 0], ['ref', -1], ['list', [['ref', 0]]], [[]]):
        with pytest.raises(ValueError):
            decode(invalid)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_object_default_feature, '__defaults__', (_object_default_feature,))
        with pytest.raises(ValueError, match='cycles'):
            encode(_object_default_feature)
    config = _configuration()
    original = build_manifest(config)
    plan = freeze_manifest(original, config)
    assert plan.manifest.architecture_function is not original.architecture_function
    assert clone_manifest(plan.manifest).architecture_function is original.architecture_function
    assert clone_manifest(plan.manifest).prediction_calibration_config.threshold_func is original.prediction_calibration_config.threshold_func
    assert plan.pipeline_digest == freeze_manifest(build_manifest(config), config).pipeline_digest
    assert len(pickle.dumps(plan)) <= 65536
    from limen.experiment import _parallel_identity as identities
    def no_discovery():
        raise AssertionError('plan transport repeated provider discovery')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(identities, 'package_distributions', no_discovery)
        restore, arguments = plan.__reduce__()
        transported = restore(*arguments)
        assert (transported.pipeline_digest, transported.resource_digest) == (plan.pipeline_digest, plan.resource_digest)
    declare_parallel(_caller_feature, ParallelRoundContract(_caller_validator, ('window',)))
    native = copy.deepcopy(plan.manifest)
    native.add_feature(_caller_feature, window='{window}')
    with startup_limits(), ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        declare_parallel(_caller_feature, ParallelRoundContract(_closed_validator, ('window',)))
        validator_manifest = copy.deepcopy(plan.manifest).add_feature(_caller_feature, window=3)
        validator_plan = freeze_manifest(validator_manifest, None)
        validator_digest = digest(validator_plan)
        for attribute, owner, changed in [('__kwdefaults__', _closed_validator, {'maximum': 1}),
                                          ('cell_contents', _closed_validator.__closure__[0], 5)]:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(owner, attribute, changed)
                invalid_validator_plan = freeze_manifest(validator_manifest, None)
                assert (invalid_validator_plan.pipeline_digest, invalid_validator_plan.resource_digest) != (validator_plan.pipeline_digest, validator_plan.resource_digest)
                with pytest.raises(ValueError, match='window outside captured bounds'):
                    validate_resources(invalid_validator_plan, _parameters())
                assert digest(validator_plan) == validator_digest
                for _ in range(2):
                    actual = pool.submit(_frozen_probe, validator_plan, _parameters() | {'window': 3}).result(timeout=30)
                    expected = _frozen_probe(validator_plan, _parameters() | {'window': 3})
                    assert actual[:2] == expected[:2] and actual[2] is False and expected[2] is True
                    assert actual[5:] == expected[5:] and actual[3] != os.getpid() and digest(validator_plan) == validator_digest
        declare_parallel(_caller_feature, ParallelRoundContract(_caller_validator, ('window',)))
        cases = [(native, _parameters()), (native, _parameters() | {'scaler_type': 'rank_gauss'})]
        fixed = copy.deepcopy(native).set_scaler(RobustScaler, extra_params={'quantile_range': (0.1, 0.9)})
        dynamic = copy.deepcopy(native).set_scaler_from_params(extra_params={'quantile_range': '{range}'})
        renamed = copy.deepcopy(native).set_scaler_from_params(param_name='scale_choice')
        cases += [(fixed, _parameters()), (dynamic, _parameters()),
                  (renamed, {key: value for key, value in _parameters().items() if key != 'scaler_type'} | {'scale_choice': 'robust'})]
        for manifest, params in cases:
            frozen = freeze_manifest(manifest, None)
            actual = pool.submit(_frozen_probe, frozen, params).result(timeout=30)
            expected = _frozen_probe(frozen, params)
            assert actual[:3] == (frozen.pipeline_digest, True, False)
            assert actual[3] != os.getpid() and actual[4] == Path(__file__).resolve().as_posix()
            assert actual[5:] == expected[5:]
        for component in (_wrapped_feature, MethodOnlyRolling(3).transform, OpaqueUnusedCallRolling(3).transform, partial(_kwargs_feature, n_jobs=1, window=3), partial(_caller_feature, window=3), partial(CallerRolling(3)), partial(CallerRolling(3).transform), CallerRolling(3).transform, CallerRolling(3), CallerRolling.class_transform, SlottedRolling(2), InheritedCallable(3), InheritedDataclass(3), NormalizedRolling(2), MappingRolling(3), OrderedRolling(3), DictRolling(window=3), ListRolling([3]), _partial_stage()):
            imports = ('tests.test_parallel_rounds',) if isinstance(component, CallerRolling) else ()
            contract = ParallelRoundContract(_partial_validator, ('multiplier', 'bias')) if isinstance(component, CallerPartial) else ParallelRoundContract(_caller_validator, ('window',), imports)
            if isinstance(component, (DictRolling, ListRolling)):
                contract = ParallelRoundContract(_dict_validator if isinstance(component, DictRolling) else _list_validator, ('__parallel_native_state__',))
            declare_parallel(component, contract)
            carried = copy.deepcopy(plan.manifest).add_feature(component)
            frozen = freeze_manifest(carried, None)
            assert digest(decode(encode(frozen))) == digest(frozen)
            if isinstance(component, MappingRolling):
                restored = decode(encode(component))
                assert type(restored) is MappingRolling and restored.window == 3 and callable(restored)
                assert dict(restored) == {'window': 3}
                validate_resources(frozen, _parameters())
            if isinstance(component, OrderedRolling):
                restored = decode(encode(component))
                assert list(restored.steps) == ['offset', 'multiply']
                restored.steps = dict(reversed(list(restored.steps.items())))
                declare_parallel(restored, contract)
                reordered = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(restored), None)
                assert reordered.pipeline_digest != frozen.pipeline_digest and reordered.resource_digest != frozen.resource_digest
            if isinstance(component, (DictRolling, ListRolling)):
                restored = decode(encode(component))
                assert type(restored) is type(component) and restored == component and callable(restored)
                restored['window' if isinstance(restored, dict) else 0] = 4
                assert digest(restored) != digest(component)
                validate_resources(frozen, _parameters())
                invalid = copy.deepcopy(component)
                invalid['window' if isinstance(invalid, dict) else 0] = 0
                rejected = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(invalid), None)
                with pytest.raises(ValueError, match=r'feature_transforms.*positive integer window'):
                    validate_domain(rejected, {key: [value] for key, value in _parameters().items()}, None)
            if isinstance(component, NormalizedRolling):
                assert component.window == decode(encode(component)).window == 3
            if isinstance(component, CallerPartial):
                restored = decode(encode(component))
                assert type(restored) is CallerPartial and restored.func.__code__ is _caller_feature.__code__
                assert restored.func.__defaults__ == _caller_feature.__defaults__
                assert restored.keywords == {'window': 3} and (restored.multiplier, restored.bias) == (2, 1)
                restored.bias += 1
                declare_parallel(restored, contract)
                altered = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(restored), None)
                assert altered.pipeline_digest != frozen.pipeline_digest and altered.resource_digest != frozen.resource_digest
            if isinstance(component, SlottedRolling):
                validate_resources(frozen, _parameters())
                changed = copy.deepcopy(component)
                changed.window += 1
                changed_plan = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(changed), None)
                assert changed_plan.pipeline_digest != frozen.pipeline_digest
                assert changed_plan.resource_digest != frozen.resource_digest
                restored = decode(encode(component))
                assert (restored.window, restored.offset, restored.label) == (2, 1, 'caller')
                assert not hasattr(restored, 'unused')
            if getattr(component, '__name__', None) == 'transform' or isinstance(component, (InheritedCallable, InheritedDataclass, NormalizedRolling)):
                _assert_inherited_identity(frozen)
            actual, pid = pool.submit(_carried_probe, frozen, _parameters()).result(timeout=30)
            expected, _ = _carried_probe(frozen, _parameters())
            assert actual == expected and pid != os.getpid()
        for component, attribute, defaults in ((_caller_feature, '__defaults__', (3,)), (_keyword_feature, '__kwdefaults__', {'window': 3})):
            declare_parallel(component, ParallelRoundContract(_caller_validator, ('window',)))
            baseline = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(component), None)
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(component, attribute, defaults)
                frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(component), None)
            assert frozen.pipeline_digest != baseline.pipeline_digest and frozen.resource_digest != baseline.resource_digest
            validate_resources(frozen, _parameters())
            actual, pid = pool.submit(_carried_probe, frozen, _parameters()).result(timeout=30)
            expected, _ = _carried_probe(frozen, _parameters())
            assert actual == expected and pid != os.getpid()
            assert frozen.manifest.feature_transforms[-1].func.__defaults__ == ((3,) if attribute == '__defaults__' else None)
        declare_parallel(_mutable_feature, ParallelRoundContract(deterministic_stage))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_mutable_feature, '__defaults__', ([2],))
            frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(_mutable_feature).add_feature(_mutable_feature), None)
            actual, pid = pool.submit(_repeated_carried_probe, frozen, _parameters()).result(timeout=30)
            expected, _ = _repeated_carried_probe(frozen, _parameters())
            assert actual == expected and pid != os.getpid()
        declare_parallel(_closed_feature, ParallelRoundContract(deterministic_stage))
        frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(_closed_feature), None)
        before = digest(frozen)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_closed_feature.__closure__[0], 'cell_contents', 3)
            changed = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(_closed_feature), None)
            assert changed.pipeline_digest != frozen.pipeline_digest and changed.resource_digest != frozen.resource_digest
            actual, pid = pool.submit(_repeated_carried_probe, frozen, _parameters()).result(timeout=30)
            expected, _ = _repeated_carried_probe(frozen, _parameters())
            assert actual == expected and pid != os.getpid() and digest(frozen) == before
        component = ClosureRolling()
        declare_parallel(component, ParallelRoundContract(deterministic_stage))
        frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(component), None)
        actual, pid = pool.submit(_repeated_carried_probe, frozen, _parameters()).result(timeout=30)
        expected, _ = _repeated_carried_probe(frozen, _parameters())
        assert actual == expected and pid != os.getpid()
        from tests.test_parallel_rounds import recorded_source
        expected_values = _caller_feature(recorded_source().head(24).select('close'), 3)['caller_mean'].to_list()
        shared = [2]
        for component, window in ((FunctionRolling(_mutable_feature), 2), (FunctionRolling(ClosureRolling()), 3), (AliasedRolling(shared, shared), 3), (KeyedRolling({1: shared, 0: shared}), 3), (DefaultMappingRolling(defaultdict(int)), 3)):
            declare_parallel(component, ParallelRoundContract(deterministic_stage))
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(_mutable_feature, '__defaults__', ([2],))
                frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(component), None)
            actual, pid = pool.submit(_repeated_carried_probe, frozen, _parameters()).result(timeout=30)
            expected, _ = _repeated_carried_probe(frozen, _parameters())
            assert actual == expected == _caller_feature(recorded_source().head(24).select('close'), window)['caller_mean'].to_list() and pid != os.getpid()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_mutable_feature, '__defaults__', ([2],))
            first, second = FunctionRolling(_mutable_feature), FunctionRolling(_mutable_feature)
            for component in (first, second):
                declare_parallel(component, ParallelRoundContract(deterministic_stage))
            frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(first).add_feature(second), None)
        actual, pid = pool.submit(_shared_carried_probe, frozen).result(timeout=30)
        expected, _ = _shared_carried_probe(frozen)
        oracle = [_caller_feature(recorded_source().head(24).select('close'), window)['caller_mean'].to_list() for window in (2, 3)]
        assert actual == expected == oracle and pid != os.getpid()
        for identity, components in (('cell', (_cell_increment, _cell_read)), ('owner', (owner.transform, owner))):
            for component in components:
                declare_parallel(component, ParallelRoundContract(deterministic_stage))
            frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(components[0]).add_feature(components[1]), None)
            actual, pid = pool.submit(_shared_carried_probe, frozen, identity).result(timeout=30)
            expected, _ = _shared_carried_probe(frozen, identity)
            oracle = [_caller_feature(recorded_source().head(24).select('close'), 3)['caller_mean'].to_list()] * 2
            assert actual == expected == oracle and pid != os.getpid()
            if identity == 'owner':
                _assert_inherited_identity(frozen)
        assert _cell_increment.__closure__[0] is _cell_read.__closure__[0]
        assert _cell_increment.__closure__[0].cell_contents == 2
        declare_parallel(_mapping_feature, ParallelRoundContract(deterministic_stage))
        options = defaultdict(int)
        frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(_mapping_feature, options=options), None)
        assert frozen.manifest.feature_transforms[-1].params['options'].default_factory is int
        actual, pid = pool.submit(_repeated_carried_probe, frozen, _parameters()).result(timeout=30)
        expected, _ = _repeated_carried_probe(frozen, _parameters())
        assert actual == expected == expected_values and pid != os.getpid() and not options
        declare_parallel(_nested_default_feature, ParallelRoundContract(deterministic_stage))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_nested_default_feature, '__defaults__', ([ClosureRolling()],))
            nested = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(_nested_default_feature), None)
        actual, pid = pool.submit(_repeated_carried_probe, nested, _parameters()).result(timeout=30)
        expected, _ = _repeated_carried_probe(nested, _parameters())
        assert actual == expected == expected_values and pid != os.getpid()
        for feature, counter in ((_object_default_feature, RollingCounter()), (_object_default_feature, CallableCounter()),
                                 (_object_default_feature, DefaultRollingFactory(2)), (_callable_default_feature, DefaultRollingFactory(2))):
            declare_parallel(feature, ParallelRoundContract(deterministic_stage))
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(feature, '__defaults__', (counter,))
                frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(feature), None)
            actual, pid = pool.submit(_repeated_carried_probe, frozen, _parameters()).result(timeout=30)
            expected, _ = _repeated_carried_probe(frozen, _parameters())
            assert actual == expected == expected_values and pid != os.getpid() and counter.window == 2
        declare_parallel(ClosureTarget, ParallelRoundContract(deterministic_stage))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ClosureTarget.__init__.__closure__[0], 'cell_contents', 2)
            patch.setattr(ClosureTarget.transform.__closure__[0], 'cell_contents', 2)
            manifest = copy.deepcopy(plan.manifest).with_target_label('closed', ClosureTarget)
            native = manifest.prepare_data(recorded_source(), _parameters())
            expected = {key: native[key].to_list() for key in ('y_train', 'y_val', 'y_test')}
            frozen = freeze_manifest(manifest, None)
            before = digest(frozen)
        for _ in range(2):
            assert pool.submit(_target_probe, frozen, _parameters()).result(timeout=30) == _target_probe(frozen, _parameters()) == expected
        assert digest(frozen) == before
        declare_parallel(AllocatedTarget, ParallelRoundContract(deterministic_stage))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(AllocatedTarget.__new__.__closure__[0], 'cell_contents', 2)
            manifest = copy.deepcopy(plan.manifest).with_target_label('allocated', AllocatedTarget)
            native = manifest.prepare_data(recorded_source(), _parameters())
            expected = {key: native[key].to_list() for key in ('y_train', 'y_val', 'y_test')}
            frozen_allocator = freeze_manifest(manifest, None)
            before = digest(frozen_allocator)
            patch.setattr(AllocatedTarget.__new__.__closure__[0], 'cell_contents', 3)
            changed = freeze_manifest(manifest, None)
            assert changed.pipeline_digest != frozen_allocator.pipeline_digest and changed.resource_digest != frozen_allocator.resource_digest
        for _ in range(2):
            assert pool.submit(_target_probe, frozen_allocator, _parameters()).result(timeout=30) == _target_probe(frozen_allocator, _parameters()) == expected
        assert digest(frozen_allocator) == before and AllocatedTarget.__new__.__closure__[0].cell_contents == 1
        declare_parallel(DualDefaultTarget, ParallelRoundContract(deterministic_stage))
        manifest = copy.deepcopy(plan.manifest).with_target_label('dual', DualDefaultTarget)
        native = manifest.prepare_data(recorded_source(), _parameters())
        expected = {key: native[key].to_list() for key in ('y_train', 'y_val', 'y_test')}
        dual = freeze_manifest(manifest, None)
        before = digest(dual)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(DualDefaultTarget.__new__, '__defaults__', (4,))
            patch.setattr(DualDefaultTarget.__init__, '__defaults__', (4,))
            for _ in range(2):
                assert pool.submit(_target_probe, dual, _parameters()).result(timeout=30) == _target_probe(dual, _parameters()) == expected
        assert digest(dual) == before
        component = partial(_kwargs_feature, n_jobs=2)
        declare_parallel(component, ParallelRoundContract(deterministic_stage))
        frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(component), None)
        with pytest.raises(ValueError, match='effective n_jobs=1 required'):
            validate_resources(frozen, _parameters())
        with pytest.raises(ValueError, match='effective n_jobs=1 required'):
            validate_domain(frozen, {key: [value] for key, value in _parameters().items()}, None)
        component = CallerRolling(3)
        declare_parallel(component, ParallelRoundContract(_caller_validator, ('window',)))
        vars(component)['cached_result'] = []
        with pytest.raises(ValueError, match='undeclared runtime state'):
            freeze_manifest(copy.deepcopy(plan.manifest).add_feature(component), None)
        from tests.test_parallel_rounds import parallel_config
        declare_parallel(CallerTarget, ParallelRoundContract(seeded_stage, ('random_state',), ('numpy',)))
        target = build_manifest(parallel_config()).with_target_label('noise', CallerTarget, fit_params={'random_state': '{target_seed}'})
        frozen = freeze_manifest(target, None)
        params = _parameters() | {'target_seed': 42}
        assert pool.submit(_target_probe, frozen, params).result(timeout=30) == _target_probe(frozen, params)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ShadowedSeedRolling.__call__, '__defaults__', (42,))
            component = ShadowedSeedRolling(None)
            declare_parallel(component, ParallelRoundContract(seeded_stage, ('random_state',)))
            frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(component), None)
            before = digest(frozen)
        assert digest(frozen) == before
        validate_resources(frozen, params)
        actual, pid = pool.submit(_carried_probe, frozen, params).result(timeout=30)
        expected, _ = _carried_probe(frozen, params)
        assert actual == expected and pid != os.getpid()
        declare_parallel(TransformSeedTarget, ParallelRoundContract(seeded_stage, ('random_state',), ('numpy',)))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(TransformSeedTarget.transform, '__defaults__', (42,))
            manifest = build_manifest(parallel_config()).with_target_label('noise', TransformSeedTarget, fit_params={'random_state': 42})
            frozen = freeze_manifest(manifest, None)
            before = digest(frozen)
        assert digest(frozen) == before
        validate_resources(frozen, params)
        assert pool.submit(_target_probe, frozen, params).result(timeout=30) == _target_probe(frozen, params)
        shadowed = ShadowedSeedRolling()
        signature = SignatureSeedRolling(None)
        for component in (shadowed, signature):
            declare_parallel(component, ParallelRoundContract(seeded_stage, ('random_state',)))
        unsafe = copy.deepcopy(plan.manifest).add_feature(shadowed)
        rejected = freeze_manifest(unsafe, None)
        with pytest.raises(ValueError, match=r'feature_transforms.*fixed integer seed'):
            validate_resources(rejected, params)
        with pytest.raises(ValueError, match=r'feature_transforms.*fixed integer seed'):
            validate_domain(rejected, {key: [value] for key, value in params.items()}, None)
        for component, arguments in ((shadowed, {'random_state': 42}), (signature, {})):
            seeded = copy.deepcopy(plan.manifest).add_feature(component, **arguments)
            frozen = freeze_manifest(seeded, None)
            validate_domain(frozen, {key: [value] for key, value in params.items()}, None)
            actual, pid = pool.submit(_carried_probe, frozen, params).result(timeout=30)
            expected, _ = _carried_probe(frozen, params)
            assert actual == expected and pid != os.getpid()
        for seed in (42, None):
            component = partial(_seed_first_feature, seed)
            component.random_state = None if seed == 42 else 42
            declare_parallel(component, ParallelRoundContract(seeded_stage, ('random_state',)))
            frozen = freeze_manifest(copy.deepcopy(plan.manifest).add_feature(component), None)
            if seed is None:
                with pytest.raises(ValueError, match=r'feature_transforms.*fixed integer seed'):
                    validate_domain(frozen, {key: [value] for key, value in params.items()}, None)
            else:
                validate_resources(frozen, params)
                actual, pid = pool.submit(_carried_probe, frozen, params).result(timeout=30)
                expected, _ = _carried_probe(frozen, params)
                assert actual == expected and pid != os.getpid()
        aliased = copy.deepcopy(plan.manifest).add_feature(shadowed, random_state='{caller_seed}')
        with pytest.raises(ValueError, match=r'feature_transforms.*fixed integer seed'):
            validate_domain(freeze_manifest(aliased, None),
                            {key: [value] for key, value in params.items()} | {'caller_seed': [42, None]}, None)
        extension = CallerExtension()
        declare_parallel(extension, ParallelRoundContract(seeded_stage, ('random_state',)))
        extended = copy.deepcopy(target).add_to_data_dict(extension)
        frozen = freeze_manifest(extended, None)
        validate_domain(frozen, {key: [value] for key, value in params.items()}, None)
        actual, pid = pool.submit(_extension_probe, frozen, params).result(timeout=30)
        expected, _ = _extension_probe(frozen, params)
        assert actual == expected and actual['caller_seed'] == 42 and actual['caller_close'] and pid != os.getpid()
        _assert_transform_bindings(params, pool)
        invalid = copy.deepcopy(extended)
        invalid.data_dict_extension.random_state = None
        rejected = freeze_manifest(invalid, None)
        assert rejected.pipeline_digest != frozen.pipeline_digest and rejected.resource_digest != frozen.resource_digest
        for validate, arguments in ((validate_resources, (rejected, params)),
                                    (validate_domain, (rejected, {key: [value] for key, value in params.items()}, None))):
            with pytest.raises(ValueError, match='data_dict_extension: fixed integer seed'):
                validate(*arguments)
    opaque = OpaqueSequenceRolling((3,))
    declare_parallel(opaque, ParallelRoundContract(_caller_validator, ('window',)))
    with pytest.raises(TypeError, match=r'object\.__new__.*not safe'):
        freeze_manifest(copy.deepcopy(plan.manifest).add_feature(opaque), None)
    original.split_config = (2, 1, 1)
    assert plan.manifest.split_config != original.split_config
    with pytest.raises(ValueError, match='unchanged compiled YAML'):
        freeze_manifest(original, config)
    values = {'tuple': (1, 0.5, None), 'list': [1, 0.5, None], 'bool': True}
    assert decode(json.loads(json.dumps(encode(values)))) == values
    assert digest(values['tuple']) != digest(values['list'])
    assert digest(values) == digest(dict(reversed(list(values.items()))))
    _assert_plan_identity(plan)
    _assert_fresh_identity_snapshot(plan)
    _assert_selector_and_metric_identity(plan)


def _assert_transform_bindings(params, pool):
    from tests.test_parallel_rounds import parallel_config
    base = build_manifest(parallel_config())
    domain = {key: [value] for key, value in params.items()}
    declare_parallel(TransformSeedTarget, ParallelRoundContract(seeded_stage, ('random_state',), ('numpy',)))
    target = copy.deepcopy(base).with_target_label('noise', TransformSeedTarget, fit_params={'random_state': 42})
    with pytest.raises(ValueError, match=r'target\.transform: fixed integer seed'):
        validate_resources(freeze_manifest(target, None), params)
    target.with_target_label('noise', TransformSeedTarget, fit_params={'random_state': 42},
                             transform_params={'random_state': '{transform_seed}'})
    frozen = freeze_manifest(target, None)
    for candidates, context in ((domain | {'transform_seed': [42, None]}, None),
                                (domain | {'transform_seed': [42]}, {'transform_seed': None})):
        with pytest.raises(ValueError, match=r'target\.transform: fixed integer seed'):
            validate_domain(frozen, candidates, context)
    safe = params | {'transform_seed': 42}
    validate_domain(frozen, domain | {'transform_seed': [42]}, None)
    assert pool.submit(_target_probe, frozen, safe).result(timeout=30) == _target_probe(frozen, safe)
    for transform in (_caller_scaler, CallerScalerTransform()):
        declare_parallel(transform, ParallelRoundContract(deterministic_stage, ('n_jobs',)))
        scaler = copy.deepcopy(base)
        computations = scaler.scaler[0]
        scaler.scaler = (computations, transform, {})
        with pytest.raises(ValueError, match=r'scaler\.transform: effective n_jobs=1'):
            validate_resources(freeze_manifest(scaler, None), params)
        if transform is _caller_scaler:
            scaler.scaler = (computations, transform, {'n_jobs': '{scale_jobs}'})
            frozen = freeze_manifest(scaler, None)
            with pytest.raises(ValueError, match=r'scaler\.transform: effective n_jobs=1'):
                validate_domain(frozen, domain | {'scale_jobs': [1, 2]}, None)
            candidate = params | {'scale_jobs': 1}
        else:
            transform.n_jobs = 1
            frozen, candidate = freeze_manifest(scaler, None), params
        validate_resources(frozen, candidate)
        actual, pid = pool.submit(_prepared_probe, frozen, candidate).result(timeout=30)
        expected, _ = _prepared_probe(frozen, candidate)
        assert actual == expected and actual['x_test'] and pid != os.getpid()


def _assert_fresh_identity_snapshot(plan):
    from limen.experiment import _parallel_identity as identities
    from limen.experiment._parallel_plan import plan_digests
    from importlib import import_module
    read_bytes = Path.read_bytes
    source = Path(import_module('limen.scalers.robust_scaler').__file__)
    reads = []
    changed = False
    def read(path):
        result = read_bytes(path)
        if path == source:
            reads.append(path)
            if changed:
                return result + b'\n# changed source'
        return result
    with pytest.MonkeyPatch.context() as patch, identities.distribution_providers(plan.distribution_providers):
        patch.setattr(Path, 'read_bytes', read)
        assert plan_digests(plan.manifest, plan.stages) == (plan.pipeline_digest, plan.resource_digest)
        assert reads == [source]
        changed = True
        altered = plan_digests(plan.manifest, plan.stages)
        assert reads == [source, source]
        assert altered[0] != plan.pipeline_digest and altered[1] != plan.resource_digest


def _assert_inherited_identity(plan):
    from limen.experiment import _parallel_encoding as codec
    from limen.experiment._parallel_plan import plan_digests
    original = codec.module_identity
    def identity(name):
        result = original(name)
        return [result[0], 'changed-inherited-source', result[2]] if name == 'tests.test_parallel_rounds' else result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(codec, 'module_identity', identity)
        changed = plan_digests(plan.manifest, plan.stages)
    assert changed[0] != plan.pipeline_digest and changed[1] != plan.resource_digest


def _assert_selector_and_metric_identity(plan):
    from limen.experiment import _parallel_encoding as codec, _parallel_plan as plans
    fixed = freeze_manifest(copy.deepcopy(plan.manifest).set_scaler(RobustScaler), None)
    registry = list(plans.SCALER_REGISTRY.items())
    swapped = dict([(registry[1][0], registry[0][1]), (registry[0][0], registry[1][1]), *registry[2:]])
    assert list(swapped.values()) == list(plans.SCALER_REGISTRY.values())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plans, 'SCALER_REGISTRY', swapped)
        changed = plans.plan_digests(plan.manifest, plan.stages)
        assert changed[0] != plan.pipeline_digest and changed[1] != plan.resource_digest
        assert plans.plan_digests(fixed.manifest, fixed.stages) == (fixed.pipeline_digest, fixed.resource_digest)
    config = _configuration()
    config['sfd']['manifest']['calibration']['threshold_function']['params'] = {}
    default_metric = freeze_manifest(build_manifest(config), config)
    original = codec.module_identity
    def identity(name):
        result = original(name)
        return [result[0], 'changed-default-metric', result[2]] if name == 'limen.metrics.balanced_metric' else result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(codec, 'module_identity', identity)
        patch.setattr(plans, 'module_identity', identity)
        changed = plans.plan_digests(default_metric.manifest, default_metric.stages)
    assert changed[0] != default_metric.pipeline_digest and changed[1] != default_metric.resource_digest


def _assert_plan_identity(plan):
    from tests.test_parallel_rounds import recorded_source
    import limen.experiment._parallel_encoding as codec
    import limen.experiment._parallel_plan as plans
    def check(value):
        assert not isinstance(value, (pl.DataFrame, pl.LazyFrame)) and not hasattr(value, 'read')
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                check(getattr(value, field.name))
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item)
        assert not any(name.endswith('_') and not name.startswith('_') for name in vars(value)) if hasattr(value, '__dict__') else True
    check(plan)
    source = recorded_source()
    expected = source_digest((source, None, None))
    assert source_digest((pl.concat([source.head(10), source.slice(10)], rechunk=False), None, None)) == expected
    for changed in (source.reverse(), source.slice(1), source.select(reversed(source.columns))):
        assert source_digest((changed, None, None)) != expected
    original = codec.module_identity
    for module, changes in (('limen.scalers.robust_scaler', True), ('limen.indicators._atr', True), ('limen.indicators.obv', False)):
        def identity(name, module=module):
            result = original(name)
            return [result[0], 'changed-source', result[2]] if name == module else result
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(codec, 'module_identity', identity)
            patch.setattr(plans, 'module_identity', identity)
            altered = freeze_manifest(build_manifest(_configuration()), _configuration())
            assert (altered.pipeline_digest != plan.pipeline_digest) is changes
            assert (altered.resource_digest != plan.resource_digest) is changes
    cyclic = copy.deepcopy(plan.manifest)
    cyclic.params = {}  # Deliberately undeclared state is rejected rather than transported.
    with pytest.raises(ValueError, match='undeclared runtime state'):
        freeze_manifest(cyclic, None)



def _factory_probe(plan, factory, extra=None):
    from limen.experiment._parallel_identity import distribution_providers
    from limen.experiment._parallel_plan import plan_digests
    from tests.test_parallel_rounds import recorded_source
    with distribution_providers(plan.distribution_providers):
        assert plan_digests(plan.manifest, plan.stages) == (plan.pipeline_digest, plan.resource_digest)
    params = _parameters() | {'auxiliary_factory': factory} | (extra or {})
    validate_resources(plan, params)
    manifest = clone_manifest(plan.manifest)
    kwargs = manifest.resolve_model_kwargs(params)
    result = manifest.architecture_function(recorded_source().head(24), **kwargs)
    return result['caller_mean'].to_list(), os.getpid()


def _assert_factory_resources(plan, params):
    from limen.experiment._parallel_resources import unfrozen_stage
    from limen.experiment._parallel_resources import deterministic_stage
    declare_parallel(_factory_architecture, ParallelRoundContract(deterministic_stage, prediction_kind='binary'))
    declare_parallel(_auxiliary_factory, ParallelRoundContract(unfrozen_stage))
    manifest = copy.deepcopy(plan.manifest).with_reference_architecture(_factory_architecture)
    frozen = freeze_manifest(manifest, None)
    with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory: unfrozen external state'):
        validate_resources(frozen, params)
    with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory: unfrozen external state'):
        validate_domain(frozen, {key: [value] for key, value in params.items()}, None)
    declare_parallel(_auxiliary_factory, ParallelRoundContract(_caller_validator, ('window',)))
    frozen = freeze_manifest(manifest, None)
    validate_resources(frozen, params)
    with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory: positive integer window required'):
        validate_domain(frozen, {key: [value] for key, value in params.items()} |
                        {'auxiliary_params': [{'window': 2}, {'window': 0}]}, None)
    declare_parallel(_optional_factory_architecture, ParallelRoundContract(deterministic_stage, prediction_kind='binary'))
    manifest = copy.deepcopy(plan.manifest).with_reference_architecture(_optional_factory_architecture)
    candidate = params | {'auxiliary_factory': _auxiliary_factory}
    options = {'auxiliary_factory': [_auxiliary_factory, _other_auxiliary_factory]}
    opaque = OpaqueFactory((1,))
    declare_parallel(opaque, ParallelRoundContract(_caller_validator, ('window',)))
    with pytest.raises(TypeError, match=r'object\.__new__.*not safe'):
        freeze_manifest(manifest, None, {'auxiliary_factory': [opaque]})
    with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory: missing frozen independence contract'):
        validate_resources(freeze_manifest(manifest, None), candidate)
    declare_parallel(_auxiliary_factory, ParallelRoundContract(unfrozen_stage))
    with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory: unfrozen external state'):
        validate_resources(freeze_manifest(manifest, None, {'auxiliary_factory': [_auxiliary_factory]}), candidate)
    declare_parallel(_auxiliary_factory, ParallelRoundContract(_caller_validator, ('window',)))
    declare_parallel(_other_auxiliary_factory, ParallelRoundContract(_caller_validator, ('window',)))
    frozen = freeze_manifest(manifest, None, options)
    validate_resources(frozen, candidate)
    validate_resources(frozen, params | {'auxiliary_factory': _other_auxiliary_factory})
    validate_domain(frozen, {key: [value] for key, value in params.items()} | options, None)
    assert digest(decode(encode(frozen))) == digest(frozen)
    with startup_limits(), ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        for factory in options['auxiliary_factory']:
            actual, pid = pool.submit(_factory_probe, frozen, factory).result(timeout=30)
            expected, _ = _factory_probe(frozen, factory)
            assert actual == expected and pid != os.getpid()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_auxiliary_factory, '__kwdefaults__', {'window': 4})
            captured_function = freeze_manifest(manifest, None, {'auxiliary_factory': [_auxiliary_factory]})
        before = digest(captured_function)
        validate_domain(captured_function, {key: [value] for key, value in params.items()} | {'auxiliary_factory': [_auxiliary_factory]}, None)
        from tests.test_parallel_rounds import recorded_source
        for _ in range(2):
            actual, pid = pool.submit(_factory_probe, captured_function, _auxiliary_factory).result(timeout=30)
            expected, _ = _factory_probe(captured_function, _auxiliary_factory)
            assert actual == expected == _caller_feature(recorded_source().head(24), 4)['caller_mean'].to_list() and pid != os.getpid()
        assert digest(captured_function) == before and _auxiliary_factory.__kwdefaults__ == {'window': 3}
        for bound in ({}, {'window': 5}):
            factory = partial(_auxiliary_factory, **bound)
            declare_parallel(factory, ParallelRoundContract(_caller_validator, ('window',)))
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(_auxiliary_factory, '__kwdefaults__', {'window': 4})
                captured_partial = freeze_manifest(manifest, None, {'auxiliary_factory': [factory]})
            before = digest(captured_partial)
            validate_domain(captured_partial, {key: [value] for key, value in params.items()} | {'auxiliary_factory': [factory]}, None)
            for _ in range(2):
                actual, pid = pool.submit(_factory_probe, captured_partial, factory).result(timeout=30)
                expected, _ = _factory_probe(captured_partial, factory)
                assert actual == expected == _caller_feature(recorded_source().head(24), bound.get('window', 4))['caller_mean'].to_list() and pid != os.getpid()
            assert digest(captured_partial) == before and _auxiliary_factory.__kwdefaults__ == {'window': 3}
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_auxiliary_factory, '__kwdefaults__', {'window': 0})
            invalid_function = freeze_manifest(manifest, None, {'auxiliary_factory': [_auxiliary_factory]})
        with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory: positive integer window required'):
            validate_domain(invalid_function, {key: [value] for key, value in params.items()} | {'auxiliary_factory': [_auxiliary_factory]}, None)
        factory = DefaultRollingFactory()
        declare_parallel(factory, ParallelRoundContract(_caller_validator, ('window',)))
        frozen_factory = freeze_manifest(manifest, None, {'auxiliary_factory': [factory]})
        validate_domain(frozen_factory, {key: [value] for key, value in params.items()} | {'auxiliary_factory': [factory]}, None)
        actual, pid = pool.submit(_factory_probe, frozen_factory, factory).result(timeout=30)
        expected, _ = _factory_probe(frozen_factory, factory)
        from tests.test_parallel_rounds import recorded_source
        assert actual == expected == _caller_feature(recorded_source().head(24), 3)['caller_mean'].to_list() and pid != os.getpid()
        declare_parallel(CallerRolling, ParallelRoundContract(_caller_validator, ('window',)))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(CallerRolling.__init__, '__defaults__', (3,))
            frozen_class = freeze_manifest(manifest, None, {'auxiliary_factory': [CallerRolling]})
            validate_domain(frozen_class, {key: [value] for key, value in params.items()} | {'auxiliary_factory': [CallerRolling]}, None)
        before = digest(frozen_class)
        for _ in range(2):
            actual, pid = pool.submit(_factory_probe, frozen_class, CallerRolling).result(timeout=30)
            expected, _ = _factory_probe(frozen_class, CallerRolling)
            assert actual == expected == _caller_feature(recorded_source().head(24), 3)['caller_mean'].to_list() and pid != os.getpid()
        assert digest(frozen_class) == before and CallerRolling.__init__.__defaults__ is None
        declare_parallel(AllocatedRolling, ParallelRoundContract(seeded_stage, ('seed',)))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(AllocatedRolling.__new__, '__kwdefaults__', {'seed': 42})
            frozen_allocator = freeze_manifest(manifest, None, {'auxiliary_factory': [AllocatedRolling]})
            validate_domain(frozen_allocator, {key: [value] for key, value in params.items()} | {'auxiliary_factory': [AllocatedRolling]}, None)
        before = digest(frozen_allocator)
        for _ in range(2):
            actual, pid = pool.submit(_factory_probe, frozen_allocator, AllocatedRolling).result(timeout=30)
            expected, _ = _factory_probe(frozen_allocator, AllocatedRolling)
            assert actual == expected == _caller_feature(recorded_source().head(24), 3)['caller_mean'].to_list() and pid != os.getpid()
        assert digest(frozen_allocator) == before and AllocatedRolling.__new__.__kwdefaults__ == {'seed': None}
        with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory: fixed integer seed required'):
            validate_domain(freeze_manifest(manifest, None, {'auxiliary_factory': [AllocatedRolling]}),
                            {key: [value] for key, value in params.items()} | {'auxiliary_factory': [AllocatedRolling]}, None)
        declare_parallel(DualSeedRolling, ParallelRoundContract(seeded_stage, ('seed',)))
        dual = freeze_manifest(manifest, None, {'auxiliary_factory': [DualSeedRolling]})
        with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory\.allocate: fixed integer seed required'):
            validate_resources(dual, params | {'auxiliary_factory': DualSeedRolling})
        with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory\.allocate: fixed integer seed required'):
            validate_domain(dual, {key: [value] for key, value in params.items()} | {'auxiliary_factory': [DualSeedRolling]}, None)
        safe = params | {'auxiliary_factory': DualSeedRolling, 'auxiliary_params': {'seed': 42}}
        validate_domain(dual, {key: [value] for key, value in safe.items()}, None)
        actual, pid = pool.submit(_factory_probe, dual, DualSeedRolling, {'auxiliary_params': {'seed': 42}}).result(timeout=30)
        expected, _ = _factory_probe(dual, DualSeedRolling, {'auxiliary_params': {'seed': 42}})
        assert actual == expected == _caller_feature(recorded_source().head(24), 3)['caller_mean'].to_list() and pid != os.getpid()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(CallerRolling.__init__, '__defaults__', (0,))
            invalid_class = freeze_manifest(manifest, None, {'auxiliary_factory': [CallerRolling]})
            assert invalid_class.pipeline_digest != frozen_class.pipeline_digest
            assert invalid_class.resource_digest != frozen_class.resource_digest
            with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory: positive integer window required'):
                validate_domain(invalid_class, {key: [value] for key, value in params.items()} | {'auxiliary_factory': [CallerRolling]}, None)
    declare_parallel(_other_auxiliary_factory, ParallelRoundContract(unfrozen_stage))
    with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory: unfrozen external state'):
        validate_domain(freeze_manifest(manifest, None, options),
                        {key: [value] for key, value in params.items()} | options, None)
    declare_parallel(_other_auxiliary_factory, ParallelRoundContract(_caller_validator, ('window',)))
    with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory: missing frozen independence contract'):
        validate_resources(freeze_manifest(manifest, None, {'auxiliary_factory': [_auxiliary_factory]}),
                           params | {'auxiliary_factory': _other_auxiliary_factory})
    with pytest.raises(ValueError, match=r'architecture\.auxiliary_factory: positive integer window required'):
        validate_domain(frozen, {key: [value] for key, value in params.items()} | options |
                        {'auxiliary_params': [{'window': 2}, {'window': 0}]}, None)



def test_effective_resources_and_output_evidence(tmp_path) -> None:
    declare_parallel(_caller_feature, ParallelRoundContract(_caller_validator, ('window',)))
    config = _configuration()
    plan = freeze_manifest(build_manifest(config), config)
    params = _parameters()
    validate_resources(plan, params)
    validate_domain(plan, {key: [value] for key, value in params.items()}, None)
    _assert_resource_projection(plan, params)
    _assert_factory_resources(plan, params)
    class_weights = {0: 1, 1: 2}
    assert decode(encode(class_weights)) == class_weights
    validate_resources(plan, params | {'class_weight': class_weights})
    assert parameter_matches('(1, 2)', [1, 2], legacy=True)
    assert not parameter_matches('(1, 2)', [1, 2])
    shared = [1]
    assert parameter_matches('[[1], [1]]', [shared, shared])
    assert not parameter_matches('[[2], [1]]', [shared, shared])
    ablation = copy.deepcopy(plan.manifest).set_feature_ablation()
    validate_resources(freeze_manifest(ablation, None), params | {'feature_drop_seed': None})
    disabled = copy.deepcopy(plan.manifest).add_feature(_caller_feature, include_if='optional', window='{unused_window}')
    without_calibration = {key: value for key, value in params.items() if not key.startswith(('cal_', 'threshold_'))}
    validate_resources(freeze_manifest(disabled, None), without_calibration | {'use_calibration': False, 'use_threshold': False})
    with pytest.raises(ValueError, match=r'feature_transforms.*unresolved parameter unused_window'):
        validate_resources(freeze_manifest(disabled, None), params | {'optional': True})
    from limen.indicators.obv import obv
    from limen.features.log_returns import log_returns
    validate_resources(freeze_manifest(copy.deepcopy(plan.manifest).add_feature(obv).add_feature(log_returns), None), params)
    for jobs in (None, True, 0, -1, 2):
        with pytest.raises(ValueError, match='effective n_jobs=1'):
            validate_resources(plan, params | {'n_jobs': jobs})
    for seed in (None, True, '42'):
        with pytest.raises(ValueError, match='fixed integer seed'):
            validate_resources(plan, params | {'random_state': seed})
    with pytest.raises(ValueError, match='fixed integer seed'):
        validate_domain(plan, {'n_jobs': [1], 'solver': ['lbfgs', 'saga'], 'random_state': [None],
                               **{key: [value] for key, value in params.items() if key not in ('n_jobs', 'solver', 'random_state')}}, None)
    sklearn_stage({'solver': 'lbfgs', 'random_state': None})
    for option in ({'device': 'cuda'}, {'tree_method': 'gpu_hist'}, {'sampling_method': 'gradient_based'}):
        with pytest.raises(ValueError, match='CPU backend required'):
            xgboost_stage({'n_jobs': 1, 'random_state': 42} | option)
    for value in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='nonfinite'):
            encode(value)
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match='cycles'):
        encode(cycle)
    target = copy.deepcopy(plan.manifest)
    target.with_target_label('noise', RandomBinaryTarget, fit_params={'random_state': '{target_seed}'})
    seeded = freeze_manifest(target, None)
    validate_resources(seeded, params | {'target_seed': 42})
    for seed in (None, True):
        with pytest.raises(ValueError, match='target: fixed integer seed'):
            validate_resources(seeded, params | {'target_seed': seed})
    with pytest.raises(ValueError, match='target: fixed integer seed'):
        validate_domain(seeded, {key: [value] for key, value in params.items()}, {'target_seed': None})
    omitted = copy.deepcopy(plan.manifest).with_target_label('noise', RandomBinaryTarget)
    with pytest.raises(ValueError, match='target: fixed integer seed'):
        validate_resources(freeze_manifest(omitted, None), params)
    _assert_dispatch_refusal(plan, seeded, params)
    from tests.test_parallel_rounds import assert_runs_equal, parallel_config, records, run_loop
    configuration = parallel_config()
    configuration['uel']['n_permutations'] = 2
    configuration['sfd']['params']['tuple_tag'] = [(1, 2), (3, 4)]
    configuration['sfd']['params']['alias_tag'] = [[shared, shared]]
    configuration['sfd']['params']['target_seed'] = [42]
    configuration['sfd']['manifest']['target'] = {'name': 'noise', 'class': 'limen.targets.RandomBinaryTarget',
                                               'fit_params': {'random_state': '{target_seed}'}}
    configuration['uel'].pop('pruning_strategies')
    directory = tmp_path / 'recorded'
    loop = run_loop(configuration, directory, workers=2)
    serial = tmp_path / 'serial-target'
    run_loop(configuration, serial, workers=1)
    assert_runs_equal(directory, serial)
    saved = records(directory)
    assert len(saved) == 2 and all(type(row['round_params']['tuple_tag']) is tuple for row in saved)
    contract = next(stage.contract for stage in vars(loop)['_parallel_plan'].stages if stage.path == 'architecture')
    for row in saved:
        validate_record(row, record_context(row), contract, execution=True, outputs=True, snapshot=True)
        assert wire_record(wire_record(row), restore=True)['round_params'] == row['round_params']
        assert row['parallel_output']['count'] == len(row['alignment']['test_datetimes']) == len(row['preds'])
    for fault in ('count', 'kind', 'digest', 'probabilities', 'prediction', 'occurrence', 'params', 'missing'):
        row = copy.deepcopy(saved[0])
        if fault in ('count', 'kind', 'digest'):
            row['parallel_output'][fault] = 0 if fault == 'count' else 'incorrect'
        elif fault == 'probabilities':
            row['probs'] = None
        elif fault == 'prediction':
            row['preds'][0] = 1 - row['preds'][0]
        elif fault == 'occurrence':
            row['round_id'] = saved[1]['round_id']
        elif fault == 'params':
            row['round_params']['C'] *= 2
        else:
            row.pop('preds')
        with pytest.raises(ValueError, match=r'output|digest|evidence'):
            validate_record(row, record_context(row), contract, execution=True, outputs=True)
    rejected = tmp_path / 'corrupt'
    shutil.copytree(directory, rejected)
    corrupted = records(rejected, raw=True)
    corrupted[0]['execution']['net'][0] += 0.01  # Deliberate saved-ledger corruption.
    storage.round_path(rejected).write_bytes(b''.join(gzip.compress((json.dumps(row) + '\n').encode(), mtime=0) for row in corrupted))
    before = {str(path.relative_to(rejected)): path.read_bytes() for path in rejected.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='digest'):
        run_loop(configuration, rejected, resume=True)
    assert {str(path.relative_to(rejected)): path.read_bytes() for path in rejected.rglob('*') if path.is_file()} == before
    resumed = run_loop(configuration, directory, resume=True)
    assert resumed.experiment_log.height == 2
    assert all(type(row['round_params']['tuple_tag']) is tuple for row in records(directory))
    from limen.experiment.checkpoint_manager import CheckpointManager
    checkpoint = CheckpointManager().load(directory)
    assert checkpoint['domain_state']['tuple_tag'] == [(1, 2), (3, 4)]
    aliases = checkpoint['domain_state']['alias_tag'][0]
    assert aliases == [[1], [1]] and aliases[0] is aliases[1]
    _assert_recovery_faults(configuration, directory, tmp_path)
    _assert_legacy_recovery(configuration, directory, tmp_path)
    _assert_native_output_forms()


def _assert_resource_projection(plan, params):
    from limen.experiment import _parallel_resources as resources
    original = vars(resources)['_validate_stage']
    calls = 0
    def counted(*args):
        nonlocal calls
        calls += 1
        assert calls <= 500, 'unrelated grid axes multiplied resource validation'
        return original(*args)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(resources, '_validate_stage', counted)
        for additions in ({f'unconsumed_{index}': list(range(10)) for index in range(10)},
                          {key: list(range(10)) for key in ('nthread', 'num_threads', 'device', 'device_type', 'seed')}):
            validate_domain(plan, {key: [value] for key, value in params.items()} | additions, None)
    conditional = copy.deepcopy(plan.manifest).add_feature(_caller_feature, include_if='optional', window='{candidate_window}')
    domain = {key: [value] for key, value in params.items()} | {'optional': [False, True], 'candidate_window': [2, 0]}
    with pytest.raises(ValueError, match=r'feature_transforms.*positive integer window'):
        validate_domain(freeze_manifest(conditional, None), domain, None)


def _assert_dispatch_refusal(plan, seeded, params):
    from importlib import import_module
    from tests.test_parallel_rounds import recorded_source
    from limen.experiment import _parallel_worker as worker
    from limen.experiment.manifest_core import MLManifest
    tabpfn_module = import_module('limen.sfd.reference_architecture.tabpfn_binary')
    def no_preparation(*_args, **_kwargs):
        raise AssertionError('unsafe candidate reached preparation or optional backend')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MLManifest, 'prepare_data', no_preparation)
        patch.setattr(worker, '_worker_sources', worker.FrozenRoundSources(recorded_source(), None, None))
        for selected, candidate, error in ((plan, params | {'n_jobs': None}, 'effective n_jobs=1'),
                                          (seeded, params | {'target_seed': None}, 'fixed integer seed')):
            patch.setattr(worker, '_worker_plan', selected)
            with pytest.raises(RuntimeError, match=error):
                worker.evaluate_round(candidate | {'_id': 'refused-control', '_round_index': 0})
        tab_config, errors = parse(Path('limen/yaml/templates/tabpfn_binary.yaml').read_text())
        assert not errors
        tab_plan = freeze_manifest(build_manifest(tab_config), tab_config)
        patch.setattr(tabpfn_module, 'require_optional', no_preparation)
        tab_params = {key: values[0] for key, values in tab_config['sfd']['params'].items()}
        with pytest.raises(ValueError, match='architecture: unfrozen external state'):
            validate_resources(tab_plan, tab_params)


def _assert_recovery_faults(configuration, directory, temporary):
    from tests.test_parallel_rounds import run_loop
    for fault in ('missing_plan', 'version', 'marker', 'worker', 'resource', 'downgraded'):
        rejected = temporary / fault
        shutil.copytree(directory, rejected)
        path = rejected / 'metadata.json'
        metadata = json.loads(path.read_text())
        if fault == 'missing_plan':
            metadata.pop('parallel_plan')
        elif fault == 'version':
            metadata['limen_version'] = 'unknown'
        elif fault == 'downgraded':
            metadata['limen_version'] = '5.45.7'
        elif fault == 'worker':
            metadata['worker_native_threads'] = 2
        elif fault == 'marker':
            metadata['parallel_plan']['version'] = True
        else:
            metadata['parallel_plan']['resource_digest'] = 'changed-control'
        path.write_text(json.dumps(metadata))
        before = {str(item.relative_to(rejected)): item.read_bytes() for item in rejected.rglob('*') if item.is_file()}
        with pytest.raises(ValueError, match=r'identity|version|policy'):
            run_loop(configuration, rejected, resume=True)
        assert {str(item.relative_to(rejected)): item.read_bytes() for item in rejected.rglob('*') if item.is_file()} == before


def _assert_legacy_recovery(configuration, directory, temporary):
    from limen.experiment.checkpoint_manager import CheckpointManager
    from tests.test_parallel_rounds import records, run_loop
    legacy = temporary / 'legacy'
    shutil.copytree(directory, legacy)
    path = legacy / 'metadata.json'
    metadata = json.loads(path.read_text())
    metadata.update(limen_version='5.45.7')
    metadata.pop('parallel_plan')
    path.write_text(json.dumps(metadata))
    checkpoint = CheckpointManager().load(legacy)
    checkpoint.pop('parameter_encoding')
    checkpoint['metadata']['content_hash'] = CheckpointManager.compute_content_hash(configuration['sfd']['params'])
    (legacy / 'checkpoint.json').write_text(json.dumps(checkpoint))
    converted = [wire_record(record, legacy=True) for record in records(directory)]
    storage.round_path(legacy).write_bytes(b''.join(gzip.compress((json.dumps(record)+'\n').encode(),mtime=0) for record in converted))
    resumed = run_loop(configuration, legacy, resume=True)
    assert resumed.experiment_log.height == 2
    assert all('parallel_plan' not in record and record['round_params']['tuple_tag'] in [[1,2],[3,4]] for record in records(legacy))


def _assert_native_output_forms():
    from threadpoolctl import threadpool_limits
    from tests.test_parallel_rounds import recorded_source
    from tests.test_record_model_outputs import _config
    from limen.experiment._parallel_plan import plan_identity
    from limen.experiment._parallel_evidence import round_record
    from limen.experiment.manifest_core import MLManifest, ProductConfig
    from limen.sfd.reference_architecture.direction_sizing import direction_sizing
    from limen.targets.trade_outcome import TradeOutcomeTarget

    source = recorded_source()
    continuous = build_manifest(_config('ridge'))
    rule_config, errors = parse(Path('limen/yaml/templates/rule_based.yaml').read_text())
    assert not errors
    rule = build_manifest(rule_config)
    rule.split_dates = continuous.split_dates
    exposure = MLManifest().set_data_source(recorded_source, params={'kline_size': 3600})
    exposure.set_split_config(6, 2, 2)
    exposure.with_target_label('outcome', TradeOutcomeTarget).with_reference_architecture(direction_sizing)
    exposure.set_backtest_config(prediction_mode='target_exposure', product=ProductConfig('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0), max_holding_seconds=7200)
    cases = [(continuous, {'alpha': 0.1, 'solver': 'svd'}, 'continuous'),
             (rule, {key: values[0] for key, values in rule_config['sfd']['params'].items()}, 'binary'),
             (exposure, {'seed': 42, 'folds': 3, 'min_train_samples': 20, 'sizing_params': {'solver': 'svd'}}, 'target_exposure'),
             (copy.deepcopy(exposure), {'seed': 42, 'folds': 3, 'min_train_samples': 20, 'sizing_params': {'solver': 'svd'}, 'conditional_size': True}, 'target_exposure')]
    for manifest, params, kind in cases:
        plan = freeze_manifest(manifest, None)
        assert digest(decode(encode(plan.manifest))) == digest(plan.manifest)
        validate_resources(plan, params)
        vars(manifest)['_parallel_evidence'] = True
        with threadpool_limits(1):
            data = manifest.prepare_data(source, params)
            data.update(_record_execution=True, _record_model_outputs=True)
            results = manifest.run_model(data, params)
        alignment = {key: value for key, value in data['_alignment'].items() if not key.startswith('_')}
        contract = next(stage.contract for stage in plan.stages if stage.path == 'architecture')
        count = data['x_test'].height if isinstance(manifest, MLManifest) else data['test'].height
        record = round_record({'_id': f'private-{kind}', '_round_index': 0}, params, results, alignment,
                              plan_identity(plan, (source, None, None)), contract, count=count, execution=True, outputs=True)
        assert record['parallel_output']['kind'] == kind and record['parallel_output']['count'] == count
        assert record['probs'] is None and record['parallel_output']['has_probabilities'] is False
        validate_record(record, record_context(record), contract, execution=True, outputs=True,
                        event=kind == 'target_exposure')
