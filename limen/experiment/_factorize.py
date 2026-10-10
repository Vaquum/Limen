"""Narrow, opt-in reuse of deterministic snapshot sweep signal work."""

from __future__ import annotations

import copy
import inspect
import json
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import NoReturn, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.experiment.errors import StrictModeError
from limen.experiment.manifest_core import MLManifest, Manifest, RuleBasedManifest
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.features.dollar_bar_crash_reversal import dollar_bar_crash_reversal
from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return
from limen.metrics.rule_based_metrics import rule_based_metrics
from limen.sfd.reference_architecture import _backtest_evaluation
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.sfd.reference_architecture.dlinear_regressor import dlinear_regressor
from limen.sfd.reference_architecture.lightgbm_binary import lightgbm_binary
from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy, rule_based
from limen.targets.next_return import NextReturnTarget
from limen.targets.quantile_binary import QuantileBinaryTarget

_BACKTEST_FIELDS = ('fee_bps', 'slip_bps', 'notional_rate', 'take_profit_bps', 'stop_loss_bps')
_RESERVED = frozenset(('bar_type', 'feature_groups', 'use_calibration', 'use_threshold'))
_FIELD_ROOT = re.compile(r'^[a-zA-Z_][a-zA-Z_0-9]*')
_FEATURES = frozenset((window_return, lag_range, dollar_bar_crash_reversal))
_ML_FEATURES = frozenset((window_return, lag_range))


def _fail(reason: str) -> NoReturn:
    raise ValueError(f'factorize requires an independent deterministic snapshot grid: {reason}')


def _references(value: object, name: str) -> bool:
    """Cover bare, braced and formatted manifest parameter consumers."""
    if isinstance(value, str):
        return value == name or any(
            (field is not None and (root := _FIELD_ROOT.match(field)) is not None and root[0] == name)
            or _references(spec, name) for _, field, spec, _ in Formatter().parse(value)
        )
    if isinstance(value, Mapping):
        return any(_references(v, name) for v in cast(Mapping[object, object], value).values())
    if isinstance(value, (list, tuple)):
        return any(_references(v, name) for v in cast(list[object] | tuple[object, ...], value))
    if hasattr(value, '__dataclass_fields__'):
        return any(_references(v, name) for k, v in cast(Mapping[str, object], vars(value)).items()
                   if k not in ('backtest_config',))
    return False


def _axes(manifest: Manifest, domain: Mapping[str, list[object]]) -> frozenset[str]:
    config = manifest.backtest_config
    if config is None:
        _fail('missing backtest parameters')
    axes: set[str] = set()
    for field in _BACKTEST_FIELDS:
        value = getattr(config, field)
        if isinstance(value, str):
            ref = value.strip()
            ref = ref[1:-1] if ref.startswith('{') and ref.endswith('}') else ref
            if ref not in domain:
                _fail(f'unknown backtest parameter {ref}')
            axes.add(ref)
    if not axes:
        _fail('no backtest-only search axes')
    return frozenset(axes)


def _check_manifest(manifest: Manifest, domain: Mapping[str, list[object]], axes: frozenset[str]) -> None:
    function = manifest.architecture_function
    if type(manifest) is MLManifest:
        if function not in (dlinear_regressor, lightgbm_binary):
            _fail('unsupported ML architecture')
        if manifest.target_class_config is None or manifest.target_class_config.target_class not in (NextReturnTarget, QuantileBinaryTarget):
            _fail('unsupported ML target')
        if (manifest.scaler is not None or manifest.ablation_config is not None
                or manifest.pca_compression_config is not None or manifest.prediction_calibration_config is not None
                or manifest.data_dict_extension is not None or manifest.objective is not None):
            _fail('ML calibration, scaler, ablation, compression, extension or objective')
        if any(entry.func not in _ML_FEATURES for entry in manifest.feature_transforms):
            _fail('unrecognized ML feature transform')
    elif type(manifest) is RuleBasedManifest:
        if function is not rule_based or manifest.target_class_config is not None:
            _fail('unsupported rule-based architecture')
        if any(entry.func not in _FEATURES for entry in manifest.feature_transforms):
            _fail('unrecognized rule-based transform')
        if manifest.strategy is None:
            _fail('missing rule-based strategy')
    else:
        _fail('only the shipped ML and rule-based manifests are supported')
    if (manifest.pre_split_data_selector is not None or manifest.bar_formation is not None
            or manifest.split_walk_forward is not None or manifest.metrics_params):
        _fail('dynamic preparation, walk-forward or metrics')
    if any(entry.include_if is not None or entry.group is not None for entry in manifest.feature_transforms):
        _fail('dynamic feature selection')

    if function is None:
        _fail('missing architecture')
    assert function is not None
    sig = inspect.signature(function)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        _fail('opaque architecture kwargs')
    for key in axes:
        if key in _RESERVED or key in sig.parameters:
            _fail(f'backtest parameter {key} affects signal generation')
        if any(_references(value, key) for attr, value in vars(manifest).items()
               if attr not in ('backtest_config', 'architecture_function', 'data_source_config')):
            _fail(f'backtest parameter {key} is also consumed by the manifest')
        if manifest.data_source_config is not None and _references(manifest.data_source_config.params, key):
            _fail(f'backtest parameter {key} is also consumed by the source')
    for params in _candidate_settings(domain, axes):
        if manifest.resolve_trade_policy(params) is not None:
            _fail('event execution')
        if function is lightgbm_binary:
            opts = manifest.resolve_model_kwargs(params)
            if (opts['deterministic'] is not True or opts['force_row_wise'] is not True
                    or opts['boosting_type'] != 'gbdt' or type(opts['random_state']) is not int
                    or opts['n_jobs'] != 1 or opts['subsample_freq'] != 0
                    or opts['subsample'] != 1.0 or opts['colsample_bytree'] != 1.0):
                _fail('unsafe LightGBM settings')


def _candidate_settings(domain: Mapping[str, list[object]], axes: frozenset[str]) -> list[dict[str, object]]:
    """Only values that can change eligibility need a scan, not the Cartesian grid."""
    settings = {key: values[0] for key, values in domain.items()}
    checks = [settings]
    for key, values in domain.items():
        if key not in axes:
            checks.extend([{**settings, key: value} for value in values[1:]])
    return checks


@dataclass
class _Cached:
    data: dict[str, object]
    result: dict[str, object]
    warnings: list[str]
    positions: dict[str, npt.NDArray[np.int64]] | None


class FactorizedRounds:
    """One MSQ invocation's successful signal computations, never persistent."""

    def __init__(self, *, manifest: Manifest, strategy: object, domain: Mapping[str, list[object]],
                 pruning: bool, callback: bool, context: Mapping[str, object] | None,
                 prep: Callable[..., dict[str, object]], model: Callable[..., dict[str, object]],
                 data: pl.DataFrame, record_execution: bool, record_model_outputs: bool,
                 intervention_path: Path | None = None) -> None:
        super().__init__()
        if type(strategy) is not GridStrategy:
            _fail('unshuffled GridStrategy is required')
        assert isinstance(strategy, GridStrategy)
        if strategy.get_state()['shuffle']:
            _fail('unshuffled GridStrategy is required')
        if pruning or callback:
            _fail('pruning and callbacks are not supported')
        if prep != manifest.prepare_data or model != manifest.run_model or any(name in vars(manifest) for name in ('prepare_data', 'run_model')):
            _fail('overridden manifest pipeline')
        self.axes = _axes(manifest, domain)
        if context and self.axes.intersection(context):
            _fail('context overrides a backtest-only parameter')
        self.domain = {key: list(values) for key, values in domain.items()} | {key: [value] for key, value in (context or {}).items()}
        _check_manifest(manifest, self.domain, self.axes)
        self.intervention_path = intervention_path
        self._check_interventions()
        self.manifest = manifest
        self.data = data
        self.prep, self.model = prep, model
        self.record_execution, self.record_model_outputs = record_execution, record_model_outputs
        self.cache: dict[str, _Cached] = {}

    def _check_interventions(self) -> None:
        if self.intervention_path is not None and self.intervention_path.exists():
            _fail('interventions are not supported')

    def evaluate(self, params: dict[str, object]) -> tuple[dict[str, object], dict[str, object], list[str]]:
        self._check_interventions()
        if set(params) != set(self.domain) or any(not any(type(value) is type(admitted) and value == admitted for admitted in self.domain[key]) for key, value in params.items()):
            _fail('candidate is outside the preflight domain')
        key = json.dumps({k: v for k, v in params.items() if k not in self.axes}, sort_keys=True, default=str)
        entry = self.cache.get(key)
        if entry is None:
            caught: list[warnings.WarningMessage] = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    data = self.prep(self.data, round_params=params)
                    data['_record_execution'] = self.record_execution
                    data['_record_model_outputs'] = self.record_model_outputs
                    if isinstance(data.get('_alignment'), dict):
                        alignment = cast(dict[str, object], data['_alignment'])
                        for field in ('execution', 'market', 'model_outputs'):
                            _ = alignment.pop(field, None)
                    result = self.model(data=data, round_params=params)
            except StrictModeError:
                for warning in caught:
                    warnings.warn(str(warning.message), warning.category, stacklevel=2)
                raise
            recorded = list(dict.fromkeys(str(w.message) for w in caught))
            if not isinstance(data.get('price_data_for_backtest'), pl.DataFrame) and type(self.manifest) is MLManifest:
                _fail('missing snapshot test prices')
            positions: dict[str, npt.NDArray[np.int64]] | None = None
            if type(self.manifest) is RuleBasedManifest:
                rule = RuleBasedStrategy()
                strategy_cfg = cast(dict[str, object], data['strategy'])
                positions = {}
                logic = cast(Callable[..., pl.Series], vars(RuleBasedStrategy)['_apply_logic'])
                for split in ('train', 'val', 'test'):
                    frame = data[split]
                    if not isinstance(frame, pl.DataFrame):
                        _fail('rule-based split must be a DataFrame')
                    positions[split] = np.asarray(logic(rule, frame, strategy_cfg).fill_null(False).to_numpy(), dtype=np.int64)
            evidence = {name: data[name] for name in ('_alignment', '_backtest_provenance', 'price_data_for_backtest') if name in data}
            if positions is not None:
                for split in positions:
                    frame = cast(pl.DataFrame, data[split])
                    evidence[split] = frame.select(col for col in ('datetime', 'open', 'high', 'low', 'close') if col in frame.columns)
            self.cache[key] = _Cached(evidence, {name: value for name, value in result.items() if name != '_model'}, recorded, positions)
            return data, result, recorded

        data = dict(entry.data)
        data['_alignment'] = copy.deepcopy(entry.data['_alignment'])
        data['_record_execution'] = self.record_execution
        data['_record_model_outputs'] = self.record_model_outputs
        apply_costs = cast(Callable[..., None], vars(Manifest)['_apply_backtest_cost'])
        apply_costs(self.manifest, data, params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if entry.positions is None:
                predictions = np.asarray(entry.result['_preds'])
                if self.manifest.architecture_function is dlinear_regressor:
                    predictions = (predictions > 0).astype(int)
                compute = cast(Callable[[object, Mapping[str, object]], dict[str, float]], vars(_backtest_evaluation)['compute_backtest'])
                metrics = compute(predictions, data)
                result = {name: metrics.get(name, value)
                          for name, value in entry.result.items()}
            else:
                rule = RuleBasedStrategy()
                cost_args = cast(Callable[..., dict[str, object]], vars(ReferenceModel)['_cost_kwargs'])
                costs = cost_args(rule, data)
                run_split = cast(Callable[..., dict[str, float]], vars(RuleBasedStrategy)['_backtest_split'])
                summaries: dict[str, dict[str, float]] = {}
                for split in ('train', 'val', 'test'):
                    if split == 'test' and self.record_execution:
                        costs.update(_record_execution=True, _alignment=data['_alignment'])
                    frame = data[split]
                    if not isinstance(frame, pl.DataFrame):
                        _fail('rule-based split must be a DataFrame')
                    summaries[split] = run_split(rule, frame, entry.positions[split], costs)
                summarize = cast(Callable[..., dict[str, object]], rule_based_metrics)
                fresh = summarize(entry.positions, summaries)
                result = {name: fresh.get(name, value)
                          for name, value in entry.result.items()}
        recorded = list(dict.fromkeys([*entry.warnings, *(str(w.message) for w in caught)]))
        return data, result, recorded
