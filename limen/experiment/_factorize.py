"""Narrow, opt-in reuse of deterministic snapshot sweep signal work."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import NoReturn, cast

import numpy as np
import numpy.typing as npt
import polars as pl

import limen.experiment._holdout_composition as _composition
from limen.experiment._factorized_admission import validate_candidate, validate_domain
from limen.experiment._factorized_key import parameter_key
from limen.experiment._factorized_replay import replay, validate_output
from limen.experiment._factorized_native import native_threads
from limen.experiment.errors import StrictModeError
from limen.experiment.convergence_report import convergence_warning
from limen.experiment.manifest_core import MLManifest, Manifest, RuleBasedManifest
from limen.experiment._factorized_prep import FittedPreparation, supports_preparation
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.experiment._grid_resume import validate_factorized_grid
from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy

_BACKTEST_FIELDS = ('fee_bps', 'slip_bps', 'notional_rate', 'take_profit_bps', 'stop_loss_bps')
_RESERVED = frozenset(('bar_type', 'feature_groups', 'use_calibration', 'use_threshold'))
_CACHE_LIMIT = 128


def _fail(reason: str) -> NoReturn:
    raise ValueError(f'factorize requires an independent deterministic snapshot grid: {reason}')


def _references(value: object, name: str) -> bool:
    """Cover bare, braced and formatted manifest parameter consumers."""
    if isinstance(value, str):
        return value == name or any(
            (field is not None and field.split('.', 1)[0].split('[', 1)[0] == name)
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


@dataclass
class _Cached:
    data: dict[str, object]
    result: dict[str, object]
    warnings: list[str]
    positions: dict[str, npt.NDArray[np.int64]] | None
    convergence: bool
    derived: dict[str, object]


class FactorizedRounds:
    """One MSQ invocation's successful signal computations, never persistent."""

    def __init__(self, *, manifest: Manifest, strategy: object, domain: Mapping[str, list[object]],
                 pruning: bool, callback: bool, context: Mapping[str, object] | None,
                 prep: Callable[..., dict[str, object]], model: Callable[..., dict[str, object]],
                 data: pl.DataFrame, record_execution: bool, record_model_outputs: bool,
                 intervention_path: Path | None = None) -> None:
        super().__init__()
        if type(strategy) is not GridStrategy:
            _fail('GridStrategy is required')
        assert isinstance(strategy, GridStrategy)
        validate_factorized_grid(strategy)
        if pruning or callback:
            _fail('pruning and callbacks are not supported')
        if prep != manifest.prepare_data or model != manifest.run_model or any(name in vars(manifest) for name in ('prepare_data', 'run_model')):
            _fail('overridden manifest pipeline')
        self.axes = _axes(manifest, domain)
        if context and self.axes.intersection(context):
            _fail('context overrides a backtest-only parameter')
        self.domain = {key: list(values) for key, values in domain.items()} | {key: [value] for key, value in (context or {}).items()}
        for key in self.axes:
            if key in _RESERVED:
                _fail(f'backtest parameter {key} affects signal generation')
            if any(_references(value, key) for attr, value in vars(manifest).items()
                   if attr not in ('backtest_config', 'architecture_function', 'data_source_config')):
                _fail(f'backtest parameter {key} is also consumed by the manifest')
            if manifest.data_source_config is not None and _references(manifest.data_source_config.params, key):
                _fail(f'backtest parameter {key} is also consumed by the source')
        self.contract = validate_domain(manifest, self.domain, self.axes)
        self.intervention_path = intervention_path
        self._check_interventions()
        self.manifest = manifest
        self.data, self._source = _composition.snapshot(manifest, data)
        self.prep, self.model = prep, model
        self.record_execution, self.record_model_outputs = record_execution, record_model_outputs
        self.cache: dict[str, _Cached] = {}
        self.preparation = FittedPreparation(manifest, self.domain, self.axes, _references) if isinstance(manifest, MLManifest) and supports_preparation(manifest) else None

    def _check_interventions(self) -> None:
        if self.intervention_path is not None and self.intervention_path.exists():
            _fail('interventions are not supported')

    def evaluate(self, params: dict[str, object]) -> tuple[dict[str, object], dict[str, object], list[str], bool]:
        self._check_interventions()
        if set(params) != set(self.domain) or any(not any(type(value) is type(admitted) and value == admitted for admitted in self.domain[key]) for key, value in params.items()):
            _fail('candidate is outside the preflight domain')
        _ = validate_candidate(self.manifest, params, self.axes)
        key = parameter_key({k: v for k, v in params.items() if k not in self.axes})
        entry = _composition.signal_cache(self.manifest, self.data, self._source, self.cache, key)
        if entry is None:
            prep_key: str | None = None
            fitted: dict[str, object] = {}
            caught: list[warnings.WarningMessage] = []
            try:
                with warnings.catch_warnings(record=True) as caught, native_threads():
                    warnings.simplefilter('always')
                    if self.preparation is None:
                        data = self.prep(self.data, round_params=params)
                    else:
                        data, prep_key, fitted = self.preparation.prepare(self.data, params, self.prep)
                    data['_record_execution'] = self.record_execution
                    data['_record_model_outputs'] = self.record_model_outputs
                    if isinstance(data.get('_alignment'), dict):
                        alignment = cast(dict[str, object], data['_alignment'])
                        for field in ('execution', 'market', 'model_outputs'):
                            _ = alignment.pop(field, None)
                    payload = copy.deepcopy({name: data[name] for name in self.contract.replay_inputs})
                    result = self.model(data=data, round_params=params)
            except StrictModeError:
                for warning in caught:
                    warnings.warn(str(warning.message), warning.category, stacklevel=2)
                raise
            validate_output(self.contract, data, result)
            recorded = list(dict.fromkeys(str(w.message) for w in caught))
            if not isinstance(data.get('price_data_for_backtest'), pl.DataFrame) and type(self.manifest) is MLManifest:
                _fail('missing snapshot test prices')
            positions: dict[str, npt.NDArray[np.int64]] | None = None
            if type(self.manifest) is RuleBasedManifest and self.contract.replay is None:
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
            artifacts = cast(dict[str, object], data.get('_fitted_params', {}))
            evidence.update(copy.deepcopy({'_fitted_params': artifacts, **artifacts}))
            evidence.update(payload)
            if positions is not None:
                for split in positions:
                    frame = cast(pl.DataFrame, data[split])
                    evidence[split] = frame.select(col for col in ('datetime', 'open', 'high', 'low', 'close') if col in frame.columns)
            with warnings.catch_warnings(record=True) as economic, native_threads():
                warnings.simplefilter('always')
                _ = replay(self.manifest, self.contract, result, copy.deepcopy(evidence), params.copy(), positions, self.record_execution)
            scientific_warnings = list(caught)
            for warning in economic:
                match = next((item for item in scientific_warnings if item.category is warning.category and str(item.message) == str(warning.message)), None)
                if match is not None:
                    scientific_warnings.remove(match)
            converged_warning = bool(convergence_warning(scientific_warnings, True))
            if self.preparation is not None and prep_key is not None:
                if prep_key not in self.preparation.cache and len(self.preparation.cache) >= _CACHE_LIMIT:
                    oldest = next(iter(self.preparation.cache))
                    del self.preparation.warning_cache[oldest]
                    _composition.evict_fitted(self.preparation.cache, self.preparation.proofs)
                _composition.remember_fitted(self.preparation, prep_key, fitted, data, self.data)
                self.preparation.warning_cache[prep_key] = list(self.preparation.pending_warnings)
            if len(self.cache) >= _CACHE_LIMIT:
                del self.cache[next(iter(self.cache))]
            derived = {name: value for name, value in params.items() if name not in self.domain}
            self.cache[key] = _Cached(copy.deepcopy(evidence), copy.deepcopy({name: value for name, value in result.items() if name != '_model'}), list(dict.fromkeys(str(w.message) for w in scientific_warnings)), positions, converged_warning, copy.deepcopy(derived))
            return data, result, recorded, bool(convergence_warning(caught, True))

        self.cache[key] = entry
        data = copy.deepcopy(entry.data)
        params.update(copy.deepcopy(entry.derived))
        if self.preparation is not None:
            data.update(self.preparation.evidence(cast(dict[str, object], entry.data['_fitted_params'])))
        data['_alignment'] = copy.deepcopy(entry.data['_alignment'])
        data['_record_execution'] = self.record_execution
        data['_record_model_outputs'] = self.record_model_outputs
        with warnings.catch_warnings(record=True) as caught, native_threads():
            warnings.simplefilter('always')
            result = replay(self.manifest, self.contract, entry.result, data, params, entry.positions, self.record_execution)
        recorded = list(dict.fromkeys([*entry.warnings, *(str(w.message) for w in caught)]))
        return _composition.signal_result(self.manifest, self.data, data, result, recorded, entry.convergence, caught)
