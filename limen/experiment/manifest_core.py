from limen.backtest.trade_contract import TradePolicy
from limen.experiment._walk_forward_split import WalkForwardConfig, resolve_walk_forward_split as _walk_forward_split, validate_fold_splits as _validate_fold_splits
from limen.targets.trade_outcome import OutcomeLabels
from limen.experiment._prepare_trade_context import source_interval as _source_interval, attach_outcomes, resolve_component_kwargs, sensor_decisions, target_context
from limen.experiment._resolve_trade_policy import BacktestConfig, FundingConfig, ProductConfig, resolve_number as _resolve_trade_number, resolve_trade_policy as _resolve_trade_policy
from limen.experiment._prepare_trade_context import PreparedTradeContext, validate_cached_context as _validate_cached_context, finish_trade_result as _finish_trade_result, attach_trade_context as _attach_trade_context, prepare_trade_context as _prepare_trade_context
from limen.experiment._prepare_backtest_data import prepare_backtest_data as _prepare_backtest_data
from limen.experiment._objective import ObjectiveConfig, prepare_objective as _prepare_objective, score_objective as _score_objective, threshold_params as _objective_threshold_params
from limen.experiment._backtest_provenance import SOURCE_ROW as _SOURCE_ROW, attach_witness as _attach_witness, capture_backtest as _capture_backtest, restore_source_rows as _restore_source_rows, preflight_backtest as _preflight_backtest, validate_witness as _validate_witness
from collections.abc import Mapping
from limen.experiment._resolve_backtest_config import BACKTEST_KEYS, resolve_backtest_config as _resolve_backtest_config, configured_barriers as _configured_barriers
import copy
import inspect
import importlib
import logging
import random
import re
from functools import partial
from datetime import date
from datetime import datetime
from itertools import pairwise
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

if TYPE_CHECKING:
    from limen.sfd.rule_based.config import RuleBasedConfig

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.decomposition import PCA
from typing_extensions import override

from limen.calibration.pipeline import CalibratorProtocol
from limen.calibration.pipeline import ThresholdOptimizerProtocol
from limen.data.utils import split_by_dates
from limen.data.utils import split_data_to_prep_output
from limen.data.utils import split_data_to_rule_based_prep_output
from limen.data.utils import split_sequential
from limen.experiment.errors import StrictModeError
from limen.scalers.robust_scaler import RobustScaler
from limen.scalers.registry import SCALER_REGISTRY
logger = logging.getLogger(__name__)

_TManifest = TypeVar('_TManifest', bound='Manifest')

ParamValue = Any | Callable[[dict[str, Any]], Any]
PipelineStep = tuple[Callable[..., pl.DataFrame], dict[str, ParamValue]]

FittedParamsComputationEntry = tuple[str, Callable[..., Any], dict[str, ParamValue]]


@dataclass
class TransformEntry:

    '''Feature or indicator transform with optional perturbation metadata.'''

    func: Callable[..., Any]
    params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])
    group: str | None = None
    include_if: str | None = None


@dataclass
class AblationConfig:

    '''Configuration for random feature ablation (Drop-N).'''

    drop_count_key: str
    seed_key: str


@dataclass
class PCACompressionConfig:

    '''Configuration for optional manifest-level PCA feature compression.'''

    enabled_param: str
    n_components_param: str
    scaler_param_name: str
    component_prefix: str


FittedTransformEntry = tuple[
    list[FittedParamsComputationEntry],
    Callable[..., pl.DataFrame],
    dict[str, ParamValue]
]


@dataclass
class TargetClassConfig:

    '''Configuration for a class-based target transform.'''

    target_class: type
    fit_params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])
    transform_params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])


@dataclass
class CalibrationConfig:

    '''Stores probability calibration and threshold function references with their params.'''

    calibration_func: CalibratorProtocol | None = None
    calibration_params: dict[str, Any] = field(default_factory=dict[str, Any])
    threshold_func: ThresholdOptimizerProtocol | None = None
    threshold_params: dict[str, Any] = field(default_factory=dict[str, Any])

    def resolve(self, round_params: dict[str, Any]) -> 'CalibrationConfig':

        '''Return a new config with string params resolved from round_params.'''

        return CalibrationConfig(
            calibration_func=self.calibration_func,
            calibration_params=_resolve_params(self.calibration_params, round_params),
            threshold_func=self.threshold_func,
            threshold_params=_resolve_params(self.threshold_params, round_params),
        )


class CalibrationBuilder:

    '''Fluent builder for calibration configuration.'''

    def __init__(self, manifest: object) -> None:

        super().__init__()

        if not isinstance(manifest, MLManifest):
            raise ValueError(
                f"CalibrationBuilder requires an MLManifest, got {type(manifest).__name__}. Use MLManifest().with_calibration() to configure calibration."
            )
        self._manifest = manifest
        self._calibration_func: CalibratorProtocol | None = None
        self._calibration_params: dict[str, Any] = {}
        self._threshold_func: ThresholdOptimizerProtocol | None = None
        self._threshold_params: dict[str, Any] = {}

    def probability_calibration(self, func: CalibratorProtocol, **params: Any) -> 'CalibrationBuilder':

        '''Register a calibrator; resolve round-parameter references at runtime.'''

        self._calibration_func = func
        self._calibration_params = params
        return self

    def threshold_function(self, func: ThresholdOptimizerProtocol, **params: Any) -> 'CalibrationBuilder':

        '''Register a threshold optimizer; resolve round-parameter references at runtime.'''

        self._threshold_func = func
        self._threshold_params = params
        return self

    def done(self) -> 'MLManifest':

        '''
        Finalise calibration configuration and return the manifest.

        Returns:
            MLManifest: The parent manifest with prediction_calibration_config set

        Raises:
            ValueError: If neither probability_calibration() nor threshold_function() was called before done()
        '''

        if self._calibration_func is None and self._threshold_func is None:
            raise ValueError('CalibrationBuilder at least one of probability_calibration() or threshold_function() must be called before done()')
        self._manifest.prediction_calibration_config = CalibrationConfig(
            calibration_func=self._calibration_func,
            calibration_params=dict(self._calibration_params),
            threshold_func=self._threshold_func,
            threshold_params=dict(self._threshold_params),
        )
        return self._manifest


@dataclass
class DataSourceConfig:

    '''Declarative configuration for data fetching in manifests.'''

    method: Callable[..., Any]
    params: dict[str, Any] = field(default_factory=dict[str, Any])


class DataSourceResolver:

    '''Resolves data source config to DataFrame.'''

    @staticmethod
    def resolve(config: DataSourceConfig) -> pl.DataFrame:

        '''
        Execute data source config and return DataFrame.

        Args:
            config: DataSourceConfig instance

        Returns:
            pl.DataFrame: Fetched data
        '''

        method = config.method
        params = config.params

        bound_self = getattr(method, '__self__', None)
        if inspect.ismethod(method) or bound_self is not None:
            result = method(**params)
            if bound_self is not None and hasattr(bound_self, 'data'):
                return bound_self.data
            return result

        if inspect.isfunction(method):
            if '.' in method.__qualname__:
                module_name = method.__module__
                class_name = method.__qualname__.rsplit('.', 1)[0]

                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)

                instance = cls()
                bound_method = getattr(instance, method.__name__)
                bound_method(**params)

                if hasattr(instance, 'data'):
                    return instance.data
                raise ValueError(
                    f"DataSourceResolver Method {method.__qualname__} executed successfully but instance does not have 'data' attribute. Expected data source methods to populate instance.data"
                )
            return method(**params)

        raise ValueError(f"DataSourceResolver Unsupported callable type: {type(method)}")


@dataclass
class Manifest:

    '''Base manifest with shared data pipeline configuration for Loop experiments.'''

    data_source_config: DataSourceConfig | None = None
    test_data_source_config: DataSourceConfig | None = None
    pre_split_data_selector: PipelineStep | None = None
    split_config: tuple[int, int, int] = (8, 1, 2)
    split_dates: tuple[date | Any, date | Any, date | Any, date | Any, date | Any, date | Any] | None = None
    split_walk_forward: WalkForwardConfig | None = None
    _walk_forward_fold: int | None = field(default=None, repr=False)
    val_predict_guard: bool = True
    test_predict_guard: bool = True
    bar_formation: PipelineStep | None = None
    required_bar_columns: list[str] = field(default_factory=list[str])
    feature_transforms: list[TransformEntry] = field(default_factory=list[TransformEntry])
    target_column: str | None = None
    target_class_config: TargetClassConfig | None = None

    architecture_function: Callable[..., dict[str, Any]] | None = None
    architecture_params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])
    metrics_params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])
    backtest_config: BacktestConfig | None = None

    def _add_transform(self,
                       func: Callable[..., Any],
                       group: str | None = None,
                       include_if: str | None = None,
                       **params: Any) -> 'Manifest':

        entry = TransformEntry(
            func=func,
            params=params,
            group=group,
            include_if=include_if,
        )
        self.feature_transforms.append(entry)

        return self

    def set_data_source(self,
                       method: Callable[..., Any],
                       params: dict[str, Any] | None = None) -> 'Manifest':

        '''
        Configure production data source for the manifest.

        Args:
            method (Callable): Method or function reference (e.g., HistoricalData.get_spot_klines)
            params (dict): Parameters to pass to the method

        Returns:
            Manifest: Self for method chaining
        '''

        self.data_source_config = DataSourceConfig(
            method=method,
            params=params or {}
        )

        return self

    def set_test_data_source(self,
                            method: Callable[..., Any],
                            params: dict[str, Any] | None = None) -> 'Manifest':

        '''
        Configure test data source for the manifest.

        Args:
            method (Callable): Function reference (e.g., HistoricalData.get_spot_klines)
            params (dict): Parameters to pass to the function

        Returns:
            Manifest: Self for method chaining
        '''

        self.test_data_source_config = DataSourceConfig(
            method=method,
            params=params or {}
        )

        return self

    def fetch_data(self) -> pl.DataFrame:

        '''Fetch data using configured data source.'''

        if self.data_source_config is None:
            raise ValueError('Manifest No data source configured')

        return DataSourceResolver.resolve(self.data_source_config)

    def fetch_test_data(self) -> pl.DataFrame:

        '''Fetch data using configured test data source.'''

        if self.test_data_source_config is None:
            raise ValueError('Manifest No test data source configured')

        return DataSourceResolver.resolve(self.test_data_source_config)

    def add_feature(self,
                    func: Callable[..., Any],
                    group: str | None = None,
                    include_if: str | None = None,
                    **params: Any) -> 'Manifest':

        '''
        Add feature transformation to the manifest.

        Args:
            func (Callable): Feature transformation function
            group (str | None): Perturbation group tag for feature filtering
            include_if (str | None): round_params key that controls inclusion
            **params: Parameters for the transformation

        Returns:
            Manifest: Self for method chaining
        '''

        return self._add_transform(func, group=group, include_if=include_if, **params)


    def add_indicator(self,
                      func: Callable[..., Any],
                      group: str | None = None,
                      include_if: str | None = None,
                      **params: Any) -> 'Manifest':

        '''
        Add indicator transformation to the manifest.

        Args:
            func (Callable): Indicator transformation function
            group (str | None): Perturbation group tag for feature filtering
            include_if (str | None): round_params key that controls inclusion
            **params: Parameters for the transformation

        Returns:
            Manifest: Self for method chaining
        '''

        return self._add_transform(func, group=group, include_if=include_if, **params)

    def set_pre_split_data_selector(self, func: Callable[..., pl.DataFrame], **params: Any) -> 'Manifest':

        '''
        Set pre-split data selector function and parameters.

        Args:
            func (Callable): Data selector function
            **params: Parameters for data selection

        Returns:
            Manifest: Self for method chaining
        '''

        self.pre_split_data_selector = (func, params)
        return self

    def set_bar_formation(self, func: Callable[..., pl.DataFrame], **params: Any) -> 'Manifest':

        '''
        Set bar formation function and parameters.

        Args:
            func (Callable): Bar formation function
            **params: Parameters for bar formation

        Returns:
            Manifest: Self for method chaining
        '''

        self.bar_formation = (func, params)

        return self


    def set_required_bar_columns(self, columns: list[str]) -> 'Manifest':

        '''
        Set required columns after bar formation.

        Args:
            columns (List[str]): List of required column names

        Returns:
            Manifest: Self for method chaining
        '''

        self.required_bar_columns = columns

        return self

    def set_split_config(self, train: int, val: int, test: int) -> 'Manifest':

        '''
        Set data split configuration.

        Args:
            train (int): Training split ratio
            val (int): Validation split ratio
            test (int): Test split ratio

        Returns:
            Manifest: Self for method chaining

        Raises:
            ValueError: If train is not positive, or if val or test is negative
        '''

        if train <= 0:
            raise ValueError('Manifest train split ratio must be positive')
        if val < 0 or test < 0:
            raise ValueError('Manifest val and test split ratios must be non-negative')

        self.split_config = (train, val, test)

        return self

    def set_split_walk_forward(
        self: _TManifest, *, n_folds: int, test_bars: int,
        purge_bars: int, embargo_bars: int, anchored: bool,
    ) -> _TManifest:
        if self.split_dates is not None:
            raise ValueError('Manifest split_walk_forward conflicts with split_dates')
        self.split_walk_forward = WalkForwardConfig(n_folds, test_bars, purge_bars, embargo_bars, anchored)
        self._walk_forward_fold = None
        return self

    def set_split_dates(
        self,
        train_start: date | Any, train_end: date | Any,
        val_start: date | Any, val_end: date | Any,
        test_start: date | Any, test_end: date | Any,
        *,
        val_predict_guard: bool | Any = True,
        test_predict_guard: bool | Any = True,
    ) -> 'Manifest':

        '''Pin half-open date windows; guards mask validation/test Sensor predictions.'''

        if self.split_walk_forward is not None:
            raise ValueError('Manifest split_dates conflicts with split_walk_forward')

        bounds = [
            ('train_start', train_start), ('train_end', train_end),
            ('val_start', val_start), ('val_end', val_end),
            ('test_start', test_start), ('test_end', test_end),
        ]
        for name, value in bounds:
            if not isinstance(value, date):
                raise TypeError(
                    f"Manifest {name} must be a date or datetime instance, got {type(value).__name__}: {value!r}"
                )
        for (a_name, a), (b_name, b) in pairwise(bounds):
            if a > b:
                raise ValueError(
                    f"Manifest {a_name}={a!r} must be <= {b_name}={b!r}; bounds must be in non-decreasing order (gaps between adjacent windows allowed)"
                )

        for name, flag in (('val_predict_guard', val_predict_guard),
                           ('test_predict_guard', test_predict_guard)):
            if not isinstance(flag, bool):
                raise TypeError(
                    f"Manifest {name} must be a bool, got {type(flag).__name__}: {flag!r}"
                )

        self.split_dates = (
            train_start, train_end,
            val_start, val_end,
            test_start, test_end,
        )
        self.val_predict_guard = val_predict_guard
        self.test_predict_guard = test_predict_guard

        return self

    def with_target_label(self,
                          target_name: str,
                          target_class: type,
                          fit_params: dict[str, Any] | None = None,
                          transform_params: dict[str, Any] | None = None) -> 'Manifest':

        '''
        Configure a class-based target transform.

        The class must accept (train_data, target_name, **fit_params) in __init__
        and expose transform(data, **transform_params) -> pl.DataFrame.
        Fitting happens once on the training split; the fitted instance is reused
        for validation and test splits.

        Args:
            target_name (str): Name of the target column to create
            target_class (type): Target class whose __init__ accepts (train_data, target_name, **fit_params)
                and whose transform() accepts (data, **transform_params) returning a pl.DataFrame
            fit_params (dict[str, ParamValue]): Parameters forwarded to __init__ after train_data and target_name
            transform_params (dict[str, ParamValue]): Parameters forwarded to transform()

        Returns:
            Manifest: Self for method chaining
        '''

        self.target_column = target_name
        self.target_class_config = TargetClassConfig(
            target_class=target_class,
            fit_params=dict(fit_params) if fit_params else {},
            transform_params=dict(transform_params) if transform_params else {},
        )
        return self

    def with_reference_architecture(self: _TManifest, architecture_function: Callable[..., dict[str, Any]]) -> _TManifest:

        '''
        Configure reference architecture function for training and evaluation.

        Args:
            architecture_function (Callable): Architecture function that takes (data, **params) and returns results

        Returns:
            Manifest: Self for method chaining

        NOTE: The architecture function should accept data dict and return results dict with metrics and predictions.
        Parameters are auto-mapped from round_params based on function signature.
        '''

        self.architecture_function = architecture_function

        return self

    def with_params_override(self, **overrides: Any) -> 'Manifest':

        '''
        Create a deep copy of this manifest with overridden parameters.

        Args:
            **overrides: Parameters to override. 'split_config' overrides the split
                ratios directly. All other keys are treated as data source param
                overrides and are validated against the data source method signature

        Returns:
            Manifest: New manifest with overridden parameters

        Raises:
            ValueError: If split ratios are invalid or a key is not
                'split_config' and not accepted by the data source method
        '''

        new_manifest = copy.deepcopy(self)

        if 'split_config' in overrides:
            sc = overrides['split_config']
            _split_len = 3
            if not isinstance(sc, tuple):
                raise ValueError(f"split_config must be a 3-tuple of ints, got {sc!r}")
            sc = cast(tuple[Any, ...], sc)
            if not (len(sc) == _split_len
                    and all(isinstance(v, int) and not isinstance(v, bool) for v in sc)):
                raise ValueError(f"split_config must be a 3-tuple of ints, got {sc!r}")
            if any(v < 0 for v in sc):
                raise ValueError(f"split_config ratios must be non-negative, got {sc!r}")
            if sum(sc) == 0:
                raise ValueError('split_config ratios must not all be zero')
            _ = new_manifest.set_split_config(*cast(tuple[int, int, int], sc))
            # Ratio override supersedes a previously-pinned date split.
            # Without this, _resolve_split would keep using split_dates and
            # the override would silently no-op (e.g. Trainer.train_sensors
            # passing (1, 0, 0) to retrain on all available data).
            new_manifest.split_dates = None

        ds_overrides = {k: v for k, v in overrides.items() if k != 'split_config'}
        if ds_overrides:
            if new_manifest.data_source_config is None:
                raise ValueError('Manifest Cannot override data source params: no data source configured')
            method_params = set(inspect.signature(
                new_manifest.data_source_config.method
            ).parameters.keys()) - {'self', 'cls'}
            unknown = set(ds_overrides) - method_params
            if unknown:
                raise ValueError(
                    f"Manifest Unknown data source params: {sorted(unknown)}. Accepted by {new_manifest.data_source_config.method.__name__}: {sorted(method_params)}"
                )
            new_manifest.data_source_config.params = dict(new_manifest.data_source_config.params)
            new_manifest.data_source_config.params.update(ds_overrides)

        return new_manifest

    def compute_test_bars(self, raw_data: pl.DataFrame, round_params: dict[str, Any]) -> pl.DataFrame:

        '''
        Compute test split bar data from raw data using manifest bar formation configuration.

        NOTE: Used by Log system to reconstruct the same test bar data that was used in training.

        Args:
            raw_data (pl.DataFrame): Raw input dataset
            round_params (Dict[str, Any]): Parameter values for current round

        Returns:
            pl.DataFrame: Bar-formed test split data
        '''

        if self.pre_split_data_selector:
            func, base_params = self.pre_split_data_selector
            resolved = _resolve_params(base_params, round_params)
            raw_data = func(raw_data, **resolved)

        # compute_test_bars consumes only split_data[2], so materialising all
        # three windows here would do 2x wasted filter work on the date path
        # (3x O(N) filters vs the legacy O(1) slice, with two thirds discarded).
        # The Log system calls this per round; the waste compounds. Take the
        # one slice we actually need.
        if self.split_walk_forward is not None:
            test_split = _resolve_split(self, raw_data)[2]
        elif self.split_dates is not None:
            *_, test_start, test_end = self.split_dates
            test_split = raw_data.filter(
                (pl.col('datetime') >= test_start) & (pl.col('datetime') < test_end)
            )
        else:
            test_split = split_sequential(raw_data, self.split_config)[2]
        _, test_bar_data = _process_bars(self, test_split, round_params)

        return test_bar_data

    def prepare_data(
        self,
        raw_data: pl.DataFrame,
        round_params: dict[str, Any]
    ) -> dict[str, Any]:

        '''
        Interface method — implemented by MLManifest and RuleBasedManifest.

        Raises:
            NotImplementedError: Always. Construct MLManifest or RuleBasedManifest instead of Manifest directly.
        '''

        raise NotImplementedError(
            f"{type(self).__name__} must implement prepare_data(). Use MLManifest for ML pipelines or RuleBasedManifest for rule-based pipelines."
        )

    def resolve_model_kwargs(self, round_params: dict[str, Any]) -> dict[str, Any]:

        '''
        Resolve model function kwargs from round_params using signature inspection.

        Maps round_params keys to model function parameters, falling back
        to defaults for unspecified parameters.

        Args:
            round_params (dict[str, Any]): Parameter values for current round

        Returns:
            dict[str, Any]: Keyword arguments for the model function

        Raises:
            ValueError: If model function is not configured or required parameters
                are missing from round_params

        '''

        if self.architecture_function is None:
            raise ValueError('Manifest Architecture function not configured. Use .with_reference_architecture(func) before run_model() or resolve_model_kwargs().')

        sig = inspect.signature(self.architecture_function)
        model_kwargs: dict[str, Any] = {}
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        for param_name, param_obj in sig.parameters.items():
            if param_name == 'data':
                continue
            if param_obj.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                continue

            if param_name in round_params:
                model_kwargs[param_name] = round_params[param_name]
            elif param_obj.default != inspect.Parameter.empty:
                model_kwargs[param_name] = param_obj.default
            else:
                raise ValueError(
                    f"Manifest Missing required parameter '{param_name}' for model function. It must be provided in round_params."
                )

        if has_var_keyword:
            for k, v in round_params.items():
                if not k.startswith('_') and k not in model_kwargs:
                    model_kwargs[k] = v

        return model_kwargs


    def set_backtest_config(self, fee_bps: float | str = 5.0, slip_bps: float | str = 5.0,
                            notional_rate: float | str = 1.0, *, take_profit_bps: float | str | None = None,
                            stop_loss_bps: float | str | None = None,
                            prediction_mode: Literal['binary', 'target_exposure'] = 'binary',
                            product: ProductConfig | None = None, initial_equity: float | str = 10000.0,
                            max_exposure: float | str = 1.0, signal_change_bps: float | str = 0.0,
                            flat_threshold: float | str = 0.0, max_holding_seconds: float | str | None = None,
                            timer_interval_seconds: float | str | None = None, timer_phase_utc_seconds: float | str = 0.0,
                            execution_lag_seconds: float | str = 0.0, max_price_gap_seconds: float | str | None = None,
                            execution_data_source: DataSourceConfig | None = None, funding: FundingConfig | None = None) -> 'Manifest':
        self.backtest_config = BacktestConfig(fee_bps, slip_bps, notional_rate, take_profit_bps, stop_loss_bps,
            prediction_mode, product, initial_equity, max_exposure, signal_change_bps, flat_threshold,
            max_holding_seconds, timer_interval_seconds, timer_phase_utc_seconds, execution_lag_seconds,
            max_price_gap_seconds, execution_data_source, funding)
        return self

    def resolve_trade_policy(self, round_params: Mapping[str, object]) -> TradePolicy | None:
        return _resolve_trade_policy(self.backtest_config, round_params)

    def resolve_backtest_config(self, round_params: Mapping[str, object]) -> dict[str, float | None]:
        """Resolve all backtest fields; an absent configuration returns an empty mapping."""
        return _resolve_backtest_config(self.backtest_config, round_params)

    def _apply_backtest_cost(self, data: dict[str, Any], round_params: dict[str, Any]) -> None:
        policy = self.resolve_trade_policy(round_params)
        _validate_cached_context(self.backtest_config, policy, data, round_params)
        if _configured_barriers(self.backtest_config) and policy is None:
            data['_backtest_configured'] = True
        else:
            data.pop('_backtest_configured', None)
        data.pop('_trade_ledger', None)
        _preflight_backtest(data)
        resolved = self.resolve_backtest_config(round_params)
        for key in BACKTEST_KEYS:
            if key in resolved:
                data[f'backtest_{key}'] = resolved[key]
            else:
                data.pop(f'backtest_{key}', None)

    def run_model(self, data: dict[str, Any], round_params: dict[str, Any]) -> dict[str, Any]:

        '''
        Execute model training and evaluation using configured functions.

        Args:
            data (dict): Prepared data dictionary
            round_params (dict[str, Any]): Parameter values for current round

        Returns:
            dict: Results including predictions, metrics, and optional extras

        Raises:
            ValueError: If required model function parameters are missing from round_params

        NOTE: Auto-maps parameters from round_params to model function signature.
        Parameters in round_params override model function defaults.
        Parameters not in round_params use model function defaults.
        Required parameters (no defaults) must be in round_params.
        '''

        model_kwargs = self.resolve_model_kwargs(round_params)
        self._apply_backtest_cost(data, round_params)
        if self.architecture_function is None:
            raise ValueError('Manifest run_model requires a configured architecture_function')
        return _finish_trade_result(data, self.architecture_function(data, **model_kwargs))


@dataclass
class MLManifest(Manifest):

    '''Manifest for ML pipelines with scaler, ablation, and calibration support.'''

    scaler: FittedTransformEntry | None = None
    ablation_config: AblationConfig | None = None
    pca_compression_config: PCACompressionConfig | None = None
    data_dict_extension: Callable[..., dict[str, Any]] | None = None
    prediction_calibration_config: CalibrationConfig | None = None
    decoder_lookback: int = 1
    strict_mode: bool = False
    objective: ObjectiveConfig | None = None

    def set_objective(self, metric: str = 'backtest_total_return', direction: str = 'maximize') -> 'MLManifest':
        self.objective = ObjectiveConfig(metric, direction)
        return self

    def set_scaler(self,
                   transform_class: Any,
                   param_name: str = '_scaler',
                   extra_params: dict[str, Any] | None = None) -> 'MLManifest':

        '''
        Set scaler transformation using make_fitted_scaler.

        Args:
            transform_class: Transform class to use for scaling
            param_name (str): Parameter name for fitted scaler
            extra_params (dict | None): Additional keyword arguments passed to the transform constructor

        Returns:
            MLManifest: Self for method chaining
        '''

        self.scaler = make_fitted_scaler(param_name, transform_class, extra_params)

        return self


    def set_strict_mode(self, strict_mode: bool) -> 'MLManifest':

        '''
        Enable or disable strict mode for null detection after CCO dislodgement.

        Args:
            strict_mode (bool): If True, unexpected nulls in feature columns raise StrictModeError

        Returns:
            MLManifest: Self for method chaining
        '''

        self.strict_mode = strict_mode
        return self


    def set_scaler_from_params(self,
                               param_name: str = 'scaler_type',
                               extra_params: dict[str, Any] | None = None) -> 'MLManifest':

        '''
        Configure scaler selection from round_params at runtime.

        The scaler class is resolved from the scaler registry using
        the value of round_params[param_name].

        Args:
            param_name (str): round_params key that holds the scaler type string

        Returns:
            MLManifest: Self for method chaining
        '''

        _static, _dynamic = _split_extra_params(extra_params)

        def _scaler_factory(data: 'pl.DataFrame',
                            scaler_type: str = '',
                            **dyn: Any) -> Any:

            if scaler_type not in SCALER_REGISTRY:
                if scaler_type == param_name:
                    raise ValueError(
                        f"round_params['{param_name}'] is required when using set_scaler_from_params(). Available types: {sorted(SCALER_REGISTRY)}"
                    )
                raise ValueError(
                    f"MLManifest Unknown scaler type '{scaler_type}'. Available: {sorted(SCALER_REGISTRY)}"
                )
            return SCALER_REGISTRY[scaler_type](data, **_static, **dyn)

        self.scaler = (
            [('_scaler', _scaler_factory, {'scaler_type': param_name, **_dynamic})],
            _apply_fitted_transform,
            {'fitted_transform': '_scaler'},
        )

        return self

    def set_feature_ablation(self,
                             drop_count_key: str = 'feature_drop_count',
                             seed_key: str = 'feature_drop_seed') -> 'MLManifest':

        '''
        Configure random feature ablation (Drop-N).

        Randomly drops N feature columns per permutation using a
        deterministic seed from round_params. Runs after feature and
        target transforms in the prepare_data pipeline.

        Args:
            drop_count_key (str): round_params key for number of columns to drop
            seed_key (str): round_params key for random seed

        Returns:
            MLManifest: Self for method chaining
        '''

        self.ablation_config = AblationConfig(
            drop_count_key=drop_count_key,
            seed_key=seed_key,
        )
        return self


    def set_pca_compression(self,
                            enabled_param: str | Any = 'auto_pca',
                            n_components_param: str | Any = 'pca_k',
                            scaler_param_name: str | Any = '_scaler',
                            component_prefix: str | Any = 'pc_') -> 'MLManifest':

        '''
        Configure optional PCA compression over the finalized feature surface.

        Args:
            enabled_param (str): round_params key that enables PCA when True
            n_components_param (str): round_params key for PCA component count
            scaler_param_name (str): fitted RobustScaler key in data_dict
            component_prefix (str): Prefix for emitted component columns

        Returns:
            MLManifest: Self for method chaining
        '''

        for name, value in {
            'enabled_param': enabled_param,
            'n_components_param': n_components_param,
            'scaler_param_name': scaler_param_name,
            'component_prefix': component_prefix,
        }.items():
            if not isinstance(value, str) or not value:
                raise TypeError(f'MLManifest {name} must be a non-empty string')

        self.pca_compression_config = PCACompressionConfig(
            enabled_param=enabled_param,
            n_components_param=n_components_param,
            scaler_param_name=scaler_param_name,
            component_prefix=component_prefix,
        )
        return self


    def add_to_data_dict(self, func: Callable[..., dict[str, Any]]) -> 'MLManifest':

        '''
        Configure data_dict extension function to add custom entries after data preparation.

        Args:
            func (Callable): Extension function with signature (data_dict, split_data, round_params, fitted_params) -> dict

        Returns:
            MLManifest: Self for method chaining

        NOTE: The extension function receives the base data_dict and full split DataFrames.
        It should modify and return the data_dict with any additional custom entries needed by the model.
        '''

        self.data_dict_extension = func
        return self

    def with_calibration(self) -> 'CalibrationBuilder':

        '''
        Configure calibration or threshold selection; call done() to finalise.
        '''

        return CalibrationBuilder(self)

    @override
    def prepare_data(
        self,
        raw_data: pl.DataFrame,
        round_params: dict[str, Any]
    ) -> dict[str, Any]:


        split_data, all_datetimes, price_data_for_backtest, sources, trade = _run_prepare_setup(self, raw_data, round_params)
        outcome_target = self.target_class_config is not None and bool(getattr(self.target_class_config.target_class, 'requires_trade_context', False))
        private_labels: list[OutcomeLabels] = []
        raw_features: pl.DataFrame | None = None

        all_fitted_params: dict[str, Any] = {}
        columns_to_drop: list[str] | None = None
        pre_transform_columns = frozenset(split_data[0].columns)

        cco_indicator_rows: int = 0
        scaler_context_rows: int = 0
        n_raw_cco: int = 0
        cco_block: pl.DataFrame | None = None

        for i, split in enumerate(split_data):
            is_train = i == 0

            if not is_train and cco_block is not None and n_raw_cco > 0:
                if scaler_context_rows > 0 and len(cco_block) < n_raw_cco:
                    split_name = _SPLIT_NAMES[i] if i < len(_SPLIT_NAMES) else str(i)
                    shortfall = n_raw_cco - len(cco_block)
                    msg = (
                        f"under-warmed CCO: {split_name} split requires {n_raw_cco} context rows but only {len(cco_block)} available (shortfall {shortfall})"
                    )
                    if self.strict_mode:
                        raise StrictModeError(msg)
                    logger.warning(msg)
                raw_input = pl.concat([cco_block, split])
            else:
                raw_input = split

            lazy = raw_input.lazy()
            lazy = _apply_feature_transforms(self, lazy, round_params)
            data = lazy.collect()

            if self.target_class_config is not None:
                data, all_fitted_params = _apply_class_based_target(
                    self, data, round_params, all_fitted_params, is_train,
                    trade_context=target_context(trade, i) if outcome_target else None, labels=private_labels
                )

            if self.ablation_config is not None:
                data, columns_to_drop = _apply_feature_ablation(
                    data, self, round_params, columns_to_drop, pre_transform_columns,
                )

            data = data.fill_nan(None)
            feature_surface = data.drop(self.target_column) if outcome_target and self.target_column is not None else data
            n_leading = _count_leading_nulls(feature_surface)
            data = data.slice(n_leading)

            n_cco_feature_rows = max(0, len(cco_block) - n_leading) if (not is_train and cco_block is not None) else 0

            _check_unexpected_nulls(self, data, i, 'A')

            if is_train and outcome_target:
                raw_features = data.drop([name for name in (self.target_column, _SOURCE_ROW) if name is not None and name in data.columns])
            data, all_fitted_params = _apply_scaler(self, data, round_params, all_fitted_params, is_train)

            if n_cco_feature_rows > 0:
                data = data.slice(n_cco_feature_rows)

            _check_unexpected_nulls(self, data, i, 'B')

            subset = [name for name in data.columns if name != self.target_column] if outcome_target else data.columns
            split_data[i] = data.fill_nan(None).drop_nulls(subset=subset)

            if is_train:
                cco_indicator_rows = n_leading
                scaler_context_rows = max(
                    (getattr(v, 'context_rows', 0) for v in all_fitted_params.values()),
                    default=0,
                )
                n_raw_cco = cco_indicator_rows + scaler_context_rows

            cco_block = split.tail(n_raw_cco) if n_raw_cco > 0 else None

        split_data, witness = _capture_backtest(sources, split_data, ml=True)
        if _configured_barriers(self.backtest_config) and self.resolve_trade_policy(round_params) is None:
            _ = _validate_witness(witness)
        split_data = _align_split_columns(split_data)
        split_data, all_fitted_params = _apply_pca_compression(
            self, split_data, round_params, all_fitted_params
        )

        if price_data_for_backtest is not None:
            final_datetimes = split_data[2].select('datetime')
            price_data_for_backtest = final_datetimes.join(
                # colliding bars share the first bar's OHLC; see #557
                price_data_for_backtest.unique(
                    subset='datetime', keep='first', maintain_order=True
                ),
                on='datetime', how='left',
                maintain_order='left'
            )

        if self.split_walk_forward is not None:
            _validate_fold_splits(split_data, require_validation=_requires_fold_validation(self, round_params))
        data_dict = _finalize_to_data_dict(self, split_data, all_datetimes, all_fitted_params, round_params, price_data_for_backtest)
        _attach_witness(data_dict, witness)
        _attach_trade_context(data_dict, trade, split_data)
        if outcome_target:
            attach_outcomes(data_dict, private_labels, raw_features, split_data[0],
                scale=partial(_apply_scaler, self, round_params=round_params), compress=partial(_apply_pca_compression, self, round_params=round_params),
                deterministic=self.scaler is None and all(bool(getattr(entry.func, 'deterministic', False)) for entry in self.feature_transforms))
        return data_dict

    @override
    def run_model(self, data: dict[str, Any], round_params: dict[str, Any]) -> dict[str, Any]:


        model_kwargs = self.resolve_model_kwargs(round_params)
        if self.architecture_function is None:
            raise ValueError('MLManifest run_model requires a configured architecture_function')
        if self.prediction_calibration_config is not None:
            use_calibration = round_params.get('use_calibration', True)
            use_threshold = round_params.get('use_threshold', True)
            if use_calibration or use_threshold:
                arch_sig = inspect.signature(self.architecture_function)
                accepts_config = (
                    'prediction_calibration_config' in arch_sig.parameters
                    or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in arch_sig.parameters.values()
                    )
                )
                if not accepts_config:
                    raise ValueError(
                        'MLManifest Calibration is configured but the architecture function does not accept `prediction_calibration_config`. Add it as a named parameter or use **kwargs.'
                    )
                resolved = self.prediction_calibration_config.resolve(round_params)
                config = CalibrationConfig(
                    calibration_func=resolved.calibration_func if use_calibration else None,
                    calibration_params=resolved.calibration_params,
                    threshold_func=resolved.threshold_func if use_threshold else None,
                    threshold_params=resolved.threshold_params,
                )
                model_kwargs['prediction_calibration_config'] = config
        self._apply_backtest_cost(data, round_params)
        resolve_component_kwargs(self.architecture_function, data, model_kwargs, round_params)
        scorer = None
        if self.objective is not None:
            scorer = _prepare_objective(data, self.architecture_function, self.prediction_calibration_config)
            config = model_kwargs.get('prediction_calibration_config')
            if config is not None:
                if not isinstance(config, CalibrationConfig):
                    raise ValueError('Objective requires a manifest calibration configuration')
                model_kwargs['prediction_calibration_config'] = replace(config, threshold_params=_objective_threshold_params(config, scorer, self.objective))
        result = self.architecture_function(data, **model_kwargs)
        if scorer is not None and self.objective is not None:
            result[self.objective.column] = _score_objective(data, result, scorer)
        return _finish_trade_result(data, result)

    def sensor_input_prep(
            self,
            raw_klines: pl.DataFrame,
            fitted_params: dict[str, Any],
            round_params: dict[str, Any],
    ) -> tuple[pl.DataFrame, int]:

        '''
        Prepare raw klines for live inference without splitting or dropping nulls.

        Args:
            raw_klines (pl.DataFrame): Raw klines covering at least indicator warm-up + decoder window
            fitted_params (dict[str, Any]): Scaler and PCA state stored from training
            round_params (dict[str, Any]): Parameter values from the winning round

        Returns:
            tuple[pl.DataFrame, int]: Prepared feature DataFrame and indicator_lookback count

        NOTE: Does not call drop_nulls. Leading null rows represent indicator warm-up bars.
        The returned indicator_lookback count is the number of leading null rows so the
        caller can determine which bars are valid for prediction.
        '''

        _, data = _process_bars(self, raw_klines, round_params)
        decisions = None
        if self.resolve_trade_policy(round_params) is not None:
            interval = _source_interval(self.data_source_config)
            interval = None if interval is None else _resolve_trade_number(interval, round_params, 'recorded source interval')
            decisions = sensor_decisions(raw_klines, data, interval_seconds=interval)
            data = data.drop([name for name in ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns') if name in data.columns])

        lazy = data.lazy()
        lazy = _apply_feature_transforms(self, lazy, round_params)
        data = lazy.collect()

        dropped_features: list[str] = round_params.get('_dropped_features') or []
        if dropped_features:
            data = data.drop([c for c in dropped_features if c in data.columns])

        data = data.fill_nan(None)
        indicator_lookback = _count_leading_nulls(data)

        all_fitted_params = dict(fitted_params)
        data, _ = _apply_scaler(self, data, round_params, all_fitted_params, is_training=False)
        data = data.fill_nan(None)

        data = _apply_sensor_pca(self, data, round_params, all_fitted_params)

        if decisions is not None:
            data = data.join(decisions, on='datetime', how='left', maintain_order='left')
        return data, indicator_lookback


@dataclass
class RuleBasedManifest(Manifest):

    '''Manifest for rule-based pipelines with predicate conditions and entry signals.'''

    strategy: 'RuleBasedConfig | None' = field(default=None, init=False, repr=False)

    def with_strategy(self, conditions: list[dict[str, Any]], entry: str) -> 'RuleBasedManifest':

        '''
        Configure rule-based strategy conditions and entry signal.

        Args:
            conditions (list[dict]): List of predicate and compound operator condition configs
            entry (str): ID of the condition that produces the per-bar position signal

        Returns:
            RuleBasedManifest: Self for method chaining
        '''

        from limen.sfd.rule_based.config import RuleBasedConfig  # local to avoid circular import
        self.strategy = RuleBasedConfig(conditions=list(conditions), entry=entry)

        return self

    @override
    def prepare_data(
        self,
        raw_data: pl.DataFrame,
        round_params: dict[str, Any]
    ) -> dict[str, Any]:

        '''
        Compute final data dictionary from raw data using the rule-based pipeline.

        Args:
            raw_data (pl.DataFrame): Raw input dataset
            round_params (Dict[str, Any]): Parameter values for current round

        Returns:
            dict: Final data dictionary ready for model training
        '''

        if self.strategy is None:
            raise ValueError(
                'RuleBasedManifest.prepare_data() called without a strategy. Call with_strategy(conditions, entry=...) before running.'
            )

        split_data, all_datetimes, _, sources, trade = _run_prepare_setup(self, raw_data, round_params)

        all_fitted_params: dict[str, Any] = {}

        for i, split in enumerate(split_data):
            lazy = split.lazy()
            lazy = _apply_feature_transforms(self, lazy, round_params)
            data = lazy.collect()

            if self.target_class_config is not None:
                data, all_fitted_params = _apply_class_based_target(
                    self, data, round_params, all_fitted_params, i == 0
                )

            data = data.fill_nan(None).drop_nulls()
            split_data[i] = data.fill_nan(None).drop_nulls()

        split_data = _align_split_columns(split_data)
        if self.split_walk_forward is not None:
            _validate_fold_splits(split_data, require_validation=_requires_fold_validation(self, round_params))

        split_data, witness = _capture_backtest(sources, split_data, ml=False)
        if _configured_barriers(self.backtest_config) and self.resolve_trade_policy(round_params) is None:
            _ = _validate_witness(witness)
        data_dict = _finalize_rule_based_data(self, split_data, all_datetimes, round_params)
        _attach_witness(data_dict, witness)
        _attach_trade_context(data_dict, trade, split_data)
        return data_dict


def _apply_fitted_transform(data: pl.DataFrame, fitted_transform: Any) -> pl.DataFrame:

    '''
    Compute transformed data using fitted transform instance.

    Args:
        data (pl.DataFrame): Data to transform
        fitted_transform: Fitted transform instance with .transform() method

    Returns:
        pl.DataFrame: Transformed data
    '''

    return fitted_transform.transform(data)


def _split_extra_params(extra_params: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    _extra = dict(extra_params or {})
    _static = {k: v for k, v in _extra.items() if not isinstance(v, str)}
    _dynamic = {k: v for k, v in _extra.items() if isinstance(v, str)}
    return _static, _dynamic


def make_fitted_scaler(param_name: str,
                       transform_class: Any,
                       extra_params: dict[str, Any] | None = None) -> FittedTransformEntry:

    '''
    Create fitted transform entry for scaling.

    Args:
        param_name (str): Name for the fitted parameter
        transform_class: Transform class to instantiate
        extra_params (dict | None): Additional keyword arguments passed to the transform constructor

    Returns:
        FittedTransformEntry: Complete fitted transform configuration
    '''

    _static, _dynamic = _split_extra_params(extra_params)

    def _factory(data: 'pl.DataFrame',
                 _cls: Any = transform_class,
                 _p: dict[str, Any] = _static,
                 **dyn: Any) -> Any:
        return _cls(data, **_p, **dyn)

    return (
        [(param_name, _factory, _dynamic)],
        _apply_fitted_transform,
        {'fitted_transform': param_name},
    )


def _resolve_params(params: dict[str, Any], round_params: dict[str, Any]) -> dict[str, Any]:

    '''
    Resolve parameters using just-in-time detection with actual round_params.

    Args:
        params (Dict[str, Any]): Parameter specification dictionary
        round_params (Dict[str, Any]): Round-specific parameter values

    Returns:
        Dict[str, Any]: Resolved parameter dictionary
    '''

    resolved: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str):
            if value in round_params:
                resolved[key] = round_params[value]
            elif value.startswith('_'):
                resolved[key] = value
            elif '{' in value and '}' in value:
                m = re.fullmatch(r'\{(\w+)\}', value.strip())
                if m:
                    resolved[key] = round_params[m.group(1)]
                else:
                    # template string like "roc_{roc_period}" — format produces a string
                    resolved[key] = value.format(**round_params)
            else:
                resolved[key] = value
        else:
            resolved[key] = value

    return resolved


def _process_bars(
        manifest: Manifest,
        data: pl.DataFrame,
        round_params: dict[str, Any]
) -> tuple[list[datetime] | list[int], pl.DataFrame]:

    '''
    Compute bar formation on data and return post-bar datetimes.

    Args:
        manifest (Manifest): Experiment manifest containing bar formation config
        data (pl.DataFrame): Input raw dataset
        round_params (Dict[str, Any]): Parameter values for current round

    Returns:
        Tuple[List, pl.DataFrame]: Post-bar datetimes and processed data
    '''

    if manifest.bar_formation and round_params.get('bar_type', 'base') != 'base':
        func, base_params = manifest.bar_formation
        resolved = _resolve_params(base_params, round_params)
        bar_data = data.pipe(func, **resolved)
        all_datetimes = bar_data['datetime'].to_list()
    else:
        all_datetimes = data['datetime'].to_list()
        bar_data = data

    # Validate required columns are present after bar formation
    available_cols = list(bar_data.columns)
    for required_col in manifest.required_bar_columns:
        if required_col not in available_cols:
            raise AssertionError(f"Required bar column '{required_col}' not found after bar formation")

    return all_datetimes, bar_data


def _should_include_transform(entry: TransformEntry, round_params: dict[str, Any]) -> bool:

    if entry.include_if is not None:
        if entry.include_if not in round_params:
            return False
        flag = round_params[entry.include_if]
        if not isinstance(flag, bool):
            raise TypeError(
                f"round_params['{entry.include_if}'] must be a bool, got {flag!r}"
            )
        if not flag:
            return False

    if entry.group is None:
        return True

    return _is_group_active(entry.group, round_params)


def _is_group_active(group: str, round_params: dict[str, Any]) -> bool:

    '''
    Check whether a feature group is active for the current round.

    The 'feature_groups' round param is a pipe-delimited string of
    active group names. The sentinel value 'all' activates every group.
    When absent or None, all groups are active by default.

    Args:
        group (str): Group name to check
        round_params (dict[str, Any]): Current round parameters

    Returns:
        bool: Whether the group should be included
    '''

    feature_groups = round_params.get('feature_groups')
    if feature_groups is None or feature_groups == 'all':
        return True

    if not isinstance(feature_groups, str):
        raise TypeError(
            f"round_params['feature_groups'] must be a string, got {type(feature_groups).__name__}"
        )

    return group in feature_groups.split('|')


_SPLIT_NAMES = ['train', 'val', 'test']


def _check_unexpected_nulls(manifest: 'MLManifest',
                             data: pl.DataFrame,
                             split_index: int,
                             checkpoint: str) -> None:

    exclude = {'datetime', manifest.target_column}
    feature_cols = [c for c in data.columns if c not in exclude]
    counts = data.select([pl.col(c).null_count().alias(c) for c in feature_cols])
    null_info: dict[str, list[Any]] = {}
    for col in feature_cols:
        if counts[col][0] > 0:
            null_info[col] = data.filter(pl.col(col).is_null())['datetime'].to_list()
    if not null_info:
        return

    split_name = _SPLIT_NAMES[split_index] if split_index < len(_SPLIT_NAMES) else str(split_index)
    detail = '; '.join(f"{col} @ {ts}" for col, ts in null_info.items())
    msg = f"Unexpected nulls in {split_name} split · Checkpoint {checkpoint} · {detail}"

    if manifest.strict_mode:
        raise StrictModeError(msg)
    logger.warning(msg)


def _count_leading_nulls(data: pl.DataFrame) -> int:

    feature_cols = [c for c in data.columns if c != 'datetime']
    if not feature_cols:
        return 0
    null_mask = pl.any_horizontal([pl.col(c).is_null() for c in feature_cols])
    has_null = data.select(null_mask.alias('_has_null'))['_has_null'].cast(pl.UInt8)
    first_valid = has_null.arg_min()
    if first_valid is None or bool(has_null[first_valid]):
        return len(data)
    return int(first_valid)


def _apply_sensor_pca(
        manifest: 'MLManifest',
        data: pl.DataFrame,
        round_params: dict[str, Any],
        all_fitted_params: dict[str, Any],
) -> pl.DataFrame:

    config = manifest.pca_compression_config
    if config is None:
        return data

    enabled = round_params.get(config.enabled_param, False)
    if not enabled:
        return data

    pca = all_fitted_params.get('_pca')
    input_feature_names = all_fitted_params.get('_pca_input_feature_names')
    component_cols = all_fitted_params.get('_pca_feature_names')

    if pca is None or input_feature_names is None or component_cols is None:
        raise ValueError(
            'PCA was not fitted — fitted_params missing _pca, _pca_input_feature_names, or _pca_feature_names'
        )

    target_col = manifest.target_column
    null_mask = data.select(
        pl.any_horizontal([pl.col(c).is_null() for c in input_feature_names]).alias('_is_null')
    )['_is_null'].to_list()
    valid_indices = [i for i, is_null in enumerate(null_mask) if not is_null]

    n_rows = len(data)
    component_arrays: dict[str, list[float | None]] = {col: [None] * n_rows for col in component_cols}
    if valid_indices:
        valid_np = data[valid_indices].select(input_feature_names).to_numpy()
        components = pca.transform(valid_np)
        for arr_idx, row_idx in enumerate(valid_indices):
            for col_idx, col in enumerate(component_cols):
                component_arrays[col][row_idx] = float(components[arr_idx, col_idx])

    out = data.select('datetime').hstack(pl.DataFrame(component_arrays))
    if target_col and target_col in data.columns:
        out = out.with_columns(data[target_col])
    return out


def _apply_feature_ablation(
        data: pl.DataFrame,
        manifest: 'MLManifest',
        round_params: dict[str, Any],
        columns_to_drop: list[str] | None,
        pre_transform_columns: frozenset[str],
) -> tuple[pl.DataFrame, list[str] | None]:

    '''
    Drop random feature columns from data for ablation.

    NOTE: Mutates round_params by adding '_dropped_features' key
    with the sorted list of dropped column names.
    '''

    config = manifest.ablation_config
    if config is None:
        raise ValueError('_apply_feature_ablation manifest has no ablation_config')

    raw_drop_count = round_params.get(config.drop_count_key)
    drop_count = 0 if raw_drop_count is None else raw_drop_count
    if not isinstance(drop_count, int) or isinstance(drop_count, bool) or drop_count < 0:
        raise ValueError(
            f"round_params['{config.drop_count_key}'] must be a non-negative int, got {raw_drop_count!r}"
        )

    raw_seed = round_params.get(config.seed_key)
    seed = 0 if raw_seed is None else raw_seed
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(
            f"round_params['{config.seed_key}'] must be an int, got {raw_seed!r}"
        )

    if drop_count == 0:
        round_params.pop('_dropped_features', None)
        return data, None

    if columns_to_drop is None:
        protected = pre_transform_columns
        if manifest.target_column:
            protected = protected | {manifest.target_column}
        eligible = sorted(
            c for c in data.columns if c not in protected
        )

        if drop_count > len(eligible):
            raise ValueError(
                f"{config.drop_count_key} ({drop_count}) exceeds eligible feature columns ({len(eligible)})"
            )

        rng = random.Random(seed)
        columns_to_drop = rng.sample(eligible, drop_count)
        round_params['_dropped_features'] = sorted(columns_to_drop)

    return data.drop(columns_to_drop), columns_to_drop


def _apply_feature_transforms(manifest: Manifest, lazy_data: pl.LazyFrame, round_params: dict[str, Any]) -> pl.LazyFrame:

    for entry in manifest.feature_transforms:
        if not _should_include_transform(entry, round_params):
            continue
        resolved = _resolve_params(entry.params, round_params)
        if _SOURCE_ROW not in lazy_data.collect_schema():
            lazy_data = lazy_data.pipe(entry.func, **resolved)
            continue
        source_data = lazy_data.collect()
        public_data = source_data.drop(_SOURCE_ROW) if _SOURCE_ROW in source_data.columns else source_data
        transformed = public_data.lazy().pipe(entry.func, **resolved).collect()
        lazy_data = _restore_source_rows(source_data, transformed).lazy()

    return lazy_data


def _apply_fitted_transforms(
        transform_entries: list[FittedTransformEntry],
        data: pl.DataFrame,
        round_params: dict[str, Any],
        all_fitted_params: dict[str, Any],
        is_training: bool
) -> tuple[pl.DataFrame, dict[str, Any]]:

    '''
    Compute fitted transforms on eager DataFrame.

    Args:
        transform_entries (List[FittedTransformEntry]): List of fitted transform configurations
        data (pl.DataFrame): DataFrame to apply transforms to
        round_params (Dict[str, Any]): Parameter values for current round
        all_fitted_params (Dict[str, Any]): Previously fitted parameters
        is_training (bool): Whether this is training data for fitting

    Returns:
        Tuple[pl.DataFrame, Dict[str, Any]]: Transformed data and updated fitted parameters
    '''

    for fitted_param_computations, func, base_params in transform_entries:
        # Fit parameters on training data only
        for param_name, compute_func, compute_base_params in fitted_param_computations:
            if param_name not in all_fitted_params and is_training:
                resolved = _resolve_params(compute_base_params, round_params)
                value = compute_func(data, **resolved)
                all_fitted_params[param_name] = value

        # Apply transform using fitted parameters
        combined_round_params = {**round_params, **all_fitted_params}
        resolved = _resolve_params(base_params, combined_round_params)
        data = func(data, **resolved)

    return data, all_fitted_params


def _apply_class_based_target(
        manifest: Manifest,
        data: pl.DataFrame,
        round_params: dict[str, Any],
        all_fitted_params: dict[str, Any],
        is_training: bool, *, trade_context: object = None, labels: list[OutcomeLabels] | None = None
) -> tuple[pl.DataFrame, dict[str, Any]]:


    source_data = data
    if _SOURCE_ROW in data.columns:
        data = data.drop(_SOURCE_ROW)
    config = manifest.target_class_config
    if config is None:
        raise ValueError('_apply_class_based_target manifest has no target_class_config')
    target_name = manifest.target_column
    instance_key = f'_target_cls_{target_name}'
    needs_context = bool(getattr(config.target_class, 'requires_trade_context', False))
    if needs_context and (trade_context is None or 'trade_context' in config.fit_params or 'trade_context' in config.transform_params):
        raise ValueError('Trade outcome context is reserved and must be supplied by the framework')

    if is_training:
        resolved_fit = _resolve_params(config.fit_params, round_params)
        if needs_context:
            resolved_fit['trade_context'] = trade_context
        instance = config.target_class(
            train_data=data,
            target_name=target_name,
            **resolved_fit
        )
        all_fitted_params[instance_key] = instance
    else:
        if instance_key not in all_fitted_params:
            raise RuntimeError(
                f"Target instance '{instance_key}' not found — training split must run before validation/test."
            )
        instance = all_fitted_params[instance_key]

    resolved_transform = _resolve_params(config.transform_params, round_params)
    if needs_context:
        resolved_transform['trade_context'] = trade_context
    data = instance.transform(data, **resolved_transform)
    if needs_context:
        outcomes = instance.outcomes
        if not isinstance(outcomes, OutcomeLabels) or labels is None:
            raise ValueError('Context target must expose private OutcomeLabels')
        labels.append(outcomes)

    return _restore_source_rows(source_data, data), all_fitted_params


def _apply_scaler(
        manifest: 'MLManifest',
        data: pl.DataFrame,
        round_params: dict[str, Any],
        all_fitted_params: dict[str, Any],
        is_training: bool
) -> tuple[pl.DataFrame, dict[str, Any]]:

    source_data = data
    if _SOURCE_ROW in data.columns:
        data = data.drop(_SOURCE_ROW)
    if manifest.scaler:
        target_col = manifest.target_column
        target_data = None
        if target_col and target_col in data.columns:
            target_data = data[target_col]
            data = data.drop(target_col)

        data, all_fitted_params = _apply_fitted_transforms(
            [manifest.scaler], data, round_params,
            all_fitted_params, is_training
        )

        if target_data is not None:
            data = data.with_columns(target_data)

    return _restore_source_rows(source_data, data), all_fitted_params


class _PCATransformer(Protocol):

    '''Typed facade over the fitted sklearn PCA surface used for compression.'''

    def fit_transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.floating[Any]]: ...

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.floating[Any]]: ...


def _pca_transformer(n_components: int) -> _PCATransformer:

    '''Create a full-SVD sklearn PCA instance behind the typed facade.'''

    return PCA(n_components=n_components, svd_solver='full', whiten=False)


def _apply_pca_compression(
        manifest: 'MLManifest',
        split_data: list[pl.DataFrame],
        round_params: dict[str, Any],
        all_fitted_params: dict[str, Any],
) -> tuple[list[pl.DataFrame], dict[str, Any]]:

    config = manifest.pca_compression_config
    if config is None:
        return split_data, all_fitted_params

    enabled = round_params.get(config.enabled_param, False)
    if not isinstance(enabled, bool):
        raise TypeError(
            f"round_params['{config.enabled_param}'] must be a bool, got {enabled!r}"
        )
    if not enabled:
        return split_data, all_fitted_params

    if config.n_components_param not in round_params:
        raise ValueError(
            f"round_params['{config.n_components_param}'] is required when round_params['{config.enabled_param}'] is True"
        )

    scaler = all_fitted_params.get(config.scaler_param_name)
    if scaler is None:
        raise ValueError(
            f"PCA compression could not find a fitted scaler at all_fitted_params['{config.scaler_param_name}']. Configure .set_scaler(RobustScaler), or pass a matching scaler_param_name to .set_pca_compression()."
        )
    if not isinstance(scaler, RobustScaler):
        raise ValueError(
            f"PCA compression requires a fitted RobustScaler at all_fitted_params['{config.scaler_param_name}'], got {type(scaler).__name__}."
        )

    target_col = manifest.target_column
    excluded_cols = {'datetime', target_col}
    feature_cols = [col for col in split_data[0].columns if col not in excluded_cols]
    _validate_pca_feature_columns(split_data, feature_cols)

    k = round_params[config.n_components_param]
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError(
            f"round_params['{config.n_components_param}'] must be an int, got {k!r}"
        )
    if k < 1 or k > len(feature_cols):
        raise ValueError(
            f"round_params['{config.n_components_param}'] must be between 1 and the feature count ({len(feature_cols)}), got {k}"
        )
    if k > split_data[0].height:
        raise ValueError(
            f"round_params['{config.n_components_param}'] must be no larger than the train row count ({split_data[0].height}), got {k}"
        )

    pca = _pca_transformer(k)
    component_cols = [f'{config.component_prefix}{i}' for i in range(k)]
    train_components = pca.fit_transform(split_data[0].select(feature_cols).to_numpy())

    transformed_splits = [
        _build_pca_split(
            split_data[0], train_components, component_cols, target_col
        )
    ]
    for split in split_data[1:]:
        if split.height == 0:
            components = np.empty((0, k))
        else:
            components = pca.transform(split.select(feature_cols).to_numpy())
        transformed_splits.append(
            _build_pca_split(split, components, component_cols, target_col)
        )

    all_fitted_params['_pca'] = pca
    all_fitted_params['_pca_input_feature_names'] = feature_cols
    all_fitted_params['_pca_feature_names'] = component_cols
    all_fitted_params['_pca_n_components'] = k

    return transformed_splits, all_fitted_params


def _validate_pca_feature_columns(
        split_data: list[pl.DataFrame],
        feature_cols: list[str],
) -> None:

    if not feature_cols:
        raise ValueError('PCA compression requires at least one feature column')

    for split_idx, split in enumerate(split_data):
        if split.height == 0:
            continue
        non_numeric = [
            col for col in feature_cols
            if not split[col].dtype.is_numeric()
        ]
        if non_numeric:
            raise ValueError(
                f"PCA compression requires numeric feature columns; split {split_idx} has non-numeric columns: {non_numeric}"
            )


def _build_pca_split(
        source: pl.DataFrame,
        components: Any,
        component_cols: list[str],
        target_col: str | None,
) -> pl.DataFrame:

    component_data = {
        col: components[:, idx]
        for idx, col in enumerate(component_cols)
    }
    out = source.select('datetime').hstack(pl.DataFrame(component_data))

    if target_col and target_col in source.columns:
        out = out.with_columns(source[target_col])

    return out


def _run_prepare_setup(
        manifest: Manifest,
        raw_data: pl.DataFrame,
        round_params: dict[str, Any],
) -> tuple[list[pl.DataFrame], list[datetime] | list[int], pl.DataFrame | None, list[pl.DataFrame] | None, PreparedTradeContext | None]:

    if manifest.pre_split_data_selector:
        func, base_params = manifest.pre_split_data_selector
        resolved = _resolve_params(base_params, round_params)
        raw_data = func(raw_data, **resolved)

    raw_splits = _resolve_split(manifest, raw_data, require_validation=_requires_fold_validation(manifest, round_params))
    split_data = raw_splits

    datetime_bar_pairs = [_process_bars(manifest, split, round_params) for split in split_data]
    all_datetimes = cast(list[datetime] | list[int], [dt for datetimes, _ in datetime_bar_pairs for dt in datetimes])
    split_data = [bar_data for _, bar_data in datetime_bar_pairs]

    policy = manifest.resolve_trade_policy(round_params)
    trade = None
    if policy is not None and manifest.backtest_config is not None:
        interval = _source_interval(manifest.data_source_config)
        trade = _prepare_trade_context(manifest.backtest_config, policy, raw_splits, split_data, round_params, interval_seconds=interval)
        split_data = [split.drop([name for name in ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns') if name in split.columns]) for split in split_data]
    prepared = _prepare_backtest_data(split_data, all_datetimes, configured=policy is None and _configured_barriers(manifest.backtest_config))
    return (*prepared, trade)


def _requires_fold_validation(manifest: Manifest, round_params: Mapping[str, object]) -> bool:
    if not isinstance(manifest, MLManifest):
        return False
    config = manifest.prediction_calibration_config
    return manifest.objective is not None or (config is not None and (
        (config.calibration_func is not None and bool(round_params.get('use_calibration', True)))
        or (config.threshold_func is not None and bool(round_params.get('use_threshold', True)))
    ))


def _resolve_split(
    manifest: Manifest, raw_data: pl.DataFrame, *, require_validation: bool = False,
) -> list[pl.DataFrame]:
    if manifest.split_walk_forward is not None:
        if manifest.split_dates is not None:
            raise ValueError('Manifest split_walk_forward conflicts with split_dates')
        return _walk_forward_split(raw_data, manifest.split_walk_forward, getattr(manifest, '_walk_forward_fold'),
                                   manifest.split_config, require_validation=require_validation)
    if manifest.split_dates is not None:
        return split_by_dates(raw_data, *manifest.split_dates)
    return split_sequential(raw_data, manifest.split_config)


def _align_split_columns(split_data: list[pl.DataFrame]) -> list[pl.DataFrame]:

    non_empty_splits = [s for s in split_data if s.height > 0]
    if not non_empty_splits:
        return split_data

    reference_cols = non_empty_splits[0].columns

    if len(non_empty_splits) > 1:
        common_cols = set(reference_cols)
        for split in non_empty_splits[1:]:
            common_cols &= set(split.columns)

        for i, split in enumerate(split_data):
            if split.height == 0:
                continue
            extra = set(split.columns) - common_cols
            if extra:
                logger.warning(
                    'Dropping columns %s from split %d — not present in all splits',
                    sorted(extra), i,
                )
                ordered_cols = [c for c in split.columns if c in common_cols]
                split_data[i] = split.select(ordered_cols)
        reference_cols = [c for c in reference_cols if c in common_cols]

    for i, split in enumerate(split_data):
        if split.height == 0:
            missing = set(reference_cols) - set(split.columns)
            if missing:
                split_data[i] = split.with_columns(
                    [pl.lit(None).alias(c) for c in reference_cols if c in missing]
                ).select(reference_cols)
            else:
                split_data[i] = split.select(reference_cols)

    return split_data


def _finalize_to_data_dict(
        manifest: 'MLManifest',
        split_data: list[pl.DataFrame],
        all_datetimes: list[datetime] | list[int],
        fitted_params: dict[str, Any],
        round_params: dict[str, Any],
        price_data_for_backtest: pl.DataFrame | None = None,
) -> dict[str, Any]:

    for i, split_df in enumerate(split_data):
        assert 'datetime' in split_df.columns, f"Split {i} missing 'datetime' column"

    if manifest.target_column:
        for i, split_df in enumerate(split_data):
            cols = list(split_df.columns)
            if manifest.target_column in cols:
                cols.remove(manifest.target_column)
                cols.append(manifest.target_column)
                split_data[i] = split_df.select(cols)
            else:
                raise ValueError(f"Split {i} missing target column '{manifest.target_column}'")

    cols = list(split_data[0].columns)

    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)

    for param_name, param_value in fitted_params.items():
        data_dict[param_name] = param_value

    data_dict['_feature_names'] = cols
    data_dict['_fitted_params'] = dict(fitted_params)

    if price_data_for_backtest is not None:
        data_dict['price_data_for_backtest'] = price_data_for_backtest

    if manifest.data_dict_extension:
        data_dict = manifest.data_dict_extension(
            data_dict=data_dict,
            split_data=split_data,
            round_params=round_params,
            fitted_params=fitted_params
        )

    return data_dict


def _finalize_rule_based_data(
        manifest: RuleBasedManifest,
        split_data: list[pl.DataFrame],
        all_datetimes: list[datetime] | list[int],
        round_params: dict[str, Any],
) -> dict[str, Any]:

    from limen.sfd.rule_based.predicates import build_predicate  # avoid circular import at module level

    data_dict = split_data_to_rule_based_prep_output(split_data, all_datetimes)

    config = manifest.strategy
    if config is None:
        raise ValueError('_finalize_rule_based_data manifest has no strategy configured')
    predicate_conditions = [c for c in config.conditions if 'type' in c]
    predicate_ids = [c['id'] for c in predicate_conditions]
    predicate_exprs = [
        build_predicate(condition, round_params).fill_null(False).alias(condition['id'])
        for condition in predicate_conditions
    ]
    if predicate_exprs:
        for split in ('train', 'val', 'test'):
            collisions = sorted(set(data_dict[split].columns) & set(predicate_ids))
            if collisions:
                raise ValueError(
                    f"Rule-based condition ids collide with existing columns in {split!r} split: {collisions}. Rename the affected conditions."
                )
            data_dict[split] = data_dict[split].with_columns(predicate_exprs)

    data_dict['strategy'] = {
        'conditions': config.conditions,
        'entry': config.entry,
    }

    return data_dict
