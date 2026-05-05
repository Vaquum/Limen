import copy
import inspect
import importlib
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from limen.calibration.pipeline import CalibratorProtocol
from limen.calibration.pipeline import ThresholdOptimizerProtocol

if TYPE_CHECKING:
    from limen.sfd.rule_based.config import RuleBasedConfig

import polars as pl

from limen.data.utils import split_data_to_prep_output
from limen.data.utils import split_data_to_rule_based_prep_output
from limen.data.utils import split_sequential
from limen.scalers.registry import SCALER_REGISTRY
logger = logging.getLogger(__name__)

ParamValue = Any | Callable[[dict[str, Any]], Any]
PipelineStep = tuple[Callable[..., pl.DataFrame], dict[str, ParamValue]]

FittedParamsComputationEntry = tuple[str, Callable[..., Any], dict[str, ParamValue]]


@dataclass
class TransformEntry:

    '''Feature or indicator transform with optional perturbation metadata.'''

    func: Callable
    params: dict[str, ParamValue] = field(default_factory=dict)
    group: str | None = None
    include_if: str | None = None


@dataclass
class AblationConfig:

    '''Configuration for random feature ablation (Drop-N).'''

    drop_count_key: str
    seed_key: str

FittedTransformEntry = tuple[
    list[FittedParamsComputationEntry],
    Callable[..., pl.LazyFrame],
    dict[str, ParamValue]
]


@dataclass
class TargetClassConfig:

    '''Configuration for a class-based target transform.'''

    target_class: type
    fit_params: dict[str, ParamValue] = field(default_factory=dict)
    transform_params: dict[str, ParamValue] = field(default_factory=dict)


@dataclass
class CalibrationConfig:

    '''Stores probability calibration and threshold function references with their params.'''

    calibration_func: CalibratorProtocol | None = None
    calibration_params: dict[str, Any] = field(default_factory=dict)
    threshold_func: ThresholdOptimizerProtocol | None = None
    threshold_params: dict[str, Any] = field(default_factory=dict)

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

    def __init__(self, manifest: 'Manifest') -> None:

        self._manifest = manifest
        self._calibration_func: CalibratorProtocol | None = None
        self._calibration_params: dict[str, Any] = {}
        self._threshold_func: ThresholdOptimizerProtocol | None = None
        self._threshold_params: dict[str, Any] = {}

    def probability_calibration(self, func: CalibratorProtocol, **params: Any) -> 'CalibrationBuilder':

        '''
        Configure the probability calibration function.

        Args:
            func (Callable): Calibration function with signature (clf, x_val, y_val, **params) -> fitted model
            **params: Extra keyword arguments forwarded to func; string values matching round_params keys are resolved at runtime

        Returns:
            CalibrationBuilder: Self for method chaining
        '''

        self._calibration_func = func
        self._calibration_params = params
        return self

    def threshold_function(self, func: ThresholdOptimizerProtocol, **params: Any) -> 'CalibrationBuilder':

        '''
        Configure the threshold optimisation function.

        Args:
            func (Callable): Threshold function with signature (y_val, val_proba, **params) -> tuple[float, float]
            **params: Extra keyword arguments forwarded to func; string values matching round_params keys are resolved at runtime

        Returns:
            CalibrationBuilder: Self for method chaining
        '''

        self._threshold_func = func
        self._threshold_params = params
        return self

    def done(self) -> 'Manifest':

        '''
        Finalise calibration configuration and return the manifest.

        Returns:
            Manifest: The parent manifest with prediction_calibration_config set

        Raises:
            ValueError: If neither probability_calibration() nor threshold_function() was called before done()
        '''

        if self._calibration_func is None and self._threshold_func is None:
            raise ValueError('at least one of probability_calibration() or threshold_function() must be called before done()')
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

    method: Callable
    params: dict[str, Any] = field(default_factory=dict)


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

        if inspect.ismethod(method) or (hasattr(method, '__self__') and method.__self__ is not None):
            result = method(**params)
            if hasattr(method.__self__, 'data'):
                return method.__self__.data
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
                    f"Method {method.__qualname__} executed successfully but "
                    f"instance does not have 'data' attribute. Expected data source "
                    f"methods to populate instance.data"
                )
            return method(**params)

        raise ValueError(f"Unsupported callable type: {type(method)}")


@dataclass
class Manifest:

    '''Defines manifest for Loop experiments.'''

    data_source_config: DataSourceConfig = None
    test_data_source_config: DataSourceConfig = None
    pre_split_data_selector: PipelineStep = None
    split_config: tuple[int, int, int] = (8, 1, 2)
    bar_formation: PipelineStep = None
    required_bar_columns: list[str] = field(default_factory=list)
    feature_transforms: list[TransformEntry] = field(default_factory=list)
    target_column: str | None = None
    target_class_config: TargetClassConfig | None = None
    scaler: FittedTransformEntry = None
    ablation_config: AblationConfig | None = None
    data_dict_extension: Callable = None

    architecture_function: Callable = None
    architecture_params: dict[str, ParamValue] = field(default_factory=dict)
    metrics_params: dict[str, ParamValue] = field(default_factory=dict)
    prediction_calibration_config: CalibrationConfig | None = None
    _rule_based: 'RuleBasedConfig | None' = field(default=None, init=False, repr=False)

    def _add_transform(self,
                       func: Callable,
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
                       method: Callable,
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
                            method: Callable,
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
            raise ValueError('No data source configured')

        return DataSourceResolver.resolve(self.data_source_config)

    def fetch_test_data(self) -> pl.DataFrame:

        '''Fetch data using configured test data source.'''

        if self.test_data_source_config is None:
            raise ValueError('No test data source configured')

        return DataSourceResolver.resolve(self.test_data_source_config)

    def add_feature(self,
                    func: Callable,
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
                      func: Callable,
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

    def set_pre_split_data_selector(self, func: Callable, **params: Any) -> 'Manifest':

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

    def set_bar_formation(self, func: Callable, **params: Any) -> 'Manifest':

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
            raise ValueError('train split ratio must be positive')
        if val < 0 or test < 0:
            raise ValueError('val and test split ratios must be non-negative')

        self.split_config = (train, val, test)

        return self

    def set_scaler(self, transform_class: Any, param_name: str = '_scaler') -> 'Manifest':

        '''
        Set scaler transformation using make_fitted_scaler.

        Args:
            transform_class: Transform class to use for scaling
            param_name (str): Parameter name for fitted scaler

        Returns:
            Manifest: Self for method chaining
        '''

        self.scaler = make_fitted_scaler(param_name, transform_class)

        return self


    def set_scaler_from_params(self,
                               param_name: str = 'scaler_type') -> 'Manifest':

        '''
        Configure scaler selection from round_params at runtime.

        The scaler class is resolved from the scaler registry using
        the value of round_params[param_name].

        Args:
            param_name (str): round_params key that holds the scaler type string

        Returns:
            Manifest: Self for method chaining
        '''

        def _scaler_factory(data: 'pl.DataFrame',
                            scaler_type: str = '') -> Any:

            if scaler_type not in SCALER_REGISTRY:
                if scaler_type == param_name:
                    raise ValueError(
                        f"round_params['{param_name}'] is required when using "
                        f"set_scaler_from_params(). "
                        f"Available types: {sorted(SCALER_REGISTRY)}"
                    )
                raise ValueError(
                    f"Unknown scaler type '{scaler_type}'. "
                    f"Available: {sorted(SCALER_REGISTRY)}"
                )
            return SCALER_REGISTRY[scaler_type](data)

        self.scaler = (
            [('_scaler', _scaler_factory, {'scaler_type': param_name})],
            _apply_fitted_transform,
            {'fitted_transform': '_scaler'},
        )

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

    def with_reference_architecture(self, architecture_function: Callable) -> 'Manifest':

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

    def with_calibration(self) -> 'CalibrationBuilder':

        '''
        Begin fluent calibration configuration.

        Returns:
            CalibrationBuilder: Builder for configuring probability calibration and threshold optimisation

        NOTE: Call .probability_calibration(), optionally .threshold_function(), then .done() to finalise.
        '''

        return CalibrationBuilder(self)

    def with_strategy(self, conditions: list[dict], entry: str) -> 'Manifest':

        '''
        Configure rule-based strategy conditions and entry signal.

        Args:
            conditions (list[dict]): List of predicate and compound operator condition configs
            entry (str): ID of the condition that produces the per-bar position signal

        Returns:
            Manifest: Self for method chaining
        '''

        from limen.sfd.rule_based.config import RuleBasedConfig  # local to avoid circular import
        self._rule_based = RuleBasedConfig(conditions=list(conditions), entry=entry)

        return self

    def set_feature_ablation(self,
                             drop_count_key: str = 'feature_drop_count',
                             seed_key: str = 'feature_drop_seed') -> 'Manifest':

        '''
        Configure random feature ablation (Drop-N).

        Randomly drops N feature columns per permutation using a
        deterministic seed from round_params. Runs after feature and
        target transforms in the prepare_data pipeline.

        Args:
            drop_count_key (str): round_params key for number of columns to drop
            seed_key (str): round_params key for random seed

        Returns:
            Manifest: Self for method chaining
        '''

        self.ablation_config = AblationConfig(
            drop_count_key=drop_count_key,
            seed_key=seed_key,
        )
        return self


    def add_to_data_dict(self, func: Callable) -> 'Manifest':

        '''
        Configure data_dict extension function to add custom entries after data preparation.

        Args:
            func (Callable): Extension function with signature (data_dict, split_data, round_params, fitted_params) -> dict

        Returns:
            Manifest: Self for method chaining

        NOTE: The extension function receives the base data_dict and full split DataFrames.
        It should modify and return the data_dict with any additional custom entries needed by the model.
        '''

        self.data_dict_extension = func
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
            ValueError: If a key is not 'split_config' and not accepted by the
                data source method
        '''

        new_manifest = copy.deepcopy(self)

        if 'split_config' in overrides:
            sc = overrides['split_config']
            _split_len = 3
            if not (isinstance(sc, tuple) and len(sc) == _split_len
                    and all(isinstance(v, int) and not isinstance(v, bool) for v in sc)):
                raise ValueError(f"split_config must be a 3-tuple of ints, got {sc!r}")
            if any(v < 0 for v in sc):
                raise ValueError(f"split_config ratios must be non-negative, got {sc!r}")
            if sum(sc) == 0:
                raise ValueError('split_config ratios must not all be zero')
            new_manifest.split_config = sc

        ds_overrides = {k: v for k, v in overrides.items() if k != 'split_config'}
        if ds_overrides:
            if new_manifest.data_source_config is None:
                raise ValueError('Cannot override data source params: no data source configured')
            method_params = set(inspect.signature(
                new_manifest.data_source_config.method
            ).parameters.keys()) - {'self', 'cls'}
            unknown = set(ds_overrides) - method_params
            if unknown:
                raise ValueError(
                    f"Unknown data source params: {sorted(unknown)}. "
                    f"Accepted by {new_manifest.data_source_config.method.__name__}: "
                    f"{sorted(method_params)}"
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

        split_data = split_sequential(raw_data, self.split_config)
        test_split = split_data[2]
        _, test_bar_data = _process_bars(self, test_split, round_params)

        return test_bar_data

    def prepare_data(
        self,
        raw_data: pl.DataFrame,
        round_params: dict[str, Any]
    ) -> dict:

        '''
        Compute final data dictionary from raw data using manifest configuration.

        Args:
            raw_data (pl.DataFrame): Raw input dataset
            round_params (Dict[str, Any]): Parameter values for current round

        Returns:
            dict: Final data dictionary ready for model training
        '''

        if self._rule_based is not None:
            if self.scaler is not None:
                raise ValueError(
                    'Scalers cannot be used with rule-based SFDs — predicates depend on '
                    'original indicator scales and produce incorrect signals on scaled values.'
                )
            if self.ablation_config is not None:
                raise ValueError(
                    'Feature ablation cannot be used with rule-based SFDs — predicate '
                    'columns are derived from specific indicator columns.'
                )

        if self.pre_split_data_selector:
            func, base_params = self.pre_split_data_selector
            resolved = _resolve_params(base_params, round_params)
            raw_data = func(raw_data, **resolved)

        split_data = split_sequential(raw_data, self.split_config)

        datetime_bar_pairs = [_process_bars(self, split, round_params) for split in split_data]
        all_datetimes = [dt for datetimes, _ in datetime_bar_pairs for dt in datetimes]
        split_data = [bar_data for _, bar_data in datetime_bar_pairs]

        price_cols = ['datetime', 'open', 'high', 'low', 'close']
        test_split = split_data[2]
        available = [c for c in price_cols if c in test_split.columns]
        price_data_for_backtest = test_split.select(available) if len(available) == len(price_cols) else None

        all_fitted_params = {}
        columns_to_drop: list[str] | None = None
        pre_transform_columns = frozenset(split_data[0].columns)

        for i in range(len(split_data)):
            lazy_data = split_data[i].lazy()

            lazy_data = _apply_feature_transforms(self, lazy_data, round_params)

            data = lazy_data.collect()

            if self.target_class_config is not None:
                data, all_fitted_params = _apply_class_based_target(
                    self, data, round_params, all_fitted_params, is_training=(i == 0)
                )

            if self.ablation_config is not None:
                data, columns_to_drop = _apply_feature_ablation(
                    data, self, round_params, columns_to_drop,
                    pre_transform_columns,
                )

            data = data.fill_nan(None).drop_nulls()

            data, all_fitted_params = _apply_scaler(
                self, data, round_params, all_fitted_params, is_training=(i == 0)
            )

            split_data[i] = data.fill_nan(None).drop_nulls()

        non_empty_splits = [s for s in split_data if s.height > 0]
        if non_empty_splits:
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
                            'Dropping columns %s from split %d — '
                            'not present in all splits',
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

        if price_data_for_backtest is not None:
            final_datetimes = split_data[2].select('datetime')
            price_data_for_backtest = final_datetimes.join(
                price_data_for_backtest, on='datetime', how='left',
                maintain_order='left'
            )

        if self._rule_based is not None:
            return _finalize_rule_based_data(self, split_data, all_datetimes, round_params)
        return _finalize_to_data_dict(self, split_data, all_datetimes, all_fitted_params, round_params, price_data_for_backtest)

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
            raise ValueError('Architecture function not configured. Use .with_reference_architecture(func) before run_model() or resolve_model_kwargs().')

        sig = inspect.signature(self.architecture_function)
        model_kwargs: dict[str, Any] = {}

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
                    f"Missing required parameter '{param_name}' for model function. "
                    'It must be provided in round_params.'
                )

        return model_kwargs


    def run_model(self, data: dict, round_params: dict[str, Any]) -> dict:

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
        if self.prediction_calibration_config is not None:
            use_calibration = round_params.get('use_calibration', True)
            use_threshold = round_params.get('use_threshold', True)
            if use_calibration or use_threshold:
                resolved = self.prediction_calibration_config.resolve(round_params)
                config = CalibrationConfig(
                    calibration_func=resolved.calibration_func if use_calibration else None,
                    calibration_params=resolved.calibration_params,
                    threshold_func=resolved.threshold_func if use_threshold else None,
                    threshold_params=resolved.threshold_params,
                )
                model_kwargs['prediction_calibration_config'] = config
        round_results = self.architecture_function(data, **model_kwargs)

        return round_results



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


def make_fitted_scaler(param_name: str, transform_class: Any) -> FittedTransformEntry:

    '''
    Create fitted transform entry for scaling.

    Args:
        param_name (str): Name for the fitted parameter
        transform_class: Transform class to instantiate

    Returns:
        FittedTransformEntry: Complete fitted transform configuration
    '''

    return ([
        (param_name, lambda data: transform_class(data), {})
    ],
    _apply_fitted_transform, {
        'fitted_transform': param_name
    })


def _resolve_params(params: dict[str, Any], round_params: dict[str, Any]) -> dict[str, Any]:

    '''
    Resolve parameters using just-in-time detection with actual round_params.

    Args:
        params (Dict[str, Any]): Parameter specification dictionary
        round_params (Dict[str, Any]): Round-specific parameter values

    Returns:
        Dict[str, Any]: Resolved parameter dictionary
    '''

    resolved = {}
    for key, value in params.items():
        if isinstance(value, str):
            if value.startswith('_') or value in round_params:
                resolved[key] = round_params[value]
            elif '{' in value and '}' in value:
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
) -> tuple[list, pl.DataFrame]:

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
        assert required_col in available_cols, (
            f"Required bar column '{required_col}' not found after bar formation"
        )

    return all_datetimes, bar_data


def _should_include_transform(entry: TransformEntry, round_params: dict[str, Any]) -> bool:

    if entry.include_if is not None and entry.include_if in round_params:
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
            f"round_params['feature_groups'] must be a string, "
            f"got {type(feature_groups).__name__}"
        )

    return group in feature_groups.split('|')


def _apply_feature_ablation(
        data: pl.DataFrame,
        manifest: Manifest,
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

    raw_drop_count = round_params.get(config.drop_count_key)
    drop_count = 0 if raw_drop_count is None else raw_drop_count
    if not isinstance(drop_count, int) or isinstance(drop_count, bool) or drop_count < 0:
        raise ValueError(
            f"round_params['{config.drop_count_key}'] must be a non-negative int, "
            f"got {raw_drop_count!r}"
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
                f"{config.drop_count_key} ({drop_count}) exceeds "
                f"eligible feature columns ({len(eligible)})"
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
        lazy_data = lazy_data.pipe(entry.func, **resolved)

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
        is_training: bool
) -> tuple[pl.DataFrame, dict[str, Any]]:

    '''
    Fit or reuse the configured target class and apply it to the split.

    Args:
        manifest (Manifest): Manifest holding the target class config
        data (pl.DataFrame): Split DataFrame to transform
        round_params (dict[str, Any]): Current round parameters for template resolution
        all_fitted_params (dict[str, Any]): Shared store for fitted instances across splits
        is_training (bool): Whether this is the training split; fits the instance if True

    Returns:
        tuple[pl.DataFrame, dict[str, Any]]: Transformed data and updated fitted params
    '''

    config = manifest.target_class_config
    target_name = manifest.target_column
    instance_key = f'_target_cls_{target_name}'

    if is_training:
        resolved_fit = _resolve_params(config.fit_params, round_params)
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
    data = instance.transform(data, **resolved_transform)

    return data, all_fitted_params


def _apply_scaler(
        manifest: Manifest,
        data: pl.DataFrame,
        round_params: dict[str, Any],
        all_fitted_params: dict[str, Any],
        is_training: bool
) -> tuple[pl.DataFrame, dict[str, Any]]:

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

    return data, all_fitted_params


def _finalize_to_data_dict(
        manifest: Manifest,
        split_data: list[pl.DataFrame],
        all_datetimes: list,
        fitted_params: dict[str, Any],
        round_params: dict[str, Any],
        price_data_for_backtest: pl.DataFrame | None = None
) -> dict:

    # Validate all splits have datetime column
    for i, split_df in enumerate(split_data):
        assert 'datetime' in split_df.columns, f"Split {i} missing 'datetime' column"

    # Ensure target_column is last column in all splits
    if manifest.target_column:
        for i, split_df in enumerate(split_data):
            cols = list(split_df.columns)
            if manifest.target_column in cols:
                # Move target_column to end
                cols.remove(manifest.target_column)
                cols.append(manifest.target_column)
                split_data[i] = split_df.select(cols)
            else:
                raise ValueError(f"Split {i} missing target column '{manifest.target_column}'")

    cols = list(split_data[0].columns)

    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)

    # Add fitted parameters to data_dict
    for param_name, param_value in fitted_params.items():
        data_dict[param_name] = param_value

    data_dict['_feature_names'] = cols

    if price_data_for_backtest is not None:
        data_dict['price_data_for_backtest'] = price_data_for_backtest

    # Apply data_dict extension if configured
    if manifest.data_dict_extension:
        data_dict = manifest.data_dict_extension(
            data_dict=data_dict,
            split_data=split_data,
            round_params=round_params,
            fitted_params=fitted_params
        )

    return data_dict


def _finalize_rule_based_data(
        manifest: Manifest,
        split_data: list[pl.DataFrame],
        all_datetimes: list,
        round_params: dict[str, Any],
) -> dict:

    from limen.sfd.rule_based.predicates import build_predicate  # avoid circular import at module level

    data_dict = split_data_to_rule_based_prep_output(split_data, all_datetimes)

    config = manifest._rule_based
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
                    f'Rule-based condition ids collide with existing columns in {split!r} split: '
                    f'{collisions}. Rename the affected conditions.'
                )
            data_dict[split] = data_dict[split].with_columns(predicate_exprs)

    data_dict['strategy'] = {
        'conditions': config.conditions,
        'entry': config.entry,
    }

    return data_dict
