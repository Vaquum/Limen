'''LoopSFD: compile a Loop web UI payload into a Limen-compatible SFD object.

A LoopSFD instance satisfies UEL's duck-typed SFD contract — it exposes
params() and manifest() methods, plus a __name__ attribute used by
UniversalExperimentLoop._write_metadata. UEL accepts any object with this
shape (see limen/experiment/experiment_core.py:70-75).

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands.
'''

from typing import Any

from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.sfd.loop.meta import (
    FITTED_LABELS,
    SCALER_NAME_MAP,
    FittedLabelConfig,
    get_target_column,
)
from limen.sfd.loop.reference_defaults import get_reference_architecture_params
from limen.sfd.loop.registry import (
    FEATURE_REGISTRY,
    INDICATOR_REGISTRY,
    MODEL_REGISTRY,
    SCALER_REGISTRY,
)


_DEFAULT_DATA_SOURCE: dict[str, Any] = {
    'method': HistoricalData.get_spot_klines,
    'params': {'kline_size': 3600, 'start_date_limit': '2025-01-01'},
}


class LoopSFD:

    '''Compile a Loop web UI payload into a Limen SFD-shaped object.'''

    def __init__(self,
                 payload: dict[str, Any],
                 reference_architecture_params: dict[str, list] | None = None,
                 data_source_config: dict[str, Any] | None = None) -> None:

        '''
        Initialize the LoopSFD with a Loop payload and optional overrides.

        Args:
            payload (dict): Loop web UI experiment design payload
            reference_architecture_params (dict | None): Override for the model
                hyperparam search space. When None, derived from the matching
                foundational SFD's params() filtered by function signature
            data_source_config (dict | None): Override for the production data
                source. Must have 'method' and optional 'params' keys. Defaults
                to HistoricalData.get_spot_klines with kline_size=3600

        Raises:
            KeyError: If payload's referenceArchitecture is not in MODEL_REGISTRY

        '''

        self._payload = payload
        self._arch = payload['referenceArchitecture']
        self._arch_func = MODEL_REGISTRY[self._arch]

        if reference_architecture_params is not None:
            self._arch_params = dict(reference_architecture_params)
        else:
            self._arch_params = get_reference_architecture_params(
                self._arch, self._arch_func,
            )

        self._data_source_config = (
            dict(data_source_config) if data_source_config is not None
            else dict(_DEFAULT_DATA_SOURCE)
        )

        # Required by UniversalExperimentLoop at experiment_core.py:70 and by
        # UEL._write_metadata which raises if the SFD has no __name__
        self.__name__ = f"loop_{self._arch}"

    def params(self) -> dict[str, list]:

        '''
        Compute the merged parameter search space.

        Combines payload-supplied component params (with namespaced keys like
        'indicator_bbands_period') with SFD-supplied model hyperparams (with
        un-namespaced keys like 'C', 'solver' that resolve_model_kwargs matches
        against the model function signature).

        Returns:
            dict[str, list]: Mapping from param name to candidate values list

        '''

        component_params = self._filtered_parameter_space()
        return {**component_params, **self._arch_params}

    def manifest(self) -> Manifest:

        '''
        Compute a Manifest configured from the Loop payload.

        Build order:
            1. Data source (default or constructor override)
            2. Split config from payload.inputData.splitRatios
            3. Indicators from payload.indicators
            4. Features from payload.features
            5. Label as target transform (first label only). Labels
               registered in FITTED_LABELS are wired via add_fitted_transform
               so a training-fit param (e.g. a quantile cutoff) is computed
               before the transform runs; other labels use plain add_transform
            6. Scaler from payload.scaler.scalingMethod
            7. Model from payload.referenceArchitecture

        Returns:
            Manifest: Configured manifest ready for UEL execution

        '''

        m = Manifest()

        m.set_data_source(**self._data_source_config)
        m.set_test_data_source(method=HistoricalData._get_data_for_test)

        sr = self._payload['inputData']['splitRatios']
        m.set_split_config(sr['train'], sr['val'], sr['test'])

        for ind in self._payload.get('indicators', []) or []:
            func = INDICATOR_REGISTRY[ind['name']]
            wired = self._wire_params('indicator', ind['name'], ind.get('params', {}))
            m.add_indicator(func, **wired)

        for feat in self._payload.get('features', []) or []:
            func = FEATURE_REGISTRY[feat['name']]
            wired = self._wire_params('feature', feat['name'], feat.get('params', {}))
            m.add_feature(func, **wired)

        # Label becomes the target transform. Fitted labels (see
        # FITTED_LABELS in meta.py) are wired via add_fitted_transform
        # so compute_func runs on the training split before apply.
        labels = self._payload.get('labels', []) or []
        if labels:
            label = labels[0]
            label_name = label['name']
            label_func = FEATURE_REGISTRY[label_name]
            target_col = get_target_column(label_name)
            user_params = label.get('params', {}) or {}
            target_builder = m.with_target(target_col)

            fitted_config = FITTED_LABELS.get(label_name)
            if fitted_config is not None:
                target_builder = self._wire_fitted_label(
                    target_builder, label_func, label_name, user_params, fitted_config,
                )
            else:
                wired = self._wire_params('label', label_name, user_params)
                target_builder = target_builder.add_transform(label_func, **wired)

            m = target_builder.done()

        scaling_method = (self._payload.get('scaler') or {}).get('scalingMethod')
        if scaling_method:
            registry_key = SCALER_NAME_MAP.get(scaling_method)
            if registry_key is None:
                raise ValueError(
                    f"Unknown scaler '{scaling_method}'. "
                    f"Known: {sorted(SCALER_NAME_MAP)}"
                )
            m.set_scaler(SCALER_REGISTRY[registry_key])

        m.with_model(self._arch_func)

        return m

    def _filtered_parameter_space(self) -> dict[str, list]:

        '''
        Compute parameterSpace with UI-form metadata and reference arch params stripped.

        Removed keys:
            - *_selected_items (UI form selection metadata)
            - reference_architecture (top-level UI metadata; wired via payload['referenceArchitecture'])
            - scaling_method (superseded by scaler.scalingMethod)
            - input_split_* (split config handled via inputData.splitRatios)
            - input_<arch>_* (reference architecture params, sourced from SFD)
            - scaler_* (defensive: stripped for legacy payloads that carried
              selectedItems-shaped scaler metadata)
            - transform_* (defensive: stripped for legacy payloads that carried
              a top-level TRANSFORM section — no longer supported)

        Returns:
            dict[str, list]: Filtered component parameter search space

        '''

        arch_prefix = f"input_{self._arch}_"
        result: dict[str, list] = {}
        for key, value in (self._payload.get('parameterSpace') or {}).items():
            if key.endswith('_selected_items'):
                continue
            if key == 'scaling_method':
                continue
            if key == 'reference_architecture':
                continue
            if key.startswith('input_split_'):
                continue
            if key.startswith(arch_prefix):
                continue
            if key.startswith('scaler_'):
                continue
            if key.startswith('transform_'):
                continue
            result[key] = value
        return result

    def _wire_params(self,
                     component_type: str,
                     name: str,
                     params: dict[str, Any]) -> dict[str, Any]:

        '''
        Compute manifest-level params for a component, wiring search params via round_params.

        For each parameter:
            - If the namespaced key ({component_type}_{name}_{param_name}) exists
              in parameterSpace, the param is passed as a string reference so
              Manifest._resolve_params will substitute the round_params value
              at runtime
            - Otherwise the param is passed as a literal value

        Args:
            component_type (str): One of 'indicator', 'feature', 'label'
            name (str): Component short name (e.g. 'bbands')
            params (dict): Component params from the payload entry

        Returns:
            dict[str, Any]: Params dict ready to pass to manifest builder methods

        '''

        result: dict[str, Any] = {}
        param_space = self._payload.get('parameterSpace') or {}
        for pname, pvalue in params.items():
            namespaced = f"{component_type}_{name}_{pname}"
            result[pname] = namespaced if namespaced in param_space else pvalue
        return result

    def _wire_fitted_label(self,
                           target_builder: Any,
                           label_func: Any,
                           label_name: str,
                           user_params: dict[str, Any],
                           config: FittedLabelConfig) -> Any:

        '''
        Wire a fitted-transform label onto the target builder.

        The fitted label pattern computes a scalar from the training split
        (e.g. a quantile cutoff) before applying the label function. This
        method translates the FittedLabelConfig into
        `.add_fitted_transform().fit_param().with_params()` calls.

        Args:
            target_builder: Limen TargetBuilder returned from .with_target()
            label_func (Callable): The label function (from FEATURE_REGISTRY)
            label_name (str): Label name as used in the payload
            user_params (dict): Label params from the payload entry
            config (FittedLabelConfig): Wiring spec from FITTED_LABELS

        Returns:
            TargetBuilder ready for .done()

        Raises:
            KeyError: If a key listed in compute_param_keys is missing from
                user_params

        '''

        wired = self._wire_params('label', label_name, user_params)

        compute_params = {key: wired[key] for key in config.compute_param_keys}

        apply_params = {key: wired[key] for key in config.apply_param_keys}
        apply_params[config.apply_fitted_as] = config.fitted_param_name

        return (
            target_builder
            .add_fitted_transform(label_func)
            .fit_param(config.fitted_param_name, config.compute_func, **compute_params)
            .with_params(**apply_params)
        )


__all__ = ['LoopSFD']
