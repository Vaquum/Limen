'''LoopSFD: compile a Loop web UI payload into a Limen-compatible SFD object.

A LoopSFD instance satisfies UEL's duck-typed SFD contract — it exposes
params() and manifest() methods, plus a __name__ attribute used by
UniversalExperimentLoop._write_metadata. UEL accepts any object with this
shape (see limen/experiment/experiment_core.py:70-75).

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands. See README.md.
'''

import logging
from typing import Any

from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.sfd.loop.meta import SCALER_NAME_MAP, get_target_column
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


logger = logging.getLogger(__name__)


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

        NOTE: The payload's `transforms` array is intentionally ignored in
        this iteration. Transforms will be wired in a follow-up once the
        correct execution context (target context, post-label) is validated
        against representative payloads.

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

        ignored_transforms = self._payload.get('transforms') or []
        if ignored_transforms:
            logger.info(
                'Loop payload contains %d transform(s); ignoring in this '
                'iteration. Names: %s',
                len(ignored_transforms),
                [t.get('name') for t in ignored_transforms],
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
            5. Transforms from payload.transforms (added as feature transforms)
            6. Label as target transform (first label only)
            7. Scaler from payload.scaler.scalingMethod
            8. Model from payload.referenceArchitecture

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

        # Label becomes the target transform. The payload's `transforms`
        # array is intentionally ignored in this iteration — see __init__
        # docstring NOTE.
        labels = self._payload.get('labels', []) or []
        if labels:
            label = labels[0]
            label_func = FEATURE_REGISTRY[label['name']]
            target_col = get_target_column(label['name'])
            wired = self._wire_params('label', label['name'], label.get('params', {}))
            m = (
                m.with_target(target_col)
                .add_transform(label_func, **wired)
                .done()
            )

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
            - scaler_* (UI form metadata for the dropped selectedItems scaler)
            - transform_* (transforms are ignored in this iteration)

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
            component_type (str): One of 'indicator', 'feature', 'transform', 'label'
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

__all__ = ['LoopSFD']
