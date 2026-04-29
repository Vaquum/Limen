'''LoopSFD: compile a Loop web UI payload into a Limen-compatible SFD object.

A LoopSFD instance satisfies UEL's duck-typed SFD contract — it exposes
params() and manifest() methods, plus a __name__ attribute used by
UniversalExperimentLoop._write_metadata. UEL accepts any object with this
shape (see limen/experiment/experiment_core.py:70-75).

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands.
'''

import inspect
import re
from typing import Any

from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.sfd.loop.meta import (
    DATA_SOURCE_REGISTRY,
    TARGET_LABEL_REGISTRY,
    get_target_column,
)
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


def _coerce_data_source_params(method: Any, params: dict[str, Any]) -> dict[str, Any]:

    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return dict(params)

    coerced: dict[str, Any] = {}
    for pname, pvalue in params.items():
        if not isinstance(pvalue, str):
            coerced[pname] = pvalue
            continue
        param_sig = sig.parameters.get(pname)
        ann = param_sig.annotation if param_sig else inspect.Parameter.empty
        if ann is int or (
            hasattr(ann, '__args__') and int in ann.__args__
        ):
            try:
                coerced[pname] = int(pvalue)
                continue
            except (ValueError, TypeError):
                raise ValueError(
                    f"Cannot coerce param '{pname}'='{pvalue}' to int "
                    f"as required by {method.__name__} signature"
                ) from None
        coerced[pname] = pvalue
    return coerced


def _resolve_data_source_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:

    '''
    Extract a Limen set_data_source(**config) dict from the payload's
    inputData.selectedItems entry if present.

    Returns None when the payload does not carry a data_source selectedItem,
    in which case the caller falls back to _DEFAULT_DATA_SOURCE or a
    constructor override.

    Args:
        payload (dict): Loop web UI experiment design payload

    Returns:
        dict | None: {'method': callable, 'params': {...}} or None
    '''

    input_data = payload.get('inputData') or {}
    selected_items = input_data.get('selectedItems') or []
    for item in selected_items:
        if (item or {}).get('name') != 'data_source':
            continue
        raw_params = dict(item.get('params') or {})
        method_name = raw_params.pop('method', None)
        if not method_name:
            raise ValueError(
                "data_source item is missing required 'method' key"
            )
        method = DATA_SOURCE_REGISTRY.get(method_name)
        if method is None:
            raise ValueError(
                f"Unknown data source method '{method_name}'. "
                f"Known: {sorted(DATA_SOURCE_REGISTRY)}"
            )
        resolved_params = {k: v for k, v in raw_params.items() if v is not None}
        resolved_params = _coerce_data_source_params(method, resolved_params)
        return {'method': method, 'params': resolved_params}
    return None


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
                hyperparam search space. When None, derived from the payload's
                parameterSpace via _build_arch_params(), which extracts keys
                matching the reference architecture function signature
            data_source_config (dict | None): Override for the production data
                source. Must have 'method' and optional 'params' keys. When
                None, the data source is extracted from the payload's
                inputData.selectedItems[0] entry named 'data_source'; if that
                is also absent, falls back to HistoricalData.get_spot_klines
                with kline_size=3600

        Raises:
            KeyError: If payload's referenceArchitecture is not in MODEL_REGISTRY

        '''

        self._payload = payload
        self._arch = payload['referenceArchitecture']
        self._arch_func = MODEL_REGISTRY[self._arch]
        self._sig_param_names = {
            p.name
            for p in inspect.signature(self._arch_func).parameters.values()
            if p.name != 'data'
        }

        if reference_architecture_params is not None:
            self._arch_params = dict(reference_architecture_params)
        else:
            self._arch_params = self._build_arch_params()

        if data_source_config is not None:
            self._data_source_config = dict(data_source_config)
        else:
            payload_config = _resolve_data_source_from_payload(payload)
            self._data_source_config = (
                payload_config if payload_config is not None
                else dict(_DEFAULT_DATA_SOURCE)
            )

        # Map from foundational-SFD style placeholder names (e.g. 'roc_period')
        # to the namespaced round_params keys LoopSFD actually produces
        # (e.g. 'indicator_roc_period'). Used by _rewrite_format_string to
        # translate user-supplied format strings at compile time so that
        # Manifest._resolve_params can interpolate them against round_params
        # at runtime.
        self._component_alias_map = self._build_component_alias_map()

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
            1. Data source (from payload.inputData.selectedItems[0] when present,
               otherwise constructor override or default)
            2. Split config from payload.inputData.splitRatios
            3. Indicators from payload.indicators
            4. Features from payload.features
            5. Label as target transform (first label only) via with_target_label()
            6. Scaler from payload.scaler.selectedItems[0].name
            7. Model from payload.referenceArchitecture

        Returns:
            Manifest: Configured manifest ready for UEL execution

        '''

        m = Manifest()

        m.set_data_source(**self._data_source_config)

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

        labels = self._payload.get('labels', []) or []
        if labels:
            label = labels[0]
            label_name = label['name']
            target_col = get_target_column(label_name)
            user_params = label.get('params', {}) or {}

            entry = TARGET_LABEL_REGISTRY[label_name]
            wired = self._wire_params('label', label_name, user_params)

            fit_params = {
                class_key: wired[payload_key]
                for payload_key, class_key in entry.fit_param_aliases.items()
                if payload_key in wired
            }
            transform_params = {
                class_key: wired[payload_key]
                for payload_key, class_key in entry.transform_param_aliases.items()
                if payload_key in wired
            }

            m.with_target_label(
                target_col, entry.target_class,
                fit_params=fit_params,
                transform_params=transform_params,
            )

        scaler_name = self._resolve_scaler_name()
        if scaler_name is not None:
            scaler_class = SCALER_REGISTRY.get(scaler_name)
            if scaler_class is None:
                raise ValueError(
                    f"Unknown scaler '{scaler_name}'. "
                    f"Known: {sorted(SCALER_REGISTRY)}"
                )
            m.set_scaler(scaler_class)

        m.with_reference_architecture(self._arch_func)

        return m

    def _resolve_scaler_name(self) -> str | None:

        '''
        Read the scaler name (lowercase registry key) from the payload.

        Reads from payload.scaler.selectedItems[0].name. Returns None when
        no scaler is selected.

        Returns:
            str | None: Registry key for SCALER_REGISTRY (e.g. 'logreg')

        '''

        scaler = self._payload.get('scaler') or {}
        selected_items = scaler.get('selectedItems') or []
        if not selected_items:
            return None
        first = selected_items[0] or {}
        name = first.get('name')
        if not isinstance(name, str) or not name.strip():
            return None
        return name.strip()

    def _build_arch_params(self) -> dict[str, list]:

        '''
        Compute the model hyperparam search space from the payload.

        All model hyperparameters come from the payload's parameterSpace.
        Each key is matched against the reference architecture function
        signature — keys that do not match any signature parameter are silently
        skipped.

        Raises:
            ValueError: If a payload-provided arch param value is not a list

        Returns:
            dict[str, list]: Arch param name → candidate values

        '''

        # Build a case-insensitive lookup: lowercase param name → actual name
        # e.g. 'c' → 'C', 'solver' → 'solver'
        sig_lower = {p.lower(): p for p in self._sig_param_names}
        arch_prefix = f'reference_architecture_{self._arch}_'

        ps = self._payload.get('parameterSpace') or {}
        arch_params: dict[str, list] = {}

        for key, value in ps.items():
            if key not in self._sig_param_names:
                continue
            canonical = sig_lower.get(key.lower())
            if canonical is None:
                continue
            if not isinstance(value, list):
                raise ValueError(
                    f"parameterSpace['{key}'] must be a list of candidate values, "
                    f"got {type(value).__name__}"
                )
            arch_params[canonical] = value

        for key, value in ps.items():
            if not key.startswith(arch_prefix):
                continue
            bare = key[len(arch_prefix):]
            canonical = sig_lower.get(bare.lower())
            if canonical is None:
                continue
            if not isinstance(value, list):
                raise ValueError(
                    f"parameterSpace['{key}'] must be a list of candidate values, "
                    f"got {type(value).__name__}"
                )
            arch_params[canonical] = value

        return arch_params


    def _filtered_parameter_space(self) -> dict[str, list]:

        '''
        Compute parameterSpace with UI-form metadata and out-of-band keys stripped.

        Removed keys:
            - *_selected_items (UI form selection metadata)
            - reference_architecture (top-level UI metadata; wired via payload['referenceArchitecture'])
            - scaling_method (legacy UI metadata)
            - input_split_* (split config handled via inputData.splitRatios)
            - input_data_source_* (data source config handled via inputData.selectedItems)
            - input_<arch>_* (reference architecture params, sourced from SFD)
            - bare arch signature keys (model hyperparams — handled via _build_arch_params)
            - scaler_* (defensive: stripped for legacy payloads that carried
              selectedItems-shaped scaler metadata)
            - transform_* (defensive: stripped for legacy payloads that carried
              a top-level TRANSFORM section — no longer supported)

        Returns:
            dict[str, list]: Filtered component parameter search space

        '''

        old_arch_prefix = f"input_{self._arch}_"
        ra_prefix = f"reference_architecture_{self._arch}_"
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
            if key.startswith('input_data_source_'):
                continue
            if key.startswith(old_arch_prefix):
                continue
            if key.startswith(ra_prefix):
                continue
            if key in self._sig_param_names:
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
            - If the value is a format-string template (contains '{...}'),
              rewrite any placeholders that match the un-namespaced alias of a
              known component param (e.g. 'roc_period' -> 'indicator_roc_period')
              and keep it as a literal. Manifest._resolve_params will then
              interpolate it against round_params at runtime via str.format
            - Otherwise if the namespaced key ({component_type}_{name}_{param_name})
              exists in parameterSpace, the param is passed as a string reference
              so Manifest._resolve_params will substitute the round_params value
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
            if isinstance(pvalue, str) and '{' in pvalue and '}' in pvalue:
                # Format strings bypass the parameterSpace-reference path.
                # Rewrite placeholders to namespaced round_params keys so the
                # runtime str.format resolves them correctly, then pass as a
                # literal. If we routed through parameterSpace instead,
                # _resolve_params would return the value from round_params
                # terminally (no .format call) and the curly braces would end
                # up as literal characters in the eventual column name.
                result[pname] = self._rewrite_format_string(pvalue)
                continue
            namespaced = f"{component_type}_{name}_{pname}"
            result[pname] = namespaced if namespaced in param_space else pvalue
        return result

    def _build_component_alias_map(self) -> dict[str, str]:

        '''
        Compute un-namespaced → namespaced alias map from payload components.

        For each indicator/feature/label in the payload, emit an alias that
        mirrors the foundational-SFD convention ('{component_name}_{param_name}'
        → 'indicator_{component_name}_{param_name}', etc). This lets users
        write format strings like 'roc_{roc_period}' that look identical to
        the foundational SFD style while LoopSFD silently translates them
        into the round_params namespace.

        Returns:
            dict[str, str]: Alias map keyed on the un-namespaced placeholder
                name (e.g. 'roc_period'), valued with the namespaced
                round_params key (e.g. 'indicator_roc_period')

        '''

        alias_map: dict[str, str] = {}
        component_groups = (
            ('indicator', self._payload.get('indicators') or []),
            ('feature', self._payload.get('features') or []),
            ('label', self._payload.get('labels') or []),
        )
        for component_type, components in component_groups:
            for component in components:
                comp_name = (component or {}).get('name')
                if not comp_name:
                    continue
                comp_params = (component or {}).get('params') or {}
                for param_name in comp_params:
                    alias = f"{comp_name}_{param_name}"
                    namespaced = f"{component_type}_{comp_name}_{param_name}"
                    # First write wins so later components cannot clobber
                    # earlier aliases if two components happen to share a
                    # (name, param) pair — unlikely but defensive
                    alias_map.setdefault(alias, namespaced)
        return alias_map

    def _rewrite_format_string(self, value: str) -> str:

        '''
        Compute a format string with un-namespaced placeholders rewritten to namespaced keys.

        For each '{placeholder}' in the input, if 'placeholder' is a key in
        the component alias map, replace it with '{alias_map[placeholder]}'.
        Placeholders that are not in the alias map are left unchanged —
        those may already be namespaced keys (e.g. 'indicator_roc_period')
        that resolve directly at runtime, or they may be unresolvable and
        raise KeyError at str.format time (same failure mode as before).

        Args:
            value (str): A format-string template (may contain 0..N
                '{name}' placeholders)

        Returns:
            str: The input with matching placeholders rewritten

        '''

        if not self._component_alias_map:
            return value

        def _replace(match: re.Match) -> str:
            placeholder = match.group(1)
            namespaced = self._component_alias_map.get(placeholder)
            if namespaced is None:
                return match.group(0)
            return '{' + namespaced + '}'

        return re.sub(r'\{([^{}]+)\}', _replace, value)

__all__ = ['LoopSFD']
