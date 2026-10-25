from __future__ import annotations

from collections.abc import Mapping
import inspect
from string import Formatter
from typing import cast

from limen.experiment._factorized_native import contract_for
from limen.experiment.manifest_core import MLManifest, Manifest
from limen.scalers import SCALER_REGISTRY


def parameter_references(value: object, domain: Mapping[str, list[object]]) -> set[str]:
    if isinstance(value, str):
        found: set[str] = {value} if value in domain else set()
        for _, field, spec, _ in Formatter().parse(value):
            if field is not None:
                name = field.split('.', 1)[0].split('[', 1)[0]
                if name in domain:
                    found.add(name)
            found.update(parameter_references(spec, domain))
        return found
    if isinstance(value, Mapping):
        return set[str]().union(*(parameter_references(item, domain) for item in cast(Mapping[object, object], value).values()))
    return set()


def projections(manifest: Manifest, domain: Mapping[str, list[object]]) -> list[frozenset[str]]:
    groups: list[frozenset[str]] = []
    selectors = {'feature_groups'} & set(domain)
    selectors.update(entry.include_if for entry in manifest.feature_transforms if entry.include_if is not None and entry.include_if in domain)

    def add(component: object, arguments: Mapping[str, object] | None, *, activation: set[str] | None = None) -> None:
        declaration = contract_for(component)
        keys = None if declaration is None else declaration.validation_keys
        names: set[str] = set(domain) if arguments is None and keys is None else set()
        if keys is None and arguments is not None:
            names.update(parameter_references(arguments, domain))
        elif keys is not None and declaration is not None:
            for name in (*keys, *declaration.parameter_keys):
                names.update(parameter_references(name if arguments is None else arguments.get(name, name), domain))
        groups.append(frozenset(names | selectors | (activation or set())))

    add(manifest.architecture_function, None)
    for entry in manifest.feature_transforms:
        add(entry.func, entry.params)
    for name in ('pre_split_data_selector', 'bar_formation'):
        step = getattr(manifest, name)
        if step is not None:
            add(step[0], step[1], activation={'bar_type'} & set(domain) if name == 'bar_formation' else set())
    target = manifest.target_class_config
    if target is not None:
        add(target.target_class, {**target.fit_params, **target.transform_params})
    if isinstance(manifest, MLManifest):
        if manifest.scaler is not None:
            _, factory, dynamic = manifest.scaler[0][0]
            parameters = inspect.signature(factory).parameters
            if '_cls' in parameters:
                add(parameters['_cls'].default, {**parameters['_p'].default, **dynamic})
            else:
                state = inspect.getclosurevars(factory).nonlocals
                selector = dynamic.get('scaler_type')
                active = parameter_references(selector, domain)
                for component in SCALER_REGISTRY.values():
                    add(component, {**state['_static'], **dynamic}, activation=active)
        config = manifest.prediction_calibration_config
        if config is None and manifest.architecture_function is not None:
            parameter = inspect.signature(manifest.architecture_function).parameters.get('prediction_calibration_config')
            if parameter is not None and parameter.default is not inspect.Parameter.empty:
                config = parameter.default
        if config is not None:
            add(config.calibration_func, config.calibration_params, activation={'use_calibration'} & set(domain))
            add(config.threshold_func, config.threshold_params, activation={'use_threshold'} & set(domain))
        if manifest.data_dict_extension is not None:
            add(manifest.data_dict_extension, None)
    return list(dict.fromkeys(groups))
