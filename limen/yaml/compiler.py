from datetime import date
from typing import Any

from limen.experiment.manifest_core import MLManifest
from limen.experiment.manifest_core import Manifest
from limen.experiment.manifest_core import RuleBasedManifest
from limen.yaml.errors import ResolutionError
from limen.yaml.resolver import resolve


def _resolve_func_params(params: dict[str, Any]) -> dict[str, Any]:

    '''
    Resolve any string values that are valid limen.* paths to their Python objects.

    Leaves non-resolvable strings (e.g. round_params keys like 'threshold_min') unchanged.

    Args:
        params (dict): Raw params dict from YAML

    Returns:
        dict: Params with callable paths resolved to Python objects

    '''

    result = {}
    for k, v in params.items():
        if isinstance(v, str):
            try:
                result[k] = resolve(v)
            except ResolutionError:
                result[k] = v
        else:
            result[k] = v
    return result


def build_manifest(yaml_dict: dict[str, Any]) -> Manifest:

    '''
    Build a Manifest from a validated YAML experiment dict.

    Branches on sfd.manifest.type to instantiate MLManifest or RuleBasedManifest.

    Args:
        yaml_dict (dict): Validated YAML dict from validator.validate()

    Returns:
        Manifest: Configured MLManifest or RuleBasedManifest

    Raises:
        ValueError: If manifest type is unknown or required fields are missing

    '''

    m = yaml_dict['sfd']['manifest']
    manifest_type = m['type']

    if manifest_type == 'ml':
        return _build_ml_manifest(m)
    if manifest_type == 'rule_based':
        return _build_rule_based_manifest(m)
    raise ValueError(f"Unknown manifest type '{manifest_type}'")


def _build_ml_manifest(m: dict[str, Any]) -> MLManifest:

    manifest = MLManifest()
    _apply_base(manifest, m)
    _apply_transforms(manifest, m)
    _apply_scaler(manifest, m)
    _apply_target(manifest, m)
    _apply_feature_ablation(manifest, m)
    _apply_pca_compression(manifest, m)
    _apply_calibration(manifest, m)
    _apply_ml_extras(manifest, m)
    manifest.with_reference_architecture(resolve(m['reference_architecture']))
    return manifest


def _build_rule_based_manifest(m: dict[str, Any]) -> RuleBasedManifest:

    manifest = RuleBasedManifest()
    _apply_base(manifest, m)
    strat = m['strategy']
    manifest.with_strategy(
        conditions=[dict(c) for c in strat['conditions']],
        entry=strat['entry'],
    )
    manifest.with_reference_architecture(resolve(m['reference_architecture']))
    return manifest


def _apply_base(manifest: Manifest, m: dict[str, Any]) -> None:

    ds = m['data_source']
    manifest.set_data_source(method=resolve(ds['method']), params=dict(ds.get('params') or {}))

    tds = m.get('test_data_source')
    if tds is not None:
        manifest.set_test_data_source(method=resolve(tds['method']), params=dict(tds.get('params') or {}))

    _apply_split(manifest, m)

    cols = m.get('required_columns')
    if cols is not None:
        if not isinstance(cols, list):
            raise ValueError(f"'required_columns' must be a list, got {type(cols).__name__}")
        manifest.set_required_bar_columns(list(cols))


def _apply_split(manifest: Manifest, m: dict[str, Any]) -> None:

    sd = m.get('split_dates')
    if sd is not None:
        manifest.set_split_dates(
            date.fromisoformat(sd['train_start']),
            date.fromisoformat(sd['train_end']),
            date.fromisoformat(sd['val_start']),
            date.fromisoformat(sd['val_end']),
            date.fromisoformat(sd['test_start']),
            date.fromisoformat(sd['test_end']),
        )
    else:
        sc = m['split_config']
        manifest.set_split_config(sc['train'], sc['val'], sc['test'])


def _apply_transforms(manifest: MLManifest, m: dict[str, Any]) -> None:

    psds = m.get('pre_split_data_selector')
    if psds is not None:
        manifest.set_pre_split_data_selector(
            resolve(psds['func']),
            **_resolve_func_params(dict(psds.get('params') or {})),
        )

    bf = m.get('bar_formation')
    if bf is not None:
        manifest.set_bar_formation(
            resolve(bf['func']),
            **_resolve_func_params(dict(bf.get('params') or {})),
        )

    for item in m.get('indicators') or []:
        manifest.add_indicator(resolve(item['func']), **_resolve_func_params(dict(item.get('params') or {})))

    for item in m.get('features') or []:
        manifest.add_feature(resolve(item['func']), **_resolve_func_params(dict(item.get('params') or {})))


def _apply_scaler(manifest: MLManifest, m: dict[str, Any]) -> None:

    scaler = m.get('scaler')
    if scaler is None:
        return
    if 'from_params' in scaler:
        manifest.set_scaler_from_params(param_name=scaler['from_params'])
    else:
        manifest.set_scaler(resolve(scaler['class']))


def _apply_target(manifest: MLManifest, m: dict[str, Any]) -> None:

    t = m['target']
    manifest.with_target_label(
        target_name=t['name'],
        target_class=resolve(t['class']),
        fit_params=dict(t.get('fit_params') or {}),
        transform_params=dict(t.get('transform_params') or {}),
    )


def _apply_feature_ablation(manifest: MLManifest, m: dict[str, Any]) -> None:

    fa = m.get('feature_ablation')
    if fa is None:
        return
    manifest.set_feature_ablation(
        drop_count_key=fa.get('drop_count_key', 'feature_drop_count'),
        seed_key=fa.get('seed_key', 'feature_drop_seed'),
    )


def _apply_pca_compression(manifest: MLManifest, m: dict[str, Any]) -> None:

    pca = m.get('pca_compression')
    if pca is not None:
        manifest.set_pca_compression(**dict(pca))


def _apply_calibration(manifest: MLManifest, m: dict[str, Any]) -> None:

    cal = m.get('calibration')
    if cal is None:
        return
    builder = manifest.with_calibration()
    prob = cal.get('probability_calibration')
    if prob is not None:
        builder.probability_calibration(
            func=resolve(prob['func']),
            **_resolve_func_params(dict(prob.get('params') or {})),
        )
    thresh = cal.get('threshold_function')
    if thresh is not None:
        builder.threshold_function(
            func=resolve(thresh['func']),
            **_resolve_func_params(dict(thresh.get('params') or {})),
        )
    if prob is not None or thresh is not None:
        builder.done()


def _apply_ml_extras(manifest: MLManifest, m: dict[str, Any]) -> None:

    dde = m.get('data_dict_extension')
    if dde is not None:
        manifest.add_to_data_dict(resolve(dde['func']))

    po = m.get('params_override')
    if po is not None:
        manifest.with_params_override(**dict(po))

    mp = m.get('metrics_params')
    if mp is not None:
        manifest.metrics_params = dict(mp)


class CompiledSFD:

    '''SFD-compatible object built from a validated YAML experiment dict.'''

    def __init__(self, yaml_dict: dict[str, Any]) -> None:

        self._yaml = yaml_dict
        self._manifest_cache: Manifest | None = None
        self.__name__ = f"yaml:{yaml_dict['metadata']['name']}"

    def params(self) -> dict[str, list[Any]]:

        '''Return the parameter search space.'''

        return {k: list(v) for k, v in self._yaml['sfd']['params'].items()}

    def manifest(self) -> Manifest:

        '''Return the compiled Manifest, built once and cached.'''

        if self._manifest_cache is None:
            self._manifest_cache = build_manifest(self._yaml)
        return self._manifest_cache
