from datetime import date
from typing import Any

from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.experiment.param_search.random_strategy import RandomStrategy
from limen.experiment.manifest_core import MLManifest
from limen.experiment.manifest_core import Manifest
from limen.experiment.manifest_core import RuleBasedManifest
from limen.experiment.param_domain import ParamDomain
from limen.yaml.errors import ResolutionError
from limen.yaml.resolver import resolve


def _resolve_func_params(params: dict[str, Any]) -> dict[str, Any]:

    '''
    Resolve any string values that are valid limen.* paths to their Python objects.

    limen.* strings are resolved eagerly and raise ResolutionError on failure.
    Other strings (round_params refs like '{threshold_min}', literals) are passed through unchanged.

    Args:
        params (dict): Raw params dict from YAML

    Returns:
        dict: Params with callable paths resolved to Python objects

    '''

    result = {}
    for k, v in params.items():
        if not isinstance(v, str):
            result[k] = v
        elif v.startswith('limen.'):
            resolved = resolve(v)
            if not callable(resolved):
                raise ValueError(
                    f"'{v}' resolves to {type(resolved).__name__}, not a callable"
                )
            result[k] = resolved
        else:
            try:
                result[k] = resolve(v)
            except ResolutionError:
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
    _apply_indicators(manifest, m)
    strat = m['strategy']
    manifest.with_strategy(
        conditions=[dict(c) for c in strat['conditions']],
        entry=strat['entry'],
    )
    manifest.with_reference_architecture(resolve(m['reference_architecture']))
    return manifest


def _apply_indicators(manifest: Manifest, m: dict[str, Any]) -> None:

    for item in m.get('indicators') or []:
        manifest.add_indicator(
            resolve(item['func']),
            include_if=item.get('include_if'),
            **_resolve_func_params(dict(item.get('params') or {})),
        )


def _apply_base(manifest: Manifest, m: dict[str, Any]) -> None:

    ds = m['data_source']
    params = dict(ds.get('params') or {})
    sd = m['split_dates']
    params['start_date_limit'] = sd['train_start']
    params['end_date_limit'] = sd['test_end']
    manifest.set_data_source(method=resolve(ds['method']), params=params)

    _apply_split(manifest, m)

    cols = m.get('required_columns')
    if cols is not None:
        if not isinstance(cols, list):
            raise ValueError(f"'required_columns' must be a list, got {type(cols).__name__}")
        manifest.set_required_bar_columns(list(cols))


def _apply_split(manifest: Manifest, m: dict[str, Any]) -> None:

    sd = m['split_dates']
    manifest.set_split_dates(
        date.fromisoformat(sd['train_start']),
        date.fromisoformat(sd['train_end']),
        date.fromisoformat(sd['val_start']),
        date.fromisoformat(sd['val_end']),
        date.fromisoformat(sd['test_start']),
        date.fromisoformat(sd['test_end']),
    )


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

    _apply_indicators(manifest, m)

    for item in m.get('features') or []:
        manifest.add_feature(
            resolve(item['func']),
            include_if=item.get('include_if'),
            **_resolve_func_params(dict(item.get('params') or {})),
        )


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

    manifest.decoder_lookback = int(m.get('decoder_lookback') or 1)


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


def build_search_strategy(yaml_dict: dict[str, Any]) -> RandomStrategy | GridStrategy:

    '''
    Build a search strategy from a validated YAML experiment dict.

    Args:
        yaml_dict (dict): Parsed and validated YAML experiment dict

    Returns:
        RandomStrategy | GridStrategy: Configured search strategy

    Raises:
        ValueError: If uel.search_strategy is not a mapping or has an unknown type

    '''

    uel_cfg = yaml_dict.get('uel') or {}
    sfd_cfg = yaml_dict.get('sfd') or {}
    strategy_cfg = uel_cfg.get('search_strategy', {})
    if not isinstance(strategy_cfg, dict):
        raise ValueError(
            f"'uel.search_strategy' must be a mapping, got {type(strategy_cfg).__name__}"
        )
    params = {k: list(v) for k, v in (sfd_cfg.get('params') or {}).items()}
    domain = ParamDomain(params)
    strategy_type = strategy_cfg.get('type', 'random')
    if strategy_type == 'grid':
        return GridStrategy(domain)
    if strategy_type == 'random':
        return RandomStrategy(domain)
    raise ValueError(f"Unknown search strategy type: '{strategy_type}'. Expected 'random' or 'grid'.")
