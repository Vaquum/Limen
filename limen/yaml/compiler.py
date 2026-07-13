import inspect
from datetime import date
from typing import Any
from typing import cast

from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.experiment.param_search.random_strategy import RandomStrategy
from limen.experiment.manifest_core import MLManifest
from limen.experiment.manifest_core import Manifest
from limen.experiment.manifest_core import RuleBasedManifest
from limen.experiment.param_domain import ParamDomain
from limen.yaml.config import is_list
from limen.yaml.config import is_mapping
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

    result: dict[str, Any] = {}
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
    _apply_strict_mode(manifest, m)
    _ = manifest.with_reference_architecture(resolve(m['reference_architecture']))
    return manifest


def _build_rule_based_manifest(m: dict[str, Any]) -> RuleBasedManifest:

    manifest = RuleBasedManifest()
    _apply_base(manifest, m)
    _apply_indicators(manifest, m)
    strat = m['strategy']
    _ = manifest.with_strategy(
        conditions=[dict(c) for c in strat['conditions']],
        entry=strat['entry'],
    )
    _ = manifest.with_reference_architecture(resolve(m['reference_architecture']))
    return manifest


def _apply_indicators(manifest: Manifest, m: dict[str, Any]) -> None:

    items: list[Any] = m.get('indicators') or []
    for item in items:
        _ = manifest.add_indicator(
            resolve(item['func']),
            include_if=item.get('include_if'),
            **_resolve_func_params(dict(cast(dict[str, Any], item.get('params') or {}))),
        )


def _apply_base(manifest: Manifest, m: dict[str, Any]) -> None:

    ds = m['data_source']
    params = dict(ds.get('params') or {})
    sd = m['split_dates']
    method = resolve(ds['method'])
    sig = inspect.signature(method)
    sig_params = sig.parameters
    accepts_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
    )
    if 'start_date_limit' in sig_params or accepts_var_keyword:
        params['start_date_limit'] = sd['train_start']
    if 'end_date_limit' in sig_params or accepts_var_keyword:
        params['end_date_limit'] = sd['test_end']
    _ = manifest.set_data_source(method=method, params=params)

    _apply_split(manifest, m)

    cols = m.get('required_columns')
    if cols is not None:
        if not is_list(cols):
            raise ValueError(f"'required_columns' must be a list, got {type(cols).__name__}")
        _ = manifest.set_required_bar_columns(list(cols))

    _apply_backtest(manifest, m)


def _apply_backtest(manifest: Manifest, m: dict[str, Any]) -> None:

    backtest = m.get('backtest')
    if not backtest:
        return
    kwargs = {key: backtest[key] for key in ('fee_bps', 'slip_bps', 'notional_rate') if key in backtest}
    _ = manifest.set_backtest_config(**kwargs)


def _apply_split(manifest: Manifest, m: dict[str, Any]) -> None:

    sd = m['split_dates']
    _ = manifest.set_split_dates(
        date.fromisoformat(sd['train_start']),
        date.fromisoformat(sd['train_end']),
        date.fromisoformat(sd['val_start']),
        date.fromisoformat(sd['val_end']),
        date.fromisoformat(sd['test_start']),
        date.fromisoformat(sd['test_end']),
        val_predict_guard=sd.get('val_predict_guard', True),
        test_predict_guard=sd.get('test_predict_guard', True),
    )


def _apply_transforms(manifest: MLManifest, m: dict[str, Any]) -> None:

    psds = m.get('pre_split_data_selector')
    if psds is not None:
        _ = manifest.set_pre_split_data_selector(
            resolve(psds['func']),
            **_resolve_func_params(dict(cast(dict[str, Any], psds.get('params') or {}))),
        )

    bf = m.get('bar_formation')
    if bf is not None:
        _ = manifest.set_bar_formation(
            resolve(bf['func']),
            **_resolve_func_params(dict(cast(dict[str, Any], bf.get('params') or {}))),
        )

    _apply_indicators(manifest, m)

    items: list[Any] = m.get('features') or []
    for item in items:
        _ = manifest.add_feature(
            resolve(item['func']),
            include_if=item.get('include_if'),
            **_resolve_func_params(dict(cast(dict[str, Any], item.get('params') or {}))),
        )


def _apply_scaler(manifest: MLManifest, m: dict[str, Any]) -> None:

    scaler = m.get('scaler')
    if scaler is None:
        return
    extra_params = dict(scaler.get('params') or {})
    if 'from_params' in scaler:
        _ = manifest.set_scaler_from_params(param_name=scaler['from_params'], extra_params=extra_params)
    else:
        _ = manifest.set_scaler(resolve(scaler['class']), extra_params=extra_params)


def _apply_strict_mode(manifest: MLManifest, m: dict[str, Any]) -> None:

    if m.get('strict_mode'):
        _ = manifest.set_strict_mode(True)


def _apply_target(manifest: MLManifest, m: dict[str, Any]) -> None:

    t = m['target']
    _ = manifest.with_target_label(
        target_name=t['name'],
        target_class=resolve(t['class']),
        fit_params=dict(t.get('fit_params') or {}),
        transform_params=dict(t.get('transform_params') or {}),
    )


def _apply_feature_ablation(manifest: MLManifest, m: dict[str, Any]) -> None:

    fa = m.get('feature_ablation')
    if fa is None:
        return
    _ = manifest.set_feature_ablation(
        drop_count_key=fa.get('drop_count_key', 'feature_drop_count'),
        seed_key=fa.get('seed_key', 'feature_drop_seed'),
    )


def _apply_pca_compression(manifest: MLManifest, m: dict[str, Any]) -> None:

    pca = m.get('pca_compression')
    if pca is not None:
        _ = manifest.set_pca_compression(**dict(pca))


def _apply_calibration(manifest: MLManifest, m: dict[str, Any]) -> None:

    cal = m.get('calibration')
    if cal is None:
        return
    builder = manifest.with_calibration()
    prob = cal.get('probability_calibration')
    if prob is not None:
        _ = builder.probability_calibration(
            func=resolve(prob['func']),
            **_resolve_func_params(dict(prob.get('params') or {})),
        )
    thresh = cal.get('threshold_function')
    if thresh is not None:
        _ = builder.threshold_function(
            func=resolve(thresh['func']),
            **_resolve_func_params(dict(thresh.get('params') or {})),
        )
    if prob is not None or thresh is not None:
        _ = builder.done()


def _apply_ml_extras(manifest: MLManifest, m: dict[str, Any]) -> None:

    dde = m.get('data_dict_extension')
    if dde is not None:
        _ = manifest.add_to_data_dict(resolve(dde['func']))

    po = m.get('params_override')
    if po is not None:
        _ = manifest.with_params_override(**dict(po))

    mp = m.get('metrics_params')
    if mp is not None:
        manifest.metrics_params = dict(mp)

    manifest.decoder_lookback = int(m.get('decoder_lookback') or 1)


class CompiledSFD:

    '''SFD-compatible object built from a validated YAML experiment dict.'''

    def __init__(self, yaml_dict: dict[str, Any]) -> None:

        super().__init__()

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

    uel_cfg: dict[str, Any] = yaml_dict.get('uel') or {}
    sfd_cfg: dict[str, Any] = yaml_dict.get('sfd') or {}
    strategy_cfg = uel_cfg.get('search_strategy', {})
    if not is_mapping(strategy_cfg):
        raise ValueError(
            f"'uel.search_strategy' must be a mapping, got {type(strategy_cfg).__name__}"
        )
    raw_params: dict[str, Any] = sfd_cfg.get('params') or {}
    params = {k: list(v) for k, v in raw_params.items()}
    domain = ParamDomain(params)
    strategy_type = strategy_cfg.get('type', 'random')
    seed = strategy_cfg.get('seed')
    if seed is not None:
        seed = int(seed)
    if strategy_type == 'grid':
        return GridStrategy(domain)
    if strategy_type == 'random':
        return RandomStrategy(domain, seed=seed)
    raise ValueError(f"Unknown search strategy type: '{strategy_type}'. Expected 'random' or 'grid'.")
