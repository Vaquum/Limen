import ast
from pathlib import Path
from catalogue_tests import validate_test_reference

REGISTRIES = {
    'search': ('limen/experiment/param_search/registry.py', 'STRATEGY_REGISTRY', 'docs/Advanced-Search.md', ['python', 'compiled'], 'search'),
    'scalers': ('limen/scalers/registry.py', 'SCALER_REGISTRY', 'docs/Scalers.md', ['python', 'compiled'], 'selection'),
    'reducers': ('limen/experiment/reducer/registry.py', 'REDUCER_REGISTRY', 'docs/Reducers-And-Feedback.md', ['python', 'compiled'], 'search'),
    'selectors': ('limen/cohort/sfc/__init__.py', 'BUILTIN_SELECTORS', 'docs/Cohort.md', ['compiled'], 'walk_forward'),
    'funding': ('limen/backtest/funding_presets.py', 'PRESETS', 'docs/Backtest.md', ['python', 'compiled'], 'execution'),
}


def assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                raise ValueError('Public declaration has no value: ' + name)
            return node.value
    raise ValueError('Missing declared public surface: ' + name)


def topic(module):
    if module == 'limen.experiment.holdout':
        return 'holdout'
    if module == 'limen.backtest.observed_costs':
        return 'observed_costs'
    if module == 'limen.sfd.reference_architecture.meta_allocation':
        return 'allocation'
    if module.startswith('limen.cli') or module == 'limen.yaml':
        return 'command_line'
    if module.startswith('limen.experiment.reducer'):
        return 'search'
    if module.startswith('limen.calibration'):
        return 'calibration'
    if module.startswith('limen.backtest'):
        return 'execution'
    if module.startswith(('limen.data', 'limen.features', 'limen.indicators', 'limen.transforms')):
        return 'bars'
    if module.startswith(('limen.experiment.param_search', 'limen.sfd')):
        return 'search'
    return 'selection'


def context(interfaces, scenario):
    return {'interfaces': interfaces, 'project_context': [interface + '/' + scenario for interface in interfaces], 'proof': 'public-signature-inventory'}


def inventory(root: Path):
    exports = {}
    for source in sorted((root / 'limen').rglob('*.py')):
        if any(part.startswith('_') and part != '__init__.py' for part in source.relative_to(root).parts):
            continue
        tree = ast.parse(source.read_text())
        declarations = [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(isinstance(target, ast.Name) and target.id == '__all__' for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
        if not declarations:
            continue
        names = sorted(ast.literal_eval(assignment(tree, '__all__')))
        if not names:
            continue
        module = '.'.join((source.parent if source.name == '__init__.py' else source.with_suffix('')).relative_to(root).parts)
        interfaces = (['python'] if module in ('limen.backtest.observed_costs', 'limen.backtest.short_flat_strategy') else ['compiled']
                      if module.startswith(('limen.cli', 'limen.yaml', 'limen.sfd.foundational_sfd', 'limen.sfd.reference_architecture.meta_allocation', 'limen.sfd.reference_architecture.direction_sizing', 'limen.targets.trade_outcome')) or module in ('limen.sfd', 'limen.experiment.holdout') else ['python', 'compiled'])
        reference = next(directory / 'README.md' for directory in source.parents if (directory / 'README.md').is_file())
        if module == 'limen':
            reference = root / 'README.md'
        elif module == 'limen.backtest.observed_costs':
            reference = root / 'docs/Backtest.md'
        exports[module] = context(interfaces, topic(module)) | {'source': str(source.relative_to(root)), 'entry_point': module,
                                                             'names': names, 'reference': str(reference.relative_to(root))}
        overrides = {'limen': ['MLManifest', 'Manifest', 'ReconstructionError', 'RuleBasedManifest', 'Trainer', 'sfd'],
                     'limen.experiment': ['CalibrationConfig', 'MLManifest', 'Manifest', 'ReuseContract', 'RuleBasedManifest', 'declare_reuse'],
                     'limen.inference': ['ReconstructionError', 'Trainer'],
                     'limen.sfd.reference_architecture': ['DirectionSizingModel', 'direction_sizing'],
                     'limen.targets': ['TradeOutcomeTarget']}
        if module in overrides:
            exports[module]['interface_overrides'] = dict.fromkeys(overrides[module], ['compiled'])
    registries = {}
    for name, (source, symbol, reference, interfaces, scenario) in REGISTRIES.items():
        value = assignment(ast.parse((root / source).read_text()), symbol)
        if not isinstance(value, ast.Dict):
            raise ValueError('Registry declaration must expose its current keys: ' + symbol)
        keys = []
        for key in value.keys:
            if key is None:
                raise ValueError('Registry inventory requires explicit keys: ' + symbol)
            keys.append(ast.literal_eval(key))
        registries[name] = context(interfaces, scenario) | {'source': source, 'entry_point': symbol, 'names': sorted(keys), 'reference': reference}
    schema = ast.parse((root / 'limen/yaml/schema.py').read_text() + '\n' + (root / 'limen/yaml/_assessment_spec.py').read_text())
    groups = {}
    for node in schema.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        names = [target.id for target in targets if isinstance(target, ast.Name) and target.id.endswith(('_REQUIRED', '_OPTIONAL', '_OPTIONAL_SHARED'))]
        for name in names:
            value = assignment(schema, name)
            if name == 'BACKTEST_OPTIONAL':
                trade = ast.parse((root / 'limen/experiment/_resolve_trade_policy.py').read_text())
                config = next(item for item in trade.body if isinstance(item, ast.ClassDef) and item.name == 'BacktestConfig')
                fields = [item.target.id for item in config.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)]
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'set' and not value.args:
                fields = []
            else:
                fields = ast.literal_eval(value)
            groups[name] = sorted(fields)
    manifest = context(['compiled'], 'selection') | {'source': 'limen/yaml/schema.py', 'entry_point': 'limen.yaml.validate',
                                                     'reference': 'docs/Experiment-Manifest.md', 'field_groups': groups}
    return {'public_exports': exports, 'registries': registries, 'manifest_fields': manifest}


def validate_surface_catalogue(root, catalogue):
    if catalogue['surfaces'] != inventory(root):
        raise ValueError('Public surface catalogue differs from the current exports, registries or Manifest fields')
    for entry in [*catalogue['surfaces']['public_exports'].values(), *catalogue['surfaces']['registries'].values(), catalogue['surfaces']['manifest_fields']]:
        if not (root / entry['reference']).is_file():
            raise ValueError('Public surface has no canonical reference: ' + entry['entry_point'])
        if entry['proof'] not in catalogue['reference_proofs'] or any(identity not in catalogue['scenario_details'] for identity in entry['project_context']):
            raise ValueError('Public surface has no reference proof or project context')
    expected = {interface + '/' + scenario for scenario, interfaces in catalogue['required_scenarios'].items() for interface in interfaces}
    if set(catalogue['scenario_details']) != expected:
        raise ValueError('Scenario descriptions do not cover the required interface matrix')
    for identity, entry in catalogue['scenario_details'].items():
        validate_test_reference(root, entry['test'])
        for reference in entry.get('extra_tests', []):
            validate_test_reference(root, reference)
        if identity != entry['interface'] + '/' + entry['scenario'] or entry['role'] != 'executed':
            raise ValueError('An execution obligation was relabelled or assigned to the wrong interface')
        source = root / entry['source']
        if not source.resolve().is_relative_to((root / f"docs/examples/{entry['interface']}-research").resolve()):
            raise ValueError('Wrong project source in the scenario catalogue')
        text = source.read_text()
        if text.count('# region ' + entry['region'] + '\n') != 1 or text.count('# endregion ' + entry['region']) != 1:
            raise ValueError('Scenario has no unique canonical source region')
        if not (root / entry['page']).is_file() or not entry['recorded_window'] or not entry['prerequisites'] or not entry['assertions']:
            raise ValueError('Scenario lacks a page, input window, prerequisites or independent assertions')
    for proof in catalogue['reference_proofs'].values():
        for reference in proof['tests']:
            validate_test_reference(root, reference)
