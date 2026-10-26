"""Inventory the validator's admitted paths and the nested parsers it delegates to."""
import ast
import hashlib
import json
from pathlib import Path

# Open parameter mappings delegate to the named callable's argument contract.
NESTED = {
    'uel': ('limen/yaml/rules.py', 'UelSpec'),
    'metadata': ('limen/yaml/_metadata_spec.py', 'check_metadata'),
    'sfd.manifest.required_columns': ('limen/experiment/_required_columns.py', 'column_issue'),
    'schema_version': ('limen/yaml/_schema_version.py', 'schema_errors'),
    'sfd.manifest': ('limen/yaml/_parameter_refs.py', 'lookup_references'),
    'sfd.params': ('limen/yaml/rules.py', 'ParamCoverage'),
    'sfd.manifest.strict_mode': ('limen/yaml/_boolean_spec.py', 'BooleanControlsSpec'),
    'sfd.params.use_calibration[]': ('limen/yaml/_boolean_spec.py', 'BooleanControlsSpec'),
    'sfd.params.use_threshold[]': ('limen/yaml/_boolean_spec.py', 'BooleanControlsSpec'),
    'sfd.manifest.split_walk_forward': ('limen/experiment/_walk_forward_split.py', 'WalkForwardConfig'),
    'sfd.manifest.backtest.product': ('limen/experiment/_resolve_trade_policy.py', 'ProductConfig'),
    'sfd.manifest.backtest.funding': ('limen/experiment/_resolve_trade_policy.py', 'FundingConfig'),
    'sfd.manifest.allocation': ('limen/experiment/_allocation.py', 'AllocationConfig'),
    'sfd.manifest.feature_shims[]': ('limen/experiment/_column_shims.py', '_FIELDS'),
    'sfd.manifest.objective': ('limen/yaml/_objective_spec.py', '_read_objective'),
    'sfd.manifest.acceptance': ('limen/experiment/_acceptance_settings.py', 'read_acceptance'),
    'sfd.manifest.strategy.conditions[]': ('limen/yaml/rules.py', 'ConditionsList'),
    'uel.intra_callback': ('limen/yaml/_callback_spec.py', 'callback_error'),
    'uel.pruning_strategies[]': ('limen/yaml/rules.py', 'PruningStrategiesSpec'),
    'sfd.manifest.indicators[]': ('limen/yaml/rules.py', '_check_func_block'),
    'uel.recording': ('limen/experiment/_recording.py', 'parse_recording'),
    'uel.output_path': ('limen/cli/commands/_results_dir.py', 'admit_output'),
    'assessment': ('limen/yaml/_assessment_spec.py', 'assessment_spec'),
    'holdout': ('limen/yaml/_assessment_spec.py', 'holdout_spec'),
}


def inventory(root: Path):
    from limen.yaml import validator
    rules = []

    def visit(items, condition='always'):
        for rule in items:
            kind, settings = type(rule).__name__, vars(rule)
            if kind == 'When':
                visit(settings['_rules'], settings['_condition_path'] + ' = ' + str(settings['_condition_value']))
            elif kind in ('Required', 'OneOf', 'NoUnknownKeys', 'Resolvable', 'FuncList', 'SingleFuncBlock', 'ConditionsList', 'PruningStrategiesSpec', 'ParamKeyFields'):
                entry = {'kind': kind, 'path': settings.get('_path', settings.get('path')), 'when': condition}
                for name in ('_known', '_choices', '_fields', '_severity', '_rejected'):
                    key = name if name in settings else name[1:]
                    if key in settings and (name != '_rejected' or settings[key]):
                        value = settings[key]
                        entry[name[1:]] = sorted(value) if isinstance(value, (set, frozenset, tuple)) else value
                if kind in ('FuncList', 'SingleFuncBlock', 'ConditionsList', 'PruningStrategiesSpec'):
                    entry['severity'] = vars(validator.NoUnknownKeys('', set()))['_severity']
                if kind == 'Required':
                    entry['type'] = settings['_type'].__name__ if settings['_type'] else None
                rules.append(entry)
    visit(vars(validator._TOP_LEVEL_ENGINE)['_rules'])
    visit(vars(validator._MAIN_ENGINE)['_rules'])
    nested = {}
    for path, (source, symbol) in NESTED.items():
        text = (root / source).read_text()
        tree = ast.parse(text)
        node = next(item for item in tree.body if getattr(item, 'name', None) == symbol or
                    isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == symbol for target in item.targets))
        segment = ast.get_source_segment(text, node)
        if segment is None:
            raise ValueError('Nested YAML contract has no source: ' + source + ':' + symbol)
        entry: dict[str, object] = {'source': source, 'symbol': symbol, 'sha256': hashlib.sha256(segment.encode()).hexdigest()}
        if isinstance(node, ast.ClassDef) and any(isinstance(child, ast.AnnAssign) for child in node.body):
            entry['fields'] = {child.target.id: {'type': ast.unparse(child.annotation),
                'default': ast.unparse(child.value) if child.value is not None else 'required'}
                for child in node.body if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)}
        elif isinstance(node, ast.Assign):
            entry['fields'] = sorted(ast.literal_eval(node.value))
        nested[path] = entry
    return {'rules': rules, 'nested': nested}


def validate(root):
    if json.loads((root / 'docs/examples/yaml-contracts.json').read_text()) != inventory(root):
        raise ValueError('YAML path, conditional rule or nested parser contract changed')
