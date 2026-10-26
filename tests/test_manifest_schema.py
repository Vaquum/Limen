from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

from markdown_it import MarkdownIt
from jsonschema import Draft7Validator
from jsonschema.validators import validator_for
import pytest

from limen.yaml import parse, validate
from limen.yaml.config import tomllib
from limen.yaml.schema import VERSION
from limen.yaml.store import commit_manifest, fork_manifest, resolve_manifest_uri

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('manifest_contracts', ROOT / 'docs/examples/schema_contracts.py')
assert SPEC is not None and SPEC.loader is not None
CONTRACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACTS)


def leaves(errors):
    for error in errors:
        yield error
        yield from leaves(error.context)


def at(document, path):
    for key in path.split('.') if path else ():
        if not isinstance(document, dict) or key not in document:
            return None
        document = document[key]
    return document


def schema_node(schema, path, condition='always'):
    def concrete(node):
        if '$ref' in node:
            return concrete(schema['definitions'][node['$ref'].rsplit('/', 1)[1]])
        if 'anyOf' in node and 'properties' not in node:
            return concrete(next(item for item in node['anyOf'] if item.get('type') != 'null'))
        if 'then' in node:
            return concrete(node['then'])
        return node
    node = schema
    fields = path.split('.') if path else []
    for index, key in enumerate(fields):
        node = concrete(node)['properties'][key]
        if fields[:index + 1] == ['sfd', 'manifest'] and (condition != 'always' or (len(fields) > 2 and fields[2] != 'type')):
            node = schema['definitions']['ruleManifest' if condition.endswith(' = rule_based') else 'mlManifest']
    return concrete(node)


@pytest.fixture(scope='module')
def schema_validator():
    schema = json.loads((ROOT / 'limen/yaml/manifest.schema.json').read_text())
    assert validator_for(schema) is Draft7Validator
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema)


@pytest.fixture(scope='module')
def corpus():
    documents = []
    for directory in ('limen/yaml/templates', 'docs/examples', 'tests/fixtures'):
        for path in sorted((ROOT / directory).rglob('*.yaml')):
            document, errors = parse(path)
            if errors or not isinstance(document.get('sfd'), dict):
                continue
            native = validate(document)
            assert native.valid, (path, native.errors)
            documents.append((path, document))
    for path in sorted((ROOT / 'docs').rglob('*.md')):
        for index, token in enumerate(MarkdownIt().parse(path.read_text())):
            if token.type != 'fence' or token.info.strip() != 'yaml':
                continue
            document, errors = parse(token.content)
            if errors or not {'schema_version', 'metadata', 'sfd', 'uel'} <= document.keys():
                continue
            assert validate(document).valid, (path, index)
            documents.append((str(path) + ':' + str(index), document))
    assert documents and {item['sfd']['manifest']['type'] for _, item in documents} == {'ml', 'rule_based'}
    baseline = next(item for path, item in documents if path == ROOT / 'docs/examples/compiled-research/selection.yaml')
    for name, value in (('data_dict_extension', {'func': 'limen.data.HistoricalData.get_spot_klines'}), ('feature_ablation', {}), ('pca_compression', {}), ('feature_shims', [{'columns': ['close'], 'method': 'log1p'}])):
        document = deepcopy(baseline)
        document['sfd']['manifest'][name] = value
        assert validate(document).valid
        documents.append((name + ' structural admission control', document))
    return documents


@pytest.fixture
def recorded_manifest():
    document, errors = parse(ROOT / 'docs/examples/compiled-research/selection.yaml')
    assert not errors and validate(document).valid
    return document


def test_manifest_schema_owns_the_current_version_and_inventory(schema_validator):
    schema = schema_validator.schema
    assert schema['properties']['schema_version']['const'] == VERSION
    assert schema['x-limen-package-version'] == tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']
    assert schema['x-limen-contract-inventory-sha256'] == hashlib.sha256(
        (ROOT / 'docs/examples/yaml-contracts.json').read_bytes()).hexdigest()
    assert '$id' not in schema
    def refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == '$ref':
                    yield value
                yield from refs(value)
        elif isinstance(node, list):
            for value in node:
                yield from refs(value)
    assert all(value.startswith('#/definitions/') for value in refs(schema))
    CONTRACTS.validate(ROOT)


def test_every_native_inventory_shape_matches_the_schema(schema_validator):
    schema = schema_validator.schema
    rules = CONTRACTS.inventory(ROOT)['rules']
    for rule in rules:
        if rule['kind'] in ('Resolvable', 'ParamKeyFields'):
            continue
        node = schema_node(schema, rule['path'], rule['when'])
        kind = rule['kind']
        if kind in ('NoUnknownKeys', 'SingleFuncBlock', 'FuncList', 'ConditionsList', 'PruningStrategiesSpec'):
            if kind in ('FuncList', 'ConditionsList', 'PruningStrategiesSpec'):
                node = schema_node({'properties': {'item': node['items']}, 'definitions': schema['definitions']}, 'item')
            peers = [item for item in rules if item['path'] == rule['path'] and item['when'] == rule['when'] and 'known' in item]
            allowed = set.intersection(*(set(item['known']) - set(item.get('rejected', {})) for item in peers))
            assert set(node['properties']) == allowed, rule
            assert node['additionalProperties'] is (rule['severity'] != 'error'), rule
        elif kind == 'OneOf':
            required = any(item['kind'] == 'Required' and item['path'] == rule['path'] and item['when'] == rule['when'] for item in rules)
            assert set(node['enum']) == set(rule['choices']) | (set() if required else {None}), rule
        elif kind == 'Required':
            parent, _, key = rule['path'].rpartition('.')
            owner = schema_node(schema, parent, rule['when'])
            assert key in owner['required'], rule
            expected = {'dict': 'object', 'str': 'string', 'int': 'integer', 'list': 'array'}[rule['type']]
            assert node.get('type') == expected or (expected == 'string' and all(isinstance(value, str) for value in node['enum'])), rule


def test_all_discovered_native_manifests_pass_portable_schema(schema_validator, corpus):
    for path, document in corpus:
        assert not list(schema_validator.iter_errors(document)), path


def test_delegated_helpers_constants_and_external_fields_are_bound(tmp_path):
    original = CONTRACTS.inventory(ROOT)
    for source in original['sources']:
        dest = tmp_path / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, dest)
    assert CONTRACTS.inventory(tmp_path) == original
    for source, before, after in (
        ('limen/yaml/_metadata_spec.py', "v?(?:[0-9]+!)?", "v?(?:[1-9]+!)?"),
        ('limen/experiment/_required_columns.py', '.strip()', '.lstrip()'),
        ('limen/experiment/_resolve_trade_policy.py', "'max_exposure', 'signal_change_bps'", "'signal_change_bps'"),
        ('limen/experiment/reducer/registry.py', "'budget': BudgetReducer", "'budget_changed': BudgetReducer"),
    ):
        path = tmp_path / source
        text = path.read_text()
        assert before in text
        path.write_text(text.replace(before, after, 1))
        changed = CONTRACTS.inventory(tmp_path)
        assert changed['sources'][source] != original['sources'][source]
        assert changed != original
        path.write_text(text)
    assert CONTRACTS.inventory(tmp_path) == original


@pytest.mark.parametrize('lineage', [17, [], None])
def test_manifest_schema_preserves_native_lineage_admission(schema_validator, recorded_manifest, lineage):
    recorded_manifest['lineage'] = lineage
    assert validate(recorded_manifest).valid
    assert not list(schema_validator.iter_errors(recorded_manifest))


def test_real_commit_fork_and_recommit_lineage(schema_validator, tmp_path):
    shutil.copyfile(ROOT / 'docs/examples/compiled-research/limen.toml', tmp_path / 'limen.toml')
    source = tmp_path / 'selection.yaml'
    shutil.copyfile(ROOT / 'docs/examples/compiled-research/selection.yaml', source)
    identity, existed = commit_manifest(source, tmp_path)
    assert not existed
    committed, _ = resolve_manifest_uri('manifest://' + identity, tmp_path)
    fork = tmp_path / 'fork.yaml'
    assert fork_manifest(committed, fork, 'schema_recorded_fork') == identity
    working, errors = parse(fork)
    assert not errors and working['lineage'] == {'parent_id': identity}
    child, existed = commit_manifest(fork, tmp_path)
    assert not existed and child != identity
    recommitted, _ = resolve_manifest_uri('manifest://' + child, tmp_path)
    for path in (committed, fork, recommitted):
        document, errors = parse(path)
        assert not errors and validate(document).valid
        assert not list(schema_validator.iter_errors(document))
    final, _ = parse(recommitted)
    assert set(final['lineage']) == {'id', 'committed_at', 'parent_id'}
    assert final['lineage']['id'] == child and final['lineage']['parent_id'] == identity


def test_manifest_schema_preserves_distinct_integer_and_float_choices(schema_validator, recorded_manifest):
    recorded_manifest['sfd']['params']['C'] = [1, 1.0]
    assert validate(recorded_manifest).valid
    assert not list(schema_validator.iter_errors(recorded_manifest))


def test_native_contracts_generate_located_structural_negatives(schema_validator, corpus):
    count = 0
    for rule in CONTRACTS.inventory(ROOT)['rules']:
        if rule['kind'] not in ('NoUnknownKeys', 'Required', 'OneOf') or rule['path'] == 'lineage':
            continue
        for source, baseline in corpus:
            if rule['when'] != 'always' and not rule['when'].endswith(' = ' + baseline['sfd']['manifest']['type']):
                continue
            original = at(baseline, rule['path'])
            if original is None or (rule['kind'] == 'NoUnknownKeys' and not isinstance(original, dict)):
                continue
            document = deepcopy(baseline)
            path = rule['path']
            if rule['kind'] == 'NoUnknownKeys':
                at(document, path)['unknown_contract_field'] = True
                native_path = (path + '.' if path else '') + 'unknown_contract_field'
                schema_path, keyword = path, 'additionalProperties'
            else:
                parent, _, field = path.rpartition('.')
                owner = at(document, parent)
                if rule['kind'] == 'Required':
                    del owner[field]
                    schema_path, keyword = parent, 'required'
                else:
                    owner[field] = '__invalid_enum__'
                    schema_path, keyword = path, 'enum'
                native_path = path
            result = validate(document)
            assert any(error.path == native_path for error in result.errors), (rule, source, result.errors)
            failures = list(leaves(schema_validator.iter_errors(document)))
            assert any('.'.join(map(str, error.absolute_path)) == schema_path and error.validator == keyword for error in failures), (rule, source, failures)
            count += 1
            break
        else:
            pytest.fail('No recorded positive for native rule: ' + str(rule))
    assert count > 30


# Delegated parsers publish owner scopes. Preserve both locations and require the
# corresponding message; an unrelated native error cannot satisfy this proof.
OWNERS = {
    ('assessment', 'top_k'): ('assessment/holdout', 'assessment.top_k'),
    ('uel.recording', 'predictions'): ('uel.recording', 'predictions'),
    ('sfd.manifest.objective', 'direction'): ('sfd.manifest.objective', 'Objective'),
}


@pytest.mark.parametrize(('owner', 'field'), OWNERS)
def test_delegated_owner_diagnostics_match_the_schema_offender(schema_validator, corpus, owner, field):
    native_owner, token = OWNERS[(owner, field)]
    baseline = next(document for _, document in corpus if isinstance(at(document, owner), dict) and field in at(document, owner))
    assert validate(deepcopy(baseline)).valid
    document = deepcopy(baseline)
    at(document, owner)[field] = '__wrong_literal__'
    native = validate(document)
    assert any(error.path == native_owner and token in error.message for error in native.errors)
    errors = list(leaves(schema_validator.iter_errors(document)))
    keyword = 'enum' if field == 'direction' else 'type'
    assert any('.'.join(map(str, error.absolute_path)) == owner + '.' + field and error.validator == keyword for error in errors)
    unrelated = deepcopy(baseline)
    unrelated['metadata']['name'] = False
    assert not any(error.path == native_owner and token in error.message for error in validate(unrelated).errors)


def test_delegated_closed_fields_match_their_actual_sources(schema_validator):
    current = CONTRACTS.inventory(ROOT)['nested']
    schema = schema_validator.schema
    for path, definition in (
        ('sfd.manifest.split_walk_forward', 'splitWalkForward'),
        ('sfd.manifest.backtest.product', 'product'),
        ('sfd.manifest.backtest.funding', 'funding'),
        ('sfd.manifest.allocation', 'tradeSource'),
        ('sfd.manifest.feature_shims[]', 'featureShim'),
    ):
        assert set(schema['definitions'][definition]['properties']) == set(current[path]['fields']), path


def positions(document, path):
    if not path:
        yield (), document
        return
    head, _, tail = path.partition('.')
    array = head.endswith('[]')
    key = head[:-2] if array else head
    if not isinstance(document, dict) or key not in document:
        return
    value = document[key]
    if array:
        if isinstance(value, list):
            for index, item in enumerate(value):
                for suffix, found in positions(item, tail):
                    yield (key, index, *suffix), found
    else:
        for suffix, found in positions(value, tail):
            yield (key, *suffix), found


def path_string(position):
    result = ''
    for value in position:
        result += f'[{value}]' if isinstance(value, int) else ('.' if result else '') + value
    return result


def member(document, position):
    for key in position:
        document = document[key]
    return document


@pytest.mark.parametrize('path', [
    'sfd.manifest.split_walk_forward', 'sfd.manifest.acceptance',
    'sfd.manifest.backtest.product', 'sfd.manifest.backtest.funding',
    'sfd.manifest.allocation', 'sfd.manifest.strategy.conditions[]', 'sfd.manifest.feature_shims[]',
    'uel.pruning_strategies[]', 'uel.recording', 'assessment', 'holdout',
    'holdout.scenarios[]', 'holdout.evaluation_target',
])
def test_unknown_delegated_fields_refuse_the_same_offender(schema_validator, corpus, path):
    for _, baseline in corpus:
        for position, value in positions(baseline, path):
            if not isinstance(value, dict):
                continue
            document = deepcopy(baseline)
            member(document, position)['unknown_contract_field'] = True
            native = validate(document)
            owner = path_string(position)
            if path.startswith(('assessment', 'holdout')):
                native_owner = 'assessment/holdout'
            elif path in ('uel.recording', 'sfd.manifest.feature_shims[]'):
                native_owner = path.removesuffix('[]')
            elif path in ('sfd.manifest.backtest.product', 'sfd.manifest.backtest.funding', 'sfd.manifest.split_walk_forward', 'sfd.manifest.acceptance', 'sfd.manifest.allocation'):
                native_owner = owner
            else:
                native_owner = owner + '.unknown_contract_field'
            assert any(error.path == native_owner and ('unknown_contract_field' in error.message or 'exactly' in error.message or 'only' in error.message) for error in native.errors), (path, native.errors)
            errors = list(leaves(schema_validator.iter_errors(document)))
            assert any(tuple(error.absolute_path) == position and error.validator == 'additionalProperties' for error in errors), (path, errors)
            unrelated = deepcopy(baseline)
            unrelated['metadata']['name'] = False
            assert not any(error.path == native_owner for error in validate(unrelated).errors)
            return
    pytest.fail('No recorded positive for closed delegated block: ' + path)


@pytest.mark.parametrize(('path', 'wrong'), [
    ('uel.pruning_strategies', None), ('uel.pruning_strategies[].params', None),
    ('sfd.manifest.split_walk_forward', None), ('sfd.manifest.acceptance', None),
    ('sfd.manifest.feature_shims', None), ('sfd.manifest.decoder_lookback', '1'),
    ('sfd.manifest.decoder_lookback', True), ('sfd.manifest.decoder_lookback', {}),
    ('uel.workers', '{workers}'), ('uel.recording.predictions', '{predictions}'),
    ('sfd.manifest.split_walk_forward.n_folds', '{folds}'),
    ('sfd.manifest.acceptance.max_pbo', '{pbo}'),
    ('sfd.manifest.objective.direction', '{direction}'),
])
def test_present_nulls_wrong_types_and_forbidden_references(schema_validator, corpus, recorded_manifest, path, wrong):
    baselines = [item for _, item in corpus]
    if path in ('sfd.manifest.feature_shims', 'sfd.manifest.decoder_lookback', 'uel.workers'):
        baseline = deepcopy(recorded_manifest)
        owner, _, key = path.rpartition('.')
        at(baseline, owner)[key] = {'feature_shims': [{'columns': ['close'], 'method': 'log1p'}], 'decoder_lookback': 1, 'workers': 1}[key]
        baselines.append(baseline)
    for baseline in baselines:
        for position, _ in positions(baseline, path):
            assert validate(deepcopy(baseline)).valid
            document = deepcopy(baseline)
            member(document, position[:-1])[position[-1]] = wrong
            native = validate(document)
            native_path = path_string(position)
            for prefix in ('sfd.manifest.split_walk_forward', 'sfd.manifest.acceptance', 'sfd.manifest.objective', 'uel.recording'):
                if native_path.startswith(prefix):
                    native_path = prefix
            assert any(error.path == native_path for error in native.errors), (path, native.errors)
            failures = list(leaves(schema_validator.iter_errors(document)))
            assert any(tuple(error.absolute_path) == position and error.validator in ('type', 'enum') for error in failures), (path, failures)
            return
    pytest.fail('No recorded positive for contract: ' + path)


@pytest.mark.parametrize('kind', ['ml', 'rule_based'])
@pytest.mark.parametrize('split', ['missing', 'both'])
def test_exactly_one_split_mode_is_required(schema_validator, corpus, kind, split):
    baseline = next(item for _, item in corpus if item['sfd']['manifest']['type'] == kind and 'split_dates' in item['sfd']['manifest'])
    document = deepcopy(baseline)
    manifest = document['sfd']['manifest']
    if split == 'missing':
        del manifest['split_dates']
    else:
        walk_forward = next(item['sfd']['manifest']['split_walk_forward'] for _, item in corpus if 'split_walk_forward' in item['sfd']['manifest'])
        manifest['split_walk_forward'] = deepcopy(walk_forward)
    native = validate(document)
    assert any(error.path == 'sfd.manifest' + ('.split_walk_forward' if split == 'both' else '') for error in native.errors)
    assert any(list(error.absolute_path) == ['sfd', 'manifest'] and error.validator == 'oneOf' for error in leaves(schema_validator.iter_errors(document)))


def test_required_native_types_reject_at_the_same_location(schema_validator, corpus):
    for rule in CONTRACTS.inventory(ROOT)['rules']:
        if rule['kind'] != 'Required':
            continue
        baseline = next(item for _, item in corpus if at(item, rule['path']) is not None and
            (rule['when'] == 'always' or rule['when'].endswith(' = ' + item['sfd']['manifest']['type'])))
        document = deepcopy(baseline)
        parent, _, field = rule['path'].rpartition('.')
        at(document, parent)[field] = {'dict': False, 'list': 17, 'str': 17, 'int': 'wrong'}[rule['type']]
        native = validate(document)
        assert any(error.path == rule['path'] for error in native.errors), (rule, native.errors)
        assert any('.'.join(map(str, error.absolute_path)) == rule['path'] and error.validator in ('type', 'enum')
                   for error in leaves(schema_validator.iter_errors(document))), rule


@pytest.mark.parametrize(('path', 'value'), [
    ('uel.output_format', None), ('uel.search_strategy.type', None),
    ('sfd.manifest.decoder_lookback', None), ('sfd.manifest.backtest.product', {}),
    ('sfd.manifest.backtest.funding.params', 17), ('sfd.manifest.backtest.funding.adapter', 17),
])
def test_native_omission_and_deferred_consumer_forms_remain_admitted(schema_validator, recorded_manifest, path, value):
    owner = recorded_manifest
    keys = path.split('.')
    for key in keys[:-1]:
        owner = owner.setdefault(key, {})
    owner[keys[-1]] = value
    assert validate(recorded_manifest).valid
    assert not list(schema_validator.iter_errors(recorded_manifest))


def test_condition_leaf_contracts_require_the_native_fields(schema_validator, corpus):
    from limen.yaml.rules import _CONDITION_LEAF_FIELDS
    from limen.experiment.reducer.registry import REDUCER_REGISTRY
    assert set(schema_validator.schema['definitions']['condition']['properties']['type']['enum']) == set(_CONDITION_LEAF_FIELDS)
    assert set(schema_validator.schema['definitions']['uel']['properties']['pruning_strategies']['items']['properties']['type']['enum']) == set(REDUCER_REGISTRY)
    for kind, fields in _CONDITION_LEAF_FIELDS.items():
        candidates = [(document, position) for _, document in corpus
                      for position, condition in positions(document, 'sfd.manifest.strategy.conditions[]')
                      if isinstance(condition, dict) and condition.get('type') == kind]
        assert candidates, kind
        baseline, position = candidates[0]
        for field in ('id', 'name', *fields):
            document = deepcopy(baseline)
            del member(document, position)[field]
            assert any(error.path == path_string(position) + '.' + field for error in validate(document).errors)
            assert any(tuple(error.absolute_path) == position and error.validator == 'required'
                       for error in leaves(schema_validator.iter_errors(document)))
