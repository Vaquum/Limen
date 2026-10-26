import hashlib
import json
import math
import re
import shlex
from pathlib import Path
import subprocess

from conftest import PROOF, ROOT
from jsonschema import Draft7Validator
from markdown_it import MarkdownIt
import pytest

from limen.yaml import parse, validate


def json_compatible(value):
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(json_compatible(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and json_compatible(item) for key, item in value.items())
    return False


def editor_cases(root):
    cases = []
    for directory in ('limen/yaml/templates', 'docs/examples', 'tests/fixtures'):
        for path in sorted((root / directory).rglob('*.yaml')):
            raw = path.read_text()
            document, errors = parse(raw)
            if not errors and isinstance(document.get('sfd'), dict):
                assert validate(document).valid, path
                cases.append({'id': 'corpus-' + str(len(cases)), 'source': str(path.relative_to(root)), 'yaml': raw})
    for path in sorted({root / 'README.md', *(root / 'docs').rglob('*.md'), *(root / 'limen').rglob('README.md')}):
        for index, token in enumerate(MarkdownIt().parse(path.read_text())):
            if token.type != 'fence' or token.info.strip() != 'yaml':
                continue
            document, errors = parse(token.content)
            if not errors and {'schema_version', 'metadata', 'sfd', 'uel'} <= document.keys():
                assert validate(document).valid, (path, index)
                cases.append({'id': 'corpus-' + str(len(cases)), 'source': str(path.relative_to(root)) + ':' + str(index), 'yaml': token.content})
    raw = (root / 'docs/examples/compiled-research/selection.yaml').read_text()
    variants = {
        'valid': raw,
        'unicode-padded-reference': raw.replace('fee_bps: 1.0', 'fee_bps: ' + json.dumps('\u0085{C}\u001c')),
        'wrong-type': raw.replace('n_permutations: 2', 'n_permutations: wrong'),
        'unknown-ml-key': raw.replace('type: ml', 'type: ml\n    unknown_contract_field: 1'),
        'integer-float': raw.replace('n_permutations: 2', 'n_permutations: 2.0'),
        'lexical-int': raw.replace('type: grid', 'type: grid\n    seed: 0x1\n    shuffle: true'),
        'unquoted-date': raw.replace('"2025-01-01"', '2025-01-01'),
        'unquoted-timestamp': raw.replace('mode: development', 'mode: development\n  created_at: 2025-01-01T00:00:00Z'),
        'non-string-key': raw.replace('C: [1.0, 0.5]', 'C: [{1: value}, 0.5]'),
        'duplicate-key': raw.replace('n_permutations: 2', 'n_permutations: 1\n  n_permutations: 2'),
        'complex-mapping-key': raw + '\n? {one: two}\n: true\n',
        'complex-sequence-key': raw + '\n? [one, two]\n: true\n',
        'malformed-yaml': raw + '\nincomplete: [\n',
        'nonfinite': raw.replace('C: [1.0, 0.5]', 'C: [.inf, 0.5]'),
        'tagged-binary': raw.replace('C: [1.0, 0.5]', 'C: [!!binary SGVsbG8=, 0.5]'),
    }
    for name, columns in (('blank-space', [' ']), ('blank-nel', ['\u0085']), ('blank-separator', ['\u001c']),
                          ('nonblank-zero-width', ['\u200b']), ('padding-order', [' close ', 'close'])):
        variants[name] = raw.replace('type: ml', 'type: ml\n    required_columns: ' + json.dumps(columns))
    cases.extend({'id': name, 'source': 'recorded selection structural/data-model control', 'yaml': yaml} for name, yaml in variants.items())
    return cases


def run_editor(root, schema_path, cases, temporary):
    inputs, output = temporary / 'cases.json', temporary / 'editor.json'
    inputs.write_text(json.dumps(cases))
    subprocess.run(['node', str(root / 'docs-site/scripts/prove-manifest-schema.mjs'), str(schema_path), str(inputs), str(output)],
                   check=True, cwd=temporary, timeout=30)
    result = json.loads(output.read_text())
    for source, case in zip(cases, result['cases'], strict=True):
        document, errors = parse(source['yaml'])
        native = validate(document) if not errors else None
        case['source'] = source['source']
        case['native'] = {'parser_errors': [{'path': error.path, 'message': error.message} for error in errors],
                          'valid': native.valid if native is not None else None,
                          'errors': [{'path': error.path, 'message': error.message} for error in native.errors] if native is not None else [],
                          'json_compatible': json_compatible(document) if not errors else False}
        if case['id'] in ('integer-float', 'lexical-int', 'unquoted-date', 'unquoted-timestamp', 'non-string-key', 'tagged-binary'):
            observed = {'integer-float': document['uel']['n_permutations'], 'lexical-int': document['uel']['search_strategy'].get('seed'),
                        'unquoted-date': document['sfd']['manifest']['split_dates']['train_start'],
                        'unquoted-timestamp': document['metadata'].get('created_at'),
                        'non-string-key': next(iter(document['sfd']['params']['C'][0])) if isinstance(document['sfd']['params']['C'][0], dict) else None,
                        'tagged-binary': document['sfd']['params']['C'][0]}[case['id']]
            case['native']['observed_type'] = type(observed).__name__
    return result


@pytest.fixture(scope='session')
def editor_proof(research_proof, request, tmp_path_factory):
    wheel = request.config.stash[PROOF]
    location = Path(wheel['installed']['import_location'])
    assert not location.is_relative_to(ROOT) and 'site-packages' in location.parts
    python = location.parents[4] / 'bin/python'
    temporary = tmp_path_factory.mktemp('manifest_editor')
    installed_copy = temporary / 'limen-manifest.schema.json'
    guide = (ROOT / 'docs/Experiment-Manifest.md').read_text()
    command = re.search(r'`(python -I -c "[^`]+")`', guide)
    assert command is not None
    argv = shlex.split(command[1])
    subprocess.run([str(python), *argv[1:]], cwd=temporary, check=True, timeout=30)
    inspect = "import json,sys,limen; from importlib import metadata; from pathlib import Path; assert 'limen.yaml' not in sys.modules; print(json.dumps({'import_location':str(Path(limen.__file__).resolve()),'version':metadata.version('vaquum-limen')}))"
    copied = json.loads(subprocess.check_output([str(python), '-I', '-c', inspect], cwd=temporary, text=True))
    assert copied == {'import_location': str(location), 'version': wheel['installed']['version']}
    assert installed_copy.read_bytes() == (ROOT / 'limen/yaml/manifest.schema.json').read_bytes()
    result = run_editor(ROOT, installed_copy, editor_cases(ROOT), temporary)
    assert result['schema_sha256'] == hashlib.sha256(installed_copy.read_bytes()).hexdigest()
    result['installed'] = copied | {'wheel_sha256': wheel['wheel_sha256']}
    target = ROOT / '.docs-evidence/manifest-editor.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def test_installed_schema_association_in_the_pinned_editor(editor_proof):
    assert editor_proof['engine']['version'] == '1.25.0'
    assert editor_proof['schema_store'] is False
    assert editor_proof['requests'] == [editor_proof['schema_uri']]
    cases = {item['id']: item for item in editor_proof['cases']}
    corpus = [item for item in editor_proof['cases'] if item['id'].startswith('corpus-')]
    assert len(corpus) >= 76
    validator = Draft7Validator(json.loads((ROOT / 'limen/yaml/manifest.schema.json').read_text()))
    for item in [*corpus, cases['valid'], cases['unicode-padded-reference']]:
        assert item['native']['valid'] and item['native']['json_compatible'], item['id']
        assert item['json_compatible'] and not item['parser_diagnostics'] and not item['diagnostics'], item['id']
        assert len(item['models']) == 1 and not list(validator.iter_errors(item['models'][0])), item['id']
    for identity, token, native_path in (
        ('wrong-type', 'wrong', 'uel.n_permutations'),
        ('unknown-ml-key', 'unknown_contract_field', 'sfd.manifest.unknown_contract_field'),
    ):
        item = cases[identity]
        assert not item['parser_diagnostics']
        assert any(error['path'] == native_path for error in item['native']['errors'])
        raw = next(source['yaml'] for source in editor_cases(ROOT) if source['id'] == identity)
        diagnostics = [entry for entry in item['diagnostics'] if entry['source'].startswith('yaml-schema:')]
        assert diagnostics and any(token in raw.splitlines()[entry['range']['start']['line']][entry['range']['start']['character']:entry['range']['end']['character']] for entry in diagnostics)
        assert list(validator.iter_errors(item['models'][0]))


def test_actual_editor_representation_gaps_are_explicit(editor_proof):
    cases = {item['id']: item for item in editor_proof['cases']}
    for identity, locus in (('integer-float', 'uel.n_permutations'), ('lexical-int', 'uel.search_strategy.seed'),
                            ('unquoted-date', 'sfd.manifest.split_dates.train_start'), ('unquoted-timestamp', 'metadata.created_at')):
        item = cases[identity]
        assert any(error['path'] == locus for error in item['native']['errors'])
        assert not item['parser_diagnostics'] and not item['diagnostics']
    assert cases['integer-float']['native']['observed_type'] == 'ScalarFloat'
    assert cases['integer-float']['models'][0]['uel']['n_permutations'] == 2
    assert cases['lexical-int']['native']['observed_type'] == 'HexInt'
    assert cases['lexical-int']['models'][0]['uel']['search_strategy']['seed'] == 1
    assert cases['unquoted-date']['native']['observed_type'] == 'date'
    assert cases['unquoted-date']['models'][0]['sfd']['manifest']['split_dates']['train_start'] == '2025-01-01'
    assert cases['unquoted-timestamp']['native']['observed_type'] == 'TimeStamp'
    for identity in ('duplicate-key', 'malformed-yaml'):
        assert cases[identity]['native']['parser_errors']
        assert cases[identity]['parser_diagnostics']
        assert any(item.get('source') == 'YAML' for item in cases[identity]['diagnostics'])
    assert cases['complex-mapping-key']['native']['parser_errors']
    assert not cases['complex-sequence-key']['native']['json_compatible']
    assert cases['non-string-key']['native']['valid'] and not cases['non-string-key']['native']['json_compatible']
    assert cases['non-string-key']['native']['observed_type'] == 'int'
    assert cases['non-string-key']['models'][0]['sfd']['params']['C'][0] == {'1': 'value'}
    assert cases['nonfinite']['native']['valid'] and not cases['nonfinite']['native']['json_compatible']
    assert not cases['nonfinite']['json_compatible'] and cases['nonfinite']['non_json_values'] == [{'path': '[0].sfd.params.C[0]', 'value': 'Infinity'}]
    assert 'models' not in cases['nonfinite']
    assert cases['tagged-binary']['native']['valid'] and not cases['tagged-binary']['native']['json_compatible']
    assert cases['tagged-binary']['native']['observed_type'] == 'bytes'
    assert cases['tagged-binary']['models'][0]['sfd']['params']['C'][0] == 'SGVsbG8='


def test_required_column_unicode_equality_padding_and_order_in_the_editor(editor_proof):
    cases = {item['id']: item for item in editor_proof['cases']}
    for identity in ('blank-space', 'blank-nel', 'blank-separator'):
        item = cases[identity]
        assert not item['native']['valid'] and any(error['path'] == 'sfd.manifest.required_columns[0]' for error in item['native']['errors'])
        assert not item['parser_diagnostics'] and item['diagnostics']
    for identity, expected in (('nonblank-zero-width', ['\u200b']), ('padding-order', [' close ', 'close'])):
        item = cases[identity]
        assert item['native']['valid'] and not item['parser_diagnostics'] and not item['diagnostics']
        assert item['models'][0]['sfd']['manifest']['required_columns'] == expected
