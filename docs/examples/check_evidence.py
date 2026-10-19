from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def region(path: Path, name: str) -> str:
    source = path.read_text()
    start, end = f'# region {name}\n', f'# endregion {name}'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f'Region is missing or duplicated: {path}:{name}')
    return source.split(start, 1)[1].split(end, 1)[0].rstrip('\n') + '\n'


def bindings(root: Path, document: Path) -> list[dict[str, str]]:
    text = document.read_text()
    pattern = r'<!-- example: (\S+) source=(\S+) region=(\S+) scenario=(\S+) interface=(\S+) role=executed -->\n```python\n(.*?)\n```'
    found = []
    for match in re.finditer(pattern, text, re.S):
        identity, source, name, scenario, interface, excerpt = match.groups()
        requested = root / source
        path = requested.resolve()
        if not path.is_relative_to(root.resolve()) or any(part.is_symlink() for part in (requested, *requested.parents)):
            raise ValueError('Example source escapes the repository')
        canonical = region(path, name)
        if canonical != excerpt + '\n':
            raise ValueError(f'Documentation differs from executable source: {identity}')
        if interface not in ('python', 'compiled') or f'/{interface}-research/' not in source:
            raise ValueError(f'Wrong research interface: {identity}')
        found.append({'id': identity, 'path': source, 'region': name, 'scenario': scenario,
                      'interface': interface, 'sha256': digest(canonical.encode())})
    if len(found) != text.count('role=executed') or len({item['id'] for item in found}) != len(found):
        raise ValueError('Missing or duplicate executable bindings')
    return found


def output_bindings(root: Path, document: Path) -> list[dict[str, str]]:
    text = document.read_text()
    pattern = r'<!-- example-output: (\S+) source=(\S+) scenario=(\S+) interface=(\S+) role=observed -->\n```json\n(.*?)\n```'
    found = []
    for match in re.finditer(pattern, text, re.S):
        identity, source, scenario, interface, excerpt = match.groups()
        path = (root / source).resolve()
        if not path.is_relative_to(root.resolve()) or path.read_text() != excerpt + '\n':
            raise ValueError('Observed output differs from its retained artifact')
        found.append({'id': identity, 'path': source, 'scenario': scenario, 'interface': interface, 'sha256': digest(path.read_bytes())})
    if len(found) != text.count('role=observed') or len({item['id'] for item in found}) != len(found):
        raise ValueError('Missing or duplicate observed output bindings')
    return found


def compare_output(current, retained, tolerances, at='') -> None:
    if at in tolerances:
        limits = tolerances[at]
        if type(current) not in (int, float) or type(retained) not in (int, float) or not math.isfinite(current) or not math.isfinite(retained) or not math.isclose(current, retained, abs_tol=limits['absolute_tolerance'], rel_tol=limits['relative_tolerance']):
            raise ValueError('Observed metric exceeds its declared tolerance: ' + at)
    elif isinstance(retained, dict):
        if not isinstance(current, dict) or set(current) != set(retained):
            raise ValueError('Observed output fields differ')
        for key, value in retained.items():
            compare_output(current[key], value, tolerances, key if not at else at + '.' + key)
    elif isinstance(retained, list):
        if not isinstance(current, list) or len(current) != len(retained):
            raise ValueError('Observed output population differs')
        for index, value in enumerate(retained):
            compare_output(current[index], value, tolerances, at + '.' + str(index))
    elif type(current) is not type(retained) or current != retained:
        raise ValueError('Observed exact value differs: ' + at)


def validate_shape(value, schema, at='$') -> None:
    if 'oneOf' in schema:
        failures = []
        for choice in schema['oneOf']:
            try:
                validate_shape(value, choice, at)
            except ValueError as error:
                failures.append(str(error))
        if len(schema['oneOf']) - len(failures) != 1:
            raise ValueError(f'{at}: expected exactly one schema choice; {failures}')
        return
    if 'const' in schema and (value != schema['const'] or (type(value) is bool) != (type(schema['const']) is bool)):
        raise ValueError(f'{at}: unexpected constant')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{at}: unexpected enum')
    kinds = schema.get('type', [])
    if isinstance(kinds, str):
        kinds = [kinds]
    matches = {'null': value is None, 'boolean': type(value) is bool,
               'integer': type(value) is int, 'number': type(value) in (int, float),
               'string': isinstance(value, str), 'array': isinstance(value, list),
               'object': isinstance(value, dict)}
    if kinds and not any(matches[kind] for kind in kinds):
        raise ValueError(f'{at}: wrong value type')
    if isinstance(value, dict):
        properties = schema.get('properties', {})
        if set(schema.get('required', [])) - value.keys():
            raise ValueError(f'{at}: missing required identity')
        extra = value.keys() - properties.keys()
        if schema.get('additionalProperties') is False and extra:
            raise ValueError(f'{at}: unknown identity fields')
        for key, item in value.items():
            child = properties.get(key, schema.get('additionalProperties'))
            if isinstance(child, dict):
                validate_shape(item, child, f'{at}.{key}')
    if isinstance(value, list):
        if len(value) < schema.get('minItems', 0):
            raise ValueError(f'{at}: missing execution records')
        if 'items' in schema:
            for index, item in enumerate(value):
                validate_shape(item, schema['items'], f'{at}[{index}]')
    if isinstance(value, str):
        if len(value) < schema.get('minLength', 0) or ('pattern' in schema and not re.fullmatch(schema['pattern'], value)):
            raise ValueError(f'{at}: invalid identity')
    if type(value) in (int, float) and (not math.isfinite(value) or value < schema.get('minimum', value)):
        raise ValueError(f'{at}: invalid numeric bound')


def validate_scenarios(records, catalogue) -> None:
    expected = {(interface, scenario) for scenario, interfaces in catalogue['required_scenarios'].items() for interface in interfaces}
    actual = [(item['interface'], item['id']) for item in records]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Research scenario catalogue is incomplete or duplicated')
    if any(item['status'] != 'passed' or not item['tests'] or not item['assertions'] for item in records):
        raise ValueError('Research scenario lacks successful semantic proof')


def validate_evidence(record, root: Path, *, head: str, tree: str, version: str) -> None:
    schema = json.loads((root / 'docs/examples/evidence.schema.json').read_text())
    validate_shape(record, schema)
    catalogue = root / 'docs/examples/catalogue.json'
    if record['validation']['catalogue_sha256'] != digest(catalogue.read_bytes()):
        raise ValueError('Wrong research catalogue identity')
    validate_scenarios(record['validation']['scenarios'], json.loads(catalogue.read_text()))
    retained = json.loads((root / 'docs/examples/retained/selection.json').read_text())
    displays = {item['id']: item for item in record['displays']}
    if displays != {item['id']: item for item in retained['displays']}:
        raise ValueError('Retained display identity was replaced by a current run')
    for binding in output_bindings(root, root / 'docs/Research-Book.md'):
        display = displays[binding['id']]
        if (display['path'], display['sha256']) != (binding['path'], binding['sha256']) or display['environment_sha256'] != digest(json.dumps(retained['environment'], sort_keys=True).encode()):
            raise ValueError('Observed output identity differs')
        current = next(item for item in record['validation']['scenarios'] if item['interface'] == binding['interface'] and item['id'] == binding['scenario'])
        source = next(item for item in record['sources'] if f"/{binding['interface']}-research/" in item['path'] and item['region'] == 'experiment')
        if display['source_sha256'] != source['sha256'] or display['retained_commit'] != retained['revision']['tested_commit']:
            raise ValueError('Retained source or run identity differs')
        compare_output(json.loads((root / current['output_path']).read_text()), json.loads((root / display['path']).read_text()), display['metrics'])
    if record['revision']['head'] != head or record['revision']['tree'] != tree:
        raise ValueError('Wrong execution revision')
    if record['revision']['dirty'] or record['package']['version'] != version:
        raise ValueError('Dirty or wrong-package proof')
    if record['mode'] == 'production' and record['publication'] is None:
        raise ValueError('Missing released-package proof')
    if record['mode'] == 'candidate' and (record['publication'] is not None or any(source['status'] == 'merged' and not source['merged_pr'] for source in record['sources'])):
        raise ValueError('Candidate cannot manufacture merged/released proof')
    from provenance import validate_provenance
    provenance = json.loads((root / '.docs-evidence/source-provenance.json').read_text())
    validate_provenance(root, provenance)
    files = {item['path']: item for item in provenance['files']}
    for source in record['sources']:
        expected = files[source['path']]
        if any(source[key] != expected[key] for key in ('commit', 'status', 'merged_pr')):
            raise ValueError('Source provenance was substituted')
        if source['region'] == '__file__' and source['sha256'] != expected['sha256']:
            raise ValueError('Source file identity differs')
    for binding in bindings(root, root / 'docs/Research-Book.md'):
        source = next((item for item in record['sources'] if item['path'] == binding['path'] and item['region'] == binding['region']), None)
        if source is None or source['sha256'] != binding['sha256']:
            raise ValueError('Stale executable binding')
        if not any(item['id'] == binding['scenario'] and item['interface'] == binding['interface'] and item['status'] == 'passed' for item in record['validation']['scenarios']):
            raise ValueError('Unexecuted example binding')

    tested = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if record['revision']['tested_commit'] != tested:
        raise ValueError('Tested-merge identity was substituted')
    from build_bundle import INPUTS
    expected_inputs = {'tests/fixtures/' + name: sha for name, sha in INPUTS.items()}
    if {item['path']: item['sha256'] for item in record['inputs']} != expected_inputs:
        raise ValueError('Recorded inputs are missing or changed')
    def verified_file(relative, expected):
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError('Retained artifact is missing or changed')
    for item in record['inputs']:
        verified_file(item['path'], item['sha256'])
    verified_file(record['bundle']['path'], record['bundle']['sha256'])
    wheels = [path for path in (root / '.docs-evidence').glob(f'vaquum_limen-{version}-*.whl') if digest(path.read_bytes()) == record['package']['wheel_sha256']]
    if len(wheels) != 1:
        raise ValueError('Installed wheel artifact does not match')
    for item in [*record['displays'], *record['validation']['scenarios']]:
        verified_file(item.get('path', item.get('output_path')), item.get('sha256', item.get('output_sha256')))
    for source in record['sources']:
        if source['status'] == 'candidate' and source['merged_pr'] is not None:
            raise ValueError('Candidate source cannot claim a merged PR')
        if record['mode'] == 'candidate':
            base = subprocess.check_output(['git', 'merge-base', 'HEAD', 'origin/main'], cwd=root, text=True).strip()
            changed = subprocess.run(['git', 'diff', '--quiet', base, 'HEAD', '--', source['path']], cwd=root, check=False).returncode
            if changed and source['status'] != 'candidate':
                raise ValueError('Unmerged source presented as merged')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Verify execution, source, input and retained-output identities.')
    for name in ('evidence', 'head', 'tree', 'version'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    validate_evidence(json.loads(Path(args.evidence).read_text()), root, head=args.head, tree=args.tree, version=args.version)
    print('Research evidence matches source, package, inputs and retained observations.')
