from __future__ import annotations

import hashlib
import json
import re
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
        path = (root / source).resolve()
        if not path.is_relative_to(root.resolve()) or path.is_symlink():
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


def validate_shape(value, schema, at='$') -> None:
    if 'oneOf' in schema:
        successes = 0
        for choice in schema['oneOf']:
            try:
                validate_shape(value, choice, at)
                successes += 1
            except ValueError:
                continue
        if successes != 1:
            raise ValueError(f'{at}: expected exactly one schema choice')
        return
    if 'const' in schema and value != schema['const']:
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
    if type(value) in (int, float) and value < schema.get('minimum', value):
        raise ValueError(f'{at}: invalid numeric bound')


def validate_evidence(record, root: Path, *, head: str, tree: str, version: str) -> None:
    schema = json.loads((root / 'docs/examples/evidence.schema.json').read_text())
    validate_shape(record, schema)
    if record['revision']['head'] != head or record['revision']['tree'] != tree:
        raise ValueError('Wrong execution revision')
    if record['revision']['dirty'] or record['package']['version'] != version:
        raise ValueError('Dirty or wrong-package proof')
    if record['mode'] == 'production' and record['publication'] is None:
        raise ValueError('Missing released-package proof')
    if record['mode'] == 'candidate' and (record['publication'] is not None or any(source['status'] == 'merged' and not source['merged_pr'] for source in record['sources'])):
        raise ValueError('Candidate cannot manufacture merged/released proof')
    for binding in bindings(root, root / 'docs/Research-Book.md'):
        source = next((item for item in record['sources'] if item['path'] == binding['path'] and item['region'] == binding['region']), None)
        if source is None or source['sha256'] != binding['sha256']:
            raise ValueError('Stale executable binding')
        if not any(item['id'] == binding['scenario'] and item['interface'] == binding['interface'] and item['status'] == 'passed' for item in record['validation']['scenarios']):
            raise ValueError('Unexecuted example binding')
