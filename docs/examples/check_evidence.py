from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import json
import math
import re
import subprocess
import tarfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

from catalogue_surfaces import validate_surface_catalogue
from catalogue_tests import validate_test_reference

EXECUTED_FENCE = r'<!-- example: (\S+) source=(\S+) region=(\S+) scenario=(\S+) interface=(\S+) role=executed -->\n```(?:python|bash|yaml)\n(.*?)\n```'
OBSERVED_FENCE = r'<!-- example-output: (\S+) source=(\S+) scenario=(\S+) interface=(\S+) role=observed -->\n```json\n(.*?)\n```'
REFERENCE_FENCE = r'<!-- reference: (\S+) source=(\S+) region=(\S+) role=(signature|definition|layout) -->\n```(?:python|python-fragment|text)\n(.*?)\n```'
RESULT_MARKER = r'<!-- result: (\S+) interface=(python|compiled) scenario=(\S+) fields=([\w.\[\],]+) -->'
COMPILED_DRIVERS = {
    'selection': ('selection',), 'search': ('source_formats', 'preparation', 'helpers', 'targets', 'reducers', 'search', 'parallel', 'recording', 'models'),
    'recovery': ('recovery',), 'walk_forward': ('walk_forward',), 'calibration': ('calibration',),
    'execution': ('execution',), 'bars': ('bars',), 'allocation': ('allocation',),
    'holdout': ('holdout',), 'command_line': ('command_line',),
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@lru_cache(maxsize=4096)
def package_reference(source: str, name: str) -> str:
    kind, symbol = name.split(':', 1)
    parts = symbol.split('.')
    if any(not part.isidentifier() or part.startswith('_') and part not in ('__init__', '__call__') for part in parts):
        raise ValueError('Invalid reference symbol: ' + symbol)
    scope = ast.parse(source).body
    selected = None
    for index, part in enumerate(parts):
        matches: list[ast.stmt] = [node for node in scope if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == part]
        if kind == 'definition' and len(parts) == 1:
            matches += [node for node in scope if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == part for target in node.targets)]
            matches += [node for node in scope if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == part]
        if len(matches) > 1 and kind == 'signature' and index == len(parts) - 1 and all(
                isinstance(node, ast.FunctionDef) and any(
                    isinstance(decorator, ast.Name) and decorator.id == 'property' or
                    isinstance(decorator, ast.Attribute) and decorator.attr == 'setter'
                    for decorator in node.decorator_list) for node in matches):
            members = [node for node in matches if isinstance(node, ast.FunctionDef)]
            for member in members:
                member.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
            return ast.unparse(ast.fix_missing_locations(ast.Module(body=[*members], type_ignores=[]))) + '\n'
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous reference symbol: ' + symbol)
        selected = matches[0]
        if index < len(parts) - 1:
            if not isinstance(selected, ast.ClassDef):
                raise ValueError('Reference members require a class: ' + symbol)
            scope = selected.body
    if selected is None:
        raise ValueError('Missing reference symbol: ' + symbol)
    if kind == 'signature':
        if isinstance(selected, ast.ClassDef):
            selected.body = [node for node in selected.body if isinstance(node, ast.AnnAssign)] or [ast.Expr(value=ast.Constant(value=Ellipsis))]
        elif isinstance(selected, (ast.FunctionDef, ast.AsyncFunctionDef)):
            selected.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        else:
            raise ValueError('A signature requires a function or class')
    elif not isinstance(selected, (ast.Assign, ast.AnnAssign)):
        raise ValueError('A definition requires a module constant')
    return ast.unparse(ast.fix_missing_locations(selected)) + '\n'


def region(path: Path, name: str) -> str:
    if name == '__file__':
        return path.read_text()
    if name.startswith('layout:'):
        if name != 'layout:package' or path.name != 'README.md':
            raise ValueError('A package layout must name its package README')
        directory = path.parent.resolve()
        tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', '.'], cwd=directory, text=True)
        names = set()
        for filename in tracked.split('\0'):
            if filename:
                parts = Path(filename).parts
                names.add(parts[0] + ('/' if len(parts) > 1 else ''))
        ordered = sorted(names)
        if not ordered:
            raise ValueError('Package layout has no tracked members')
        return directory.name + '/\n' + ''.join(('└── ' if index == len(ordered) - 1 else '├── ') + member + '\n' for index, member in enumerate(ordered))
    if name.startswith(('signature:', 'definition:')):
        if path.suffix != '.py':
            raise ValueError('Invalid reference symbol: ' + name.split(':', 1)[1])
        return package_reference(path.read_text(), name)
    if path.suffix == '.json':
        command = json.loads(path.read_text())[name]
        if not isinstance(command, str) or '\n' in command or not command.strip():
            raise ValueError('Invalid canonical command: ' + name)
        return command + '\n'
    source = path.read_text()
    start, end = f'# region {name}\n', f'# endregion {name}'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f'Region is missing or duplicated: {path}:{name}')
    return source.split(start, 1)[1].split(end, 1)[0].rstrip('\n') + '\n'


def bindings(root: Path, document: Path) -> list[dict[str, str]]:
    text = document.read_text()
    found = []
    for match in re.finditer(EXECUTED_FENCE, text, re.S):
        identity, source, name, scenario, interface, excerpt = match.groups()
        requested = root / source
        path = requested.resolve()
        if not path.is_relative_to(root.resolve()) or any(part.is_symlink() for part in (requested, *requested.parents)):
            raise ValueError('Example source escapes the repository')
        if interface == 'compiled' and path.suffix not in ('.yaml', '.sh', '.json'):
            raise ValueError(f'Wrong research interface: Compiled Research requires YAML and CLI source: {identity}')
        if interface == 'compiled' and '/compiled-research/' in source:
            driver = root / 'docs/examples/compiled-research/run.sh'
            regions = COMPILED_DRIVERS.get(scenario, ())
            if not regions:
                raise ValueError('Unknown Compiled consuming scenario: ' + scenario)
            commands = '\n'.join(region(driver, part) for part in regions)
            relative = str(path.relative_to(root / 'docs/examples'))
            if path == driver:
                consumed = name in regions or name == '__file__'
            elif scenario == 'holdout' and '/final' in relative:
                consumed = relative.rsplit('/', 1)[0] + '/run.sh' in commands
            elif scenario == 'command_line':
                project_commands = (root / 'docs/examples/compiled-research/project.sh').read_text()
                consumed = relative == 'compiled-research/project.sh' or relative in project_commands
            else:
                consumed = relative in commands
            if not consumed:
                raise ValueError('Example is not consumed by its Compiled scenario: ' + identity)
        canonical = region(path, name)
        if canonical != excerpt + '\n':
            raise ValueError(f'Documentation differs from executable source: {identity}')
        shared = json.loads((root / 'docs/examples/catalogue.json').read_text())['shared_sources']
        if interface not in ('python', 'compiled') or (f'/{interface}-research/' not in source and interface not in shared.get(source, [])):
            raise ValueError(f'Wrong research interface: {identity}')
        found.append({'id': identity, 'path': source, 'region': name, 'scenario': scenario,
                      'interface': interface, 'sha256': digest(canonical.encode())})
    if len(found) != text.count('role=executed') or len({item['id'] for item in found}) != len(found):
        raise ValueError('Missing or duplicate executable bindings')
    return found


def maintained_documents(root: Path) -> list[Path]:
    return sorted({root / 'README.md', *(root / 'docs').rglob('*.md'), *(root / 'limen').rglob('README.md')})


def result_bindings(root: Path, document: Path, records=None):
    text = document.read_text()
    catalogue = json.loads((root / 'docs/examples/catalogue.json').read_text())
    found = []
    for match in re.finditer(RESULT_MARKER, text):
        identity, interface, scenario, fields = match.groups()
        key = interface + '/' + scenario
        if key not in catalogue['scenario_details'] or any(not field or not all((part[:-2] if part.endswith('[]') else part).isidentifier() for part in field.split('.')) for field in fields.split(',')):
            raise ValueError('Unknown current result or field: ' + identity)
        if records is not None:
            record = next(item for item in records if item['interface'] == interface and item['id'] == scenario)
            path = (root / record['output_path']).resolve()
            if not path.is_relative_to((root / '.docs-evidence').resolve()) or digest(path.read_bytes()) != record['output_sha256']:
                raise ValueError('Current result identity differs: ' + identity)
            value = json.loads(path.read_text())
            def select(selected, parts, field):
                if not parts:
                    return selected
                part = parts[0]
                name = part[:-2] if part.endswith('[]') else part
                if not isinstance(selected, dict) or name not in selected:
                    raise ValueError('Missing current result field: ' + field)
                selected = selected[name]
                if part.endswith('[]'):
                    if not isinstance(selected, list):
                        raise ValueError('Current result field is not a row collection: ' + field)
                    return [select(item, parts[1:], field) for item in selected]
                return select(selected, parts[1:], field)
            for field in fields.split(','):
                select(value, field.split('.'), field)
        found.append({'id': identity, 'interface': interface, 'scenario': scenario, 'fields': fields.split(',')})
    if len(found) != text.count('<!-- result:') or len({item['id'] for item in found}) != len(found):
        raise ValueError('Malformed or duplicate current result')
    return found


def validate_page_catalogue(root: Path, catalogue):
    pages = catalogue['candidate_pages']
    expected = {str(path.relative_to(root)) for path in maintained_documents(root)}
    if len(pages) != len(set(pages)) or set(pages) != expected:
        raise ValueError('Bound page catalogue must cover every maintained document exactly once')
    contexts = catalogue['page_contexts']
    if set(contexts) != expected:
        raise ValueError('Bound page contexts differ from the maintained documents')
    for page, scenarios in contexts.items():
        if not scenarios or len(set(scenarios)) != len(scenarios) or any(name not in catalogue['scenario_details'] for name in scenarios):
            raise ValueError('Page has no complete scenario context: ' + page)
    return contexts


def all_bindings(root: Path) -> list[dict[str, str]]:
    found = [item for document in maintained_documents(root) for item in bindings(root, document)]
    if len({item['id'] for item in found}) != len(found):
        raise ValueError('Duplicate executable binding across maintained documents')
    return found


def output_bindings(root: Path, document: Path) -> list[dict[str, str]]:
    text = document.read_text()
    found = []
    for match in re.finditer(OBSERVED_FENCE, text, re.S):
        identity, source, scenario, interface, excerpt = match.groups()
        path = (root / source).resolve()
        if not path.is_relative_to(root.resolve()) or path.read_text() != excerpt + '\n':
            raise ValueError('Observed output differs from its retained artifact')
        found.append({'id': identity, 'path': source, 'scenario': scenario, 'interface': interface, 'sha256': digest(path.read_bytes())})
    if len(found) != text.count('role=observed') or len({item['id'] for item in found}) != len(found):
        raise ValueError('Missing or duplicate observed output bindings')
    return found


def reference_bindings(root: Path, document: Path) -> list[dict[str, str]]:
    text = document.read_text()
    found = []
    for match in re.finditer(REFERENCE_FENCE, text, re.S):
        identity, source, name, role, excerpt = match.groups()
        requested = root / source
        path = requested.resolve()
        if not path.is_relative_to((root / 'limen').resolve()) or any(part.is_symlink() for part in (requested, *requested.parents)):
            raise ValueError('Reference source escapes the package')
        if not name.startswith(role + ':') or region(path, name) != excerpt + '\n':
            raise ValueError('Reference differs from current package source: ' + identity)
        if role == 'layout' and not match.group(0).split('\n', 1)[1].startswith('```text\n'):
            raise ValueError('A package layout requires a text fence')
        found.append({'id': identity, 'path': source, 'region': name, 'role': role, 'sha256': digest((excerpt + '\n').encode())})
    if len(found) != len(re.findall(r'<!-- reference:', text)) or len({item['id'] for item in found}) != len(found):
        raise ValueError('Missing or duplicate reference bindings')
    return found


def all_reference_bindings(root: Path) -> list[dict[str, str]]:
    found = [item for document in maintained_documents(root) for item in reference_bindings(root, document)]
    identities = [item['id'] for item in [*all_bindings(root), *found]]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate source binding across maintained documents')
    return found


@lru_cache(maxsize=8)
def markdown_documents(site: Path, texts: tuple[str, ...]) -> dict:
    parser = '''import MarkdownIt from 'markdown-it';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const markdown = new MarkdownIt({html:true});
console.log(JSON.stringify(JSON.parse(input).map(text => {
  const tokens = markdown.parse(text, {});
  return {
    fences: tokens.filter(token => ['fence', 'code_block'].includes(token.type)).map(token => token.map[0]),
    links: tokens.flatMap(token => token.children || []).filter(token => token.type === 'link_open').map(token => token.attrGet('href'))
  };
})));'''
    parsed = subprocess.run(['node', '--input-type=module', '-e', parser], cwd=site,
                            input=json.dumps(texts), text=True, capture_output=True, check=True)
    return {text: (tuple(record['fences']), tuple(record['links']))
            for text, record in zip(texts, json.loads(parsed.stdout), strict=True)}


@lru_cache(maxsize=256)
def markdown_structure(site: Path, text: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    documents = tuple(document.read_text() for document in maintained_documents(site.parent))
    return markdown_documents(site, documents if text in documents else (text,))[text]


def validate_fences(root: Path, document: Path, *, interfaces: tuple[str, ...] = ()) -> None:
    text = document.read_text()
    bindings(root, document)
    output_bindings(root, document)
    reference_bindings(root, document)
    result_bindings(root, document)
    fence_lines, links = markdown_structure(root / 'docs-site', text)
    expected = sorted(text[:match.start()].count('\n') + 1 for pattern in (EXECUTED_FENCE, OBSERVED_FENCE, REFERENCE_FENCE)
                      for match in re.finditer(pattern, text, re.S))
    if fence_lines != tuple(expected):
        raise ValueError('Unbound code fence or indented block in candidate documentation: ' + str(document))
    repository = json.loads((root / 'docs-site/reader/content.json').read_text())['product']['repository']
    source_prefix = repository + '/blob/main/'
    targets = []
    for href in links:
        if href.startswith(source_prefix):
            targets.append((root / unquote(urlsplit(href[len(source_prefix):]).path)).resolve())
        elif not urlsplit(href).scheme and not href.startswith(('/', '#')):
            targets.append((document.parent / unquote(urlsplit(href).path)).resolve())
    for interface in interfaces:
        if interface not in ('python', 'compiled'):
            raise ValueError('Unknown project interface: ' + interface)
        project = (root / f'docs/examples/{interface}-research/README.md').resolve()
        if document.resolve() != project and project not in targets:
            raise ValueError('Missing visible canonical project connection: ' + interface)


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
    if isinstance(value, (int, float)) and not isinstance(value, bool) and (not math.isfinite(value) or value < schema.get('minimum', value)):
        raise ValueError(f'{at}: invalid numeric bound')


def validate_retained_source(root, retained, source, display) -> None:
    commit = retained['revision']['tested_commit']
    if not re.fullmatch(r'[0-9a-f]{40}', commit) or display['retained_commit'] != commit:
        raise ValueError('Retained run identity differs')
    historical = next(item for item in retained['sources'] if item['path'] == source['path'] and item['region'] == source['region'])
    text = subprocess.check_output(['git', 'show', f"{commit}:{source['path']}"], cwd=root, text=True)
    start, end = '# region ' + source['region'] + '\n', '# endregion ' + source['region']
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('Retained source region is unavailable')
    observed = text.split(start, 1)[1].split(end, 1)[0].rstrip('\n') + '\n'
    if display['source_sha256'] != historical['sha256'] or historical['sha256'] != digest(observed.encode()):
        raise ValueError('Retained source identity differs from its immutable execution revision')


def validate_scenarios(records, catalogue, root: Path) -> None:
    expected = {(interface, scenario) for scenario, interfaces in catalogue['required_scenarios'].items() for interface in interfaces}
    actual = [(item['interface'], item['id']) for item in records]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Research scenario catalogue is incomplete or duplicated')
    if any(item['status'] != 'passed' or not item['tests'] or not item['assertions'] for item in records):
        raise ValueError('Research scenario lacks successful semantic proof')
    for item in records:
        entry = catalogue['scenario_details'][item['interface'] + '/' + item['id']]
        required = entry['test'] + '[' + item['interface'] + '-' + item['id'] + ']'
        if required not in item['tests'] or entry['assertions'] not in item['assertions']:
            raise ValueError('Research scenario lacks its required semantic test or assertion')
        if not set(entry.get('extra_tests', [])) <= set(item['tests']):
            raise ValueError('Research scenario lacks its required integrity tests')
        for reference in item['tests']:
            validate_test_reference(root, reference.split('[', 1)[0])


def validate_model_catalogue(root, catalogue) -> None:
    module = ast.parse((root / 'limen/sfd/reference_architecture/__init__.py').read_text())
    exports = next(ast.literal_eval(node.value) for node in module.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '__all__' for target in node.targets))
    if set(catalogue['model_families']) != {name for name in exports if name[0].islower()}:
        raise ValueError('Public model catalogue is incomplete')
    for family, entry in catalogue['model_families'].items():
        if 'retained_execution' in entry:
            from optional_models import verify
            if 'limen.sfd.reference_architecture.' + family not in (root / entry['configuration']).read_text():
                raise ValueError('Model definition differs from the public catalogue')
            verify(root, json.loads((root / entry['retained_execution']).read_text()))
            validate_test_reference(root, entry['test'])
            continue
        if 'unproved' in entry:
            if entry['unproved'] not in catalogue['unproved'].values():
                raise ValueError('Unproved model has no tracked execution boundary')
            continue
        if entry['interface'] not in catalogue['required_scenarios'][entry['scenario']]:
            raise ValueError('Model has no required interface scenario')
        source = root / entry['configuration']
        if not source.resolve().is_relative_to((root / 'docs/examples/compiled-research').resolve()) or not source.is_file():
            raise ValueError('Missing canonical model definition')
        if 'limen.sfd.reference_architecture.' + family not in source.read_text():
            raise ValueError('Model definition differs from the public catalogue')


def validate_cli_catalogue(root, catalogue) -> None:
    from catalogue_admission import validate_cli_contracts
    validate_cli_contracts(root)
    from limen.cli.main import cli
    commands = set(cli.commands)
    if set(catalogue['cli_commands']) != commands:
        raise ValueError('Public CLI catalogue is incomplete')
    for entry in catalogue['cli_commands'].values():
        if entry['interface'] != 'compiled' or entry['interface'] not in catalogue['required_scenarios'][entry['scenario']]:
            raise ValueError('CLI command has no required compiled scenario')


def validate_source_status(root, sources, candidate_base):
    changed = set(subprocess.check_output(['git', 'diff', '--no-renames', '--name-only', '-z', candidate_base, 'HEAD'], cwd=root).decode().split('\0')) - {''} if candidate_base is not None else set()
    for source in sources:
        if source['status'] == 'candidate' and source['merged_pr'] is not None:
            raise ValueError('Candidate source cannot claim a merged PR')
        if source['path'] in changed and source['status'] != 'candidate':
            raise ValueError('Unmerged source presented as merged')


def validate_evidence(record, root: Path, *, head: str, tree: str, version: str) -> None:
    schema = json.loads((root / 'docs/examples/evidence.schema.json').read_text())
    validate_shape(record, schema)
    catalogue = root / 'docs/examples/catalogue.json'
    if record['validation']['catalogue_sha256'] != digest(catalogue.read_bytes()):
        raise ValueError('Wrong research catalogue identity')
    catalogue_data = json.loads(catalogue.read_text())
    validate_scenarios(record['validation']['scenarios'], catalogue_data, root)
    validate_model_catalogue(root, catalogue_data)
    validate_cli_catalogue(root, catalogue_data)
    from schema_contracts import validate as validate_yaml_contracts
    validate_yaml_contracts(root)
    validate_surface_catalogue(root, catalogue_data)
    from catalogue_admission import validate_admission
    validate_admission(root, json.loads((root / 'docs/examples/api-contracts.json').read_text()))
    contexts = validate_page_catalogue(root, catalogue_data)
    for page, scenarios in contexts.items():
        interfaces = tuple(sorted({catalogue_data['scenario_details'][name]['interface'] for name in scenarios}))
        validate_fences(root, root / page, interfaces=interfaces)
    results = [item for document in maintained_documents(root) for item in result_bindings(root, document, record['validation']['scenarios'])]
    if len({item['id'] for item in results}) != len(results):
        raise ValueError('Duplicate current result across maintained documents')
    retained = json.loads((root / 'docs/examples/retained/selection.json').read_text())
    displays = {item['id']: item for item in record['displays']}
    if displays != {item['id']: item for item in retained['displays']}:
        raise ValueError('Retained display identity was replaced by a current run')
    for binding in [item for document in maintained_documents(root) for item in output_bindings(root, document)]:
        display = displays[binding['id']]
        if (display['path'], display['sha256']) != (binding['path'], binding['sha256']) or display['environment_sha256'] != digest(json.dumps(retained['environment'], sort_keys=True).encode()):
            raise ValueError('Observed output identity differs')
        current = next(item for item in record['validation']['scenarios'] if item['interface'] == binding['interface'] and item['id'] == binding['scenario'])
        source = next(item for item in retained['sources'] if f"/{binding['interface']}-research/" in item['path'] and item['sha256'] == display['source_sha256'])
        validate_retained_source(root, retained, source, display)
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
    for binding in all_bindings(root):
        source = next((item for item in record['sources'] if item['path'] == binding['path'] and item['region'] == binding['region']), None)
        if source is None or source['sha256'] != binding['sha256']:
            raise ValueError('Stale executable binding')
        if not any(item['id'] == binding['scenario'] and item['interface'] == binding['interface'] and item['status'] == 'passed' for item in record['validation']['scenarios']):
            raise ValueError('Unexecuted example binding')
    for binding in all_reference_bindings(root):
        source = next((item for item in record['sources'] if item['path'] == binding['path'] and item['region'] == binding['region']), None)
        if source is None or source['sha256'] != binding['sha256']:
            raise ValueError('Stale package reference binding')

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
    locks = [root / 'requirements/ci/research-env.txt', root / 'docs/examples/requirements-research-macos.txt']
    if record['environment']['lock_sha256'] not in {digest(path.read_bytes()) for path in locks}:
        raise ValueError('Research dependency lock is unrecognized')
    verified_file('docs/examples/requirements-build.txt', record['environment']['build_lock_sha256'])
    verified_file('.docs-evidence/environment-wheels.json', record['environment']['wheelhouse_manifest_sha256'])
    verified_file(record['bundle']['path'], record['bundle']['sha256'])
    verified_file(record['native_outputs']['path'], record['native_outputs']['sha256'])
    with tarfile.open(root / record['native_outputs']['path']) as archive:
        members = archive.getmembers()
        if len({item.name for item in members}) != len(members) or any(not item.isfile() for item in members):
            raise ValueError('Native output archive contains substituted or duplicate members')
        contents = {}
        for member in members:
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('Missing native output member')
            with stream:
                contents[member.name] = stream.read()
        inventory = json.loads(contents['files.json'])
        if inventory != {name: digest(data) for name, data in contents.items() if name != 'files.json'}:
            raise ValueError('Native output archive inventory differs')
        installed = json.loads(contents['environment/installed.json'])
        if digest(json.dumps(installed['dependencies']).encode()) != record['environment']['dependencies_sha256']:
            raise ValueError('Retained installed dependency identity differs')
        for scenario in record['validation']['scenarios']:
            interface, name = scenario['interface'], scenario['id']
            directory = f'results/{interface}' if name == 'selection' else f'results/{interface}-all/{name}'
            if interface == 'compiled':
                index = json.loads(contents['results/compiled-artifacts.json'])[name]
                if not index or any(path not in contents or digest(contents[path]) != value for path, value in index.items()):
                    raise ValueError('A compiled scenario lost its actual CLI artifacts')
            if not any(path.startswith(directory + '/') for path in contents) and interface != 'compiled':
                raise ValueError('A required scenario has no retained native artifacts')
            summary = json.loads(contents[f'results/{interface}/summary.json'] if name == 'selection' else contents[f'results/{interface}-all/summary.json'])
            native_result = summary if name == 'selection' else summary['scenarios'][name]
            if native_result != json.loads((root / scenario['output_path']).read_text()):
                raise ValueError('Retained native summary differs from validated output')
    wheels = [path for path in (root / '.docs-evidence').glob(f'vaquum_limen-{version}-*.whl') if digest(path.read_bytes()) == record['package']['wheel_sha256']]
    if len(wheels) != 1:
        raise ValueError('Installed wheel artifact does not match')
    for item in [*record['displays'], *record['validation']['scenarios']]:
        verified_file(item.get('path', item.get('output_path')), item.get('sha256', item.get('output_sha256')))
    candidate_base = subprocess.check_output(['git', 'merge-base', 'HEAD', 'origin/main'], cwd=root, text=True).strip() if record['mode'] == 'candidate' else None
    validate_source_status(root, record['sources'], candidate_base)
    if record['mode'] == 'production':
        from publication import validate_production
        validate_production(root, record)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Verify execution, source, input and retained-output identities.')
    for name in ('evidence', 'head', 'tree', 'version'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    validate_evidence(json.loads(Path(args.evidence).read_text()), root, head=args.head, tree=args.tree, version=args.version)
    print('Research evidence matches source, package, inputs and retained observations.')
