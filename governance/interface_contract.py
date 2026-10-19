"""Limen-specific declaration admission; candidate Python is inspected, never run."""
from __future__ import annotations

import ast
import base64
import binascii
from collections.abc import Callable
from dataclasses import dataclass
import io
import json
from pathlib import PurePosixPath
import re
import subprocess
import sys
import tokenize
from typing import NoReturn, cast
from urllib.parse import urlencode

import yaml

__all__ = ['CandidateEvidence', 'interface_contract_failures']

_SHA = re.compile(r'[0-9a-f]{40}')
_HEADING = re.compile(r'^(#{1,6})[ \t]+(.+?)[ \t]*$')
_FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')
_TEST_PATH = re.compile(r'tests/(?:[A-Za-z0-9_-]+/)*test_[A-Za-z0-9_]+\.py')
_TEST_NAME = re.compile(r'test_[A-Za-z0-9_]+')
_TEST_CLASS = re.compile(r'Test[A-Za-z0-9_]*')
_PLACEHOLDER = re.compile(r'(?:_No response_|(?:TODO|TBD)\b.*|REPLACE:.*|N/?A|\.\.\.|<[^>]+>)', re.IGNORECASE | re.DOTALL)
_INTERFACES = frozenset({'python', 'compiled'})
_ROLES = frozenset({'public', 'prerequisite', 'maintenance'})
_BLOCKING_MARKS = frozenset({'skip', 'skipif', 'xfail'})


def _setup(message: str, cause: Exception | None = None) -> NoReturn:
    print(f'interface contract setup: {message}', file=sys.stderr)
    raise SystemExit(2) from cause


def _api(endpoint: str) -> object:
    try:
        result = subprocess.run(['gh', 'api', '--method', 'GET', endpoint],
                                check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        _setup(f'gh unavailable: {exc}', exc)
    if result.returncode:
        _setup(f'{endpoint}: {result.stderr.strip()}')
    try:
        payload: object = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        _setup(f'malformed JSON from {endpoint}', exc)
    return payload


def _object(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _setup(f'malformed {description}')
    return cast(dict[str, object], value)


def _blob_bytes(value: object, *, expected_sha: str | None = None) -> bytes:
    blob = _object(value, 'blob')
    content = blob.get('content')
    if blob.get('encoding') != 'base64' or not isinstance(content, str):
        _setup('blob requires base64 content')
    if expected_sha is not None and blob.get('sha') != expected_sha:
        _setup('blob identity does not match the candidate tree')
    try:
        data = base64.b64decode(''.join(content.splitlines()), validate=True)
    except (ValueError, binascii.Error) as exc:
        _setup('malformed base64 content', exc)
    size = blob.get('size')
    if isinstance(size, bool) or not isinstance(size, int) or size != len(data):
        _setup('blob size does not match content')
    return data


class CandidateEvidence:
    """Read candidate blobs and required contexts from their distinct authorities."""

    def __init__(self, repo: str, head_sha: str) -> None:
        if not isinstance(head_sha, str) or _SHA.fullmatch(head_sha) is None:
            raise ValueError('Interface Contract requires a full 40-character lowercase head SHA.')
        self.repo = repo
        self.head_sha = head_sha
        self._tree: dict[str, tuple[str, str, str]] | None = None
        self._sources: dict[str, str] = {}
        self._checks: set[str] | None = None

    def _entries(self) -> dict[str, tuple[str, str, str]]:
        if self._tree is None:
            payload = _object(_api(f'repos/{self.repo}/git/trees/{self.head_sha}?recursive=1'), 'Git tree')
            entries = payload.get('tree')
            if payload.get('truncated') is not False or not isinstance(entries, list):
                _setup('missing or truncated candidate Git tree')
            self._tree = {}
            for value in entries:
                entry = _object(value, 'Git tree entry')
                fields = [entry.get(key) for key in ('path', 'mode', 'type', 'sha')]
                if any(not isinstance(item, str) or not item for item in fields):
                    _setup('incomplete candidate Git tree entry')
                path, mode, kind, sha = cast(list[str], fields)
                if _SHA.fullmatch(sha) is None or path in self._tree:
                    _setup('invalid candidate Git tree identity')
                self._tree[path] = (mode, kind, sha)
        return self._tree

    def __call__(self, path: str) -> str:
        if path not in self._sources:
            entry = self._entries().get(path)
            if entry is None:
                raise ValueError(f'candidate proof file does not exist: {path}')
            mode, kind, sha = entry
            if kind != 'blob' or mode not in ('100644', '100755'):
                raise ValueError(f'candidate proof must be a regular file, not a symlink or submodule: {path}')
            data = _blob_bytes(_api(f'repos/{self.repo}/git/blobs/{sha}'), expected_sha=sha)
            try:
                encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
                self._sources[path] = data.decode(encoding)
            except (LookupError, SyntaxError, UnicodeDecodeError) as exc:
                raise ValueError(f'candidate proof has invalid Python source encoding: {path}') from exc
        return self._sources[path]

    @property
    def required_checks(self) -> set[str]:
        if self._checks is None:
            repository = _object(_api(f'repos/{self.repo}'), 'repository')
            branch = repository.get('default_branch')
            if not isinstance(branch, str) or not branch:
                _setup('default branch is unavailable')
            query = urlencode({'ref': branch})
            data = _blob_bytes(_api(f'repos/{self.repo}/contents/.github/rulesets/main.json?{query}'))
            try:
                payload: object = json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                _setup('malformed default-branch ruleset', exc)
            rules = _object(payload, 'default-branch ruleset').get('rules')
            if not isinstance(rules, list):
                _setup('default-branch rules are unavailable')
            required = [_object(rule, 'ruleset rule') for rule in rules
                        if _object(rule, 'ruleset rule').get('type') == 'required_status_checks']
            if len(required) != 1:
                _setup('expected one required-status-check rule')
            checks = _object(required[0].get('parameters'), 'required-check parameters').get('required_status_checks')
            if not isinstance(checks, list) or not checks:
                _setup('required contexts are unavailable')
            contexts = [_object(check, 'required check').get('context') for check in checks]
            if any(not isinstance(context, str) or not context for context in contexts):
                _setup('malformed required context')
            self._checks = set(cast(list[str], contexts))
        return set(self._checks)


class _ContractLoader(yaml.SafeLoader):
    def compose_node(self, parent: yaml.Node | None, index: object) -> yaml.Node:
        if self.check_event(yaml.AliasEvent):
            raise ValueError('YAML aliases are not permitted')
        return super().compose_node(parent, index)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[str, object]:
        result: dict[str, object] = {}
        for key_node, value_node in node.value:
            if key_node.tag == 'tag:yaml.org,2002:merge':
                raise ValueError('YAML merge keys are not permitted')
            key: object = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError('mapping keys must be strings')
            if key in result:
                raise ValueError(f'duplicate YAML key: {key}')
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _markdown(body: str) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]:
    headings: list[tuple[int, int, str]] = []
    fences: list[tuple[int, int, str]] = []
    opened: tuple[int, str, str] | None = None
    for index, line in enumerate(body.splitlines()):
        fence = _FENCE.fullmatch(line)
        if opened is not None:
            start, marker, language = opened
            if fence and fence[1][0] == marker[0] and len(fence[1]) >= len(marker) and not fence[2].strip():
                fences.append((start, index, language))
                opened = None
        elif fence:
            opened = (index, fence[1], fence[2].strip())
        else:
            heading = _HEADING.fullmatch(line)
            if heading:
                headings.append((index, len(heading[1]), heading[2]))
    if opened is not None:
        start, _, language = opened
        fences.append((start, -1, language))
    return headings, fences


@dataclass(frozen=True)
class _Contract:
    delivery: str
    interfaces: frozenset[str]
    scope: str
    proof: dict[str, tuple[str, ...]]


def _parse(body: str, *, required: bool) -> _Contract | None:
    normalized = body.replace('\r\n', '\n')
    headings, fences = _markdown(normalized)
    sections = [(index, level) for index, level, title in headings
                if title == 'Interface Contract' and level in (2, 3)]
    if not sections and not required:
        return None
    if len(sections) != 1:
        raise ValueError('provide exactly one ## or ### Interface Contract section')
    start, level = sections[0]
    end = next((index for index, depth, _ in headings if index > start and depth <= level), len(normalized.splitlines()))
    blocks = [(first, last, language) for first, last, language in fences if start < first < end]
    if len(blocks) != 1 or blocks[0][2] != 'yaml':
        raise ValueError('Interface Contract must contain exactly one yaml fence')
    first, last, _ = blocks[0]
    if last < first:
        raise ValueError('close the Interface Contract yaml fence')
    loader = _ContractLoader('\n'.join(normalized.splitlines()[first + 1:last]))
    try:
        raw: object = loader.get_single_data()
    except yaml.YAMLError as exc:
        raise ValueError('malformed Interface Contract YAML; quote scope values containing colons') from exc
    finally:
        loader.dispose()
    if not isinstance(raw, dict) or set(raw) != {'delivery', 'interfaces', 'scope', 'proof'}:
        raise ValueError('use exactly delivery, interfaces, scope and proof keys')
    delivery, interfaces, scope, proof = (raw[key] for key in ('delivery', 'interfaces', 'scope', 'proof'))
    if not isinstance(delivery, str) or delivery not in _ROLES:
        raise ValueError('delivery must be public, prerequisite or maintenance')
    if not isinstance(interfaces, list) or any(not isinstance(item, str) or item not in _INTERFACES for item in interfaces):
        raise ValueError('interfaces must list only python and compiled')
    if len(interfaces) != len(set(interfaces)):
        raise ValueError('interfaces must not contain duplicates')
    if delivery == 'public' and not interfaces:
        raise ValueError('public delivery requires at least one interface')
    if not isinstance(scope, str) or not scope.strip() or _PLACEHOLDER.fullmatch(scope.strip()):
        raise ValueError('scope must describe the capability and exclusions, not a placeholder')
    if not isinstance(proof, dict) or set(proof) != set(interfaces):
        raise ValueError('proof keys must exactly match interfaces; empty interfaces use proof: {}')
    references: dict[str, tuple[str, ...]] = {}
    for interface, refs in proof.items():
        if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
            raise ValueError(f'{interface} proof must be a nonempty list of references')
        references[interface] = tuple(sorted(refs))
    return _Contract(delivery, frozenset(interfaces), scope.strip(), references)


def _test_reference(reference: str) -> tuple[str, list[str]]:
    parts = reference.removeprefix('test:').split('::')
    path, symbols = parts[0], parts[1:]
    if _TEST_PATH.fullmatch(path) is None or str(PurePosixPath(path)) != path:
        raise ValueError('test proof must use a normal tests/.../test_*.py path')
    if len(symbols) not in (1, 2) or _TEST_NAME.fullmatch(symbols[-1]) is None:
        raise ValueError('test proof requires ::test_name or ::TestClass::test_name')
    if len(symbols) == 2 and _TEST_CLASS.fullmatch(symbols[0]) is None:
        raise ValueError('test proof classes must begin with Test')
    return path, symbols


def _marked(node: ast.AST) -> bool:
    return any((isinstance(part, ast.Attribute) and part.attr in _BLOCKING_MARKS)
               or (isinstance(part, ast.Name) and part.id in _BLOCKING_MARKS) for part in ast.walk(node))


def _unavailable(nodes: list[ast.stmt], decorators: list[ast.expr]) -> bool:
    if any(_marked(decorator) for decorator in decorators):
        return True
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if '__test__' in names and isinstance(node.value, ast.Constant) and node.value.value is False:
                return True
            if 'pytestmark' in names and node.value is not None and _marked(node.value):
                return True
    return False


def _test_exists(source: str, symbols: list[str]) -> None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise ValueError('candidate proof is not parseable Python') from exc
    nodes = tree.body
    if _unavailable(nodes, []):
        raise ValueError('candidate proof module disables collection or declares skip/xfail')
    if len(symbols) == 2:
        classes = [node for node in nodes if isinstance(node, ast.ClassDef) and node.name == symbols[0]]
        if len(classes) != 1:
            raise ValueError('candidate proof class does not exist directly in its module')
        cls = classes[0]
        if _unavailable(cls.body, cls.decorator_list) or any(isinstance(node, ast.FunctionDef) and node.name in ('__init__', '__new__') for node in cls.body):
            raise ValueError('candidate proof class is not normally collectible or declares skip/xfail')
        nodes = cls.body
    tests = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == symbols[-1]]
    if len(tests) != 1:
        raise ValueError('candidate proof test does not exist directly in its module/class')
    if _unavailable([], tests[0].decorator_list):
        raise ValueError('candidate proof test declares skip/xfail')


def _proof_failures(contract: _Contract, *, resolve: bool,
                    read_source: Callable[[str], str], required_checks: set[str]) -> list[str]:
    failures: list[str] = []
    trees: dict[str, str] = {}
    for interface, references in contract.proof.items():
        for reference in references:
            try:
                if reference.startswith('test:'):
                    path, symbols = _test_reference(reference)
                    if resolve:
                        if path not in trees:
                            trees[path] = read_source(path)
                        _test_exists(trees[path], symbols)
                elif reference.startswith('check:') and reference[6:].strip():
                    if contract.delivery == 'public':
                        raise ValueError('public capability proof requires named tests, not check references')
                    if resolve and reference[6:] not in required_checks:
                        raise ValueError('check proof is not a trusted default-branch required context')
                else:
                    raise ValueError('proof must use test:<path>::<symbol> or check:<required context>')
            except ValueError as exc:
                failures.append(f'{interface} proof {reference!r}: {exc}')
    return failures


def interface_contract_failures(pr_body: str, issue_body: str, *, parent_body: str | None,
                                closing_parent: bool, read_source: Callable[[str], str],
                                required_checks: set[str]) -> list[str]:
    """Reconcile authoritative declarations and verify their permitted proof references."""
    contracts: dict[str, _Contract | None] = {}
    failures: list[str] = []
    bodies = [('PR', pr_body, True), ('slice', issue_body, False)]
    if parent_body is not None:
        bodies.append(('parent', parent_body, True))
    for label, body, required in bodies:
        try:
            contracts[label] = _parse(body, required=required)
        except ValueError as exc:
            failures.append(f'{label} Interface Contract: {exc}')
    if failures:
        return failures
    child = contracts['PR']
    assert child is not None
    if contracts['slice'] is not None and child != contracts['slice']:
        failures.append('PR Interface Contract must agree with the slice declaration.')
    parent = contracts.get('parent')
    if parent is not None:
        if child.delivery == 'public' and parent.delivery != 'public':
            failures.append('A public child requires a public parent Interface Contract.')
        if child.delivery in ('public', 'prerequisite') and not child.interfaces <= parent.interfaces:
            failures.append('Child interface targets must be a subset of the parent targets.')
        failures.extend(f'parent Interface Contract: {message}' for message in _proof_failures(
            parent, resolve=closing_parent, read_source=read_source, required_checks=required_checks))
    failures.extend(f'PR Interface Contract: {message}' for message in _proof_failures(
        child, resolve=True, read_source=read_source, required_checks=required_checks))
    return failures
