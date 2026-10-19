from __future__ import annotations

import base64
import json

import pytest
import yaml

from governance import interface_contract as contract

SHA = 'a' * 40
BLOB = 'b' * 40
PATH = 'tests/test_method.py'
PYTHON = f'test:{PATH}::test_python_capability'
COMPILED = f'test:{PATH}::TestCompiled::test_capability'
CHECK = 'check:pr_checks_ruleset'
SOURCE = '''
raise RuntimeError('Candidate source must never execute in admission')

def test_python_capability():
    assert True

class TestCompiled:
    def test_capability(self):
        assert True
'''


def _body(delivery='public', interfaces=('python',), *, scope='A complete declared capability.', proof=None):
    if proof is None:
        proof = {name: [PYTHON if name == 'python' else COMPILED] for name in interfaces}
    value = {'delivery': delivery, 'interfaces': list(interfaces), 'scope': scope, 'proof': proof}
    return '## Interface Contract\n\n```yaml\n' + yaml.safe_dump(value) + '```\n'


def _source(path):
    if path != PATH:
        raise ValueError(f'candidate proof file does not exist: {path}')
    return SOURCE


def _admit(body, issue='', parent=None, *, closing=False, read_source=_source):
    return contract.interface_contract_failures(body, issue, parent_body=parent,
        closing_parent=closing, read_source=read_source, required_checks={'pr_checks_ruleset'})


@pytest.mark.parametrize(('delivery', 'interfaces', 'proof'), [
    ('public', ('python',), None),
    ('public', ('compiled',), None),
    ('public', ('python', 'compiled'), None),
    ('prerequisite', ('python',), {'python': [CHECK]}),
    ('prerequisite', (), {}),
    ('maintenance', ('compiled',), {'compiled': [CHECK]}),
    ('maintenance', (), {}),
])
def test_complete_and_staged_delivery(delivery, interfaces, proof):
    assert _admit(_body(delivery, interfaces, proof=proof)) == []


def test_contract_order_crlf_and_embedded_examples():
    body = _body(interfaces=('python', 'compiled'), proof={'python': [PYTHON, COMPILED], 'compiled': [COMPILED]})
    issue = _body(interfaces=('compiled', 'python'), proof={'compiled': [COMPILED], 'python': [COMPILED, PYTHON]})
    example = '## Signature Diff\n```text\n## Interface Contract\nexample only\n```\n'
    assert _admit(example + body.replace('\n', '\r\n'), issue.replace('## Interface', '### Interface')) == []


@pytest.mark.parametrize(('mutation', 'message'), [
    ('missing', 'exactly one'), ('duplicate', 'exactly one'),
    ('two_fences', 'exactly one yaml fence'), ('wrong_fence', 'exactly one yaml fence'),
    ('unclosed', 'close the Interface Contract yaml fence'),
    ('duplicate_key', 'duplicate YAML key'), ('duplicate_proof_key', 'duplicate YAML key'),
    ('alias', 'aliases'), ('merge', 'merge keys'), ('malformed', 'quote scope'),
])
def test_contract_sections_and_yaml_are_strict(mutation, message):
    body = _body()
    mutations = {
        'missing': '# Other\n_No response_\n',
        'duplicate': body + body,
        'two_fences': body + '```yaml\nother: data\n```\n',
        'wrong_fence': body.replace('```yaml', '```python'),
        'unclosed': body.removesuffix('```\n'),
        'duplicate_key': body.replace('delivery: public', 'delivery: public\ndelivery: maintenance'),
        'duplicate_proof_key': body.replace('proof:\n', 'proof:\n  python: []\n'),
        'alias': body.replace('interfaces:\n- python', 'interfaces: &targets [python]').replace('scope:', 'scope: *targets\nunused:'),
        'merge': body.replace('delivery: public', '<<: {delivery: public}\ndelivery: public'),
        'malformed': body.replace('scope: A complete declared capability.', 'scope: Preserve: both interfaces'),
    }
    failures = _admit(mutations[mutation])
    assert any(message in failure for failure in failures), failures


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('delivery', 'partial', 'delivery must'), ('delivery', True, 'delivery must'),
    ('interfaces', [True], 'interfaces must'), ('interfaces', 'python', 'interfaces must'),
    ('interfaces', ['python', 'python'], 'duplicates'), ('interfaces', [], 'public delivery requires'),
    ('scope', '', 'scope must'), ('scope', 'TODO describe the capability', 'scope must'),
    ('scope', 'REPLACE: complete capability and justified exclusions.', 'scope must'),
    ('scope', '_No response_', 'scope must'), ('scope', '<scope>', 'scope must'),
    ('proof', {}, 'proof keys'), ('proof', {'python': []}, 'nonempty list'),
    ('proof', {'python': [True]}, 'nonempty list'),
    ('proof', {'python': [CHECK]}, 'public capability proof requires named tests'),
    ('unknown', 'value', 'exactly delivery'),
])
def test_rejects_invalid_contract(field, value, message):
    data = {'delivery': 'public', 'interfaces': ['python'], 'scope': 'Complete capability.', 'proof': {'python': [PYTHON]}}
    data[field] = value
    body = '## Interface Contract\n```yaml\n' + yaml.safe_dump(data) + '```\n'
    failures = _admit(body)
    assert any(message in failure for failure in failures), failures


@pytest.mark.parametrize(('reference', 'source', 'message'), [
    ('test:tests/../governance/test_x.py::test_x', SOURCE, 'normal tests'),
    ('test:/tests/test_x.py::test_x', SOURCE, 'normal tests'),
    ('test:tests//test_x.py::test_x', SOURCE, 'normal tests'),
    ('test:tests/benchmark_x.py::test_x', SOURCE, 'normal tests'),
    ('test:governance/tests/test_x.py::test_x', SOURCE, 'normal tests'),
    ('test:tests/test_absent.py::test_x', SOURCE, 'does not exist'),
    (f'test:{PATH}::helper', SOURCE, 'test_name'),
    (f'test:{PATH}::test_python_capability[case]', SOURCE, 'test_name'),
    (PYTHON, 'def outer():\n    def test_python_capability():\n        assert True\n', 'directly'),
    (PYTHON, 'if False:\n    def test_python_capability():\n        assert True\n', 'directly'),
    (PYTHON, 'def test_python_capability(:\n', 'parseable'),
    (PYTHON, '__test__ = False\n' + SOURCE, 'disables collection'),
    (PYTHON, 'pytestmark = pytest.mark.skipif(False)\n' + SOURCE, 'skip/xfail'),
    (PYTHON, '@pytest.mark.xfail\ndef test_python_capability():\n    assert True\n', 'skip/xfail'),
    (COMPILED, '@pytest.mark.skip\nclass TestCompiled:\n    def test_capability(self):\n        assert True\n', 'skip/xfail'),
    (COMPILED, 'class TestCompiled:\n    def __init__(self):\n        self.value = 1\n    def test_capability(self):\n        assert True\n', 'collectible'),
])
def test_rejects_missing_or_uncollectable_proof(reference, source, message):
    def read_source(path):
        if path != PATH:
            raise ValueError(f'candidate proof file does not exist: {path}')
        return source
    failures = _admit(_body(proof={'python': [reference]}), read_source=read_source)
    assert any(message in failure for failure in failures), failures


def test_runtime_skip_is_explicitly_a_review_obligation():
    source = 'def test_python_capability():\n    pytest.skip("runtime environment")\n'
    assert _admit(_body(), read_source=lambda _path: source) == []


def test_parent_contract_is_authoritative():
    parent = _body(interfaces=('python', 'compiled'))
    assert _admit(_body(), parent=parent) == []
    assert _admit(_body('prerequisite', ('python',), proof={'python': [CHECK]}), parent=parent) == []
    assert _admit(_body('maintenance', ('compiled',), proof={'compiled': [CHECK]}), parent=_body()) == []
    assert any('subset' in failure for failure in _admit(_body(interfaces=('compiled',)), parent=_body()))
    assert any('subset' in failure for failure in _admit(_body('prerequisite', ('compiled',)), parent=_body()))
    assert any('public parent' in failure for failure in _admit(_body(), parent=_body('maintenance', ())))
    assert any('parent Interface Contract' in failure for failure in _admit(_body(), parent='Legacy parent without a contract.'))
    assert any('agree' in failure for failure in _admit(_body(), issue=_body(scope='Different scope.')))


def test_parent_proof_existence_is_deferred_until_parent_closure():
    parent = _body(proof={'python': ['test:tests/test_future.py::test_future']})
    assert _admit(_body(), parent=parent) == []
    assert any('parent Interface Contract' in failure and 'does not exist' in failure
               for failure in _admit(_body(), parent=parent, closing=True))
    malformed = _body(proof={'python': ['test:../../test_future.py::test_future']})
    assert any('normal tests' in failure for failure in _admit(_body(), parent=malformed))


def test_missing_pr_contract_never_inherits_or_bypasses():
    assert any('PR Interface Contract' in failure for failure in _admit('', issue=_body()))
    assert _admit(_body(), issue='Legacy slice without a declaration.') == []
    assert _admit(_body(), issue='Legacy code example\n```python\nunfinished') == []
    assert any('trusted default-branch required' in failure for failure in
               _admit(_body('maintenance', ('python',), proof={'python': ['check:invented']})))


def _blob(source, sha=BLOB):
    data = source.encode() if isinstance(source, str) else source
    return {'content': base64.b64encode(data).decode(), 'encoding': 'base64', 'size': len(data), 'sha': sha}


def _tree(*, mode='100644', kind='blob', truncated=False):
    return {'truncated': truncated, 'tree': [{'path': PATH, 'type': kind, 'mode': mode, 'sha': BLOB}]}


def test_candidate_reads_exact_regular_blobs_and_trusted_contexts(monkeypatch):
    rules = {'rules': [{'type': 'required_status_checks', 'parameters': {'required_status_checks': [{'context': 'pr_checks_ruleset'}]}}]}
    responses = {
        f'repos/Vaquum/Limen/git/trees/{SHA}?recursive=1': _tree(),
        f'repos/Vaquum/Limen/git/blobs/{BLOB}': _blob(SOURCE),
        'repos/Vaquum/Limen': {'default_branch': 'main'},
        'repos/Vaquum/Limen/contents/.github/rulesets/main.json?ref=main': _blob(json.dumps(rules)),
    }
    calls = []
    def api(endpoint):
        calls.append(endpoint)
        return responses[endpoint]
    monkeypatch.setattr(contract, '_api', api)
    evidence = contract.CandidateEvidence('Vaquum/Limen', SHA)
    assert evidence(PATH) == evidence(PATH) == SOURCE
    assert evidence.required_checks == evidence.required_checks == {'pr_checks_ruleset'}
    assert len(calls) == len(responses)
    assert _admit(_body(), read_source=evidence) == []


@pytest.mark.parametrize(('mode', 'kind'), [('120000', 'blob'), ('160000', 'commit'), ('040000', 'tree')])
def test_candidate_symlinks_submodules_and_directories_are_not_proof(monkeypatch, mode, kind):
    monkeypatch.setattr(contract, '_api', lambda _endpoint: _tree(mode=mode, kind=kind))
    failures = _admit(_body(), read_source=contract.CandidateEvidence('Vaquum/Limen', SHA))
    assert any('regular file' in failure for failure in failures)


@pytest.mark.parametrize('head', [None, '', 'main', 'a' * 39, 'g' * 40])
def test_candidate_requires_full_head_sha(head):
    with pytest.raises(ValueError, match='40-character'):
        contract.CandidateEvidence('Vaquum/Limen', head)


@pytest.mark.parametrize('payload', [None, {}, {'tree': [], 'truncated': True}, {'tree': 'wrong', 'truncated': False}])
def test_unavailable_or_truncated_candidate_evidence_is_setup_failure(monkeypatch, payload):
    monkeypatch.setattr(contract, '_api', lambda _endpoint: payload)
    with pytest.raises(SystemExit) as error:
        _admit(_body(), read_source=contract.CandidateEvidence('Vaquum/Limen', SHA))
    assert error.value.code == 2


def test_source_setup_failure_propagates_instead_of_becoming_missing_proof():
    def read_source(_path):
        raise SystemExit(2)
    with pytest.raises(SystemExit) as error:
        _admit(_body(), read_source=read_source)
    assert error.value.code == 2
