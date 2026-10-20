import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from provenance import base_revision, git

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / 'docs/examples/retained_checker.py'


@pytest.fixture(scope='module')
def retained_checkout(tmp_path_factory):
    root = tmp_path_factory.mktemp('retained-checker') / 'product'
    recorded = json.loads((ROOT / 'docs/examples/tests/fixtures/retained-source-prs.json').read_text())
    subprocess.run(['git', 'clone', '--shared', '--no-checkout', str(ROOT), str(root)], check=True, capture_output=True)
    subprocess.run(['git', 'fetch', 'origin', 'refs/remotes/origin/main:refs/remotes/origin/main'], cwd=root, check=True, capture_output=True)
    subprocess.run(['git', 'checkout', '--detach', recorded['head']], cwd=root, check=True, capture_output=True)
    record = {name: recorded[name] for name in ('schema_version', 'base', 'files', 'responses')}
    output = root / '.docs-evidence/source-provenance.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record))
    assert record['base'] == base_revision(root)
    assert git(root, 'merge-base', 'HEAD', 'origin/main') == recorded['head']
    launcher = root / '.docs-evidence/reviewed-python'
    subprocess.run([sys.executable, str(DRIVER), '--install-launcher', str(launcher)], cwd=root, check=True, capture_output=True)
    return root, record, launcher


@pytest.mark.parametrize('corrupt_source', [False, True])
def test_retained_original_imports_use_reviewed_checker_without_changing_product(retained_checkout, corrupt_source):
    root, record, launcher = retained_checkout
    selected = copy.deepcopy(record)
    if corrupt_source:
        selected['files'][0]['sha256'] = '0' * 64
    (root / '.docs-evidence/source-provenance.json').write_text(json.dumps(selected))
    probe = root / '.docs-evidence/test_reviewed_checker.py'
    probe.write_text('''import json
from pathlib import Path
import check_evidence
import provenance

def test_original_checker_binding():
    root = Path.cwd()
    assert Path(check_evidence.__file__).resolve() == root / 'docs/examples/check_evidence.py'
    assert Path(provenance.__file__).resolve() == Path(''' + repr(str(DRIVER.with_name('provenance.py'))) + ''')
    provenance.validate_provenance(root, json.loads((root / '.docs-evidence/source-provenance.json').read_text()))
''')
    before = git(root, 'rev-parse', 'HEAD^{tree}')
    prior_receipts = set((root / '.docs-evidence/retained-checker').glob('*.json'))
    original = 'docs/examples/tests/test_provenance.py::test_source_association_requires_its_exact_merged_main_pr'
    result = subprocess.run([str(launcher), '-m', 'pytest', '--import-mode=importlib', original, str(probe), '-q'], cwd=root, text=True, capture_output=True)
    assert result.returncode == int(corrupt_source), result.stdout + result.stderr
    if corrupt_source:
        assert 'Source-changing commit or source bytes differ' in result.stdout
    evidence = json.loads((root / '.docs-evidence/current.json').read_text())
    assert evidence['status'] == ('failed' if corrupt_source else 'incomplete')
    created = set((root / '.docs-evidence/retained-checker').glob('*.json')) - prior_receipts
    assert len(created) == 1
    receipt = json.loads(created.pop().read_text())
    assert receipt['exit_code'] == result.returncode
    assert receipt['product'] == {'head': git(root, 'rev-parse', 'HEAD'), 'tree': before}
    assert str(DRIVER.with_name('provenance.py')) in receipt['checker']
    assert git(root, 'rev-parse', 'HEAD^{tree}') == before
    assert git(root, 'status', '--porcelain', '--untracked-files=normal') == ''
    original_evidence = root / 'docs/examples/check_evidence.py'
    prior_receipts = set((root / '.docs-evidence/retained-checker').glob('*.json'))
    refused = subprocess.run([str(launcher), str(original_evidence), '--evidence', str(root / '.docs-evidence/current.json'), '--head', git(root, 'rev-parse', 'HEAD'), '--tree', before, '--version', '5.45.12'], cwd=root, text=True, capture_output=True)
    assert refused.returncode == 1
    assert 'check_evidence.py' in refused.stderr
    created = set((root / '.docs-evidence/retained-checker').glob('*.json')) - prior_receipts
    assert len(created) == 1
    assert json.loads(created.pop().read_text())['exit_code'] == refused.returncode


@pytest.mark.parametrize('arguments,code,error', [
    (['docs/examples/released_wheel.py'], 1, 'only runs the original documentation checks'),
    (['docs/examples/provenance.py', '--unexpected-option'], 2, '--output'),
])
def test_retained_checker_refuses_an_unrelated_script(retained_checkout, arguments, code, error):
    root, _, launcher = retained_checkout
    prior_receipts = set((root / '.docs-evidence/retained-checker').glob('*.json'))
    result = subprocess.run([str(launcher), *arguments], cwd=root, text=True, capture_output=True)
    assert result.returncode == code
    assert error in result.stderr
    created = set((root / '.docs-evidence/retained-checker').glob('*.json')) - prior_receipts
    assert len(created) == 1
    assert json.loads(created.pop().read_text())['exit_code'] == code
