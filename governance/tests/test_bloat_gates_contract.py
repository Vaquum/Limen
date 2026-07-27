"""Contract tests for the hard-mechanical bloat gates (slice #11)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
BUDGET_JSON: Final[Path] = REPO_ROOT / '.github/module_budgets.json'
LINT_WORKFLOW: Final[Path] = REPO_ROOT / '.github/workflows/pr_checks_lint.yml'
GOVERNANCE_DIR: Final[Path] = REPO_ROOT / 'governance'
PYPROJECT: Final[Path] = REPO_ROOT / 'pyproject.toml'

GATE_SCRIPTS: Final[list[str]] = [
    'check_module_budgets.py',
    'check_test_code_ratio.py',
    'check_module_docstrings.py',
    'check_docstrings.py',
    'check_file_size_balance.py',
    'check_coverage_floor.py',
    'check_coverage_ratchet.py',
    'check_dependency_vulnerabilities.py',
    'check_diff_coverage.py',
    'check_budget_ratchet.py',
    'check_no_swallowed_violations.py',
    'check_test_fallbacks.py',
]

GATE_BANNERS: Final[dict[str, str]] = {
    'check_module_budgets.py': 'MODULE BUDGET GATE',
    'check_test_code_ratio.py': 'TEST/CODE RATIO GATE',
    'check_module_docstrings.py': 'MODULE DOCSTRING GATE',
    'check_docstrings.py': 'DOCSTRING CONVENTIONS GATE',
    'check_file_size_balance.py': 'FILE SIZE BALANCE GATE',
    'check_coverage_floor.py': 'COVERAGE FLOOR GATE',
    'check_coverage_ratchet.py': 'COVERAGE RATCHET GATE',
    'check_dependency_vulnerabilities.py': 'DEPENDENCY VULNERABILITY GATE',
    'check_diff_coverage.py': 'DIFF COVERAGE GATE',
    'check_budget_ratchet.py': 'BUDGET RATCHET GATE',
    'check_no_swallowed_violations.py': 'NO SWALLOWED VIOLATIONS GATE',
    'check_test_fallbacks.py': 'TEST FALLBACK GATE',
}


def _run(script: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(GOVERNANCE_DIR / script), *args]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=cwd or REPO_ROOT)


def test_module_budgets_is_valid_json() -> None:
    data = json.loads(BUDGET_JSON.read_text(encoding='utf-8'))
    assert isinstance(data, dict)
    assert all(isinstance(k, str) for k in data)
    assert all(isinstance(v, int) and v > 0 for v in data.values())


def test_module_budgets_covers_every_package_path() -> None:
    data = json.loads(BUDGET_JSON.read_text(encoding='utf-8'))
    package_paths = {p for p in data if p.startswith('limen/')}
    script_paths = {p for p in data if p.startswith('governance/')}
    # Every .py under the package root is declared in
    # module_budgets.json. Otherwise a new module could silently escape
    # the line-count budget gate.
    actual_paths = _actual_package_paths()
    assert package_paths == actual_paths, (
        f'module_budgets.json paths diverge from actual source tree: '
        f'extra={sorted(package_paths - actual_paths)}, '
        f'missing={sorted(actual_paths - package_paths)}'
    )
    # Every top-level gate module carries a declared budget. The count is
    # derived rather than hardcoded so adding a gate does not require
    # editing a magic number in two places.
    expected_scripts = {
        f'governance/{p.name}'
        for p in (REPO_ROOT / 'governance').glob('*.py')
    }
    assert script_paths == expected_scripts, (
        f'extra={sorted(script_paths - expected_scripts)}, '
        f'missing={sorted(expected_scripts - script_paths)}'
    )


def _actual_package_paths() -> set[str]:
    root = REPO_ROOT / 'limen'
    return {
        str(p.relative_to(REPO_ROOT)).replace('\\', '/')
        for p in root.rglob('*.py')
        if '__pycache__' not in p.parts
    }


def test_all_gate_scripts_exist_and_are_executable() -> None:
    for name in GATE_SCRIPTS:
        path = GOVERNANCE_DIR / name
        assert path.is_file(), f'{path} missing'
        assert path.stat().st_mode & 0o111, f'{path} not executable'


def test_all_wired_scripts_pass_on_current_repo() -> None:
    _run('check_module_budgets.py').check_returncode()
    _run('check_no_swallowed_violations.py').check_returncode()
    _run(
        'check_budget_ratchet.py',
        '--base-file', '/dev/null',
        '--pr-body-file', '/dev/null',
    ).check_returncode()


# Gates that ship but are not wired into pr_checks_lint, and do not pass on
# this repository. Each is named in CLAUDE.md law 6 with its reason. They are
# still exercised here for banner and exit-code shape, just not for a PASS.
UNWIRED_GATES: Final[frozenset[str]] = frozenset({
    # Coverage is enforced by `PR Checks Coverage` at a 92% floor, computed
    # from the suite's own run. Wiring these into pr_checks_lint would run the
    # suite a second time to enforce a weaker bound.
    'check_coverage_floor.py',
    'check_coverage_ratchet.py',
    'check_diff_coverage.py',
    'check_test_code_ratio.py',
    'check_module_docstrings.py',
    'check_docstrings.py',
    'check_file_size_balance.py',
    'check_test_fallbacks.py',
})


def test_pass_banners_printed_on_success() -> None:
    for name in ('check_module_budgets.py',):
        result = _run(name)
        assert result.returncode == 0, result.stderr
        banner = GATE_BANNERS[name]
        assert f'{banner} -- PASS' in result.stdout, f'{name} stdout: {result.stdout!r}'


def test_unwired_gates_still_report_under_their_banner() -> None:
    # They are not required, but they must remain runnable and honest: a
    # non-zero exit under the gate's own FAIL banner, never a crash and never
    # a silent zero.
    # The ratchet gates take a required --base-ref/--base-file, so a bare
    # run is an argparse usage error rather than a gate verdict. They are
    # exercised with arguments in test_budget_ratchet_vacuous_when_base_missing.
    for name in sorted(UNWIRED_GATES - {'check_coverage_ratchet.py', 'check_diff_coverage.py'}):
        result = _run(name)
        banner = GATE_BANNERS[name]
        # 1 is a violation, 2 is a setup failure (a gate that cannot run
        # fails closed rather than passing over an empty tree). Both are
        # honest; a silent 0 would not be.
        assert result.returncode in (1, 2), f'{name} exited {result.returncode}'
        assert f'{banner} -- FAIL' in result.stdout + result.stderr, name


def test_fail_banners_are_declared_in_each_script_source() -> None:
    for name in GATE_SCRIPTS:
        source = (GOVERNANCE_DIR / name).read_text(encoding='utf-8')
        banner = GATE_BANNERS[name]
        assert f'{banner} -- FAIL' in source, f'{name} missing FAIL banner literal'
        assert f'{banner} -- PASS' in source, f'{name} missing PASS banner literal'


def test_workflow_invokes_every_wired_gate() -> None:
    workflow = LINT_WORKFLOW.read_text(encoding='utf-8')
    for script in GATE_SCRIPTS:
        if script in UNWIRED_GATES:
            assert f'governance/{script}' not in workflow, (
                f'{script} is named unwired but the workflow invokes it')
            continue
        assert f'governance/{script}' in workflow, f'{script} not invoked by workflow'
    assert 'steps.package.outputs.package_root' in workflow
    assert 'ruff check "${{ steps.package.outputs.package_root }}" governance tests' in workflow


def test_no_soft_fail_pathway_in_workflow() -> None:
    workflow = LINT_WORKFLOW.read_text(encoding='utf-8')
    assert '|| true' not in workflow
    assert 'continue-on-error' not in workflow
    forbidden_flags = re.compile(r'--warn-only|--no-fail|--soft(-fail)?')
    assert forbidden_flags.search(workflow) is None


def test_scripts_are_self_budgeted() -> None:
    data = json.loads(BUDGET_JSON.read_text(encoding='utf-8'))
    for name in GATE_SCRIPTS:
        key = f'governance/{name}'
        assert key in data, f'{key} missing from module_budgets.json'
        # 130 rather than upstream's 120: three gates carry a guarded
        # `tomllib` import for the 3.10 floor (see each module's docstring
        # and Vaquum/new-repository-template#97).
        assert data[key] <= 130, f'{key} budget {data[key]} exceeds the 130-line self-limit'


def test_ruff_select_includes_new_rules() -> None:
    cfg = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))
    select = cfg['tool']['ruff']['lint']['select']
    # This repository selects rule families rather than upstream's individual
    # rules, and deliberately does not enable the complexity/arity family --
    # see Vaquum/Limen#772 for why that was withdrawn rather than adopted.
    for rule in ('E', 'F', 'W', 'N', 'UP', 'ANN', 'S', 'BLE', 'B', 'PL', 'RUF'):
        assert rule in select, f'ruff select missing {rule}'


def test_budget_ratchet_vacuous_when_base_missing() -> None:
    result = _run('check_budget_ratchet.py', '--base-file', '/dev/null', '--pr-body-file', '/dev/null')
    assert result.returncode == 0
    assert 'BUDGET RATCHET GATE -- PASS' in result.stdout
    assert 'vacuous' in result.stdout.lower()


def test_budget_ratchet_accepts_marker(tmp_path: Path) -> None:
    # Build a self-contained repo layout in tmp_path that actually has a
    # budget raise between base and head. Previous version ran against
    # the real head budget, so the base's `foo.py` key never matched
    # anything in head and the marker logic was never exercised.
    (tmp_path / '.github').mkdir()
    head = {'new_repository_template/foo.py': 200}
    base = {'new_repository_template/foo.py': 100}
    (tmp_path / '.github' / 'module_budgets.json').write_text(json.dumps(head), encoding='utf-8')
    base_file = tmp_path / 'base.json'
    base_file.write_text(json.dumps(base), encoding='utf-8')
    body_file = tmp_path / 'body.txt'
    body_file.write_text(
        '[budget-raise: new_repository_template/foo.py: legitimate growth]\n',
        encoding='utf-8',
    )
    scripts_dir = tmp_path / 'governance'
    scripts_dir.mkdir()
    (scripts_dir / '__init__.py').write_text('', encoding='utf-8')
    import shutil
    shutil.copy2(GOVERNANCE_DIR / '_common.py', scripts_dir / '_common.py')
    shutil.copy2(GOVERNANCE_DIR / 'check_budget_ratchet.py', scripts_dir / 'check_budget_ratchet.py')
    result = subprocess.run(
        [sys.executable, str(scripts_dir / 'check_budget_ratchet.py'),
         '--base-file', str(base_file), '--pr-body-file', str(body_file)],
        check=False, capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'BUDGET RATCHET GATE -- PASS' in result.stdout
