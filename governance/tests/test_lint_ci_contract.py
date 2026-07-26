from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib

import yaml
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[2]
LINT_WORKFLOW: Final[Path] = REPO_ROOT / '.github/workflows/pr_checks_lint.yml'
RULESET_WORKFLOW: Final[Path] = REPO_ROOT / '.github/workflows/pr_checks_ruleset.yml'
RULESET_SNAPSHOT: Final[Path] = REPO_ROOT / '.github/rulesets/main.json'
DEV_ENV_IN: Final[Path] = REPO_ROOT / 'requirements/ci/dev-env.in'
DEV_ENV_TXT: Final[Path] = REPO_ROOT / 'requirements/ci/dev-env.txt'
BAD_FIXTURE: Final[Path] = REPO_ROOT / 'governance/tests/fixtures/lint/bad_imports.py'
RUFF_VERSION: Final[str] = yaml.safe_load(
    (REPO_ROOT / 'governance.yml').read_text(encoding='utf-8')
)['toolchain']['ruff_version']
EXPECTED_RUFF_POLICY: Final[dict[str, object]] = {
    "exclude": [
        "governance/tests/fixtures"
    ],
    "select": [
        "F",
        "E",
        "W",
        "N",
        "UP",
        "ANN",
        "S",
        "BLE",
        "B",
        "A",
        "C4",
        "G",
        "T20",
        "PT",
        "RET",
        "SIM",
        "ARG",
        "PTH1",
        "ERA",
        "PL",
        "RUF"
    ],
    "ignore": [
        "D201",
        "D202",
        "D203",
        "D212",
        "E501",
        "I",
        "D200",
        "D100",
        "ANN201",
        "D102",
        "PLR0913",
        "D107",
        "S101",
        "S113",
        "S608",
        "S311",
        "N806",
        "PLC0415",
        "PLR0912",
        "PLR0915",
        "RUF002",
        "PT",
        "N803",
        "ANN401",
        "C401",
        "RET504",
        "PLW0108"
    ],
    "per-file-ignores": {
        "governance/*.py": [
            "T201",
            "S603",
            "S607",
            "PLR5501",
            "PLR0911",
            "PLR2004",
            "B007",
            "PTH123"
        ],
        "governance/tests/**/*.py": [
            "S101",
            "BLE001",
            "PLR2004",
            "B011",
            "ANN",
            "ARG",
            "S603",
            "S607",
            "S102",
            "S108"
        ],
        "tests/**/*.py": [
            "S101",
            "BLE001",
            "PLR2004",
            "B011",
            "ANN",
            "ARG",
            "S603",
            "S607"
        ],
        "limen/targets/*.py": [
            "ARG002"
        ],
        "limen/cli/**/*.py": [
            "S603"
        ]
    }
}


def _required_status_contexts() -> list[str]:
    payload = json.loads(RULESET_SNAPSHOT.read_text(encoding='utf-8'))
    for rule in payload['rules']:
        if rule['type'] == 'required_status_checks':
            checks = rule['parameters']['required_status_checks']
            return [entry['context'] for entry in checks]
    raise AssertionError('required_status_checks rule missing from ruleset snapshot')


def _run_ruff(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-m', 'ruff', *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_pr_checks_lint_workflow_exists() -> None:
    assert LINT_WORKFLOW.exists()


def test_ruleset_snapshot_requires_pr_checks_lint() -> None:
    assert 'pr_checks_lint' in _required_status_contexts()


def test_pr_checks_lint_runs_pinned_ruff_on_tools_and_tests_tools() -> None:
    # Name kept for slice #11 Tests-table backward compatibility. The
    # assertions inside now cover the broader post-#11 surface
    # (ruff + every gate script + no-soft-fail). A future slice may
    # split or rename this; doing so requires updating the slice body's
    # Tests table in lockstep.
    workflow = LINT_WORKFLOW.read_text(encoding='utf-8')

    assert '--require-hashes -r requirements/ci/dev-env.txt' in workflow
    assert '--no-deps -e .' in workflow
    assert 'id: package' in workflow
    assert 'package_root="$(python - <<' in workflow
    assert 'python -m ruff check "${{ steps.package.outputs.package_root }}" governance tests' in workflow
    assert 'continue-on-error' not in workflow
    # Each gate invocation must appear verbatim somewhere in the workflow.
    assert 'governance/check_module_budgets.py' in workflow
    assert 'governance/check_no_swallowed_violations.py' in workflow
    assert 'governance/check_dependency_vulnerabilities.py' in workflow
    assert 'governance/check_budget_ratchet.py' in workflow
    # vulture is named in the workflow header explaining its absence, so
    # assert it is not *invoked* rather than not mentioned.
    assert '-m vulture' not in workflow

    # Deliberately absent, and asserted absent so their omission is a recorded
    # decision rather than drift. Each is named in CLAUDE.md law 6 with its
    # reason; see also the workflow's own header comment.
    for withdrawn in (
        'governance/check_module_docstrings.py',
        'governance/check_docstrings.py',
        'governance/check_test_fallbacks.py',
        'governance/check_test_code_ratio.py',
        'governance/check_file_size_balance.py',
    ):
        assert withdrawn not in workflow, withdrawn

    # Coverage lives in `PR Checks Tests`/`PR Checks Coverage`, which already
    # runs the suite once and enforces a 92% floor; wiring the floor gate here
    # too would run the suite twice for a weaker bound. Docs-site checks live
    # in `PR Checks Docs Site`, built on this repo's own docs generator rather
    # than the upstream docs-map model.
    assert 'governance/check_coverage_floor.py' not in workflow
    assert 'npm --prefix docs-site' not in workflow

    # No soft-fail pathway: no `|| true`, no continue-on-error on any step.
    assert '|| true' not in workflow


def test_pr_checks_ruleset_runs_test_lint_ci_contract() -> None:
    workflow = RULESET_WORKFLOW.read_text(encoding='utf-8')

    assert '--require-hashes -r requirements/ci/dev-env.txt' in workflow
    assert 'governance/tests/test_lint_ci_contract.py' in workflow


def test_pinned_ruff_fails_on_known_bad_fixture() -> None:
    version = _run_ruff('--version')
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == f'ruff {RUFF_VERSION}'

    result = _run_ruff('check', str(BAD_FIXTURE))

    assert result.returncode == 1
    assert 'bad_imports.py' in f'{result.stdout}\n{result.stderr}'


def test_ruff_pin_is_consistent_across_requirement_sets() -> None:
    # The lint and ruleset venvs install the compiled dev-env set, so
    # the ruff the gates run is whatever dev-env pins; source (.in) and
    # compiled (.txt) must both carry exactly the contract version.
    files = [DEV_ENV_IN, DEV_ENV_TXT]
    pins = sorted({
        pin
        for source in files
        for pin in re.findall(r'^ruff==([0-9.]+)', source.read_text(encoding='utf-8'), re.MULTILINE)
    })

    assert pins == [RUFF_VERSION]


def test_pyproject_ruff_policy_contract() -> None:
    data = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    ruff = data['tool']['ruff']
    actual_policy = {
        'exclude': ruff.get('exclude'),
        'select': ruff['lint'].get('select'),
        'ignore': ruff['lint'].get('ignore'),
        'per-file-ignores': ruff['lint'].get('per-file-ignores'),
    }

    assert actual_policy == EXPECTED_RUFF_POLICY

    # Config-independent blind-except scan, scoped to the gate code. A bare
    # `except Exception` in a gate can hide the very violation the gate exists
    # to report, so it is forbidden there regardless of per-file-ignores.
    # `tests/` is governed by the pyproject per-file-ignores instead: this
    # suite deliberately catches broadly where it parses arbitrary content and
    # aggregates every failure for one assertion (see
    # tests/test_docs_surface.py::_assert_fences_parse), which is the opposite
    # of swallowing.
    result = _run_ruff('check', '--isolated', '--select', 'BLE001', 'governance')
    assert result.returncode == 0, result.stdout + result.stderr
