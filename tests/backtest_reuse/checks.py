"""Run the published backtest slice's local validation planes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def base_budget(name: str, directory: Path) -> str:
    result = subprocess.run(['git', 'show', f'origin/main:.github/{name}.json'],
                            cwd=ROOT, check=True, capture_output=True)
    path = directory / f'{name}.json'
    path.write_bytes(result.stdout)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plane', choices=('suite', 'static', 'typing'))
    args = parser.parse_args()
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
    directory = Path(tempfile.mkdtemp(prefix='limen-backtest-check-'))
    sys.stdout.write(f'Validation artifacts: {directory}\n')
    sys.stdout.flush()
    if args.plane == 'suite':
        os.environ['LIMEN_COVERAGE_RUN'] = '1'
        os.environ['COVERAGE_CORE'] = 'sysmon'
        profile = directory / 'test_runtime_profile.json'
        os.environ['LIMEN_RUNTIME_PROFILE_PATH'] = str(profile)
        run('-m', 'coverage', 'run', '-m', 'tests.run')
        run('-m', 'coverage', 'report', '--fail-under=92')
        run('-m', 'tests.runtime_gate', '--profile', str(profile),
            '--budget', 'tests/runtime_budget.json', '--enforce')
    elif args.plane == 'static':
        run('-m', 'ruff', 'check', 'limen', 'governance', 'tests')
        for gate in ('check_module_budgets', 'check_no_swallowed_violations'):
            run(f'governance/{gate}.py')
        run('governance/fail_loud_gate.py', '--base-budget',
            base_budget('fail_loud_budget', directory))
        body = subprocess.run(['gh', 'pr', 'view', 'https://github.com/Vaquum/Limen/pull/974',
                               '--json', 'body', '--jq', '.body'],
                              cwd=ROOT, check=True, capture_output=True, text=True)
        body_path = directory / 'pr-body.md'
        body_path.write_text(body.stdout)
        run('governance/check_budget_ratchet.py', '--base-file',
            base_budget('module_budgets', directory), '--pr-body-file', str(body_path))
        run('scripts/package_audit.py')
    else:
        result = subprocess.run([sys.executable, '-m', 'pyright', '--outputjson'],
                                cwd=ROOT, check=False, capture_output=True, text=True)
        path = directory / 'pyright.json'
        path.write_text(result.stdout)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr or result.stdout)
        report = json.loads(result.stdout)
        sys.stdout.write(json.dumps(report['summary'], indent=2) + '\n')
        sys.stdout.flush()
        run('governance/typing_gate.py', '--pyright-json', str(path),
            '--base-budget', base_budget('typing_budget', directory))


if __name__ == '__main__':
    main()
