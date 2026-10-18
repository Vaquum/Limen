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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plane', choices=('suite', 'static', 'typing'))
    args = parser.parse_args()
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
    if args.plane == 'suite':
        os.environ['LIMEN_COVERAGE_RUN'] = '1'
        os.environ['COVERAGE_CORE'] = 'sysmon'
        run('-m', 'coverage', 'run', '-m', 'tests.run')
        run('-m', 'coverage', 'report', '--fail-under=92')
        run('-m', 'tests.runtime_gate', '--profile', 'coverage-artifacts/test_runtime_profile.json',
            '--budget', 'tests/runtime_budget.json', '--enforce')
    elif args.plane == 'static':
        run('-m', 'ruff', 'check', 'limen', 'governance', 'tests')
        for gate in ('check_module_budgets', 'check_no_swallowed_violations',
                     'fail_loud_gate'):
            run(f'governance/{gate}.py')
        run('governance/check_budget_ratchet.py', '--base-ref', 'origin/main', '--pr-number', '974')
        run('scripts/package_audit.py')
    else:
        result = subprocess.run([sys.executable, '-m', 'pyright', '--outputjson'],
                                cwd=ROOT, check=False, capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr or result.stdout)
        report = json.loads(result.stdout)
        sys.stdout.write(json.dumps(report['summary'], indent=2) + '\n')
        sys.stdout.flush()
        with tempfile.TemporaryDirectory(prefix='limen-backtest-typing-') as directory:
            path = Path(directory) / 'pyright.json'
            path.write_text(result.stdout)
            run('governance/typing_gate.py', '--pyright-json', str(path))


if __name__ == '__main__':
    main()
