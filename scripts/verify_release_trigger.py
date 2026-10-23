"""Bind package publication to the exact successful manual release source."""
import json
from pathlib import Path
import re
import subprocess
import sys


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(['git', *arguments], cwd=root, text=True).strip()


def verify_trigger(root: Path, run) -> None:
    head = run.get('head_sha')
    if (run.get('path') != '.github/workflows/pr_post_release.yml'
            or run.get('event') != 'workflow_dispatch'
            or run.get('conclusion') != 'success'
            or run.get('head_repository', {}).get('full_name') != 'Vaquum/Limen'
            or not isinstance(head, str) or not re.fullmatch('[0-9a-f]{40}', head)
            or git(root, 'rev-parse', 'HEAD') != head):
        raise ValueError('Package publication requires the exact successful manual release source')
    subprocess.run(['git', 'merge-base', '--is-ancestor', head, 'origin/main'], cwd=root, check=True)
    branch = run.get('head_branch')
    if branch == 'main':
        return
    version = re.search(r'^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"',
                        (root / 'pyproject.toml').read_text(), re.M)
    if version is None or branch != f'v{version.group(1)}-dispatch-{head}':
        raise ValueError('Manual release tag must bind the package version and full merged commit')
    if git(root, 'rev-parse', f'refs/tags/{branch}^{{commit}}') != head:
        raise ValueError('Manual release tag points to another commit')


if __name__ == '__main__':
    verify_trigger(Path(__file__).resolve().parents[1], json.loads(Path(sys.argv[1]).read_text()))
