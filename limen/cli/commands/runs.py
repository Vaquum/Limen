from pathlib import Path

import click

from limen.yaml.config import find_project_root


def run_runs(start: Path) -> bool:

    '''
    List all run directories under results/ in the current project.

    Args:
        start (Path): Directory to start searching for the project root

    Returns:
        bool: True on success, False if no project found

    '''

    project_root = find_project_root(start)
    if project_root is None:
        click.secho('  ✗ No limen project found. Run this command from inside a Limen project.', fg='red')
        return False

    results_root = project_root / 'results'
    runs = sorted(
        (csv.parent for csv in results_root.rglob('results.csv')),
        key=lambda d: str(d),
    ) if results_root.exists() else []

    if not runs:
        click.echo('  No runs yet. Use limen run to execute an experiment.')
        return True

    click.echo(f"Runs ({len(runs)}):\n")
    for run_dir in runs:
        rel = run_dir.relative_to(project_root)
        kind = 'dev' if rel.parts[1:2] == ('dev',) else 'committed'
        permutations = _count_permutations(run_dir / 'results.csv')
        click.echo(f"  {rel!s:<48}  {permutations:>4} perms  [{kind}]")

    return True


def _count_permutations(csv_path: Path) -> int:

    try:
        with csv_path.open(encoding='utf-8') as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return 0
