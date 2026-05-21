import shutil
from pathlib import Path

import click

from limen.cli.commands._constants import TEMPLATES_DIR


def run_init(output: Path, template_name: str | None) -> bool:

    '''
    Scaffold a new YAML experiment file from a template.

    Args:
        output (Path): Destination path for the new experiment file
        template_name (str | None): Template stem name (e.g. 'logreg_binary'); if None,
            lists available templates and returns False

    Returns:
        bool: True on success, False on failure

    '''

    available = {p.stem: p for p in sorted(TEMPLATES_DIR.glob('*.yaml'))}

    if template_name is None:
        click.echo('Available templates:')
        for name in available:
            click.echo(f'  {name}')
        click.echo()
        click.secho('Usage: limen init <output.yaml> --template <name>', fg='yellow')
        return False

    if template_name not in available:
        click.secho(
            f"Template '{template_name}' not found. "
            f"Available: {', '.join(available)}",
            fg='red',
        )
        return False

    if output.exists():
        click.secho(f"'{output}' already exists — will not overwrite.", fg='red')
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(available[template_name], output)

    _update_experiment_name(output, output.stem)

    click.secho(f"  ✓ Created '{output}' from template '{template_name}'", fg='green')
    click.echo(f"  Edit the file, then run: limen validate {output}")
    return True


def _update_experiment_name(path: Path, name: str) -> None:

    text = path.read_text()
    lines = text.splitlines(keepends=True)

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('name:'):
            indent = line[: len(line) - len(stripped)]
            lines[i] = f'{indent}name: {name}\n'
            break

    path.write_text(''.join(lines))
