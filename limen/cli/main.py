from pathlib import Path

import click

from limen.cli.commands.run import run_experiment
from limen.cli.commands.validate import run_validate


@click.group()
def cli() -> None:

    '''Limen — declarative experiment runner.'''


@cli.command()
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
def validate(yaml_file: Path) -> None:

    '''Validate a YAML experiment file.'''

    ok = run_validate(yaml_file)
    raise SystemExit(0 if ok else 1)


@cli.command()
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
@click.option('--dry-run', is_flag=True, default=False,
              help='Validate only — do not execute the experiment')
def run(yaml_file: Path, dry_run: bool) -> None:

    '''Validate, compile, and run a YAML experiment file.'''

    ok = run_experiment(yaml_file, dry_run=dry_run)
    raise SystemExit(0 if ok else 1)
