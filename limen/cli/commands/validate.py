from pathlib import Path

import click

from limen.yaml.parser import parse
from limen.yaml.validator import validate


def run_validate(yaml_path: Path) -> bool:

    '''
    Parse and validate a YAML experiment file.

    Args:
        yaml_path (Path): Path to the YAML experiment file

    Returns:
        bool: True if valid, False if errors were found

    '''

    click.echo(f"Validating {yaml_path} ...")

    yaml_dict, parse_errors = parse(yaml_path)

    if parse_errors:
        for e in parse_errors:
            location = f' (line {e.line})' if e.line else ''
            click.secho(f'  PARSE ERROR{location}: {e.message}', fg='red')
        return False

    result = validate(yaml_dict)

    for e in result.errors:
        location = f' (line {e.line})' if e.line else ''
        path = f'  [{e.path}]' if e.path else ''
        suggestion = f'\n    → {e.suggestion}' if e.suggestion else ''
        click.secho(f'  ERROR{path}{location}: {e.message}{suggestion}', fg='red')

    for w in result.warnings:
        location = f' (line {w.line})' if w.line else ''
        path = f'  [{w.path}]' if w.path else ''
        click.secho(f'  WARN{path}{location}: {w.message}', fg='yellow')

    if result.valid:
        click.secho('  ✓ Valid', fg='green')
    else:
        click.secho(f'  ✗ {len(result.errors)} error(s) found', fg='red')

    return result.valid
