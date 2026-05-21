from pathlib import Path

import click

from limen.yaml.parser import parse

_TEMPLATES_DIR = Path(__file__).parent.parent.parent / 'yaml' / 'templates'


def run_list_templates() -> None:

    '''Print all available YAML experiment templates with their descriptions.'''

    templates = sorted(_TEMPLATES_DIR.glob('*.yaml'))

    if not templates:
        click.echo('No templates found.')
        return

    click.echo(f'Available templates ({_TEMPLATES_DIR}):')
    click.echo()

    name_width = max(len(t.stem) for t in templates)

    for path in templates:
        yaml_dict, errors = parse(path)
        description = ''
        if not errors:
            description = yaml_dict.get('metadata', {}).get('description', '')
        click.echo(f'  {path.stem:<{name_width}}    {description}')
