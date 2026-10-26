import click

from limen.yaml._callback_spec import callback_error
from limen.yaml._schema_version import schema_errors
from limen.yaml.validator import unknown_errors

__all__: list[str] = []


def admit_saved_yaml(document: dict[str, object]) -> bool:
    '''Refuse an unsupported schema or structural typo before reconstruction.'''
    errors = schema_errors(document)
    if not errors and (message := callback_error(document)):
        click.secho(f'  ✗ {message}', fg='red')
        return False
    if not errors:
        errors = unknown_errors(document)
    for error in errors:
        detail = f'{error.message}. {error.suggestion}' if error.suggestion else error.message
        click.secho(f'  ✗ {error.path}: {detail}', fg='red')
    return not errors
