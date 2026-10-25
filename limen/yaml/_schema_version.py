from limen.yaml.errors import YAMLError
from limen.yaml.rules import Required, SchemaVersion
from limen.yaml.schema import VERSION

__all__: list[str] = []


def schema_errors(yaml_dict: dict[str, object]) -> list[YAMLError]:
    errors: list[YAMLError] = []
    Required('schema_version', str, suggestion=f'Set schema_version: "{VERSION}"').check(yaml_dict, errors, [])
    if not errors:
        SchemaVersion(VERSION).check(yaml_dict, errors, [])
    return errors
