from collections.abc import Mapping
from datetime import date, datetime
import re
from typing import cast

from limen.yaml.errors import YAMLError
from limen.yaml.schema import METADATA_OPTIONAL

__all__: list[str] = []

_CREATED = re.compile(
    r'[0-9]{4}-[0-9]{2}-[0-9]{2}'
    r'(?:T[0-9]{2}:[0-5][0-9]:[0-5][0-9](?:\.(?P<fraction>[0-9]{1,6}))?'
    r'(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?)?'
)
# PyPA component spellings; this check never normalizes or compares versions.
_VERSION = re.compile(
    r'v?(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*'
    r'(?:[-_.]?(?:a|b|c|rc|alpha|beta|pre|preview)[-_.]?[0-9]*)?'
    r'(?:(?:-[0-9]+)|(?:[-_.]?(?:post|rev|r)[-_.]?[0-9]*))?'
    r'(?:[-_.]?dev[-_.]?[0-9]*)?(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?', re.IGNORECASE | re.ASCII,
)


def _created_at(value: object) -> bool:
    if not isinstance(value, str) or (match := _CREATED.fullmatch(value)) is None:
        return False
    normalized = value.replace('Z', '+00:00')
    fraction = match.group('fraction')
    if fraction:
        normalized = normalized.replace('.' + fraction, '.' + fraction.ljust(6, '0'), 1)
    try:
        _ = datetime.fromisoformat(normalized) if 'T' in normalized else date.fromisoformat(normalized)
    except ValueError:
        return False
    return True


def _error(errors: list[YAMLError], path: str, message: str, suggestion: str | None = None) -> None:
    errors.append(YAMLError(message=message, path='metadata.' + path, suggestion=suggestion))


def check_metadata(document: Mapping[str, object], errors: list[YAMLError]) -> None:
    raw = document.get('metadata')
    if not isinstance(raw, Mapping):
        return
    metadata = cast(Mapping[str, object], raw)
    for name in sorted(METADATA_OPTIONAL):
        value = metadata.get(name)
        if value is None:
            continue
        if name == 'author':
            if not isinstance(value, str) or not value.strip():
                _error(errors, name, 'author must be a nonblank string')
        elif name == 'description':
            if not isinstance(value, str):
                _error(errors, name, 'description must be a string')
        elif name == 'tags':
            if not isinstance(value, list):
                _error(errors, name, 'tags must be a list of nonblank strings')
            else:
                for index, tag in enumerate(cast(list[object], value)):
                    if not isinstance(tag, str) or not tag.strip():
                        _error(errors, f'tags[{index}]', 'tag must be a nonblank string', 'Use quoted tag strings')
        elif name == 'created_at':
            if not _created_at(value):
                _error(errors, name, 'created_at must be a quoted ISO date or timestamp',
                    'Quote a real YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS[.ffffff][Z or +/-HH:MM] value')
        elif name == 'limen_version' and (not isinstance(value, str) or _VERSION.fullmatch(value) is None):
            _error(errors, name, 'limen_version must be a Python package-version string',
                'Quote the version; metadata does not establish schema compatibility')
