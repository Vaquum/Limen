from typing import cast

__all__: list[str] = []


def column_issue(value: object) -> tuple[int | None, str, type[TypeError] | type[ValueError]] | None:
    if not isinstance(value, list):
        return None, f'required_columns must be a list (got {type(value).__name__}); omit the key to use the default', TypeError
    if not value:
        return None, 'required_columns must be nonempty; omit the key to use the default', ValueError
    seen: dict[str, int] = {}
    for index, column in enumerate(cast(list[object], value)):
        if not isinstance(column, str):
            return index, f'required_columns[{index}] must be a string (got {type(column).__name__})', TypeError
        if not column.strip():
            return index, f'required_columns[{index}] must not be blank', ValueError
        if column in seen:
            return index, f'required_columns[{index}] duplicates required_columns[{seen[column]}] ({column!r})', ValueError
        seen[column] = index
    return None


def admit_columns(value: object) -> list[str]:
    if issue := column_issue(value):
        _, message, error_type = issue
        raise error_type(message)
    return cast(list[str], value).copy()
