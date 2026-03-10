'''
Declarative filter type constants and builders for named filters.

Each constant defines a filter_type string used in intervention dicts.
FILTER_BUILDERS maps these to factory functions that create filter callables
from filter_params dicts.

'''

import hashlib
from collections.abc import Callable
from typing import Any

FILTER_EXCLUDE_VALUE = 'exclude_value'
FILTER_KEEP_VALUES = 'keep_values'
FILTER_KEEP_BETWEEN = 'keep_between'
FILTER_SAMPLE = 'sample'

FILTER_BUILDERS: dict[str, Callable[[dict[str, Any]], Callable[[dict[str, Any]], bool]]] = {
    FILTER_EXCLUDE_VALUE: lambda fp: lambda c: c[fp['param']] == fp['value'],
    FILTER_KEEP_VALUES: lambda fp: lambda c: c[fp['param']] not in fp['values'],
    FILTER_KEEP_BETWEEN: lambda fp: lambda c: not (fp['lower'] <= c[fp['param']] <= fp['upper']),
    FILTER_SAMPLE: lambda fp: lambda c: (
        c.get(fp['param']) == fp['value']
        and int(hashlib.sha256(
            str(sorted(c.items())).encode()
        ).hexdigest(), 16) % 10_000 >= round(fp['fraction'] * 10_000)
    ),
}
