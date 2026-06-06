from typing import Any

import numpy as np
import pandas as pd


def select(context: dict[str, Any],
           *,
           column: str,
           n: int,
           ascending: bool = False) -> list[int | str]:
    '''Return the top n ids from results.csv ordered by one numeric column.'''

    if not isinstance(n, int) or n <= 0:
        raise ValueError('top_n n must be a positive integer')

    results = context.get('results')
    if results is None:
        raise ValueError('top_n selector requires results.csv data in context["results"]')
    if not isinstance(results, pd.DataFrame):
        raise ValueError('top_n context["results"] must be a pandas DataFrame')

    missing = [col for col in ('id', column) if col not in results.columns]
    if missing:
        raise ValueError(f'top_n selector input is missing required columns: {missing}')

    def coerce_id(value: Any) -> int | str:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('top_n selector returned a boolean permutation id')
        if pd.isna(value):
            raise ValueError('top_n selector returned a missing permutation id')
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
            if not stripped:
                raise ValueError('top_n selector returned an empty permutation id')
            return stripped

        coerced = str(value).strip()
        if not coerced:
            raise ValueError('top_n selector returned an empty permutation id')
        return coerced

    work = results[['id', column]].copy()
    work[column] = pd.to_numeric(work[column], errors='coerce')
    work = work.dropna(subset=[column])
    if work.empty:
        return []

    work['_id_sort_key'] = work['id'].map(lambda value: str(coerce_id(value)))
    ranked = work.sort_values(
        by=[column, '_id_sort_key'],
        ascending=[ascending, True],
        kind='mergesort',
    )

    return [coerce_id(value) for value in ranked.head(n)['id'].tolist()]
