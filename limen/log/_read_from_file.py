import pandas as pd
from pathlib import Path


from typing import Any


def _read_from_file(_self: Any, file_path: str) -> pd.DataFrame:

    '''
    Create cleaned experiment log DataFrame from file.

    Args:
        file_path (str): Path to experiment log CSV file

    Returns:
        pd.DataFrame: Cleaned log data with whitespace-trimmed object columns
    '''

    with Path(file_path).open() as f:

        lines = f.readlines()

        for i, line in enumerate(lines):

            if i != 0 and line.startswith('recall'):
                lines.pop(i)

    with Path('__temp__.csv').open('w') as f:

        f.writelines(lines)

    data = pd.read_csv('__temp__.csv')

    for col in data.columns:
        if data[col].dtype == object:

            mask = data[col].notnull()
            data[col] = data[col].mask(mask, data[col].astype(str).str.strip())

    return data
