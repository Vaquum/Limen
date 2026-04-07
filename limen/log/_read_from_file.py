from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd


def _read_from_file(_self: Any, file_path: str) -> pd.DataFrame:

    '''
    Create cleaned experiment log DataFrame from file.

    Args:
        file_path (str): Path to experiment log CSV file

    Returns:
        pd.DataFrame: Cleaned log data with whitespace-trimmed object columns
    '''

    with Path(file_path).open() as f:

        lines = [
            line
            for i, line in enumerate(f)
            if i == 0 or not line.startswith('recall')
        ]

    data = pd.read_csv(StringIO(''.join(lines)))

    for col in data.columns:
        if pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_string_dtype(data[col]):

            data[col] = data[col].str.strip()

    return data
