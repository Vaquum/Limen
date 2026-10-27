from io import StringIO
from pathlib import Path
from typing import Any

import polars as pl


def read_from_file(_self: Any, file_path: str) -> pl.DataFrame:

    '''
    Create cleaned experiment log DataFrame from file.

    Args:
        file_path (str): Path to experiment log CSV file

    Returns:
        pl.DataFrame: Cleaned log data with whitespace-trimmed string columns
    '''

    with Path(file_path).open() as f:

        header_line = f.readline()

        if header_line:
            header = header_line.strip()
            lines = [
                header_line,
                *[
                    line
                    for line in f
                    if line.strip() != header
                ],
            ]
        else:
            lines = []

    data = pl.read_csv(StringIO(''.join(lines)), infer_schema_length=None,
                       null_values=['#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
                                    '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>',
                                    'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null'])
    return data.with_columns(pl.col(pl.String).str.strip_chars())
