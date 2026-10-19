import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import polars as pl
import limen.experiment._round_storage as _storage

logger = logging.getLogger(__name__)


def load_round_data(directory: Path) -> dict[str, dict[str, object]]:
    path = _storage.round_path(directory)
    result: dict[str, dict[str, object]] = {}
    if not path.exists():
        raise FileNotFoundError(f'round_data.jsonl not found in {directory}. Cannot load permutation parameters.')
    with _storage.open_text(path) as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                value: object = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Malformed JSON in round_data.jsonl line {line_number}') from exc
            if not isinstance(value, Mapping):
                raise ValueError(f'Invalid round_data.jsonl line {line_number}: expected object')
            entry = dict(cast(Mapping[str, object], value))
            if not {'round_id', 'round_params'} <= set(entry):
                raise ValueError(f'Invalid round_data.jsonl line {line_number}: requires round_id and round_params')
            if not isinstance(entry['round_params'], dict):
                raise ValueError(f'Invalid round_data.jsonl line {line_number}: round_params must be an object')
            result[str(entry['round_id'])] = entry
    return result


def load_original_log(directory: Path) -> dict[str, dict[str, object]] | None:
    path = directory / 'results.csv'
    try:
        frame = pl.read_csv(path)
    except FileNotFoundError:
        logger.warning('results.csv not found in %s — skipping validation', directory)
        return None
    return {str(row['id']): dict(cast(Mapping[str, object], row)) for row in frame.iter_rows(named=True)}
