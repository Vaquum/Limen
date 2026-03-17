import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import polars as pl

from limen.experiment.sensor import Sensor

__all__ = ['Trainer']

logger = logging.getLogger(__name__)


class Trainer:

    '''
    Retrain selected permutations from a completed experiment.

    NOTE: Pass 1 implementation — validates permutations by re-running
    manifest.prepare_data() → manifest.run_model(). Returns Sensor
    instances with results but no trained model object. Pass 2
    (full-data retraining with model class) will be added in a later PR.
    '''

    def __init__(self,
                 experiment_dir: str | Path,
                 data: pl.DataFrame | None = None) -> None:

        '''
        Args:
            experiment_dir (str | Path): Path to completed experiment directory
            data (pl.DataFrame | None): Data to use for training. If None,
                fetches from manifest data source config

        '''

        self._experiment_dir = Path(experiment_dir)

        metadata_path = self._experiment_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self._experiment_dir}. "
                f"Only experiments created with experiment_dir support training."
            )

        with metadata_path.open('r') as f:
            self._metadata = json.load(f)

        if 'sfd_module' not in self._metadata:
            raise ValueError(
                'metadata.json missing required key: sfd_module'
            )

        sfd_module_name = self._metadata['sfd_module']
        sfd = importlib.import_module(sfd_module_name)

        self._manifest = sfd.manifest()
        self._params = sfd.params()
        self._round_data = self._load_round_data()

        if data is not None:
            self._data = data
        else:
            env = os.getenv('LOOP_ENV', 'test')
            if env == 'test' and self._manifest.test_data_source_config is not None:
                self._data = self._manifest.fetch_test_data()
            else:
                self._data = self._manifest.fetch_data()

    def _load_round_data(self) -> dict[int, dict[str, Any]]:

        '''
        Load round_data.jsonl into a dict keyed by round_id.

        Returns:
            dict[int, dict[str, Any]]: Mapping of round_id to round entry

        '''

        round_data_path = self._experiment_dir / 'round_data.jsonl'
        if not round_data_path.exists():
            raise FileNotFoundError(
                f"round_data.jsonl not found in {self._experiment_dir}. "
                f"Cannot load permutation parameters."
            )

        result: dict[int, dict[str, Any]] = {}
        with round_data_path.open('r') as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                result[entry['round_id']] = entry

        return result

    def train(self, permutation_ids: list[int]) -> list[Sensor]:

        '''
        Run Pass 1 training for selected permutations.

        Args:
            permutation_ids (list[int]): Round IDs from experiment_log to retrain

        Returns:
            list[Sensor]: Sensor instances with validation results

        Raises:
            ValueError: If any permutation ID is not found in round_data

        '''

        missing = [
            pid for pid in permutation_ids
            if pid not in self._round_data
        ]
        if missing:
            raise ValueError(
                f"Permutation IDs not found in round_data: {missing}"
            )

        sensors: list[Sensor] = []

        for pid in permutation_ids:
            round_params = self._round_data[pid]['round_params']

            data_dict = self._manifest.prepare_data(self._data, round_params)
            results = self._manifest.run_model(data_dict, round_params)

            sensor = Sensor(
                model=None,
                round_params=round_params,
                metadata=self._metadata,
                results=results,
            )
            sensors.append(sensor)

            logger.info('Trained sensor for permutation %d', pid)

        return sensors
