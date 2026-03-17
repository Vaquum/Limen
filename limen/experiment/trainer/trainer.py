import importlib
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

from limen.experiment.trainer.sensor import Sensor

logger = logging.getLogger(__name__)

_SKIP_COLUMNS = frozenset({
    'id', '_id', 'execution_time', '_warnings',
})

_FLOAT_TOLERANCE = 1e-6


class Trainer:

    '''
    Retrain selected permutations from a completed experiment.

    NOTE: Pass 1 implementation — validates permutations by re-running
    manifest.prepare_data() → manifest.run_model(). Compares results
    against the original experiment log to detect pipeline drift.
    Returns Sensor instances with results but no trained model object.
    Pass 2 (full-data retraining with model class) will be added later
    '''

    def __init__(self,
                 experiment_dir: str | Path,
                 data: pl.DataFrame | None = None) -> None:

        '''
        Create a Trainer from a completed experiment directory.

        Args:
            experiment_dir (str | Path): Path to completed experiment directory
            data (pl.DataFrame | None): Data to use for training. If None,
                fetches from manifest data source config

        '''

        self._experiment_dir = Path(experiment_dir)

        metadata_path = self._experiment_dir / 'metadata.json'
        try:
            with metadata_path.open('r') as f:
                self._metadata = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"metadata.json not found in {self._experiment_dir}. "
                f"Only experiments created with experiment_dir support training."
            ) from None

        if 'sfd_module' not in self._metadata:
            raise ValueError(
                'metadata.json missing required key: sfd_module'
            )

        sfd_module_name = self._metadata['sfd_module']
        sfd = importlib.import_module(sfd_module_name)

        self._manifest = sfd.manifest()
        self._params = sfd.params()
        self._param_keys = frozenset(self._params.keys())
        self._round_data = self._load_round_data()
        self._original_log = self._load_original_log()

        if data is not None:
            self._data = data
        else:
            self._data = self._manifest.fetch_data_for_env()


    def _load_round_data(self) -> dict[int, dict[str, Any]]:

        '''
        Load round_data.jsonl into a dict keyed by round_id.

        Returns:
            dict[int, dict[str, Any]]: Mapping of round_id to round entry

        '''

        round_data_path = self._experiment_dir / 'round_data.jsonl'

        result: dict[int, dict[str, Any]] = {}
        try:
            f = round_data_path.open('r')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"round_data.jsonl not found in {self._experiment_dir}. "
                f"Cannot load permutation parameters."
            ) from None

        with f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                result[entry['round_id']] = entry

        return result


    def _load_original_log(self) -> dict[int, dict[str, Any]] | None:

        '''
        Load results.csv into a dict keyed by permutation id.

        Returns:
            dict[int, dict[str, Any]] | None: Mapping of id to row, or None if not found

        '''

        csv_path = self._experiment_dir / 'results.csv'
        try:
            df = pl.read_csv(csv_path)
        except FileNotFoundError:
            logger.warning(
                'results.csv not found in %s — skipping validation',
                self._experiment_dir,
            )
            return None

        result: dict[int, dict[str, Any]] = {}
        for row in df.iter_rows(named=True):
            result[row['id']] = row
        return result


    def _validate_metrics(self,
                          permutation_id: int,
                          results: dict[str, Any]) -> list[str]:

        '''
        Compare Pass 1 results against original experiment log entry.

        Args:
            permutation_id (int): Round ID to validate
            results (dict[str, Any]): Pass 1 results from run_model

        Returns:
            list[str]: List of mismatch descriptions, empty if all match

        '''

        if self._original_log is None:
            return []

        original = self._original_log.get(permutation_id)
        if original is None:
            return [f"permutation {permutation_id} not found in results.csv"]
        mismatches: list[str] = []

        for key, new_value in results.items():
            if key.startswith('_') or key in _SKIP_COLUMNS or key in self._param_keys:
                continue
            if key not in original:
                continue

            original_value = original[key]

            if not isinstance(new_value, (int, float)):
                continue
            if original_value is None:
                continue

            if isinstance(new_value, float):
                if abs(new_value - original_value) > _FLOAT_TOLERANCE:
                    mismatches.append(
                        f"{key}: original={original_value}, new={new_value}"
                    )
            elif new_value != original_value:
                mismatches.append(
                    f"{key}: original={original_value}, new={new_value}"
                )

        return mismatches


    def train(self, permutation_ids: list[int]) -> list[Sensor]:

        '''
        Run Pass 1 validation for selected permutations.

        Re-runs the pipeline and compares metrics against the original
        experiment log. Logs warnings for any metric mismatches.

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
            round_params = dict(self._round_data[pid]['round_params'])

            data_dict = self._manifest.prepare_data(self._data, round_params)
            results = self._manifest.run_model(data_dict, round_params)

            mismatches = self._validate_metrics(pid, results)
            if mismatches:
                logger.warning(
                    'Permutation %d: metric mismatch detected — %s',
                    pid, '; '.join(mismatches),
                )

            sensor = Sensor(
                model=None,
                round_params=round_params,
                metadata=self._metadata,
                results=results,
            )
            sensors.append(sensor)

            logger.info('Validated sensor for permutation %d', pid)

        return sensors
