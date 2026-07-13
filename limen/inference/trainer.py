import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

from limen.inference.errors import ReconstructionError
from limen.inference.sensor import Sensor
from limen.yaml.compiler import CompiledSFD
from limen.yaml.config import is_mapping
from limen.yaml.errors import ResolutionError

logger = logging.getLogger(__name__)

_SKIP_COLUMNS = frozenset({
    'id', '_id', '_round_index', 'execution_time', '_warnings',
})

_FLOAT_TOLERANCE = 1e-6
_STOCHASTIC_TOLERANCE = 0.01


class Trainer:

    '''
    Retrain selected permutations from a completed YAML experiment.

    Validates permutations by re-running manifest.prepare_data() and
    manifest.run_model(), comparing metrics against the original experiment
    log. The validated model is used directly as the Sensor model.
    '''

    def __init__(self,
                 experiment_dir: str | Path,
                 data: pl.DataFrame | None = None) -> None:

        '''
        Create a Trainer from a completed YAML experiment directory.

        Args:
            experiment_dir (str | Path): Path to completed experiment directory
            data (pl.DataFrame | None): Data to use for training. If None,
                fetches from manifest data source config

        Raises:
            FileNotFoundError: If metadata.json is not found
            ValueError: If metadata.json does not contain a valid yaml_reference

        '''

        super().__init__()

        self._experiment_dir = Path(experiment_dir)

        metadata_path = self._experiment_dir / 'metadata.json'
        try:
            with metadata_path.open('r') as f:
                self._metadata = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"metadata.json not found in {self._experiment_dir}. Only experiments created with experiment_dir support training."
            ) from None

        yaml_reference = self._metadata.get('yaml_reference')
        if yaml_reference is None:
            raise ValueError(
                'metadata.json missing required key: yaml_reference. Trainer requires a YAML-based experiment.'
            )
        if not is_mapping(yaml_reference):
            raise ValueError(
                'metadata.json key \'yaml_reference\' must be an object'
            )
        self._yaml_reference = yaml_reference
        self._manifest_id: str | None = self._metadata.get('manifest_id')

        try:
            sfd = CompiledSFD(yaml_reference)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "metadata.json key 'yaml_reference' is not a valid YAML SFD reference"
            ) from exc

        if not hasattr(sfd, 'manifest') or not hasattr(sfd, 'params'):
            raise ValueError(
                'yaml_reference does not produce a manifest-based SFD. Trainer requires a manifest-based SFD.'
            )

        try:
            self._manifest = sfd.manifest()
            params = sfd.params()
        except (KeyError, ResolutionError, TypeError, ValueError) as exc:
            raise ValueError(
                "metadata.json key 'yaml_reference' is not a valid YAML SFD reference"
            ) from exc

        self._param_keys = frozenset(params.keys())
        self._round_data = self._load_round_data()
        self._original_log = self._load_original_log()

        if data is not None:
            self._data = data
        else:
            self._data = self._manifest.fetch_data()


    def _load_round_data(self) -> dict[str, dict[str, Any]]:

        '''
        Load round_data.jsonl into a dict keyed by round_id.

        Returns:
            dict[str, dict[str, Any]]: Mapping of round_id to round entry

        '''

        round_data_path = self._experiment_dir / 'round_data.jsonl'

        result: dict[str, dict[str, Any]] = {}
        try:
            f = round_data_path.open('r')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"round_data.jsonl not found in {self._experiment_dir}. Cannot load permutation parameters."
            ) from None

        with f:
            for line_number, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in round_data.jsonl line {line_number}"
                    ) from exc
                if not is_mapping(entry):
                    raise ValueError(
                        f"Invalid round_data.jsonl line {line_number}: expected object"
                    )
                if 'round_id' not in entry or 'round_params' not in entry:
                    raise ValueError(
                        f"Invalid round_data.jsonl line {line_number}: requires round_id and round_params"
                    )
                if not isinstance(entry['round_params'], dict):
                    raise ValueError(
                        f"Invalid round_data.jsonl line {line_number}: round_params must be an object"
                    )
                result[str(entry['round_id'])] = entry

        return result


    def _load_original_log(self) -> dict[str, dict[str, Any]] | None:

        '''
        Load results.csv into a dict keyed by permutation id.

        Returns:
            dict[str, dict[str, Any]] | None: Mapping of id to row, or None if not found

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

        result: dict[str, dict[str, Any]] = {}
        for row in df.iter_rows(named=True):
            result[str(row['id'])] = row
        return result


    def _validate_metrics(self,
                          permutation_id: str,
                          results: dict[str, Any],
                          is_deterministic: bool) -> list[str]:

        '''
        Compare retrained results against original experiment log entry.

        Args:
            permutation_id (str): Round ID to validate
            results (dict[str, Any]): Results from run_model
            is_deterministic (bool): Whether to use exact-match tolerance

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
            if not isinstance(original_value, (int, float)):
                continue

            if is_deterministic:
                if isinstance(new_value, float) or isinstance(original_value, float):
                    if abs(float(new_value) - float(original_value)) > _FLOAT_TOLERANCE:
                        mismatches.append(
                            f"{key}: original={original_value}, new={new_value}"
                        )
                elif new_value != original_value:
                    mismatches.append(
                        f"{key}: original={original_value}, new={new_value}"
                    )
            else:
                diff = abs(float(new_value) - float(original_value))
                scale = max(abs(float(original_value)), abs(float(new_value)), 1.0)
                if diff > _FLOAT_TOLERANCE and diff / scale > _STOCHASTIC_TOLERANCE:
                    mismatches.append(
                        f"{key}: original={original_value}, new={new_value}"
                    )

        return mismatches


    def train(self, permutation_ids: list[str]) -> list[Sensor]:

        '''
        Retrain selected permutations and return Sensor instances.

        Re-runs the pipeline and compares metrics against the original experiment
        log. Raises ReconstructionError on mismatch. The validated model is used
        directly as the Sensor model.

        Args:
            permutation_ids (list[str]): Round IDs from experiment_log to retrain

        Returns:
            list[Sensor]: Sensor instances wrapping validated models

        Raises:
            ValueError: If any permutation ID is not found in round_data
            ReconstructionError: If metrics deviate beyond tolerance

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

            model = results.pop('_model', None)
            if model is None:
                raise ValueError(
                    f"Permutation {pid}: architecture result does not contain '_model'. Trainer requires a reference architecture that returns the trained model via result['_model']. Rule-based architectures are not supported."
                )
            fitted_params = data_dict.pop('_fitted_params', {})

            mismatches = self._validate_metrics(pid, results, model.deterministic)
            if mismatches:
                raise ReconstructionError(
                    f"Permutation {pid}: metric mismatch detected — "
                    + '; '.join(mismatches)
                    + f" (round_params: {json.dumps(round_params, sort_keys=True, default=str)})"
                )

            sensor = Sensor(
                yaml_reference=self._yaml_reference,
                model=model,
                fitted_params=fitted_params,
                round_params=round_params,
                permutation_id=pid,
                manifest_id=self._manifest_id,
            )
            sensors.append(sensor)

            logger.info('Trained sensor for permutation %s', pid)

        return sensors
