import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

from limen.experiment.trainer.errors import ReconstructionError
from limen.experiment.trainer.sensor import Sensor
from limen.sfd.reference_architecture.base import ReferenceModel

logger = logging.getLogger(__name__)

_SKIP_COLUMNS = frozenset({
    'id', '_id', 'execution_time', '_warnings',
})

_FLOAT_TOLERANCE = 1e-6
_STOCHASTIC_TOLERANCE = 0.01


class Trainer:

    '''
    Retrain selected permutations from a completed experiment.

    Pass 1 validates permutations by re-running manifest.prepare_data()
    and manifest.run_model(), comparing metrics against the original
    experiment log. Pass 2 retrains on the full dataset with
    split_config=(1,0,0) using the model class directly. Returns Sensor
    instances wrapping trained ReferenceModel objects.
    '''

    def __init__(self,
                 experiment_dir: str | Path,
                 data: pl.DataFrame | None = None) -> None:

        '''
        Create a Trainer from a completed experiment directory.

        NOTE: experiment_dir must be trusted. The SFD module path stored
        in metadata.json is imported via importlib.import_module(), which
        executes arbitrary code from that module.

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

        if not hasattr(sfd, 'manifest') or not hasattr(sfd, 'params'):
            raise ValueError(
                f"SFD module '{sfd_module_name}' does not have manifest() and "
                f"params(). Trainer requires a manifest-based SFD."
            )

        self._manifest = sfd.manifest()
        params = sfd.params()
        self._param_keys = frozenset(params.keys())
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
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning(
                        'Skipping malformed line in round_data.jsonl: %s',
                        stripped[:80],
                    )
                    continue
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


    def _resolve_model_class(self) -> type[ReferenceModel]:

        '''
        Resolve the ReferenceModel subclass from the model function's module.

        Returns:
            type[ReferenceModel]: The model class

        Raises:
            ValueError: If zero or multiple ReferenceModel subclasses found

        '''

        if self._manifest.model_function is None:
            raise ValueError(
                'Model function not configured on manifest. '
                'Cannot resolve model class.'
            )

        module_name = self._manifest.model_function.__module__
        module = importlib.import_module(module_name)

        classes = [
            cls for _, cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(cls, ReferenceModel) and cls is not ReferenceModel
        ]

        if len(classes) == 0:
            raise ValueError(
                f"No ReferenceModel subclass found in '{module_name}'"
            )
        if len(classes) > 1:
            raise ValueError(
                f"Multiple ReferenceModel subclasses found in '{module_name}': "
                f"{[c.__name__ for c in classes]}"
            )

        return classes[0]


    def _validate_metrics(self,
                          permutation_id: int,
                          results: dict[str, Any],
                          is_deterministic: bool) -> list[str]:

        '''
        Compare Pass 1 results against original experiment log entry.

        Args:
            permutation_id (int): Round ID to validate
            results (dict[str, Any]): Pass 1 results from run_model
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
                denom = max(abs(float(original_value)), 1e-9)
                if diff / denom > _STOCHASTIC_TOLERANCE:
                    mismatches.append(
                        f"{key}: original={original_value}, new={new_value}"
                    )

        return mismatches


    def train(self, permutation_ids: list[int]) -> list[Sensor]:

        '''
        Run 2-pass training for selected permutations.

        Pass 1 re-runs the pipeline and compares metrics against the
        original experiment log. Raises ReconstructionError on mismatch.
        Pass 2 retrains on the full dataset with split_config=(1,0,0)
        using the model class directly.

        Args:
            permutation_ids (list[int]): Round IDs from experiment_log to retrain

        Returns:
            list[Sensor]: Sensor instances wrapping trained models

        Raises:
            ValueError: If any permutation ID is not found in round_data
            ReconstructionError: If Pass 1 metrics deviate beyond tolerance

        '''

        missing = [
            pid for pid in permutation_ids
            if pid not in self._round_data
        ]
        if missing:
            raise ValueError(
                f"Permutation IDs not found in round_data: {missing}"
            )

        model_class = self._resolve_model_class()
        manifest_full = self._manifest.with_params_override(split_config=(1, 0, 0))
        sensors: list[Sensor] = []

        for pid in permutation_ids:
            round_params = dict(self._round_data[pid]['round_params'])

            # Pass 1: validate with original split
            data_dict = self._manifest.prepare_data(self._data, round_params)
            results = self._manifest.run_model(data_dict, round_params)

            mismatches = self._validate_metrics(pid, results, model_class.deterministic)
            if mismatches:
                raise ReconstructionError(
                    f"Permutation {pid}: metric mismatch detected — "
                    + '; '.join(mismatches)
                )

            # Pass 2: retrain on full data
            full_data = manifest_full.prepare_data(self._data, round_params)
            model_kwargs = self._manifest.resolve_model_kwargs(round_params)
            trained_model = model_class().train(full_data, **model_kwargs)

            sensor = Sensor(
                model=trained_model,
                permutation_id=pid,
                round_params=round_params,
                metadata=self._metadata,
                results=results,
            )
            sensors.append(sensor)

            logger.info('Trained sensor for permutation %d', pid)

        return sensors
