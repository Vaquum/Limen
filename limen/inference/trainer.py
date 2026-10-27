from limen.experiment._target_identity import bind as bind_target_identity
import limen.experiment._holdout_research as _holdout
from limen.inference._trainer_records import load_round_data, load_original_log
from limen.experiment._recording import project_record
import json
from collections.abc import Mapping
from limen.backtest.trade_contract import JsonValue
import logging
import math
from pathlib import Path
from typing import Any, cast

import polars as pl

from limen.inference.errors import ReconstructionError
from limen.inference.sensor import Sensor
import limen.inference._fold_trainer as _fold_replay
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
                 data: pl.DataFrame | None = None, *,
                 project_root: str | Path | None = None) -> None:

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
            sfd = CompiledSFD(yaml_reference, project_root=project_root or self._experiment_dir)
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

        from limen.experiment._dynamic_barriers import reject_promotion
        reject_promotion(self._manifest)
        self._param_keys = frozenset(params.keys())
        self._round_data = self._load_round_data()
        self._original_log = None if self._metadata.get('split_walk_forward') is not None else self._load_original_log()

        self._data = _holdout.initialize_trainer(self, data, project_root=project_root)

    def _load_round_data(self) -> dict[str, dict[str, Any]]:
        return load_round_data(self._experiment_dir)

    def _load_original_log(self) -> dict[str, dict[str, Any]] | None:
        return load_original_log(self._experiment_dir)

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

            values = (original_value, new_value)
            if any(isinstance(value, float) and not math.isfinite(value) for value in values):
                if original_value != new_value and not all(isinstance(value, float) and math.isnan(value) for value in values):
                    mismatches.append(f"{key}: original={original_value}, new={new_value}")
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


    @_holdout.train_guard
    def train(self, permutation_ids: list[str], *, fold: int | None = None) -> list[Sensor]:


        fold_sensors = _fold_replay.train(self, permutation_ids, fold)
        if fold_sensors is not None:
            return fold_sensors

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

            expected = self._round_data[pid]
            _holdout.trainer_round(self._manifest, expected)
            with bind_target_identity(self._manifest, round_params, pid):
                data_dict = self._manifest.prepare_data(self._data, round_params)
                policy, frozen_contract = self._metadata.get('recording'), expected.get('trade_contract')
                if frozen_contract is not None and (project_record(expected, policy).get('trade_contract') != project_record(data_dict, policy).get('trade_contract') or expected.get('trade_contract_digest') != data_dict.get('trade_contract_digest')):
                    raise ReconstructionError(f'Permutation {pid}: exact trade rules/source/output identity changed')
                results = self._manifest.run_model(data_dict, round_params)
            binding = expected.get('learning_binding')
            if binding is not None or '_learning_binding' in data_dict:
                if not isinstance(binding, Mapping):
                    raise ReconstructionError(f'Permutation {pid}: missing frozen learning identity')
                identity = cast(Mapping[str, object], binding)
                if identity.get('round_id') != pid or identity.get('manifest_id') != self._manifest_id or identity.get('model') != data_dict.get('_learning_binding'):
                    raise ReconstructionError(f'Permutation {pid}: exact learning/source/factory identity changed')

            model = results.pop('_model', None)
            if model is None:
                raise ValueError(
                    f"Permutation {pid}: architecture result does not contain '_model'. Trainer requires a reference architecture that returns the trained model via result['_model']."
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
                trade_contract=cast(Mapping[str, JsonValue] | None, data_dict.get('trade_contract') if frozen_contract is not None else None),
            )
            sensors.append(sensor)

            logger.info('Trained sensor for permutation %s', pid)

        return sensors
