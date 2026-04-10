import json
from pathlib import Path
from typing import Any

import numpy as np


class Cohort:

    '''
    Construct a decoder cohort from a completed experiment and selected permutations.

    This initial constructor implementation focuses on source resolution and
    permutation_id validation. Prediction and aggregation behavior are added
    separately.
    '''

    _PROBABILITY_CAPABLE_HINTS = (
        'logreg',
        'tabpfn',
        'random_binary',
        'random forest',
        'random_forest',
        'lightgbm',
        'lgbm',
        'ridge',
        'svm',
        'hmm',
        'garch',
        'lstm',
        'cnn',
    )

    def __init__(self,
                 *,
                 experiment_id: str | None = None,
                 experiment_log_path: str | None = None,
                 permutation_ids: list[int | str] | None = None) -> None:
        '''
        Construct an inference-only Cohort from one experiment source and selected permutations.

        Args:
            experiment_id (str | None): Experiment identifier used to resolve one
                experiment directory containing decoder permutations.
            experiment_log_path (str | None): Explicit path to an experiment
                directory used to reconstruct selected decoders.
            permutation_ids (list[int | str] | None): Specific permutation IDs to
                include. If omitted, all available permutations in the resolved
                experiment are selected.

        Returns:
            None

        '''

        if experiment_id is None and experiment_log_path is None:
            raise ValueError(
                'Provide exactly one of: experiment_id or experiment_log_path.')

        if experiment_id is not None and experiment_log_path is not None:
            raise ValueError('Cohort accepts exactly one experiment source.')

        experiment_dir = (
            self._resolve_experiment_id(experiment_id)
            if experiment_id is not None
            else Path(experiment_log_path).expanduser().resolve(strict=False)
        )

        if not experiment_dir.exists() or not experiment_dir.is_dir():
            raise FileNotFoundError(
                f'Experiment log path is missing or unreadable: {experiment_dir}'
            )

        metadata_path = experiment_dir / 'metadata.json'
        round_data_path = experiment_dir / 'round_data.jsonl'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f'Experiment log path is missing or unreadable: {experiment_dir}'
            )
        if not round_data_path.exists():
            raise FileNotFoundError(
                f'Experiment log path is missing or unreadable: {experiment_dir}'
            )

        with metadata_path.open('r') as f:
            metadata = json.load(f)

        round_entries = self._load_round_entries(round_data_path)
        available_ids = set(round_entries.keys())
        if not available_ids:
            raise ValueError('Resolved experiment contains no permutations.')

        if permutation_ids is None:
            selected_ids = sorted(available_ids)
        else:
            if not permutation_ids:
                raise ValueError(
                    'permutation_ids must be a non-empty list when provided.')

            normalized = [self._normalize_permutation_id(
                pid) for pid in permutation_ids]
            if len(normalized) != len(set(normalized)):
                raise ValueError('permutation_ids must be unique.')

            missing_ids = [
                pid for pid in normalized if pid not in available_ids]
            if missing_ids:
                raise ValueError(
                    f'Unknown permutation_ids requested: {missing_ids}')

            selected_ids = normalized

        architecture_id = self._resolve_cohort_architecture(
            selected_ids,
            round_entries,
            metadata,
        )
        supports_probabilities = self._architecture_supports_probabilities(
            architecture_id,
        )

        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.available_permutation_ids = sorted(available_ids)
        self.permutation_ids = selected_ids
        self.architecture_id = architecture_id
        self.supports_probabilities = supports_probabilities
        self.aggregation_mode = (
            'probability_weighted' if supports_probabilities else 'majority_vote'
        )
        self.metadata = metadata
        self._members: list[Any] = []

    def set_members(self, members: list[Any]) -> None:

        if not members:
            raise ValueError('Cohort members must be a non-empty list.')

        self._members = list(members)

    def predict(self,
                X: Any,
                *,
                return_probs: bool = False,
                return_meta: bool = False) -> np.ndarray | tuple:
        '''
        Run cohort members on input data and return aggregated binary predictions.

        Args:
            X (Any): Input batch to score with all cohort members. Accepts either
                a pre-built model input dict or raw test features.
            return_probs (bool): Whether to also return raw per-member probability
                arrays (P(1)). Only valid in probability-weighted mode.
            return_meta (bool): Whether to also return structured cohort metadata
                (placeholder schema) alongside predictions.

        Returns:
            np.ndarray | tuple: Aggregated binary predictions by default. Optional
                tuple variants are:
                - (predictions, probabilities)
                - (predictions, metadata)
                - (predictions, probabilities, metadata)

        '''

        if return_probs and self.aggregation_mode != 'probability_weighted':
            raise ValueError(
                'Probabilities are unavailable for this Cohort architecture.')

        if not self._members:
            raise RuntimeError(
                'Cohort has no bound decoder members. '
                'Use set_members(...) before predict().'
            )

        member_input = X if isinstance(X, dict) else {'x_test': X}

        # Single-decoder cohort short-circuit: pass through unchanged predictions.
        if len(self._members) == 1:
            result = self._members[0].predict(member_input)
            if not isinstance(result, dict) or '_preds' not in result:
                raise ValueError(
                    'Single-decoder cohort member must return a dict with _preds.'
                )

            preds = np.asarray(result['_preds'])
            probs = None
            if return_probs:
                if '_probs' not in result:
                    raise ValueError(
                        'Decoder in probability mode must return a dict with _probs.'
                    )
                probs = [np.asarray(result['_probs'], dtype=float)]

            return self._format_predict_output(
                preds,
                probs,
                return_probs=return_probs,
                return_meta=return_meta,
            )

        if self.aggregation_mode == 'probability_weighted':
            member_probs: list[np.ndarray] = []

            for member in self._members:
                result = member.predict(member_input)
                if not isinstance(result, dict) or '_probs' not in result:
                    raise ValueError(
                        'Decoder in probability mode must return a dict with _probs.'
                    )

                probs = np.asarray(result['_probs'], dtype=float)
                member_probs.append(probs)

            preds = self._probability_weighted_vote(member_probs)
            return self._format_predict_output(
                preds,
                member_probs if return_probs else None,
                return_probs=return_probs,
                return_meta=return_meta,
            )

        if self.aggregation_mode == 'majority_vote':
            member_preds: list[np.ndarray] = []

            for member in self._members:
                result = member.predict(member_input)
                if not isinstance(result, dict) or '_preds' not in result:
                    raise ValueError(
                        'Decoder in majority_vote mode must return a dict with _preds.'
                    )

                preds = np.asarray(result['_preds'], dtype=float)
                member_preds.append(preds)

            preds = self._majority_vote(member_preds)
            return self._format_predict_output(
                preds,
                None,
                return_probs=return_probs,
                return_meta=return_meta,
            )

        raise ValueError(f'Unknown aggregation_mode: {self.aggregation_mode}')

    @staticmethod
    def _probability_weighted_vote(member_probs: list[np.ndarray]) -> np.ndarray:

        if not member_probs:
            raise ValueError('member_probs must be a non-empty list.')

        base_shape = member_probs[0].shape
        for probs in member_probs[1:]:
            if probs.shape != base_shape:
                raise ValueError('Decoder outputs must share the same shape.')

        probs_matrix = np.vstack(member_probs)
        mean_p1 = np.mean(probs_matrix, axis=0)

        return (mean_p1 > 0.5).astype(np.int8)

    @staticmethod
    def _majority_vote(member_preds: list[np.ndarray]) -> np.ndarray:

        if not member_preds:
            raise ValueError('member_preds must be a non-empty list.')

        base_shape = member_preds[0].shape
        for preds in member_preds[1:]:
            if preds.shape != base_shape:
                raise ValueError('Decoder outputs must share the same shape.')

        preds_matrix = np.vstack(member_preds)
        mean_vote = np.mean(preds_matrix, axis=0)

        return (mean_vote > 0.5).astype(np.int8)

    def _format_predict_output(self,
                               predictions: np.ndarray,
                               probs: list[np.ndarray] | None,
                               *,
                               return_probs: bool,
                               return_meta: bool) -> np.ndarray | tuple:

        if not return_probs and not return_meta:
            return predictions

        out: list[Any] = [predictions]
        if return_probs:
            out.append(probs)
        if return_meta:
            out.append(self._predict_meta())

        return tuple(out)

    def _predict_meta(self) -> dict[str, Any]:

        return {
            'permutation_ids': list(self.permutation_ids),
            'decoder_count': len(self._members),
            'architecture_id': self.architecture_id,
            'aggregation_mode': self.aggregation_mode,
        }

    @staticmethod
    def _normalize_permutation_id(pid: int | str) -> int | str:

        if isinstance(pid, bool):
            raise ValueError(
                'permutation_ids entries must be int or str identifiers.')

        if isinstance(pid, int):
            return pid

        if not isinstance(pid, str):
            raise ValueError(
                'permutation_ids entries must be int or str identifiers.')

        stripped = pid.strip()
        if stripped.isdigit():
            return int(stripped)

        return stripped

    @staticmethod
    def _load_round_entries(round_data_path: Path) -> dict[int | str, dict]:

        entries: dict[int | str, dict] = {}
        with round_data_path.open('r') as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue

                entry = json.loads(stripped)
                if 'round_id' not in entry:
                    continue
                entries[entry['round_id']] = entry

        return entries

    @classmethod
    def _resolve_cohort_architecture(cls,
                                     selected_ids: list[int | str],
                                     round_entries: dict[int | str, dict],
                                     metadata: dict) -> str:

        architecture_ids = {
            cls._extract_architecture_id(round_entries[pid], metadata)
            for pid in selected_ids
        }

        if len(architecture_ids) != 1:
            raise ValueError(
                'All selected permutation_ids must belong to the same architecture.'
            )

        return next(iter(architecture_ids))

    @staticmethod
    def _extract_architecture_id(round_entry: dict, metadata: dict) -> str:

        round_params = round_entry.get('round_params', {})
        for key in (
            'architecture',
            'model_architecture',
            '_architecture',
            'reference_architecture',
            'decoder_architecture',
            'model_name',
        ):
            value = round_params.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        sfd_module = metadata.get('sfd_module')
        if isinstance(sfd_module, str) and sfd_module.strip():
            return sfd_module.strip()

        return 'unknown_architecture'

    @classmethod
    def _architecture_supports_probabilities(cls, architecture_id: str) -> bool:

        normalized = architecture_id.strip().lower()

        if any(hint in normalized for hint in cls._PROBABILITY_CAPABLE_HINTS):
            return True

        return False

    @staticmethod
    def _resolve_experiment_id(experiment_id: str) -> Path:

        direct = Path(experiment_id).expanduser().resolve(strict=False)
        if direct.exists() and direct.is_dir():
            return direct

        cwd = Path.cwd()
        matches: list[Path] = []
        for metadata_path in cwd.rglob('metadata.json'):
            parent = metadata_path.parent
            if parent.name != experiment_id:
                continue
            if (parent / 'round_data.jsonl').exists():
                matches.append(parent.resolve())

        if not matches:
            raise ValueError(
                f'Unable to resolve experiment_id: {experiment_id}')

        unique_matches = sorted(set(matches))
        if len(unique_matches) > 1:
            raise ValueError(
                f'experiment_id resolved to multiple experiment logs: {experiment_id}'
            )

        return unique_matches[0]
