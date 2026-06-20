import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from typing import ClassVar

import numpy as np
import pandas as pd
import polars as pl

from limen.cohort.sfc import BUILTIN_SELECTORS
from limen.cohort.sfc import Selector
from limen.inference.sensor import BarPrediction
from limen.inference.sensor import PredictionReason
from limen.yaml.store import _SHA256_PREFIX


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
    _VOTE_THRESHOLD = 0.5
    _REASON_PRIORITY: ClassVar[dict[PredictionReason, int]] = {
        'inside-training-window': 2,
        'null-features': 1,
        'warm-up': 0,
    }

    _logger = logging.getLogger(__name__)

    def __init__(self,
                 *,
                 experiment_id: str | None = None,
                 experiment_log_path: str | None = None,
                 permutation_ids: list[str] | None = None,
                 selector: str | Selector | None = None,
                 selector_params: dict[str, Any] | None = None) -> None:
        '''
        Construct an inference-only Cohort from one experiment source and selected permutations.

        Args:
            experiment_id (str | None): Experiment identifier used to resolve one
                experiment directory containing decoder permutations.
            experiment_log_path (str | None): Explicit path to an experiment
                directory used to reconstruct selected decoders.
            permutation_ids (list[str] | None): Specific permutation IDs to
                include. If omitted, all available permutations in the resolved
                experiment are selected.
            selector (str | Selector | None): Built-in selector name or callable
                using the `select(context, **params) -> list[str]`
                contract. Defaults to `all` when permutation_ids is omitted.
            selector_params (dict[str, Any] | None): Keyword arguments passed to
                the selector callable.

        Returns:
            None

        '''

        if experiment_id is None and experiment_log_path is None:
            raise ValueError(
                'Cohort Provide exactly one of: experiment_id or experiment_log_path.')

        if experiment_id is not None and experiment_log_path is not None:
            raise ValueError('Cohort accepts exactly one experiment source.')

        if permutation_ids is not None and selector is not None:
            raise ValueError('Cohort accepts permutation_ids or selector, not both.')

        if selector_params is not None and not isinstance(selector_params, dict):
            raise ValueError('Cohort selector_params must be a dict when provided.')

        if selector_params is not None and selector is None:
            raise ValueError('Cohort selector_params requires an explicit selector.')

        experiment_dir = (
            self._resolve_experiment_id(experiment_id)
            if experiment_id is not None
            else Path(experiment_log_path).expanduser().resolve(strict=False)
        )

        if not experiment_dir.exists() or not experiment_dir.is_dir():
            raise FileNotFoundError(
                f'Cohort Experiment log path is missing or unreadable: {experiment_dir}'
            )

        metadata_path = experiment_dir / 'metadata.json'
        round_data_path = experiment_dir / 'round_data.jsonl'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f'Cohort Experiment log path is missing or unreadable: {experiment_dir}'
            )
        if not round_data_path.exists():
            raise FileNotFoundError(
                f'Cohort Experiment log path is missing or unreadable: {experiment_dir}'
            )

        with metadata_path.open('r') as f:
            metadata = json.load(f)

        round_entries = self._load_round_entries(round_data_path)
        available_ids = set(round_entries.keys())
        if not available_ids:
            raise ValueError('Cohort Resolved experiment contains no permutations.')

        if permutation_ids is None:
            context = self._build_selector_context(
                experiment_dir,
                metadata,
                round_entries,
                available_ids,
            )
            selected_ids = self._select_permutation_ids(
                context,
                selector,
                selector_params or {},
            )
        else:
            if not permutation_ids:
                raise ValueError(
                    'Cohort permutation_ids must be a non-empty list when provided.')

            normalized = [self._normalize_permutation_id(
                pid) for pid in permutation_ids]
            if len(normalized) != len(set(normalized)):
                raise ValueError('Cohort permutation_ids must be unique.')

            missing_ids = [
                pid for pid in normalized if pid not in available_ids]
            if missing_ids:
                raise ValueError(
                    f'Cohort Unknown permutation_ids requested: {missing_ids}')

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
        self.available_permutation_ids = self._sort_permutation_ids(available_ids)
        self.permutation_ids = selected_ids
        self.architecture_id = architecture_id
        self.supports_probabilities = supports_probabilities
        self.aggregation_mode = (
            'probability_weighted' if supports_probabilities else 'majority_vote'
        )
        self.metadata = metadata
        self.manifest_id: str | None = metadata.get('manifest_id')
        self.cohort_id: str | None = None
        self._members: list[Any] = []

    def set_members(self, members: list[Any]) -> None:

        if not members:
            raise ValueError('Cohort members must be a non-empty list.')

        if len(members) != len(self.permutation_ids):
            raise ValueError(
                'Cohort member count must match selected permutation_ids count.'
            )

        by_pid: dict[str, Any] = {}
        for member in members:
            if not hasattr(member, 'permutation_id'):
                raise ValueError(
                    'Each cohort member must expose permutation_id for binding.'
                )

            pid = self._normalize_permutation_id(member.permutation_id)
            if pid in by_pid:
                raise ValueError(
                    'Cohort members must have unique permutation_id values.')

            by_pid[pid] = member

            if hasattr(member, 'round_params'):
                arch = self._extract_architecture_id(
                    {'round_params': member.round_params},
                    self.metadata,
                )
                if arch.strip().lower() != self.architecture_id.strip().lower():
                    raise ValueError(
                        'Bound members must match cohort architecture_id.'
                    )

            member_mid = getattr(member, 'manifest_id', None)
            if member_mid is not None:
                if self.manifest_id is None:
                    self.manifest_id = member_mid
                elif self.manifest_id != member_mid:
                    raise ValueError(
                        f"All cohort members must share the same manifest_id. "
                        f"Got {member_mid!r}, expected {self.manifest_id!r}."
                    )

        missing = [pid for pid in self.permutation_ids if pid not in by_pid]
        if missing:
            raise ValueError(
                f'Cohort Missing bound members for permutation_ids: {missing}'
            )

        self._members = [by_pid[pid] for pid in self.permutation_ids]

        payload = json.dumps(
            {
                'aggregation_mode': self.aggregation_mode,
                'architecture_id': self.architecture_id,
                'manifest_id': self.manifest_id,
                'permutation_ids': sorted(self.permutation_ids),
            },
            sort_keys=True,
        )
        self.cohort_id = f'{_SHA256_PREFIX}{hashlib.sha256(payload.encode()).hexdigest()}'


    def __call__(self, raw_klines: pl.DataFrame) -> list[BarPrediction]:

        return self.predict_all(raw_klines)


    def _assert_members(self) -> None:

        if not self._members:
            raise RuntimeError(
                'Cohort has no bound members. Use set_members() before predicting.'
            )


    def predict(self, raw_klines: pl.DataFrame) -> BarPrediction:

        '''
        Run all member sensors on raw klines and return one aggregated prediction for the last bar.

        Args:
            raw_klines (pl.DataFrame): Raw klines from live feed, same schema as
                the manifest data source

        Returns:
            BarPrediction: Aggregated prediction for the last bar

        Raises:
            RuntimeError: If no members have been bound via set_members()

        '''

        self._assert_members()
        member_bars = [m.predict(raw_klines) for m in self._members]
        return self._aggregate_bar_predictions(member_bars)


    def predict_all(self, raw_klines: pl.DataFrame) -> list[BarPrediction]:

        '''
        Run all member sensors on raw klines and return one aggregated prediction per bar.

        Args:
            raw_klines (pl.DataFrame): Raw klines from live feed, same schema as
                the manifest data source

        Returns:
            list[BarPrediction]: One aggregated entry per bar in the post-bar-formation data

        Raises:
            RuntimeError: If no members have been bound via set_members()
            ValueError: If member sensors return different-length prediction lists

        '''

        self._assert_members()

        all_bars = [m.predict_all(raw_klines) for m in self._members]

        fully_errored = [all(b.reason == 'sensor-error' for b in bars) for bars in all_bars]
        healthy_lengths = [len(bars) for bars, err in zip(all_bars, fully_errored, strict=True) if not err]

        if not healthy_lengths:
            return [
                BarPrediction(datetime=None, prediction=None, probability=None, reason='sensor-error')
                for _ in range(len(all_bars[0]))
            ]

        n = healthy_lengths[0]
        if any(length != n for length in healthy_lengths[1:]):
            raise ValueError('Cohort Member sensors returned different-length prediction lists.')

        n_errored_members = sum(fully_errored)
        if n_errored_members:
            self._logger.warning(
                'Cohort predict_all: %d of %d member(s) fully errored and were excluded from aggregation.',
                n_errored_members, len(self._members),
            )

        aligned = [
            [BarPrediction(datetime=None, prediction=None, probability=None, reason='sensor-error')
             for _ in range(n)]
            if err else bars
            for bars, err in zip(all_bars, fully_errored, strict=True)
        ]

        return [
            self._aggregate_bar_predictions([aligned[i][t] for i in range(len(self._members))])
            for t in range(n)
        ]


    def _aggregate_bar_predictions(self, member_bars: list[BarPrediction]) -> BarPrediction:

        '''
        Aggregate per-member BarPredictions into one cohort-level BarPrediction.

        Args:
            member_bars (list[BarPrediction]): One prediction per bound member,
                all sharing the same datetime

        Returns:
            BarPrediction: Aggregated prediction for the bar

        Raises:
            ValueError: If member_bars is empty, or if any member has a None
                probability in probability_weighted mode, or a None prediction
                in majority_vote mode

        '''

        if not member_bars:
            raise ValueError('Cohort member_bars must be a non-empty list.')

        dt = next((b.datetime for b in member_bars if b.datetime is not None), None)

        valid_bars = [b for b in member_bars if b.reason != 'sensor-error']
        if not valid_bars:
            return BarPrediction(datetime=dt, prediction=None, probability=None, reason='sensor-error')

        if len(valid_bars) < len(member_bars):
            n_errors = len(member_bars) - len(valid_bars)
            self._logger.debug(
                'Cohort aggregation: %d of %d member(s) returned sensor-error and were excluded.',
                n_errors, len(member_bars),
            )

        blocking = [b for b in valid_bars if b.reason is not None]
        if blocking:
            worst = max(blocking, key=lambda b: self._REASON_PRIORITY.get(b.reason or '', -1))
            return BarPrediction(datetime=dt, prediction=None, probability=None, reason=worst.reason)

        if self.aggregation_mode == 'probability_weighted':
            if any(b.probability is None for b in valid_bars):
                raise ValueError(
                    'Cohort probability_weighted mode requires a finite probability from all members.'
                )
            probs = np.array([b.probability for b in valid_bars], dtype=float)
            self._validate_probability_range(probs)
            mean_prob = float(np.mean(probs))
            pred = int(mean_prob > self._VOTE_THRESHOLD)
            return BarPrediction(datetime=dt, prediction=pred, probability=mean_prob, reason=None)

        if any(b.prediction is None for b in valid_bars):
            raise ValueError(
                'Cohort majority_vote mode requires a numeric prediction from all members.'
            )
        threshold = self._fallback_vote_threshold()
        votes = np.array([int(float(b.prediction) > threshold) for b in valid_bars])
        pred = int(np.mean(votes) > self._VOTE_THRESHOLD)
        return BarPrediction(datetime=dt, prediction=pred, probability=None, reason=None)


    @staticmethod
    def _validate_probability_range(probs: np.ndarray) -> None:

        if not np.isfinite(probs).all():
            raise ValueError(
                'Cohort Decoder probabilities must be finite values in [0, 1].')

        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError('Cohort Decoder probabilities must lie within [0, 1].')


    def _fallback_vote_threshold(self) -> float:

        arch = self.architecture_id.strip().lower()
        if 'regressor' in arch:
            return 0.0
        return Cohort._VOTE_THRESHOLD


    @staticmethod
    def _normalize_permutation_id(pid: str) -> str:

        if not isinstance(pid, str):
            raise ValueError('Cohort permutation_ids entries must be str identifiers.')

        return pid.strip()


    @staticmethod
    def _sort_permutation_ids(values: set[str]) -> list[str]:

        return sorted(values)

    @classmethod
    def _select_permutation_ids(cls,
                                context: dict[str, Any],
                                selector: str | Selector | None,
                                selector_params: dict[str, Any]) -> list[str]:

        resolved_selector = cls._resolve_selector(selector)
        try:
            selected = resolved_selector(context, **selector_params)
        except TypeError as e:
            raise ValueError(
                f'Cohort selector could not be called with selector_params: {e}'
            ) from e

        if not isinstance(selected, list):
            raise ValueError('Cohort selector must return a list of permutation ids.')

        if not selected:
            raise ValueError('Cohort selector returned no permutation ids.')

        normalized = [cls._normalize_permutation_id(pid) for pid in selected]
        if len(normalized) != len(set(normalized)):
            raise ValueError('Cohort selector returned duplicate permutation ids.')

        available_ids = set(context['available_permutation_ids'])
        missing_ids = [pid for pid in normalized if pid not in available_ids]
        if missing_ids:
            raise ValueError(
                f'Cohort selector returned unknown permutation ids: {missing_ids}'
            )

        return normalized

    @staticmethod
    def _resolve_selector(selector: str | Selector | None) -> Selector:

        if selector is None:
            return BUILTIN_SELECTORS['all']

        if isinstance(selector, str):
            if selector not in BUILTIN_SELECTORS:
                known = sorted(BUILTIN_SELECTORS)
                raise ValueError(
                    f'Unknown Cohort selector: {selector}. Known selectors: {known}'
                )
            return BUILTIN_SELECTORS[selector]

        if callable(selector):
            return selector

        raise ValueError('Cohort selector must be None, a built-in selector name, or a callable.')

    @staticmethod
    def _build_selector_context(experiment_dir: Path,
                                metadata: dict,
                                round_entries: dict[str, dict],
                                available_ids: set[str]) -> dict[str, Any]:

        context: dict[str, Any] = {
            'experiment_dir': experiment_dir,
            'metadata': metadata,
            'round_entries': round_entries,
            'available_permutation_ids': Cohort._sort_permutation_ids(available_ids),
        }

        results_path = experiment_dir / 'results.csv'
        if results_path.exists():
            context['results'] = pd.read_csv(results_path)

        return context

    @staticmethod
    def _load_round_entries(round_data_path: Path) -> dict[str, dict]:

        entries: dict[str, dict] = {}
        with round_data_path.open('r') as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue

                entry = json.loads(stripped)
                if 'round_id' not in entry:
                    continue
                entries[str(entry['round_id'])] = entry

        return entries

    @classmethod
    def _resolve_cohort_architecture(cls,
                                     selected_ids: list[str],
                                     round_entries: dict[str, dict],
                                     metadata: dict) -> str:

        architecture_ids = {
            cls._extract_architecture_id(round_entries[pid], metadata)
            for pid in selected_ids
        }

        if len(architecture_ids) != 1:
            raise ValueError(
                'Cohort All selected permutation_ids must belong to the same architecture.'
            )

        return next(iter(architecture_ids))

    @classmethod
    def _extract_architecture_id(cls, round_entry: dict, metadata: dict) -> str:

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

        yaml_architecture = cls._extract_yaml_reference_architecture(metadata)
        if yaml_architecture is not None:
            return yaml_architecture

        sfd_module = metadata.get('sfd_module')
        if isinstance(sfd_module, str) and sfd_module.strip():
            return sfd_module.strip()

        return 'unknown_architecture'

    @staticmethod
    def _extract_yaml_reference_architecture(metadata: dict) -> str | None:

        yaml_reference = metadata.get('yaml_reference')
        if not isinstance(yaml_reference, dict):
            return None

        sfd_config = yaml_reference.get('sfd')
        if not isinstance(sfd_config, dict):
            return None

        manifest_config = sfd_config.get('manifest')
        if not isinstance(manifest_config, dict):
            return None

        reference_architecture = manifest_config.get('reference_architecture')
        if isinstance(reference_architecture, str) and reference_architecture.strip():
            return reference_architecture.strip()

        return None

    @classmethod
    def _architecture_supports_probabilities(cls, architecture_id: str) -> bool:

        normalized = architecture_id.strip().lower()

        return any(hint in normalized for hint in cls._PROBABILITY_CAPABLE_HINTS)

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
                f'Cohort Unable to resolve experiment_id: {experiment_id}')

        unique_matches = sorted(set(matches))
        if len(unique_matches) > 1:
            raise ValueError(
                f'Cohort experiment_id resolved to multiple experiment logs: {experiment_id}'
            )

        return unique_matches[0]
