import limen.experiment._round_storage as _storage
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.cohort.sfc import BUILTIN_SELECTORS, Selector
from limen.cohort._round_sources import RoundSources, architecture_identity, same_architecture
from limen.cohort._fold_sources import FoldSources, bind_native_output, mixed_outputs, select_sources
from limen.inference._fold_sensor import reject_legacy_members
from limen.inference.sensor import BarPrediction, PredictionReason, Sensor
from limen.cohort._python_cohort import Aggregate, CohortMap, CohortPrepare, HistoryT, PythonCohortBase, cohort_fields
from limen.yaml.config import is_mapping
from limen.yaml.store import SHA256_PREFIX


class Cohort:

    '''
    Combine explicitly selected, reconstructed sensors for inference.
    '''

    _PROBABILITY_CAPABLE_HINTS = ('logreg', 'tabpfn', 'random_binary', 'random forest', 'random_forest', 'lightgbm',
                                'lgbm', 'svm', 'hmm', 'garch', 'lstm', 'cnn',
                                )
    _VOTE_THRESHOLD = 0.5
    _REASON_PRIORITY: ClassVar[dict[PredictionReason, int]] = {
        'inside-training-window': 2,
        'null-features': 1,
        'warm-up': 0,
    }

    _logger = logging.getLogger(__name__)
    member_ids: list[tuple[str | None, str] | tuple[str, str, int, str]]
    architecture_id: str
    supports_probabilities: bool | None
    aggregation_mode: str | None

    @staticmethod
    def from_python(sensors: Sequence[Sensor], *, aggregate: Aggregate = 'mean', prepare: CohortPrepare[HistoryT] | None = None, map_output: CohortMap | None = None, mapped_output_id: str | None = None, state_id: str | None = None, selection_cutoff: datetime | None = None) -> '_PythonCohort[HistoryT]':
        members, identity, cutoff = cohort_fields(sensors, aggregate, prepare, map_output, mapped_output_id, state_id, selection_cutoff)
        return _PythonCohort(members, identity, cutoff, aggregate, prepare, map_output)

    def __init__(self,
                 *,
                 experiment_id: str | None = None,
                 experiment_log_path: str | None = None,
                 permutation_ids: list[str] | None = None,
                 selector: str | Selector | None = None,
                 selector_params: Any = None,
                 rounds: list[tuple[str | Path, str] | tuple[str | Path, str, int]] | None = None) -> None:
        '''Select named rounds across sources, or use the existing single-source selectors.'''

        super().__init__()

        self._round_sources: RoundSources[Cohort] | FoldSources | None = None
        if rounds is not None:
            if any(arg is not None for arg in (experiment_id, experiment_log_path, permutation_ids, selector, selector_params)):
                raise ValueError('Cohort rounds conflicts with single-source or selector arguments.')
            selected: RoundSources[Cohort] | FoldSources = select_sources(rounds, self, lambda plain: RoundSources[Cohort](
                plain, lambda path, ids: type(self)(experiment_log_path=str(path), permutation_ids=ids),
                self._load_round_entries, self._extract_architecture_id, self._normalize_permutation_id),
                self._normalize_permutation_id, self._extract_architecture_id, self._load_round_entries)
            source = next(iter(selected.sources.values()))
            single = len(selected.sources) == 1
            self.experiment_dir = source.experiment_dir if single else None
            self.experiment_id = source.experiment_id if single else None
            self.available_permutation_ids = source.available_permutation_ids if single else None
            self.permutation_ids = source.permutation_ids if single else None
            self.metadata = source.metadata if single else None
            self.manifest_id = selected.manifest_id
            self.architecture_id = selected.architecture_id
            self.supports_probabilities = source.supports_probabilities
            self.aggregation_mode = source.aggregation_mode
            self.member_ids = list(selected.member_ids)
            self.source_metadata = selected.source_metadata
            self.cohort_id = None
            self._members = []
            self._round_sources = selected
            if self.architecture_id.startswith('mixed-binary-v1:'):
                self.supports_probabilities = self.aggregation_mode = None
            return

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
            else Path(cast(str, experiment_log_path)).expanduser().resolve(strict=False)
        )

        if not experiment_dir.exists() or not experiment_dir.is_dir():
            raise FileNotFoundError(
                f'Cohort Experiment log path is missing or unreadable: {experiment_dir}'
            )

        metadata_path = experiment_dir / 'metadata.json'
        round_data_path = _storage.round_path(experiment_dir)
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
        if metadata.get('split_walk_forward') is not None:
            raise ValueError('Cohort walk-forward rounds cannot be promoted.')

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
            resolved_selector_params = cast(dict[str, Any], selector_params or {})
            selected_ids = self._select_permutation_ids(
                context,
                selector,
                resolved_selector_params,
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
        self.member_ids = [(self.manifest_id, pid) for pid in selected_ids]
        self.source_metadata = {experiment_dir: metadata}
        self._selected_architectures = {pid: self._extract_architecture_id(round_entries[pid], metadata) for pid in selected_ids}
        if self.architecture_id.startswith('mixed-binary-v1:'):
            if not isinstance(self.manifest_id, str) or not self.manifest_id.strip():
                raise ValueError('Mixed Cohort requires a known selected manifest_id')
            self.supports_probabilities = self.aggregation_mode = None

    def set_members(self, members: list[Any]) -> None:

        if not members:
            raise ValueError('Cohort members must be a non-empty list.')
        mixed = self.architecture_id.startswith('mixed-binary-v1:')
        reject_legacy_members(members, qualified_folds=getattr(self._round_sources, 'qualified_folds', False) is True)

        if self._round_sources is not None:
            bound = self._round_sources.bind(members)
            if not isinstance(self._round_sources, FoldSources):
                bind_native_output(self, bound, self._architecture_supports_probabilities(self.architecture_id), self._round_sources.architectures)
            assert self.aggregation_mode is not None
            payload = json.dumps(self._round_sources.identity(self.aggregation_mode) | ({'binary_contracts': vars(self)['_mixed_policy']} if mixed else {}), sort_keys=True)
            self._members = bound
            self.cohort_id = f'{SHA256_PREFIX}{hashlib.sha256(payload.encode()).hexdigest()}'
            return

        assert self.permutation_ids is not None
        assert self.metadata is not None

        if len(members) != len(self.permutation_ids):
            raise ValueError('Cohort member count must match selected permutation_ids count.')

        by_pid: dict[str, Any] = {}
        for member in members:
            if getattr(member, 'prediction_mode', 'binary') == 'target_exposure':
                raise ValueError('Binary Cohort cannot aggregate target-exposure members; use the signed Sensor directly')
            if not hasattr(member, 'permutation_id'):
                raise ValueError('Each cohort member must expose permutation_id for binding.')

            pid = self._normalize_permutation_id(member.permutation_id)
            if mixed and (pid not in self._selected_architectures or getattr(member, 'manifest_id', None) != self.manifest_id or not hasattr(member, 'round_params')):
                raise ValueError('Mixed Cohort requires each selected member manifest_id and architecture')
            if pid in by_pid:
                raise ValueError('Cohort members must have unique permutation_id values.')

            by_pid[pid] = member

            if hasattr(member, 'round_params'):
                arch = self._extract_architecture_id(
                    {'round_params': member.round_params},
                    {'yaml_reference': getattr(member, '_yaml_reference', None)} if mixed else self.metadata,
                )
                if not same_architecture(arch, self._selected_architectures[pid] if mixed else self.architecture_id):
                    raise ValueError('Bound members must match cohort architecture_id.')

            member_mid = getattr(member, 'manifest_id', None)
            if member_mid is not None:
                if self.manifest_id is None:
                    self.manifest_id = member_mid
                elif self.manifest_id != member_mid:
                    raise ValueError(
                        f"All cohort members must share the same manifest_id. Got {member_mid!r}, expected {self.manifest_id!r}."
                    )

        missing = [pid for pid in self.permutation_ids if pid not in by_pid]
        if missing:
            raise ValueError(f'Cohort Missing bound members for permutation_ids: {missing}')

        bound = [by_pid[pid] for pid in self.permutation_ids]
        bind_native_output(self, bound, self._architecture_supports_probabilities(self.architecture_id), [self._selected_architectures[pid] for pid in self.permutation_ids])
        self._members = bound

        payload = json.dumps(
            {
                'aggregation_mode': self.aggregation_mode,
                'architecture_id': self.architecture_id,
                'manifest_id': self.manifest_id,
                'permutation_ids': sorted(self.permutation_ids),
            } | ({'binary_contracts': vars(self)['_mixed_policy']} if mixed else {}),
            sort_keys=True,
        )
        self.cohort_id = f'{SHA256_PREFIX}{hashlib.sha256(payload.encode()).hexdigest()}'


    def __call__(self, raw_klines: pl.DataFrame) -> list[BarPrediction]:

        return self.predict_all(raw_klines)


    def _assert_members(self) -> None:

        if not self._members:
            raise RuntimeError(
                'Cohort has no bound members. Use set_members() before predicting.'
            )


    def predict(self, raw_klines: pl.DataFrame) -> BarPrediction:

        '''Return the aggregate for the members' aligned last bar.'''

        self._assert_members()
        member_bars = [m.predict(raw_klines) for m in self._members]
        return self._aggregate_bar_predictions(member_bars)


    def predict_all(self, raw_klines: pl.DataFrame) -> list[BarPrediction]:

        '''Aggregate aligned member predictions for every formed bar.'''

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

        '''Aggregate aligned member outputs using existing blocking and error rules.'''

        if not member_bars:
            raise ValueError('Cohort member_bars must be a non-empty list.')

        valid_bars = [b for b in member_bars if b.reason != 'sensor-error']
        dt = next((b.datetime for b in valid_bars if b.datetime is not None), None)
        if not valid_bars:
            dt = next((b.datetime for b in member_bars if b.datetime is not None), None)
            return BarPrediction(datetime=dt, prediction=None, probability=None, reason='sensor-error')

        for i, bar in enumerate(member_bars):
            if bar.reason == 'sensor-error':
                continue
            member_id = self.member_ids[i] if i < len(self.member_ids) else i
            if bar.reason is None and bar.datetime is None:
                raise ValueError(f'Cohort member {member_id}: actionable prediction requires a timestamp.')
            if bar.datetime is not None and bar.datetime != dt:
                raise ValueError(f'Cohort member {member_id}: prediction timestamps do not align.')

        if len(valid_bars) < len(member_bars):
            n_errors = len(member_bars) - len(valid_bars)
            self._logger.debug(
                'Cohort aggregation: %d of %d member(s) returned sensor-error and were excluded.',
                n_errors, len(member_bars),
            )

        blocking = [b for b in valid_bars if b.reason is not None]
        if blocking:
            worst = max(blocking, key=lambda b: self._REASON_PRIORITY.get(b.reason, -1) if b.reason is not None else -1)
            return BarPrediction(datetime=dt, prediction=None, probability=None, reason=worst.reason)

        if self.architecture_id.startswith('mixed-binary-v1:'):
            valid_bars = mixed_outputs(self, member_bars)

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
        votes = np.array([int(float(cast(float, b.prediction)) > threshold) for b in valid_bars])
        pred = int(np.mean(votes) > self._VOTE_THRESHOLD)
        return BarPrediction(datetime=dt, prediction=pred, probability=None, reason=None)


    @staticmethod
    def _validate_probability_range(probs: npt.NDArray[np.float64]) -> None:

        if not np.isfinite(probs).all():
            raise ValueError(
                'Cohort Decoder probabilities must be finite values in [0, 1].')

        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError('Cohort Decoder probabilities must lie within [0, 1].')


    def _fallback_vote_threshold(self) -> float:

        arch = self.architecture_id.strip().lower()
        if not arch.startswith('mixed-binary-v1:') and 'regressor' in arch and not getattr(self._round_sources, 'qualified_folds', False):
            return 0.0
        return Cohort._VOTE_THRESHOLD


    @staticmethod
    def _normalize_permutation_id(pid: int | str) -> str:

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
            selected: Any = resolved_selector(context, **selector_params)
        except TypeError as e:
            raise ValueError(
                f'Cohort selector could not be called with selector_params: {e}'
            ) from e

        if not isinstance(selected, list):
            raise ValueError('Cohort selector must return a list of permutation ids.')

        if not selected:
            raise ValueError('Cohort selector returned no permutation ids.')

        selected_ids = cast(list[Any], selected)
        normalized = [cls._normalize_permutation_id(pid) for pid in selected_ids]
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
                                metadata: dict[str, Any],
                                round_entries: dict[str, dict[str, Any]],
                                available_ids: set[str]) -> dict[str, Any]:

        context: dict[str, Any] = {
            'experiment_dir': experiment_dir,
            'metadata': metadata,
            'round_entries': round_entries,
            'available_permutation_ids': Cohort._sort_permutation_ids(available_ids),
        }

        results_path = experiment_dir / 'results.csv'
        if results_path.exists():
            context['results'] = pl.read_csv(results_path, infer_schema_length=None)

        return context

    @staticmethod
    def _load_round_entries(round_data_path: Path) -> dict[str, dict[str, Any]]:

        entries: dict[str, dict[str, Any]] = {}
        with _storage.open_text(round_data_path) as f:
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
                                     round_entries: dict[str, dict[str, Any]],
                                     metadata: dict[str, Any]) -> str:

        return architecture_identity({cls._extract_architecture_id(round_entries[pid], metadata) for pid in selected_ids})

    @classmethod
    def _extract_architecture_id(cls, round_entry: Mapping[str, object], metadata: Mapping[str, object]) -> str:

        round_params = cast(Mapping[str, object], round_entry.get('round_params', {}))
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
    def _extract_yaml_reference_architecture(metadata: Mapping[str, object]) -> str | None:

        yaml_reference = metadata.get('yaml_reference')
        if not is_mapping(yaml_reference):
            return None

        sfd_config = yaml_reference.get('sfd')
        if not is_mapping(sfd_config):
            return None

        manifest_config = sfd_config.get('manifest')
        if not is_mapping(manifest_config):
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
            if _storage.round_path(parent).exists():
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


class _PythonCohort(PythonCohortBase[HistoryT], Cohort):
    '''Native Cohort over caller-owned live Sensors.'''
