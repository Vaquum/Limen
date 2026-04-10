import json
from pathlib import Path


class Cohort:

    '''
    Construct a decoder cohort from a completed experiment and selected permutations.

    This initial constructor implementation focuses on source resolution and
    permutation_id validation. Prediction and aggregation behavior are added
    separately.
    '''

    def __init__(self,
                 *,
                 experiment_id: str | None = None,
                 experiment_log_path: str | None = None,
                 permutation_ids: list[int | str] | None = None) -> None:

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

        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.available_permutation_ids = sorted(available_ids)
        self.permutation_ids = selected_ids
        self.architecture_id = architecture_id
        self.metadata = metadata

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
