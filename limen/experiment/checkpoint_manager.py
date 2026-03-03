import hashlib
import json
import logging
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain

logger = logging.getLogger(__name__)


class CheckpointManager:

    '''Manage experiment checkpoints — save, load, and validate state.'''

    def __init__(self, *, checkpoint_interval: int = 1000) -> None:

        '''
        Initialize the CheckpointManager.

        Args:
            checkpoint_interval (int): Save a checkpoint every N rounds

        '''

        if checkpoint_interval < 1:
            raise ValueError(
                f"checkpoint_interval must be >= 1, got {checkpoint_interval}"
            )
        self._checkpoint_interval = checkpoint_interval


    def should_checkpoint(self, current_round: int) -> bool:

        '''
        Compute whether a checkpoint is due at the current round.

        Args:
            current_round (int): Current experiment round number

        Returns:
            bool: True if a checkpoint should be saved

        '''

        return current_round > 0 and current_round % self._checkpoint_interval == 0


    @staticmethod
    def compute_content_hash(content: dict) -> str:

        '''
        Compute a SHA-256 hex digest of a dict, serialized with sorted keys for determinism.

        Args:
            content (dict): Content to hash

        Returns:
            str: 64-character lowercase hex SHA-256 digest

        '''

        raw = json.dumps(content, sort_keys=True)
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()


    def initialize_fresh(self, checkpoint_dir: Path) -> Path:

        '''
        Create the checkpoint directory if it does not exist, and return its path.

        Args:
            checkpoint_dir (Path): Path to create

        Returns:
            Path: Created directory path

        '''

        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        logger.info('Initialized checkpoint directory: %s', path)
        return path


    def save(self,
             checkpoint_dir: Path,
             msq: MSQ,
             domain: ParamDomain,
             current_round: int,
             target_permutations: int,
             *,
             strategy_type: str,
             content_hash: str,
             feedback_controller: Any | None = None,
             pruning_strategies: list | None = None) -> None:

        '''
        Write a checkpoint file into checkpoint_dir.

        NOTE: All state is written atomically to a single file via a
        write-then-rename pattern. The previous checkpoint remains intact
        until the new one is fully written.

        Args:
            checkpoint_dir (Path): Directory to write checkpoint file
            msq (MSQ): MSQ instance to checkpoint
            domain (ParamDomain): ParamDomain instance to checkpoint
            current_round (int): Round number at checkpoint time
            target_permutations (int): Total rounds planned for the run
            strategy_type (str): Class name of the search strategy
            content_hash (str): SHA-256 digest of the experiment content
            feedback_controller (Any | None): FeedbackController to checkpoint
            pruning_strategies (list | None): PruningStrategy instances to checkpoint

        '''

        checkpoint = {
            'metadata': {
                'experiment_round': current_round,
                'target_permutations': target_permutations,
                'strategy_type': strategy_type,
                'content_hash': content_hash,
                'saved_at': datetime.now(tz=timezone.utc).isoformat(),
            },
            'msq_state': msq.get_state(),
            'domain_state': domain.get_state(),
        }

        if feedback_controller is not None:
            checkpoint['feedback_controller_state'] = (
                feedback_controller.get_state()
            )

        if pruning_strategies is not None:
            checkpoint['pruning_strategy_states'] = [
                ps.get_state() for ps in pruning_strategies
            ]

        self._write_json(Path(checkpoint_dir) / 'checkpoint.json', checkpoint)

        logger.info('Checkpoint saved at round %d → %s', current_round, checkpoint_dir)


    def load(self, checkpoint_dir: Path) -> dict[str, Any]:

        '''
        Load checkpoint from checkpoint_dir.

        Args:
            checkpoint_dir (Path): Directory containing checkpoint file

        Returns:
            dict: Keys 'metadata', 'msq_state', 'domain_state'

        Raises:
            ValueError: If checkpoint is missing or corrupt

        '''

        try:
            data = self._read_json(Path(checkpoint_dir) / 'checkpoint.json')
        except FileNotFoundError as e:
            raise ValueError(
                f"No checkpoint found in '{checkpoint_dir}'."
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Corrupt checkpoint in '{checkpoint_dir}': {e}"
            ) from e

        if not isinstance(data, dict):
            raise ValueError(
                f"Corrupt checkpoint in '{checkpoint_dir}': "
                f"expected object, got {type(data).__name__}."
            )

        return data


    def validate(self,
                 checkpoint_dir: Path,
                 *,
                 content_hash: str,
                 strategy_type: str) -> dict[str, Any]:

        '''
        Validate a checkpoint's structure, content hash, and strategy type against the current configuration.

        Args:
            checkpoint_dir (Path): Directory containing checkpoint file
            content_hash (str): Expected SHA-256 digest
            strategy_type (str): Expected strategy class name

        Returns:
            dict: Validated checkpoint data with keys 'metadata', 'msq_state', 'domain_state'

        Raises:
            ValueError: If checkpoint is missing, corrupt, or configuration does not match

        '''

        data = self.load(checkpoint_dir)
        self._validate_structure(data, checkpoint_dir)

        metadata = data['metadata']

        saved_hash = metadata['content_hash']
        if saved_hash != content_hash:
            raise ValueError(
                f"Content hash mismatch: checkpoint was created with hash "
                f"'{saved_hash[:16]}...', current is '{content_hash[:16]}...'. "
                f"Experiment configuration has changed. "
                f"Delete the checkpoint directory to start fresh."
            )

        saved_strategy = metadata['strategy_type']
        if saved_strategy != strategy_type:
            raise ValueError(
                f"Strategy type mismatch: checkpoint used '{saved_strategy}', "
                f"current is '{strategy_type}'. "
                f"Cannot resume with a different search strategy."
            )

        return data


    @staticmethod
    def _validate_structure(data: Any, checkpoint_dir: Path) -> None:

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid checkpoint format in '{checkpoint_dir}': "
                f"top-level JSON must be an object, got {type(data).__name__}."
            )

        if 'metadata' not in data:
            raise ValueError(
                f"Invalid checkpoint format in '{checkpoint_dir}': missing 'metadata' key."
            )

        metadata = data['metadata']

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Invalid checkpoint format in '{checkpoint_dir}': 'metadata' must be an object."
            )

        for key in ('experiment_round', 'target_permutations', 'content_hash', 'strategy_type'):
            if key not in metadata:
                raise ValueError(
                    f"Invalid checkpoint format in '{checkpoint_dir}': missing metadata key '{key}'."
                )

        for key, expected in (('experiment_round', int), ('target_permutations', int),
                              ('content_hash', str), ('strategy_type', str)):
            if not isinstance(metadata[key], expected):
                raise ValueError(
                    f"Invalid checkpoint format in '{checkpoint_dir}': "
                    f"metadata key '{key}' must be {expected.__name__}, "
                    f"got {type(metadata[key]).__name__}."
                )

        for key in ('msq_state', 'domain_state'):
            if key not in data:
                raise ValueError(
                    f"Invalid checkpoint format in '{checkpoint_dir}': missing '{key}' key."
                )
            if not isinstance(data[key], dict):
                raise ValueError(
                    f"Invalid checkpoint format in '{checkpoint_dir}': "
                    f"'{key}' must be an object, got {type(data[key]).__name__}."
                )


    @staticmethod
    def _write_json(path: Path, data: dict) -> None:

        tmp = path.with_suffix('.tmp')
        try:
            with tmp.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise


    @staticmethod
    def _read_json(path: Path) -> Any:

        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
