import hashlib
import json
import logging
from datetime import datetime, timezone
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

        self._checkpoint_interval = checkpoint_interval


    def should_checkpoint(self, current_round: int) -> bool:

        '''
        Return True if a checkpoint should be saved at this round.

        Fires every checkpoint_interval rounds, never at round 0.

        Args:
            current_round (int): Current experiment round number

        Returns:
            bool: True if checkpoint should fire

        '''

        return current_round > 0 and current_round % self._checkpoint_interval == 0


    @staticmethod
    def compute_content_hash(content: dict) -> str:

        '''
        Compute a SHA-256 hex digest of a dict.

        Serialized to JSON with sorted keys for determinism.

        Args:
            content (dict): Content to hash

        Returns:
            str: 64-character lowercase hex SHA-256 digest

        '''

        raw = json.dumps(content, sort_keys=True)
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()


    def initialize_fresh(self, checkpoint_dir: Path) -> Path:

        '''
        Create a checkpoint directory and return its path.

        NOTE: Idempotent — safe to call on an existing directory.

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
             content_hash: str = '') -> None:

        '''
        Write checkpoint files into checkpoint_dir.

        NOTE: Each call overwrites the previous checkpoint. Files are written
        atomically using a write-then-rename pattern to avoid corrupt
        partial writes on crash.

        Args:
            checkpoint_dir (Path): Directory to write checkpoint files
            msq (MSQ): MSQ instance to checkpoint
            domain (ParamDomain): ParamDomain instance to checkpoint
            current_round (int): Round number at checkpoint time
            target_permutations (int): Total rounds planned for the run
            strategy_type (str): Class name of the search strategy
            content_hash (str): SHA-256 digest of the experiment content

        '''

        checkpoint_dir = Path(checkpoint_dir)

        metadata = {
            'experiment_round': current_round,
            'target_permutations': target_permutations,
            'strategy_type': strategy_type,
            'content_hash': content_hash,
            'saved_at': datetime.now(tz=timezone.utc).isoformat(),
        }

        self._write_json(checkpoint_dir / 'checkpoint_metadata.json', metadata)
        self._write_json(checkpoint_dir / 'msq_state.json', msq.get_state())
        self._write_json(checkpoint_dir / 'domain_state.json', domain.get_state())

        logger.info('Checkpoint saved at round %d → %s', current_round, checkpoint_dir)


    def load(self, checkpoint_dir: Path) -> dict[str, Any]:

        '''
        Load all checkpoint files from checkpoint_dir.

        Args:
            checkpoint_dir (Path): Directory containing checkpoint files

        Returns:
            dict: Keys 'metadata', 'msq_state', 'domain_state'

        Raises:
            FileNotFoundError: If any checkpoint file is missing
            json.JSONDecodeError: If any file contains invalid JSON

        '''

        checkpoint_dir = Path(checkpoint_dir)

        return {
            'metadata': self._read_json(checkpoint_dir / 'checkpoint_metadata.json'),
            'msq_state': self._read_json(checkpoint_dir / 'msq_state.json'),
            'domain_state': self._read_json(checkpoint_dir / 'domain_state.json'),
        }


    def validate(self,
                 checkpoint_dir: Path,
                 *,
                 content_hash: str,
                 strategy_type: str) -> None:

        '''
        Validate a checkpoint against the current experiment configuration.

        Checks content hash and strategy type. Raises ValueError with a
        descriptive message on any mismatch.

        Args:
            checkpoint_dir (Path): Directory containing checkpoint files
            content_hash (str): Expected SHA-256 digest
            strategy_type (str): Expected strategy class name

        Raises:
            ValueError: If content hash or strategy type do not match

        '''

        metadata = self.load(checkpoint_dir)['metadata']

        saved_hash = metadata.get('content_hash', '')
        if saved_hash != content_hash:
            raise ValueError(
                f"Content hash mismatch: checkpoint was created with hash "
                f"'{saved_hash[:16]}...', current is '{content_hash[:16]}...'. "
                f"Experiment configuration has changed. "
                f"Delete the checkpoint directory to start fresh."
            )

        saved_strategy = metadata.get('strategy_type', '')
        if saved_strategy != strategy_type:
            raise ValueError(
                f"Strategy type mismatch: checkpoint used '{saved_strategy}', "
                f"current is '{strategy_type}'. "
                f"Cannot resume with a different search strategy."
            )


    @staticmethod
    def _write_json(path: Path, data: dict) -> None:

        tmp = path.with_suffix('.tmp')
        try:
            with tmp.open('w') as f:
                json.dump(data, f, indent=2)
            tmp.replace(path)
        except OSError:
            if tmp.exists():
                tmp.unlink()
            raise


    @staticmethod
    def _read_json(path: Path) -> dict:

        with path.open('r') as f:
            return json.load(f)
