'''CLI entry point for running a Loop payload through UEL.

Invocation:
    python -m limen.sfd.loop.run <payload.json> --out <experiment_dir> --n <N>

The runner owns the experiment directory: it creates the dir, copies the
payload there as an audit trail, then hands the dir to UEL which writes all
its standard artifacts (results.csv, checkpoint.json, audit.jsonl, etc.).
The progress callback writes progress.json on every round so a polling
backend can serve live updates.

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands. See README.md.
'''

import argparse
import json
import logging
import sys
from pathlib import Path

from limen import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy
from limen.sfd.loop.loop_sfd import LoopSFD
from limen.sfd.loop.progress import make_progress_callback


logger = logging.getLogger(__name__)


def run_experiment(payload_path: Path,
                   experiment_dir: Path,
                   n_permutations: int = 100,
                   seed: int = 42) -> None:

    '''
    Compute and execute a Loop experiment from a JSON payload file.

    Args:
        payload_path (Path): Path to the Loop payload JSON file
        experiment_dir (Path): Directory to write all experiment artifacts to.
            Created if missing
        n_permutations (int): Number of parameter combinations to try
        seed (int): Random seed for the search strategy

    '''

    payload_path = Path(payload_path)
    experiment_dir = Path(experiment_dir)

    experiment_dir.mkdir(parents=True, exist_ok=True)

    payload_text = payload_path.read_text()
    payload = json.loads(payload_text)

    # Audit copy — survives the run, helps reproduce later
    (experiment_dir / 'payload.json').write_text(payload_text)

    sfd = LoopSFD(payload)

    domain = ParamDomain(sfd.params())
    strategy = RandomStrategy(domain, seed=seed)

    progress_file = experiment_dir / 'progress.json'
    progress_callback = make_progress_callback(progress_file, total=n_permutations)

    uel = UniversalExperimentLoop(
        sfd=sfd,
        search_strategy=strategy,
        feedback_interval=1,
        experiment_dir=experiment_dir,
        intra_callback=progress_callback,
    )

    logger.info(
        'Starting Loop experiment: payload=%s out=%s n=%d',
        payload_path, experiment_dir, n_permutations,
    )

    uel.run(
        experiment_name=str(experiment_dir / 'run'),
        n_permutations=n_permutations,
    )

    logger.info('Experiment complete: %s', experiment_dir)


def main(argv: list[str] | None = None) -> int:

    '''
    Compute and execute a Loop experiment from command-line arguments.

    Args:
        argv (list | None): Command-line arguments excluding the program name

    Returns:
        int: Exit code (0 on success, 1 on failure)

    '''

    parser = argparse.ArgumentParser(
        description='Run a Loop web UI experiment payload through Limen UEL.',
    )
    parser.add_argument(
        'payload',
        type=Path,
        help='Path to the Loop payload JSON file',
    )
    parser.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Experiment directory for all output artifacts',
    )
    parser.add_argument(
        '--n',
        type=int,
        default=100,
        help='Number of parameter combinations to try (default: 100)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for the search strategy (default: 42)',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)',
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    )

    try:
        run_experiment(
            payload_path=args.payload,
            experiment_dir=args.out,
            n_permutations=args.n,
            seed=args.seed,
        )
    except Exception:
        logger.exception('Experiment failed')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
