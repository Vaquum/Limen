import json
from pathlib import Path
from typing import Any

import click

from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.yaml.compiler import CompiledSFD
from limen.yaml.compiler import build_search_strategy


def run_resume(results_dir: Path) -> bool:

    '''
    Resume an experiment from a checkpoint directory.

    Reads yaml_reference from metadata.json to reconstruct the SFD and search
    strategy. Reads target_permutations from checkpoint.json to continue to
    the original round target.

    Args:
        results_dir (Path): Path to the experiment results directory containing
            metadata.json and checkpoint.json

    Returns:
        bool: True on success, False on failure

    '''

    click.echo(f"Resuming from {results_dir} ...")

    yaml_reference = _load_yaml_reference(results_dir)
    if yaml_reference is None:
        return False

    target_permutations = _load_target_permutations(results_dir)
    if target_permutations is None:
        return False


    uel_cfg = yaml_reference.get('uel', {})
    experiment_name: str = yaml_reference['metadata']['name']
    prep_each_round: bool = bool(uel_cfg.get('prep_each_round', True))
    test_mode: bool = yaml_reference['metadata'].get('mode', 'development') == 'development'
    feedback_interval: int = int(uel_cfg.get('feedback_interval', 100))
    checkpoint_interval: int = int(uel_cfg.get('checkpoint_interval', 1000))

    try:
        compiled = CompiledSFD(yaml_reference)
        search_strategy = build_search_strategy(yaml_reference)
    except Exception as exc:  # noqa: BLE001
        click.secho(f'  ✗ Failed to reconstruct experiment: {exc}', fg='red')
        return False

    click.echo(f"Resuming '{experiment_name}' (target: {target_permutations} permutations) ...")

    try:
        uel = UniversalExperimentLoop(
            sfd=compiled,
            search_strategy=search_strategy,
            experiment_dir=results_dir,
            test_mode=test_mode,
            feedback_interval=feedback_interval,
            checkpoint_interval=checkpoint_interval,
            yaml_reference=yaml_reference,
        )
        uel.run(
            experiment_name=experiment_name,
            n_permutations=target_permutations,
            prep_each_round=prep_each_round,
            resume=True,
        )
    except Exception as exc:  # noqa: BLE001
        click.secho(f'  ✗ Experiment failed: {exc}', fg='red')
        return False

    click.secho('  ✓ Experiment complete', fg='green')
    return True


def _load_yaml_reference(results_dir: Path) -> dict[str, Any] | None:

    metadata_path = results_dir / 'metadata.json'
    if not metadata_path.exists():
        click.secho(
            f"  ✗ No metadata.json found in '{results_dir}' — not a valid experiment directory.",
            fg='red',
        )
        return None

    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        click.secho(f"  ✗ Cannot read metadata.json: {exc}", fg='red')
        return None

    if not isinstance(metadata, dict):
        click.secho("  ✗ metadata.json is not a JSON object.", fg='red')
        return None

    yaml_reference = metadata.get('yaml_reference')

    if not isinstance(yaml_reference, dict):
        click.secho(
            "  ✗ metadata.json has no valid 'yaml_reference' — experiment was not started from a YAML file.",
            fg='red',
        )
        return None

    if not isinstance(yaml_reference.get('metadata'), dict) or 'name' not in yaml_reference['metadata']:
        click.secho(
            "  ✗ 'yaml_reference' is missing 'metadata.name' — cannot identify experiment.",
            fg='red',
        )
        return None

    return yaml_reference


def _load_target_permutations(results_dir: Path) -> int | None:

    try:
        data = CheckpointManager().load(results_dir)
        return int(data['metadata']['target_permutations'])
    except (ValueError, KeyError, TypeError) as exc:
        click.secho(f'  ✗ Cannot load checkpoint: {exc}', fg='red')
        return None
