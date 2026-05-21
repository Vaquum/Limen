import json
from pathlib import Path
from typing import Any

import click

from limen.experiment import GridStrategy
from limen.experiment import RandomStrategy
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.yaml.compiler import CompiledSFD


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
    sfd_cfg = yaml_reference.get('sfd', {})
    experiment_name: str = yaml_reference['metadata']['name']
    prep_each_round: bool = bool(uel_cfg.get('prep_each_round', True))
    test_mode: bool = yaml_reference['metadata'].get('mode', 'development') == 'development'
    feedback_interval: int = int(uel_cfg.get('feedback_interval', 100))
    checkpoint_interval: int = int(uel_cfg.get('checkpoint_interval', 1000))

    try:
        compiled = CompiledSFD(yaml_reference)
        search_strategy = _build_search_strategy(uel_cfg, sfd_cfg)
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

    metadata = json.loads(metadata_path.read_text())
    yaml_reference = metadata.get('yaml_reference')

    if yaml_reference is None:
        click.secho(
            "  ✗ metadata.json has no 'yaml_reference' — experiment was not started from a YAML file.",
            fg='red',
        )
        return None

    return yaml_reference


def _load_target_permutations(results_dir: Path) -> int | None:

    checkpoint_path = results_dir / 'checkpoint.json'
    if not checkpoint_path.exists():
        click.secho(
            f"  ✗ No checkpoint.json found in '{results_dir}' — nothing to resume.",
            fg='red',
        )
        return None

    try:
        data = json.loads(checkpoint_path.read_text())
        return int(data['metadata']['target_permutations'])
    except (KeyError, ValueError, TypeError) as exc:
        click.secho(f'  ✗ Could not read target_permutations from checkpoint: {exc}', fg='red')
        return None


def _build_search_strategy(uel_cfg: dict[str, Any],
                            sfd_cfg: dict[str, Any]) -> RandomStrategy | GridStrategy:

    strategy_type = (uel_cfg.get('search_strategy') or {}).get('type', 'random')
    params = {k: list(v) for k, v in (sfd_cfg.get('params') or {}).items()}
    domain = ParamDomain(params)

    if strategy_type == 'grid':
        return GridStrategy(domain)
    return RandomStrategy(domain)
