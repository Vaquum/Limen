import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from limen.cli.commands._load_yaml import load_and_validate
from limen.cli.commands.profile import format_space
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.yaml.compiler import CompiledSFD
from limen.yaml.compiler import build_search_strategy


def run_experiment(yaml_path: Path, dry_run: bool = False) -> bool:

    '''
    Validate, compile, and execute a YAML experiment file.

    Args:
        yaml_path (Path): Path to the YAML experiment file
        dry_run (bool): When True, validate only — do not execute

    Returns:
        bool: True on success, False on validation failure

    '''

    click.echo(f"Loading {yaml_path} ...")

    yaml_dict, valid = load_and_validate(yaml_path)
    if not valid:
        return False

    if dry_run:
        try:
            CompiledSFD(yaml_dict).manifest()
        except Exception as exc:  # noqa: BLE001
            click.secho(f'  ✗ Compilation failed: {exc}', fg='red')
            return False
        click.echo('  Dry run — skipping execution')
        return True

    uel_cfg = yaml_dict.get('uel', {})

    experiment_name: str = yaml_dict['metadata']['name']
    n_permutations: int = uel_cfg.get('n_permutations', 10000)
    prep_each_round: bool = bool(uel_cfg.get('prep_each_round', True))
    test_mode: bool = yaml_dict['metadata'].get('mode', 'development') == 'development'

    results_dir = _build_results_dir(uel_cfg, experiment_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(yaml_path, results_dir / yaml_path.name)

    search_strategy = build_search_strategy(yaml_dict)
    compiled = CompiledSFD(yaml_dict)

    _params = compiled.params()
    total_space = math.prod(len(v) for v in _params.values())
    strategy_type: str = uel_cfg.get('search_strategy', {}).get('type', 'random')
    click.echo(
        f"Running '{experiment_name}' — "
        f"{n_permutations:,} of {format_space(total_space)} permutations ({strategy_type})"
    )
    click.echo(f"  Results → {results_dir}")

    feedback_interval: int = int(uel_cfg.get('feedback_interval', 100))
    checkpoint_interval: int = int(uel_cfg.get('checkpoint_interval', 1000))

    try:
        uel = UniversalExperimentLoop(
            sfd=compiled,
            search_strategy=search_strategy,
            experiment_dir=results_dir,
            test_mode=test_mode,
            feedback_interval=feedback_interval,
            checkpoint_interval=checkpoint_interval,
            yaml_reference=dict(yaml_dict),
        )
        uel.run(
            experiment_name=experiment_name,
            n_permutations=n_permutations,
            prep_each_round=prep_each_round,
        )
    except Exception as exc:  # noqa: BLE001
        click.secho(f'  ✗ Experiment failed: {exc}', fg='red')
        return False

    output_format: str = uel_cfg.get('output_format', 'csv')
    if output_format == 'parquet' and uel.experiment_log is not None:
        parquet_path = results_dir / 'results.parquet'
        uel.experiment_log.write_parquet(str(parquet_path))
        click.echo(f"  Parquet → {parquet_path}")

    click.secho('  ✓ Experiment complete', fg='green')

    return True


def _build_results_dir(uel_cfg: dict[str, Any], experiment_name: str) -> Path:

    output_path_template: str = uel_cfg.get('output_path', './results/{name}_{datetime}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Path(
        output_path_template
        .replace('{name}', experiment_name)
        .replace('{datetime}', timestamp)
    )


