import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from limen.experiment import GridStrategy
from limen.experiment import RandomStrategy
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.yaml.compiler import CompiledSFD
from limen.yaml.parser import parse
from limen.yaml.validator import validate


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

    yaml_dict, parse_errors = parse(yaml_path)
    if parse_errors:
        for e in parse_errors:
            location = f' (line {e.line})' if e.line else ''
            click.secho(f'  PARSE ERROR{location}: {e.message}', fg='red')
        return False

    result = validate(yaml_dict)
    for e in result.errors:
        location = f' (line {e.line})' if e.line else ''
        path = f'  [{e.path}]' if e.path else ''
        suggestion = f'\n    → {e.suggestion}' if e.suggestion else ''
        click.secho(f'  ERROR{path}{location}: {e.message}{suggestion}', fg='red')
    for w in result.warnings:
        location = f' (line {w.line})' if w.line else ''
        path = f'  [{w.path}]' if w.path else ''
        click.secho(f'  WARN{path}{location}: {w.message}', fg='yellow')

    if not result.valid:
        click.secho(f'  ✗ {len(result.errors)} validation error(s) — aborting', fg='red')
        return False

    click.secho('  ✓ Valid', fg='green')

    if dry_run:
        try:
            CompiledSFD(yaml_dict).manifest()
        except Exception as exc:  # noqa: BLE001
            click.secho(f'  ✗ Compilation failed: {exc}', fg='red')
            return False
        click.echo('  Dry run — skipping execution')
        return True

    uel_cfg = yaml_dict.get('uel', {})
    sfd_cfg = yaml_dict.get('sfd', {})

    experiment_name: str = yaml_dict['metadata']['name']
    n_permutations: int = uel_cfg.get('n_permutations', 10000)
    prep_each_round: bool = bool(uel_cfg.get('prep_each_round', True))
    test_mode: bool = yaml_dict['metadata'].get('mode', 'development') == 'development'

    results_dir = _build_results_dir(uel_cfg, experiment_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(yaml_path, results_dir / yaml_path.name)

    search_strategy = _build_search_strategy(uel_cfg, sfd_cfg)
    compiled = CompiledSFD(yaml_dict)

    click.echo(f"Running '{experiment_name}' ({n_permutations} permutations) ...")
    click.echo(f"  Results → {results_dir}")

    uel = UniversalExperimentLoop(
        sfd=compiled,
        search_strategy=search_strategy,
        experiment_dir=results_dir,
        test_mode=test_mode,
    )

    try:
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


def _build_search_strategy(uel_cfg: dict[str, Any],
                            sfd_cfg: dict[str, Any]) -> RandomStrategy | GridStrategy:

    strategy_cfg = uel_cfg.get('search_strategy', {})
    if not isinstance(strategy_cfg, dict):
        raise ValueError(
            f"'uel.search_strategy' must be a mapping, got {type(strategy_cfg).__name__}"
        )
    strategy_type = strategy_cfg.get('type', 'random')
    # ruamel.yaml returns CommentedMap/CommentedSeq — convert to plain Python types
    params = {k: list(v) for k, v in (sfd_cfg.get('params') or {}).items()}
    domain = ParamDomain(params)

    if strategy_type == 'grid':
        return GridStrategy(domain)
    return RandomStrategy(domain)
