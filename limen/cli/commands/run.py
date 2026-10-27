from contextlib import ExitStack
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from limen.cli.commands._export_results import export_results
from limen.cli.commands._load_yaml import load_and_validate
from limen.cli.commands._results_dir import admit_output
from limen.cli.commands.profile import format_space
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment._round_errors import admit
from limen.yaml.compiler import CompiledSFD
from limen.yaml.compiler import build_pruning_strategies
from limen.yaml.compiler import build_search_strategy

DEFAULT_RESULTS_BASE: Path = Path('.')


def run_experiment(yaml_path: Path,
                   dry_run: bool = False,
                   manifest_id: str | None = None,
                   results_base: Path = DEFAULT_RESULTS_BASE,
                   progress_bar: bool = True,
                   workers: int | None = None) -> bool:

    '''Run the declared experiment, preserving its YAML and committed-manifest output layout.'''

    with ExitStack() as resources:
        click.echo(f"Loading {yaml_path} ...")

        yaml_dict, valid = load_and_validate(yaml_path)
        if not valid:
            return False

        uel_cfg = yaml_dict.get('uel', {})
        experiment_name: str = yaml_dict['metadata']['name']
        test_mode: bool = yaml_dict['metadata'].get('mode', 'development') == 'development'
        try:
            results_dir = _build_results_dir(uel_cfg, experiment_name, test_mode, manifest_id, results_base)
            if dry_run:
                _ = CompiledSFD(yaml_dict, project_root=yaml_path.parent).manifest()
        except Exception as exc:  # noqa: BLE001
            click.secho(f'  ✗ Compilation failed: {exc}', fg='red')
            return False
        if dry_run:
            click.echo('  Dry run — skipping execution')
            return True

        n_permutations: int = uel_cfg.get('n_permutations', 10000)
        prep_each_round: bool = bool(uel_cfg.get('prep_each_round', True))
        effective_workers = int(uel_cfg.get('workers', 1)) if workers is None else workers
        feedback_interval: int = int(uel_cfg.get('feedback_interval', 100))
        checkpoint_interval: int = int(uel_cfg.get('checkpoint_interval', 1000))

        try:
            from limen.cli.assessment import require_reservation
            _ = resources.enter_context(require_reservation(yaml_dict, yaml_path))
            search_strategy = build_search_strategy(yaml_dict)
            pruning_strategies = build_pruning_strategies(yaml_dict)
            compiled = CompiledSFD(yaml_dict, project_root=yaml_path.parent)
            uel = UniversalExperimentLoop(
                sfd=compiled, search_strategy=search_strategy,
                pruning_strategies=pruning_strategies, experiment_dir=results_dir,
                test_mode=test_mode, feedback_interval=feedback_interval,
                checkpoint_interval=checkpoint_interval,
                yaml_reference=dict(yaml_dict),
                on_round_error=uel_cfg.get('on_round_error', 'abort'), max_round_errors=uel_cfg.get('max_round_errors'),
            )
            from limen.experiment._holdout_research import reject_modes
            reject_modes(uel.manifest)
            uel.check_parallel_admission(effective_workers, prep_each_round,
                                         factorize=uel_cfg.get('factorize', False))
            uel.check_fresh_directory()
            admit(uel, {'workers': effective_workers, 'factorize': uel_cfg.get('factorize', False)})
        except Exception as exc:  # noqa: BLE001
            click.secho(f'  ✗ Failed to construct experiment: {exc}', fg='red')
            return False

        results_dir.mkdir(parents=True, exist_ok=True)
        _ = shutil.copy2(yaml_path, results_dir / ('manifest.yaml' if manifest_id is not None else yaml_path.name))

        total_space = math.prod(len(v) for v in compiled.params().values())
        strategy_type: str = uel_cfg.get('search_strategy', {}).get('type', 'random')
        click.echo(
            f"Running '{experiment_name}' — {n_permutations:,} of {format_space(total_space)} permutations ({strategy_type})"
        )
        click.echo(f"  Results → {results_dir}")

        try:
            uel.run(
                experiment_name=experiment_name,
                n_permutations=n_permutations,
                prep_each_round=prep_each_round,
                progress_bar=progress_bar,
                record_execution=uel_cfg.get('record_execution', False),
                record_model_outputs=uel_cfg.get('record_model_outputs', False),
                recording=uel_cfg.get('recording'),
                factorize=uel_cfg.get('factorize', False),
                workers=effective_workers,
            )
            if uel.interrupted:
                raise InterruptedError('Interrupted; checkpoint retained.')
        except Exception as exc:  # noqa: BLE001
            click.secho(f'  ✗ Experiment failed: {exc}', fg='red')
            return False

        export_results(uel_cfg, uel.experiment_log, results_dir)

        click.secho('  ✓ Experiment complete', fg='green')

        return True


def _build_results_dir(uel_cfg: dict[str, Any],
                       experiment_name: str,
                       test_mode: bool,
                       manifest_id: str | None = None,
                       results_base: Path = DEFAULT_RESULTS_BASE) -> Path:

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_root = results_base / 'results'
    prefix = results_root / 'dev' if test_mode else results_root

    if manifest_id is not None:
        return prefix / manifest_id[:8] / timestamp

    template: str = uel_cfg.get('output_path', '{name}_{datetime}')
    return admit_output(prefix, template.replace('{name}', experiment_name).replace('{datetime}', timestamp))
