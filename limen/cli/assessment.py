from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
import sys
from collections.abc import Iterator, Mapping
import json
from pathlib import Path
from typing import cast

import click


def _project(path: Path) -> Path:
    from limen.yaml.config import find_project_root
    root = find_project_root(path.parent)
    if root is None:
        raise click.ClickException('A guarded assessment requires a Git project with limen.toml')
    return root


def _write(path: Path, value: object) -> None:
    _ = path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _assess(yaml_file: Path, results: Path, reservation: str | None) -> None:
    import polars as pl
    from limen import Cohort, Trainer
    from limen.cli.commands._load_yaml import load_and_validate
    from limen.yaml._assessment_spec import assessment_spec, holdout_spec
    document, valid = load_and_validate(yaml_file)
    if not valid:
        raise click.ClickException('Invalid assessment declaration')
    spec = assessment_spec(document)
    if spec is None:
        raise click.ClickException('Declare assessment in the YAML before selecting results')
    holdout = holdout_spec(document)
    if (holdout is None) != (reservation is None):
        raise click.ClickException('Guarded assessment requires holdout YAML and its --reservation identity together')
    saved = json.loads((results / 'metadata.json').read_text())['yaml_reference']
    if saved != document:
        raise click.ClickException('Assessment YAML differs from the recorded experiment declaration')
    if holdout is not None and reservation is not None:
        from limen.experiment._holdout_attempts import reservation as declared_reservation
        _ = declared_reservation(_project(yaml_file), reservation, str(holdout['start']), str(holdout['end']), str(holdout['context_start']))
    log = pl.read_csv(results / 'results.csv')
    if spec.metric not in log.columns:
        raise click.ClickException('The declared validation metric is absent from the result log')
    ranked = log.filter(pl.col(spec.metric).is_finite()).sort([spec.metric, 'id'], descending=[spec.direction == 'maximize', False])
    if ranked.height < spec.top_k:
        raise click.ClickException('Fewer successful finite validation results than assessment.top_k')
    ids = [str(identity) for identity in ranked['id'].head(spec.top_k).to_list()]
    trainer = Trainer(results, project_root=yaml_file.parent)
    if spec.folds:
        rounds: list[tuple[str | Path, str] | tuple[str | Path, str, int]] = [(results, identity, fold) for identity in ids for fold in spec.folds]
        sensors = [trainer.train([identity], fold=fold)[0] for identity in ids for fold in spec.folds]
        cohort = Cohort(rounds=rounds)
    else:
        sensors = trainer.train(ids)
        cohort = Cohort(experiment_log_path=str(results), permutation_ids=ids)
    cohort.set_members(sensors)
    report = {'selected_ids': ids, 'members': len(sensors), 'member_ids': cohort.member_ids,
              'folds': list(spec.folds), 'metric': spec.metric, 'direction': spec.direction}
    if holdout is not None:
        from limen.experiment.holdout import final_holdout_report, freeze_holdout_selection
        if reservation is None:
            raise click.ClickException('Guarded final assessment requires --reservation from limen reserve')
        target = None
        if 'evaluation_target' in holdout:
            from limen.yaml.resolver import resolve
            declaration = cast(dict[str, object], holdout['evaluation_target'])
            target = resolve(str(declaration['class']))(pl.DataFrame(), str(declaration['name']), **cast(dict[str, object], declaration.get('params', {})))
        frozen = freeze_holdout_selection(cohort, evaluation_target=target, context_start=str(holdout['context_start']),
                                          scenarios=cast(list[Mapping[str, object]], holdout['scenarios']))
        final = final_holdout_report(frozen, project_root=_project(yaml_file), reservation_id=reservation, start=str(holdout['start']), end=str(holdout['end']))
        _write(results / 'final-report.json', final)
    elif reservation is not None:
        raise click.ClickException('--reservation requires a holdout declaration')
    if spec.predict:
        from limen.yaml import CompiledSFD
        manifest = CompiledSFD(document).manifest()
        if spec.inference_source is None:
            raw = manifest.fetch_data()
        else:
            from limen.data import HistoricalData
            raw = HistoricalData().get_any_file(spec.inference_source)
        predictions = cohort.predict_all(raw)
        pl.DataFrame([asdict(prediction) for prediction in predictions], infer_schema_length=None).write_parquet(results / 'predictions.parquet')
    _write(results / 'assessment.json', report)
    click.echo(f'Validated {len(sensors)} Sensor members → {results / "assessment.json"}')


@contextmanager
def require_reservation(document: dict[str, object], yaml_path: Path) -> Iterator[None]:
    from limen.yaml._assessment_spec import holdout_spec
    spec = holdout_spec(document)
    if spec is None:
        yield
        return
    from limen.experiment._holdout_history import admission, bind, utc_day
    with admission(_project(yaml_path)):
        binding = bind(yaml_path.parent)
        if binding is None or binding.cutoff_ns is None or binding.cutoff_ns > utc_day(str(spec['start'])) or binding.interval_seconds != spec['interval_seconds']:
            raise ValueError('Run limen reserve for the declared final window before research')
        yield


@click.command('assess')
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
@click.option('--results', required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--reservation', default=None, help='Reservation identity returned by limen reserve.')
def assess(yaml_file: Path, results: Path, reservation: str | None) -> None:
    '''Select validation-ranked rounds, reconstruct Sensors and assess the declared Cohort.'''
    try:
        _assess(yaml_file, results, reservation)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc

@click.command('reserve')
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
def reserve(yaml_file: Path) -> None:
    '''Reserve the YAML's final window before research; print its identity.'''
    from limen.cli.commands._load_yaml import load_and_validate
    from limen.yaml._assessment_spec import holdout_spec
    from limen.experiment.holdout import reserve_holdout
    with redirect_stdout(sys.stderr):
        document, valid = load_and_validate(yaml_file)
    if not valid:
        raise click.ClickException('Invalid holdout declaration')
    spec = holdout_spec(document)
    if spec is None:
        raise click.ClickException('Declare holdout in the YAML before reserving')
    try:
        identity = reserve_holdout(_project(yaml_file), str(spec['start']), str(spec['end']),
                                    interval_seconds=cast(int, spec['interval_seconds']))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(identity)


def register(cli: click.Group) -> None:
    cli.add_command(assess)
    cli.add_command(reserve)
