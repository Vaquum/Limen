from pathlib import Path

import click

from limen.cli.commands.run import run_experiment
from limen.cli.commands.validate import run_validate


@click.group()
def cli() -> None:

    '''
    Limen — declarative ML experiment runner.

    \b
    Define experiments as YAML files and run them with a single command.
    Limen handles validation, compilation, parameter search, and result saving.

    \b
    Quick start:
      limen validate experiment.yaml       Check your YAML for errors
      limen run experiment.yaml            Run the experiment
      limen run --dry-run experiment.yaml  Validate + compile only, no execution

    \b
    YAML structure:
      schema_version: "1.0"
      metadata:
        name: my_experiment
        mode: development          # development | production
      sfd:
        manifest:
          type: ml                 # ml | rule_based
          ...
        params:
          lookback: [12, 24, 48]
          ...
      uel:
        n_permutations: 100
        search_strategy:
          type: random             # random | grid
        output_format: csv         # csv | parquet

    \b
    Templates are in limen/yaml/templates/:
      logreg_binary.yaml           Logistic regression binary classifier
    '''


@cli.command()
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
def validate(yaml_file: Path) -> None:

    '''
    Validate a YAML experiment file.

    \b
    Checks:
      - YAML syntax and structure
      - Required fields (metadata, sfd, uel)
      - Schema version
      - All limen.* paths are resolvable (funcs, classes, methods)
      - Parameter lists are non-empty
      - Split config sums and types
      - Search strategy and output format values

    \b
    Exits 0 if valid, 1 if errors are found.

    \b
    Examples:
      limen validate experiment.yaml
      limen validate limen/yaml/templates/logreg_binary.yaml
    '''

    ok = run_validate(yaml_file)
    raise SystemExit(0 if ok else 1)


@cli.command()
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
@click.option('--dry-run', is_flag=True, default=False,
              help='Validate and compile only — do not execute the experiment.')
@click.option('--production', is_flag=True, default=False,
              help='Force production mode, overriding metadata.mode: development in the YAML.')
def run(yaml_file: Path, dry_run: bool, production: bool) -> None:

    '''
    Validate, compile, and run a YAML experiment file.

    \b
    Steps:
      1. Parse the YAML file
      2. Validate structure and all limen.* paths
      3. Compile the manifest (resolve all callables)
      4. Build the parameter search domain
      5. Run the UniversalExperimentLoop
      6. Save results to the configured output path

    \b
    Mode behaviour:
      metadata.mode: development   Uses test data (HuggingFace), small dataset
      metadata.mode: production    Uses live data source configured in the manifest
      --production flag            Overrides development mode without editing the YAML

    \b
    Output:
      Results are written to ./results/{name}_{timestamp}/results.csv by default.
      Override with uel.output_path and uel.output_format in the YAML.

    \b
    Exits 0 on success, 1 on validation failure or runtime error.

    \b
    Examples:
      limen run experiment.yaml
      limen run --dry-run experiment.yaml
      limen run --production experiment.yaml
      limen run --production --dry-run experiment.yaml
    '''

    ok = run_experiment(yaml_file, dry_run=dry_run, production=production)
    raise SystemExit(0 if ok else 1)
