from pathlib import Path

import click

from limen.cli.commands.commit import run_commit
from limen.cli.commands.init import run_init
from limen.cli.commands.list_templates import run_list_templates
from limen.cli.commands.new import run_new
from limen.cli.commands.profile import run_profile
from limen.cli.commands.resume import run_resume
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
      limen profile experiment.yaml        Profile permutation space and runtime
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
@click.option('--parent', default=None, metavar='MANIFEST_ID',
              help='Parent manifest ID for lineage tracking (e.g. sha256:abc123...).')
@click.option('--message', '-m', default=None,
              help='Git commit message. Auto-generated if omitted.')
def commit(yaml_file: Path, parent: str | None, message: str | None) -> None:

    '''
    Validate and commit a YAML experiment file to the manifest store.

    \b
    Steps:
      1. Find the limen project root (walks up from the YAML file location)
      2. Validate the YAML file
      3. Content-address and store in manifests/committed/
      4. Update manifests/committed/index.json
      5. Git add + commit inside the project

    \b
    The committed file has a lineage block injected with its ID and timestamp.
    Committing the same file twice is a no-op (idempotent).

    \b
    Examples:
      limen commit manifests/examples/logreg_binary.yaml
      limen commit manifests/examples/logreg_binary.yaml --message "tuned lookback"
      limen commit manifests/examples/logreg_binary.yaml --parent sha256:abc123...
    '''

    ok = run_commit(yaml_file, parent, message)
    raise SystemExit(0 if ok else 1)


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
      - Split config types and valid ranges (train > 0, val/test >= 0)
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


@cli.command('profile')
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
def profile_cmd(yaml_file: Path) -> None:

    '''
    Profile a YAML experiment — permutation space and runtime estimate.

    \b
    Always computed:
      - Total permutations and complexity rating (low / medium / high / extreme)
      - Per-parameter value counts

    \b
    Computed when test_data_source is configured:
      - Average time per permutation (covering array sampling)
      - Estimated total runtime across the full permutation space
      - Errors encountered during sampling (small data, NaN, class imbalance)

    \b
    Exits 0 on success, 1 on validation or compilation failure.
    Sampling errors are non-fatal and reported in the output.

    \b
    Examples:
      limen profile experiment.yaml
      limen profile limen/yaml/templates/logreg_binary.yaml
    '''

    ok = run_profile(yaml_file)
    raise SystemExit(0 if ok else 1)


@cli.command()
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option('--dry-run', is_flag=True, default=False,
              help='Validate and compile only — do not execute the experiment.')
@click.option('--resume', type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, metavar='RESULTS_DIR',
              help='Resume from a checkpoint directory instead of starting fresh.')
def run(yaml_file: Path | None, dry_run: bool, resume: Path | None) -> None:

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
    Mode behaviour is controlled by metadata.mode in the YAML:
      development    Uses test data (HuggingFace), small dataset
      production     Uses live data source configured in the manifest

    \b
    Output:
      Results are written to ./results/{name}_{datetime}/results.csv by default.
      Set uel.output_format: parquet to also write results.parquet.
      Override the output path with uel.output_path in the YAML.

    \b
    Exits 0 on success, 1 on validation failure or runtime error.

    \b
    Examples:
      limen run experiment.yaml
      limen run --dry-run experiment.yaml
      limen run --resume ./results/my_experiment_20260521_120000
    '''

    if resume is not None:
        if yaml_file is not None:
            click.secho('Cannot specify both a YAML file and --resume.', fg='red')
            raise SystemExit(1)
        if dry_run:
            click.secho('--dry-run has no effect with --resume.', fg='red')
            raise SystemExit(1)
        ok = run_resume(resume)
    elif yaml_file is not None:
        ok = run_experiment(yaml_file, dry_run=dry_run)
    else:
        click.secho('Provide a YAML file or --resume <results-dir>.', fg='red')
        raise SystemExit(1)

    raise SystemExit(0 if ok else 1)


@cli.command('list-templates')
def list_templates() -> None:

    '''
    List all available YAML experiment templates.

    \b
    Examples:
      limen list-templates
    '''

    run_list_templates()


@cli.command()
@click.argument('output', type=click.Path(path_type=Path))
@click.option('--template', default=None,
              help='Template name to scaffold from (e.g. logreg_binary).')
def init(output: Path, template: str | None) -> None:

    '''
    Scaffold a new YAML experiment file from a template.

    \b
    Copies the selected template to OUTPUT and sets metadata.name
    to the output filename stem.

    \b
    Examples:
      limen init my_experiment.yaml --template logreg_binary
      limen init my_experiment.yaml               # lists templates when --template omitted
    '''

    ok = run_init(output, template)
    raise SystemExit(0 if ok else 1)


@cli.command()
@click.argument('project_name')
@click.option('--backup-remote', default=None,
              help='Git remote URL for manifest store backup (can also be set interactively).')
def new(project_name: str, backup_remote: str | None) -> None:

    '''
    Create a new Limen project from the official project template.

    \b
    Clones Vaquum/limen-project-template and optionally sets up a backup remote.

    \b
    Examples:
      limen new my-project
      limen new my-project --backup-remote git@github.com:user/my-project.git
    '''

    ok = run_new(project_name, backup_remote)
    raise SystemExit(0 if ok else 1)
