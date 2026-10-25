# Command-Line Interface

Compiled Research uses YAML files and CLI commands from declaration through Sensor/Cohort assessment. The [Compiled project](examples/compiled-research/README.md) supplies its complete commands and recorded inputs. [Python Research](examples/python-research/README.md) owns ordinary Python functions.

## Prerequisites

Use Python 3.10–3.13, the matching wheel and example bundle, the declared model extras, and Git for committed projects and guarded holdout history. Run the worked commands from the bundle root.

## First YAML Run

<!-- example: cli-project-lifecycle source=docs/examples/compiled-research/project.sh region=lifecycle scenario=command_line interface=compiled role=executed -->
```bash
bundle=$(pwd)
limen --version
limen list-templates
mkdir -p results/compiled-project
cp compiled-research/limen.toml results/compiled-project/limen.toml
cp compiled-research/project.yaml results/compiled-project/research.yaml
cp -R "$bundle/inputs" results/compiled-project/inputs
printf 'inputs/\nresults/dev/\n' > results/compiled-project/.gitignore
git init --quiet --bare results/compiled-backup.git
git init --quiet results/compiled-project
cd results/compiled-project
git config user.name "Limen documentation"
git config user.email "docs@example.invalid"
git config commit.gpgsign false
limen init scaffold.yaml --template logreg_binary
limen validate research.yaml
limen profile research.yaml
limen run --dry-run research.yaml
limen commit research.yaml --message "docs: record the research definition"
limen ls
for definition in manifests/committed/*.yaml; do
  identity=$(basename "$definition" .yaml)
done
limen lineage "$identity"
limen fork "$identity" recorded-child
cp "$bundle/compiled-research/project_child.yaml" manifests/recorded-child-production.yaml
limen commit manifests/recorded-child-production.yaml --parent "sha256:$identity" --message "docs: record the child definition"
limen reindex
limen run --no-progress-bar "manifest://sha256:$identity"
limen backup
cd "$bundle"
limen new results/compiled-restored --from "$bundle/results/compiled-backup.git"
```

The project script preserves its source YAML and uses a local Git remote. Runtime outputs and guarded holdout history have their own preservation requirements; Git backup follows the project's ignore rules. The template scaffold is separate from the recorded experiment.

A multi-file run stops after a signalled interruption and retains the current experiment's checkpoint; later YAML files are not launched.

Fresh runs check for existing run artifacts before copying their YAML declaration. Use `--resume` when the saved recording policy permits recovery, or choose a new output path. A directory containing only an old YAML copy still accepts a fresh run; an existing file at the declaration's copy destination is replaced.

## Sensor and Cohort assessment

`limen assess YAML --results DIRECTORY` requires the exact declaration retained in that directory. A guarded assessment also requires the supplied reservation to contain the YAML-declared final window. Its `assessment` block names an explicit validation metric, direction and member count. Optional fold numbers select qualified recorded folds. The command reconstructs and validates each selected Sensor, binds the Cohort and writes `assessment.json`. `predict: true` additionally writes `predictions.parquet` with decision timestamps, unavailable reasons and optional native availability values.

<!-- example: cli-assess-worked source=docs/examples/commands.json region=compiled-assess scenario=search interface=compiled role=executed -->
```bash
limen assess compiled-research/research.yaml --results results/dev/compiled-research
```

## Guarded final assessment

`limen reserve YAML` reserves the YAML's final interval in its Git project before research. It prints the reservation identity to standard output; validation messages use standard error. `limen assess YAML --results DIRECTORY --reservation ID` freezes the selected Cohort and writes the native guarded `final-report.json`. Ordinary prediction is unavailable for a declaration with a guarded final assessment.

<!-- example: cli-holdout-worked source=docs/examples/compiled-research/final/run.sh region=final_assessment scenario=holdout interface=compiled role=executed -->
```bash
git init --quiet compiled-research/final
reservation=$(limen reserve compiled-research/final/holdout.yaml)
limen run --no-progress-bar compiled-research/final/holdout.yaml
limen assess compiled-research/final/holdout.yaml --results results/dev/compiled-holdout --reservation "$reservation"
```

The [event-contract variant](Final-Holdout.md#preserve-event-economics) declares one selected Sensor and `holdout.scenarios: []`. The CLI freezes its recorded event policy and round identity, then writes the schema-2 execution report.

## Commands

| Command | Purpose | Main output |
| --- | --- | --- |
| `limen assess <yaml_file> --results <directory>` | Select validation-ranked rounds, validate reconstructed Sensors and assess their Cohort. | `assessment.json`, optional `predictions.parquet` or guarded `final-report.json`. |
| `limen reserve <yaml_file>` | Reserve the declared final interval before research. | Reservation identity and native project history. |
| `limen validate <yaml_file>` | Validate YAML syntax, schema, references, parameter lists, split settings, search strategy, and output format. | Exit code plus validation messages. |
| `limen profile <yaml_file>` | Estimate parameter-space size and sample runtime against the real data source. | Complexity rating, parameter counts, per-permutation runtime estimate, and data-quality warnings. |
| `limen run <target>` | Validate, compile, and execute a YAML file or a committed manifest (`sha256:` / `manifest://sha256:`). | Results directory with `results.csv`, optional `results.parquet`, metadata, round data, and checkpoints when configured. |
| `limen run --dry-run <target>` | Validate and compile without executing permutations. | Compile success or validation/runtime setup errors. |
| `limen run --resume <results_dir>` | Resume a saved YAML/MSQ run from a checkpoint directory. | Updated result artifacts in the existing run directory. |
| `limen init <output> [--template <name>]` | Scaffold a YAML file from a template; resolves a bare name into `manifests/` inside a project and prompts for a template when omitted. | New YAML file. |
| `limen list-templates` | List bundled YAML templates under `limen/yaml/templates`. | Template names. |
| `limen commit <yaml_file>` | Validate, content-address, store, index, and git-commit a manifest inside a Limen project. | Committed manifest under `manifests/committed/` plus index update. |
| `limen ls` | List committed manifests in the current project. | Short id, name, commit timestamp, and parent. |
| `limen fork <manifest_id> [name]` | Copy a committed manifest into `manifests/` as a development working file with lineage to its parent. | New working YAML with `lineage.parent_id`. |
| `limen lineage <manifest_id>` | Show the parent chain for a committed manifest. | Lineage tree, root-first, with the target marked. |
| `limen reindex` | Rebuild `manifests/committed/index.json` from the committed files. | Repaired index. |
| `limen new <project_name>` | Create a project from the official template, or restore one with `--from <remote>`. | New project directory. |
| `limen backup` | Snapshot the project and push it to the configured `backup_remote`. | Project pushed to the backup remote. |

## Command Details

### `limen validate <yaml_file>`

Runs YAML parsing plus structural validation. It checks required top-level blocks, schema version, `metadata.mode`, manifest type, supported output formats, search strategy values, `split_dates`, parameter lists, and resolvability of `limen.*` dotted references.

Exit behavior:

- `0` when validation succeeds
- `1` when parsing or validation fails

### `limen profile <yaml_file>`

Validates and compiles the YAML, then reports parameter-space size and complexity. It also fetches the manifest's real data source and runs a covering-array sample of permutations to estimate per-permutation runtime and surface data-quality warnings (e.g. unexpected nulls). Strict mode is temporarily disabled during sampling so thin parameter combinations do not abort profiling.

### `limen run <yaml_file> [<yaml_file> ...]`

Execute one or more YAML declarations sequentially. A failure stops before the next file. Resume accepts one results directory and cannot be combined with YAML targets. Each declaration follows the same execution path:

1. parse the YAML file
2. validate structure and references
3. compile YAML into a manifest-backed SFD
4. build the configured search strategy
5. run UEL
6. write result artifacts

Options:

| Option | Behavior |
| --- | --- |
| `--dry-run` | validate and compile only; no permutations execute |
| `--resume <results_dir>` | resume from a checkpoint directory; cannot be combined with a YAML file |
| `--no-progress-bar` | disable the experiment progress bar |
| `--workers <positive integer>` | override `uel.workers` for a run; on resume, override the saved worker count under the same parallel admission rules |

The normal result directory contains:

| File | Meaning |
| --- | --- |
| copied YAML manifest | the source manifest copied beside the run outputs; committed manifest URI runs use `manifest.yaml` |
| `metadata.json` | experiment metadata, including `yaml_reference` for Trainer and resume flows |
| `results.csv` | streaming round log |
| `round_data.jsonl.gz` | round params, predictions, alignment metadata, and [optional execution](Universal-Experiment-Loop.md#record-execution) |
| `checkpoint.json` | checkpoint state when checkpointing has run |
| `audit.jsonl` | feedback-cycle audit trail when feedback has run |

### `limen commit <yaml_file>`

Commits a manifest into a project-local content-addressed store. The YAML file must be inside a Limen project root containing `limen.toml`, and `metadata.mode` must be `production`. The command validates the YAML, writes `manifests/committed/<hex>.yaml` where `<hex>` is the manifest id without the `sha256:` prefix, updates `manifests/committed/index.json`, and then attempts a git commit in the project.

Failure boundaries:

- `--parent` must be `sha256:<64-hex-chars>`.
- validation or non-production mode stops before store mutation.
- an existing manifest repairs the store index when needed and may create a repair commit.
- git failure after store write is reported as a warning; the command still returns success because the manifest is stored but not version-controlled.

Options:

| Option | Behavior |
| --- | --- |
| `--parent <manifest_id>` | records lineage to an earlier committed manifest |
| `-m`, `--message <message>` | overrides the generated git commit message |

### `limen init <output> [--template <name>]`

Copies a bundled YAML template to `output` and updates `metadata.name` to the output filename stem. The `.yaml` extension is optional, and a bare name is resolved into the project's `manifests/` directory when run inside a Limen project (a warning is printed otherwise). Without `--template`, the command lists templates and prompts for one, defaulting to `logreg_binary`.

### `limen list-templates`

Lists package-data templates under `limen/yaml/templates`. Use the command's output for the current template inventory; [Built-In SFDs](Built-In-SFDs.md) describes their model families. `tabpfn_binary` is listed without importing TabPFN but requires the optional `tabpfn` extra before execution.

### `limen ls`

Reads `manifests/committed/index.json` from the current project and prints committed manifest metadata. Run it from inside a Limen project directory.

### `limen new <project_name>`

Creates a new project from the official project template. New projects are initialized on the `main` branch so they line up with an empty backup remote (which GitHub also defaults to `main`).

Options:

| Option | Behavior |
| --- | --- |
| `--backup-remote <url>` | sets a backup remote during project creation |
| `--from <remote_url>` | restores a project by cloning a backup remote instead of scaffolding from the template; cannot be combined with `--backup-remote` |

### `limen fork <manifest_id> [name]`

Copies a committed manifest into `manifests/` as a development-mode working file and records the source as `lineage.parent_id`. When `name` is omitted, it prompts with an auto-incremented default (e.g. `<parent>_v2`). On the next `limen commit`, the parent lineage is preserved automatically.

### `limen lineage <manifest_id>`

Walks the `parent_id` chain from a committed manifest to its root and prints it root-first, with the target manifest marked.

### `limen reindex`

Rebuilds `manifests/committed/index.json` from the committed manifest files (the source of truth). Use it to repair the index after a bad merge or a pull that left the index out of sync. Only the index is rewritten; committed manifests are never modified.

### `limen backup`

Snapshots the project — stages and commits the current state, honoring `.gitignore` so development runs under `results/dev/` stay local — then pushes to `backup_remote` from `limen.toml`. It never force-pushes: on a rejected push it explains the remote has diverged and points to the raw `git push` command. Restore on another machine with `limen new <name> --from <remote_url>`.

## YAML Run Contract

`limen run` accepts either a YAML file path or a committed manifest reference — a bare `sha256:<hash>` (the form `limen commit` prints) or a full `manifest://sha256:<hash>` URI. The command resolves the manifest, validates it, compiles all `limen.*` references, builds the parameter search domain, and then runs UEL.

Committed manifest URIs may use a full `manifest://sha256:<64-hex>` ID or an unambiguous short prefix. Resolution happens against the current project-local manifest store. A short prefix is expanded to the stored full manifest file name before execution; the result path uses the first eight hex characters, the copied YAML is named `manifest.yaml`, and `metadata.json` records the full canonical `manifest_id`. Use the full URI in external provenance records.

`metadata.mode` controls the default result path:

| Mode | Default output |
| --- | --- |
| `development` | `./results/dev/{name}_{datetime}/results.csv` |
| `production` | `./results/{name}_{datetime}/results.csv` |
| committed manifest URI | `./results/[dev/]<short-hash>/<timestamp>/results.csv` |

Set `uel.output_format: parquet` to also write `results.parquet` after successful execution. A successful `--resume` uses this setting from the saved YAML and writes or refreshes the file in the supplied results directory, including when no new rounds run. Rejected or interrupted resumes leave any existing Parquet file unchanged.

Set `uel.output_path` to a relative template within the direct run's mode directory;
see the [output destination contract](Experiment-Manifest.md#output-destinations)
for containment, relocation and resume behavior. Committed manifest URI runs
ignore it and use the hash/timestamp path under the project `results` directory.

## Validation Boundaries

`limen validate` and the validation step inside `limen run` check structure and resolvability before execution. They do not prove that custom callables have predictive value, that a remote data source is available at run time, or that a production run is valid for downstream promotion.

Runtime boundaries:

- `limen run --resume` requires full [recording evidence](Universal-Experiment-Loop.md#recording-policy) enabled on the initial run, an existing results directory with `metadata.json` containing `yaml_reference`, a readable checkpoint, consistent `round_data.jsonl.gz` through the checkpoint round, and `results.csv`.
- `limen run --dry-run` does not test data-source availability beyond compilation needs.
- CLI commands report process success through exit codes; they do not write an external audit record unless the command itself creates project artifacts.
- `limen commit` mutates git state in the target project; use it only when the manifest store commit is intended.

## Read Next

- [Experiment Manifest](Experiment-Manifest.md)
- [Universal Experiment Loop](Universal-Experiment-Loop.md)
- [Trainer](Trainer.md)
- [Developer Guidelines](Developer/README.md)

## Complete argument and option index

These tables come from the current Click declarations. Required values and positional arguments have no declared default. Omitted optional flags use the listed value; automatic help and version options are included. Path constraints are checked before command execution. The command descriptions above explain state changes and exit conditions.

### `limen` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `--version` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show the version and exit. |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen assess` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `yaml_file` | `{"allow_dash": false, "dir_okay": true, "exists": true, "file_okay": true, "name": "path", "param_type": "Path", "readable": true, "writable": false}` | — | true | 1 | Positional yaml_file. |
| `--results` | `{"allow_dash": false, "dir_okay": true, "exists": true, "file_okay": false, "name": "directory", "param_type": "Path", "readable": true, "writable": false}` | — | true | 1 | Positional results. |
| `--reservation` | `{"name": "text", "param_type": "String"}` | `null` | false | 1 | Reservation identity returned by limen reserve. |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen backup` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen commit` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `yaml_file` | `{"allow_dash": false, "dir_okay": true, "exists": true, "file_okay": true, "name": "path", "param_type": "Path", "readable": true, "writable": false}` | — | true | 1 | Positional yaml_file. |
| `--parent` | `{"name": "text", "param_type": "String"}` | `null` | false | 1 | Parent manifest ID for lineage tracking (e.g. sha256:abc123...). |
| `--message`, `-m` | `{"name": "text", "param_type": "String"}` | `null` | false | 1 | Git commit message. Auto-generated if omitted. |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen fork` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `manifest_ref` | `{"name": "text", "param_type": "String"}` | — | true | 1 | Positional manifest_ref. |
| `name` | `{"name": "text", "param_type": "String"}` | — | false | 1 | Positional name. |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen init` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `output` | `{"allow_dash": false, "dir_okay": true, "exists": false, "file_okay": true, "name": "path", "param_type": "Path", "readable": true, "writable": false}` | — | true | 1 | Positional output. |
| `--template` | `{"name": "text", "param_type": "String"}` | `null` | false | 1 | Template name to scaffold from (e.g. logreg_binary). |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen lineage` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `manifest_ref` | `{"name": "text", "param_type": "String"}` | — | true | 1 | Positional manifest_ref. |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen list-templates` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen ls` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen new` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `project_name` | `{"name": "text", "param_type": "String"}` | — | true | 1 | Positional project_name. |
| `--backup-remote` | `{"name": "text", "param_type": "String"}` | `null` | false | 1 | Git remote URL for manifest store backup (can also be set interactively). |
| `--from` | `{"name": "text", "param_type": "String"}` | `null` | false | 1 | Restore a project from a backup remote instead of the template. |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen profile` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `yaml_file` | `{"allow_dash": false, "dir_okay": true, "exists": true, "file_okay": true, "name": "path", "param_type": "Path", "readable": true, "writable": false}` | — | true | 1 | Positional yaml_file. |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen reindex` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen reserve` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `yaml_file` | `{"allow_dash": false, "dir_okay": true, "exists": true, "file_okay": true, "name": "path", "param_type": "Path", "readable": true, "writable": false}` | — | true | 1 | Positional yaml_file. |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen run` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `target` | `{"name": "text", "param_type": "String"}` | — | false | -1 | Positional target. |
| `--dry-run` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Validate and compile only — do not execute the experiment. |
| `--resume` | `{"allow_dash": false, "dir_okay": true, "exists": true, "file_okay": false, "name": "directory", "param_type": "Path", "readable": true, "writable": false}` | `null` | false | 1 | Resume from a checkpoint directory instead of starting fresh. |
| `--no-progress-bar` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Disable the experiment progress bar. |
| `--workers` | `{"clamp": false, "max": null, "max_open": false, "min": 1, "min_open": false, "name": "integer range", "param_type": "IntRange"}` | `null` | false | 1 | Concurrent rounds; overrides YAML, or uses saved settings with --resume. |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |

### `limen validate` options

| Argument or option | Type and constraint | Default | Required | Cardinality | Meaning |
| --- | --- | --- | --- | --- | --- |
| `yaml_file` | `{"allow_dash": false, "dir_okay": true, "exists": true, "file_okay": true, "name": "path", "param_type": "Path", "readable": true, "writable": false}` | — | true | 1 | Positional yaml_file. |
| `--help` | `{"name": "boolean", "param_type": "Bool"}` | `false` | false | 1 | Show this message and exit. |
