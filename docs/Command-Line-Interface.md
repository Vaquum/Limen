# Command Line Interface

The Limen CLI is the supported shell surface for YAML-first experiment work. It provides validation, profiling, template scaffolding, committed manifests, resumable runs, and standard result directories without Python orchestration code.

## Scope

The CLI owns project and manifest operations around declarative YAML experiments. It does not replace the Python API for custom SFD authoring, custom feature code, or direct `UniversalExperimentLoop` integration.

## Commands

| Command | Purpose | Main output |
| --- | --- | --- |
| `limen validate <yaml_file>` | Validate YAML syntax, schema, references, parameter lists, split settings, search strategy, and output format. | Exit code plus validation messages. |
| `limen profile <yaml_file>` | Estimate YAML parameter-space size without executing data or model work. | Complexity rating, parameter counts, and runtime-sampling skipped status. |
| `limen run <yaml_file>` | Validate, compile, and execute a YAML experiment. | Results directory with `results.csv`, optional `results.parquet`, metadata, round data, and checkpoints when configured. |
| `limen run --dry-run <yaml_file>` | Validate and compile without executing permutations. | Compile success or validation/runtime setup errors. |
| `limen run --resume <results_dir>` | Resume an artifact-backed run from a checkpoint directory. | Updated result artifacts in the existing run directory. |
| `limen init <output> --template <name>` | Copy a bundled YAML template and set `metadata.name` from the output filename. | New YAML file. |
| `limen list-templates` | List bundled YAML templates under `limen/yaml/templates`. | Template names. |
| `limen commit <yaml_file>` | Validate, content-address, store, index, and git-commit a manifest inside a Limen project. | Committed manifest under `manifests/committed/` plus index update. |
| `limen ls` | List committed manifests in the current project. | Short id, name, and commit timestamp. |
| `limen new <project_name>` | Create a project from the official Limen project template. | New project directory. |

## Command Details

### `limen validate <yaml_file>`

Runs YAML parsing plus structural validation. It checks required top-level blocks, schema version, `metadata.mode`, manifest type, supported output formats, search strategy values, `split_dates`, parameter lists, and resolvability of `limen.*` dotted references.

Exit behavior:

- `0` when validation succeeds
- `1` when parsing or validation fails

### `limen profile <yaml_file>`

Validates and compiles the YAML, then reports parameter-space size and complexity. Validated CLI YAML manifests do not accept `test_data_source`, so this command is static: runtime sampling is reported as skipped and no data source, preparation function, or model function is executed.

### `limen run <yaml_file>`

Runs the normal YAML execution path:

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

### `limen init <output> --template <name>`

Copies a bundled YAML template to `output` and updates `metadata.name` to the output filename stem. Without `--template`, the command lists available templates instead of guessing.

### `limen list-templates`

Lists package-data templates under `limen/yaml/templates`. Shipped templates include logreg, LightGBM, TabPFN, XGBoost, and rule-based examples when present in the installed package.

### `limen ls`

Reads `manifests/committed/index.json` from the current project and prints committed manifest metadata. Run it from inside a Limen project directory.

### `limen new <project_name>`

Creates a new project from the official project template.

Options:

| Option | Behavior |
| --- | --- |
| `--backup-remote <url>` | sets a backup remote during project creation |

## YAML Run Contract

`limen run` accepts either a YAML file path or a committed `manifest://sha256:` URI. The command resolves the manifest, validates it, compiles all `limen.*` references, builds the parameter search domain, and then runs UEL.

`metadata.mode` controls the default result path:

| Mode | Default output |
| --- | --- |
| `development` | `./results/dev/{name}_{datetime}/results.csv` |
| `production` | `./results/{name}_{datetime}/results.csv` |
| committed manifest URI | `./results/[dev/]<short-hash>/<timestamp>/results.csv` |

Set `uel.output_format: parquet` to also write `results.parquet`. Set `uel.output_path` to override the default output directory for direct YAML runs. Committed manifest URI runs ignore `uel.output_path` and always use the hash/timestamp path under the project `results` directory.

## Validation Boundaries

`limen validate` and the validation step inside `limen run` check structure and resolvability before execution. They do not prove that custom callables have predictive value, that a remote data source is available at run time, or that a production run is valid for downstream promotion.

Runtime boundaries:

- `limen run --resume` requires an existing results directory with `metadata.json` containing `yaml_reference`, a readable checkpoint, consistent `round_data.jsonl` through the checkpoint round, and `results.csv`.
- `limen run --dry-run` does not test data-source availability beyond compilation needs.
- CLI commands report process success through exit codes; they do not write an external audit record unless the command itself creates project artifacts.
- `limen commit` mutates git state in the target project; use it only when the manifest store commit is intended.

## Read Next

- [Experiment Manifest](Experiment-Manifest.md)
- [Universal Experiment Loop](Universal-Experiment-Loop.md)
- [Trainer](Trainer.md)
- [Developer Guidelines](Developer/README.md)
