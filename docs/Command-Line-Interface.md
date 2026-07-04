# Command Line Interface

The Limen CLI is the supported shell surface for YAML-first experiment work. It provides validation, profiling, template scaffolding, committed manifests, resumable runs, and standard result directories without Python orchestration code.

## Scope

The CLI owns project and manifest operations around declarative YAML experiments. It does not replace the Python API for custom SFD authoring, custom feature code, or direct `UniversalExperimentLoop` integration.

## First YAML Run

```bash
limen init logreg-first.yaml --template logreg_binary
limen validate logreg-first.yaml
limen profile logreg-first.yaml
limen run --dry-run logreg-first.yaml
limen run logreg-first.yaml
```

This is the preferred user journey: create or edit YAML, validate it, profile the search space, compile it without execution, then run it through the UEL engine.

## Commands

| Command | Purpose | Main output |
| --- | --- | --- |
| `limen validate <yaml_file>` | Validate YAML syntax, schema, references, parameter lists, split settings, search strategy, and output format. | Exit code plus validation messages. |
| `limen profile <yaml_file>` | Estimate parameter-space size and sample runtime against the real data source. | Complexity rating, parameter counts, per-permutation runtime estimate, and data-quality warnings. |
| `limen run <target>` | Validate, compile, and execute a YAML file or a committed manifest (`sha256:` / `manifest://sha256:`). | Results directory with `results.csv`, optional `results.parquet`, metadata, round data, and checkpoints when configured. |
| `limen run --dry-run <target>` | Validate and compile without executing permutations. | Compile success or validation/runtime setup errors. |
| `limen run --resume <results_dir>` | Resume an artifact-backed run from a checkpoint directory. | Updated result artifacts in the existing run directory. |
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
| `--no-progress-bar` | disable the experiment progress bar |

The normal result directory contains:

| File | Meaning |
| --- | --- |
| copied YAML manifest | the source manifest copied beside the run outputs; committed manifest URI runs use `manifest.yaml` |
| `metadata.json` | experiment metadata, including `yaml_reference` for Trainer and resume flows |
| `results.csv` | streaming round log |
| `round_data.jsonl` | round params, predictions, and alignment metadata |
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

Lists package-data templates under `limen/yaml/templates`. Shipped templates include logreg, LightGBM, TabPFN, XGBoost, and rule-based manifests when present in the installed package. `tabpfn_binary` is listed as a template but requires installing the optional `tabpfn` extra before execution.

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
