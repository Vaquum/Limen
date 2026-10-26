# `limen.cli.commands`

> One thin handler per `limen` subcommand, keeping the Click group in `main.py` declarative.

## Canonical docs

- [Command-Line Interface](../../../docs/Command-Line-Interface.md)

## In the research projects

The [Compiled project](../../../docs/examples/compiled-research/README.md) `command_line` scenario exercises the installed handlers through their public commands. Its assertions check exit status, parent/child lineage, completed result identities and restored campaign bytes.

## What this package owns

Owns the implementation body behind the project and run subcommands: argument handling, calls into `limen.yaml` and `limen.experiment`, and console output.
Does **not** own the Click command declarations (those live in `limen/cli/main.py`), manifest semantics, or experiment execution internals.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `run.py:run_experiment()` | Compile a YAML manifest and run the Universal Experiment Loop | Backs `limen run`; accepts YAML files and `manifest://sha256:` URIs |
| `resume.py:run_resume()` | Continue a saved YAML/MSQ run from a checkpoint | Backs `limen run --resume` |
| `validate.py:run_validate()` | Validate a YAML manifest | Backs `limen validate` |
| `profile.py:run_profile()` | Parameter-space and runtime profiling | Backs `limen profile`; samples the configured data source with a covering array |
| `init.py` / `new.py` / `list_templates.py` | Scaffold projects and discover bundled templates | Reads `limen/yaml/templates` |
| `commit.py` / `ls.py` / `fork.py` / `lineage.py` / `backup.py` / `reindex.py` | Committed manifest storage and lineage | Operate on project-local `manifests/committed` state |

`limen assess` and `limen reserve` are registered and implemented in the sibling `limen.cli.assessment` module. They are part of the public CLI; they are not handlers in this directory.

## Adjacent modules

- `limen.cli.main` declares the Click group and registers every handler here.
- `limen.yaml` parses, validates, compiles, stores, and profiles the manifests these commands act on.
- `limen.experiment` executes the compiled SFD through UEL.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `_constants.py` | Shared command constants |
| `_export_results.py` | Shared successful-run Parquet export |
| `_load_yaml.py` | Shared YAML loading helper |
| `_results_dir.py` | Direct-YAML output containment |
| `_resume_admission.py` | Saved schema, callback and structural admission |
| `run.py` | limen run |
| `resume.py` | limen run --resume |
| `validate.py` | limen validate |
| `profile.py` | limen profile |
| `init.py` | limen init |
| `new.py` | limen new |
| `list_templates.py` | limen list-templates |
| `commit.py` | limen commit |
| `ls.py` | limen ls |
| `fork.py` | limen fork |
| `lineage.py` | limen lineage |
| `backup.py` | limen backup |
| `reindex.py` | limen reindex |

<!-- reference: package-cli-commands-layout source=limen/cli/commands/README.md region=layout:package role=layout -->
```text
commands/
├── README.md
├── __init__.py
├── _constants.py
├── _export_results.py
├── _load_yaml.py
├── _results_dir.py
├── _resume_admission.py
├── backup.py
├── commit.py
├── fork.py
├── init.py
├── lineage.py
├── list_templates.py
├── ls.py
├── new.py
├── profile.py
├── reindex.py
├── resume.py
├── run.py
└── validate.py
```

## Things to know

- Handlers should stay thin. YAML semantics belong in `limen.yaml` and execution semantics belong in `limen.experiment`.
- Each handler returns a `bool` (or `None`) and prints its own operator-facing output; `main.py` maps that to the process exit code.
- Modules prefixed with `_` are shared internals, not commands.

## Read next

- [Command-Line Interface](../../../docs/Command-Line-Interface.md)
- [Experiment Manifest](../../../docs/Experiment-Manifest.md)
