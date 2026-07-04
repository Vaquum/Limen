# `limen.cli.commands`

> One thin handler per `limen` subcommand, keeping the Click group in `main.py` declarative.

## Canonical docs

- [Command Line Interface](../../../docs/Command-Line-Interface.md)

## What this package owns

Owns the implementation body behind each `limen` subcommand: argument handling, calls into `limen.yaml` and `limen.experiment`, and console output.
Does **not** own the Click command declarations (those live in `limen/cli/main.py`), manifest semantics, or experiment execution internals.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `run.py:run_experiment()` | Compile a YAML manifest and run the Universal Experiment Loop | Backs `limen run`; accepts YAML files and `manifest://sha256:` URIs |
| `resume.py:run_resume()` | Continue a checkpointed artifact-backed run | Backs `limen run --resume` |
| `validate.py:run_validate()` | Validate a YAML manifest | Backs `limen validate` |
| `profile.py:run_profile()` | Static permutation profiling | Backs `limen profile`; runtime sampling is skipped for validated CLI YAML |
| `init.py` / `new.py` / `list_templates.py` | Scaffold projects and discover bundled templates | Reads `limen/yaml/templates` |
| `commit.py` / `ls.py` / `fork.py` / `lineage.py` / `backup.py` / `reindex.py` | Committed manifest storage and lineage | Operate on project-local `manifests/committed` state |

## Adjacent modules

- `limen.cli.main` declares the Click group and registers every handler here.
- `limen.yaml` parses, validates, compiles, stores, and profiles the manifests these commands act on.
- `limen.experiment` executes the compiled SFD through UEL.

## Quick orientation

```text
commands/
├── _constants.py        # Shared command constants
├── _load_yaml.py        # Shared YAML loading helper
├── run.py               # limen run
├── resume.py            # limen run --resume
├── validate.py          # limen validate
├── profile.py           # limen profile
├── init.py              # limen init
├── new.py               # limen new
├── list_templates.py    # limen list-templates
├── commit.py            # limen commit
├── ls.py                # limen ls
├── fork.py              # limen fork
├── lineage.py           # limen lineage
├── backup.py            # limen backup
└── reindex.py           # limen reindex
```

## Things to know

- Handlers should stay thin. YAML semantics belong in `limen.yaml` and execution semantics belong in `limen.experiment`.
- Each handler returns a `bool` (or `None`) and prints its own operator-facing output; `main.py` maps that to the process exit code.
- Modules prefixed with `_` (`_constants.py`, `_load_yaml.py`) are shared internals, not commands.

## Read next

- [Command Line Interface](../../../docs/Command-Line-Interface.md)
- [Experiment Manifest](../../../docs/Experiment-Manifest.md)
