# `limen.cli`

> Provide the `limen` shell command for YAML validation, profiling, execution, manifest storage, and project scaffolding.

## Canonical docs

- [Command Line Interface](../../docs/Command-Line-Interface.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Universal Experiment Loop](../../docs/Universal-Experiment-Loop.md)

## What this package owns

Owns command parsing and command-specific orchestration around YAML manifests.
Does **not** own manifest semantics, experiment execution internals, model behavior, or docs-site build behavior.

## Key entry points

| Entry point | Use case | Notes |
| --- | --- | --- |
| `limen.cli.main:cli` | Click command group exported by the package script | Registered as the `limen` console script. |
| `commands/validate.py` | YAML validation from the shell | Uses `limen.yaml` validation. |
| `commands/profile.py` | Static permutation profiling for YAML manifests | Runtime sampling is skipped for validated CLI YAML. |
| `commands/run.py` | Execute a YAML experiment | Compiles YAML into a manifest and runs UEL. |
| `commands/resume.py` | Continue a checkpointed artifact-backed run | Used through `limen run --resume`. |
| `commands/init.py` and `commands/list_templates.py` | Bundled template discovery or scaffolding | Reads `limen/yaml/templates`. |
| `commands/commit.py` and `commands/ls.py` | Committed manifest storage | Writes project-local `manifests/committed` state. |
| `commands/new.py` | New project scaffold | Clones the official project template. |

## Adjacent modules

- `limen.yaml` parses, validates, compiles, stores, and profiles YAML manifests.
- `limen.experiment` executes the compiled SFD through UEL.
- `limen.sfd` supplies packaged decoder templates.

## Quick orientation

```text
cli/
|-- main.py       # Click command group and command registration
|-- git_utils.py  # git helper calls for project commands
`-- commands/
    |-- validate.py
    |-- profile.py
    |-- run.py
    |-- resume.py
    |-- init.py
    |-- list_templates.py
    |-- commit.py
    |-- ls.py
    `-- new.py
```

## Things to know

- `limen profile` is static for validated CLI YAML because YAML manifests reject `test_data_source`.
- `limen run` accepts both direct YAML files and committed `manifest://sha256:` URIs.
- `limen commit` writes the project manifest store before attempting the git commit.
- Command code should stay thin; YAML semantics belong in `limen.yaml` and execution semantics belong in `limen.experiment`.

## Read next

- [Command Line Interface](../../docs/Command-Line-Interface.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
