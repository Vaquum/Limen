# `limen.yaml`

> Parse, validate, compile, store, and profile declarative YAML experiment manifests.

## Canonical docs

- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Command Line Interface](../../docs/Command-Line-Interface.md)
- [Advanced Search](../../docs/Advanced-Search.md)

## What this package owns

Owns YAML schema constants, parsing, validation, object resolution, manifest compilation, profiling, and committed-manifest lookup.
Does **not** own Click command routing, UEL execution, model implementations, or feature/target business logic.

## Key entry points

| Entry point | Use it when | Notes |
| --- | --- | --- |
| `parser.py` | You need to read YAML into structured config | Handles file parsing before validation. |
| `schema.py` | You need the authoritative YAML field sets and allowed values | Defines schema version `1.0`, modes, manifest types, output formats, and search strategies. |
| `validator.py` | You need to validate structure and supported field names | Used by CLI validation and run paths. |
| `resolver.py` | You need to resolve `limen.*` dotted references | Bridges YAML text to Python callables/classes. |
| `compiler.py` | You need to compile YAML config into executable Limen objects | Produces manifest and UEL-ready configuration. |
| `profiler.py` | You need permutation and runtime estimates | Powers `limen profile`. |
| `store.py` | You need committed-manifest storage and `manifest://` lookup | Used by `limen commit`, `limen ls`, and `limen run manifest://...`. |
| `templates/` | You need shipped starting YAML files | Used by `limen init` and `limen list-templates`. |

## Adjacent modules

- `limen.cli` exposes this package through shell commands.
- `limen.experiment.manifest_core` is the Python object model that compiled YAML targets.
- `limen.sfd.foundational_sfd` mirrors several YAML templates as packaged SFDs.

## Quick orientation

```text
yaml/
|-- parser.py     # YAML text to Python data
|-- schema.py     # version, modes, manifest types, and allowed values
|-- rules.py      # validation rule implementations
|-- validator.py  # validation orchestration and result objects
|-- resolver.py   # limen.* dotted-reference resolution
|-- compiler.py   # YAML config to executable manifest-backed SFD
|-- profiler.py   # parameter-space profiling
|-- store.py      # committed manifest store and URI resolution
|-- config.py     # project-root and limen.toml helpers
`-- templates/    # bundled starter YAML manifests
```

## Things to know

- `schema_version` is currently `1.0`.
- YAML validation rejects `sfd.manifest.test_data_source`; runtime date limits come from `split_dates`.
- The resolver only treats `limen.*` references as supported dotted references.
- Store functions back `limen commit`, `limen ls`, and committed-manifest URI runs.

## Read next

- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Command Line Interface](../../docs/Command-Line-Interface.md)
