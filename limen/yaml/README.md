# `limen.yaml`

> Parse, validate, compile, store, and profile declarative YAML experiment manifests.

## Canonical docs

- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Command-Line Interface](../../docs/Command-Line-Interface.md)
- [Advanced Search](../../docs/Advanced-Search.md)

## In the research projects

The [Compiled project](../../docs/examples/compiled-research/README.md) `command_line` scenario validates, commits, forks, runs and restores its declared experiment. Its source keeps the installed CLI and the compiler on the same definition; Compiled Research authors YAML and runs the CLI. Direct compiler APIs remain optional component integration.

## What this package owns

Owns the declarative manifest surface behind the preferred CLI path: YAML schema constants, parsing, validation, object resolution, manifest compilation, profiling, and committed-manifest lookup.
Does **not** own Click command routing, UEL execution, model implementations, or feature/target business logic.

## Key entry points

Package-root exports are grouped by responsibility:

| Surface | Public exports |
| --- | --- |
| Parse, validate, resolve | `parse`, `validate`, `ValidationResult`, `resolve` |
| Compile | `CompiledSFD`, `build_manifest`, `build_search_strategy` |
| Profile | `profile`, `ProfileResult`, `make_covering_array` |
| Schema | `VERSION` |
| Project and store | `find_project_root`, `get_store_path`, `read_limen_toml`, `resolve_manifest_uri` |
| Errors | `YAMLError`, `ValidationError`, `ResolutionError`, `GitError` |

The package also owns `parser.py`, `schema.py`, `validator.py`, `resolver.py`, `compiler.py`, `profiler.py`, `store.py`, and the shipped `templates/` directory behind those exports.

## Adjacent modules

- `limen.cli` exposes this package through shell commands.
- `limen.experiment.manifest_core` is the Python object model that compiled YAML targets.
- `limen.sfd.foundational_sfd` mirrors YAML templates as packaged SFDs.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `parser.py` | YAML text to Python data |
| `schema.py` | version, modes, manifest types, and allowed values |
| `rules.py` | validation rule implementations |
| `validator.py` | validation orchestration and result objects |
| `resolver.py` | limen.* dotted-reference resolution |
| `compiler.py` | YAML config to executable manifest-backed SFD |
| `profiler.py` | parameter-space profiling |
| `store.py` | committed manifest store and URI resolution |
| `config.py` | project-root and limen.toml helpers |
| `templates/` | bundled starter YAML manifests |

<!-- reference: package-yaml-layout source=limen/yaml/README.md region=layout:package role=layout -->
```text
yaml/
├── README.md
├── __init__.py
├── _assessment_spec.py
├── _backtest_spec.py
├── _boolean_spec.py
├── _callback_spec.py
├── _feature_shim_spec.py
├── _metadata_spec.py
├── _objective_spec.py
├── _parameter_refs.py
├── _parser_yaml.py
├── _recording_spec.py
├── _schema_version.py
├── _temporal_spec.py
├── compiler.py
├── config.py
├── errors.py
├── parser.py
├── profiler.py
├── resolver.py
├── rules.py
├── schema.py
├── store.py
├── templates/
└── validator.py
```

## Things to know

- `schema_version` is currently `1.0`.
- YAML manifests are the canonical operator-facing experiment definition; Python builders are the extension-equivalent surface.
- YAML validation rejects `sfd.manifest.test_data_source`; see the [source/split profiling contract](../../docs/Command-Line-Interface.md#limen-profile-yaml_file).
- The resolver only treats `limen.*` references as supported dotted references.
- Store functions back `limen commit`, `limen ls`, and committed-manifest URI runs.

## Read next

- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Command-Line Interface](../../docs/Command-Line-Interface.md)
