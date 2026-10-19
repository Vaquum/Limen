# Contributing

Start with [CLAUDE.md](CLAUDE.md) and the [developer docs](docs/Developer/README.md). Changes to research examples and the reader follow the [documentation system contract](docs/Developer/Documentation-System.md).

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Canonical local validation:

```bash
python -m tests.run
python -m build
ruff check .
```

Docs-site checks live under `docs-site`:

```bash
cd docs-site
npm ci
npm run check
```

The candidate reader also requires current execution evidence from both projects. Follow the [reader validation procedure](docs/Developer/Documentation-System.md#validating-the-candidate-reader) for research dependencies, source provenance, wheel execution and browser checks. A retained renderer build alone does not validate the candidate reader.

## Pull Requests

Commit a useful increment, push it and open a non-draft PR before local checks, as required by [CLAUDE.md](CLAUDE.md#workflow). Then:

- read the relevant docs page for the touched subsystem
- update tests and docs with the behavior change
- update `CHANGELOG.md`, package version, citation and embedded version carriers on every PR
- run `python3 -m tests.run`
- run `python3 -m build` when packaging metadata, package data, or README links change
- run `ruff check .` when Python source changes
- run docs-site checks when docs-site or public docs change

Use the issue templates for bug reports, feature requests, support requests, and security reports.
