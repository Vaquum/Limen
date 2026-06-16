# Contributing

Start with the developer docs:

- docs/Developer/README.md

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

## Pull Requests

Before opening a PR:

- read the relevant docs page for the touched subsystem
- update tests and docs with the behavior change
- update `CHANGELOG.md` and version metadata when public package behavior or metadata changes
- run `python3 -m tests.run`
- run `python3 -m build` when packaging metadata, package data, or README links change
- run `ruff check .` when Python source changes
- run docs-site checks when docs-site or public docs change

Use the issue templates for bug reports, feature requests, support requests, and security reports.
