# Contributing

Start with the developer docs:

- docs/Developer/README.md

## Local Setup

```bash
python3 -m pip install -e .
python3 -m pip install build pytest ruff
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
- run docs-site checks when docs-site or public docs change

Use the issue templates for bug reports, feature requests, support requests, and security reports.
