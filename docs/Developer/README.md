# Developer home

This page routes Limen contribution work to the matching maintenance path before code, docs, or release metadata changes.

For cross-product Vaquum process and organization-wide norms, see the external [Vaquum Developer Docs](https://dev-docs.vaquum.fi/#/). Limen release and versioning contracts live in this repository.

## Prerequisites

- a Limen repository checkout
- the task's authoritative issue and repository workflow instructions

## Read this first

Before opening or updating a Limen PR:

- read the relevant Limen page for the task
- check the repo PR template and satisfy every applicable item
- update docs, changelog, tests, and version metadata when the change requires it

## Route by task

| Task | Read this next | Why |
|---|---|---|
| changing docs structure, navigation, or page roles | [Documentation System Contract](Documentation-System.md) | Defines the docs architecture, page types, site model, and rewrite rules. |
| updating reducer behavior | [Pruning Strategies](Pruning-Strategies.md) | Defines reducer semantics, YAML shape, testing, and failure modes. |
| updating or adding public functions, classes, or modules | [Writing Docstrings](Writing-Docstrings.md) | Defines Limen's docstring expectations and the repo's current house style. |
| adding a new foundational experiment template | [Contributing Foundational SFDs](Contributing-Foundational-SFDs.md) | Covers research expectations, file ownership, and review criteria for foundational SFDs. |
| preparing a release or checking release automation | [Making a Release](Making-Release.md) | Defines the local evidence required before Limen release publication. |
| changing release or publish-path controls | [Release Policy](Release-Policy.md) | Defines the publish-path gates, model authority boundary, release deliverables, and version-reuse rules. |
| changing package metadata, artifacts, extras, or packaging gates | [Packaging](Packaging.md) | Defines the wheel/sdist, dependency, reproducibility, and provenance contract. |
| deciding how to bump the version | [Semantic Versioning](../Semantic-Versioning.md) | Defines the Limen-local version surfaces and bump rules. |
| assessing recorded known risk | [Technical Debt](../TechnicalDebt.md) | Tracks accepted debt, trigger conditions, and candidate remedies. |

## Contributor workflow

1. Start with data: confirm whether the task uses bundled YAML templates, direct `HistoricalData`, or an external OHLC-compatible source.
2. Confirm the signal layer: read the relevant [Indicators](../Indicators.md), [Features](../Features.md), [Transforms](../Transforms.md), and [Scalers](../Scalers.md) pages before changing prep.
3. Confirm prep semantics: preserve split-first preparation, CCO behavior, strict-mode null handling, and train-fitted transformations.
4. Confirm the target: read [Targets](../Targets.md) and make sure fitting happens only on the training split when the target learns thresholds.
5. Confirm hyperparameters: route search-space changes through the manifest/YAML params surface unless the task is Python-extension-only.
6. Train through the current run surface: prefer YAML plus CLI for operator-facing work, and direct UEL only for Python extension or test work.
7. Benchmark outcomes with [Log](../Log.md), [Benchmark](../Benchmark.md), and [Backtest](../Backtest.md) before claiming the change improves research behavior.
8. Make code, docs, tests, changelog, and version changes together when they belong together.
9. Review the full GitHub diff before requesting review.
10. Confirm the PR template items are true, not just checked.

Canonical bootstrap:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Canonical validation paths:

```bash
python -m pytest
python -m tests.run
python -m build
python scripts/package_audit.py
ruff check .
```

`python -m tests.run` delegates to pytest collection and adds the CI runtime profile artifact. It is not a separate curated test list.

Docs-site validation:

```bash
cd docs-site
npm ci
npm run check
```

Limen does not define a tox, nox, or Makefile contract; these direct commands are the authoritative local workflow.

## Test runtime budget

- `PR Validation` publishes a `test-runtime-profile` artifact from `python -m coverage run -m tests.run`, which uses pytest collection.
- `PR Checks Runtime` enforces the suite ceiling committed in `tests/runtime_budget.json`.
- Update `tests/runtime_budget.json` only when recent green `main` CI runs show a real new baseline, and keep that evidence in the linked issue or PR.
- Do not raise the budget to absorb avoidable slow tests; every pytest test executed through `tests/run.py` is timed automatically.

## Scope notes

- `/docs` is the canonical public docs layer.
- `/docs/Developer` is the canonical Limen contributor layer.
- package `README`s under `/limen` are orientation pages, not the main contributor process docs.
- release and versioning policy lives in this repository; shared Vaquum docs are background only.

## Read next

- [Documentation System Contract](Documentation-System.md)
- [End-to-End Workflow](../End-to-End-Workflow.md)
- [Glossary](../Glossary.md)
- [Pruning Strategies](Pruning-Strategies.md)
- [Writing Docstrings](Writing-Docstrings.md)
- [Contributing Foundational SFDs](Contributing-Foundational-SFDs.md)
- [Making a Release](Making-Release.md)
- [Release Policy](Release-Policy.md)
- [Packaging](Packaging.md)
- [Semantic Versioning](../Semantic-Versioning.md)
- [Technical Debt](../TechnicalDebt.md)
