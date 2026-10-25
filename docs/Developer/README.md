# Developer home

Use the maintenance path that matches the change: code, documentation or release metadata\.

Follow changes through the [Python project](../examples/python-research/README.md) and [Compiled project](../examples/compiled-research/README.md). They carry the shared source, documentation and execution contract; update the affected case with the behavior it explains.

For cross-product Vaquum process and organization-wide norms, see the external [Vaquum Developer Docs](https://dev-docs.vaquum.fi/#/). Limen release and versioning contracts live in this repository.

## Prerequisites

- a Limen repository checkout
- the task's authoritative issue and repository workflow instructions

## Read this first

Before opening or updating a Limen PR:

- read the relevant Limen page for the task
- check the repo PR template and satisfy every applicable item
- update the affected project source, docs and tests together; version and changelog are cut at release time, not per PR

## Route by task

| Task | Read this next | Why |
|---|---|---|
| planning or delivering research, maintenance or private prerequisites | [Constitution](https://github.com/Vaquum/Limen/blob/main/CLAUDE.md) | Defines the delivery laws: every PR closes one issue, `feat` work goes through a PRD. |
| changing docs structure, navigation, or page roles | [Documentation System Contract](Documentation-System.md) | Defines the docs architecture, page types, site model, and rewrite rules. |
| updating reducer behavior | [Pruning Strategies](Pruning-Strategies.md) | Defines reducer semantics, YAML shape, testing, and failure modes. |
| updating or adding public functions, classes, or modules | [Writing Docstrings](Writing-Docstrings.md) | Defines Limen's docstring expectations and the repo's current house style. |
| adding a new foundational experiment template | [Contributing Foundational SFDs](Contributing-Foundational-SFDs.md) | Covers research expectations, file ownership, and review criteria for foundational SFDs. |
| preparing a release or checking release automation | [Making a Release](Making-Release.md) | Defines the local evidence required before Limen release publication. |
| changing release or publish-path controls | [Release Policy](Release-Policy.md) | Defines the publish-path gates, model authority boundary, release deliverables, and version-reuse rules. |
| changing package metadata, artifacts, extras, or packaging gates | [Packaging](Packaging.md) | Defines the wheel/sdist, dependency, reproducibility, and provenance contract. |
| measuring or optimizing indicator calculations | [Indicator Performance](Indicator-Performance.md#performance-measurements) | Records real-data timings, memory, exact equivalence and regression checks. |
| cutting a release version | [Semantic Versioning](../Semantic-Versioning.md) | Defines the Limen-local version surfaces and release-time rules. |
| assessing recorded known risk | [Technical Debt](../TechnicalDebt.md) | Tracks accepted debt, trigger conditions, and candidate remedies. |

## Contributor workflow

The [constitution](https://github.com/Vaquum/Limen/blob/main/CLAUDE.md) is canonical. Every PR closes exactly one issue — a PRD for `feat` work, any issue otherwise. All PRs satisfy the required checks.

For research behavior changes:

1. Identify the interface outcomes: caller-owned Python SFDs through UEL, or the Manifest/CLI contract for Compiled Research. Verify the Compiled claim through unchanged YAML and actual CLI execution; Python component APIs are optional integration surfaces. Confirm the source: bundled templates, direct `HistoricalData`, or caller-owned data.
2. Confirm the signal layer: read the relevant [Indicators](../Indicators.md), [Features](../Features.md), [Transforms](../Transforms.md), and [Scalers](../Scalers.md) pages before changing prep.
3. For Compiled Research prep, preserve split-first preparation, CCO behavior, strict-mode null handling, and train-fitted transformations. Python Research keeps methodology caller-owned.
4. For target changes, read [Targets](../Targets.md); preserve training-only threshold fitting in the governed pipeline.
5. Declare Compiled Research search spaces through Manifest/YAML params; preserve caller-owned parameter and search choices in Python Research.
6. Run Compiled experiments through unchanged YAML and CLI, or caller-owned Python SFDs through UEL with a search strategy. Both routes share MSQ execution machinery.
7. Benchmark outcomes with [Log](../Log.md), [Benchmark](../Benchmark.md), and [Backtest](../Backtest.md) before claiming the change improves research behavior.
8. Make affected code, canonical examples, docs and tests change together. Leave version surfaces alone; they move only at release time.
9. Review the full GitHub diff before requesting review.
10. Confirm the PR template items are true, not just checked.

Complete single-interface delivery may ship while another interface remains unavailable. Private groundwork must not expose an unfinished public capability.

Canonical bootstrap:

| Step | Command |
|---|---|
| 1 | `python3 -m venv .venv` |
| 2 | `source .venv/bin/activate` |
| 3 | `python -m pip install --upgrade pip` |
| 4 | `python -m pip install -e ".[dev]"` |

Package validation commands (run from the repository root; the exact required planes are listed in [CLAUDE.md](https://github.com/Vaquum/Limen/blob/main/CLAUDE.md#the-laws)):

| Step | Command |
|---|---|
| 1 | `python -m pytest` |
| 2 | `python -m tests.run` |
| 3 | `python -m build` |
| 4 | `python scripts/package_audit.py` |
| 5 | `ruff check limen governance tests` |

`python -m tests.run` delegates to complete pytest collection in four separate worker processes, groups each file on one worker, and disables automatic worker restart. All explicit arguments pass through unchanged: a file path alone retains pytest's serial selection, `-n 0 tests` selects the complete serial suite, and explicit worker counts or distribution modes remain caller-owned.

The existing `python -m coverage run -m tests.run` command starts real coverage collectors before worker collection and merges their databases after workers terminate. Python 3.12 CI with `LIMEN_COVERAGE_RUN=1 COVERAGE_CORE=sysmon` requires actual SysMonitor collectors. The profile retains native collection and execution counts, worker PIDs, backend names, database and executed-line digests, and combination time. Default file distribution requires exactly one completion per collected case; explicit `--dist each` retains each worker's execution separately. Failed or interrupted runs retain their nonzero result and partial profile.

Covered idle workers and workers executing only data-independent test utilities can have genuine empty package coverage. Their databases and receipts are still required. A temporary `no-data-collected` option is confined to each collector's empty-data save or merge boundary and restored immediately; project warning policy remains unchanged. The complete proof still requires the runtime ceiling in `tests/runtime_budget.json` and the repository’s coverage floor. An individual timing establishes that run's result, without a future-runtime guarantee.

Docs-site validation also needs the successful same-head Python research evidence produced by the [documentation workflow](https://github.com/Vaquum/Limen/blob/main/.github/workflows/pr_checks_docs_site.yml). Its offline wheel preparation, source-provenance acquisition and project proof precede the reader build; `npm run check` alone does not replace that proof. From `docs-site/`:

| Step | Command |
|---|---|
| 1 | `npm ci` |
| 2 | `npm run check` |

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

The [indicator performance evidence](Indicator-Performance.md) retains calculation measurements and the numerical-parity conditions that changes to project preparation must preserve.
