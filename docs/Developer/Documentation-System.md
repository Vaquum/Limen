# Documentation system contract

This page defines how Limen documentation is owned, written, assembled, and verified. The target is one coherent documentation product, not a collection of individually plausible pages.

## Prerequisites

- a Limen repository checkout
- Python development dependencies and Node.js 20 or later for the required verification gates

## Quality bar

Limen documentation is release-ready only when it is:

- correct against current source, exports, templates, workflows, and runtime behavior
- complete across the maintained public Markdown corpus
- coherent from product entry through workflows, reference, maintenance, and package boundaries
- runnable in the dependency environment each example declares
- consistent in terminology, units, page roles, links, and style
- mechanically protected by tests, lint, link checking, and the rendered-site build

## Source ownership

- [README.md](../../README.md) is the product home and first-success path.
- [docs/README.md](../README.md) is the public task router.
- `/docs` owns public concepts, workflows, guides, and reference.
- `/docs/Developer` owns contributor and maintainer guidance.
- `README.md` files under `/limen` explain package ownership, boundaries, and entry points; they route to canonical docs rather than duplicating them.
- `docs-site/scripts/assemble-docs.mjs` is the complete source-to-route map for the hosted site.

Author a claim once whenever practical. Secondary pages should summarize and link to the canonical explanation.

## Information architecture

The Docusaurus site presents five top-level sections:

| Section | Responsibility |
| --- | --- |
| Overview | product boundary, system story, and task routing |
| Guides | end-to-end jobs and operational workflows |
| Reference | interfaces, schemas, defaults, outputs, and edge cases |
| Developer | contribution, documentation, release, packaging, and maintenance |
| Packages | module ownership, public entry points, and nested package boundaries |

Top-level categories are collapsed by default. Nested package READMEs live under a collapsed Internal packages group so full corpus coverage does not flatten the sidebar.

## Narrative spine

Major pages must agree on this sequence:

1. `HistoricalData` or an external compatible frame supplies Bitcoin market data.
2. Optional data bars, indicators, features, transforms, scalers, and targets define the research surface.
3. A YAML manifest is the default operator-facing experiment definition.
4. The CLI validates, profiles, dry-runs, and runs that manifest.
5. Universal Experiment Loop is the engine beneath CLI execution and the direct Python extension surface.
6. Log, benchmark, and backtest surfaces explain experiment outcomes.
7. Trainer replays selected rounds with their original manifest configuration, validates metrics, and returns Sensors.
8. Cohort binds selected Sensors into a multi-member inference surface.
9. Trade decisioning and execution occur outside Limen.

## Page contracts

### Product home

- state what Limen is and is not
- show the minimum runnable install and first experiment
- name produced files
- route by reader task

### Docs hub

- route by task and audience
- show the system sequence
- link every top-level site section

### Guide

- state the job and current scope
- declare prerequisites or say that none are required
- show at least one concrete command or example
- state expected output, artifact, or observable result
- link the next task

### Reference

- state the covered surface
- document names, signatures, parameters, defaults, and return behavior
- document edge cases and optional dependencies
- distinguish public exports from module-only symbols

### Developer page

- state purpose and prerequisites
- give an executable process or checklist
- name failure and review boundaries
- link related maintenance surfaces

### Package README

- state what the package owns and does not own
- list source-true public entry points
- identify adjacent packages or optional dependencies
- link canonical public docs

## Writing rules

- Lead with current behavior and reader impact.
- Prefer exact commands, paths, values, units, and return fields over abstractions.
- Use American English for shared prose: `artifact`, `behavior`, `optimization`.
- Use canonical component capitalization: `Single-File Decoder`, `Reference Architecture`, `HistoricalData`, `Universal Experiment Loop`.
- Use `python` fences only for standalone parseable code. Use `python-fragment` for fluent-chain fragments or partial code.
- State required extras before the first example that imports an optional dependency.
- Do not present local measurements as stable API guarantees.
- Do not describe planned, historical, or external behavior as current Limen behavior.
- End task-oriented pages with an explicit next route.

## Source-backed claims

Use the narrowest authoritative source:

| Claim | Authority |
| --- | --- |
| import or export | package `__init__.py` and an import smoke test |
| callable arguments/defaults | current function or class signature |
| YAML field or template | schema/rules/compiler plus bundled template |
| reducer or scaler name | current registry |
| result field or artifact | implementation and focused test |
| package dependency | `pyproject.toml` |
| release behavior | current workflow and script |
| hosted route | assembler map and Docusaurus build |

When prose and source disagree, fix the prose or explicitly route a separate behavior defect. Documentation work must not silently change runtime contracts.

## Examples

Examples must satisfy the level they imply:

- syntax examples parse
- import examples import in the declared extras environment
- command examples use current argument order and names
- runnable workflows complete against a bounded fixture or isolated environment
- output examples contain only fields the implementation can produce in that mode

Base `Manifest` is an abstract interface whose `prepare_data()` raises `NotImplementedError`; runnable examples must instantiate `MLManifest` or `RuleBasedManifest`.

## Links and edit paths

- Local links and fragments must resolve in source and in the assembled site.
- Public links must use the canonical non-broken route shape.
- Every assembled page receives a `custom_edit_url` targeting its real source under GitHub `/edit/main/`.
- Repository files not mapped as docs may link to GitHub; maintained documentation must be assembled instead of silently falling back to a blob link.

## Site assembly

`docs-site/scripts/assemble-docs.mjs`:

1. maps every maintained source to one destination and route
2. writes source-aware front matter
3. rewrites relative documentation links to assembled destinations
4. preserves repository links only for non-doc files
5. creates collapsed category metadata

The source inventory test must fail if a maintained Markdown source is added without a route or if the map references a missing source.

## Required verification

From the repository root:

```bash
python -m pytest -q tests/test_docs_surface.py
python -m tests.run
```

From `docs-site`:

```bash
npm ci
npm run check
```

`npm run check` runs Markdown lint, assembles the corpus, and builds Docusaurus with broken-link failures enabled. The Python docs test owns source-linked semantic inventories, versions, local links and anchors, fence parsing, and abstract-example guards.

For a changed first-run path, also prove the documented install in a fresh environment. For layout or navigation changes, inspect the rendered desktop and mobile site before review.

## Review checklist

- Is every changed claim true at the named source?
- Does the page keep one primary role?
- Are prerequisites and outputs explicit?
- Do equivalent terms, units, and examples agree across pages?
- Is every maintained source assembled exactly once?
- Do edit links target real source files?
- Do docs tests, Markdown lint, and the site build pass?

## Read next

- [Docs hub](../README.md)
- [Developer home](README.md)
- [Writing docstrings](Writing-Docstrings.md)
- [Packaging](Packaging.md)
