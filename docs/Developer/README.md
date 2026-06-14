# Developer home

This page routes Limen contribution work to the matching maintenance path before code, docs, or release metadata changes.

For cross-product Vaquum process and organization-wide norms, see the external [Vaquum Developer Docs](https://dev-docs.vaquum.fi/#/). Release process and versioning guidance now also live there. The pages below cover Limen-specific contribution and maintenance rules that still belong in this repo.

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
| preparing a release or checking release automation | [Making a Release](https://github.com/Vaquum/dev-docs/blob/main/src/Making-Release.md) | Uses the shared Vaquum release process that the Limen release script now fetches directly. |
| deciding how to bump the version | [Semantic Versioning](https://github.com/Vaquum/dev-docs/blob/main/src/Semantic-Versioning.md) | Uses the shared Vaquum versioning guidance rather than a repo-local copy. |
| assessing recorded known risk | [Technical Debt](../TechnicalDebt.md) | Tracks accepted debt, trigger conditions, and candidate remedies. |

## Contributor workflow

1. Understand the affected subsystem and read the canonical page for it.
2. Make the code change, doc change, or release metadata change together when they belong together.
3. Run the relevant validation locally.
4. Review the full GitHub diff before requesting review.
5. Confirm the PR template items are true, not just checked.

## Test runtime budget

- `PR Validation` now publishes a `test-runtime-profile` artifact from the canonical `python -m coverage run -m tests.run` path.
- `PR Checks Runtime` enforces the suite ceiling committed in `tests/runtime_budget.json`.
- Update `tests/runtime_budget.json` only when recent green `main` CI runs show a real new baseline, and keep that evidence in the linked issue or PR.
- Do not raise the budget to absorb avoidable slow tests; everything executed through `tests/run.py` is timed automatically.

## Scope notes

- `/docs` is the canonical public docs layer.
- `/docs/Developer` is the canonical Limen contributor layer.
- package `README`s under `/limen` are orientation pages, not the main contributor process docs.
- release and versioning policy lives in the shared Vaquum developer docs; Limen pages record local release surfaces, automation inputs, and review notes.

## Read next

- [Documentation System Contract](Documentation-System.md)
- [Pruning Strategies](Pruning-Strategies.md)
- [Writing Docstrings](Writing-Docstrings.md)
- [Contributing Foundational SFDs](Contributing-Foundational-SFDs.md)
- [Making a Release](https://github.com/Vaquum/dev-docs/blob/main/src/Making-Release.md)
- [Semantic Versioning](https://github.com/Vaquum/dev-docs/blob/main/src/Semantic-Versioning.md)
- [Technical Debt](../TechnicalDebt.md)
