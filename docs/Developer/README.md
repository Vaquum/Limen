# Developer Home

This is the starting point for contributing to Limen itself. Use it to find the right maintenance path before you change code, docs, or release metadata.

For cross-product Vaquum process and organization-wide norms, see the external [Vaquum Developer Docs](https://dev-docs.vaquum.fi/#/). For Limen-specific contribution and maintenance rules, use the pages below as canonical.

## Read This First

Before opening or updating a Limen PR:

- read the relevant Limen page for the task you are doing
- check the repo PR template and satisfy every applicable item
- update docs, changelog, tests, and version metadata when the change requires it

## Route By Task

| If you are doing this | Read this next | Why |
|---|---|---|
| changing docs structure, navigation, or page roles | [Documentation System Contract](Documentation-System.md) | Defines the docs architecture, page types, site model, and rewrite rules. |
| updating or adding public functions, classes, or modules | [Writing Docstrings](Writing-Docstrings.md) | Defines Limen's docstring expectations and the repo's current house style. |
| adding a new foundational experiment template | [Contributing Foundational SFDs](Contributing-Foundational-SFDs.md) | Covers research expectations, file ownership, and review criteria for foundational SFDs. |
| preparing a release or checking release automation | [Making a Release](Making-Release.md) | Describes the actual release flow on `main`, preflight checks, and manual fallback. |
| deciding how to bump the version | [Semantic Versioning](../Semantic-Versioning.md) | Defines Limen's versioning policy and how it maps to actual shipped changes. |

## Common Contributor Workflow

1. Understand the affected subsystem and read the canonical page for it.
2. Make the code change, doc change, or release metadata change together when they belong together.
3. Run the relevant validation locally.
4. Review the full GitHub diff yourself before requesting review.
5. Make sure the PR template items are genuinely true, not just checked.

## Scope Notes

- `/docs` is the canonical public docs layer.
- `/docs/Developer` is the canonical Limen contributor layer.
- package `README`s under `/limen` are orientation pages, not the main contributor process docs.

## Read Next

- [Documentation System Contract](Documentation-System.md)
- [Writing Docstrings](Writing-Docstrings.md)
- [Contributing Foundational SFDs](Contributing-Foundational-SFDs.md)
- [Making a Release](Making-Release.md)
- [Semantic Versioning](../Semantic-Versioning.md)
