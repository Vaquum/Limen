# Third-Party Notices

Limen depends on Python and JavaScript open source packages declared in:

- `pyproject.toml`
- `docs-site/package.json`
- `docs-site/package-lock.json`

Dependency license review is required before release when dependencies are added, removed, or materially upgraded.

## Current Known Notes

- Docs-site `npm audit --audit-level=high` must pass before closeout of #619.
- Docusaurus may retain upstream moderate advisories until its dependency chain releases fixed versions.
- Optional `tabpfn` support is not installed by default; install it only when the TabPFN workflow is required.
- The `tabpfn` extra must be reviewed as a separate optional license surface before release because its upstream package metadata can include attribution terms beyond a plain Apache-2.0-only summary.
