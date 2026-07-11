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
- `typing_extensions` (PSF-2.0, pure Python, no transitive dependencies) is a runtime dependency for `@override` on Python 3.10; when the Python floor reaches 3.12 the import moves to `typing` and the dependency is removed.
- The `tabpfn` extra must be reviewed as a separate optional license surface before release because its upstream package metadata can include attribution terms beyond a plain Apache-2.0-only summary.
