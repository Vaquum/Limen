# Semantic Versioning

The shared versioning policy now lives in the Vaquum developer docs.

Use the current guide here:

- [Semantic Versioning in Vaquum Developer Docs](https://github.com/Vaquum/dev-docs/blob/main/src/Semantic-Versioning.md)

That external page is now the canonical source for version-bump policy across Vaquum projects.

## Limen Local Scope

Use this page to find the policy, then apply the result to Limen's local version surfaces.

Limen-local version surfaces:

| Surface | Role |
| --- | --- |
| `pyproject.toml` | canonical package version for builds and releases |
| `CHANGELOG.md` | human-readable release history |
| git tag `v<version>` | release identity expected by `scripts/create_release.py` |
| docs and package references | update only when the changed behavior affects documented public surfaces |

## Local Rules

- Do not invent a Limen-specific version policy here; use the shared Vaquum policy.
- Keep `pyproject.toml`, changelog entry, release tag, and release notes aligned to the same version.
- Treat docs-only changes as version-affecting only when the shared policy says the published package metadata should move.
- If code behavior changes, update docs and changelog in the same PR when they are part of the public surface.

## Review Notes

Version review should answer:

1. Which shared-policy category applies?
2. Which local files need the version or changelog update?
3. Does the release script already see the intended version in `pyproject.toml`?

## Read Next

- [Making a Release](Developer/Making-Release.md)
- [Developer Home](Developer/README.md)
