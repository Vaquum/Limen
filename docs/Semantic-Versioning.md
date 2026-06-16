# Semantic Versioning

This page is the Limen-local versioning contract.

## Limen local scope

Use this page to decide the version bump and apply the result to Limen's local version surfaces.

Limen-local version surfaces:

| Surface | Role |
| --- | --- |
| `pyproject.toml` | canonical package version for builds and releases |
| `CHANGELOG.md` | human-readable release history |
| git tag `v<version>` | release identity expected by `scripts/create_release.py` |
| docs and package references | update only when the changed behavior affects documented public surfaces |

## Local rules

- MAJOR: incompatible public API, CLI, schema, artifact, or package-contract change.
- MINOR: new compatible public capability.
- PATCH: compatible fix, docs correction, package metadata correction, dependency/security refresh, or proof-gate change.
- Keep `pyproject.toml`, template `metadata.limen_version`, changelog entry, release tag, and release notes aligned to the same version.
- Treat docs-only changes as version-affecting when the published package metadata, README, docs artifacts, or public support/security surface changes.
- If code behavior changes, update docs and changelog in the same PR when they are part of the public surface.

## Review notes

Version review records the local-policy category, the files requiring version or changelog updates, and whether the release script reads the intended version from `pyproject.toml`.

## Read next

- [Making a Release](Developer/Making-Release.md)
- [Developer Home](Developer/README.md)
