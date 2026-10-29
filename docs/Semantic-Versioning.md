# Semantic Versioning

This page is the Limen-local versioning contract.

## Limen local scope

Use this page to cut the release version and apply it to Limen's release surfaces. Both the [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) execute against that package identity; their source and results must identify the version they actually validate.

## Prerequisites

- the approved change scope and its public compatibility impact
- the current package version, changelog, citation metadata, templates, and release state

Limen-local version surfaces:

| Surface | Role |
| --- | --- |
| `pyproject.toml` | canonical package version for builds and releases |
| `CHANGELOG.md` | human-readable release history |
| `CITATION.cff` | citation version and release date |
| git tag `v<version>` | release identity expected by `scripts/create_release.py` |
| docs and package references | update only when the changed behavior affects documented public surfaces |

## Local rules

- MAJOR: incompatible public API, CLI, schema, artifact, or package-contract change.
- MINOR: new compatible public capability.
- PATCH: compatible fix, docs correction, package metadata correction, dependency/security refresh, or proof-gate change.
- Keep `pyproject.toml`, `CITATION.cff`, template `metadata.limen_version`, changelog entry, release tag, and release notes aligned to the same version.
- Ordinary PRs do not touch version surfaces. The release cuts the package version and changelog entry under [repository law 5](../.github/REPOSITORY-LAW.md#the-laws). Update affected public docs in the PR that changes the behavior they describe.

## Documentation compatibility

A reader or content revision identifies its source tree separately from the package version it validates. Example and evidence schema changes require their own compatibility decision: update consumers, exact bindings and publication checks together. A source hash identifies bytes; current execution against the matching wheel establishes compatibility.

## Review notes

Version review records the local-policy category, the files requiring version or changelog updates, and whether the release script reads the intended version from `pyproject.toml`.

## Read next

- [Making a Release](Developer/Making-Release.md)
- [Developer Home](Developer/README.md)
