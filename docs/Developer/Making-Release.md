# Making a Release

This page is the Limen-local release contract. Shared Vaquum release docs may provide organization-wide background, but Limen release evidence must be reproducible from this repository.

## Limen Local Scope

Keep shared process references as background only. The local surfaces below are the authoritative Limen release inputs.

Limen-local release surfaces:

| Surface | Role |
| --- | --- |
| `pyproject.toml` | source of the package version used by the release script |
| `CHANGELOG.md` | human-readable release history expected by PR discipline |
| `scripts/create_release.py` | local automation for generating and publishing a GitHub release |
| `RELEASE_DOCS_URL` | optional override for the shared release guide fetched by the script |

## Automation Inputs

`scripts/create_release.py` expects:

- `ANTHROPIC_API_KEY`
- `GITHUB_TOKEN`
- optional `ANTHROPIC_MODEL`
- optional `RELEASE_DOCS_URL`
- a version already set in `pyproject.toml`
- git history and tags available locally
- `gh` authenticated enough to create the GitHub release

The script creates a tag with the `v<version>` shape and uses GitHub release notes generated from the shared release guide plus git history since the latest tag.

Release notes must be tied to local evidence before publication:

- `CHANGELOG.md` entry for the version
- release PR number and merge SHA
- compare link from previous tag to new tag
- built artifact names and hashes
- CI run links for tests, package build, docs-site build, and docs-site audit where applicable

## Failure Notes

- If the target tag already exists locally or remotely, the script skips release creation.
- If the shared release guide cannot be fetched, release-note generation fails before tagging.
- If credentials are missing, the script exits before calling the model or GitHub.
- The script does not decide the version bump; use the Limen-local [Semantic Versioning](../Semantic-Versioning.md) contract before running it.
- LLM-generated prose is draft material until a maintainer checks it against the changelog, merge SHA, compare link, artifact hashes, and CI evidence.

## Read Next

- [Semantic Versioning](../Semantic-Versioning.md)
- [Developer Home](README.md)
