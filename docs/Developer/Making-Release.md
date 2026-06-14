# Making a Release

The full release process now lives in the shared Vaquum developer docs.

Use the current guide here:

- [Making a Release in Vaquum Developer Docs](https://github.com/Vaquum/dev-docs/blob/main/src/Making-Release.md)

Limen's release automation fetches that shared guide directly when it generates release notes, so the external page is the canonical source.

## Limen Local Scope

This page is the Limen-local pointer, not a fork of the shared release process. Keep the actual release procedure in the Vaquum developer docs unless a step is specific to this repository.

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

## Failure Notes

- If the target tag already exists locally or remotely, the script skips release creation.
- If the shared release guide cannot be fetched, release-note generation fails before tagging.
- If credentials are missing, the script exits before calling the model or GitHub.
- The script does not decide the version bump; use the shared semantic-versioning policy before running it.

## Read Next

- [Semantic Versioning](../Semantic-Versioning.md)
- [Developer Home](README.md)
