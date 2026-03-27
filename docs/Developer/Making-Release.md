# Making a Release

This page documents how releases actually work in Limen today.

The important fact is that release creation is automated on pushes to `main`. Maintainers normally prepare the release in the PR by updating version metadata and changelog content, then let the GitHub workflow create the tag and GitHub release after merge.

## Current Release Model

On every push to `main`, the workflow in `.github/workflows/pr_post_release.yml` runs `scripts/create_release.py`.

That script:

1. reads the version from `pyproject.toml`
2. reads this release guide as part of its prompting instructions
3. gathers the git log since the latest tag
4. asks the release model to produce a release title and notes
5. creates tag `v<version>` if that tag does not already exist
6. creates the GitHub release for that tag

If the tag already exists, the automation exits cleanly without creating a duplicate release.

## What Must Be Ready Before Merge

Before merging a release-bearing PR:

- `pyproject.toml` has the intended new version
- `CHANGELOG.md` is updated if the change is not docs-only or otherwise non-code
- docs are updated if public behavior, API, workflow, or outputs changed
- tests and checks are green
- the full PR diff has been reviewed carefully

## Version And Tag Rules

- Use the version in `pyproject.toml` as the release version.
- Tags must always use lowercase `v`, such as `v1.48.0`.
- Never create uppercase `V` tags.
- Limen keeps `CHANGELOG.md` in oldest-first order. New release entries should follow that convention.

## Release Notes Rules

The automated release script is prompted from this page, so these rules matter in practice.

Release notes should contain:

- `## Summary`: short bullet points covering the most important shipped changes
- `## Details`: a fuller narrative explanation of what changed and why it matters

The release title should be a creative lunar-calendar-inspired name, because that is part of the current automated release convention.

## Normal Maintainer Flow

1. Decide the correct version bump using [Semantic Versioning](../Semantic-Versioning.md).
2. Update `pyproject.toml`.
3. Update `CHANGELOG.md` if the PR is not docs-only or another non-code change.
4. Merge to `main`.
5. Watch the `Automated Release` workflow.
6. Verify that the tag and GitHub release were created as expected.

## Manual Fallback

Use manual release steps only if the automation fails or if a release needs to be backfilled.

### If the workflow failed before creating the tag

- fix the underlying problem
- rerun the workflow or re-trigger it through a new push to `main`

### If the tag exists but the GitHub release is missing

Create the release manually for the existing tag:

```bash
gh release create v<NEW_VERSION> \
  --title "<RELEASE_NAME>" \
  --notes-file /path/to/release-notes.md
```

### If you must create the tag manually

```bash
git tag -a v<NEW_VERSION> -m "Release v<NEW_VERSION>"
git push origin v<NEW_VERSION>
```

Only do this when the automated path is genuinely blocked.

## Failure Cases To Check

- version in `pyproject.toml` was not bumped, so the workflow no-ops because the tag already exists
- changelog was not updated for a shipped code change
- tag format is wrong
- release notes are misleading because the merged git log is incomplete or noisy
- release workflow failed due to missing secrets or GitHub permissions

## Read Next

- [Semantic Versioning](../Semantic-Versioning.md)
- [Developer Home](README.md)
