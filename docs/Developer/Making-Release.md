# Making a release

Limen release publication is automated and destructive: a qualifying push to `main` can create and push a tag, publish a GitHub release, build distributions, attach provenance, and publish to PyPI. Review the release candidate before merge; the release script has no approval prompt or dry-run mode.

## Prerequisites

- maintainer authority for the repository, GitHub release, and PyPI project
- a reviewed release candidate with the required version surfaces aligned
- configured GitHub Actions secrets and trusted publishing

## Release surfaces

| Surface | Role |
| --- | --- |
| `pyproject.toml` | package version and dependency metadata |
| `CHANGELOG.md` | human-readable release history |
| `CITATION.cff` | citation version and release date |
| `limen/yaml/templates/*.yaml` | embedded `metadata.limen_version` values |
| `.github/workflows/pr_post_release.yml` | runs release creation after every push to `main` |
| `scripts/create_release.py` | generates notes, creates and pushes the tag, and publishes the GitHub release |
| `.github/workflows/pr_publish_pypi.yml` | builds, attests, attaches, and publishes distributions |

The shared Vaquum guide is not the local authority, but the script requires a pinned copy as model-prompt input. `RELEASE_DOCS_URL` can override that URL. If the fetch fails, release creation stops before tagging.

## Before merging a release candidate

Confirm that one candidate commit contains:

- the intended version in `pyproject.toml`
- a matching `CHANGELOG.md` entry
- matching `CITATION.cff` version and release date
- matching `metadata.limen_version` in every bundled YAML template
- green tests, docs-site, packaging, lint, type, and security checks
- reviewed dependency licenses, including the separate optional TabPFN surface when relevant

The automation does not verify these conditions. They are pre-merge review obligations.

## Automated sequence

After a push to `main`, `pr_post_release.yml` installs the `release` extra and runs:

```bash
python scripts/create_release.py
```

The script:

1. requires `ANTHROPIC_API_KEY` and `GITHUB_TOKEN`
2. reads the version from `pyproject.toml`
3. fetches `RELEASE_DOCS_URL`
4. sends the guide and up to 100 commits since the latest tag to the configured Anthropic model
5. parses the model response as release JSON
6. prints the first 500 characters of the notes
7. exits successfully if the local or remote `v<version>` tag already exists
8. otherwise creates an annotated tag, pushes it to `origin`, and immediately runs `gh release create`

There is no pause between the preview and publication. Do not run the script merely to preview notes.

When the GitHub release is published, `pr_publish_pypi.yml` checks out the tag, builds the wheel and sdist with fixed `SOURCE_DATE_EPOCH`, creates GitHub build-provenance attestations, uploads the distributions as workflow artifacts, attaches them to the GitHub release, and publishes them to PyPI through trusted publishing. Its secondary `workflow_run` trigger resolves a tag only when the successful automated-release run's head commit is tagged.

## Inputs

Required by `scripts/create_release.py`:

- `ANTHROPIC_API_KEY`
- `GITHUB_TOKEN`
- authenticated `gh`
- git history and tags
- network access to `RELEASE_DOCS_URL`

Optional:

- `ANTHROPIC_MODEL` (defaults to the model named in the script)
- `RELEASE_DOCS_URL`

## Evidence and limitations

The implemented workflows provide the tag, GitHub release, built distributions, GitHub provenance attestations, release assets, and PyPI publication result. They do not generate a dependency-license report, SBOM, artifact-hash manifest, compare link, release-PR reference, or maintainer sign-off record. Produce and attach any of those artifacts explicitly when the release policy requires them; do not infer their existence from a green publish workflow.

If tag creation succeeds but GitHub release creation fails, rerunning with the unchanged version exits because the tag already exists. Repair that partial release deliberately rather than expecting the script to resume it.

## Read next

- [Semantic versioning](../Semantic-Versioning.md)
- [Packaging](Packaging.md)
- [Developer home](README.md)
