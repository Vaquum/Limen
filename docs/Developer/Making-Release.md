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
| `scripts/create_release.py` | computes and validates the tag, generates notes with mechanical traceability, and publishes the GitHub release |
| `.github/workflows/pr_publish_pypi.yml` | guards filename availability, builds, attests, publishes, and attaches release assets |

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
2. reads the version from `pyproject.toml`, computes the `v<version>` tag, and validates it against `TAG_RE`; the model has no authority over identifiers
3. requires a matching `CHANGELOG.md` entry for the version and aborts before tagging when it is missing
4. exits successfully if the local or remote `v<version>` tag already exists
5. fetches `RELEASE_DOCS_URL`
6. sends the guide and up to 100 commits since the latest tag to the configured Anthropic model
7. parses the model response as prose JSON (`release_name`, `release_notes`) and rejects any identifier deviation
8. appends mechanical traceability to the notes: merged pull requests since the previous tag, the compare link, and the changelog anchor
9. prints a preview, creates an annotated tag, pushes it to `origin`, and immediately runs `gh release create`

There is no pause between the preview and publication. Do not run the script merely to preview notes.

After the automated-release run completes, `pr_publish_pypi.yml` resolves and re-validates the released tag, checks out the tag, guards that PyPI has never served the version's filenames, builds the wheel and sdist with fixed `SOURCE_DATE_EPOCH`, creates GitHub build-provenance attestations, and publishes to PyPI through trusted publishing from the `pypi` GitHub environment. After a successful publish it attaches the distributions, a CycloneDX `sbom.json`, and the `provenance.intoto.jsonl` attestation bundle to the GitHub release and appends their SHA-256 digests to the release body. Its `workflow_run` trigger resolves a tag only when the successful automated-release run's head commit is tagged; the `v*` tag ruleset ensures only that automation can create release tags. [Release Policy](Release-Policy.md) defines these controls.

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

The implemented workflows provide the tag, GitHub release, built distributions, GitHub provenance attestations (also attached as the `provenance.intoto.jsonl` release asset), release assets with a CycloneDX SBOM, SHA-256 digests in the release body, mechanical traceability (merged pull requests, compare link, changelog anchor), and the PyPI publication result. They do not generate a dependency-license report on the release or a maintainer sign-off record. Produce and attach those artifacts explicitly when the release policy requires them; do not infer their existence from a green publish workflow.

If tag creation succeeds but GitHub release creation fails, rerunning with the unchanged version exits because the tag already exists. Repair that partial release deliberately rather than expecting the script to resume it.

## Read next

- [Release Policy](Release-Policy.md)
- [Semantic versioning](../Semantic-Versioning.md)
- [Packaging](Packaging.md)
- [Developer home](README.md)
