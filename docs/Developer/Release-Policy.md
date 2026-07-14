# Release policy

This page is the repo-local contract for what may publish Limen, what a release must carry, and what may never happen twice. [Making a Release](Making-Release.md) describes the operational sequence; this page defines the controls that sequence runs under.

## Prerequisites

- the operational release sequence in [Making a Release](Making-Release.md)
- maintainer authority for repository settings and the PyPI project when changing a control

## Version rules

[Semantic versioning](../Semantic-Versioning.md) defines the bump semantics. This policy adds one rule with no exceptions: a version that has ever been tagged or uploaded is burned. PyPI permanently rejects any filename it has ever accepted, even after the file is deleted, and deleted releases disappear from PyPI's JSON API, so a reused version fails in the least visible way possible. Never reuse a version; always bump past it.

## Publish-path gates

The publish path is gated as strongly as the merge path. Every control is mechanical:

| Control | Mechanism |
| --- | --- |
| merge to `main` | required status checks on the `Protect-Main` ruleset, including the `pr_checks_supply` dependency vulnerability gate |
| tag creation | a `v*` tag ruleset restricts creating, updating, and deleting release tags to the GitHub Actions integration; no user account can tag a release |
| tag shape | `scripts/create_release.py` computes the tag from `pyproject.toml` and validates it against `TAG_RE`; `pr_publish_pypi.yml` re-validates the resolved tag and cross-checks it against the built version |
| PyPI upload | trusted publishing (OIDC) only, from the `publish_to_pypi` job running in the `pypi` GitHub environment; the PyPI trusted-publisher configuration names that environment |
| filename availability | a pre-build guard queries PyPI and fails loud, with the burned filenames named, before anything is built or uploaded |

## Model authority boundary

The release model composes prose only: the release title and the notes body. It has no authority over identifiers. The tag derives from the `pyproject.toml` version, must match `TAG_RE` (`^v\d+\.\d+\.\d+$`), and any model deviation on `tag` or `version` aborts the release before a git command runs. Traceability is appended mechanically after the model returns: the merged pull requests since the previous tag, the compare link, and the changelog anchor by `scripts/create_release.py`, and the artifact SHA-256 digests by the publish workflow once the distributions exist. `tests/test_release_contract.py` pins this boundary.

## Release deliverables

Every GitHub release carries:

- the wheel and sdist that PyPI serves, attached as release assets
- `sbom.json` (CycloneDX), generated from the released wheel environment
- SHA-256 digests of every asset, appended to the release body
- GitHub build-provenance attestations for the distributions
- a mechanical traceability section: merged pull requests, compare link, changelog anchor

Assets attach only after the PyPI publish succeeds, so a release with assets is a release that shipped.

## Dependency vulnerability gate

`pr_checks_supply` runs `governance/check_dependency_vulnerabilities.py`: pip-audit over `[project.dependencies]`. A finding blocks merge unless `.github/vuln_exceptions.json` carries an entry with `id`, `reason`, and `expiry`; an expired exception fails again. Dependabot (pip and github-actions ecosystems, weekly) proposes the upgrades that keep the gate green.

## Workflow hardening invariants

- every workflow declares least-privilege top-level `permissions:`; write scopes are job-level only
- every action reference is pinned to a full commit SHA; no mutable tags
- every checkout sets `persist-credentials: false` except the release tag push
- CI tool installs are constrained by `requirements/constraints.txt`
- release tag names reach shell steps through `env:` indirection, never inline interpolation

`tests/test_packaging_surface.py::test_supply_chain_surfaces` fails any PR that regresses these invariants; repository settings state is verified by API read and recorded in the delivering slice's closeout.

## Read next

- [Making a Release](Making-Release.md)
- [Packaging](Packaging.md)
- [Semantic versioning](../Semantic-Versioning.md)
- [Developer home](README.md)
