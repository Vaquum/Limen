# Security assurance case

This page argues, requirement by requirement, why Limen meets its security requirements, and names the mechanical evidence for each argument. It is the entry point for judging Limen's security posture; [SECURITY.md](https://github.com/Vaquum/Limen/blob/main/SECURITY.md) defines how to report a suspected failure of any claim on this page.

## Prerequisites

- none for reading; verification commands require the [GitHub CLI](https://cli.github.com)

## What Limen is, for threat modeling

Limen is a Python research library and CLI that an operator runs inside their own environment. It is not a hosted service: it has no server process, no user accounts, no authentication surface, and it stores no credentials. At runtime it reads remote Bitcoin market data over HTTPS, executes experiments the operator defines, and writes artifacts to the operator's filesystem. The security-relevant supply side is therefore the repository itself: what code can reach `main`, what dependencies ship, and whether a released artifact is exactly what CI built.

## Trust boundaries

| Boundary | What crosses it | Position |
| --- | --- | --- |
| Remote market data ingress | OHLC and trade history over HTTPS from public dataset hosts | untrusted input, validated at load |
| Operator-supplied manifests and SFD modules | YAML experiment definitions and Python decoder modules | inside the operator's own trust domain: Limen executes research code its operator wrote, like any library |
| Third-party dependencies | packages declared in `pyproject.toml` | vetted at merge and weekly |
| CI and release pipeline | workflows, tokens, PyPI publication | least privilege, mechanically pinned |
| Documentation site | assembled static site | build gated, no runtime code |

## Requirement 1: only reviewed, gate-passing code reaches main

Argument: direct pushes are impossible and every merge requires review plus a fixed set of green checks, so no single actor can land unreviewed code.

Evidence: the `Protect-Main` ruleset requires pull requests with non-author approval, code-owner review on the enforcement surfaces owned in `.github/CODEOWNERS`, and all required status checks (Ruff, Tests, Coverage, Runtime, CodeQL, Pyright, Docs Site, Packaging, Package Install on four Python versions, the slice gate, and the dependency vulnerability gate) before merge; force pushes and branch deletion are blocked. The slice gate (`governance/slice_gate.py`, run by `.github/workflows/pr_checks_slice.yml`) additionally binds every PR to a filed scope contract.

## Requirement 2: released artifacts are exactly what CI built, and users can verify it

Argument: no human hands touch the publish path, the build is reproducible, and every artifact ships with a cryptographic provenance trail a consumer can check independently.

Evidence: release tags can be created only by the GitHub Actions integration under the `v*` tag ruleset; PyPI publication uses trusted publishing (OIDC) from the `pypi` environment with no long-lived token; distributions build with a fixed `SOURCE_DATE_EPOCH` and are attested with `actions/attest-build-provenance`; every GitHub release carries the wheel, sdist, CycloneDX `sbom.json`, the `provenance.intoto.jsonl` Sigstore bundle, and a SHA-256 digest block. Verification commands are documented in [Release Policy](Release-Policy.md). `tests/test_packaging_surface.py::test_supply_chain_surfaces` pins the pipeline's invariants.

## Requirement 3: known-vulnerable dependencies cannot enter or persist silently

Argument: vulnerability findings block merges unless explicitly excepted with a reason and an expiry, and update pressure is continuous, so exposure windows are bounded and always on the record.

Evidence: `.github/workflows/pr_checks_supply.yml` runs `governance/check_dependency_vulnerabilities.py` (pip-audit over the declared dependencies) as a required check; exceptions live in `.github/vuln_exceptions.json` with mandatory `id`, `reason`, and `expiry`, and expired exceptions fail again; Dependabot proposes weekly updates for the pip and github-actions ecosystems.

## Requirement 4: CI credentials follow least privilege and stay out of PR reach

Argument: workflow tokens grant the minimum scope, third-party actions cannot drift, and checked-out credentials are not persisted, so PR-supplied code has no standing capability to escalate.

Evidence: every workflow declares least-privilege top-level `permissions:` with write scopes only at job level; every action reference is pinned to a full commit SHA; every checkout sets `persist-credentials: false` except the release tag push; tool installs are constrained by `requirements/constraints.txt`; tag names reach shell through `env:` indirection. All five invariants are pinned by `tests/test_packaging_surface.py::test_supply_chain_surfaces`.

## Requirement 5: vulnerability reports have a private, credited path

Argument: reporters have a private channel with a stated scope and a credit commitment, so disclosure incentives point toward the project rather than the public tracker.

Evidence: [SECURITY.md](https://github.com/Vaquum/Limen/blob/main/SECURITY.md) routes reports through GitHub Security Advisories, names the supported version, scope, and required report content, and commits to crediting reporters in the fix's release notes and changelog entry unless they request otherwise.

## Requirement 6: static analysis runs on every change

Argument: two independent analyzers plus a strict type gate run as required checks, so common vulnerability classes and type-level defects are caught before merge, not after release.

Evidence: CodeQL (`.github/workflows/pr_checks_codeql.yml`) and Ruff with its security rules (`.github/workflows/pr_checks_ruff.yml`) are required checks; Pyright runs in strict mode over `limen/` with an errors-and-warnings-zero baseline (`.github/workflows/pr_checks_pyright.yml`); OpenSSF Scorecard re-analyzes every push to `main` (`.github/workflows/scorecard-analysis.yml`) and publishes the result.

## Residual risks

- Limen executes operator-authored research code by design; a hostile manifest or SFD module is hostile code the operator chose to run, and no sandbox is claimed. Treat third-party manifests like third-party code.
- No fuzzing or property-based testing runs today; the [Roadmap](../Roadmap.md) carries it under platform trust.
- The contributor base is a single organization; continuity and access rules are documented in [GOVERNANCE.md](https://github.com/Vaquum/Limen/blob/main/GOVERNANCE.md) and [MAINTAINERS.md](https://github.com/Vaquum/Limen/blob/main/MAINTAINERS.md).

## Verifying the claims

```bash
gh api repos/Vaquum/Limen/rulesets --jq '.[].name'
gh attestation verify vaquum_limen-*.whl --repo Vaquum/Limen
python -m pytest -q tests/test_packaging_surface.py
```

The first command lists the active rulesets; the second verifies a downloaded release artifact against its published provenance; the third runs the supply-chain invariant pins locally.

## Read next

- [Release Policy](Release-Policy.md)
- [Making a Release](Making-Release.md)
- [Roadmap](../Roadmap.md)
- [Developer home](README.md)
