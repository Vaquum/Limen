# Security assurance case

This page argues, requirement by requirement, why Limen meets its security requirements, and names the mechanical evidence for each argument. It is the entry point for judging Limen's security posture; [SECURITY.md](https://github.com/Vaquum/Limen/blob/main/SECURITY.md) defines how to report a suspected failure of any claim on this page.

The [Python project](../examples/python-research/README.md) executes caller-owned functions and fitted state. The [Compiled project](../examples/compiled-research/README.md) is authored in YAML and run through the CLI; its declared components execute in the operator's environment. Their recorded-input and artifact checks address integrity, not sandboxing.

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
| Documentation site | assembled static site | static reading plus local browser JavaScript and a Worker; build/browser/security proof required |

## Requirement 1: only reviewed, gate-passing code reaches main

Argument: the declared branch-protection policy requires review and current checks before merge. The live ruleset audit supplies the settings evidence; this page is not an independent audit of every administrator capability.

Evidence: the `Protect-Main` ruleset requires pull requests with non-author approval, code-owner review on the enforcement surfaces owned in `.github/CODEOWNERS`, and all required status checks (Ruff, Tests, Coverage, Runtime, CodeQL, Pyright, Docs Site, Packaging, Package Install on four Python versions, the PRD gate, and the dependency vulnerability gate) before merge; force pushes and branch deletion are blocked. The PRD gate (`governance/prd_gate.py`, run by `.github/workflows/pr_checks_prd.yml`) additionally binds every `feat` PR to a filed PRD and every other PR to a filed issue.

## Requirement 2: released artifacts are exactly what CI built, and users can verify it

Argument: the workflow builds distributions from the resolved release tag with a fixed `SOURCE_DATE_EPOCH`, attests their bytes, and publishes those built distributions. Consumers can verify the source and artifact identities in the provenance independently.

Evidence: the live `Protect-Release-Tags` rule blocks updates and deletion of `v*` tags without bypass; creation follows repository write permissions. Manual-release source admission runs from trusted `main` code before checkout of the triggering source. PyPI publication uses trusted publishing (OIDC) from the `pypi` environment with no long-lived token; distributions are attested with `actions/attest-build-provenance`; successful publication attaches the wheel, sdist, CycloneDX `sbom.json`, the `provenance.intoto.jsonl` Sigstore bundle, and a SHA-256 digest block to the GitHub release. Verification commands and partial-publication recovery are documented in [Release Policy](Release-Policy.md). `tests/test_packaging_surface.py::test_supply_chain_surfaces` pins the declared pipeline invariants; live repository settings require their separate API evidence.

## Requirement 3: known-vulnerable dependencies cannot enter or persist silently

Argument: merge checks reject known dependency findings unless they have a recorded, time-limited exception. Scheduled updates provide continuing review pressure; these controls do not bound the time before an unknown vulnerability is discovered.

Evidence: `.github/workflows/pr_checks_lint.yml` runs `governance/check_dependency_vulnerabilities.py` (pip-audit over the declared dependencies) as a required check; exceptions live in `.github/vuln_exceptions.json` with mandatory `id`, `reason`, and `expiry`, and expired exceptions fail again; Dependabot proposes weekly updates for the pip and github-actions ecosystems.

## Requirement 4: CI job permissions are explicit

Argument: each job receives declared permissions, actions are pinned, and checkouts discard credentials. Privileged manual-release admission uses trusted `main` code before loading the triggering source; these controls do not establish that arbitrary PR code is harmless.

Evidence: every workflow declares least-privilege top-level `permissions:` with write scopes only at job level; every action reference is pinned to a full commit SHA; every checkout sets `persist-credentials: false` except the release tag push; CI tool installs use hash-checked locks under `requirements/ci/`, within the supported envelope in `requirements/constraints.txt`; tag names reach shell through `env:` indirection. All five invariants are pinned by `tests/test_packaging_surface.py::test_supply_chain_surfaces`.

## Requirement 5: vulnerability reports have a private, credited path

Argument: reporters have a private channel with a stated scope and a credit commitment, so disclosure incentives point toward the project rather than the public tracker.

Evidence: [SECURITY.md](https://github.com/Vaquum/Limen/blob/main/SECURITY.md) routes reports through GitHub Security Advisories, names the supported version, scope, and required report content, and commits to crediting reporters in the fix's release notes and changelog entry unless they request otherwise.

## Requirement 6: static analysis runs on every change

Argument: required CodeQL, lint and typing checks catch their supported defect classes before merge. They complement review and execution tests; they do not establish the absence of vulnerabilities.

Evidence: CodeQL (`.github/workflows/pr_checks_codeql.yml`) and Ruff in the static lint plane (`.github/workflows/pr_checks_lint.yml`) are required checks; Pyright runs through `.github/workflows/pr_checks_typing.yml` under the non-increasing budget in `.github/typing_budget.json`; the recorded Pyright error count is zero while tracked Any and suppression counts remain nonzero; OpenSSF Scorecard re-analyzes every push to `main` (`.github/workflows/scorecard-analysis.yml`) and publishes the result.

## Residual risks

- Limen executes operator-authored research code by design; a hostile manifest or SFD module is hostile code the operator chose to run, and no sandbox is claimed. Treat third-party manifests like third-party code.
- The scheduled [fuzz smoke workflow](https://github.com/Vaquum/Limen/blob/main/.github/workflows/fuzz_smoke.yml) exercises its declared test surface. It is not exhaustive fuzzing or proof of every public interface combination.
- The contributor base is a single organization; continuity and access rules are documented in [GOVERNANCE.md](https://github.com/Vaquum/Limen/blob/main/GOVERNANCE.md) and [MAINTAINERS.md](https://github.com/Vaquum/Limen/blob/main/MAINTAINERS.md).

## Verifying the claims

| Step | Command |
|---|---|
| 1 | `gh api repos/Vaquum/Limen/rulesets --jq '.[].name'` |
| 2 | `gh attestation verify vaquum_limen-*.whl --repo Vaquum/Limen` |
| 3 | `python -m pytest -q tests/test_packaging_surface.py` |

The first command lists the active rulesets; the second verifies a downloaded release artifact against its published provenance; the third runs the supply-chain invariant pins locally.

## Read next

- [Release Policy](Release-Policy.md)
- [Making a Release](Making-Release.md)
- [Roadmap](../Roadmap.md)
- [Developer home](README.md)
