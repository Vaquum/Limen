# Semantic Versioning

This page defines how Limen should choose version bumps in practice.

The repo workflow is simple:

- docs-only and other non-code changes usually do not require a version bump
- shipped code changes should update `pyproject.toml`
- releases are created from the version in `pyproject.toml` after merge to `main`

## Version Rules

### MAJOR: `X.y.z`

Use a major bump when a release introduces a breaking change for users of Limen.

Typical examples:

- changing a public API in a way that requires user code changes
- removing or renaming public imports without a compatibility path
- changing core workflow behavior in a way that invalidates existing usage assumptions
- changing persisted artifact formats or contracts in a breaking way

### MINOR: `x.Y.z`

Use a minor bump when the release adds new backwards-compatible capability.

Typical examples:

- new public indicators, features, transforms, or scalers
- new manifest capabilities that do not break existing manifests
- new experiment, cohort, trainer, or data-access functionality that extends the surface cleanly

### PATCH: `x.y.Z`

Use a patch bump for backwards-compatible fixes and improvements.

Typical examples:

- bug fixes
- correctness improvements
- performance improvements without breaking behavior
- implementation cleanups that change shipped code but not the public contract

Patch is the default version bump when code changed and neither major nor minor applies.

## Practical Decision Table

| Change type | Version bump |
|---|---|
| docs-only or non-code-only change | no version bump in normal practice |
| backwards-compatible bug fix or improvement | patch |
| backwards-compatible new capability | minor |
| breaking change | major |

## Release Hygiene Rules

- Update `pyproject.toml` in the same PR as the shipped change.
- Update `CHANGELOG.md` in the same PR when the change is not docs-only or another non-code change.
- Keep `CHANGELOG.md` in oldest-first order.
- Release tags must use lowercase `v`, such as `v1.48.0`.

## Default Questions To Ask

Before bumping the version, ask:

1. Will an existing Limen user need to change code, config, workflow, or artifact assumptions?
2. Is this adding a new public capability or only improving an existing one?
3. Is this code shipping at all, or is it only docs and repo hygiene?

If the answer to the first question is yes, lean major. If not, and the answer to the second is yes, lean minor. Otherwise, patch is usually correct for shipped code.

## Read Next

- [Making a Release](Developer/Making-Release.md)
- [Developer Home](Developer/README.md)
