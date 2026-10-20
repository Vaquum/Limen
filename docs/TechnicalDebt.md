# Technical Debt

## Prerequisites

- the current implementation, tests, and issue history for the affected surface
- evidence that the limitation remains present and materially relevant

Record intentionally accepted limitations in shipped code, with their origin, severity and migration path. The [Python project](examples/python-research/README.md) and [Compiled project](examples/compiled-research/README.md) expose different persistence boundaries: retaining caller-owned fitted objects and reconstructing a recorded Manifest are different contracts. A documented boundary alone is not an accepted defect; use this register when a specific limitation is intentionally carried.

## Register Rules

Use this page only for debt that is accepted and intentionally carried. Do not use it as a general task backlog.

Each entry should include:

- stable debt id
- affected module or public surface
- origin or evidence source
- current severity
- realistic blast radius
- trigger condition for fixing
- migration or removal path
- docs that must change when the debt is fixed

Severity should describe current Limen risk, not hypothetical downstream risk alone. If downstream live-trading assumptions raise severity, state that condition explicitly.

## Closure Rules

When an item is fixed:

1. update the code and tests in the fixing PR
2. update the canonical user/developer docs that described the old behavior
3. either remove the item from this page or move it to a short resolved note with the PR link
4. do not leave stale mitigations that imply the old risk still exists

---

## Current Register

No accepted technical-debt items are currently recorded.

Resolved or removed debt must stay in git history or linked PR discussion, not as stale active-risk text on this page.
