# Technical Debt

Known technical debt in shipped Limen code. Each item includes origin PR, severity, and migration path.

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

## TD-001: `Trainer._load_sfd_module` `import_module` fallback trusts any name resolvable on `sys.path`

**Origin**: PR #500 (Copilot pre-merge review)
**Severity**: Low (paper-trade scope; live-trading scope: Medium)
**Module**: `limen/experiment/trainer/trainer.py:_load_sfd_module`

The local-file branch validates that `sfd_module_name` is a dotted sequence of valid Python identifiers, so path-traversal (`..`, `/`, `\`) is blocked from escaping `experiment_dir`. The `importlib.import_module` fallback path passes the validated-but-not-allowlisted name straight to Python's finder system. Any module installed in site-packages is therefore loadable as the SFD by an attacker who can write `metadata.json` (e.g. via supply-chain compromise of the `experiment_runner` bundle pipeline). The `if not hasattr(sfd, 'manifest') or not hasattr(sfd, 'params')` check happens AFTER the import, so the named module's top-level code runs even when its surface ultimately fails the manifest-driven contract.

Realistic blast radius:
- Paper-trade: testnet API key exfiltration, testnet capital loss to hostile fills (no real money).
- Live-trade: live Binance API key exfiltration, full configured-strategy capital loss to hostile fills (the API key cannot withdraw by default, but it can place orders).

**When to fix**: Before any deploy that flips `TRADE_MODE=live` on a Limen-Trainer-fed strategy AND when the upstream bundle pipeline (`trainer_prep.py` / `experiment_runner` artifacts) is not under the same trust boundary as the deploy operator.

**Migration**: Replace the bare `import_module(sfd_module_name)` fallback with one of:
1. **Allowlist**: maintain a frozenset of accepted package prefixes (e.g. `frozenset({'limen.sfd.foundational_sfd'})`); reject any name not matching. Cheap, exact, but every new built-in SFD must be added.
2. **Annotation contract**: require the SFD module to set a sentinel module-level attribute (`__limen_sfd__ = True`) and verify post-import. Keeps the open extension surface but still allows arbitrary code to run before the verification trips.
3. **Explicit handler registration**: drop the `import_module` fallback entirely; require all SFDs to live inside `experiment_dir`. Hardest break — invalidates the existing UEL-built experiment_dirs that reference packaged SFDs by fully-qualified name. Would require a metadata-format migration.

Option 1 is the cheapest correct move and matches how the rest of Limen treats trusted-module surfaces. Document the allowlist in `docs/Trainer.md` and gate live-trade deploy on its presence.
