# Repository law

## The laws

Nineteen laws. Eighteen are workflow gates on every PR; the nineteenth is branch protection on `main`. Any failure blocks merge. No bypass.

The list is long because this repository ships a package: alongside the governance gates it carries packaging verification, a built-wheel install matrix across four interpreters, a coverage floor, a suite runtime budget, and a documentation site. Every one of those is a required check, and law 17 requires each to be named here.

1. **Every PR closes exactly one OPEN issue via `Closes #N`.** A `feat` PR's issue must carry the `prd` label; any other type may close any OPEN issue — or a PRD when the author wants more eyes on it. *(pr_checks_prd)*

2. **PR title and every non-merge commit match Conventional Commits v1.0.0, and no commit message or the PR title names an AI/LLM assistant.** Allowed types: `feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert`. Every commit-message closing reference, including merge commits, must also appear explicitly as a same-repository closing reference in the PR body; cross-repository commit closures are rejected. *(pr_checks_cc)*

3. **Typing discipline never weakens.** No new `Any`, `cast(..., Any)`, `# type: ignore`, `# pyright: ignore`, or `# noqa`. Pyright error count cannot rise. `.github/typing_budget.json` cannot be raised by the PR it gates. The budget is seeded at this repository's measured state, not at zero: law 3 guarantees "no worse", not "clean". *(pr_checks_typing)*

4. **Silent-fallback patterns never grow.** No new bare `except:`, empty handler (`pass`, `...`, `return`, `return None`, `continue`, `break`), `contextlib.suppress` (or any alias chain thereof), or `errors='ignore'`. `.github/fail_loud_budget.json` cannot be raised by the PR it gates. *(pr_checks_fail_loud)*

5. **Version and changelog are cut at release time.** Ordinary PRs do not touch version surfaces. When a release is cut, manually with agent help, `[project].version` advances strictly forward by `MAJOR.MINOR.PATCH`, `CHANGELOG.md` gains a `## [X.Y.Z] - YYYY-MM-DD` section with at least one imperative content line ("Add", not "Added"), and `CITATION.cff` plus every `limen_version:` in the manifest templates and public docs carry the same version. *(manual)*

6. **The static lint plane passes.** Ruff 0.15.21 across `limen/`, `governance/`, and `tests/`; every package module within its declared line budget in `.github/module_budgets.json`; no honesty-violation is swallowed; declared runtime dependencies carry no known vulnerability (pip-audit, with time-boxed `.github/vuln_exceptions.json` entries); and no budget file is raised by the PR it gates. *(pr_checks_lint)*

   Five upstream gates are deliberately **not** in this plane and are named here so their absence is a recorded decision rather than drift: `check_module_docstrings` and `check_docstrings` (this repository ignores `D100` by choice — one function per file with filename-matching names makes a module docstring a restatement, which the stance below forbids), `check_test_fallbacks` (the pattern it flags is a correct verbose `pytest.raises`, and it mis-flags `try`/`finally` cleanup), `check_test_code_ratio` (a SLOC proxy on a suite already at 93% coverage), and `check_file_size_balance` (pending a configurable bound upstream). All five ship in `governance/` and can be run by hand. Dead-code detection (vulture) is absent for the same reason: all 17 of its findings here are parameter names in Protocol stubs and sklearn-shaped signatures, so it reports no defect this codebase has.

7. **`python -m tests.run` passes.** *(PR Checks Tests)*

8. **Package coverage holds at or above 92%.** *(PR Checks Coverage)*

9. **The suite completes inside its recorded wall-clock budget** in `tests/runtime_budget.json`, with the slowest tests published to the run summary. *(PR Checks Runtime)*

10. **The documentation site lints, assembles, and builds clean, and its dependencies carry no untracked advisory.** An advisory with no patched release anywhere in the dependency graph may be tracked in `docs-site/scripts/audit-security.mjs` by its URL, with the date and the release that retires it; a tracked advisory is a dated exception, not a silence. *(PR Checks Docs Site)*

11. **Packaging holds its contract.** The source audit passes, distributions build byte-identically twice under a fixed `SOURCE_DATE_EPOCH`, and `twine check`, `check-wheel-contents`, `check-manifest` and `pyroma` all pass. *(PR Checks Packaging)*

12. **The built wheel installs and imports clean on Python 3.10**, with `__version__` agreeing with distribution metadata and no heavy optional backend imported. *(PR Checks Package Install (3.10))*

13. **The same holds on Python 3.11.** *(PR Checks Package Install (3.11))*

14. **The same holds on Python 3.12.** *(PR Checks Package Install (3.12))*

15. **The same holds on Python 3.13.** *(PR Checks Package Install (3.13))*

16. **CodeQL reports no new Python security anti-patterns.** *(PR Checks CodeQL (python))*

17. **The written laws and the enforced gates agree exactly.** The required status checks on `main` are in bijection with the workflow-gate laws here — every required check has a law, and every gated law is required. A gate added to the ruleset without a law, or a law whose gate was dropped, fails this gate. *(pr_checks_honesty)*

18. **Live branch protection on `main` matches `.github/rulesets/main.json`.** Changing branch protection out-of-band (in the GitHub UI) blocks the next PR until the snapshot is updated in a PR of its own. *(pr_checks_ruleset)*

19. **No direct push to `main`. No force-push. No branch deletion.** Branches may merge without rebasing onto latest `main`; conflicting merges stay blocked and the push-triggered suite on `main` guards against stale-green breaks. One approving review, code-owner review on the enforcement surfaces owned in `.github/CODEOWNERS`, and every review thread resolved. *(branch protection, server-side)*

Beyond the gates, `audit_main_ruleset` re-checks the live ruleset on every push to `main` with a privileged token — including `bypass_actors`, which the PR-time ruleset gate (`pr_checks_ruleset`) cannot observe. It is a post-merge alarm, not a merge gate, so it carries no law of its own. The release and publish path (`pr_post_release`, `pr_publish_pypi`), the weekly fuzz smoke, and the OpenSSF Scorecard analysis run outside the merge gate for the same reason.

