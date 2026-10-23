# CLAUDE.md

The measure is if the merge button is green or not. Never stand down before the merge button is green.

The repository law, operating discipline, and code stance — the single canonical constitution for every contributor, human or agent.

## Motivation

**This applies to both the author and the reviewer:**

We don't want saga ornamentation. We want commits that move the needle. The needle is the thing that humans actually benefit from in the software. Always ask "what are the key usage paths here, and how is this proposed change moving those?". Never be satisfied with work that seems to check the boxes, but doesn't really move the capability where the actual benefit is.

We want work that is the minimal footprint representation of the ask. Minimal, clinical interventions all the way. Get in, get out, do the thing that actually has to be done, never expand it beyond that. Keep asking "is this truly the cleanest and simplest and least intrusive way to get this done" and do it that way and only that way. Don't let others come expand it, let them come make it cleaner.

When addressing review comments on your own issue or PR, handle every one — blocking or not — with either a commit/change (and name that commit in the thread) or a reply explaining the resolution, or why it is not addressed; then resolve the thread. Both are ok, do not sink into endless rabbit holes. 

Merge stays blocked until every thread is resolved (`required_review_thread_resolution`, server-side). 

## Review work

**Reviewing a pull request?** 

When reviewing an issue, post comments directly in the thread. Do not grind endlessly turn after turn.

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

## Workflow

Branch off `main`. Commit the first working increment and push to a remote branch of the same name, then **open the PR right away — before you run the gates locally, and never as a draft.** Opening it first starts CI immediately; run the local gate suite afterwards, keep working while GitHub runs the gates too, and don't wait for CI in the foreground. Never sit on local verification without an open PR; open a real PR or none.

The human documentation editor deliberately separates Commit from Open PR: writers may accumulate prose commits on its origin branch before opening a non-draft PR. Those branch-only commits have no PR-check result. This exception does not change agent implementation delivery or any merge gate.

Each push to an open PR re-runs every gate. Prefer new commits to amends — amends don't give you anything and they muddle the PR history. Keep one logical change per commit; don't batch unrelated changes together. Before you request review, read your own full diff in GitHub — catch what you'd flag in someone else's PR.

Merge unlocks when every required gate is green. Branches need not be up-to-date with `main`; rebase only to resolve a conflict or to pick up a main-side fix the PR needs.

Reviews are triggered automatically by changes, and once the PR comes ready it will be auto-merged through triggers. The responsibility of the author is to never drop the ball before the merge button is green: watch every gate, answer every review comment, and keep pushing until the button is green.

When a gate fails, the gate's own output names the reason. Read the output, fix the code or the linked issue, push again. If the failure is the gate being wrong rather than the PR being wrong, fix the gate in its own PR — the ruleset drift gate (`pr_checks_ruleset`) will force the matching ruleset-snapshot update so no gate relaxation side-enters.

## Beyond the laws

The gates check shape, scope, format, ratchets, and named test suites. They do not check whether the PR's capability actually works. 

**Radical simplicity.** The simplest code that meets the requirement wins. Complexity earns its place by naming the specific concern it addresses — not "robustness" or "future-proofing" in general.

**No defensive fog; fail loud.** Agents are primed to produce defensible-looking code: `try/except` that swallows, fallbacks for cases that don't happen, docstrings that restate the signature, comments that narrate the line, parameters that might be useful someday. None of it belongs. When something is wrong, find the root cause and fail loudly and early — never paper over missing state with a workaround, a fallback, or a swallowed error. The fail-loud gate (`pr_checks_fail_loud`) catches the AST-detectable forms.

**No sitting in the dark.** Never suppress callable output, script output, or sub-agent logs to save context — stay fully aware of what the running process is doing.

**No long-running commands.** Do not repeatedly run long-running commands. If something needs to be run repeatedly, and immediately profile it.

**Minimal scope.** Touch only the files the task demands. Drive-by cleanups go in a separate PR.

**No synthetic data.** Ever. Inventing data is not a shortcut — it corrupts everything downstream. If the real data isn't there, stop and ask.

**Validate against the stated expectation.** The question is never "did it run" — it's "did it return what the issue promised."

**Deliver meaning not mechanics.** It's better to deliver the right meaning poorly, than deliver meaningless scaffolding and mechanics in an impressive way. 

**The smallest possible honest way always.** Issue spec, code, communication, everything, let it be the smallest possible unit size that honestly delivers what is required.

## Conventions

Concrete house style the gates don't check — not judgment calls, just the defaults to follow:

- **Dependencies.** Prefer the standard library or an existing project dependency. A new external dependency must be required by the task, not a convenience.
- **Logging.** Use `logging.getLogger(__name__)` in library code. (`print` in the package is already gate-blocked.)
- **Public surface.** Expose the public API explicitly with `__all__`; prefix internal names with `_`.
- **Resources.** Use context managers for anything that must close; avoid mutable default arguments; prefer `pathlib` over `os.path`.
- **LLM output.** A model may draft, but raw model output is never dropped in as-is — the contributor simplifies it, understands it, and owns it.
- **Docs.** Author each thing once: one page is canonical, the rest link to it. Show real, runnable examples and current behavior — never imaginary examples or aspirational framing.
- **Release notes.** Technically correct, concise in summary, specific in detail, and tied to the pushed tag.

## When in doubt, stop and reduce

For the most part, PRs become too big, and very rarely too small, so optimize towards the hard, optimize towards keeping them small and simple.

## Task-concluding messages

If a commit was made, show the hash.
If a review was left, share the link. 
If a PR was made, share the link.

The measure is if the merge button is green or not. Never stand down before the merge button is green.
