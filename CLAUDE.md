# CLAUDE.md

The repository law, operating discipline, and code stance — the single canonical constitution for every contributor, human or agent.

Adopted from [`Vaquum/new-repository-template`](https://github.com/Vaquum/new-repository-template), which is upstream for the governance plane. Divergences from upstream are deliberate and are named where they occur; a gate that is wrong here is fixed upstream in its own PR rather than forked locally.

## Motivation

We don't want saga ornamentation. We want commits that move the needle. The needle is the thing that humans actually benefit from in the software. Always ask "what are the key usage paths here, and how is this proposed change moving those?". Never be satisfied with work that seems to check the boxes, but doesn't really move the capability where the actual benefit is.

A gate earns its place by catching a defect class this repository actually has. Adopting a gate because upstream runs it is not a reason — upstream is a 2,000-line enforcement repository with no domain data, and constraints that are free there can be expensive here. When a gate demands work that delivers nothing, the gate is what changes.

## The laws

Nineteen laws. Eighteen are workflow gates on every PR; the nineteenth is branch protection on `main`. Any failure blocks merge. No bypass.

The list is long because this repository ships a package: alongside the governance gates it carries packaging verification, a built-wheel install matrix across four interpreters, a coverage floor, a suite runtime budget, and a documentation site. Every one of those is a required check, and law 17 requires each to be named here.

1. **Every PR closes exactly one OPEN slice-labelled issue; closing its native parent PRD additionally requires no other open slice children and complete parent evidence.** A completed last child may leave its parent open with `Parent remains open: <remaining outcome or next slice>` in the PR body. Every PR carries the [Interface Contract](#research-interface-contract), agrees with its slice when declared, and respects its authoritative parent contract. Public delivery must completely support each claimed interface. PR title byte-equals the slice issue's title. Diff stays within the issue's `## Surfaces` globs and touches no path in `## Out of Scope`. Issue body preserves every existing `> **Significance.**` blockquote from the slice template verbatim. Every `Done Means` checkbox is checked or carries `OVERRULED: <reason>` before merge; this cannot waive the independent interface contract. The `slice_closeout_guard` workflow writes closeout evidence when a merged PR closes the slice and reopens an evidence-less close. *(pr_checks_slice)*

2. **PR title, every non-merge commit, and the linked issue title match Conventional Commits v1.0.0, and no commit message or the PR title names an AI/LLM assistant.** Allowed types: `feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert`. Every commit-message closing reference, including merge commits, must also appear explicitly as a same-repository closing reference in the PR body; cross-repository commit closures are rejected. *(pr_checks_cc)*

3. **Typing discipline never weakens.** No new `Any`, `cast(..., Any)`, `# type: ignore`, `# pyright: ignore`, or `# noqa`. Pyright error count cannot rise. `.github/typing_budget.json` cannot be raised by the PR it gates. The budget is seeded at this repository's measured state, not at zero: law 3 guarantees "no worse", not "clean". *(pr_checks_typing)*

4. **Silent-fallback patterns never grow.** No new bare `except:`, empty handler (`pass`, `...`, `return`, `return None`, `continue`, `break`), `contextlib.suppress` (or any alias chain thereof), or `errors='ignore'`. `.github/fail_loud_budget.json` cannot be raised by the PR it gates. *(pr_checks_fail_loud)*

5. **Every PR bumps the version and leaves a CHANGELOG trail.** `[project].version` advances strictly forward by `MAJOR.MINOR.PATCH`; the one exception is a PR whose body carries `[version-rewind: <reason>]` on its own line, which may set a lower version to withdraw releases that should never have been cut. `CHANGELOG.md` carries a `## [X.Y.Z] - YYYY-MM-DD` section for the new version with at least one content line, written imperatively ("Add", not "Added") and free of leftover placeholders. Bump level meets the Conventional Commits type minimum: `type!` → major, `feat` → minor, else patch. A major release is reserved for a change to the public API and requires `[major-release: <reason>]` on its own line of the PR body naming that change; a behaviour change inside one component, however visible in its output, is a patch or a minor and its slice title carries no `!`. `CITATION.cff` and every `limen_version:` in the manifest templates and public docs carry the same version. *(pr_checks_version)*

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

19. **No direct push to `main`. No force-push. No branch deletion.** Branch must be up-to-date with `main` before merge. One approving review, code-owner review on the enforcement surfaces owned in `.github/CODEOWNERS`, and every review thread resolved. *(branch protection, server-side)*

Beyond the gates, `audit_main_ruleset` re-checks the live ruleset on every push to `main` with a privileged token — including `bypass_actors`, which the PR-time ruleset gate (`pr_checks_ruleset`) cannot observe. It is a post-merge alarm, not a merge gate, so it carries no law of its own. The release and publish path (`pr_post_release`, `pr_publish_pypi`), the weekly fuzz smoke, and the OpenSSF Scorecard analysis run outside the merge gate for the same reason.

## Research interface contract

Limen has two research interfaces. **Python Research** gives callers their own data, models, features, backtesting and selection in ordinary Python: custom SFD `params`/`prep`/`model` functions through `UniversalExperimentLoop`, caller-controlled search through MSQ, and `Sensor.from_python` for native inference handoff. It requires no new research language or built-in-model whitelist. **Compiled Research** adopts the `Manifest` (`MLManifest`/`RuleBasedManifest`) contract, directly or through `CompiledSFD` and CLI validation/compilation. A Manifest invoked from Python remains Compiled Research. UEL/MSQ, indicators, features and scalers are shared machinery, not exclusive interface markers.

A public capability must work across every semantically applicable combination admitted by each interface that exposes it. Complete Python-only or Compiled-only delivery is valid; an unfinished second interface must remain unavailable. Restrictions must follow the capability's methodology or necessary evidence, not an implementation whitelist. Shared changes preserve existing guarantees on both paths. Completeness applies to the introduced or extended user capability, not every individual helper; unrelated maintenance does not have to repair every legacy limitation.

This is a deliberate Limen-specific extension of the upstream governance plane, including law 1's independent child completion. It uses existing required contexts and introduces no additional research interface or required status context.

### Declaration

Every implementation PR has exactly one `## Interface Contract` section; issue forms may render it as `### Interface Contract`. It contains exactly one YAML fence with exactly four keys:

```yaml
delivery: maintenance
interfaces: []
scope: "Change contributor governance without changing research execution."
proof: {}
```

- `delivery` is `public`, `prerequisite` or `maintenance`. Public delivery requires at least one interface and full support for each. A prerequisite exposes no unfinished public capability. Maintenance repairs or preserves existing behavior without enlarging its public contract.
- `interfaces` is a duplicate-free subset of `python`, `compiled`. Prerequisite and maintenance work may use `[]` when no research capability changes; ordinary slice Proof and CI obligations still apply.
- `scope` is a nonblank string naming the capability and justified exclusions, or the non-public purpose. Quote prose values to avoid YAML ambiguity. It cannot redefine an implementation gap as methodological inapplicability.
- `proof` maps exactly the listed interfaces to nonempty reference lists; `interfaces: []` uses `proof: {}`. Public delivery needs at least one named test for every interface. Other roles may instead cite a relevant required check.

Duplicate or unknown keys, aliases/merge keys, non-string enum values, placeholders and duplicate sections are rejected. CRLF and interface/reference ordering do not change a declaration's meaning. New issue forms require the field; CLI-authored issues follow the same planning discipline. The new slice field has no Significance blockquote, so it does not invalidate existing issues' quotations.

### Evidence and review

Test references use `test:tests/<path>/test_<module>.py::test_<name>` or `test:tests/<path>/test_<module>.py::Test<Class>::test_<name>`; the intermediate path may be absent. They name directly declared, normally discovered package tests. Reference a whole parametrized function, not selected cases. The gate rejects unsafe paths, symlinks, missing files/symbols and statically declared skip/xfail evidence. Only prerequisite/maintenance work may use `check:<required context>`, resolved against the trusted default branch's ruleset, not the candidate snapshot.

Gate reads use the judged PR head as data; missing head identity or unavailable evidence cannot pass. Privileged issue/sweep jobs never import, collect or execute candidate tests. Reference existence and check membership do not prove execution or relevance. Required CI and semantic review must establish actual passing, unskipped execution, meaningful assertions, full claimed scope and honest exclusions. A formatting or admission check cannot prove a public research capability.

### Adoption and parent completion

The PR declaration must equal its slice's parsed declaration when present. If the slice lacks the section, its own PR supplies it; every PR still declares a contract, so no adoption-date exemption is needed. A native parent's contract must persist in that parent issue before child admission. Add it with a targeted issue edit, preserve the parent's agreed outcomes, and never replace it with a PR-body backfill.

A public child requires a public parent and may claim a subset of its interfaces. A prerequisite's interfaces must also be a subset of the parent's targets. Maintenance may additionally name existing interfaces preserved by a shared repair without enlarging the parent's delivered product scope. Validate compatibility on every child; defer only parent proof-reference lookup until closure.

A completed child may close independently. A last-child PR leaving the parent open names the remaining outcome or next slice in `Parent remains open: ...`. When ready, a new completion slice may close itself and the parent with all parent evidence. Child counts alone never establish product completion. GitHub's optional-colon closing syntax is recognized; commit-based closures must be represented in the PR body so they receive the same checks. Add a missing reference to the PR body rather than rewriting commit history.

The gate covers PR-path parent closure. Manual PRD closure has no automatic evidence guard; contributors must not use it to evade this law. Material PR/issue contract edits require renewed semantic review of both the current head and changed declaration; GitHub does not automatically dismiss approval on body edits. These are explicit mechanical limits, not completeness guarantees. After correcting a stale API-posted sweep failure, dispatch `pr_checks_slice_sweep` with `workflow_dispatch` to refresh the verdict instead of waiting six hours.

## Workflow

Branch off `main`. Commit the first working increment and push to a remote branch of the same name, then **open the PR right away — before you run the gates locally, and never as a draft.** Opening it first starts CI immediately; run the local gate suite afterwards, keep working while GitHub runs the gates too, and don't wait for CI in the foreground. Never sit on local verification without an open PR; open a real PR or none.

**`zero-bang` is the approving authority — the operator.** ("The operator" throughout this document is that human reviewer: the one who judges the work at review time and whose approval unlocks merge.) Request their review the moment the PR is open. Once every requested change is addressed, re-request `zero-bang`'s review.

Each push re-runs every gate. Prefer new commits to amends — amends don't give you anything and they muddle the PR history. Keep one logical change per commit; don't batch unrelated changes together. Before you request review, read your own full diff in GitHub — catch what you'd flag in someone else's PR.

Merge unlocks when every required gate is green **and** the branch is up-to-date with `main`. Up-to-date is enforced server-side; rebase when main advances.

When a gate fails, the gate's own output names the reason. Read the output, fix the code or the slice issue, push again. If the failure is the gate being wrong rather than the PR being wrong, fix the gate in its own PR — the ruleset drift gate (`pr_checks_ruleset`) will force the matching ruleset-snapshot update so no gate relaxation side-enters.

## Review work

**Reviewing a pull request?** The canonical brief is [`.github/copilot-instructions.md`](.github/copilot-instructions.md) — how to read a diff beyond its own lines, what to hunt, the verdict ladder, and how to post. Work entirely from it; it is also what GitHub's built-in Copilot review reads, so every reviewer (Copilot, agent, or human) holds one shared standard.

The same posture governs the rest of review work. When reviewing an issue, post comments directly in the thread. When addressing comments on your own issue or PR, handle every one — blocking or not — with either a commit (and name that commit in the thread) or a reply explaining the resolution, or why it is not addressed; then resolve the thread. Merge stays blocked until every thread is resolved (`required_review_thread_resolution`, server-side). The opinion is the deliverable; never confirm with the operator before posting — it only adds a round-trip.

## Beyond the laws

The gates check shape, scope, format, ratchets, and named test suites. They do not check whether the slice's capability actually works. The operator judges that at review time, against the following stance:

**Radical simplicity.** The simplest code that meets the requirement wins. Complexity earns its place by naming the specific concern it addresses — not "robustness" or "future-proofing" in general.

**No defensive fog; fail loud.** Agents are primed to produce defensible-looking code: `try/except` that swallows, fallbacks for cases that don't happen, docstrings that restate the signature, comments that narrate the line, parameters that might be useful someday. None of it belongs. When something is wrong, find the root cause and fail loudly and early — never paper over missing state with a workaround, a fallback, or a swallowed error. The fail-loud gate (`pr_checks_fail_loud`) catches the AST-detectable forms; the rest is operator-caught at review.

**No sitting in the dark.** Never suppress callable output, script output, or sub-agent logs to save context — stay fully aware of what the running process is doing.

**No long-running commands.** Do not repeatedly run long-running commands. If something needs to be run repeatedly, immediately profile it, and report back to operator the profile results. The operator is expert in performance hacking, you are not. 

**Minimal scope.** Touch only the files the task demands. Drive-by cleanups go in a separate slice.

**No synthetic data.** Ever. Inventing data is not a shortcut — it corrupts everything downstream. If the real data isn't there, stop and ask.

**Validate against the stated expectation.** The question is never "did it run" — it's "did it return what the slice promised."

**Deliver meaning not mechanics.** It's better to deliver the right meaning poorly, than deliver meaningless scaffolding and mechanics in an impressive way. 

**The smallest possible honest way always.** Slice spec, code, communication, everything, let it be the smallest possible unit size that honestly delivers what is required.

## Conventions

Concrete house style the gates don't check — not judgment calls, just the defaults to follow:

- **Dependencies.** Prefer the standard library or an existing project dependency. A new external dependency must be required by the task, not a convenience.
- **Logging.** Use `logging.getLogger(__name__)` in library code. (`print` in the package is already gate-blocked.)
- **Public surface.** Expose the public API explicitly with `__all__`; prefix internal names with `_`.
- **Resources.** Use context managers for anything that must close; avoid mutable default arguments; prefer `pathlib` over `os.path`.
- **LLM output.** A model may draft, but raw model output is never dropped in as-is — the contributor simplifies it, understands it, and owns it.
- **Docs.** Author each thing once: one page is canonical, the rest link to it. Show real, runnable examples and current behavior — never imaginary examples or aspirational framing.
- **Release notes.** Technically correct, concise in summary, specific in detail, and tied to the pushed tag.

## When in doubt, stop

This is collaboration. If the requirement is unclear, if the scope is ambiguous, if a gate's meaning is unobvious, if the fix would require touching something that wasn't asked for — stop and ask the operator. Proceeding through doubt is where harm accumulates.

## Task-concluding messages

If a commit was made, show the hash.
If a review was left, share the link. 
If a PR was made, share the link.
