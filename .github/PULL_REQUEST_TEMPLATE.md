**NOTE:** The author must always **first review the PR themselves**, before requesting review from others, that means carefully having reviewed the full diff in GitHub.

## What does this PR change?

_(Fill in at least a short description of the intent of the change.)_

## Interface Contract

<!-- Required on every PR. Follow CLAUDE.md#research-interface-contract.
Match the slice declaration when present; this PR may supply its own missing
slice declaration. A native parent's declaration must live in the parent issue.
Use public, prerequisite or maintenance. Public delivery needs a package test per
claimed interface. Non-public work with no research behavior change may use
interfaces: [] and proof: {}. Replace the example scope and proof below.
If the last child leaves its parent open, write `Parent remains open: <remaining
outcome or next slice>` on one line after the fence.
Represent every commit-message closing reference explicitly in this PR body.
-->

```yaml
delivery: public
interfaces: [python]
scope: "REPLACE: complete capability and justified exclusions."
proof:
  python: [test:tests/test_example.py::test_python_capability]
```

## Checklist

- [ ] I have reviewed full diff in “Files changed”
- [ ] I left no unnecessary files in the changes
- [ ] I ran `python -m tests.run` locally without errors (where applicable)
- [ ] I updated any relevant documentation (if behavior/API/config/user/etc changed)
- [ ] I added and/or updated docstrings for any changed public functions/classes (the docstring-conventions gate enforces the mechanizable rules)
- [ ] I added a `CHANGELOG.md` entry under a new `## [X.Y.Z] - YYYY-MM-DD` header (every PR — the version gate requires it)
- [ ] I bumped `[project].version` in `pyproject.toml` (every PR — the version gate requires it)
- [ ] I carried the version to `CITATION.cff` and every `limen_version:` in the manifest templates and public docs
- [ ] I added and/or updated tests (if behavior changed or new code paths added)
- [ ] I validated that the change returns what it promised — not merely that it ran
- [ ] I removed any extraneous examples/comments
- [ ] I declared complete interface outcomes and verified meaningful passing proof without skipped cases
- [ ] I requested renewed semantic review after any material PR/issue contract edit
- [ ] I linked exactly one slice to close on merge; any parent closing reference has its complete outcome evidence
