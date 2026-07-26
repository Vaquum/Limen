**NOTE:** The author must always **first review the PR themselves**, before requesting review from others, that means carefully having reviewed the full diff in GitHub.

## What does this PR change?

_(Fill in at least a short description of the intent of the change.)_

## Checklist

- [ ] I have reviewed full diff in “Files changed”
- [ ] I left no unnecessary files in the changes
- [ ] I ran `python -m tests.run` locally without errors (where applicable)
- [ ] I updated any relevant documentation (if behavior/API/config/user/etc changed)
- [ ] I added and/or updated docstrings for any changed public functions/classes (the docstring-conventions gate enforces the mechanizable rules)
- [ ] I added a `CHANGELOG.md` entry under a new `# v<X.Y.Z>` header (every PR — the version gate requires it)
- [ ] I bumped `[project].version` in `pyproject.toml` (every PR — the version gate requires it)
- [ ] I carried the version to `CITATION.cff` and every `limen_version:` in the manifest templates and public docs
- [ ] I added and/or updated tests (if behavior changed or new code paths added)
- [ ] I validated that the change returns what it promised — not merely that it ran
- [ ] I removed any extraneous examples/comments
- [ ] I linked issue to auto-close on merge (e.g., “Fixes #123”) when applicable
