# Making Pull Requests

We have PR reviews for every PR, but the responsibility is with the author of the PR to ensure that the PR promoted for review is in a mergeable state: CI is green, tests pass locally, and there are no merge conflicts. Moreover, not everything can be caught by tests, so the author is further responsible for having carefully gone through the proposed changed with regards to their implications to the codebase.

**NOTE:** The author must always **first review the PR themselves**, before requesting review from others, that means carefully having reviewed the full diff in GitHub.

## Basic Sanity

Every PR must minimally involve successfully going through the following steps: 

- [ ] I have reviewed full diff in “Files changed”
- [ ] I left no unnecessary files in the changes
- [ ] I ran `python tests/run.py` locally without errors (where applicable)
- [ ] I updated `/docs` (if behavior/API/config/user/etc changed)
- [ ] I added and/or updated docstrings as per [Writing Docstrings](https://github.com/Vaquum/Limen/blob/main/docs/Developer/Writing-Docstrings.md) (for any changed public functions/classes)
- [ ] I updated CHANGELOG.md (unless only docs or other non-code aspect was changed)
- [ ] I updated pyproject.toml (unless only docs or other non-code aspect was changed)
- [ ] I added and/or updated tests (if behavior changed or new code paths added)
- [ ] I validated changes manually
- [ ] I validated changes with LLM
- [ ] I removed any extraneous examples/comments
- [ ] I linked issue to auto-close on merge (e.g., “Fixes #123”) when applicable

Once all these boxes are checked, the PR is ready for adding reviewers.
