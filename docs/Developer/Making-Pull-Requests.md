# Making Pull Requests

We have PR reviews for every PR, but the responsibility is with the author of the PR to ensure that the PR promoted for review is in a mergeable state: CI is green, tests pass locally, and there are no merge conflicts. Moreover, not everything can be catched by tests, so the author is further responsible for having carefully gone through the proposed changed with regards to their implications to the codebase.

**NOTE:** The author must always **first review the PR themselves**, before requesting review from others, that means carefully having reviewed the full diff in GitHub.

## Basic Sanity

Every PR must minimally involve successfully going through the following steps: 

- [ ] Self-review completed (reviewed full diff in “Files changed”)
- [ ] No unnecessary files are included in the changes
- [ ] Ran `python tests/run.py` successfully (where applicable)
- [ ] Updated `/docs` (if behavior/API/config/user/etc changed)
- [ ] Added/updated docstrings per [Writing Docstrings](https://github.com/Vaquum/Limen/blob/main/docs/Developer/Writing-Docstrings.md) (for any changed public functions/classes)
- [ ] Updated CHANGELOG.md (unless only docs or other non-code aspect was changed)
- [ ] Added and/or updated tests (if behavior changed or new code paths added)
- [ ] Validated changes manually
- [ ] Validated changes with LLM
- [ ] Removed any extraneous examples/comments
- [ ] Linked issue is set to auto-close on merge (e.g., “Fixes #123”) when applicable

Once all these boxes are checked, the PR is ready for adding reviewers.
