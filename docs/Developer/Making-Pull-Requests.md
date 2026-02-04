# Making Pull Requests

We have PR reviews, but the responsibility is with the author of the PR to ensure that the PR promoted for review is in a mergeable state: CI is green, tests pass locally, and there are no merge conflicts.

**NOTE:** The author must always first review the PR themselves, before requesting review from others, that means carefully having reviewed the full diff in GitHub.

## Basic Sanity

Every code contributing PR must minimally involve successfully going through the following steps: 

- [ ] Self-review completed (reviewed full diff in “Files changed”)
- [ ] No unnecessary files are included in the changes
- [ ] Ran `python tests/run.py` successfully (record results in PR Test Plan)
- [ ] Updated /docs (if behavior/API/config/user workflow changed)
- [ ] Added/updated docstrings per [Writing Docstrings](https://github.com/Vaquum/Limen/blob/main/docs/Developer/Writing-Docstrings.md) (for any changed public functions/classes)
- [ ] Updated CHANGELOG.md (if user-facing / release-note-worthy)
- [ ] Added/updated tests (if behavior changed or new code paths added)
- [ ] Validated changes manually as needed (record steps in PR Test Plan)
- [ ] If an LLM was used, verified generated changes and removed extraneous examples/comments
- [ ] Linked issue is set to auto-close on merge (e.g., “Fixes #123”) when applicable

Once all these boxes are checked, the PR is ready for adding reviewers.
