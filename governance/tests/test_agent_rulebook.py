"""Pin the agent-facing rulebook surfaces.

Upstream pins two posted guideline artifacts (`VAQUUM_PR_GUIDELINE.md` and
`VAQUUM_REPO_SPECIFICS.md`) by digest. This repository does not carry those
files: its rulebook is the constitution plus the reviewer brief, and both are
adopted from upstream rather than authored here. What matters is that the
three surfaces exist, that exactly one of them is canonical, and that the
other two route to it rather than restating it.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSTITUTION = REPO_ROOT / 'CLAUDE.md'
AGENTS = REPO_ROOT / 'AGENTS.md'
REVIEW_BRIEF = REPO_ROOT / '.github/copilot-instructions.md'


def test_constitution_is_present_and_carries_the_laws() -> None:
    """Verify the constitution exists with a parseable laws section."""
    assert CONSTITUTION.is_file()
    text = CONSTITUTION.read_text(encoding='utf-8')
    assert '## The laws' in text
    assert '## Beyond the laws' in text


def test_agents_file_routes_to_the_constitution_without_restating_it() -> None:
    """Verify AGENTS.md is a pointer rather than a second rulebook.

    The 257-line task-type regime it replaced was a parallel operating
    document whose useful parts now live in the slice template. A second
    rulebook is how the two drift apart.
    """
    assert AGENTS.is_file()
    text = AGENTS.read_text(encoding='utf-8')
    assert '[CLAUDE.md](CLAUDE.md)' in text
    assert '.github/copilot-instructions.md' in text
    assert len(text.splitlines()) < 20, 'AGENTS.md should route, not restate'


def test_review_brief_is_the_adopted_one() -> None:
    """Verify the reviewer brief is the full brief rather than a stub."""
    assert REVIEW_BRIEF.is_file()
    text = REVIEW_BRIEF.read_text(encoding='utf-8')
    assert '# PR review guideline' in text
    assert 'catastrophe' in text
    assert 'mediocre' in text
    assert len(text.splitlines()) > 50, 'the brief must not regress to a stub'
