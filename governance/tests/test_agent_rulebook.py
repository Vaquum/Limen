"""Pin the agent-facing rulebook surfaces.

Upstream pins two posted guideline artifacts (`VAQUUM_PR_GUIDELINE.md` and
`VAQUUM_REPO_SPECIFICS.md`) by digest. This repository does not carry those
files: its rulebook is the constitution, adopted from upstream rather than
authored here, with AGENTS.md routing to it. The former reviewer brief
(`.github/copilot-instructions.md`) was removed by owner decision and must
not return. What matters is that the two surfaces exist, that exactly one
of them is canonical, and that the other routes to it rather than restating it.
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
    document whose useful parts now live in the issue templates. A second
    rulebook is how the two drift apart.
    """
    assert AGENTS.is_file()
    text = AGENTS.read_text(encoding='utf-8')
    assert '[CLAUDE.md](CLAUDE.md)' in text
    assert len(text.splitlines()) < 20, 'AGENTS.md should route, not restate'


def test_review_brief_stays_removed() -> None:
    """Verify the removed reviewer brief does not return.

    Removed by owner decision: no automated reviewer brief may exist.
    """
    assert not REVIEW_BRIEF.exists()
