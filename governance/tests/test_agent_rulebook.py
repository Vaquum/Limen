"""Pin the requested action rule and the single pointer to it."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAUDE = REPO_ROOT / 'CLAUDE.md'
AGENTS = REPO_ROOT / 'AGENTS.md'
REVIEW_BRIEF = REPO_ROOT / '.github/copilot-instructions.md'


def test_claude_file_routes_to_agents_without_restating_it() -> None:
    assert CLAUDE.is_file()
    assert CLAUDE.read_text(encoding='utf-8') == 'Read AGENTS.md.\n'


def test_agents_file_contains_the_requested_action_rule() -> None:
    assert AGENTS.is_file()
    assert AGENTS.read_text(encoding='utf-8') == (
        'If you were asked a question, answer it now. '
        'If you were asked to do work, do it now. '
        'If you need more information, ask it now.\n'
    )


def test_review_brief_stays_removed() -> None:
    """Verify the removed reviewer brief does not return.

    Removed by owner decision: no automated reviewer brief may exist.
    """
    assert not REVIEW_BRIEF.exists()
