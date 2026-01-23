# Plan: Retroactive Ruff Fix with Preserved Styling

## Goal
Fix ~700 logical/syntax errors identified by Ruff (bugbear, pyupgrade, etc.) while strictly preserving the existing code style (indentation, quotes, imports). This replaces the previous attempt which unintentionally reformatted the entire codebase.

## Steps

1.  **Reset State**
    *   Hard reset current branch `ci/configure-ruff-correctly` to commit `8395874` ("feat: configure ruff rules").
    *   *Reason:* This is the last stable point before the "bad" formatting changes.

2.  **Configure Ruff for Targeted Fixes**
    *   Modify `pyproject.toml` to:
        *   **Enable:** Logical checks (`F`, `B`, `UP`, `ANN`, `S`, etc.) and basic style checks (`W`, `E`).
        *   **Disable:** `I` (Isort/Import sorting), `D` (Docstrings), `Q` (Quotes).
        *   **Settings:** Remove strict quote enforcement settings to allow mixed usage (single for strings, double for f-strings).
    *   *Reason:* Preventing `I`, `D` and strict quote enforcement stops the massive "noise" in the diff.

3.  **Apply Fixes**
    *   Run `ruff check . --fix`.
    *   *Outcome:* This will fix unused imports, old syntax (e.g., `Type` -> `list`), and unused variables, but leave imports and docstrings alone.

4.  **Create New Commit**
    *   Commit changes with message: `fix: ruff errors (preserve styling)`.

5.  **Restore Lost History (Cherry-Pick)**
    *   Re-apply valuable commits that happened after the reset:
        1.  `d7275f8`: "ci: remove ruff formatting check"
        2.  `700f577`: "ver: bumped to `1.32.1`"
    *   *Skip:* `67daad7` ("chore: enforce ruff configuration") because it likely contains the enforced formatting we are trying to avoid.

6.  **Push**
    *   Force push to update the remote branch `ci/configure-ruff-correctly`.

## Outcome
A clean history where the codebase is modernized and bug-free, but looks visually identical to the original style, followed by the latest CI and version updates.
