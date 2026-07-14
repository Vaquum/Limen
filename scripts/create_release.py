#!/usr/bin/env python3
"""Automated release creation script using Claude AI.

The model composes prose only: the release title and the release notes
body. Every identifier is computed mechanically — the tag derives from
the pyproject.toml version, must match ``TAG_RE``, and any model
deviation on identifiers aborts the release before a git command runs.
Traceability (merged pull requests, compare link, changelog anchor) is
appended after the model returns; artifact SHA-256 digests are appended
by ``pr_publish_pypi.yml`` once the distributions exist.
"""
# ruff: noqa: T201, S607, S603, BLE001

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Final


DEFAULT_RELEASE_DOCS_URL = (
    'https://raw.githubusercontent.com/'
    'Vaquum/dev-docs/551e77b251dc3e70548b8bcd645d702c8f80e3b6/src/Making-Release.md'
)
RELEASE_DOCS_URL = os.getenv('RELEASE_DOCS_URL', DEFAULT_RELEASE_DOCS_URL)
REPO_URL: Final[str] = 'https://github.com/Vaquum/Limen'
TAG_RE: Final[re.Pattern[str]] = re.compile(r'^v\d+\.\d+\.\d+$')
URL_FETCH_TIMEOUT: Final[int] = 30
MAX_COMMITS: Final[int] = 100


def read_file(filepath: str) -> str:
    """Read content from a file."""
    with Path(filepath).open() as f:
        return f.read()


def fetch_url(url: str) -> str:
    """Fetch content from a URL with timeout and error handling."""
    try:
        with urllib.request.urlopen(url, timeout=URL_FETCH_TIMEOUT) as response:  # noqa: S310
            return response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP error fetching {url}: {e.code} {e.reason}') from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise RuntimeError(f'Timed out fetching {url} after {URL_FETCH_TIMEOUT}s') from e
        raise RuntimeError(f'URL error fetching {url}: {e.reason}') from e


def get_current_version() -> str:
    """Extract current version from pyproject.toml."""
    content = read_file('pyproject.toml')
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError('Could not find version in pyproject.toml')
    return match.group(1)


def get_previous_tag() -> str | None:
    """Return the most recent tag reachable from HEAD, if any."""
    result = subprocess.run(
        ['git', 'describe', '--tags', '--abbrev=0'],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_git_log_since(previous_tag: str | None) -> str:
    """Get git log since the previous tag, limited to prevent context overflow."""
    rev_range = [f'{previous_tag}..HEAD'] if previous_tag else []
    log_result = subprocess.run(
        ['git', 'log', *rev_range, '--oneline', '-n', str(MAX_COMMITS)],
        capture_output=True,
        text=True,
        check=True,
    )
    return log_result.stdout.strip()


def get_merged_pr_subjects(previous_tag: str | None) -> list[str]:
    """Return commit subjects since the previous tag."""
    rev_range = [f'{previous_tag}..HEAD'] if previous_tag else []
    result = subprocess.run(
        ['git', 'log', *rev_range, '--format=%s'],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def extract_pr_numbers(subjects: list[str]) -> list[int]:
    """Extract pull request numbers from merge and squash commit subjects."""
    numbers: set[int] = set()
    for subject in subjects:
        match = re.match(r'^Merge pull request #(\d+)', subject) or re.search(r'\(#(\d+)\)\s*$', subject)
        if match:
            numbers.add(int(match.group(1)))
    return sorted(numbers)


def changelog_anchor(version: str, changelog: str) -> str:
    """Derive the GitHub heading anchor of the version's changelog entry."""
    heading_re = re.compile(rf'^## \[{re.escape(version)}] - \d{{4}}-\d{{2}}-\d{{2}}$', re.MULTILINE)
    match = heading_re.search(changelog)
    if match is None:
        raise ValueError(f'create_release CHANGELOG.md has no entry for version {version}')
    heading = match.group(0).removeprefix('## ')
    return re.sub(r'[^\w\- ]', '', heading.lower()).strip().replace(' ', '-')


def compute_tag(version: str, release_info: dict[str, str]) -> str:
    """Derive the tag from the pyproject version, rejecting model deviation on identifiers."""
    tag = f'v{version}'
    if (
        not TAG_RE.match(tag)
        or release_info.get('tag') not in (None, tag)
        or release_info.get('version') not in (None, version)
    ):
        raise ValueError(f'create_release tag must be {tag} (model may not choose identifiers)')
    return tag


def build_traceability(
    tag: str,
    previous_tag: str | None,
    pr_numbers: list[int],
    anchor: str,
) -> str:
    """Compose the mechanical traceability appendix for the release notes."""
    lines = ['## Traceability', '']
    if pr_numbers:
        lines.append('- Merged pull requests: ' + ', '.join(f'#{number}' for number in pr_numbers))
    if previous_tag:
        lines.append(f'- Compare: {REPO_URL}/compare/{previous_tag}...{tag}')
    lines.append(f'- Changelog: {REPO_URL}/blob/{tag}/CHANGELOG.md#{anchor}')
    lines.append('- Artifact SHA-256 digests are appended by the publish workflow after the distributions build.')
    return '\n'.join(lines)


def create_prompt(version: str, previous_tag: str | None) -> str:
    """Create the prompt for Claude to generate the release prose."""
    docs = fetch_url(RELEASE_DOCS_URL)
    git_log = get_git_log_since(previous_tag)
    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    prompt = f"""You are creating a new release for the Limen project.

CURRENT STATE:
- Version in pyproject.toml: {version}
- Current date/time: {current_date}

RELEASE DOCUMENTATION:
{docs}

GIT CHANGES SINCE LAST RELEASE:
{git_log}

TASK:
Based on the release documentation and the git changes above, create a JSON response with the following structure:
{{
    "release_name": "<creative name based on lunar calendar animals>",
    "release_notes": "<markdown formatted release notes with Summary and Details sections>"
}}

IMPORTANT REQUIREMENTS:
1. The release_name should be a creative play on lunar calendar animals (year, month, day, hour)
2. The release_notes must include:
   - ## Summary section: concise bullet points of key changes
   - ## Details section: beautiful essay-style comprehensive description
3. Analyze the git log carefully to understand what changed
4. Return ONLY valid JSON, no other text

Generate the release information now:"""

    return prompt


def parse_claude_response(response_text: str) -> dict[str, str]:
    """Parse Claude's JSON response."""
    # Try to parse directly first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # If direct parsing fails, try to extract JSON more carefully
    # Look for the first { and find its matching }
    start = response_text.find('{')
    if start == -1:
        raise ValueError(f'Could not find JSON in response: {response_text}') from None

    # Count braces to find the matching closing brace, accounting for strings
    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start, len(response_text)):
        char = response_text[i]

        # Handle escape sequences
        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        # Track if we're inside a string
        if char == '"':
            in_string = not in_string
            continue

        # Only count braces outside of strings
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found matching brace
                    json_str = response_text[start:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        raise ValueError(f'Invalid JSON extracted: {json_str}') from e

    raise ValueError(f'Could not find complete JSON object in response: {response_text}') from None


def tag_exists(tag: str) -> bool:
    """Check if a git tag already exists locally or remotely."""
    try:
        # Check local tags first
        result = subprocess.run(
            ['git', 'tag', '-l', tag],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.stdout.strip() == tag:
            return True

        # Check remote tags
        result = subprocess.run(
            ['git', 'ls-remote', '--tags', 'origin', f'refs/tags/{tag}'],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, Exception) as e:
        # If we can't check, assume tag doesn't exist and let the
        # actual tag creation fail with a proper error message
        print(f'Warning: Could not check if tag exists: {e}')
        return False


def create_git_tag(tag: str, message: str) -> None:
    """Create and push a git tag."""
    subprocess.run(['git', 'tag', '-a', tag, '-m', message], check=True)
    subprocess.run(['git', 'push', 'origin', tag], check=True)
    print(f'Created and pushed tag: {tag}')


def create_github_release(tag: str, title: str, notes: str) -> None:
    """Create a GitHub release using gh CLI."""
    # Write notes to a temporary file to handle multiline content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        notes_file = f.name
        f.write(notes)

    try:
        subprocess.run(
            ['gh', 'release', 'create', tag, '--title', title, '--notes-file', notes_file],
            check=True,
        )
        print(f'Created GitHub release: {title} ({tag})')
    finally:
        # Clean up temporary file
        Path(notes_file).unlink(missing_ok=True)


def main() -> None:
    """Main function to orchestrate the release creation."""
    # Imported here so the module imports without the release extra,
    # which tests/test_release_contract.py requires offline.
    import anthropic

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print('Error: ANTHROPIC_API_KEY environment variable not set')
        sys.exit(1)

    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print('Error: GITHUB_TOKEN environment variable not set')
        sys.exit(1)

    # Get model from environment variable or use default
    model = os.getenv('ANTHROPIC_MODEL', 'claude-opus-4-6')
    print(f'Using model: {model}')

    print('Creating release with Claude AI...')

    version = get_current_version()
    previous_tag = get_previous_tag()

    # Compute every identifier mechanically before the model is consulted
    tag = compute_tag(version, {})
    anchor = changelog_anchor(version, read_file('CHANGELOG.md'))

    # Check if tag already exists
    if tag_exists(tag):
        print(f'\n✓ Tag {tag} already exists. Skipping release creation.')
        print('This is expected when the version in pyproject.toml has not changed.')
        sys.exit(0)

    # Create the prompt
    prompt = create_prompt(version, previous_tag)
    print(f'\nPrompt length: {len(prompt)} characters')

    # Call Claude API
    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )

        response_text = message.content[0].text
        print(f'\nClaude response received ({len(response_text)} characters)')

        # Parse the response and reject any identifier deviation
        release_info = parse_claude_response(response_text)
        tag = compute_tag(version, release_info)

        notes = release_info['release_notes'] + '\n\n' + build_traceability(
            tag,
            previous_tag,
            extract_pr_numbers(get_merged_pr_subjects(previous_tag)),
            anchor,
        )

        print('\nRelease Information:')
        print(f'  Version: {version}')
        print(f'  Tag: {tag}')
        print(f'  Name: {release_info["release_name"]}')
        print('\nRelease Notes Preview:')
        print(notes[:500] + '...')

        # Create git tag
        create_git_tag(
            tag,
            f'Release {version}: {release_info["release_name"]}'
        )

        # Create GitHub release
        create_github_release(
            tag,
            release_info['release_name'],
            notes
        )

        print('\n✓ Release created successfully!')

    except Exception as e:
        print(f'\nError creating release: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
