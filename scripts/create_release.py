#!/usr/bin/env python3
"""Automated release creation script using Claude AI."""
# ruff: noqa: T201, S607, S603, BLE001

import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import anthropic


def read_file(filepath: str) -> str:
    """Read content from a file."""
    with Path(filepath).open() as f:
        return f.read()


def get_current_version() -> str:
    """Extract current version from pyproject.toml."""
    content = read_file('pyproject.toml')
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError('Could not find version in pyproject.toml')
    return match.group(1)


def get_git_log_since_last_tag() -> str:
    """Get git log since the last tag, limited to prevent context overflow."""
    MAX_COMMITS = 100
    try:
        # Get the latest tag
        result = subprocess.run(
            ['git', 'describe', '--tags', '--abbrev=0'],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            last_tag = result.stdout.strip()
            # Get commits since that tag, limited to MAX_COMMITS
            log_result = subprocess.run(
                ['git', 'log', f'{last_tag}..HEAD', '--oneline', '-n', str(MAX_COMMITS)],
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            # No tags exist, get recent commits
            log_result = subprocess.run(
                ['git', 'log', '--oneline', '-n', str(MAX_COMMITS)],
                capture_output=True,
                text=True,
                check=True,
            )
    except subprocess.CalledProcessError as e:
        print(f'Error getting git log: {e}')
        return ''

    return log_result.stdout.strip()


def create_prompt() -> str:
    """Create the prompt for Claude to generate release information."""
    docs = read_file('docs/Developer/Making-Release.md')
    version = get_current_version()
    git_log = get_git_log_since_last_tag()
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
    "version": "{version}",
    "tag": "v{version}",
    "release_name": "<creative name based on lunar calendar animals>",
    "release_notes": "<markdown formatted release notes with Summary and Details sections>"
}}

IMPORTANT REQUIREMENTS:
1. The tag MUST use lowercase 'v' prefix (e.g., v{version})
2. The release_name should be a creative play on lunar calendar animals (year, month, day, hour)
3. The release_notes must include:
   - ## Summary section: concise bullet points of key changes
   - ## Details section: beautiful essay-style comprehensive description
4. Analyze the git log carefully to understand what changed
5. Return ONLY valid JSON, no other text

Generate the release information now:"""

    return prompt


def parse_claude_response(response_text: str) -> dict:
    """Parse Claude's JSON response."""
    import json

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

    # Create the prompt
    prompt = create_prompt()
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

        # Parse the response
        release_info = parse_claude_response(response_text)

        print('\nRelease Information:')
        print(f'  Version: {release_info["version"]}')
        print(f'  Tag: {release_info["tag"]}')
        print(f'  Name: {release_info["release_name"]}')
        print('\nRelease Notes Preview:')
        print(release_info['release_notes'][:500] + '...')

        # Create git tag
        create_git_tag(
            release_info['tag'],
            f'Release {release_info["version"]}: {release_info["release_name"]}'
        )

        # Create GitHub release
        create_github_release(
            release_info['tag'],
            release_info['release_name'],
            release_info['release_notes']
        )

        print('\n✓ Release created successfully!')

    except Exception as e:
        print(f'\nError creating release: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
