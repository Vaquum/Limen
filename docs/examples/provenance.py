import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def git(root, *arguments):
    return subprocess.check_output(['git', *arguments], cwd=root, text=True).strip()


def base_revision(root):
    if git(root, 'rev-parse', '--is-shallow-repository') != 'false':
        raise ValueError('Source provenance requires full Git history')
    main = git(root, 'rev-parse', 'origin/main')
    changes = git(root, 'rev-list', '--first-parent', '--reverse', main, '--not', 'HEAD').splitlines()
    return git(root, 'rev-parse', changes[0] + '^1') if changes else main


@lru_cache(maxsize=32)
def _first_parent_changes(root, revision):
    raw = subprocess.check_output(['git', 'log', '--first-parent', '--diff-merges=first-parent',
                                   '--format=%H', '--raw', '-z', '--no-renames', revision, '--'], cwd=root)
    entries = iter(os.fsdecode(raw).split('\0'))
    changes, commit = {}, ''
    for entry in entries:
        if entry.startswith(('\n:', ':')):
            changes.setdefault(next(entries), commit)
        elif entry:
            commit = entry.lstrip('\n')
    return changes


@lru_cache(maxsize=4096)
def history_change(root, revision, path, first_parent):
    commit = (_first_parent_changes(root, revision).get(path) if first_parent else
              git(root, 'log', '--no-patch', '-1', '--format=%H', revision, '--', path))
    if not commit:
        raise ValueError('No source change: ' + path)
    return commit


def latest_change(root, base, path):
    return history_change(root, git(root, 'rev-parse', base), path, True)


def merged_pull(commit, responses):
    matches = [item for item in responses if item['merged_at'] is not None and item['merge_commit_sha'] == commit and item['base']['ref'] == 'main' and item['base']['repo']['full_name'].lower() == 'vaquum/limen']
    if len(matches) != 1:
        raise ValueError('Source-changing main commit has no unique merged PR: ' + commit)
    return matches[0]['html_url']


def pull_associations(commit):
    command = ['gh', 'api', f'repos/Vaquum/Limen/commits/{commit}/pulls']
    for attempt in range(6):
        response = subprocess.run(command, text=True, capture_output=True, timeout=30)
        if response.returncode == 0:
            associations = json.loads(response.stdout)
            if not isinstance(associations, list):
                raise ValueError('Source PR associations must be a JSON array: ' + commit)
            return associations
        if response.stderr.strip() != 'unexpected end of JSON input' or attempt == 5:
            print(response.stderr.strip(), file=sys.stderr)
            raise subprocess.CalledProcessError(response.returncode, command, output=response.stdout, stderr=response.stderr)
        print(f'Incomplete GitHub response for {commit}; retry {attempt + 1}/5: {response.stderr.strip()}', file=sys.stderr)
        time.sleep(attempt + 1)
    raise RuntimeError('Source PR association attempts exhausted')


def source_paths(root):
    from check_evidence import all_reference_bindings
    references = {item['path'] for item in all_reference_bindings(root)}
    return sorted({'README.md', *references, *[str(path.relative_to(root)) for path in (root / 'docs').rglob('*.md')], *[str(path.relative_to(root)) for path in (root / 'limen').rglob('README.md')], *[str(path.relative_to(root)) for path in (root / 'docs/examples').rglob('*') if path.is_file() and path.suffix in ('.py', '.yaml', '.json', '.sh', '.toml') and 'tests' not in path.parts]})


def tree_objects(root, revision):
    objects = {}
    for entry in git(root, 'ls-tree', '-rz', revision).split('\0'):
        if entry:
            metadata, path = entry.split('\t', 1)
            objects[path] = metadata.split()[2]
    return objects


def identity(root, base, path, current, previous, *, head=None, ancestry=None):
    if previous.get(path) != current[path] or ancestry is not None and ancestry.get(path) != current[path]:
        return history_change(root, head or git(root, 'rev-parse', 'HEAD'), path, False), 'candidate'
    return history_change(root, base, path, True), 'merged'


def acquire(root, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"status":"running"}\n')
    base = base_revision(root)
    head = git(root, 'rev-parse', 'HEAD')
    current, previous = tree_objects(root, head), tree_objects(root, base)
    ancestry = tree_objects(root, git(root, 'merge-base', head, 'origin/main'))
    files, responses = [], {}
    for path in source_paths(root):
        commit, status = identity(root, base, path, current, previous, head=head, ancestry=ancestry)
        pull = None
        if status == 'merged':
            if commit not in responses:
                responses[commit] = pull_associations(commit)
            pull = merged_pull(commit, responses[commit])
        files.append({'path': path, 'commit': commit, 'status': status, 'merged_pr': pull, 'sha256': hashlib.sha256((root / path).read_bytes()).hexdigest()})
    record = {'schema_version': 1, 'base': base, 'files': files, 'responses': responses}
    output.write_text(json.dumps(record, indent=2) + '\n')
    return record


def validate_provenance(root, record):
    base = base_revision(root)
    if record['schema_version'] != 1 or record['base'] != base:
        raise ValueError('Source provenance belongs to another main ancestry')
    paths = source_paths(root)
    if len(record['files']) != len(paths) or {item['path'] for item in record['files']} != set(paths):
        raise ValueError('Source provenance inventory differs')
    head = git(root, 'rev-parse', 'HEAD')
    current, previous = tree_objects(root, head), tree_objects(root, base)
    ancestry = tree_objects(root, git(root, 'merge-base', head, 'origin/main'))
    for item in record['files']:
        commit, status = identity(root, base, item['path'], current, previous, head=head, ancestry=ancestry)
        if (item['commit'], item['status']) != (commit, status) or item['sha256'] != hashlib.sha256((root / item['path']).read_bytes()).hexdigest():
            raise ValueError('Source-changing commit or source bytes differ')
        expected = merged_pull(commit, record['responses'][commit]) if status == 'merged' else None
        if item['merged_pr'] != expected:
            raise ValueError('Source PR identity differs')


def main():
    parser = argparse.ArgumentParser(description='Bind canonical source changes to merged PRs on main.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2],
                        help='Inspect this full-history checkout without changing its source or revision.')
    args = parser.parse_args()
    root = args.root.resolve()
    result = acquire(root, args.output)
    validate_provenance(root, result)
    print(f"Bound {len(result['files'])} sources to first-parent main history and explicit candidate changes.")


if __name__ == '__main__':
    main()
