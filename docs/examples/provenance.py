import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git(root, *arguments):
    return subprocess.check_output(['git', *arguments], cwd=root, text=True).strip()


def base_revision(root):
    if git(root, 'rev-parse', '--is-shallow-repository') != 'false':
        raise ValueError('Source provenance requires full Git history')
    return git(root, 'merge-base', 'HEAD', 'origin/main')


def latest_change(root, base, path):
    commit = git(root, 'log', '--first-parent', '--diff-merges=first-parent', '--no-patch', '-1', '--format=%H', base, '--', path)
    if not commit:
        raise ValueError('No first-parent source change: ' + path)
    return commit


def merged_pull(commit, responses):
    matches = [item for item in responses if item['merged_at'] is not None and item['merge_commit_sha'] == commit and item['base']['ref'] == 'main' and item['base']['repo']['full_name'].lower() == 'vaquum/limen']
    if len(matches) != 1:
        raise ValueError('Source-changing main commit has no unique merged PR: ' + commit)
    return matches[0]['html_url']


def source_paths(root):
    from check_evidence import all_reference_bindings
    references = {item['path'] for item in all_reference_bindings(root)}
    return sorted({'README.md', *references, *[str(path.relative_to(root)) for path in (root / 'docs').rglob('*.md')], *[str(path.relative_to(root)) for path in (root / 'limen').rglob('README.md')], *[str(path.relative_to(root)) for path in (root / 'docs/examples').rglob('*') if path.is_file() and path.suffix in ('.py', '.yaml', '.json') and 'tests' not in path.parts]})


def identity(root, base, path):
    current = git(root, 'rev-parse', 'HEAD:' + path)
    previous = subprocess.run(['git', 'rev-parse', '--verify', base + ':' + path], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if previous.returncode or previous.stdout.strip() != current:
        return git(root, 'log', '-1', '--format=%H', 'HEAD', '--', path), 'candidate'
    return latest_change(root, base, path), 'merged'


def acquire(root, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"status":"running"}\n')
    base = base_revision(root)
    files, responses = [], {}
    for path in source_paths(root):
        commit, status = identity(root, base, path)
        pull = None
        if status == 'merged':
            if commit not in responses:
                responses[commit] = json.loads(subprocess.check_output(['gh', 'api', f'repos/Vaquum/Limen/commits/{commit}/pulls'], text=True))
            pull = merged_pull(commit, responses[commit])
        files.append({'path': path, 'commit': commit, 'status': status, 'merged_pr': pull, 'sha256': hashlib.sha256((root / path).read_bytes()).hexdigest()})
    record = {'schema_version': 1, 'base': base, 'files': files, 'responses': responses}
    output.write_text(json.dumps(record, indent=2) + '\n')
    return record


def validate_provenance(root, record):
    base = base_revision(root)
    if record['schema_version'] != 1 or record['base'] != base:
        raise ValueError('Source provenance belongs to another main ancestry')
    if len(record['files']) != len(source_paths(root)) or {item['path'] for item in record['files']} != set(source_paths(root)):
        raise ValueError('Source provenance inventory differs')
    for item in record['files']:
        commit, status = identity(root, base, item['path'])
        if (item['commit'], item['status']) != (commit, status) or item['sha256'] != hashlib.sha256((root / item['path']).read_bytes()).hexdigest():
            raise ValueError('Source-changing commit or source bytes differ')
        expected = merged_pull(commit, record['responses'][commit]) if status == 'merged' else None
        if item['merged_pr'] != expected:
            raise ValueError('Source PR identity differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bind canonical source changes to merged PRs on main.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    result = acquire(root, args.output)
    validate_provenance(root, result)
    print(f"Bound {len(result['files'])} sources to first-parent main history and explicit candidate changes.")
