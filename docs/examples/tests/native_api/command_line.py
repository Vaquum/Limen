import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from ruamel.yaml import YAML


# region lifecycle
def run(config, output):
    project = output / 'project'
    project.mkdir()
    backup = output / 'backup.git'
    subprocess.run(['git', 'init', '--bare', '--initial-branch=main', str(backup)], check=True, timeout=10)
    subprocess.run(['git', 'init', '--initial-branch=main', str(project)], check=True, timeout=10)
    for key, value in [('user.name', 'Limen documentation'), ('user.email', 'docs@example.invalid'), ('commit.gpgsign', 'false')]:
        subprocess.run(['git', '-C', str(project), 'config', key, value], check=True, timeout=10)
    (project / 'limen.toml').write_text('[project]\nname="book-command-line"\n[store]\nbackup_remote=' + json.dumps(str(backup.resolve())) + '\n')
    (project / '.gitignore').write_text('results/dev/\n')
    (project / 'manifests').mkdir()
    subprocess.run(['git', '-C', str(project), 'add', '.'], check=True, timeout=10)
    subprocess.run(['git', '-C', str(project), 'commit', '-m', 'docs: establish the recorded campaign'], check=True, timeout=10)
    executable = Path(sys.executable).with_name('limen.exe' if os.name == 'nt' else 'limen')
    calls = []

    def command(*args, cwd=project):
        completed = subprocess.run([str(executable), *map(str, args)], cwd=cwd, text=True, capture_output=True, timeout=30)
        print(completed.stdout, end='')
        print(completed.stderr, end='', file=sys.stderr)
        with (output / 'commands.log').open('a') as stream:
            stream.write(completed.stdout + completed.stderr)
        completed.check_returncode()
        calls.append({'args': list(map(str, args)), 'returncode': completed.returncode})

    command('--version')
    command('list-templates')
    command('init', 'installed-template.yaml', '--template', 'logreg_binary')
    config['metadata']['name'] = 'book_cli_research'
    config['metadata']['mode'] = 'production'
    config['uel'].update(output_format='parquet', checkpoint_interval=1,
                         recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    source = project / 'manifests/research.yaml'
    with source.open('w') as stream:
        YAML().dump(config, stream)
    command('validate', source)
    command('profile', source)
    command('run', '--dry-run', source)
    command('commit', source, '--message', 'docs: record the research definition')
    index_path = project / 'manifests/committed/index.json'
    parent_id = json.loads(index_path.read_text())['manifests'][0]['id']
    command('ls')
    command('lineage', parent_id)
    command('fork', parent_id, 'book-cli-fork')
    fork_path = project / 'manifests/book-cli-fork.yaml'
    fork = YAML(typ='safe').load(fork_path)
    fork['metadata']['mode'] = 'production'
    with fork_path.open('w') as stream:
        YAML().dump(fork, stream)
    command('commit', fork_path, '--parent', parent_id, '--message', 'docs: record the child research definition')
    command('reindex')
    command('run', '--no-progress-bar', 'manifest://' + parent_id)
    runs = list((project / 'results').rglob('metadata.json'))
    if len(runs) != 1:
        raise ValueError('The command-line scenario must create one recorded research run')
    directory = runs[0].parent
    command('run', '--resume', directory, '--no-progress-bar')
    command('backup')
    restored = output / 'restored'
    command('new', restored, '--from', backup, cwd=output)
    command('ls', cwd=restored)
    index = json.loads(index_path.read_text())
    tracked = subprocess.check_output(['git', '-C', str(project), 'ls-files', '-z'], text=True).split('\0')
    digests = {name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in tracked if name}
    return {'commands': calls, 'manifests': index['manifests'], 'parent_id': parent_id,
            'run': str(directory.relative_to(output)), 'project_files': digests,
            'backup': 'tracked campaign files through a local Git remote; ignored runtime leases and holdout history are not restored'}
# endregion lifecycle
