"""Replay named cases through the existing indicator benchmark (set LIMEN_BENCHMARK_CASES)."""
import functools
import json
import os
from pathlib import Path
import runpy
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
case_path = Path(os.environ['LIMEN_BENCHMARK_CASES']).resolve()
os.environ['LIMEN_BENCHMARK_CASES'] = str(case_path)
CASES = json.loads(case_path.read_text())
namespace = runpy.run_path(str(ROOT / 'tests/benchmark_indicators.py'))
state = namespace['_worker'].__globals__
state['__file__'] = str(Path(__file__).resolve())
state['OUTPUTS'] = {name: tuple(case['outputs']) for name, case in CASES.items()}
state['TARGETS'] = tuple(CASES)
original_worker = state['_worker']


def replay_worker(args):
    sys.path.insert(0, str(args.code_root))
    from limen import indicators
    calls = {name: functools.partial(getattr(indicators, case['function']), **case['kwargs'])
             for name, case in CASES.items()}
    for name, calculate in calls.items():
        setattr(indicators, name, calculate)
    result = original_worker(args)
    result['steady_wall_ms'] = result['steady_ms']
    result['steady_ms'] = statistics.median(result['cpu_samples_ms'])
    result['steady_clock'] = 'process_time'
    return result


state['_worker'] = replay_worker
if __name__ == '__main__':
    state['main']()
