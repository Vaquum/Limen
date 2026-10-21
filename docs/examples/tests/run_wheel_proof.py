import argparse
import json
from pathlib import Path

from wheel_proof import prove_wheel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--temporary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--released-wheel', type=Path)
    parser.add_argument('--released-bundle', type=Path)
    args = parser.parse_args()
    result = prove_wheel(args.root, args.temporary, released_wheel=args.released_wheel,
                         released_bundle=args.released_bundle)
    result['scenario_reports'] = [[interface, scenario, report]
                                 for (interface, scenario), report in result['scenario_reports'].items()]
    args.output.write_text(json.dumps(result, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
