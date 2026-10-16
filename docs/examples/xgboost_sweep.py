"""Run the built-in XGBoost research sweep against its real market data source."""
import argparse
from pathlib import Path

from limen.experiment import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy
from limen.sfd.foundational_sfd import xgboost_regressor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=('test', 'prod'), default='test')
    parser.add_argument('--permutations', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error('--permutations must be positive')
    space = xgboost_regressor.params()
    space['device'] = [args.device]
    strategy = RandomStrategy(ParamDomain(space), seed=args.seed)
    loop = UniversalExperimentLoop(sfd=xgboost_regressor, search_strategy=strategy,
                                  test_mode=args.env == 'test', experiment_dir=args.output)
    loop.run('results', n_permutations=args.permutations, prep_each_round=True)


if __name__ == '__main__':
    main()
