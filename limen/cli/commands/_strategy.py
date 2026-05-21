from typing import Any

from limen.experiment import GridStrategy
from limen.experiment import RandomStrategy
from limen.experiment.param_domain import ParamDomain


def build_search_strategy(uel_cfg: dict[str, Any],
                           sfd_cfg: dict[str, Any]) -> RandomStrategy | GridStrategy:

    '''
    Build a search strategy from UEL and SFD config dicts.

    Args:
        uel_cfg (dict): The uel section of a YAML experiment dict
        sfd_cfg (dict): The sfd section of a YAML experiment dict

    Returns:
        RandomStrategy | GridStrategy: Configured search strategy

    Raises:
        ValueError: If uel.search_strategy is not a mapping

    '''

    strategy_cfg = uel_cfg.get('search_strategy', {})
    if not isinstance(strategy_cfg, dict):
        raise ValueError(
            f"'uel.search_strategy' must be a mapping, got {type(strategy_cfg).__name__}"
        )
    strategy_type = strategy_cfg.get('type', 'random')
    params = {k: list(v) for k, v in (sfd_cfg.get('params') or {}).items()}
    domain = ParamDomain(params)

    if strategy_type == 'grid':
        return GridStrategy(domain)
    return RandomStrategy(domain)
