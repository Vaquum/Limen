from limen.scalers.causal_rolling_robust_scaler import CausalRollingRobustScaler
from limen.scalers.linear_scaler import LinearScaler
from limen.scalers.logreg_scaler import LogRegScaler
from limen.scalers.robust_scaler import RobustScaler
from limen.scalers.rank_gauss_scaler import RankGaussScaler

SCALER_REGISTRY: dict[str, type] = {
    'linear': LinearScaler,
    'logreg': LogRegScaler,
    'robust': RobustScaler,
    'rank_gauss': RankGaussScaler,
    'causal_rolling_robust': CausalRollingRobustScaler,
}
