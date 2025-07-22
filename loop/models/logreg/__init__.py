# Make logreg a Python package

import loop.models.logreg.regime_multiclass as regime_multiclass
import loop.models.logreg.breakout_regressor_ridge as breakout_regressor_ridge

__all__ = [
    'regime_multiclass',
    'breakout_regressor_ridge'
]