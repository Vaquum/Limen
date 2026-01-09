from limen.sfd.reference_architecture.logreg_binary import logreg_binary
from limen.sfd.reference_architecture.random_binary import random_binary
from limen.sfd.reference_architecture.xgboost_regressor import xgboost_regressor

# tabpfn is optional - only import if available
try:
    from limen.sfd.reference_architecture.tabpfn_binary import tabpfn_binary
except ImportError:
    tabpfn_binary = None

__all__ = [
    'logreg_binary',
    'random_binary',
    'tabpfn_binary',
    'xgboost_regressor',
]
