from limen.sfd.reference_architecture.base import ReferenceModel
from limen.sfd.reference_architecture.logreg_binary import LogRegBinary
from limen.sfd.reference_architecture.logreg_binary import logreg_binary
from limen.sfd.reference_architecture.random_binary import RandomBinary
from limen.sfd.reference_architecture.random_binary import random_binary
from limen.sfd.reference_architecture.xgboost_regressor import XGBoostRegressor
from limen.sfd.reference_architecture.xgboost_regressor import xgboost_regressor

# tabpfn is optional - only import if available
try:
    from limen.sfd.reference_architecture.tabpfn_binary import TabPFNBinary
    from limen.sfd.reference_architecture.tabpfn_binary import tabpfn_binary
except ImportError:
    TabPFNBinary = None
    tabpfn_binary = None

__all__ = [
    'LogRegBinary',
    'RandomBinary',
    'ReferenceModel',
    'TabPFNBinary',
    'XGBoostRegressor',
    'logreg_binary',
    'random_binary',
    'tabpfn_binary',
    'xgboost_regressor',
]
