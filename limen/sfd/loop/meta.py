'''Hand-maintained metadata for the Loop payload compiler.

Two small mappings that cannot be derived by introspection:

- LABEL_TARGET_COLUMNS: label function name → target column name produced
- SCALER_NAME_MAP: PascalCase class name (Loop's format) → SCALER_REGISTRY key

NOTE: This module is part of the temporary `limen.sfd.loop` subpackage that
will be removed when RFC-1005 (YAML compiler) lands. See README.md.
'''


LABEL_TARGET_COLUMNS: dict[str, str] = {
    'forward_breakout_target': 'forward_breakout',
}


def get_target_column(label_name: str) -> str:

    '''
    Compute the target column name produced by a label function.

    Args:
        label_name (str): The label function short name as used in the payload

    Returns:
        str: The target column name. Falls back to label_name itself when no
            explicit override is registered (works for labels like quantile_flag
            whose function name matches the column name they produce)

    '''

    return LABEL_TARGET_COLUMNS.get(label_name, label_name)


SCALER_NAME_MAP: dict[str, str] = {
    'LinearScaler': 'linear',
    'LogRegScaler': 'logreg',
    'RobustScaler': 'robust',
    'RankGaussScaler': 'rank_gauss',
}


__all__ = [
    'LABEL_TARGET_COLUMNS',
    'SCALER_NAME_MAP',
    'get_target_column',
]
