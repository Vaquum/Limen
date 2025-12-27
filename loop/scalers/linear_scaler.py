import re
import polars as pl

DEFAULT_SCALING_RULES = {
    r'^open$': 'standard',
    r'^close$': 'standard',
    r'^high$': 'standard',
    r'^low$': 'standard',
    r'^mean$': 'standard',
    r'^median$': 'standard',
    r'^std$': 'standard',
    r'^iqr$': 'standard',
    r'^maker_ratio$': 'none',
    r'^volume$': 'log_standard',
    r'^no_of_trades$': 'log_standard',
    r'^open_liquidity$': 'log_standard',
    r'^high_liquidity$': 'log_standard',
    r'^low_liquidity$': 'log_standard',
    r'^close_liquidity$': 'log_standard',
    r'^liquidity_sum$': 'log_standard',
    r'^maker_volume$': 'standard',
    r'^maker_liquidity$': 'standard',
    r'^roc(_\d+)?$': 'standard',
    r'^atr(_\d+)?$': 'standard',
    r'^(sma|short_sma|medium_sma|long_sma)$': 'standard',
    r'^macd$': 'standard',
    r'^ppo(_\d+_\d+)?$': 'standard',
    r'^ppo_signal(_\d+)?$': 'standard',
    r'^wilder_rsi(_\d+)?$': 'standard',
    r'^vwap$': 'standard',
    r'^imbalance$': 'standard',
    r'^body_pct$': 'standard',
    r'^returns$': 'standard',
    r'^atr_percent_sma(_\d+)?$': 'standard',
    r'^close_position$': 'standard',
    r'^close_volatility(_\d+)?$': 'standard',
    r'^trend_strength$': 'standard',
    r'^volume_regime$': 'standard',
    r'^distance_from_high(_\d+)?$': 'log_standard',
    r'^distance_from_low(_\d+)?$': 'log_standard',
    r'^breakout_ema$': 'standard',
    r'^gap_high$': 'standard',
    r'^tenkan(_\d+)?$': 'standard',
    r'^kijun(_\d+)?$': 'standard',
    r'^senkou_a(_\d+)?$': 'standard',
    r'^senkou_b(_\d+)?$': 'standard',
    r'^chikou(_\d+)?$': 'standard',
    r'^price_range_position(_\d+)?$': 'standard',
    r'^range_pct$': 'standard',
    r'.*': 'none'
}


def build_rules(
    overrides: dict[str, list[str]] | None = None,
    base_rules: dict[str, str] | None = None,
) -> dict[str, str]:
    
    """
    Build scaling rules by combining base rules and user overrides.

    Args:
        overrides: User-specified rules in sklearn style, e.g.
            {'standard': ['open', 'close'], 'log_standard': ['volume']}.
        base_rules: Regex-based rules to start with.
    """

    rules = dict(base_rules or DEFAULT_SCALING_RULES)

    if overrides:
        for rule, cols in overrides.items():
            for col in cols:
                rules[fr'^{col}$'] = rule

    return rules


def get_scaling_rule(col: str, rules: dict[str, str], default: str = 'none') -> str:

    """
    Find the matching scaling rule for a column name.

    Args:
        col: Column name.
        rules: Regex-to-rule mapping.
        default: Rule to use if no pattern matches.

    Returns:
        The scaling rule name.
    """

    for pattern, rule in rules.items():
        if re.match(pattern, col):
            return rule
        
    return default


class LinearScaler:
    def __init__(
        self,
        x_train: pl.DataFrame,
        rules: dict[str, str] | None = None,
        default: str = 'standard',
    ):
        
        """
        Linear transformation utility for scaling features.

        Args:
            x_train: Training DataFrame.
            rules: Regex-to-rule mapping.
            default: Fallback scaling rule.
        """

        self.rules = rules or DEFAULT_SCALING_RULES
        self.default = default
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}

        for col in x_train.columns:
            rule = get_scaling_rule(col, self.rules, self.default)

            if rule == "log_standard":
                mean = x_train.select(pl.col(col).log1p().mean()).item()
                std = x_train.select(pl.col(col).log1p().std(ddof=0)).item()
            elif rule == "standard":
                mean = x_train[col].mean()
                std = x_train[col].std(ddof=0)
            else:
                continue

            self.means[col] = mean
            self.stds[col] = std

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        
        """
        Apply linear scaling transformation.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """

        exprs = []
        for col in df.columns:
            rule = get_scaling_rule(col, self.rules, self.default)

            if rule == 'standard':
                exprs.append(((pl.col(col) - self.means[col]) / self.stds[col]).alias(col))
            
            elif rule == 'log_standard':
                exprs.append(((pl.col(col).log1p() - self.means[col]) / self.stds[col]).alias(col))
            
            elif rule == 'divide_100':
                exprs.append((pl.col(col) / 100).alias(col))
            
            elif rule == 'none':
                exprs.append(pl.col(col).alias(col))

        return df.with_columns(exprs)


def inverse_transform(df: pl.DataFrame, scaler: LinearScaler) -> pl.DataFrame:

    """
    Apply inverse scaling transformation.

    Args:
        df: DataFrame to inverse transform.
        scaler: LinearScaler instance with fitted parameters.

    Returns:
        DataFrame in original scale.
    """

    exprs = []
    for col in df.columns:
        rule = get_scaling_rule(col, scaler.rules, scaler.default)

        if rule == 'standard':
            exprs.append((pl.col(col) * scaler.stds[col] + scaler.means[col]).alias(col))
        
        elif rule == 'log_standard':
            exprs.append(((pl.col(col) * scaler.stds[col] + scaler.means[col]).exp() - 1).alias(col))
        
        elif rule == 'divide_100':
            exprs.append((pl.col(col) * 100).alias(col))
        
        elif rule == 'none':
            exprs.append(pl.col(col).alias(col))

    return df.with_columns(exprs)
