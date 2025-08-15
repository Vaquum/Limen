import xgboost as xgb
import polars as pl

from loop.utils.splits import split_sequential
from loop.utils.splits import split_data_to_prep_output
from loop.metrics.continuous_metrics import continuous_metrics


def params():
    return {
        'learning_rate': [0.01, 0.02, 0.03],
        'max_depth': [2, 3, 4],
        'n_estimators': [300, 500, 700],
        'min_child_weight': [5, 10, 20],
        'subsample': [0.5, 0.6, 0.7],
        'colsample_bytree': [0.5, 0.6, 0.7],
        'gamma': [0.1, 0.5, 1.0],
        'reg_alpha': [0.1, 0.5, 1.0],
        'reg_lambda': [1.0, 5.0, 10.0],
        'objective': ['reg:squarederror'],
        'booster': ['gbtree'],
        'early_stopping_rounds': [50],
    }

def prep(data: pl.DataFrame):

    all_datetimes = data['datetime'].to_list()

    data = data.with_columns([
        # Original return features (must be created first)
        (((pl.col('close') - pl.col('close').shift(1)) / pl.col('close').shift(1)) * 100).alias('return_1'),
        (((pl.col('close') - pl.col('close').shift(5)) / pl.col('close').shift(5)) * 100).alias('return_5'),
        (((pl.col('close') - pl.col('close').shift(20)) / pl.col('close').shift(20)) * 100).alias('return_20'),
    ]).with_columns([
        # Volatility features (depends on return_1)
        pl.col('return_1').rolling_std(window_size=20).alias('volatility_20'),
        ((pl.col('high').shift(1) - pl.col('low').shift(1)) / pl.col('close').shift(1) * 100).alias('price_range'),
        
        # Volume features (ALL LAGGED)
        pl.col('volume').shift(1).rolling_mean(window_size=20).alias('volume_ma20'),
        (pl.col('volume').shift(1) / pl.col('volume').shift(1).rolling_mean(window_size=20)).alias('volume_ratio'),
        (pl.col('volume').shift(1) * pl.col('close').shift(1)).alias('dollar_volume'),
        
        # Microstructure features (ALL LAGGED)
        pl.col('maker_ratio').shift(1).rolling_mean(window_size=10).alias('maker_ratio_ma10'),
        (pl.col('no_of_trades').shift(1) / pl.col('no_of_trades').shift(1).rolling_mean(window_size=20)).alias('trade_intensity'),
        
        # Order flow imbalance (LAGGED)
        ((pl.col('close').shift(1) - pl.col('open').shift(1)) / pl.col('open').shift(1) * 100).alias('period_return'),
        (pl.col('maker_ratio').shift(1) - 0.5).alias('order_flow_imbalance'),
        
        # Technical indicators (LAGGED)
        ((pl.col('close').shift(1) - pl.col('low').shift(1)) / (pl.col('high').shift(1) - pl.col('low').shift(1))).alias('stochastic_k'),
        
        # Original unused features (kept for potential future use)
        (pl.col('close') / pl.col('close').rolling_mean(window_size=10)).alias('close_to_ma10'),
        (pl.col('close') / pl.col('close').rolling_mean(window_size=50)).alias('close_to_ma50'),
        
        # Target variables
        (((pl.col('close').shift(-1) - pl.col('close')) / pl.col('close')) * 100).alias('next_return'),
        pl.col('close').shift(-1).alias('next_close'),
    ]).filter(
        pl.all_horizontal([
            pl.col('next_return').is_not_null(),
            pl.col('return_20').is_not_null(),
            pl.col('volatility_20').is_not_null(),
            pl.col('volume_ratio').is_not_null(),
            pl.col('trade_intensity').is_not_null(),
            pl.col('close_to_ma50').is_not_null()
        ])
    )
    
    cols = ['datetime', 'return_1', 'return_5', 'return_20',
            'volatility_20', 'price_range', 'volume_ratio',
            'maker_ratio_ma10', 'trade_intensity', 'period_return',
            'order_flow_imbalance', 'stochastic_k', 
            'next_close', 'next_return',]

    split_data = split_sequential(data, (8, 1, 2))
    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)
    
    return data_dict


def model(data: dict, round_params: dict):
    
    model = xgb.XGBRegressor(
        learning_rate=round_params['learning_rate'],
        max_depth=round_params['max_depth'],
        n_estimators=round_params['n_estimators'],
        min_child_weight=round_params['min_child_weight'],
        subsample=round_params['subsample'],
        colsample_bytree=round_params['colsample_bytree'],
        gamma=round_params['gamma'],
        reg_alpha=round_params['reg_alpha'],
        reg_lambda=round_params['reg_lambda'],
        objective=round_params['objective'],
        booster=round_params['booster'],
        random_state=42,
        early_stopping_rounds=round_params['early_stopping_rounds'] if round_params['early_stopping_rounds'] is not None else None,
        eval_metric=['rmse']
    )
    
    model.fit(data['x_train'],
              data['y_train'],
              eval_set=[(data['x_val'], data['y_val'])],
              verbose=False)

    preds = model.predict(data['x_test'])

    round_results = continuous_metrics(data, preds)
    round_results['_preds'] = preds

    return round_results
