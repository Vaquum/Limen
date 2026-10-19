from copy import deepcopy

import numpy as np
import polars as pl

from limen.backtest.trade_contract import TradeInputs, TradeLedger, export_trade_contract
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.sfd.reference_architecture.meta_allocation import MetaAllocationModel
from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy
from limen.experiment import RuleBasedManifest
from limen.yaml import CompiledSFD


# region allocation_case
def allocation_case(config, output):
    manifest = CompiledSFD(config).manifest()
    raw = pl.read_parquet(config['sfd']['manifest']['data_source']['params']['file_path_or_url'])
    params = {key: values[0] for key, values in config['sfd']['params'].items()}
    prepared = manifest.prepare_data(raw, params)
    result = manifest.run_model(prepared, params)
    model = result['_model']
    if not isinstance(model, MetaAllocationModel):
        raise TypeError('The declared allocation must retain its fitted model')
    batch = model.context.batches['test']
    predicted = model.predict({'allocation_batch': batch})

    # Fit the original declared primary separately, without allocation or execution.
    plain_config = deepcopy(config)
    primary_declaration = plain_config['sfd']['manifest']
    del primary_declaration['allocation'], primary_declaration['backtest']
    plain = CompiledSFD(plain_config).manifest()
    plain_data = plain.prepare_data(raw, params)
    if isinstance(plain, RuleBasedManifest):
        primary = RuleBasedStrategy(**plain.resolve_model_kwargs(params)).train(plain_data)
    else:
        primary = plain.run_model(plain_data, params)['_model']
    if not isinstance(primary, ReferenceModel):
        raise TypeError('The original primary must retain its fitted model')
    original = primary.predict(dict(batch.primary_inputs))
    native = model.primary.predict(dict(batch.primary_inputs))

    inputs, ledger = prepared['_trade_inputs'], prepared['_trade_ledger']
    if not isinstance(inputs, TradeInputs) or not isinstance(ledger, TradeLedger):
        raise TypeError('Allocation requires the native bound execution inputs and ledger')
    output.mkdir(parents=True, exist_ok=True)
    model.training_rows.write_parquet(output / 'training.parquet')
    for field in ('states', 'intents', 'fills', 'episodes', 'funding'):
        getattr(ledger, field).write_parquet(output / (field + '.parquet'))
    report = {
        'primary': np.asarray(native['_preds'], dtype=float).tolist(),
        'plain_primary': np.asarray(original['_preds'], dtype=float).tolist(),
        'exposure': np.asarray(result['_preds'], dtype=float).tolist(),
        'repeated': np.asarray(predicted['_preds'], dtype=float).tolist(),
        'take_probability': np.asarray(predicted['take_probability'], dtype=float).tolist(),
        'signed_return': np.asarray(predicted['signed_notional_return'], dtype=float).tolist(),
        'stops': batch.stop_loss_bps, 'row_ids': batch.row_ids,
        'binding': model.learning_binding, 'batch_digest': batch.digest(),
        'training_window': {key: config['sfd']['manifest']['split_dates'][key] for key in ('train_start', 'train_end')},
        'training_rows': model.context.raw_train.height,
        'contract': export_trade_contract(model.context.policy, inputs),
        'completed': int(model.training_rows['available'].sum()),
        'unavailable': int((~model.training_rows['available']).sum()),
    }
    return report, model, plain
# endregion allocation_case


# region allocation_prefix
def prefix_probe(model: MetaAllocationModel, original, params):
    raw = model.context.raw_train
    indices = np.array_split(np.arange(raw.height), 3)[1]
    start, end = int(indices[0]), int(indices[-1]) + 1
    prefix = raw.head(start - 1)
    manifest = deepcopy(original)
    manifest.split_dates = None
    manifest.split_config = (int(prefix.height * .8), int(prefix.height * .2), 0)
    prepared = manifest.prepare_data(prefix, params)
    # The validation partition supplies evaluation rows; fitting remains prefix-only.
    fitting = dict(prepared, x_test=prepared['x_val'], y_test=prepared['y_val'])
    result = manifest.run_model(fitting, params)
    primary = result['_model']
    if not isinstance(primary, ReferenceModel):
        raise TypeError('Prefix fitting must retain the original primary')
    features, _ = manifest.sensor_input_prep(raw.head(end), prepared['_fitted_params'], params)
    features = features.filter(pl.col('datetime').is_between(raw['datetime'][start], raw['datetime'][end-1]))
    features = features.select(prepared['x_train'].columns).fill_nan(None).drop_nulls()
    return {'prefix_rows': prefix.height, 'predictions': np.asarray(primary.predict({'x_test': features})['_preds'], dtype=float).tolist()}
# endregion allocation_prefix
