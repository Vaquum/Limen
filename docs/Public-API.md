# Public API index

The declared package exports and their public class members are indexed here. The [Python project](examples/python-research/README.md) owns its caller functions and fitted state. The [Compiled project](examples/compiled-research/README.md) uses the [CLI](Command-Line-Interface.md) and [YAML contract](Experiment-Manifest.md); a native Python method in this appendix does not become a Compiled authoring step.

Use the subject references for methodology, defaults, units, state, outputs and refusals. This index closes names and source ownership. A signature or definition identifies the current package contract; it does not assert that every configuration was executed by the introductory experiment. Module re-exports and version metadata are distinguished from callable contracts.

## Declared entries

| Entry | Contract kind | Authority |
| --- | --- | --- |
| [`limen.Cohort`](Public-API.md#limencohort) | `signature` | `limen/cohort/cohort.py` |
| [`limen.Cohort.__call__`](Public-API.md#limencohort__call__) | `signature` | `limen/cohort/cohort.py` |
| [`limen.Cohort.__init__`](Cohort.md#construction) | `signature` | `limen/cohort/cohort.py` |
| [`limen.Cohort.from_python`](Cohort.md#python-cohorts) | `signature` | `limen/cohort/cohort.py` |
| [`limen.Cohort.predict`](Cohort.md#predictraw_klines---barprediction) | `signature` | `limen/cohort/cohort.py` |
| [`limen.Cohort.predict_all`](Cohort.md#predict_allraw_klines---listbarprediction) | `signature` | `limen/cohort/cohort.py` |
| [`limen.Cohort.set_members`](Cohort.md#binding-members) | `signature` | `limen/cohort/cohort.py` |
| [`limen.HistoricalData`](Public-API.md#limenhistoricaldata) | `signature` | `limen/data/historical_data.py` |
| [`limen.HistoricalData.__init__`](Public-API.md#limenhistoricaldata__init__) | `signature` | `limen/data/historical_data.py` |
| [`limen.HistoricalData.get_any_file`](Historical-Data-Reference.md#get_any_file) | `signature` | `limen/data/historical_data.py` |
| [`limen.HistoricalData.get_arrow_file`](Historical-Data-Reference.md#get_arrow_file) | `signature` | `limen/data/historical_data.py` |
| [`limen.HistoricalData.get_binance_file`](Historical-Data-Reference.md#get_binance_file) | `signature` | `limen/data/historical_data.py` |
| [`limen.HistoricalData.get_spot_dollar_klines`](Historical-Data-Reference.md#get_spot_dollar_klines) | `signature` | `limen/data/historical_data.py` |
| [`limen.HistoricalData.get_spot_klines`](Historical-Data-Reference.md#get_spot_klines) | `signature` | `limen/data/historical_data.py` |
| [`limen.Log`](Public-API.md#limenlog) | `signature` | `limen/log/log.py` |
| [`limen.Log.__init__`](Log.md#file-backed-log) | `signature` | `limen/log/log.py` |
| [`limen.Log.experiment_backtest_results`](Log.md#experiment_backtest_results) | `signature` | `limen/log/_experiment_backtest_results.py` |
| [`limen.Log.experiment_confusion_metrics`](Log.md#experiment_confusion_metrics) | `signature` | `limen/log/_experiment_confusion_metrics.py` |
| [`limen.Log.experiment_parameter_correlation`](Perturbation-Strategies.md#deriving-feature-importance-from-ablation) | `signature` | `limen/log/_experiment_parameter_correlation.py` |
| [`limen.Log.experiment_parameter_importance`](Log.md#experiment_parameter_importancemetric--parameters-random_state0-n_repeats10) | `signature` | `limen/log/_experiment_parameter_importance.py` |
| [`limen.Log.permutation_confusion_metrics`](Log.md#permutation_confusion_metricsx-round_id) | `signature` | `limen/log/_permutation_confusion_metrics.py` |
| [`limen.Log.permutation_prediction_performance`](Log.md#permutation_prediction_performanceround_id) | `signature` | `limen/log/log.py` |
| [`limen.Log.read_from_file`](Log.md#read_from_filefile_path) | `signature` | `limen/log/_read_from_file.py` |
| [`limen.MLManifest`](Public-API.md#limenmlmanifest) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.MLManifest.add_to_data_dict`](Experiment-Manifest.md#add_to_data_dictfunc) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.MLManifest.prepare_data`](Public-API.md#limenmlmanifestprepare_data) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.MLManifest.run_model`](Public-API.md#limenmlmanifestrun_model) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.MLManifest.sensor_input_prep`](Public-API.md#limenmlmanifestsensor_input_prep) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.MLManifest.set_feature_ablation`](Perturbation-Strategies.md#feature-ablation) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.MLManifest.set_feature_shims`](Experiment-Manifest.md#feature-shims) | `signature` | `limen/experiment/_column_shims.py` |
| [`limen.MLManifest.set_objective`](Public-API.md#limenmlmanifestset_objective) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.MLManifest.set_pca_compression`](Experiment-Manifest.md#set_pca_compressionenabled_paramauto_pca-n_components_parampca_k-scaler_param_name_scaler-component_prefixpc_) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.MLManifest.set_scaler`](Experiment-Manifest.md#set_scalertransform_class-param_name_scaler-extra_paramsnone) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.MLManifest.set_scaler_from_params`](Perturbation-Strategies.md#scaler-from-parameters) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.MLManifest.set_strict_mode`](Experiment-Manifest.md#set_strict_modestrict_mode) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.MLManifest.with_calibration`](Calibration.md#threshold-only-path) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest`](Public-API.md#limenmanifest) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.add_feature`](Perturbation-Strategies.md#conditional-toggle-include_if) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.add_indicator`](Perturbation-Strategies.md#feature-group-selection) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.compute_test_bars`](Public-API.md#limenmanifestcompute_test_bars) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.fetch_data`](Public-API.md#limenmanifestfetch_data) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.fetch_test_data`](Public-API.md#limenmanifestfetch_test_data) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.prepare_data`](Public-API.md#limenmanifestprepare_data) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.resolve_backtest_config`](Public-API.md#limenmanifestresolve_backtest_config) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.resolve_model_kwargs`](Public-API.md#limenmanifestresolve_model_kwargs) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.resolve_trade_policy`](Public-API.md#limenmanifestresolve_trade_policy) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.run_model`](Public-API.md#limenmanifestrun_model) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.set_acceptance`](Experiment-Manifest.md#acceptance-thresholds) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.set_backtest_config`](Backtest.md#economic-inputs) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.set_bar_formation`](Experiment-Manifest.md#set_bar_formationfunc-params) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.set_data_source`](Experiment-Manifest.md#set_data_sourcemethod-paramsnone) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.set_pre_split_data_selector`](Experiment-Manifest.md#set_pre_split_data_selectorfunc-params) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.set_required_bar_columns`](Experiment-Manifest.md#set_required_bar_columnscolumns) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.set_split_config`](Experiment-Manifest.md#set_split_configtrain-val-test) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.set_split_dates`](Experiment-Manifest.md#set_split_datestrain_start-train_end-val_start-val_end-test_start-test_end--val_predict_guardtrue-test_predict_guardtrue) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.set_split_walk_forward`](Experiment-Manifest.md#walk-forward-sweeps) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.set_test_data_source`](Experiment-Manifest.md#set_test_data_sourcemethod-paramsnone) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.with_allocation`](Public-API.md#limenmanifestwith_allocation) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.with_params_override`](Perturbation-Strategies.md#manifest-override) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.with_reference_architecture`](Experiment-Manifest.md#with_reference_architecturearchitecture_function) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Manifest.with_target_label`](Experiment-Manifest.md#target-configuration) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.ReconstructionError`](Public-API.md#limenreconstructionerror) | `signature` | `limen/inference/errors.py` |
| [`limen.RuleBasedManifest`](Public-API.md#limenrulebasedmanifest) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.RuleBasedManifest.prepare_data`](Public-API.md#limenrulebasedmanifestprepare_data) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.RuleBasedManifest.sensor_input_prep`](Public-API.md#limenrulebasedmanifestsensor_input_prep) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.RuleBasedManifest.set_objective`](Public-API.md#limenrulebasedmanifestset_objective) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.RuleBasedManifest.with_strategy`](Experiment-Manifest.md#with_strategyconditions-entry) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.Sensor`](Public-API.md#limensensor) | `signature` | `limen/inference/sensor.py` |
| [`limen.Sensor.__call__`](Trainer.md#sensor-inference-api) | `signature` | `limen/inference/sensor.py` |
| [`limen.Sensor.__init__`](Public-API.md#limensensor__init__) | `signature` | `limen/inference/sensor.py` |
| [`limen.Sensor.from_model`](Trainer.md#python-construction) | `signature` | `limen/inference/sensor.py` |
| [`limen.Sensor.from_python`](Trainer.md#arbitrary-python-inference) | `signature` | `limen/inference/sensor.py` |
| [`limen.Sensor.predict`](Trainer.md#sensor-inference-api) | `signature` | `limen/inference/sensor.py` |
| [`limen.Sensor.predict_all`](Public-API.md#limensensorpredict_all) | `signature` | `limen/inference/sensor.py` |
| [`limen.Sensor.prediction_mode`](Public-API.md#limensensorprediction_mode) | `signature` | `limen/inference/sensor.py` |
| [`limen.Sensor.round_params`](Public-API.md#limensensorround_params) | `signature` | `limen/inference/sensor.py` |
| [`limen.Sensor.trade_contract`](Public-API.md#limensensortrade_contract) | `signature` | `limen/inference/sensor.py` |
| [`limen.Trainer`](Public-API.md#limentrainer) | `signature` | `limen/inference/trainer.py` |
| [`limen.Trainer.__init__`](Trainer.md#trainerexperiment_dir-datanone) | `signature` | `limen/inference/trainer.py` |
| [`limen.Trainer.train`](Trainer.md#trainpermutation_ids) | `signature` | `limen/inference/trainer.py` |
| [`limen.UniversalExperimentLoop`](Public-API.md#limenuniversalexperimentloop) | `signature` | `limen/experiment/experiment_core.py` |
| [`limen.UniversalExperimentLoop.__init__`](Universal-Experiment-Loop.md#constructor-contract) | `signature` | `limen/experiment/experiment_core.py` |
| [`limen.UniversalExperimentLoop.fold_results`](Public-API.md#limenuniversalexperimentloopfold_results) | `signature` | `limen/experiment/experiment_core.py` |
| [`limen.UniversalExperimentLoop.interrupted`](Public-API.md#limenuniversalexperimentloopinterrupted) | `signature` | `limen/experiment/experiment_core.py` |
| [`limen.UniversalExperimentLoop.run`](Universal-Experiment-Loop.md#run-contract) | `signature` | `limen/experiment/experiment_core.py` |
| `limen.__version__` | `metadata` | `limen/__init__.py` |
| [`limen.backtest.ExecutionResult`](Public-API.md#limenbacktestexecutionresult) | `signature` | `limen/backtest/long_flat_strategy.py` |
| [`limen.backtest.ProductSpec`](Backtest.md#signed-exposure-elapsed-exits-and-funding) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.TradeInputs`](Backtest.md#signed-exposure-elapsed-exits-and-funding) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.TradePolicy`](Backtest.md#signed-exposure-elapsed-exits-and-funding) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.backtest_snapshot`](Backtest.md#strategy-boundary) | `signature` | `limen/backtest/backtest_snapshot.py` |
| [`limen.backtest.backtest_snapshot.BACKTEST_SNAPSHOT_COLUMNS`](Public-API.md#limenbacktestbacktest_snapshotbacktest_snapshot_columns) | `definition` | `limen/backtest/backtest_snapshot.py` |
| [`limen.backtest.backtest_snapshot.backtest_snapshot`](Backtest.md#strategy-boundary) | `signature` | `limen/backtest/backtest_snapshot.py` |
| [`limen.backtest.execution_events.OBSERVATION_COLUMNS`](Public-API.md#limenbacktestexecution_eventsobservation_columns) | `definition` | `limen/backtest/execution_events.py` |
| [`limen.backtest.execution_events.SIGNAL_COLUMNS`](Public-API.md#limenbacktestexecution_eventssignal_columns) | `definition` | `limen/backtest/execution_events.py` |
| [`limen.backtest.execution_events.execution_events`](Public-API.md#limenbacktestexecution_eventsexecution_events) | `signature` | `limen/backtest/execution_events.py` |
| [`limen.backtest.execution_events.validate_observations`](Public-API.md#limenbacktestexecution_eventsvalidate_observations) | `signature` | `limen/backtest/execution_events.py` |
| [`limen.backtest.execution_events.with_predictions`](Public-API.md#limenbacktestexecution_eventswith_predictions) | `signature` | `limen/backtest/execution_events.py` |
| [`limen.backtest.funding_adapter.ADAPTER_VERSION`](Public-API.md#limenbacktestfunding_adapteradapter_version) | `definition` | `limen/backtest/funding_adapter.py` |
| [`limen.backtest.funding_adapter.FUNDING_KEYS`](Public-API.md#limenbacktestfunding_adapterfunding_keys) | `definition` | `limen/backtest/funding_adapter.py` |
| [`limen.backtest.funding_adapter.FundingAdapter`](Public-API.md#limenbacktestfunding_adapterfundingadapter) | `signature` | `limen/backtest/funding_adapter.py` |
| [`limen.backtest.funding_adapter.FundingAdapter.__call__`](Public-API.md#limenbacktestfunding_adapterfundingadapter__call__) | `signature` | `limen/backtest/funding_adapter.py` |
| [`limen.backtest.funding_adapter.FundingCashflow`](Public-API.md#limenbacktestfunding_adapterfundingcashflow) | `signature` | `limen/backtest/funding_adapter.py` |
| [`limen.backtest.funding_adapter.FundingContext`](Public-API.md#limenbacktestfunding_adapterfundingcontext) | `signature` | `limen/backtest/funding_adapter.py` |
| [`limen.backtest.funding_adapter.funding_adapter`](Public-API.md#limenbacktestfunding_adapterfunding_adapter) | `signature` | `limen/backtest/funding_adapter.py` |
| [`limen.backtest.funding_adapter.prepare_funding`](Public-API.md#limenbacktestfunding_adapterprepare_funding) | `signature` | `limen/backtest/funding_adapter.py` |
| [`limen.backtest.funding_adapter.resolve_adapter`](Public-API.md#limenbacktestfunding_adapterresolve_adapter) | `signature` | `limen/backtest/funding_adapter.py` |
| [`limen.backtest.funding_adapter.scheduled_times`](Public-API.md#limenbacktestfunding_adapterscheduled_times) | `signature` | `limen/backtest/funding_adapter.py` |
| [`limen.backtest.funding_presets.BASELINE_RATE_8H`](Public-API.md#limenbacktestfunding_presetsbaseline_rate_8h) | `definition` | `limen/backtest/funding_presets.py` |
| [`limen.backtest.funding_presets.FundingPreset`](Public-API.md#limenbacktestfunding_presetsfundingpreset) | `signature` | `limen/backtest/funding_presets.py` |
| [`limen.backtest.funding_presets.PRESETS`](Public-API.md#limenbacktestfunding_presetspresets) | `definition` | `limen/backtest/funding_presets.py` |
| [`limen.backtest.funding_presets.PRESET_VERSION`](Public-API.md#limenbacktestfunding_presetspreset_version) | `definition` | `limen/backtest/funding_presets.py` |
| [`limen.backtest.long_flat_strategy`](Backtest.md#usage) | `signature` | `limen/backtest/long_flat_strategy.py` |
| [`limen.backtest.long_flat_strategy.ExecutionResult`](Public-API.md#limenbacktestexecutionresult) | `signature` | `limen/backtest/long_flat_strategy.py` |
| [`limen.backtest.long_flat_strategy.long_flat_strategy`](Backtest.md#usage) | `signature` | `limen/backtest/long_flat_strategy.py` |
| [`limen.backtest.observed_costs.CalibrationReport`](Public-API.md#limenbacktestobserved_costscalibrationreport) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.ComparisonReport`](Public-API.md#limenbacktestobserved_costscomparisonreport) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostCalibration`](Public-API.md#limenbacktestobserved_costscostcalibration) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostCalibration.compare`](Public-API.md#limenbacktestobserved_costscostcalibrationcompare) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostCalibration.digest`](Public-API.md#limenbacktestobserved_costscostcalibrationdigest) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostCalibration.from_history`](Backtest.md#recorded-execution-price-costs) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostCalibration.read`](Public-API.md#limenbacktestobserved_costscostcalibrationread) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostCalibration.report`](Public-API.md#limenbacktestobserved_costscostcalibrationreport) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostCalibration.write`](Public-API.md#limenbacktestobserved_costscostcalibrationwrite) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostCandidate`](Public-API.md#limenbacktestobserved_costscostcandidate) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostClock`](Public-API.md#limenbacktestobserved_costscostclock) | `definition` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostComparison`](Public-API.md#limenbacktestobserved_costscostcomparison) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostComparison.digest`](Public-API.md#limenbacktestobserved_costscostcomparisondigest) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostComparison.ledger`](Public-API.md#limenbacktestobserved_costscostcomparisonledger) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostComparison.report`](Public-API.md#limenbacktestobserved_costscostcomparisonreport) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostComparison.write`](Public-API.md#limenbacktestobserved_costscostcomparisonwrite) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostConditions`](Public-API.md#limenbacktestobserved_costscostconditions) | `signature` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.observed_costs.CostDisposition`](Public-API.md#limenbacktestobserved_costscostdisposition) | `definition` | `limen/backtest/observed_costs.py` |
| [`limen.backtest.trade_contract.BPS`](Public-API.md#limenbacktesttrade_contractbps) | `definition` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.ExecutionEvent`](Public-API.md#limenbacktesttrade_contractexecutionevent) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.FundingPolicy`](Backtest.md#signed-exposure-elapsed-exits-and-funding) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.JsonScalar`](Public-API.md#limenbacktesttrade_contractjsonscalar) | `definition` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.JsonValue`](Public-API.md#limenbacktesttrade_contractjsonvalue) | `definition` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.NANOSECONDS`](Public-API.md#limenbacktesttrade_contractnanoseconds) | `definition` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.PredictionMode`](Public-API.md#limenbacktesttrade_contractpredictionmode) | `definition` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.ProductSpec`](Backtest.md#signed-exposure-elapsed-exits-and-funding) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.RULE_VERSION`](Public-API.md#limenbacktesttrade_contractrule_version) | `definition` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.SourceBinding`](Public-API.md#limenbacktesttrade_contractsourcebinding) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.TradeInputs`](Backtest.md#signed-exposure-elapsed-exits-and-funding) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.TradeLedger`](Public-API.md#limenbacktesttrade_contracttradeledger) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.TradePolicy`](Backtest.md#signed-exposure-elapsed-exits-and-funding) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.contract_digest`](Public-API.md#limenbacktesttrade_contractcontract_digest) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.export_trade_contract`](Public-API.md#limenbacktesttrade_contractexport_trade_contract) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.finite_number`](Public-API.md#limenbacktesttrade_contractfinite_number) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.json_value`](Public-API.md#limenbacktesttrade_contractjson_value) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_contract.source_binding`](Public-API.md#limenbacktesttrade_contractsource_binding) | `signature` | `limen/backtest/trade_contract.py` |
| [`limen.backtest.trade_execution`](Public-API.md#limenbacktesttrade_execution) | `signature` | `limen/backtest/trade_execution.py` |
| [`limen.backtest.trade_execution.trade_execution`](Public-API.md#limenbacktesttrade_execution) | `signature` | `limen/backtest/trade_execution.py` |
| [`limen.calibration.CalibratorProtocol`](Public-API.md#limencalibrationcalibratorprotocol) | `signature` | `limen/calibration/pipeline.py` |
| [`limen.calibration.CalibratorProtocol.__call__`](Calibration.md#custom-calibrators-and-optimizers) | `signature` | `limen/calibration/pipeline.py` |
| [`limen.calibration.ThresholdOptimizerProtocol`](Public-API.md#limencalibrationthresholdoptimizerprotocol) | `signature` | `limen/calibration/pipeline.py` |
| [`limen.calibration.ThresholdOptimizerProtocol.__call__`](Calibration.md#custom-calibrators-and-optimizers) | `signature` | `limen/calibration/pipeline.py` |
| [`limen.calibration.apply_calibrated_predict`](Public-API.md#limencalibrationapply_calibrated_predict) | `signature` | `limen/calibration/pipeline.py` |
| [`limen.calibration.fit_calibrator`](Public-API.md#limencalibrationfit_calibrator) | `signature` | `limen/calibration/pipeline.py` |
| [`limen.calibration.grid_threshold_optimizer`](Calibration.md#grid_threshold_optimizer-parameters) | `signature` | `limen/calibration/threshold.py` |
| [`limen.calibration.sklearn_probability_calibrator`](Calibration.md#custom-calibrators-and-optimizers) | `signature` | `limen/calibration/probability.py` |
| [`limen.cli.cli`](Public-API.md#limenclicli) | `signature` | `limen/cli/main.py` |
| [`limen.cohort.BUILTIN_SELECTORS`](Cohort.md#single-file-cohort-selectors) | `definition` | `limen/cohort/sfc/__init__.py` |
| [`limen.cohort.Cohort`](Public-API.md#limencohort) | `signature` | `limen/cohort/cohort.py` |
| [`limen.cohort.Cohort.__call__`](Public-API.md#limencohort__call__) | `signature` | `limen/cohort/cohort.py` |
| [`limen.cohort.Cohort.__init__`](Cohort.md#construction) | `signature` | `limen/cohort/cohort.py` |
| [`limen.cohort.Cohort.from_python`](Cohort.md#python-cohorts) | `signature` | `limen/cohort/cohort.py` |
| [`limen.cohort.Cohort.predict`](Cohort.md#predictraw_klines---barprediction) | `signature` | `limen/cohort/cohort.py` |
| [`limen.cohort.Cohort.predict_all`](Cohort.md#predict_allraw_klines---listbarprediction) | `signature` | `limen/cohort/cohort.py` |
| [`limen.cohort.Cohort.set_members`](Cohort.md#binding-members) | `signature` | `limen/cohort/cohort.py` |
| [`limen.cohort.sfc.BUILTIN_SELECTORS`](Cohort.md#single-file-cohort-selectors) | `definition` | `limen/cohort/sfc/__init__.py` |
| [`limen.cohort.sfc.Selector`](Public-API.md#limencohortsfcselector) | `definition` | `limen/cohort/sfc/__init__.py` |
| [`limen.cohort.sfc.SelectorContext`](Public-API.md#limencohortsfcselectorcontext) | `definition` | `limen/cohort/sfc/__init__.py` |
| [`limen.cohort.sfc.select_all`](Cohort.md#selector-contract) | `signature` | `limen/cohort/sfc/all.py` |
| [`limen.cohort.sfc.select_backtest_pareto`](Cohort.md#single-file-cohort-selectors) | `signature` | `limen/cohort/sfc/backtest_pareto.py` |
| [`limen.cohort.sfc.select_diverse_metrics`](Cohort.md#single-file-cohort-selectors) | `signature` | `limen/cohort/sfc/diverse_metrics.py` |
| [`limen.cohort.sfc.select_top_n`](Cohort.md#single-file-cohort-selectors) | `signature` | `limen/cohort/sfc/top_n.py` |
| [`limen.data.HistoricalData`](Public-API.md#limenhistoricaldata) | `signature` | `limen/data/historical_data.py` |
| [`limen.data.HistoricalData.__init__`](Public-API.md#limenhistoricaldata__init__) | `signature` | `limen/data/historical_data.py` |
| [`limen.data.HistoricalData.get_any_file`](Historical-Data-Reference.md#get_any_file) | `signature` | `limen/data/historical_data.py` |
| [`limen.data.HistoricalData.get_arrow_file`](Historical-Data-Reference.md#get_arrow_file) | `signature` | `limen/data/historical_data.py` |
| [`limen.data.HistoricalData.get_binance_file`](Historical-Data-Reference.md#get_binance_file) | `signature` | `limen/data/historical_data.py` |
| [`limen.data.HistoricalData.get_spot_dollar_klines`](Historical-Data-Reference.md#get_spot_dollar_klines) | `signature` | `limen/data/historical_data.py` |
| [`limen.data.HistoricalData.get_spot_klines`](Historical-Data-Reference.md#get_spot_klines) | `signature` | `limen/data/historical_data.py` |
| [`limen.data.bars.liquidity_bars`](Public-API.md#limendatabarsliquidity_bars) | `signature` | `limen/data/bars/standard_bars.py` |
| [`limen.data.bars.trade_bars`](Public-API.md#limendatabarstrade_bars) | `signature` | `limen/data/bars/standard_bars.py` |
| [`limen.data.bars.volume_bars`](Public-API.md#limendatabarsvolume_bars) | `signature` | `limen/data/bars/standard_bars.py` |
| [`limen.data.compute_data_bars`](Public-API.md#limendatacompute_data_bars) | `signature` | `limen/data/utils/compute_data_bars.py` |
| [`limen.data.utils.compute_data_bars`](Public-API.md#limendatacompute_data_bars) | `signature` | `limen/data/utils/compute_data_bars.py` |
| [`limen.data.utils.random_slice`](Public-API.md#limendatautilsrandom_slice) | `signature` | `limen/data/utils/random_slice.py` |
| [`limen.data.utils.split_by_dates`](Utilities.md#split_by_dates) | `signature` | `limen/data/utils/splits.py` |
| [`limen.data.utils.split_data_to_prep_output`](Public-API.md#limendatautilssplit_data_to_prep_output) | `signature` | `limen/data/utils/splits.py` |
| [`limen.data.utils.split_data_to_rule_based_prep_output`](Public-API.md#limendatautilssplit_data_to_rule_based_prep_output) | `signature` | `limen/data/utils/splits.py` |
| [`limen.data.utils.split_random`](Public-API.md#limendatautilssplit_random) | `signature` | `limen/data/utils/splits.py` |
| [`limen.data.utils.split_sequential`](Public-API.md#limendatautilssplit_sequential) | `signature` | `limen/data/utils/splits.py` |
| [`limen.data.utils.split_walk_forward`](Public-API.md#limendatautilssplit_walk_forward) | `signature` | `limen/data/utils/split_walk_forward.py` |
| [`limen.data.utils.split_walk_forward.split_walk_forward`](Public-API.md#limendatautilssplit_walk_forward) | `signature` | `limen/data/utils/split_walk_forward.py` |
| [`limen.experiment.CalibrationConfig`](Public-API.md#limenexperimentcalibrationconfig) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.CalibrationConfig.resolve`](Public-API.md#limenexperimentcalibrationconfigresolve) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.GridStrategy`](Public-API.md#limenexperimentgridstrategy) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.GridStrategy.__init__`](Public-API.md#limenexperimentgridstrategy__init__) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.GridStrategy.get_state`](Public-API.md#limenexperimentgridstrategyget_state) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.GridStrategy.is_finite`](Public-API.md#limenexperimentgridstrategyis_finite) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.GridStrategy.on_domain_changed`](Public-API.md#limenexperimentgridstrategyon_domain_changed) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.GridStrategy.set_state`](Public-API.md#limenexperimentgridstrategyset_state) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.MLManifest`](Public-API.md#limenmlmanifest) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.MLManifest.add_to_data_dict`](Experiment-Manifest.md#add_to_data_dictfunc) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.MLManifest.prepare_data`](Public-API.md#limenmlmanifestprepare_data) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.MLManifest.run_model`](Public-API.md#limenmlmanifestrun_model) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.MLManifest.sensor_input_prep`](Public-API.md#limenmlmanifestsensor_input_prep) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.MLManifest.set_feature_ablation`](Perturbation-Strategies.md#feature-ablation) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.MLManifest.set_feature_shims`](Experiment-Manifest.md#feature-shims) | `signature` | `limen/experiment/_column_shims.py` |
| [`limen.experiment.MLManifest.set_objective`](Public-API.md#limenmlmanifestset_objective) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.MLManifest.set_pca_compression`](Experiment-Manifest.md#set_pca_compressionenabled_paramauto_pca-n_components_parampca_k-scaler_param_name_scaler-component_prefixpc_) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.MLManifest.set_scaler`](Experiment-Manifest.md#set_scalertransform_class-param_name_scaler-extra_paramsnone) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.MLManifest.set_scaler_from_params`](Perturbation-Strategies.md#scaler-from-parameters) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.MLManifest.set_strict_mode`](Experiment-Manifest.md#set_strict_modestrict_mode) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.MLManifest.with_calibration`](Calibration.md#threshold-only-path) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest`](Public-API.md#limenmanifest) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.add_feature`](Perturbation-Strategies.md#conditional-toggle-include_if) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.add_indicator`](Perturbation-Strategies.md#feature-group-selection) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.compute_test_bars`](Public-API.md#limenmanifestcompute_test_bars) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.fetch_data`](Public-API.md#limenmanifestfetch_data) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.fetch_test_data`](Public-API.md#limenmanifestfetch_test_data) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.prepare_data`](Public-API.md#limenmanifestprepare_data) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.resolve_backtest_config`](Public-API.md#limenmanifestresolve_backtest_config) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.resolve_model_kwargs`](Public-API.md#limenmanifestresolve_model_kwargs) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.resolve_trade_policy`](Public-API.md#limenmanifestresolve_trade_policy) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.run_model`](Public-API.md#limenmanifestrun_model) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.set_acceptance`](Experiment-Manifest.md#acceptance-thresholds) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.set_backtest_config`](Backtest.md#economic-inputs) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.set_bar_formation`](Experiment-Manifest.md#set_bar_formationfunc-params) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.set_data_source`](Experiment-Manifest.md#set_data_sourcemethod-paramsnone) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.set_pre_split_data_selector`](Experiment-Manifest.md#set_pre_split_data_selectorfunc-params) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.set_required_bar_columns`](Experiment-Manifest.md#set_required_bar_columnscolumns) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.set_split_config`](Experiment-Manifest.md#set_split_configtrain-val-test) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.set_split_dates`](Experiment-Manifest.md#set_split_datestrain_start-train_end-val_start-val_end-test_start-test_end--val_predict_guardtrue-test_predict_guardtrue) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.set_split_walk_forward`](Experiment-Manifest.md#walk-forward-sweeps) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.set_test_data_source`](Experiment-Manifest.md#set_test_data_sourcemethod-paramsnone) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.with_allocation`](Public-API.md#limenmanifestwith_allocation) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.with_params_override`](Perturbation-Strategies.md#manifest-override) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.with_reference_architecture`](Experiment-Manifest.md#with_reference_architecturearchitecture_function) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.Manifest.with_target_label`](Experiment-Manifest.md#target-configuration) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.RandomStrategy`](Public-API.md#limenexperimentrandomstrategy) | `signature` | `limen/experiment/param_search/random_strategy.py` |
| [`limen.experiment.RandomStrategy.__init__`](Public-API.md#limenexperimentrandomstrategy__init__) | `signature` | `limen/experiment/param_search/random_strategy.py` |
| [`limen.experiment.RandomStrategy.get_state`](Public-API.md#limenexperimentrandomstrategyget_state) | `signature` | `limen/experiment/param_search/random_strategy.py` |
| [`limen.experiment.RandomStrategy.on_domain_changed`](Public-API.md#limenexperimentrandomstrategyon_domain_changed) | `signature` | `limen/experiment/param_search/random_strategy.py` |
| [`limen.experiment.RandomStrategy.set_state`](Public-API.md#limenexperimentrandomstrategyset_state) | `signature` | `limen/experiment/param_search/random_strategy.py` |
| [`limen.experiment.ReuseContract`](Universal-Experiment-Loop.md#reuse-declarations) | `signature` | `limen/experiment/_factorized_contract.py` |
| [`limen.experiment.RuleBasedManifest`](Public-API.md#limenrulebasedmanifest) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.RuleBasedManifest.prepare_data`](Public-API.md#limenrulebasedmanifestprepare_data) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.RuleBasedManifest.sensor_input_prep`](Public-API.md#limenrulebasedmanifestsensor_input_prep) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.RuleBasedManifest.set_objective`](Public-API.md#limenrulebasedmanifestset_objective) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.RuleBasedManifest.with_strategy`](Experiment-Manifest.md#with_strategyconditions-entry) | `signature` | `limen/experiment/manifest_core.py` |
| [`limen.experiment.STRATEGY_REGISTRY`](Public-API.md#limenexperimentstrategy_registry) | `definition` | `limen/experiment/param_search/registry.py` |
| [`limen.experiment.UniversalExperimentLoop`](Public-API.md#limenuniversalexperimentloop) | `signature` | `limen/experiment/experiment_core.py` |
| [`limen.experiment.UniversalExperimentLoop.__init__`](Universal-Experiment-Loop.md#constructor-contract) | `signature` | `limen/experiment/experiment_core.py` |
| [`limen.experiment.UniversalExperimentLoop.fold_results`](Public-API.md#limenuniversalexperimentloopfold_results) | `signature` | `limen/experiment/experiment_core.py` |
| [`limen.experiment.UniversalExperimentLoop.interrupted`](Public-API.md#limenuniversalexperimentloopinterrupted) | `signature` | `limen/experiment/experiment_core.py` |
| [`limen.experiment.UniversalExperimentLoop.run`](Universal-Experiment-Loop.md#run-contract) | `signature` | `limen/experiment/experiment_core.py` |
| [`limen.experiment.declare_reuse`](Universal-Experiment-Loop.md#reuse-declarations) | `signature` | `limen/experiment/_factorized_contract.py` |
| [`limen.experiment.acceptance_report.acceptance_report`](Public-API.md#limenexperimentacceptance_reportacceptance_report) | `signature` | `limen/experiment/acceptance_report.py` |
| [`limen.experiment.acceptance_report.read_acceptance`](Public-API.md#limenexperimentacceptance_reportread_acceptance) | `signature` | `limen/experiment/_acceptance_settings.py` |
| [`limen.experiment.convergence_report.convergence_header`](Public-API.md#limenexperimentconvergence_reportconvergence_header) | `signature` | `limen/experiment/convergence_report.py` |
| [`limen.experiment.convergence_report.convergence_report`](Public-API.md#limenexperimentconvergence_reportconvergence_report) | `signature` | `limen/experiment/convergence_report.py` |
| [`limen.experiment.convergence_report.convergence_warning`](Public-API.md#limenexperimentconvergence_reportconvergence_warning) | `signature` | `limen/experiment/convergence_report.py` |
| [`limen.experiment.convergence_report.convergence_warnings`](Public-API.md#limenexperimentconvergence_reportconvergence_warnings) | `signature` | `limen/experiment/convergence_report.py` |
| [`limen.experiment.family_acceptance_report.family_acceptance_report`](Benchmark.md#comparing-a-recorded-experiment-family) | `signature` | `limen/experiment/family_acceptance_report.py` |
| [`limen.experiment.holdout.FrozenHoldoutSelection`](Public-API.md#limenexperimentholdoutfrozenholdoutselection) | `signature` | `limen/experiment/_holdout_freeze.py` |
| [`limen.experiment.holdout.FrozenHoldoutSelection.__init__`](Public-API.md#limenexperimentholdoutfrozenholdoutselection__init__) | `signature` | `limen/experiment/_holdout_freeze.py` |
| [`limen.experiment.holdout.final_holdout_report`](Public-API.md#limenexperimentholdoutfinal_holdout_report) | `signature` | `limen/experiment/_holdout_assessment.py` |
| [`limen.experiment.holdout.freeze_holdout_selection`](Public-API.md#limenexperimentholdoutfreeze_holdout_selection) | `signature` | `limen/experiment/_holdout_freeze.py` |
| [`limen.experiment.holdout.read_holdout_report`](Public-API.md#limenexperimentholdoutread_holdout_report) | `signature` | `limen/experiment/_holdout_report.py` |
| [`limen.experiment.holdout.release_holdout`](Public-API.md#limenexperimentholdoutrelease_holdout) | `signature` | `limen/experiment/_holdout_history.py` |
| [`limen.experiment.holdout.reserve_holdout`](Public-API.md#limenexperimentholdoutreserve_holdout) | `signature` | `limen/experiment/_holdout_history.py` |
| [`limen.experiment.holdout.withdraw_holdout`](Public-API.md#limenexperimentholdoutwithdraw_holdout) | `signature` | `limen/experiment/_holdout_history.py` |
| [`limen.experiment.param_search.GridStrategy`](Public-API.md#limenexperimentgridstrategy) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.GridStrategy.__init__`](Public-API.md#limenexperimentgridstrategy__init__) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.GridStrategy.get_state`](Public-API.md#limenexperimentgridstrategyget_state) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.GridStrategy.is_finite`](Public-API.md#limenexperimentgridstrategyis_finite) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.GridStrategy.on_domain_changed`](Public-API.md#limenexperimentgridstrategyon_domain_changed) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.GridStrategy.set_state`](Public-API.md#limenexperimentgridstrategyset_state) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.RandomStrategy`](Public-API.md#limenexperimentrandomstrategy) | `signature` | `limen/experiment/param_search/random_strategy.py` |
| [`limen.experiment.param_search.RandomStrategy.__init__`](Public-API.md#limenexperimentrandomstrategy__init__) | `signature` | `limen/experiment/param_search/random_strategy.py` |
| [`limen.experiment.param_search.RandomStrategy.get_state`](Public-API.md#limenexperimentrandomstrategyget_state) | `signature` | `limen/experiment/param_search/random_strategy.py` |
| [`limen.experiment.param_search.RandomStrategy.on_domain_changed`](Public-API.md#limenexperimentrandomstrategyon_domain_changed) | `signature` | `limen/experiment/param_search/random_strategy.py` |
| [`limen.experiment.param_search.RandomStrategy.set_state`](Public-API.md#limenexperimentrandomstrategyset_state) | `signature` | `limen/experiment/param_search/random_strategy.py` |
| [`limen.experiment.param_search.STRATEGY_REGISTRY`](Public-API.md#limenexperimentstrategy_registry) | `definition` | `limen/experiment/param_search/registry.py` |
| [`limen.experiment.param_search.SearchStrategy`](Public-API.md#limenexperimentparam_searchsearchstrategy) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SearchStrategy.__init__`](Public-API.md#limenexperimentparam_searchsearchstrategy__init__) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SearchStrategy.compute_param_hash`](Public-API.md#limenexperimentparam_searchsearchstrategycompute_param_hash) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SearchStrategy.domain`](Public-API.md#limenexperimentparam_searchsearchstrategydomain) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SearchStrategy.generated_count`](Public-API.md#limenexperimentparam_searchsearchstrategygenerated_count) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SearchStrategy.get_state`](Public-API.md#limenexperimentparam_searchsearchstrategyget_state) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SearchStrategy.is_finite`](Public-API.md#limenexperimentparam_searchsearchstrategyis_finite) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SearchStrategy.mark_seen`](Public-API.md#limenexperimentparam_searchsearchstrategymark_seen) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SearchStrategy.on_domain_changed`](Public-API.md#limenexperimentparam_searchsearchstrategyon_domain_changed) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SearchStrategy.rebuild_seen_from_log`](Public-API.md#limenexperimentparam_searchsearchstrategyrebuild_seen_from_log) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SearchStrategy.set_state`](Public-API.md#limenexperimentparam_searchsearchstrategyset_state) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SearchStrategy.update_from_feedback`](Public-API.md#limenexperimentparam_searchsearchstrategyupdate_from_feedback) | `signature` | `limen/experiment/param_search/search_strategy.py` |
| [`limen.experiment.param_search.SobolStrategy`](Public-API.md#limenexperimentparam_searchsobolstrategy) | `signature` | `limen/experiment/param_search/sobol_strategy.py` |
| [`limen.experiment.param_search.SobolStrategy.__init__`](Public-API.md#limenexperimentparam_searchsobolstrategy__init__) | `signature` | `limen/experiment/param_search/sobol_strategy.py` |
| [`limen.experiment.param_search.SobolStrategy.get_state`](Public-API.md#limenexperimentparam_searchsobolstrategyget_state) | `signature` | `limen/experiment/param_search/sobol_strategy.py` |
| [`limen.experiment.param_search.SobolStrategy.on_domain_changed`](Public-API.md#limenexperimentparam_searchsobolstrategyon_domain_changed) | `signature` | `limen/experiment/param_search/sobol_strategy.py` |
| [`limen.experiment.param_search.SobolStrategy.set_state`](Public-API.md#limenexperimentparam_searchsobolstrategyset_state) | `signature` | `limen/experiment/param_search/sobol_strategy.py` |
| [`limen.experiment.param_search.grid_strategy.GridStrategy`](Public-API.md#limenexperimentgridstrategy) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.grid_strategy.GridStrategy.__init__`](Public-API.md#limenexperimentgridstrategy__init__) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.grid_strategy.GridStrategy.get_state`](Public-API.md#limenexperimentgridstrategyget_state) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.grid_strategy.GridStrategy.is_finite`](Public-API.md#limenexperimentgridstrategyis_finite) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.grid_strategy.GridStrategy.on_domain_changed`](Public-API.md#limenexperimentgridstrategyon_domain_changed) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.grid_strategy.GridStrategy.set_state`](Public-API.md#limenexperimentgridstrategyset_state) | `signature` | `limen/experiment/param_search/grid_strategy.py` |
| [`limen.experiment.param_search.sobol_strategy.SobolStrategy`](Public-API.md#limenexperimentparam_searchsobolstrategy) | `signature` | `limen/experiment/param_search/sobol_strategy.py` |
| [`limen.experiment.param_search.sobol_strategy.SobolStrategy.__init__`](Public-API.md#limenexperimentparam_searchsobolstrategy__init__) | `signature` | `limen/experiment/param_search/sobol_strategy.py` |
| [`limen.experiment.param_search.sobol_strategy.SobolStrategy.get_state`](Public-API.md#limenexperimentparam_searchsobolstrategyget_state) | `signature` | `limen/experiment/param_search/sobol_strategy.py` |
| [`limen.experiment.param_search.sobol_strategy.SobolStrategy.on_domain_changed`](Public-API.md#limenexperimentparam_searchsobolstrategyon_domain_changed) | `signature` | `limen/experiment/param_search/sobol_strategy.py` |
| [`limen.experiment.param_search.sobol_strategy.SobolStrategy.set_state`](Public-API.md#limenexperimentparam_searchsobolstrategyset_state) | `signature` | `limen/experiment/param_search/sobol_strategy.py` |
| [`limen.experiment.reducer.ACTION_SUGGEST`](Public-API.md#limenexperimentreduceraction_suggest) | `definition` | `limen/experiment/reducer/pruning_strategy.py` |
| [`limen.experiment.reducer.BudgetReducer`](Public-API.md#limenexperimentreducerbudgetreducer) | `signature` | `limen/experiment/reducer/budget_reducer.py` |
| [`limen.experiment.reducer.BudgetReducer.__init__`](Developer/Pruning-Strategies.md#budgetreducer) | `signature` | `limen/experiment/reducer/budget_reducer.py` |
| [`limen.experiment.reducer.BudgetReducer.analyze_and_intervene`](Public-API.md#limenexperimentreducerbudgetreduceranalyze_and_intervene) | `signature` | `limen/experiment/reducer/budget_reducer.py` |
| [`limen.experiment.reducer.BudgetReducer.get_state`](Public-API.md#limenexperimentreducerbudgetreducerget_state) | `signature` | `limen/experiment/reducer/budget_reducer.py` |
| [`limen.experiment.reducer.BudgetReducer.set_state`](Public-API.md#limenexperimentreducerbudgetreducerset_state) | `signature` | `limen/experiment/reducer/budget_reducer.py` |
| [`limen.experiment.reducer.CorrelationReducer`](Public-API.md#limenexperimentreducercorrelationreducer) | `signature` | `limen/experiment/reducer/correlation_reducer.py` |
| [`limen.experiment.reducer.CorrelationReducer.__init__`](Developer/Pruning-Strategies.md#correlationreducer) | `signature` | `limen/experiment/reducer/correlation_reducer.py` |
| [`limen.experiment.reducer.CorrelationReducer.analyze_and_intervene`](Public-API.md#limenexperimentreducercorrelationreduceranalyze_and_intervene) | `signature` | `limen/experiment/reducer/correlation_reducer.py` |
| [`limen.experiment.reducer.CorrelationReducer.get_state`](Public-API.md#limenexperimentreducercorrelationreducerget_state) | `signature` | `limen/experiment/reducer/correlation_reducer.py` |
| [`limen.experiment.reducer.CorrelationReducer.set_state`](Public-API.md#limenexperimentreducercorrelationreducerset_state) | `signature` | `limen/experiment/reducer/correlation_reducer.py` |
| [`limen.experiment.reducer.FILTER_EXCLUDE_VALUE`](Public-API.md#limenexperimentreducerfilter_exclude_value) | `definition` | `limen/experiment/reducer/filter_types.py` |
| [`limen.experiment.reducer.FILTER_KEEP_BETWEEN`](Public-API.md#limenexperimentreducerfilter_keep_between) | `definition` | `limen/experiment/reducer/filter_types.py` |
| [`limen.experiment.reducer.FILTER_KEEP_VALUES`](Public-API.md#limenexperimentreducerfilter_keep_values) | `definition` | `limen/experiment/reducer/filter_types.py` |
| [`limen.experiment.reducer.FILTER_SAMPLE`](Public-API.md#limenexperimentreducerfilter_sample) | `definition` | `limen/experiment/reducer/filter_types.py` |
| [`limen.experiment.reducer.FocusReducer`](Public-API.md#limenexperimentreducerfocusreducer) | `signature` | `limen/experiment/reducer/focus_reducer.py` |
| [`limen.experiment.reducer.FocusReducer.__init__`](Developer/Pruning-Strategies.md#focusreducer) | `signature` | `limen/experiment/reducer/focus_reducer.py` |
| [`limen.experiment.reducer.FocusReducer.analyze_and_intervene`](Public-API.md#limenexperimentreducerfocusreduceranalyze_and_intervene) | `signature` | `limen/experiment/reducer/focus_reducer.py` |
| [`limen.experiment.reducer.FocusReducer.get_state`](Public-API.md#limenexperimentreducerfocusreducerget_state) | `signature` | `limen/experiment/reducer/focus_reducer.py` |
| [`limen.experiment.reducer.FocusReducer.set_state`](Public-API.md#limenexperimentreducerfocusreducerset_state) | `signature` | `limen/experiment/reducer/focus_reducer.py` |
| [`limen.experiment.reducer.PruningStrategy`](Public-API.md#limenexperimentreducerpruningstrategy) | `signature` | `limen/experiment/reducer/pruning_strategy.py` |
| [`limen.experiment.reducer.PruningStrategy.__init__`](Public-API.md#limenexperimentreducerpruningstrategy__init__) | `signature` | `limen/experiment/reducer/pruning_strategy.py` |
| [`limen.experiment.reducer.PruningStrategy.active`](Public-API.md#limenexperimentreducerpruningstrategyactive) | `signature` | `limen/experiment/reducer/pruning_strategy.py` |
| [`limen.experiment.reducer.PruningStrategy.analyze_and_intervene`](Developer/Pruning-Strategies.md#reducer-contract) | `signature` | `limen/experiment/reducer/pruning_strategy.py` |
| [`limen.experiment.reducer.PruningStrategy.get_state`](Developer/Pruning-Strategies.md#reducer-contract) | `signature` | `limen/experiment/reducer/pruning_strategy.py` |
| [`limen.experiment.reducer.PruningStrategy.set_state`](Developer/Pruning-Strategies.md#reducer-contract) | `signature` | `limen/experiment/reducer/pruning_strategy.py` |
| [`limen.experiment.reducer.REDUCER_REGISTRY`](Public-API.md#limenexperimentreducerreducer_registry) | `definition` | `limen/experiment/reducer/registry.py` |
| [`limen.experiment.reducer.SanityReducer`](Public-API.md#limenexperimentreducersanityreducer) | `signature` | `limen/experiment/reducer/sanity_reducer.py` |
| [`limen.experiment.reducer.SanityReducer.__init__`](Developer/Pruning-Strategies.md#sanityreducer) | `signature` | `limen/experiment/reducer/sanity_reducer.py` |
| [`limen.experiment.reducer.SanityReducer.analyze_and_intervene`](Public-API.md#limenexperimentreducersanityreduceranalyze_and_intervene) | `signature` | `limen/experiment/reducer/sanity_reducer.py` |
| [`limen.experiment.reducer.SanityReducer.get_state`](Public-API.md#limenexperimentreducersanityreducerget_state) | `signature` | `limen/experiment/reducer/sanity_reducer.py` |
| [`limen.experiment.reducer.SanityReducer.set_state`](Public-API.md#limenexperimentreducersanityreducerset_state) | `signature` | `limen/experiment/reducer/sanity_reducer.py` |
| [`limen.experiment.reducer.SaturationReducer`](Public-API.md#limenexperimentreducersaturationreducer) | `signature` | `limen/experiment/reducer/saturation_reducer.py` |
| [`limen.experiment.reducer.SaturationReducer.__init__`](Developer/Pruning-Strategies.md#saturationreducer) | `signature` | `limen/experiment/reducer/saturation_reducer.py` |
| [`limen.experiment.reducer.SaturationReducer.analyze_and_intervene`](Public-API.md#limenexperimentreducersaturationreduceranalyze_and_intervene) | `signature` | `limen/experiment/reducer/saturation_reducer.py` |
| [`limen.experiment.reducer.SaturationReducer.get_state`](Public-API.md#limenexperimentreducersaturationreducerget_state) | `signature` | `limen/experiment/reducer/saturation_reducer.py` |
| [`limen.experiment.reducer.SaturationReducer.set_state`](Public-API.md#limenexperimentreducersaturationreducerset_state) | `signature` | `limen/experiment/reducer/saturation_reducer.py` |
| [`limen.experiment.reducer.TRIM_RANDOM`](Public-API.md#limenexperimentreducertrim_random) | `definition` | `limen/experiment/reducer/budget_reducer.py` |
| [`limen.experiment.reducer.TRIM_WORST_FIRST`](Public-API.md#limenexperimentreducertrim_worst_first) | `definition` | `limen/experiment/reducer/budget_reducer.py` |
| `limen.features` | `module` | `limen/features/__init__.py` |
| [`limen.features.absorption_intensity`](Features.md#limenfeaturesabsorption_intensity) | `signature` | `limen/features/absorption_intensity.py` |
| [`limen.features.active_lines`](Features.md#limenfeaturesactive_lines) | `signature` | `limen/features/active_lines.py` |
| [`limen.features.active_quantile_count`](Features.md#limenfeaturesactive_quantile_count) | `signature` | `limen/features/active_quantile_count.py` |
| [`limen.features.amihud_illiquidity`](Features.md#limenfeaturesamihud_illiquidity) | `signature` | `limen/features/amihud_illiquidity.py` |
| [`limen.features.atr_percent_sma`](Features.md#limenfeaturesatr_percent_sma) | `signature` | `limen/features/atr_percent_sma.py` |
| [`limen.features.atr_sma`](Features.md#limenfeaturesatr_sma) | `signature` | `limen/features/atr_sma.py` |
| [`limen.features.body_to_range`](Features.md#limenfeaturesbody_to_range) | `signature` | `limen/features/body_to_range.py` |
| [`limen.features.breakout_features`](Features.md#limenfeaturesbreakout_features) | `signature` | `limen/features/breakout_features.py` |
| [`limen.features.breakout_percentile_regime`](Features.md#limenfeaturesbreakout_percentile_regime) | `signature` | `limen/features/breakout_percentile_regime.py` |
| [`limen.features.bulk_volume_classification`](Features.md#limenfeaturesbulk_volume_classification) | `signature` | `limen/features/bulk_volume_classification.py` |
| [`limen.features.calendar_time_features`](Features.md#limenfeaturescalendar_time_features) | `signature` | `limen/features/calendar_time_features.py` |
| [`limen.features.close_ma_distance_atr`](Features.md#limenfeaturesclose_ma_distance_atr) | `signature` | `limen/features/close_ma_distance_atr.py` |
| [`limen.features.close_position`](Features.md#limenfeaturesclose_position) | `signature` | `limen/features/close_position.py` |
| [`limen.features.close_position_rolling`](Features.md#limenfeaturesclose_position_rolling) | `signature` | `limen/features/close_position_rolling.py` |
| [`limen.features.close_to_extremes`](Features.md#limenfeaturesclose_to_extremes) | `signature` | `limen/features/close_to_extremes.py` |
| [`limen.features.common_strategy_positions.POSITION_COLUMN`](Public-API.md#limenfeaturescommon_strategy_positionsposition_column) | `definition` | `limen/features/common_strategy_positions.py` |
| [`limen.features.common_strategy_positions.common_strategy_positions`](Public-API.md#limenfeaturescommon_strategy_positionscommon_strategy_positions) | `signature` | `limen/features/common_strategy_positions.py` |
| [`limen.features.conserved_flux_renormalization`](Features.md#limenfeaturesconserved_flux_renormalization) | `signature` | `limen/features/conserved_flux_renormalization.py` |
| [`limen.features.cusum_filter`](Features.md#limenfeaturescusum_filter) | `signature` | `limen/features/cusum_filter.py` |
| [`limen.features.cyclical_time_features`](Features.md#limenfeaturescyclical_time_features) | `signature` | `limen/features/cyclical_time_features.py` |
| [`limen.features.distance_from_high`](Features.md#limenfeaturesdistance_from_high) | `signature` | `limen/features/distance_from_high.py` |
| [`limen.features.distance_from_low`](Features.md#limenfeaturesdistance_from_low) | `signature` | `limen/features/distance_from_low.py` |
| [`limen.features.distance_from_ma`](Features.md#limenfeaturesdistance_from_ma) | `signature` | `limen/features/distance_from_ma.py` |
| [`limen.features.dollar_bar_crash_reversal`](Features.md#limenfeaturesdollar_bar_crash_reversal) | `signature` | `limen/features/dollar_bar_crash_reversal.py` |
| [`limen.features.dollar_volume`](Features.md#limenfeaturesdollar_volume) | `signature` | `limen/features/dollar_volume.py` |
| [`limen.features.downside_volatility_ratio`](Features.md#limenfeaturesdownside_volatility_ratio) | `signature` | `limen/features/downside_volatility_ratio.py` |
| [`limen.features.dynamic_stop_loss`](Features.md#limenfeaturesdynamic_stop_loss) | `signature` | `limen/features/dynamic_stop_loss.py` |
| [`limen.features.dynamic_target`](Features.md#limenfeaturesdynamic_target) | `signature` | `limen/features/dynamic_target.py` |
| [`limen.features.ema_alignment`](Features.md#limenfeaturesema_alignment) | `signature` | `limen/features/ema_alignment.py` |
| [`limen.features.entry_score_microstructure`](Features.md#limenfeaturesentry_score_microstructure) | `signature` | `limen/features/entry_score_microstructure.py` |
| [`limen.features.feature_aliases`](Features.md#limenfeaturesfeature_aliases) | `signature` | `limen/features/feature_aliases.py` |
| [`limen.features.find_min_d`](Fractional-Differentiation.md#finding-the-minimum-d) | `signature` | `limen/features/fractional_diff.py` |
| [`limen.features.fractional_diff`](Fractional-Differentiation.md#adding-it-as-a-feature) | `signature` | `limen/features/fractional_diff.py` |
| [`limen.features.gap_high`](Features.md#limenfeaturesgap_high) | `signature` | `limen/features/gap_high.py` |
| [`limen.features.garman_klass_volatility`](Features.md#limenfeaturesgarman_klass_volatility) | `signature` | `limen/features/garman_klass_volatility.py` |
| [`limen.features.hh_hl_structure_regime`](Features.md#limenfeatureshh_hl_structure_regime) | `signature` | `limen/features/hh_hl_structure_regime.py` |
| [`limen.features.hours_since_big_move`](Features.md#limenfeatureshours_since_big_move) | `signature` | `limen/features/hours_since_big_move.py` |
| [`limen.features.hours_since_quantile_line`](Features.md#limenfeatureshours_since_quantile_line) | `signature` | `limen/features/hours_since_quantile_line.py` |
| [`limen.features.ichimoku_cloud`](Features.md#limenfeaturesichimoku_cloud) | `signature` | `limen/features/ichimoku_cloud.py` |
| [`limen.features.illiquidity_shock`](Features.md#limenfeaturesilliquidity_shock) | `signature` | `limen/features/illiquidity_shock.py` |
| [`limen.features.is_funding_hour`](Features.md#limenfeaturesis_funding_hour) | `signature` | `limen/features/is_funding_hour.py` |
| [`limen.features.is_us_open_hour`](Features.md#limenfeaturesis_us_open_hour) | `signature` | `limen/features/is_us_open_hour.py` |
| [`limen.features.jump_variation_proxy`](Features.md#limenfeaturesjump_variation_proxy) | `signature` | `limen/features/jump_variation_proxy.py` |
| [`limen.features.kaufman_efficiency_ratio`](Features.md#limenfeatureskaufman_efficiency_ratio) | `signature` | `limen/features/kaufman_efficiency_ratio.py` |
| [`limen.features.kline_imbalance`](Features.md#limenfeatureskline_imbalance) | `signature` | `limen/features/kline_imbalance.py` |
| [`limen.features.lag_column`](Features.md#limenfeatureslag_column) | `signature` | `limen/features/lagged_features.py` |
| [`limen.features.lag_columns`](Features.md#limenfeatureslag_columns) | `signature` | `limen/features/lagged_features.py` |
| [`limen.features.lag_range`](Features.md#limenfeatureslag_range) | `signature` | `limen/features/lagged_features.py` |
| [`limen.features.lag_range_cols`](Features.md#limenfeatureslag_range_cols) | `signature` | `limen/features/lagged_features.py` |
| [`limen.features.liquidity_drop`](Features.md#limenfeaturesliquidity_drop) | `signature` | `limen/features/liquidity_drop.py` |
| [`limen.features.liquidity_range`](Features.md#limenfeaturesliquidity_range) | `signature` | `limen/features/liquidity_range.py` |
| [`limen.features.log_returns`](Features.md#limenfeatureslog_returns) | `signature` | `limen/features/log_returns.py` |
| [`limen.features.ma_slope_regime`](Features.md#limenfeaturesma_slope_regime) | `signature` | `limen/features/ma_slope_regime.py` |
| [`limen.features.maker_liquidity_share`](Features.md#limenfeaturesmaker_liquidity_share) | `signature` | `limen/features/maker_liquidity_share.py` |
| [`limen.features.maker_volume_ratio`](Features.md#limenfeaturesmaker_volume_ratio) | `signature` | `limen/features/maker_volume_ratio.py` |
| [`limen.features.maker_volume_share`](Features.md#limenfeaturesmaker_volume_share) | `signature` | `limen/features/maker_volume_share.py` |
| [`limen.features.market_regime`](Features.md#limenfeaturesmarket_regime) | `signature` | `limen/features/market_regime.py` |
| [`limen.features.micro_momentum`](Features.md#limenfeaturesmicro_momentum) | `signature` | `limen/features/micro_momentum.py` |
| [`limen.features.momentum_confirmation`](Features.md#limenfeaturesmomentum_confirmation) | `signature` | `limen/features/momentum_confirmation.py` |
| [`limen.features.momentum_periods`](Features.md#limenfeaturesmomentum_periods) | `signature` | `limen/features/momentum_periods.py` |
| [`limen.features.momentum_weight`](Features.md#limenfeaturesmomentum_weight) | `signature` | `limen/features/momentum_weight.py` |
| [`limen.features.narrow_range`](Features.md#limenfeaturesnarrow_range) | `signature` | `limen/features/narrow_range.py` |
| [`limen.features.order_flow_imbalance`](Features.md#limenfeaturesorder_flow_imbalance) | `signature` | `limen/features/order_flow_imbalance.py` |
| [`limen.features.parkinson_vol_of_vol`](Features.md#limenfeaturesparkinson_vol_of_vol) | `signature` | `limen/features/parkinson_vol_of_vol.py` |
| [`limen.features.parkinson_volatility`](Features.md#limenfeaturesparkinson_volatility) | `signature` | `limen/features/parkinson_volatility.py` |
| [`limen.features.position_in_candle`](Features.md#limenfeaturesposition_in_candle) | `signature` | `limen/features/position_in_candle.py` |
| [`limen.features.position_in_range`](Features.md#limenfeaturesposition_in_range) | `signature` | `limen/features/position_in_range.py` |
| [`limen.features.price_lines`](Features.md#limenfeaturesprice_lines) | `signature` | `limen/features/price_lines.py` |
| [`limen.features.price_lines.line_active_count`](Public-API.md#limenfeaturesprice_linesline_active_count) | `signature` | `limen/features/price_lines.py` |
| [`limen.features.price_lines.line_recency`](Public-API.md#limenfeaturesprice_linesline_recency) | `signature` | `limen/features/price_lines.py` |
| [`limen.features.price_lines.price_lines`](Features.md#limenfeaturesprice_lines) | `signature` | `limen/features/price_lines.py` |
| [`limen.features.price_range_position`](Features.md#limenfeaturesprice_range_position) | `signature` | `limen/features/price_range_position.py` |
| [`limen.features.price_vs_band_regime`](Features.md#limenfeaturesprice_vs_band_regime) | `signature` | `limen/features/price_vs_band_regime.py` |
| [`limen.features.quantile_line_density`](Features.md#limenfeaturesquantile_line_density) | `signature` | `limen/features/quantile_line_density.py` |
| [`limen.features.quantile_price_lines`](Features.md#limenfeaturesquantile_price_lines) | `signature` | `limen/features/quantile_price_lines.py` |
| [`limen.features.range_overlap`](Features.md#limenfeaturesrange_overlap) | `signature` | `limen/features/range_overlap.py` |
| [`limen.features.range_pct`](Features.md#limenfeaturesrange_pct) | `signature` | `limen/features/range_pct.py` |
| [`limen.features.range_per_dollar_volume`](Features.md#limenfeaturesrange_per_dollar_volume) | `signature` | `limen/features/range_per_dollar_volume.py` |
| [`limen.features.realized_kurtosis`](Features.md#limenfeaturesrealized_kurtosis) | `signature` | `limen/features/realized_kurtosis.py` |
| [`limen.features.realized_semivariance`](Features.md#limenfeaturesrealized_semivariance) | `signature` | `limen/features/realized_semivariance.py` |
| [`limen.features.realized_skewness`](Features.md#limenfeaturesrealized_skewness) | `signature` | `limen/features/realized_skewness.py` |
| [`limen.features.regime_multiplier`](Features.md#limenfeaturesregime_multiplier) | `signature` | `limen/features/regime_multiplier.py` |
| [`limen.features.rejection_intensity`](Features.md#limenfeaturesrejection_intensity) | `signature` | `limen/features/rejection_intensity.py` |
| [`limen.features.relative_range_seasonality`](Features.md#limenfeaturesrelative_range_seasonality) | `signature` | `limen/features/relative_range_seasonality.py` |
| [`limen.features.relative_volatility_seasonality`](Features.md#limenfeaturesrelative_volatility_seasonality) | `signature` | `limen/features/relative_volatility_seasonality.py` |
| [`limen.features.relative_volume_seasonality`](Features.md#limenfeaturesrelative_volume_seasonality) | `signature` | `limen/features/relative_volume_seasonality.py` |
| [`limen.features.return_autocorrelation`](Features.md#limenfeaturesreturn_autocorrelation) | `signature` | `limen/features/return_autocorrelation.py` |
| [`limen.features.return_per_dollar_volume`](Features.md#limenfeaturesreturn_per_dollar_volume) | `signature` | `limen/features/return_per_dollar_volume.py` |
| [`limen.features.return_volatility_correlation`](Features.md#limenfeaturesreturn_volatility_correlation) | `signature` | `limen/features/return_volatility_correlation.py` |
| [`limen.features.returns_lags`](Features.md#limenfeaturesreturns_lags) | `signature` | `limen/features/returns_lags.py` |
| [`limen.features.rogers_satchell_volatility`](Features.md#limenfeaturesrogers_satchell_volatility) | `signature` | `limen/features/rogers_satchell_volatility.py` |
| [`limen.features.rolling_zscore`](Features.md#limenfeaturesrolling_zscore) | `signature` | `limen/features/rolling_zscore.py` |
| [`limen.features.sma_crossover`](Features.md#limenfeaturessma_crossover) | `signature` | `limen/features/sma_crossover.py` |
| [`limen.features.sma_ratios`](Features.md#limenfeaturessma_ratios) | `signature` | `limen/features/sma_ratios.py` |
| [`limen.features.spread`](Features.md#limenfeaturesspread) | `signature` | `limen/features/spread.py` |
| [`limen.features.spread_percent`](Features.md#limenfeaturesspread_percent) | `signature` | `limen/features/spread_percent.py` |
| [`limen.features.stochastic_k_abs`](Features.md#limenfeaturesstochastic_k_abs) | `signature` | `limen/features/stochastic_k_abs.py` |
| [`limen.features.tail_event_intensity`](Features.md#limenfeaturestail_event_intensity) | `signature` | `limen/features/tail_event_intensity.py` |
| [`limen.features.taker_imbalance_ratio`](Features.md#limenfeaturestaker_imbalance_ratio) | `signature` | `limen/features/taker_imbalance_ratio.py` |
| [`limen.features.time_to_funding`](Features.md#limenfeaturestime_to_funding) | `signature` | `limen/features/time_to_funding.py` |
| [`limen.features.trade_density`](Features.md#limenfeaturestrade_density) | `signature` | `limen/features/trade_density.py` |
| [`limen.features.trade_imbalance`](Features.md#limenfeaturestrade_imbalance) | `signature` | `limen/features/trade_imbalance.py` |
| [`limen.features.trade_size_ratio`](Features.md#limenfeaturestrade_size_ratio) | `signature` | `limen/features/trade_size_ratio.py` |
| [`limen.features.trend_coherence`](Features.md#limenfeaturestrend_coherence) | `signature` | `limen/features/trend_coherence.py` |
| [`limen.features.trend_strength`](Features.md#limenfeaturestrend_strength) | `signature` | `limen/features/trend_strength.py` |
| [`limen.features.volatility_1h`](Features.md#limenfeaturesvolatility_1h) | `signature` | `limen/features/volatility_1h.py` |
| [`limen.features.volatility_barriers.volatility_barriers`](Public-API.md#limenfeaturesvolatility_barriersvolatility_barriers) | `signature` | `limen/features/volatility_barriers.py` |
| [`limen.features.volatility_measure`](Features.md#limenfeaturesvolatility_measure) | `signature` | `limen/features/volatility_measure.py` |
| [`limen.features.volatility_of_volatility`](Features.md#limenfeaturesvolatility_of_volatility) | `signature` | `limen/features/volatility_of_volatility.py` |
| [`limen.features.volatility_ratio`](Features.md#limenfeaturesvolatility_ratio) | `signature` | `limen/features/volatility_ratio.py` |
| [`limen.features.volatility_spike`](Features.md#limenfeaturesvolatility_spike) | `signature` | `limen/features/volatility_spike.py` |
| [`limen.features.volatility_term_structure`](Features.md#limenfeaturesvolatility_term_structure) | `signature` | `limen/features/volatility_term_structure.py` |
| [`limen.features.volatility_weight`](Features.md#limenfeaturesvolatility_weight) | `signature` | `limen/features/volatility_weight.py` |
| [`limen.features.volume_ratio`](Features.md#limenfeaturesvolume_ratio) | `signature` | `limen/features/volume_ratio.py` |
| [`limen.features.volume_regime`](Features.md#limenfeaturesvolume_regime) | `signature` | `limen/features/volume_regime.py` |
| [`limen.features.volume_spike`](Features.md#limenfeaturesvolume_spike) | `signature` | `limen/features/volume_spike.py` |
| [`limen.features.volume_to_range`](Features.md#limenfeaturesvolume_to_range) | `signature` | `limen/features/volume_to_range.py` |
| [`limen.features.volume_trend`](Features.md#limenfeaturesvolume_trend) | `signature` | `limen/features/volume_trend.py` |
| [`limen.features.volume_volatility_correlation`](Features.md#limenfeaturesvolume_volatility_correlation) | `signature` | `limen/features/volume_volatility_correlation.py` |
| [`limen.features.vpin`](Features.md#limenfeaturesvpin) | `signature` | `limen/features/vpin.py` |
| [`limen.features.vwap`](Features.md#limenfeaturesvwap) | `signature` | `limen/features/vwap.py` |
| [`limen.features.wick_imbalance`](Features.md#limenfeatureswick_imbalance) | `signature` | `limen/features/wick_imbalance.py` |
| [`limen.features.wick_proportion`](Features.md#limenfeatureswick_proportion) | `signature` | `limen/features/wick_proportion.py` |
| [`limen.features.window_return_regime`](Features.md#limenfeatureswindow_return_regime) | `signature` | `limen/features/window_return_regime.py` |
| [`limen.features.yang_zhang_volatility`](Features.md#limenfeaturesyang_zhang_volatility) | `signature` | `limen/features/yang_zhang_volatility.py` |
| `limen.indicators` | `module` | `limen/indicators/__init__.py` |
| [`limen.indicators.ad`](Indicators.md#limenindicatorsad) | `signature` | `limen/indicators/ad.py` |
| [`limen.indicators.adosc`](Indicators.md#limenindicatorsadosc) | `signature` | `limen/indicators/adosc.py` |
| [`limen.indicators.apo`](Indicators.md#limenindicatorsapo) | `signature` | `limen/indicators/apo.py` |
| [`limen.indicators.atr`](Indicators.md#limenindicatorsatr) | `signature` | `limen/indicators/atr.py` |
| [`limen.indicators.avgprice`](Indicators.md#limenindicatorsavgprice) | `signature` | `limen/indicators/avgprice.py` |
| [`limen.indicators.bbands`](Indicators.md#limenindicatorsbbands) | `signature` | `limen/indicators/bbands.py` |
| [`limen.indicators.body_pct`](Indicators.md#limenindicatorsbody_pct) | `signature` | `limen/indicators/body_pct.py` |
| [`limen.indicators.bollinger_bands`](Indicators.md#limenindicatorsbollinger_bands) | `signature` | `limen/indicators/bollinger_bands.py` |
| [`limen.indicators.bollinger_position`](Indicators.md#limenindicatorsbollinger_position) | `signature` | `limen/indicators/bollinger_position.py` |
| [`limen.indicators.bop`](Indicators.md#limenindicatorsbop) | `signature` | `limen/indicators/bop.py` |
| [`limen.indicators.cci`](Indicators.md#limenindicatorscci) | `signature` | `limen/indicators/cci.py` |
| [`limen.indicators.cdl2crows`](Indicators.md#limenindicatorscdl2crows) | `signature` | `limen/indicators/cdl2crows.py` |
| [`limen.indicators.cdl3blackcrows`](Indicators.md#limenindicatorscdl3blackcrows) | `signature` | `limen/indicators/cdl3blackcrows.py` |
| [`limen.indicators.cdl3inside`](Indicators.md#limenindicatorscdl3inside) | `signature` | `limen/indicators/cdl3inside.py` |
| [`limen.indicators.cdl3linestrike`](Indicators.md#limenindicatorscdl3linestrike) | `signature` | `limen/indicators/cdl3linestrike.py` |
| [`limen.indicators.cdl3starsinsouth`](Indicators.md#limenindicatorscdl3starsinsouth) | `signature` | `limen/indicators/cdl3starsinsouth.py` |
| [`limen.indicators.cdl3whitesoldiers`](Indicators.md#limenindicatorscdl3whitesoldiers) | `signature` | `limen/indicators/cdl3whitesoldiers.py` |
| [`limen.indicators.cdlabandonedbaby`](Indicators.md#limenindicatorscdlabandonedbaby) | `signature` | `limen/indicators/cdlabandonedbaby.py` |
| [`limen.indicators.cdladvancedblock`](Indicators.md#limenindicatorscdladvancedblock) | `signature` | `limen/indicators/cdladvancedblock.py` |
| [`limen.indicators.cdladvancedblock.candle_totals`](Public-API.md#limenindicatorscdladvancedblockcandle_totals) | `signature` | `limen/indicators/cdladvancedblock.py` |
| [`limen.indicators.cdladvancedblock.cdladvancedblock`](Indicators.md#limenindicatorscdladvancedblock) | `signature` | `limen/indicators/cdladvancedblock.py` |
| [`limen.indicators.cdlbelthold`](Indicators.md#limenindicatorscdlbelthold) | `signature` | `limen/indicators/cdlbelthold.py` |
| [`limen.indicators.cdlclosingmarubozu`](Indicators.md#limenindicatorscdlclosingmarubozu) | `signature` | `limen/indicators/cdlclosingmarubozu.py` |
| [`limen.indicators.cdlconcealbabyswall`](Indicators.md#limenindicatorscdlconcealbabyswall) | `signature` | `limen/indicators/cdlconcealbabyswall.py` |
| [`limen.indicators.cdlcounterattack`](Indicators.md#limenindicatorscdlcounterattack) | `signature` | `limen/indicators/cdlcounterattack.py` |
| [`limen.indicators.cdldarkcloudcover`](Indicators.md#limenindicatorscdldarkcloudcover) | `signature` | `limen/indicators/cdldarkcloudcover.py` |
| [`limen.indicators.cdldoji`](Indicators.md#limenindicatorscdldoji) | `signature` | `limen/indicators/cdldoji.py` |
| [`limen.indicators.cdldragonflydoji`](Indicators.md#limenindicatorscdldragonflydoji) | `signature` | `limen/indicators/cdldragonflydoji.py` |
| [`limen.indicators.cdlengulfing`](Indicators.md#limenindicatorscdlengulfing) | `signature` | `limen/indicators/cdlengulfing.py` |
| [`limen.indicators.cdlgravestonedoji`](Indicators.md#limenindicatorscdlgravestonedoji) | `signature` | `limen/indicators/cdlgravestonedoji.py` |
| [`limen.indicators.cdlhammer`](Indicators.md#limenindicatorscdlhammer) | `signature` | `limen/indicators/cdlhammer.py` |
| [`limen.indicators.cdlhangingman`](Indicators.md#limenindicatorscdlhangingman) | `signature` | `limen/indicators/cdlhangingman.py` |
| [`limen.indicators.cdlharami`](Indicators.md#limenindicatorscdlharami) | `signature` | `limen/indicators/cdlharami.py` |
| [`limen.indicators.cdlharamicross`](Indicators.md#limenindicatorscdlharamicross) | `signature` | `limen/indicators/cdlharamicross.py` |
| [`limen.indicators.cdlhighwave`](Indicators.md#limenindicatorscdlhighwave) | `signature` | `limen/indicators/cdlhighwave.py` |
| [`limen.indicators.cdlhikkake`](Indicators.md#limenindicatorscdlhikkake) | `signature` | `limen/indicators/cdlhikkake.py` |
| [`limen.indicators.cdlhikkakemod`](Indicators.md#limenindicatorscdlhikkakemod) | `signature` | `limen/indicators/cdlhikkakemod.py` |
| [`limen.indicators.cdlhomingpigeon`](Indicators.md#limenindicatorscdlhomingpigeon) | `signature` | `limen/indicators/cdlhomingpigeon.py` |
| [`limen.indicators.cdlidentical3crows`](Indicators.md#limenindicatorscdlidentical3crows) | `signature` | `limen/indicators/cdlidentical3crows.py` |
| [`limen.indicators.cdlinvertedhammer`](Indicators.md#limenindicatorscdlinvertedhammer) | `signature` | `limen/indicators/cdlinvertedhammer.py` |
| [`limen.indicators.cdlladderbottom`](Indicators.md#limenindicatorscdlladderbottom) | `signature` | `limen/indicators/cdlladderbottom.py` |
| [`limen.indicators.cdllongleggeddoji`](Indicators.md#limenindicatorscdllongleggeddoji) | `signature` | `limen/indicators/cdllongleggeddoji.py` |
| [`limen.indicators.cdllongline`](Indicators.md#limenindicatorscdllongline) | `signature` | `limen/indicators/cdllongline.py` |
| [`limen.indicators.cdlmarubozu`](Indicators.md#limenindicatorscdlmarubozu) | `signature` | `limen/indicators/cdlmarubozu.py` |
| [`limen.indicators.cdlmatchinglow`](Indicators.md#limenindicatorscdlmatchinglow) | `signature` | `limen/indicators/cdlmatchinglow.py` |
| [`limen.indicators.cdlmathold`](Indicators.md#limenindicatorscdlmathold) | `signature` | `limen/indicators/cdlmathold.py` |
| [`limen.indicators.cdlonneck`](Indicators.md#limenindicatorscdlonneck) | `signature` | `limen/indicators/cdlonneck.py` |
| [`limen.indicators.cdlpiercing`](Indicators.md#limenindicatorscdlpiercing) | `signature` | `limen/indicators/cdlpiercing.py` |
| [`limen.indicators.cdlrickshawman`](Indicators.md#limenindicatorscdlrickshawman) | `signature` | `limen/indicators/cdlrickshawman.py` |
| [`limen.indicators.cdlrisefall3methods`](Indicators.md#limenindicatorscdlrisefall3methods) | `signature` | `limen/indicators/cdlrisefall3methods.py` |
| [`limen.indicators.cdlseparatinglines`](Indicators.md#limenindicatorscdlseparatinglines) | `signature` | `limen/indicators/cdlseparatinglines.py` |
| [`limen.indicators.cdlshootingstar`](Indicators.md#limenindicatorscdlshootingstar) | `signature` | `limen/indicators/cdlshootingstar.py` |
| [`limen.indicators.cdlshortline`](Indicators.md#limenindicatorscdlshortline) | `signature` | `limen/indicators/cdlshortline.py` |
| [`limen.indicators.cdlspinningtop`](Indicators.md#limenindicatorscdlspinningtop) | `signature` | `limen/indicators/cdlspinningtop.py` |
| [`limen.indicators.cdlstalledpattern`](Indicators.md#limenindicatorscdlstalledpattern) | `signature` | `limen/indicators/cdlstalledpattern.py` |
| [`limen.indicators.cdlsticksandwich`](Indicators.md#limenindicatorscdlsticksandwich) | `signature` | `limen/indicators/cdlsticksandwich.py` |
| [`limen.indicators.cdltakuri`](Indicators.md#limenindicatorscdltakuri) | `signature` | `limen/indicators/cdltakuri.py` |
| [`limen.indicators.cdlthrusting`](Indicators.md#limenindicatorscdlthrusting) | `signature` | `limen/indicators/cdlthrusting.py` |
| [`limen.indicators.cdltristar`](Indicators.md#limenindicatorscdltristar) | `signature` | `limen/indicators/cdltristar.py` |
| [`limen.indicators.cdlunique3river`](Indicators.md#limenindicatorscdlunique3river) | `signature` | `limen/indicators/cdlunique3river.py` |
| [`limen.indicators.cmo`](Indicators.md#limenindicatorscmo) | `signature` | `limen/indicators/cmo.py` |
| [`limen.indicators.dema`](Indicators.md#limenindicatorsdema) | `signature` | `limen/indicators/dema.py` |
| [`limen.indicators.ema`](Indicators.md#limenindicatorsema) | `signature` | `limen/indicators/ema.py` |
| [`limen.indicators.ht_dcperiod`](Indicators.md#limenindicatorsht_dcperiod) | `signature` | `limen/indicators/ht_dcperiod.py` |
| [`limen.indicators.ht_dcphase`](Indicators.md#limenindicatorsht_dcphase) | `signature` | `limen/indicators/ht_dcphase.py` |
| [`limen.indicators.ht_phasor`](Indicators.md#limenindicatorsht_phasor) | `signature` | `limen/indicators/ht_phasor.py` |
| [`limen.indicators.ht_sine`](Indicators.md#limenindicatorsht_sine) | `signature` | `limen/indicators/ht_sine.py` |
| [`limen.indicators.ht_trendline`](Indicators.md#limenindicatorsht_trendline) | `signature` | `limen/indicators/ht_trendline.py` |
| [`limen.indicators.ht_trendmode`](Indicators.md#limenindicatorsht_trendmode) | `signature` | `limen/indicators/ht_trendmode.py` |
| [`limen.indicators.kama`](Indicators.md#limenindicatorskama) | `signature` | `limen/indicators/kama.py` |
| [`limen.indicators.linearreg`](Indicators.md#limenindicatorslinearreg) | `signature` | `limen/indicators/linearreg.py` |
| [`limen.indicators.linearreg.linearreg`](Indicators.md#limenindicatorslinearreg) | `signature` | `limen/indicators/linearreg.py` |
| [`limen.indicators.linearreg.linearreg_components`](Public-API.md#limenindicatorslinearreglinearreg_components) | `signature` | `limen/indicators/linearreg.py` |
| [`limen.indicators.linearreg_angle`](Indicators.md#limenindicatorslinearreg_angle) | `signature` | `limen/indicators/linearreg_angle.py` |
| [`limen.indicators.linearreg_intercept`](Indicators.md#limenindicatorslinearreg_intercept) | `signature` | `limen/indicators/linearreg_intercept.py` |
| [`limen.indicators.linearreg_slope`](Indicators.md#limenindicatorslinearreg_slope) | `signature` | `limen/indicators/linearreg_slope.py` |
| [`limen.indicators.ma`](Indicators.md#limenindicatorsma) | `signature` | `limen/indicators/ma.py` |
| [`limen.indicators.macd`](Indicators.md#limenindicatorsmacd) | `signature` | `limen/indicators/macd.py` |
| [`limen.indicators.macdext`](Indicators.md#limenindicatorsmacdext) | `signature` | `limen/indicators/macdext.py` |
| [`limen.indicators.macdfix`](Indicators.md#limenindicatorsmacdfix) | `signature` | `limen/indicators/macdfix.py` |
| [`limen.indicators.mama`](Indicators.md#limenindicatorsmama) | `signature` | `limen/indicators/mama.py` |
| [`limen.indicators.medprice`](Indicators.md#limenindicatorsmedprice) | `signature` | `limen/indicators/medprice.py` |
| [`limen.indicators.mfi`](Indicators.md#limenindicatorsmfi) | `signature` | `limen/indicators/mfi.py` |
| [`limen.indicators.midpoint`](Indicators.md#limenindicatorsmidpoint) | `signature` | `limen/indicators/midpoint.py` |
| [`limen.indicators.midprice`](Indicators.md#limenindicatorsmidprice) | `signature` | `limen/indicators/midprice.py` |
| [`limen.indicators.midprice.midprice`](Indicators.md#limenindicatorsmidprice) | `signature` | `limen/indicators/midprice.py` |
| [`limen.indicators.midprice.window_extrema`](Public-API.md#limenindicatorsmidpricewindow_extrema) | `signature` | `limen/indicators/midprice.py` |
| [`limen.indicators.mom`](Indicators.md#limenindicatorsmom) | `signature` | `limen/indicators/mom.py` |
| [`limen.indicators.natr`](Indicators.md#limenindicatorsnatr) | `signature` | `limen/indicators/natr.py` |
| [`limen.indicators.obv`](Indicators.md#limenindicatorsobv) | `signature` | `limen/indicators/obv.py` |
| [`limen.indicators.ppo`](Indicators.md#limenindicatorsppo) | `signature` | `limen/indicators/ppo.py` |
| [`limen.indicators.price_change_pct`](Indicators.md#limenindicatorsprice_change_pct) | `signature` | `limen/indicators/price_change_pct.py` |
| [`limen.indicators.returns`](Indicators.md#limenindicatorsreturns) | `signature` | `limen/indicators/returns.py` |
| [`limen.indicators.roc`](Developer/Writing-Docstrings.md#house-style) | `signature` | `limen/indicators/roc.py` |
| [`limen.indicators.rocp`](Indicators.md#limenindicatorsrocp) | `signature` | `limen/indicators/rocp.py` |
| [`limen.indicators.rocr`](Indicators.md#limenindicatorsrocr) | `signature` | `limen/indicators/rocr.py` |
| [`limen.indicators.rocr100`](Indicators.md#limenindicatorsrocr100) | `signature` | `limen/indicators/rocr100.py` |
| [`limen.indicators.rolling_volatility`](Indicators.md#limenindicatorsrolling_volatility) | `signature` | `limen/indicators/rolling_volatility.py` |
| [`limen.indicators.rsi`](Indicators.md#limenindicatorsrsi) | `signature` | `limen/indicators/rsi.py` |
| [`limen.indicators.rsi_sma`](Indicators.md#limenindicatorsrsi_sma) | `signature` | `limen/indicators/rsi_sma.py` |
| [`limen.indicators.sar`](Indicators.md#limenindicatorssar) | `signature` | `limen/indicators/sar.py` |
| [`limen.indicators.sarext`](Indicators.md#limenindicatorssarext) | `signature` | `limen/indicators/sarext.py` |
| [`limen.indicators.sma`](Indicators.md#limenindicatorssma) | `signature` | `limen/indicators/sma.py` |
| [`limen.indicators.sma_deviation_std`](Indicators.md#limenindicatorssma_deviation_std) | `signature` | `limen/indicators/sma_deviation_std.py` |
| [`limen.indicators.stddev`](Indicators.md#limenindicatorsstddev) | `signature` | `limen/indicators/stddev.py` |
| [`limen.indicators.stoch`](Indicators.md#limenindicatorsstoch) | `signature` | `limen/indicators/stoch.py` |
| [`limen.indicators.stochastic_oscillator`](Indicators.md#limenindicatorsstochastic_oscillator) | `signature` | `limen/indicators/stochastic_oscillator.py` |
| [`limen.indicators.stochf`](Indicators.md#limenindicatorsstochf) | `signature` | `limen/indicators/stochf.py` |
| [`limen.indicators.stochrsi`](Indicators.md#limenindicatorsstochrsi) | `signature` | `limen/indicators/stochrsi.py` |
| [`limen.indicators.t3`](Indicators.md#limenindicatorst3) | `signature` | `limen/indicators/t3.py` |
| [`limen.indicators.tema`](Indicators.md#limenindicatorstema) | `signature` | `limen/indicators/tema.py` |
| [`limen.indicators.trange`](Indicators.md#limenindicatorstrange) | `signature` | `limen/indicators/trange.py` |
| [`limen.indicators.trima`](Indicators.md#limenindicatorstrima) | `signature` | `limen/indicators/trima.py` |
| [`limen.indicators.trix`](Indicators.md#limenindicatorstrix) | `signature` | `limen/indicators/trix.py` |
| [`limen.indicators.tsf`](Indicators.md#limenindicatorstsf) | `signature` | `limen/indicators/tsf.py` |
| [`limen.indicators.typprice`](Indicators.md#limenindicatorstypprice) | `signature` | `limen/indicators/typprice.py` |
| [`limen.indicators.ultosc`](Indicators.md#limenindicatorsultosc) | `signature` | `limen/indicators/ultosc.py` |
| [`limen.indicators.var`](Indicators.md#limenindicatorsvar) | `signature` | `limen/indicators/var.py` |
| [`limen.indicators.wclprice`](Indicators.md#limenindicatorswclprice) | `signature` | `limen/indicators/wclprice.py` |
| [`limen.indicators.wilder_rsi`](Indicators.md#limenindicatorswilder_rsi) | `signature` | `limen/indicators/wilder_rsi.py` |
| [`limen.indicators.willr`](Indicators.md#limenindicatorswillr) | `signature` | `limen/indicators/willr.py` |
| [`limen.indicators.window_return`](Indicators.md#limenindicatorswindow_return) | `signature` | `limen/indicators/window_return.py` |
| [`limen.indicators.wma`](Indicators.md#limenindicatorswma) | `signature` | `limen/indicators/wma.py` |
| [`limen.inference.BarPrediction`](Trainer.md#predictraw_klines---barprediction) | `signature` | `limen/inference/_python_sensor.py` |
| [`limen.inference.ReconstructionError`](Public-API.md#limenreconstructionerror) | `signature` | `limen/inference/errors.py` |
| [`limen.inference.Sensor`](Public-API.md#limensensor) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.Sensor.__call__`](Trainer.md#sensor-inference-api) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.Sensor.__init__`](Public-API.md#limensensor__init__) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.Sensor.from_model`](Trainer.md#python-construction) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.Sensor.from_python`](Trainer.md#arbitrary-python-inference) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.Sensor.predict`](Trainer.md#sensor-inference-api) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.Sensor.predict_all`](Public-API.md#limensensorpredict_all) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.Sensor.prediction_mode`](Public-API.md#limensensorprediction_mode) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.Sensor.round_params`](Public-API.md#limensensorround_params) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.Sensor.trade_contract`](Public-API.md#limensensortrade_contract) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.Trainer`](Public-API.md#limentrainer) | `signature` | `limen/inference/trainer.py` |
| [`limen.inference.Trainer.__init__`](Trainer.md#trainerexperiment_dir-datanone) | `signature` | `limen/inference/trainer.py` |
| [`limen.inference.Trainer.train`](Trainer.md#trainpermutation_ids) | `signature` | `limen/inference/trainer.py` |
| [`limen.inference.sensor.BarPrediction`](Trainer.md#predictraw_klines---barprediction) | `signature` | `limen/inference/_python_sensor.py` |
| [`limen.inference.sensor.PredictionReason`](Public-API.md#limeninferencesensorpredictionreason) | `definition` | `limen/inference/_python_sensor.py` |
| [`limen.inference.sensor.Sensor`](Public-API.md#limensensor) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.sensor.Sensor.__call__`](Trainer.md#sensor-inference-api) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.sensor.Sensor.__init__`](Public-API.md#limensensor__init__) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.sensor.Sensor.from_model`](Trainer.md#python-construction) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.sensor.Sensor.from_python`](Trainer.md#arbitrary-python-inference) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.sensor.Sensor.predict`](Trainer.md#sensor-inference-api) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.sensor.Sensor.predict_all`](Public-API.md#limensensorpredict_all) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.sensor.Sensor.prediction_mode`](Public-API.md#limensensorprediction_mode) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.sensor.Sensor.round_params`](Public-API.md#limensensorround_params) | `signature` | `limen/inference/sensor.py` |
| [`limen.inference.sensor.Sensor.trade_contract`](Public-API.md#limensensortrade_contract) | `signature` | `limen/inference/sensor.py` |
| `limen.log` | `module` | `limen/log/__init__.py` |
| [`limen.log.ConfusionNotApplicableError`](Public-API.md#limenlogconfusionnotapplicableerror) | `signature` | `limen/log/_permutation_confusion_metrics.py` |
| [`limen.log.experiment_backtest_results`](Log.md#experiment_backtest_results) | `signature` | `limen/log/_experiment_backtest_results.py` |
| [`limen.log.experiment_confusion_metrics`](Log.md#experiment_confusion_metrics) | `signature` | `limen/log/_experiment_confusion_metrics.py` |
| [`limen.log.experiment_parameter_correlation`](Perturbation-Strategies.md#deriving-feature-importance-from-ablation) | `signature` | `limen/log/_experiment_parameter_correlation.py` |
| [`limen.log.permutation_confusion_metrics`](Log.md#permutation_confusion_metricsx-round_id) | `signature` | `limen/log/_permutation_confusion_metrics.py` |
| [`limen.log.permutation_prediction_performance`](Public-API.md#limenlogpermutation_prediction_performance) | `signature` | `limen/log/_permutation_prediction_performance.py` |
| `limen.metrics` | `module` | `limen/metrics/__init__.py` |
| [`limen.metrics.balanced_metric`](Standard-Metrics-Library.md#limenmetricsbalanced_metric) | `signature` | `limen/metrics/balanced_metric.py` |
| `limen.metrics.binary_metrics` | `module` | `limen/metrics/binary_metrics.py` |
| `limen.metrics.continuous_metrics` | `module` | `limen/metrics/continuous_metrics.py` |
| [`limen.metrics.deflated_sharpe_ratio`](Standard-Metrics-Library.md#limenmetricsdeflated_sharpe_ratio) | `signature` | `limen/metrics/deflated_sharpe_ratio.py` |
| [`limen.metrics.deflated_sharpe_ratio.deflated_sharpe_ratio`](Standard-Metrics-Library.md#limenmetricsdeflated_sharpe_ratio) | `signature` | `limen/metrics/deflated_sharpe_ratio.py` |
| `limen.metrics.multiclass_metrics` | `module` | `limen/metrics/multiclass_metrics.py` |
| [`limen.metrics.probability_of_backtest_overfitting`](Standard-Metrics-Library.md#limenmetricsprobability_of_backtest_overfitting) | `signature` | `limen/metrics/probability_of_backtest_overfitting.py` |
| [`limen.metrics.probability_of_backtest_overfitting.probability_of_backtest_overfitting`](Standard-Metrics-Library.md#limenmetricsprobability_of_backtest_overfitting) | `signature` | `limen/metrics/probability_of_backtest_overfitting.py` |
| `limen.metrics.rule_based_metrics` | `module` | `limen/metrics/rule_based_metrics.py` |
| `limen.metrics.safe_ovr_auc` | `module` | `limen/metrics/safe_ovr_auc.py` |
| `limen.scalers` | `module` | `limen/scalers/__init__.py` |
| [`limen.scalers.CausalRollingRobustScaler`](Scalers.md#limenscalerscausalrollingrobustscaler) | `signature` | `limen/scalers/causal_rolling_robust_scaler.py` |
| [`limen.scalers.CausalRollingRobustScaler.__init__`](Scalers.md#limenscalerscausalrollingrobustscaler__init__) | `signature` | `limen/scalers/causal_rolling_robust_scaler.py` |
| [`limen.scalers.CausalRollingRobustScaler.context_rows`](Scalers.md#limenscalerscausalrollingrobustscalercontext_rows) | `signature` | `limen/scalers/causal_rolling_robust_scaler.py` |
| [`limen.scalers.CausalRollingRobustScaler.transform`](Scalers.md#limenscalerscausalrollingrobustscalertransform) | `signature` | `limen/scalers/causal_rolling_robust_scaler.py` |
| [`limen.scalers.LinearScaler`](Scalers.md#limenscalerslinearscaler) | `signature` | `limen/scalers/linear_scaler.py` |
| [`limen.scalers.LinearScaler.__init__`](Scalers.md#limenscalerslinearscaler__init__) | `signature` | `limen/scalers/linear_scaler.py` |
| [`limen.scalers.LinearScaler.transform`](Scalers.md#limenscalerslinearscalertransform) | `signature` | `limen/scalers/linear_scaler.py` |
| [`limen.scalers.LogRegScaler`](Scalers.md#limenscalerslogregscaler) | `signature` | `limen/scalers/logreg_scaler.py` |
| [`limen.scalers.LogRegScaler.__init__`](Scalers.md#limenscalerslogregscaler__init__) | `signature` | `limen/scalers/logreg_scaler.py` |
| [`limen.scalers.LogRegScaler.transform`](Scalers.md#limenscalerslogregscalertransform) | `signature` | `limen/scalers/logreg_scaler.py` |
| [`limen.scalers.RankGaussScaler`](Scalers.md#limenscalersrankgaussscaler) | `signature` | `limen/scalers/rank_gauss_scaler.py` |
| [`limen.scalers.RankGaussScaler.__init__`](Scalers.md#limenscalersrankgaussscaler__init__) | `signature` | `limen/scalers/rank_gauss_scaler.py` |
| [`limen.scalers.RankGaussScaler.transform`](Scalers.md#limenscalersrankgaussscalertransform) | `signature` | `limen/scalers/rank_gauss_scaler.py` |
| [`limen.scalers.RobustScaler`](Scalers.md#limenscalersrobustscaler) | `signature` | `limen/scalers/robust_scaler.py` |
| [`limen.scalers.RobustScaler.__init__`](Scalers.md#limenscalersrobustscaler__init__) | `signature` | `limen/scalers/robust_scaler.py` |
| [`limen.scalers.RobustScaler.transform`](Scalers.md#limenscalersrobustscalertransform) | `signature` | `limen/scalers/robust_scaler.py` |
| [`limen.scalers.SCALER_REGISTRY`](Scalers.md#fit-once-then-reuse) | `definition` | `limen/scalers/registry.py` |
| `limen.sfd` | `module` | `limen/sfd/__init__.py` |
| `limen.sfd.foundational_rule_based` | `module` | `limen/sfd/foundational_sfd/rule_based.py` |
| `limen.sfd.foundational_sfd` | `module` | `limen/sfd/foundational_sfd/__init__.py` |
| `limen.sfd.foundational_sfd.common_strategies` | `module` | `limen/sfd/foundational_sfd/common_strategies.py` |
| [`limen.sfd.foundational_sfd.common_strategies.manifest`](Common-Strategies.md#run-the-eight-strategy-grid) | `signature` | `limen/sfd/foundational_sfd/common_strategies.py` |
| [`limen.sfd.foundational_sfd.common_strategies.params`](Common-Strategies.md#run-the-eight-strategy-grid) | `signature` | `limen/sfd/foundational_sfd/common_strategies.py` |
| `limen.sfd.foundational_sfd.dlinear_regressor` | `module` | `limen/sfd/foundational_sfd/dlinear_regressor.py` |
| `limen.sfd.foundational_sfd.dollar_bar_crash_reversal` | `module` | `limen/sfd/foundational_sfd/dollar_bar_crash_reversal.py` |
| `limen.sfd.foundational_sfd.lightgbm_binary` | `module` | `limen/sfd/foundational_sfd/lightgbm_binary.py` |
| `limen.sfd.foundational_sfd.logreg_binary` | `module` | `limen/sfd/foundational_sfd/logreg_binary.py` |
| `limen.sfd.foundational_sfd.random_binary` | `module` | `limen/sfd/foundational_sfd/random_binary.py` |
| `limen.sfd.foundational_sfd.ridge_regressor` | `module` | `limen/sfd/foundational_sfd/ridge_regressor.py` |
| [`limen.sfd.foundational_sfd.ridge_regressor.manifest`](Public-API.md#limensfdfoundational_sfdridge_regressormanifest) | `signature` | `limen/sfd/foundational_sfd/ridge_regressor.py` |
| [`limen.sfd.foundational_sfd.ridge_regressor.params`](Public-API.md#limensfdfoundational_sfdridge_regressorparams) | `signature` | `limen/sfd/foundational_sfd/ridge_regressor.py` |
| `limen.sfd.foundational_sfd.rule_based` | `module` | `limen/sfd/foundational_sfd/rule_based.py` |
| `limen.sfd.foundational_sfd.tabpfn_binary` | `module` | `limen/sfd/foundational_sfd/tabpfn_binary.py` |
| `limen.sfd.foundational_sfd.xgboost_regressor` | `module` | `limen/sfd/foundational_sfd/xgboost_regressor.py` |
| [`limen.sfd.foundational_sfd.xgboost_regressor.manifest`](Public-API.md#limensfdfoundational_sfdxgboost_regressormanifest) | `signature` | `limen/sfd/foundational_sfd/xgboost_regressor.py` |
| [`limen.sfd.foundational_sfd.xgboost_regressor.params`](Public-API.md#limensfdfoundational_sfdxgboost_regressorparams) | `signature` | `limen/sfd/foundational_sfd/xgboost_regressor.py` |
| `limen.sfd.logreg_binary` | `module` | `limen/sfd/foundational_sfd/logreg_binary.py` |
| `limen.sfd.random_binary` | `module` | `limen/sfd/foundational_sfd/random_binary.py` |
| [`limen.sfd.reference_architecture.DLinearRegressor`](Public-API.md#limensfdreference_architecturedlinearregressor) | `signature` | `limen/sfd/reference_architecture/dlinear_regressor.py` |
| [`limen.sfd.reference_architecture.DLinearRegressor.evaluate`](Public-API.md#limensfdreference_architecturedlinearregressorevaluate) | `signature` | `limen/sfd/reference_architecture/dlinear_regressor.py` |
| [`limen.sfd.reference_architecture.DLinearRegressor.predict`](Public-API.md#limensfdreference_architecturedlinearregressorpredict) | `signature` | `limen/sfd/reference_architecture/dlinear_regressor.py` |
| [`limen.sfd.reference_architecture.DLinearRegressor.train`](Public-API.md#limensfdreference_architecturedlinearregressortrain) | `signature` | `limen/sfd/reference_architecture/dlinear_regressor.py` |
| [`limen.sfd.reference_architecture.DirectionSizingModel`](Public-API.md#limensfdreference_architecturedirectionsizingmodel) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.DirectionSizingModel.evaluate`](Public-API.md#limensfdreference_architecturedirectionsizingmodelevaluate) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.DirectionSizingModel.predict`](Public-API.md#limensfdreference_architecturedirectionsizingmodelpredict) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.DirectionSizingModel.train`](Public-API.md#limensfdreference_architecturedirectionsizingmodeltrain) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.LightGBMBinary`](Public-API.md#limensfdreference_architecturelightgbmbinary) | `signature` | `limen/sfd/reference_architecture/lightgbm_binary.py` |
| [`limen.sfd.reference_architecture.LightGBMBinary.__init__`](Public-API.md#limensfdreference_architecturelightgbmbinary__init__) | `signature` | `limen/sfd/reference_architecture/lightgbm_binary.py` |
| [`limen.sfd.reference_architecture.LightGBMBinary.evaluate`](Public-API.md#limensfdreference_architecturelightgbmbinaryevaluate) | `signature` | `limen/sfd/reference_architecture/lightgbm_binary.py` |
| [`limen.sfd.reference_architecture.LightGBMBinary.predict`](Public-API.md#limensfdreference_architecturelightgbmbinarypredict) | `signature` | `limen/sfd/reference_architecture/lightgbm_binary.py` |
| [`limen.sfd.reference_architecture.LightGBMBinary.train`](Public-API.md#limensfdreference_architecturelightgbmbinarytrain) | `signature` | `limen/sfd/reference_architecture/lightgbm_binary.py` |
| [`limen.sfd.reference_architecture.LogRegBinary`](Public-API.md#limensfdreference_architecturelogregbinary) | `signature` | `limen/sfd/reference_architecture/logreg_binary.py` |
| [`limen.sfd.reference_architecture.LogRegBinary.__init__`](Public-API.md#limensfdreference_architecturelogregbinary__init__) | `signature` | `limen/sfd/reference_architecture/logreg_binary.py` |
| [`limen.sfd.reference_architecture.LogRegBinary.evaluate`](Public-API.md#limensfdreference_architecturelogregbinaryevaluate) | `signature` | `limen/sfd/reference_architecture/logreg_binary.py` |
| [`limen.sfd.reference_architecture.LogRegBinary.predict`](Public-API.md#limensfdreference_architecturelogregbinarypredict) | `signature` | `limen/sfd/reference_architecture/logreg_binary.py` |
| [`limen.sfd.reference_architecture.LogRegBinary.train`](Public-API.md#limensfdreference_architecturelogregbinarytrain) | `signature` | `limen/sfd/reference_architecture/logreg_binary.py` |
| [`limen.sfd.reference_architecture.RandomBinary`](Public-API.md#limensfdreference_architecturerandombinary) | `signature` | `limen/sfd/reference_architecture/random_binary.py` |
| [`limen.sfd.reference_architecture.RandomBinary.__init__`](Public-API.md#limensfdreference_architecturerandombinary__init__) | `signature` | `limen/sfd/reference_architecture/random_binary.py` |
| [`limen.sfd.reference_architecture.RandomBinary.evaluate`](Public-API.md#limensfdreference_architecturerandombinaryevaluate) | `signature` | `limen/sfd/reference_architecture/random_binary.py` |
| [`limen.sfd.reference_architecture.RandomBinary.predict`](Public-API.md#limensfdreference_architecturerandombinarypredict) | `signature` | `limen/sfd/reference_architecture/random_binary.py` |
| [`limen.sfd.reference_architecture.RandomBinary.train`](Public-API.md#limensfdreference_architecturerandombinarytrain) | `signature` | `limen/sfd/reference_architecture/random_binary.py` |
| [`limen.sfd.reference_architecture.ReferenceModel`](Public-API.md#limensfdreference_architecturereferencemodel) | `signature` | `limen/sfd/reference_architecture/base.py` |
| [`limen.sfd.reference_architecture.ReferenceModel.__init__`](Public-API.md#limensfdreference_architecturereferencemodel__init__) | `signature` | `limen/sfd/reference_architecture/base.py` |
| [`limen.sfd.reference_architecture.ReferenceModel.evaluate`](Reference-Architecture.md#referencemodel) | `signature` | `limen/sfd/reference_architecture/base.py` |
| [`limen.sfd.reference_architecture.ReferenceModel.predict`](Reference-Architecture.md#referencemodel) | `signature` | `limen/sfd/reference_architecture/base.py` |
| [`limen.sfd.reference_architecture.ReferenceModel.train`](Reference-Architecture.md#referencemodel) | `signature` | `limen/sfd/reference_architecture/base.py` |
| [`limen.sfd.reference_architecture.RidgeRegressor`](Public-API.md#limensfdreference_architectureridgeregressor) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.RidgeRegressor.__init__`](Public-API.md#limensfdreference_architectureridgeregressor__init__) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.RidgeRegressor.evaluate`](Public-API.md#limensfdreference_architectureridgeregressorevaluate) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.RidgeRegressor.predict`](Public-API.md#limensfdreference_architectureridgeregressorpredict) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.RidgeRegressor.train`](Public-API.md#limensfdreference_architectureridgeregressortrain) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.RuleBasedStrategy`](Public-API.md#limensfdreference_architecturerulebasedstrategy) | `signature` | `limen/sfd/reference_architecture/rule_based.py` |
| [`limen.sfd.reference_architecture.RuleBasedStrategy.__init__`](Public-API.md#limensfdreference_architecturerulebasedstrategy__init__) | `signature` | `limen/sfd/reference_architecture/rule_based.py` |
| [`limen.sfd.reference_architecture.RuleBasedStrategy.evaluate`](Public-API.md#limensfdreference_architecturerulebasedstrategyevaluate) | `signature` | `limen/sfd/reference_architecture/rule_based.py` |
| [`limen.sfd.reference_architecture.RuleBasedStrategy.predict`](Public-API.md#limensfdreference_architecturerulebasedstrategypredict) | `signature` | `limen/sfd/reference_architecture/rule_based.py` |
| [`limen.sfd.reference_architecture.RuleBasedStrategy.train`](Public-API.md#limensfdreference_architecturerulebasedstrategytrain) | `signature` | `limen/sfd/reference_architecture/rule_based.py` |
| [`limen.sfd.reference_architecture.TabPFNBinary`](Public-API.md#limensfdreference_architecturetabpfnbinary) | `signature` | `limen/sfd/reference_architecture/tabpfn_binary.py` |
| [`limen.sfd.reference_architecture.TabPFNBinary.__init__`](Public-API.md#limensfdreference_architecturetabpfnbinary__init__) | `signature` | `limen/sfd/reference_architecture/tabpfn_binary.py` |
| [`limen.sfd.reference_architecture.TabPFNBinary.evaluate`](Public-API.md#limensfdreference_architecturetabpfnbinaryevaluate) | `signature` | `limen/sfd/reference_architecture/tabpfn_binary.py` |
| [`limen.sfd.reference_architecture.TabPFNBinary.predict`](Public-API.md#limensfdreference_architecturetabpfnbinarypredict) | `signature` | `limen/sfd/reference_architecture/tabpfn_binary.py` |
| [`limen.sfd.reference_architecture.TabPFNBinary.train`](Public-API.md#limensfdreference_architecturetabpfnbinarytrain) | `signature` | `limen/sfd/reference_architecture/tabpfn_binary.py` |
| [`limen.sfd.reference_architecture.XGBoostRegressor`](Public-API.md#limensfdreference_architecturexgboostregressor) | `signature` | `limen/sfd/reference_architecture/xgboost_regressor.py` |
| [`limen.sfd.reference_architecture.XGBoostRegressor.evaluate`](Public-API.md#limensfdreference_architecturexgboostregressorevaluate) | `signature` | `limen/sfd/reference_architecture/xgboost_regressor.py` |
| [`limen.sfd.reference_architecture.XGBoostRegressor.predict`](Public-API.md#limensfdreference_architecturexgboostregressorpredict) | `signature` | `limen/sfd/reference_architecture/xgboost_regressor.py` |
| [`limen.sfd.reference_architecture.XGBoostRegressor.train`](Public-API.md#limensfdreference_architecturexgboostregressortrain) | `signature` | `limen/sfd/reference_architecture/xgboost_regressor.py` |
| [`limen.sfd.reference_architecture.direction_sizing`](Public-API.md#limensfdreference_architecturedirection_sizing) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.direction_sizing.ComponentEstimator`](Public-API.md#limensfdreference_architecturedirection_sizingcomponentestimator) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.direction_sizing.ComponentEstimator.fit`](Public-API.md#limensfdreference_architecturedirection_sizingcomponentestimatorfit) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.direction_sizing.ComponentEstimator.predict`](Public-API.md#limensfdreference_architecturedirection_sizingcomponentestimatorpredict) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.direction_sizing.ComponentFactory`](Public-API.md#limensfdreference_architecturedirection_sizingcomponentfactory) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.direction_sizing.ComponentFactory.__call__`](Public-API.md#limensfdreference_architecturedirection_sizingcomponentfactory__call__) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.direction_sizing.DirectionSizingModel`](Public-API.md#limensfdreference_architecturedirectionsizingmodel) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.direction_sizing.DirectionSizingModel.evaluate`](Public-API.md#limensfdreference_architecturedirectionsizingmodelevaluate) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.direction_sizing.DirectionSizingModel.predict`](Public-API.md#limensfdreference_architecturedirectionsizingmodelpredict) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.direction_sizing.DirectionSizingModel.train`](Public-API.md#limensfdreference_architecturedirectionsizingmodeltrain) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.direction_sizing.FoldFeatures`](Public-API.md#limensfdreference_architecturedirection_sizingfoldfeatures) | `signature` | `limen/experiment/_fold_preparation.py` |
| [`limen.sfd.reference_architecture.direction_sizing.FoldPreparation`](Public-API.md#limensfdreference_architecturedirection_sizingfoldpreparation) | `signature` | `limen/experiment/_fold_preparation.py` |
| [`limen.sfd.reference_architecture.direction_sizing.FoldPreparation.fit_transform`](Public-API.md#limensfdreference_architecturedirection_sizingfoldpreparationfit_transform) | `signature` | `limen/experiment/_fold_preparation.py` |
| [`limen.sfd.reference_architecture.direction_sizing.direction_sizing`](Public-API.md#limensfdreference_architecturedirection_sizing) | `signature` | `limen/sfd/reference_architecture/direction_sizing.py` |
| [`limen.sfd.reference_architecture.dlinear_regressor`](Public-API.md#limensfdreference_architecturedlinear_regressor) | `signature` | `limen/sfd/reference_architecture/dlinear_regressor.py` |
| [`limen.sfd.reference_architecture.lightgbm_binary`](Public-API.md#limensfdreference_architecturelightgbm_binary) | `signature` | `limen/sfd/reference_architecture/lightgbm_binary.py` |
| [`limen.sfd.reference_architecture.logreg_binary`](Experiment-Manifest.md#with_reference_architecturearchitecture_function) | `signature` | `limen/sfd/reference_architecture/logreg_binary.py` |
| [`limen.sfd.reference_architecture.meta_allocation.MetaAllocationModel`](Public-API.md#limensfdreference_architecturemeta_allocationmetaallocationmodel) | `signature` | `limen/sfd/reference_architecture/meta_allocation.py` |
| [`limen.sfd.reference_architecture.meta_allocation.MetaAllocationModel.evaluate`](Public-API.md#limensfdreference_architecturemeta_allocationmetaallocationmodelevaluate) | `signature` | `limen/sfd/reference_architecture/meta_allocation.py` |
| [`limen.sfd.reference_architecture.meta_allocation.MetaAllocationModel.predict`](Public-API.md#limensfdreference_architecturemeta_allocationmetaallocationmodelpredict) | `signature` | `limen/sfd/reference_architecture/meta_allocation.py` |
| [`limen.sfd.reference_architecture.meta_allocation.MetaAllocationModel.train`](Public-API.md#limensfdreference_architecturemeta_allocationmetaallocationmodeltrain) | `signature` | `limen/sfd/reference_architecture/meta_allocation.py` |
| [`limen.sfd.reference_architecture.meta_allocation.meta_allocation`](Public-API.md#limensfdreference_architecturemeta_allocationmeta_allocation) | `signature` | `limen/sfd/reference_architecture/meta_allocation.py` |
| [`limen.sfd.reference_architecture.random_binary`](Public-API.md#limensfdreference_architecturerandom_binary) | `signature` | `limen/sfd/reference_architecture/random_binary.py` |
| [`limen.sfd.reference_architecture.ridge_regressor`](Reference-Architecture.md#function-wrappers-versus-classes) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.ridge_regressor.RidgeRegressor`](Public-API.md#limensfdreference_architectureridgeregressor) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.ridge_regressor.RidgeRegressor.__init__`](Public-API.md#limensfdreference_architectureridgeregressor__init__) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.ridge_regressor.RidgeRegressor.evaluate`](Public-API.md#limensfdreference_architectureridgeregressorevaluate) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.ridge_regressor.RidgeRegressor.predict`](Public-API.md#limensfdreference_architectureridgeregressorpredict) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.ridge_regressor.RidgeRegressor.train`](Public-API.md#limensfdreference_architectureridgeregressortrain) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.ridge_regressor.ridge_regressor`](Reference-Architecture.md#function-wrappers-versus-classes) | `signature` | `limen/sfd/reference_architecture/ridge_regressor.py` |
| [`limen.sfd.reference_architecture.rule_based`](Public-API.md#limensfdreference_architecturerule_based) | `signature` | `limen/sfd/reference_architecture/rule_based.py` |
| [`limen.sfd.reference_architecture.tabpfn_binary`](Public-API.md#limensfdreference_architecturetabpfn_binary) | `signature` | `limen/sfd/reference_architecture/tabpfn_binary.py` |
| [`limen.sfd.reference_architecture.xgboost_regressor`](Public-API.md#limensfdreference_architecturexgboost_regressor) | `signature` | `limen/sfd/reference_architecture/xgboost_regressor.py` |
| [`limen.sfd.reference_architecture.xgboost_regressor.XGBoostRegressor`](Public-API.md#limensfdreference_architecturexgboostregressor) | `signature` | `limen/sfd/reference_architecture/xgboost_regressor.py` |
| [`limen.sfd.reference_architecture.xgboost_regressor.XGBoostRegressor.evaluate`](Public-API.md#limensfdreference_architecturexgboostregressorevaluate) | `signature` | `limen/sfd/reference_architecture/xgboost_regressor.py` |
| [`limen.sfd.reference_architecture.xgboost_regressor.XGBoostRegressor.predict`](Public-API.md#limensfdreference_architecturexgboostregressorpredict) | `signature` | `limen/sfd/reference_architecture/xgboost_regressor.py` |
| [`limen.sfd.reference_architecture.xgboost_regressor.XGBoostRegressor.train`](Public-API.md#limensfdreference_architecturexgboostregressortrain) | `signature` | `limen/sfd/reference_architecture/xgboost_regressor.py` |
| [`limen.sfd.reference_architecture.xgboost_regressor.xgboost_regressor`](Public-API.md#limensfdreference_architecturexgboost_regressor) | `signature` | `limen/sfd/reference_architecture/xgboost_regressor.py` |
| [`limen.sfd.rule_based.RuleBasedConfig`](Public-API.md#limensfdrule_basedrulebasedconfig) | `signature` | `limen/sfd/rule_based/config.py` |
| [`limen.sfd.rule_based.build_predicate`](Public-API.md#limensfdrule_basedbuild_predicate) | `signature` | `limen/sfd/rule_based/predicates.py` |
| [`limen.sfd.rule_based.common_strategies.STRATEGY_NAMES`](Public-API.md#limensfdrule_basedcommon_strategiesstrategy_names) | `definition` | `limen/sfd/rule_based/common_strategies.py` |
| [`limen.sfd.rule_based.common_strategies.STRATEGY_PARAMETERS`](Public-API.md#limensfdrule_basedcommon_strategiesstrategy_parameters) | `definition` | `limen/sfd/rule_based/common_strategies.py` |
| [`limen.sfd.rule_based.common_strategies.strategy_signals`](Public-API.md#limensfdrule_basedcommon_strategiesstrategy_signals) | `signature` | `limen/sfd/rule_based/common_strategies.py` |
| `limen.sfd.xgboost_regressor` | `module` | `limen/sfd/foundational_sfd/xgboost_regressor.py` |
| `limen.targets` | `module` | `limen/targets/__init__.py` |
| [`limen.targets.EmaBreakoutTarget`](Targets.md#limentargetsemabreakouttarget) | `signature` | `limen/targets/ema_breakout.py` |
| [`limen.targets.EmaBreakoutTarget.__init__`](Targets.md#limentargetsemabreakouttarget__init__) | `signature` | `limen/targets/ema_breakout.py` |
| [`limen.targets.EmaBreakoutTarget.transform`](Targets.md#limentargetsemabreakouttargettransform) | `signature` | `limen/targets/ema_breakout.py` |
| [`limen.targets.ExitQualityTarget`](Targets.md#limentargetsexitqualitytarget) | `signature` | `limen/targets/exit_quality.py` |
| [`limen.targets.ExitQualityTarget.__init__`](Targets.md#limentargetsexitqualitytarget__init__) | `signature` | `limen/targets/exit_quality.py` |
| [`limen.targets.ExitQualityTarget.transform`](Targets.md#limentargetsexitqualitytargettransform) | `signature` | `limen/targets/exit_quality.py` |
| [`limen.targets.ForwardBreakoutTarget`](Targets.md#limentargetsforwardbreakouttarget) | `signature` | `limen/targets/forward_breakout.py` |
| [`limen.targets.ForwardBreakoutTarget.__init__`](Targets.md#limentargetsforwardbreakouttarget__init__) | `signature` | `limen/targets/forward_breakout.py` |
| [`limen.targets.ForwardBreakoutTarget.transform`](Targets.md#limentargetsforwardbreakouttargettransform) | `signature` | `limen/targets/forward_breakout.py` |
| [`limen.targets.ForwardVolNormalizedReturnTarget`](Targets.md#limentargetsforwardvolnormalizedreturntarget) | `signature` | `limen/targets/forward_vol_normalized_return.py` |
| [`limen.targets.ForwardVolNormalizedReturnTarget.__init__`](Targets.md#limentargetsforwardvolnormalizedreturntarget__init__) | `signature` | `limen/targets/forward_vol_normalized_return.py` |
| [`limen.targets.ForwardVolNormalizedReturnTarget.transform`](Targets.md#limentargetsforwardvolnormalizedreturntargettransform) | `signature` | `limen/targets/forward_vol_normalized_return.py` |
| [`limen.targets.IdentityTarget`](Targets.md#limentargetsidentitytarget) | `signature` | `limen/targets/identity.py` |
| [`limen.targets.IdentityTarget.__init__`](Targets.md#limentargetsidentitytarget__init__) | `signature` | `limen/targets/identity.py` |
| [`limen.targets.IdentityTarget.transform`](Targets.md#limentargetsidentitytargettransform) | `signature` | `limen/targets/identity.py` |
| [`limen.targets.NextBarDownTarget`](Targets.md#limentargetsnextbardowntarget) | `signature` | `limen/targets/next_bar_down.py` |
| [`limen.targets.NextBarDownTarget.__init__`](Targets.md#limentargetsnextbardowntarget__init__) | `signature` | `limen/targets/next_bar_down.py` |
| [`limen.targets.NextBarDownTarget.transform`](Targets.md#limentargetsnextbardowntargettransform) | `signature` | `limen/targets/next_bar_down.py` |
| [`limen.targets.NextBarUpTarget`](Targets.md#limentargetsnextbaruptarget) | `signature` | `limen/targets/next_bar_up.py` |
| [`limen.targets.NextBarUpTarget.__init__`](Targets.md#limentargetsnextbaruptarget__init__) | `signature` | `limen/targets/next_bar_up.py` |
| [`limen.targets.NextBarUpTarget.transform`](Targets.md#limentargetsnextbaruptargettransform) | `signature` | `limen/targets/next_bar_up.py` |
| [`limen.targets.NextReturnTarget`](Targets.md#limentargetsnextreturntarget) | `signature` | `limen/targets/next_return.py` |
| [`limen.targets.NextReturnTarget.__init__`](Targets.md#limentargetsnextreturntarget__init__) | `signature` | `limen/targets/next_return.py` |
| [`limen.targets.NextReturnTarget.transform`](Targets.md#limentargetsnextreturntargettransform) | `signature` | `limen/targets/next_return.py` |
| [`limen.targets.OutcomeLabels`](Targets.md#limentargetsoutcomelabels) | `signature` | `limen/targets/trade_outcome.py` |
| [`limen.targets.QuantileBinaryTarget`](Targets.md#limentargetsquantilebinarytarget) | `signature` | `limen/targets/quantile_binary.py` |
| [`limen.targets.QuantileBinaryTarget.__init__`](Targets.md#limentargetsquantilebinarytarget__init__) | `signature` | `limen/targets/quantile_binary.py` |
| [`limen.targets.QuantileBinaryTarget.transform`](Targets.md#limentargetsquantilebinarytargettransform) | `signature` | `limen/targets/quantile_binary.py` |
| [`limen.targets.RandomBinaryTarget`](Targets.md#limentargetsrandombinarytarget) | `signature` | `limen/targets/random_binary.py` |
| [`limen.targets.RandomBinaryTarget.__init__`](Targets.md#limentargetsrandombinarytarget__init__) | `signature` | `limen/targets/random_binary.py` |
| [`limen.targets.RandomBinaryTarget.transform`](Targets.md#limentargetsrandombinarytargettransform) | `signature` | `limen/targets/random_binary.py` |
| [`limen.targets.RiskRewardRatioTarget`](Targets.md#limentargetsriskrewardratiotarget) | `signature` | `limen/targets/risk_reward_ratio.py` |
| [`limen.targets.RiskRewardRatioTarget.__init__`](Targets.md#limentargetsriskrewardratiotarget__init__) | `signature` | `limen/targets/risk_reward_ratio.py` |
| [`limen.targets.RiskRewardRatioTarget.transform`](Targets.md#limentargetsriskrewardratiotargettransform) | `signature` | `limen/targets/risk_reward_ratio.py` |
| [`limen.targets.ThresholdBinaryTarget`](Targets.md#limentargetsthresholdbinarytarget) | `signature` | `limen/targets/threshold_binary.py` |
| [`limen.targets.ThresholdBinaryTarget.__init__`](Targets.md#limentargetsthresholdbinarytarget__init__) | `signature` | `limen/targets/threshold_binary.py` |
| [`limen.targets.ThresholdBinaryTarget.transform`](Targets.md#limentargetsthresholdbinarytargettransform) | `signature` | `limen/targets/threshold_binary.py` |
| [`limen.targets.TradeOutcomeTarget`](Targets.md#limentargetstradeoutcometarget) | `signature` | `limen/targets/trade_outcome.py` |
| [`limen.targets.TradeOutcomeTarget.__init__`](Targets.md#limentargetstradeoutcometarget__init__) | `signature` | `limen/targets/trade_outcome.py` |
| [`limen.targets.TradeOutcomeTarget.outcomes`](Targets.md#limentargetstradeoutcometargetoutcomes) | `signature` | `limen/targets/trade_outcome.py` |
| [`limen.targets.TradeOutcomeTarget.transform`](Targets.md#limentargetstradeoutcometargettransform) | `signature` | `limen/targets/trade_outcome.py` |
| [`limen.targets.TradeTargetContext`](Targets.md#limentargetstradetargetcontext) | `signature` | `limen/targets/trade_outcome.py` |
| [`limen.targets.TradelineLongBinaryTarget`](Targets.md#limentargetstradelinelongbinarytarget) | `signature` | `limen/targets/tradeline_long_binary.py` |
| [`limen.targets.TradelineLongBinaryTarget.__init__`](Targets.md#limentargetstradelinelongbinarytarget__init__) | `signature` | `limen/targets/tradeline_long_binary.py` |
| [`limen.targets.TradelineLongBinaryTarget.transform`](Targets.md#limentargetstradelinelongbinarytargettransform) | `signature` | `limen/targets/tradeline_long_binary.py` |
| [`limen.targets.TripleBarrierTarget`](Targets.md#limentargetstriplebarriertarget) | `signature` | `limen/targets/triple_barrier.py` |
| [`limen.targets.TripleBarrierTarget.__init__`](Targets.md#limentargetstriplebarriertarget__init__) | `signature` | `limen/targets/triple_barrier.py` |
| [`limen.targets.TripleBarrierTarget.transform`](Targets.md#limentargetstriplebarriertargettransform) | `signature` | `limen/targets/triple_barrier.py` |
| [`limen.targets.VolNormalizedReturnTarget`](Targets.md#limentargetsvolnormalizedreturntarget) | `signature` | `limen/targets/vol_normalized_return.py` |
| [`limen.targets.VolNormalizedReturnTarget.__init__`](Targets.md#limentargetsvolnormalizedreturntarget__init__) | `signature` | `limen/targets/vol_normalized_return.py` |
| [`limen.targets.VolNormalizedReturnTarget.transform`](Targets.md#limentargetsvolnormalizedreturntargettransform) | `signature` | `limen/targets/vol_normalized_return.py` |
| [`limen.targets.trade_outcome.OutcomeLabels`](Targets.md#limentargetsoutcomelabels) | `signature` | `limen/targets/trade_outcome.py` |
| [`limen.targets.trade_outcome.TradeOutcomeTarget`](Targets.md#limentargetstradeoutcometarget) | `signature` | `limen/targets/trade_outcome.py` |
| [`limen.targets.trade_outcome.TradeOutcomeTarget.__init__`](Targets.md#limentargetstradeoutcometarget__init__) | `signature` | `limen/targets/trade_outcome.py` |
| [`limen.targets.trade_outcome.TradeOutcomeTarget.outcomes`](Targets.md#limentargetstradeoutcometargetoutcomes) | `signature` | `limen/targets/trade_outcome.py` |
| [`limen.targets.trade_outcome.TradeOutcomeTarget.transform`](Targets.md#limentargetstradeoutcometargettransform) | `signature` | `limen/targets/trade_outcome.py` |
| [`limen.targets.trade_outcome.TradeTargetContext`](Targets.md#limentargetstradetargetcontext) | `signature` | `limen/targets/trade_outcome.py` |
| `limen.transforms` | `module` | `limen/transforms/__init__.py` |
| [`limen.transforms.mad_transform`](Transforms.md#limentransformsmad_transform) | `signature` | `limen/transforms/mad_transform.py` |
| [`limen.transforms.quantile_trim_transform`](Transforms.md#limentransformsquantile_trim_transform) | `signature` | `limen/transforms/quantile_trim_transform.py` |
| [`limen.transforms.shift_column_transform`](Transforms.md#limentransformsshift_column_transform) | `signature` | `limen/transforms/shift_column_transform.py` |
| [`limen.transforms.winsorize_transform`](Transforms.md#limentransformswinsorize_transform) | `signature` | `limen/transforms/winsorize_transform.py` |
| [`limen.transforms.zscore_transform`](Transforms.md#limentransformszscore_transform) | `signature` | `limen/transforms/zscore_transform.py` |
| `limen.utils` | `module` | `limen/utils/__init__.py` |
| [`limen.utils.AdfResult`](Utilities.md#limenutilsadfresult) | `signature` | `limen/utils/adf_test.py` |
| [`limen.utils.ParamSpace`](Utilities.md#limenutilsparamspace) | `signature` | `limen/utils/param_space.py` |
| [`limen.utils.ParamSpace.__init__`](Utilities.md#limenutilsparamspace__init__) | `signature` | `limen/utils/param_space.py` |
| [`limen.utils.ParamSpace.generate`](Utilities.md#limenutilsparamspacegenerate) | `signature` | `limen/utils/param_space.py` |
| [`limen.utils.adf_test`](Fractional-Differentiation.md#finding-the-minimum-d) | `signature` | `limen/utils/adf_test.py` |
| [`limen.utils.binary_metrics`](Utilities.md#limenutilsbinary_metrics) | `signature` | `limen/metrics/binary_metrics.py` |
| [`limen.utils.confidence_filtering_system`](Utilities.md#limenutilsconfidence_filtering_system) | `signature` | `limen/utils/confidence_filtering_system.py` |
| [`limen.utils.continuous_metrics`](Utilities.md#limenutilscontinuous_metrics) | `signature` | `limen/metrics/continuous_metrics.py` |
| [`limen.utils.data_dict_to_numpy`](Utilities.md#limenutilsdata_dict_to_numpy) | `signature` | `limen/utils/data_dict_to_numpy.py` |
| [`limen.utils.filter_lines_by_quantile`](Utilities.md#limenutilsfilter_lines_by_quantile) | `signature` | `limen/utils/filter_lines_by_quantile.py` |
| [`limen.utils.find_price_lines`](Utilities.md#limenutilsfind_price_lines) | `signature` | `limen/utils/find_price_lines.py` |
| [`limen.utils.find_price_lines.LineArrays`](Public-API.md#limenutilsfind_price_lineslinearrays) | `definition` | `limen/utils/_computation_cache.py` |
| [`limen.utils.find_price_lines.find_price_lines`](Utilities.md#limenutilsfind_price_lines) | `signature` | `limen/utils/find_price_lines.py` |
| [`limen.utils.find_price_lines.price_line_arrays`](Public-API.md#limenutilsfind_price_linesprice_line_arrays) | `signature` | `limen/utils/find_price_lines.py` |
| [`limen.utils.format_report_footer`](Utilities.md#limenutilsformat_report_footer) | `signature` | `limen/utils/reporting.py` |
| [`limen.utils.format_report_header`](Utilities.md#limenutilsformat_report_header) | `signature` | `limen/utils/reporting.py` |
| [`limen.utils.format_report_section`](Utilities.md#limenutilsformat_report_section) | `signature` | `limen/utils/reporting.py` |
| [`limen.utils.safe_ovr_auc`](Utilities.md#limenutilssafe_ovr_auc) | `signature` | `limen/metrics/safe_ovr_auc.py` |
| [`limen.yaml.CompiledSFD`](Public-API.md#limenyamlcompiledsfd) | `signature` | `limen/yaml/compiler.py` |
| [`limen.yaml.CompiledSFD.__init__`](Public-API.md#limenyamlcompiledsfd__init__) | `signature` | `limen/yaml/compiler.py` |
| [`limen.yaml.CompiledSFD.manifest`](Public-API.md#limenyamlcompiledsfdmanifest) | `signature` | `limen/yaml/compiler.py` |
| [`limen.yaml.CompiledSFD.params`](Public-API.md#limenyamlcompiledsfdparams) | `signature` | `limen/yaml/compiler.py` |
| [`limen.yaml.GitError`](Public-API.md#limenyamlgiterror) | `signature` | `limen/yaml/errors.py` |
| [`limen.yaml.GitError.__init__`](Public-API.md#limenyamlgiterror__init__) | `signature` | `limen/yaml/errors.py` |
| [`limen.yaml.ProfileResult`](Public-API.md#limenyamlprofileresult) | `signature` | `limen/yaml/profiler.py` |
| [`limen.yaml.ResolutionError`](Public-API.md#limenyamlresolutionerror) | `signature` | `limen/yaml/errors.py` |
| [`limen.yaml.ResolutionError.__init__`](Public-API.md#limenyamlresolutionerror__init__) | `signature` | `limen/yaml/errors.py` |
| [`limen.yaml.VERSION`](Public-API.md#limenyamlversion) | `definition` | `limen/yaml/schema.py` |
| [`limen.yaml.ValidationError`](Public-API.md#limenyamlvalidationerror) | `signature` | `limen/yaml/errors.py` |
| [`limen.yaml.ValidationError.__init__`](Public-API.md#limenyamlvalidationerror__init__) | `signature` | `limen/yaml/errors.py` |
| [`limen.yaml.ValidationResult`](Public-API.md#limenyamlvalidationresult) | `signature` | `limen/yaml/validator.py` |
| [`limen.yaml.YAMLError`](Public-API.md#limenyamlyamlerror) | `signature` | `limen/yaml/errors.py` |
| [`limen.yaml.build_manifest`](Public-API.md#limenyamlbuild_manifest) | `signature` | `limen/yaml/compiler.py` |
| [`limen.yaml.build_search_strategy`](Public-API.md#limenyamlbuild_search_strategy) | `signature` | `limen/yaml/compiler.py` |
| [`limen.yaml.find_project_root`](Public-API.md#limenyamlfind_project_root) | `signature` | `limen/yaml/config.py` |
| [`limen.yaml.get_store_path`](Public-API.md#limenyamlget_store_path) | `signature` | `limen/yaml/config.py` |
| [`limen.yaml.make_covering_array`](Public-API.md#limenyamlmake_covering_array) | `signature` | `limen/yaml/profiler.py` |
| [`limen.yaml.parse`](Public-API.md#limenyamlparse) | `signature` | `limen/yaml/parser.py` |
| [`limen.yaml.profile`](Public-API.md#limenyamlprofile) | `signature` | `limen/yaml/profiler.py` |
| [`limen.yaml.read_limen_toml`](Public-API.md#limenyamlread_limen_toml) | `signature` | `limen/yaml/config.py` |
| [`limen.yaml.resolve`](Public-API.md#limenyamlresolve) | `signature` | `limen/yaml/resolver.py` |
| [`limen.yaml.resolve_manifest_uri`](Public-API.md#limenyamlresolve_manifest_uri) | `signature` | `limen/yaml/store.py` |
| [`limen.yaml.validate`](Public-API.md#limenyamlvalidate) | `signature` | `limen/yaml/validator.py` |

## Additional component contracts

These source-bound entries supplement the topic references. They serve direct Python use and optional native extension or maintenance. Compiled Research continues through YAML and CLI.

### `limen.Cohort`

<!-- reference: public-limen-cohort source=limen/cohort/cohort.py region=signature:Cohort role=signature -->
```python
class Cohort:
    _REASON_PRIORITY: ClassVar[dict[PredictionReason, int]] = {'inside-training-window': 2, 'null-features': 1, 'warm-up': 0}
    member_ids: list[tuple[str | None, str] | tuple[str, str, int, str]]
    architecture_id: str
    supports_probabilities: bool | None
    aggregation_mode: str | None
```

Combine explicitly selected, reconstructed sensors for inference.

### `limen.Cohort.__call__`

<!-- reference: public-limen-cohort---call-- source=limen/cohort/cohort.py region=signature:Cohort.__call__ role=signature -->
```python
def __call__(self, raw_klines: pl.DataFrame) -> list[BarPrediction]:
    ...
```

### `limen.HistoricalData`

<!-- reference: public-limen-historicaldata source=limen/data/historical_data.py region=signature:HistoricalData role=signature -->
```python
class HistoricalData:
    DEFAULT_SPOT_KLINES_DATASET_REPO: Final[str] = _DEFAULT_SPOT_DATASET_REPO
    DEFAULT_SPOT_DOLLAR_KLINES_DATASET_REPO: Final[str] = _DEFAULT_SPOT_DOLLAR_DATASET_REPO
```

Stateful file-backed data access surface for Limen.

### `limen.HistoricalData.__init__`

<!-- reference: public-limen-historicaldata---init-- source=limen/data/historical_data.py region=signature:HistoricalData.__init__ role=signature -->
```python
def __init__(self) -> None:
    ...
```

### `limen.Log`

<!-- reference: public-limen-log source=limen/log/log.py region=signature:Log role=signature -->
```python
class Log:
    ...
```

Log object for storing and analyzing experiment results.

### `limen.MLManifest`

<!-- reference: public-limen-mlmanifest source=limen/experiment/manifest_core.py region=signature:MLManifest role=signature -->
```python
@dataclass
class MLManifest(Manifest):
    scaler: FittedTransformEntry | None = None
    ablation_config: AblationConfig | None = None
    pca_compression_config: PCACompressionConfig | None = None
    data_dict_extension: Callable[..., dict[str, Any]] | None = None
    prediction_calibration_config: CalibrationConfig | None = None
    decoder_lookback: int = 1
    strict_mode: bool = False
    objective: ObjectiveConfig | None = None
    feature_shims: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
```

Manifest for ML pipelines with scaler, ablation, and calibration support.

### `limen.MLManifest.prepare_data`

<!-- reference: public-limen-mlmanifest-prepare-data source=limen/experiment/manifest_core.py region=signature:MLManifest.prepare_data role=signature -->
```python
@override
def prepare_data(self, raw_data: pl.DataFrame, round_params: dict[str, Any]) -> dict[str, Any]:
    ...
```

### `limen.MLManifest.run_model`

<!-- reference: public-limen-mlmanifest-run-model source=limen/experiment/manifest_core.py region=signature:MLManifest.run_model role=signature -->
```python
@override
def run_model(self, data: dict[str, Any], round_params: dict[str, Any]) -> dict[str, Any]:
    ...
```

### `limen.MLManifest.sensor_input_prep`

<!-- reference: public-limen-mlmanifest-sensor-input-prep source=limen/experiment/manifest_core.py region=signature:MLManifest.sensor_input_prep role=signature -->
```python
def sensor_input_prep(self, raw_klines: pl.DataFrame, fitted_params: dict[str, Any], round_params: dict[str, Any]) -> tuple[pl.DataFrame, int]:
    ...
```

Prepare live features using fitted target exclusions, scaler and PCA.

Preserve null rows and return the leading indicator warm-up count.

Fitted target metadata takes precedence over static class declarations.

### `limen.MLManifest.set_objective`

<!-- reference: public-limen-mlmanifest-set-objective source=limen/experiment/manifest_core.py region=signature:MLManifest.set_objective role=signature -->
```python
def set_objective(self, metric: str='backtest_total_return', direction: str='maximize') -> 'MLManifest':
    ...
```

### `limen.Manifest`

<!-- reference: public-limen-manifest source=limen/experiment/manifest_core.py region=signature:Manifest role=signature -->
```python
@dataclass
class Manifest:
    data_source_config: DataSourceConfig | None = None
    test_data_source_config: DataSourceConfig | None = None
    pre_split_data_selector: PipelineStep | None = None
    split_config: tuple[int, int, int] = (8, 1, 2)
    split_dates: tuple[date | Any, date | Any, date | Any, date | Any, date | Any, date | Any] | None = None
    split_walk_forward: WalkForwardConfig | None = None
    acceptance: dict[str, float] | None = None
    _walk_forward_fold: int | None = field(default=None, repr=False)
    val_predict_guard: bool = True
    test_predict_guard: bool = True
    bar_formation: PipelineStep | None = None
    required_bar_columns: list[str] = field(default_factory=list[str])
    feature_transforms: list[TransformEntry] = field(default_factory=list[TransformEntry])
    target_column: str | None = None
    target_class_config: TargetClassConfig | None = None
    architecture_function: Callable[..., dict[str, Any]] | None = None
    architecture_params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])
    metrics_params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])
    backtest_config: BacktestConfig | None = None
    allocation_config: _allocation_stage.AllocationConfig | None = None
```

Base manifest with shared data pipeline configuration for Loop experiments.

### `limen.Manifest.compute_test_bars`

<!-- reference: public-limen-manifest-compute-test-bars source=limen/experiment/manifest_core.py region=signature:Manifest.compute_test_bars role=signature -->
```python
@_holdout.preparation
def compute_test_bars(self, raw_data: pl.DataFrame, round_params: dict[str, Any]) -> pl.DataFrame:
    ...
```

Compute test split bar data from raw data using manifest bar formation configuration.

NOTE: Used by Log system to reconstruct the same test bar data that was used in training.

Args:

raw_data (pl.DataFrame): Raw input dataset

round_params (Dict[str, Any]): Parameter values for current round

Returns:

pl.DataFrame: Bar-formed test split data

### `limen.Manifest.fetch_data`

<!-- reference: public-limen-manifest-fetch-data source=limen/experiment/manifest_core.py region=signature:Manifest.fetch_data role=signature -->
```python
def fetch_data(self) -> pl.DataFrame:
    ...
```

Fetch data using configured data source.

### `limen.Manifest.fetch_test_data`

<!-- reference: public-limen-manifest-fetch-test-data source=limen/experiment/manifest_core.py region=signature:Manifest.fetch_test_data role=signature -->
```python
def fetch_test_data(self) -> pl.DataFrame:
    ...
```

Fetch data using configured test data source.

### `limen.Manifest.prepare_data`

<!-- reference: public-limen-manifest-prepare-data source=limen/experiment/manifest_core.py region=signature:Manifest.prepare_data role=signature -->
```python
def prepare_data(self, raw_data: pl.DataFrame, round_params: dict[str, Any]) -> dict[str, Any]:
    ...
```

Interface method — implemented by MLManifest and RuleBasedManifest.

Raises:

NotImplementedError: Always. Construct MLManifest or RuleBasedManifest instead of Manifest directly.

### `limen.Manifest.resolve_backtest_config`

<!-- reference: public-limen-manifest-resolve-backtest-config source=limen/experiment/manifest_core.py region=signature:Manifest.resolve_backtest_config role=signature -->
```python
def resolve_backtest_config(self, round_params: Mapping[str, object]) -> dict[str, float | None]:
    ...
```

Resolve all backtest fields; an absent configuration returns an empty mapping.

### `limen.Manifest.resolve_model_kwargs`

<!-- reference: public-limen-manifest-resolve-model-kwargs source=limen/experiment/manifest_core.py region=signature:Manifest.resolve_model_kwargs role=signature -->
```python
def resolve_model_kwargs(self, round_params: dict[str, Any]) -> dict[str, Any]:
    ...
```

Resolve model function kwargs from round_params using signature inspection.

Maps round_params keys to model function parameters, falling back

to defaults for unspecified parameters.

Args:

round_params (dict[str, Any]): Parameter values for current round

Returns:

dict[str, Any]: Keyword arguments for the model function

Raises:

ValueError: If model function is not configured or required parameters

are missing from round_params

### `limen.Manifest.resolve_trade_policy`

<!-- reference: public-limen-manifest-resolve-trade-policy source=limen/experiment/manifest_core.py region=signature:Manifest.resolve_trade_policy role=signature -->
```python
def resolve_trade_policy(self, round_params: Mapping[str, object]) -> TradePolicy | None:
    ...
```

### `limen.Manifest.run_model`

<!-- reference: public-limen-manifest-run-model source=limen/experiment/manifest_core.py region=signature:Manifest.run_model role=signature -->
```python
def run_model(self, data: dict[str, Any], round_params: dict[str, Any]) -> dict[str, Any]:
    ...
```

Execute model training and evaluation using configured functions.

Args:

data (dict): Prepared data dictionary

round_params (dict[str, Any]): Parameter values for current round

Returns:

dict: Results including predictions, metrics, and optional extras

Raises:

ValueError: If required model function parameters are missing from round_params

NOTE: Auto-maps parameters from round_params to model function signature.

Parameters in round_params override model function defaults.

Parameters not in round_params use model function defaults.

Required parameters (no defaults) must be in round_params.

### `limen.Manifest.with_allocation`

<!-- reference: public-limen-manifest-with-allocation source=limen/experiment/manifest_core.py region=signature:Manifest.with_allocation role=signature -->
```python
def with_allocation(self: _TManifest, method: Callable[..., dict[str, object]], params: Mapping[str, object] | None=None) -> _TManifest:
    ...
```

### `limen.ReconstructionError`

<!-- reference: public-limen-reconstructionerror source=limen/inference/errors.py region=signature:ReconstructionError role=signature -->
```python
class ReconstructionError(Exception):
    ...
```

Raised when metric validation detects deviation beyond tolerance.

### `limen.RuleBasedManifest`

<!-- reference: public-limen-rulebasedmanifest source=limen/experiment/manifest_core.py region=signature:RuleBasedManifest role=signature -->
```python
@dataclass
class RuleBasedManifest(Manifest):
    strategy: 'RuleBasedConfig | None' = field(default=None, init=False, repr=False)
    objective: ObjectiveConfig | None = None
```

Manifest for rule-based pipelines with predicate conditions and entry signals.

### `limen.RuleBasedManifest.prepare_data`

<!-- reference: public-limen-rulebasedmanifest-prepare-data source=limen/experiment/manifest_core.py region=signature:RuleBasedManifest.prepare_data role=signature -->
```python
@override
@_holdout.preparation
def prepare_data(self, raw_data: pl.DataFrame, round_params: dict[str, Any]) -> dict[str, Any]:
    ...
```

Compute final data dictionary from raw data using the rule-based pipeline.

Args:

raw_data (pl.DataFrame): Raw input dataset

round_params (Dict[str, Any]): Parameter values for current round

Returns:

dict: Final data dictionary ready for model training

### `limen.RuleBasedManifest.set_objective`

<!-- reference: public-limen-rulebasedmanifest-set-objective source=limen/experiment/manifest_core.py region=signature:RuleBasedManifest.set_objective role=signature -->
```python
def set_objective(self, metric: str='backtest_total_return', direction: str='maximize') -> 'RuleBasedManifest':
    ...
```

### `limen.RuleBasedManifest.sensor_input_prep`

<!-- reference: public-limen-rulebasedmanifest-sensor-input-prep source=limen/experiment/manifest_core.py region=signature:RuleBasedManifest.sensor_input_prep role=signature -->
```python
def sensor_input_prep(self, raw_klines: pl.DataFrame, fitted_params: dict[str, object], round_params: dict[str, object]) -> tuple[pl.DataFrame, int]:
    ...
```

Prepare rule features without fitting targets or recomputing evaluation labels.

### `limen.Sensor`

<!-- reference: public-limen-sensor source=limen/inference/sensor.py region=signature:Sensor role=signature -->
```python
class Sensor:
    _trade_contract: Mapping[str, JsonValue] | None = None
    _trade_digest: str | None = None
```

Inference wrapper around a trained YAML model for live bar-by-bar prediction.

### `limen.Sensor.__init__`

<!-- reference: public-limen-sensor---init-- source=limen/inference/sensor.py region=signature:Sensor.__init__ role=signature -->
```python
def __init__(self, yaml_reference: dict[str, Any], model: ReferenceModel, fitted_params: dict[str, Any], round_params: dict[str, Any], permutation_id: str | None=None, manifest_id: str | None=None, *, trade_contract: Mapping[str, JsonValue] | None=None) -> None:
    ...
```

### `limen.Sensor.predict_all`

<!-- reference: public-limen-sensor-predict-all source=limen/inference/sensor.py region=signature:Sensor.predict_all role=signature -->
```python
def predict_all(self, raw_klines: pl.DataFrame) -> list[BarPrediction]:
    ...
```

Prepare raw klines and return one prediction per input bar.

Warm-up bars and bars inside the training window have prediction=None

with reason set. Invalid numeric outputs have reason='sensor-error'.

Valid bars have predictions populated. Output length

equals the post-bar-formation row count, not necessarily len(raw_klines).

Args:

raw_klines (pl.DataFrame): Raw klines from live feed, same schema as

the manifest data source

Returns:

list[BarPrediction]: One entry per bar in the post-bar-formation data.

Length equals len(raw_klines) when bar_type is 'base' (no bar

aggregation). When bar formation is active, length equals the

aggregated bar count, which is smaller than len(raw_klines).

Returns reason='sensor-error' on unexpected exceptions rather

than raising.

### `limen.Sensor.prediction_mode`

<!-- reference: public-limen-sensor-prediction-mode source=limen/inference/sensor.py region=signature:Sensor.prediction_mode role=signature -->
```python
@property
def prediction_mode(self) -> Literal['binary', 'target_exposure', 'python']:
    ...
```

### `limen.Sensor.round_params`

<!-- reference: public-limen-sensor-round-params source=limen/inference/sensor.py region=signature:Sensor.round_params role=signature -->
```python
@property
def round_params(self) -> dict[str, Any]:
    ...
```

### `limen.Sensor.trade_contract`

<!-- reference: public-limen-sensor-trade-contract source=limen/inference/sensor.py region=signature:Sensor.trade_contract role=signature -->
```python
@property
def trade_contract(self) -> Mapping[str, JsonValue] | None:
    ...
```

### `limen.Trainer`

<!-- reference: public-limen-trainer source=limen/inference/trainer.py region=signature:Trainer role=signature -->
```python
class Trainer:
    ...
```

Retrain selected permutations from a completed YAML experiment.

Validates permutations by re-running manifest.prepare_data() and

manifest.run_model(), comparing metrics against the original experiment

log. The validated model is used directly as the Sensor model.

### `limen.UniversalExperimentLoop`

<!-- reference: public-limen-universalexperimentloop source=limen/experiment/experiment_core.py region=signature:UniversalExperimentLoop role=signature -->
```python
class UniversalExperimentLoop:
    ...
```

### `limen.UniversalExperimentLoop.fold_results`

<!-- reference: public-limen-universalexperimentloop-fold-results source=limen/experiment/experiment_core.py region=signature:UniversalExperimentLoop.fold_results role=signature -->
```python
@property
def fold_results(self) -> pl.DataFrame:
    ...
```

### `limen.UniversalExperimentLoop.interrupted`

<!-- reference: public-limen-universalexperimentloop-interrupted source=limen/experiment/experiment_core.py region=signature:UniversalExperimentLoop.interrupted role=signature -->
```python
@property
def interrupted(self) -> bool:
    ...
```

### `limen.backtest.ExecutionResult`

<!-- reference: public-limen-backtest-executionresult source=limen/backtest/long_flat_strategy.py region=signature:ExecutionResult role=signature -->
```python
class ExecutionResult(NamedTuple):
    pos: npt.NDArray[np.float64]
    gross: npt.NDArray[np.float64]
    net: npt.NDArray[np.float64]
```

Per-bar execution arrays consumed by the snapshot ledger.

Attributes:

pos (npt.NDArray[np.float64]): Position held per bar (0 or 1 under the

all-in model; a strategy may return a deployed fraction).

gross (npt.NDArray[np.float64]): Per-bar gross return before costs.

net (npt.NDArray[np.float64]): Per-bar net return after entry and exit costs.

### `limen.backtest.backtest_snapshot.BACKTEST_SNAPSHOT_COLUMNS`

<!-- reference: public-limen-backtest-backtest-snapshot-backtest-snapshot-columns source=limen/backtest/backtest_snapshot.py region=definition:BACKTEST_SNAPSHOT_COLUMNS role=definition -->
```python
BACKTEST_SNAPSHOT_COLUMNS = ['edge_bps_p5', 'edge_bps_p50', 'edge_bps_p95', 'pnl_bps_p5', 'pnl_bps_p50', 'pnl_bps_p95', 'cost_bps_p5', 'cost_bps_p50', 'cost_bps_p95', 'drawdown_bps_p5', 'drawdown_bps_p50', 'drawdown_bps_p95', 'wins_per_bar', 'pnl_per_bar_bps', 'avg_win_bps', 'avg_loss_bps', 'cvar_95_pnl_bps', 'trades_per_bar', 'inventory_per_bar', 'cost_per_bar_bps']
```

### `limen.backtest.execution_events.OBSERVATION_COLUMNS`

<!-- reference: public-limen-backtest-execution-events-observation-columns source=limen/backtest/execution_events.py region=definition:OBSERVATION_COLUMNS role=definition -->
```python
OBSERVATION_COLUMNS: tuple[str, ...] = ('row_id', 'start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns', 'open', 'high', 'low', 'close')
```

### `limen.backtest.execution_events.SIGNAL_COLUMNS`

<!-- reference: public-limen-backtest-execution-events-signal-columns source=limen/backtest/execution_events.py region=definition:SIGNAL_COLUMNS role=definition -->
```python
SIGNAL_COLUMNS: tuple[str, ...] = ('row_id', 'available_at_ns', 'target')
```

### `limen.backtest.execution_events.execution_events`

<!-- reference: public-limen-backtest-execution-events-execution-events source=limen/backtest/execution_events.py region=signature:execution_events role=signature -->
```python
def execution_events(inputs: TradeInputs, policy: TradePolicy) -> tuple[ExecutionEvent, ...]:
    ...
```

### `limen.backtest.execution_events.validate_observations`

<!-- reference: public-limen-backtest-execution-events-validate-observations source=limen/backtest/execution_events.py region=signature:validate_observations role=signature -->
```python
def validate_observations(inputs: TradeInputs, policy: TradePolicy) -> None:
    ...
```

### `limen.backtest.execution_events.with_predictions`

<!-- reference: public-limen-backtest-execution-events-with-predictions source=limen/backtest/execution_events.py region=signature:with_predictions role=signature -->
```python
def with_predictions(inputs: TradeInputs, predictions: object) -> TradeInputs:
    ...
```

### `limen.backtest.funding_adapter.ADAPTER_VERSION`

<!-- reference: public-limen-backtest-funding-adapter-adapter-version source=limen/backtest/funding_adapter.py region=definition:ADAPTER_VERSION role=definition -->
```python
ADAPTER_VERSION = 'limen-funding-v1'
```

### `limen.backtest.funding_adapter.FUNDING_KEYS`

<!-- reference: public-limen-backtest-funding-adapter-funding-keys source=limen/backtest/funding_adapter.py region=definition:FUNDING_KEYS role=definition -->
```python
FUNDING_KEYS = frozenset(('rate', 'rate_unit', 'rate_basis_seconds', 'mechanism', 'settlement_interval_seconds', 'settlement_phase_utc_seconds', 'cash_settlement_interval_seconds', 'cash_settlement_phase_utc_seconds', 'valuation', 'currency', 'history_interpretation', 'approximation'))
```

### `limen.backtest.funding_adapter.FundingAdapter`

<!-- reference: public-limen-backtest-funding-adapter-fundingadapter source=limen/backtest/funding_adapter.py region=signature:FundingAdapter role=signature -->
```python
class FundingAdapter(Protocol):
    ...
```

### `limen.backtest.funding_adapter.FundingAdapter.__call__`

<!-- reference: public-limen-backtest-funding-adapter-fundingadapter---call-- source=limen/backtest/funding_adapter.py region=signature:FundingAdapter.__call__ role=signature -->
```python
def __call__(self, context: FundingContext, *, params: Mapping[str, JsonValue]) -> FundingCashflow:
    ...
```

### `limen.backtest.funding_adapter.FundingCashflow`

<!-- reference: public-limen-backtest-funding-adapter-fundingcashflow source=limen/backtest/funding_adapter.py region=signature:FundingCashflow role=signature -->
```python
@dataclass(frozen=True)
class FundingCashflow:
    event_id: str
    time_ns: int
    recognized_delta: float
    cash_delta: float
    currency: str
```

### `limen.backtest.funding_adapter.FundingContext`

<!-- reference: public-limen-backtest-funding-adapter-fundingcontext source=limen/backtest/funding_adapter.py region=signature:FundingContext role=signature -->
```python
@dataclass(frozen=True)
class FundingContext:
    event_id: str
    start_ns: int
    end_ns: int
    kind: Literal['payment', 'accrual', 'cash_settlement']
    accrued_funding: float
    quantity: float
    valuation_price: float
    rate_decimal: float
    rate_basis_seconds: float
    mechanism: Literal['discrete', 'continuous']
    currency: str
```

### `limen.backtest.funding_adapter.funding_adapter`

<!-- reference: public-limen-backtest-funding-adapter-funding-adapter source=limen/backtest/funding_adapter.py region=signature:funding_adapter role=signature -->
```python
def funding_adapter(context: FundingContext, *, params: Mapping[str, JsonValue]) -> FundingCashflow:
    ...
```

### `limen.backtest.funding_adapter.prepare_funding`

<!-- reference: public-limen-backtest-funding-adapter-prepare-funding source=limen/backtest/funding_adapter.py region=signature:prepare_funding role=signature -->
```python
def prepare_funding(policy: FundingPolicy, history: pl.DataFrame | None, start: int, end: int) -> pl.DataFrame:
    ...
```

### `limen.backtest.funding_adapter.resolve_adapter`

<!-- reference: public-limen-backtest-funding-adapter-resolve-adapter source=limen/backtest/funding_adapter.py region=signature:resolve_adapter role=signature -->
```python
def resolve_adapter(policy: FundingPolicy) -> FundingAdapter:
    ...
```

### `limen.backtest.funding_adapter.scheduled_times`

<!-- reference: public-limen-backtest-funding-adapter-scheduled-times source=limen/backtest/funding_adapter.py region=signature:scheduled_times role=signature -->
```python
def scheduled_times(start: int, end: int, interval_seconds: object, phase_seconds: object) -> range:
    ...
```

### `limen.backtest.funding_presets.BASELINE_RATE_8H`

<!-- reference: public-limen-backtest-funding-presets-baseline-rate-8h source=limen/backtest/funding_presets.py region=definition:BASELINE_RATE_8H role=definition -->
```python
BASELINE_RATE_8H = 2.8e-05
```

### `limen.backtest.funding_presets.FundingPreset`

<!-- reference: public-limen-backtest-funding-presets-fundingpreset source=limen/backtest/funding_presets.py region=signature:FundingPreset role=signature -->
```python
@dataclass(frozen=True)
class FundingPreset:
    instrument: str
    params: Mapping[str, JsonValue]
    calibration: Mapping[str, JsonValue]
```

### `limen.backtest.funding_presets.PRESETS`

<!-- reference: public-limen-backtest-funding-presets-presets source=limen/backtest/funding_presets.py region=definition:PRESETS role=definition -->
```python
PRESETS: Mapping[str, FundingPreset] = {'binance_btcusdt': FundingPreset('BTCUSDT', {'rate': BASELINE_RATE_8H, 'mechanism': 'discrete', 'rate_unit': 'decimal', 'rate_basis_seconds': 28800.0, 'settlement_interval_seconds': 28800.0, 'settlement_phase_utc_seconds': 0.0, 'valuation': 'execution_proxy', 'currency': 'USDT', 'approximation': 'scenario'}, _calibration()), 'hyperliquid_btc': FundingPreset('BTC', {'rate': BASELINE_RATE_8H / 8, 'mechanism': 'discrete', 'rate_unit': 'decimal', 'rate_basis_seconds': 3600.0, 'settlement_interval_seconds': 3600.0, 'settlement_phase_utc_seconds': 0.0, 'valuation': 'execution_proxy', 'currency': 'USDC', 'approximation': 'scenario'}, _calibration()), 'deribit_btc_usdc': FundingPreset('BTC_USDC-PERPETUAL', {'rate': BASELINE_RATE_8H, 'mechanism': 'continuous', 'rate_unit': 'decimal', 'rate_basis_seconds': 28800.0, 'cash_settlement_interval_seconds': 86400.0, 'cash_settlement_phase_utc_seconds': 28800.0, 'valuation': 'execution_proxy', 'currency': 'USDC', 'approximation': 'scenario'}, _calibration())}
```

### `limen.backtest.funding_presets.PRESET_VERSION`

<!-- reference: public-limen-backtest-funding-presets-preset-version source=limen/backtest/funding_presets.py region=definition:PRESET_VERSION role=definition -->
```python
PRESET_VERSION = '2026-10-08'
```

### `limen.backtest.observed_costs.CalibrationReport`

<!-- reference: public-limen-backtest-observed-costs-calibrationreport source=limen/backtest/observed_costs.py region=signature:CalibrationReport role=signature -->
```python
class CalibrationReport(TypedDict):
    digest: str
    method: Literal['minute_midpoint_1s_v1']
    source: JsonValue
    clocks: JsonValue
    estimates: JsonValue
    support: JsonValue
    sensitivity: JsonValue
    exclusions: JsonValue
    limitations: JsonValue
```

### `limen.backtest.observed_costs.ComparisonReport`

<!-- reference: public-limen-backtest-observed-costs-comparisonreport source=limen/backtest/observed_costs.py region=signature:ComparisonReport role=signature -->
```python
class ComparisonReport(TypedDict):
    digest: str
    calibration_digest: str
    candidates: JsonValue
    ranks: JsonValue
    failures: JsonValue
    unavailable: JsonValue
    limitations: JsonValue
```

### `limen.backtest.observed_costs.CostCalibration`

<!-- reference: public-limen-backtest-observed-costs-costcalibration source=limen/backtest/observed_costs.py region=signature:CostCalibration role=signature -->
```python
@dataclass(frozen=True, slots=True, eq=False)
class CostCalibration:
    _payload: dict[str, JsonValue]
    _digest: str = field(init=False)
```

Immutable recorded-price stress with separately reported account fees.

Pin ``digest`` in the caller's research specification before later ``read``.

The method preserves original candidate fees; observed cross-account fees

never establish the fees a different account will pay.

### `limen.backtest.observed_costs.CostCalibration.compare`

<!-- reference: public-limen-backtest-observed-costs-costcalibration-compare source=limen/backtest/observed_costs.py region=signature:CostCalibration.compare role=signature -->
```python
def compare(self, candidates: Mapping[str, CostCandidate]) -> CostComparison:
    ...
```

### `limen.backtest.observed_costs.CostCalibration.digest`

<!-- reference: public-limen-backtest-observed-costs-costcalibration-digest source=limen/backtest/observed_costs.py region=signature:CostCalibration.digest role=signature -->
```python
@property
def digest(self) -> str:
    ...
```

Return the versioned canonical content identity.

### `limen.backtest.observed_costs.CostCalibration.read`

<!-- reference: public-limen-backtest-observed-costs-costcalibration-read source=limen/backtest/observed_costs.py region=signature:CostCalibration.read role=signature -->
```python
@classmethod
def read(cls, path: str | Path, *, expected_digest: str) -> CostCalibration:
    ...
```

Read externally pinned content without re-estimating its rates.

### `limen.backtest.observed_costs.CostCalibration.report`

<!-- reference: public-limen-backtest-observed-costs-costcalibration-report source=limen/backtest/observed_costs.py region=signature:CostCalibration.report role=signature -->
```python
def report(self) -> CalibrationReport:
    ...
```

Return copied estimates, source claims and explicit transfer limits.

### `limen.backtest.observed_costs.CostCalibration.write`

<!-- reference: public-limen-backtest-observed-costs-costcalibration-write source=limen/backtest/observed_costs.py region=signature:CostCalibration.write role=signature -->
```python
def write(self, path: str | Path) -> str:
    ...
```

Publish once; the same canonical bytes may be written idempotently.

### `limen.backtest.observed_costs.CostCandidate`

<!-- reference: public-limen-backtest-observed-costs-costcandidate source=limen/backtest/observed_costs.py region=signature:CostCandidate role=signature -->
```python
@dataclass(frozen=True, eq=False)
class CostCandidate:
    inputs: InitVar[TradeInputs]
    policy: InitVar[TradePolicy]
    research_id: InitVar[str]
    conditions: InitVar[CostConditions]
    benchmark: InitVar[pl.DataFrame]
    _snapshot: dict[str, object] = field(init=False)
    _seal: str = field(init=False)
```

### `limen.backtest.observed_costs.CostClock`

<!-- reference: public-limen-backtest-observed-costs-costclock source=limen/backtest/observed_costs.py region=definition:CostClock role=definition -->
```python
CostClock = Literal['assumed_exchange_label']
```

### `limen.backtest.observed_costs.CostComparison`

<!-- reference: public-limen-backtest-observed-costs-costcomparison source=limen/backtest/observed_costs.py region=signature:CostComparison role=signature -->
```python
@dataclass(frozen=True, slots=True, eq=False)
class CostComparison:
    _payload: dict[str, JsonValue]
    _ledgers: dict[tuple[str, str], TradeLedger]
    _digest: str = field(init=False)
```

### `limen.backtest.observed_costs.CostComparison.digest`

<!-- reference: public-limen-backtest-observed-costs-costcomparison-digest source=limen/backtest/observed_costs.py region=signature:CostComparison.digest role=signature -->
```python
@property
def digest(self) -> str:
    ...
```

### `limen.backtest.observed_costs.CostComparison.ledger`

<!-- reference: public-limen-backtest-observed-costs-costcomparison-ledger source=limen/backtest/observed_costs.py region=signature:CostComparison.ledger role=signature -->
```python
def ledger(self, candidate: str, scenario: Literal['original', 'calibrated']) -> TradeLedger:
    ...
```

### `limen.backtest.observed_costs.CostComparison.report`

<!-- reference: public-limen-backtest-observed-costs-costcomparison-report source=limen/backtest/observed_costs.py region=signature:CostComparison.report role=signature -->
```python
def report(self) -> ComparisonReport:
    ...
```

### `limen.backtest.observed_costs.CostComparison.write`

<!-- reference: public-limen-backtest-observed-costs-costcomparison-write source=limen/backtest/observed_costs.py region=signature:CostComparison.write role=signature -->
```python
def write(self, path: str | Path) -> str:
    ...
```

### `limen.backtest.observed_costs.CostConditions`

<!-- reference: public-limen-backtest-observed-costs-costconditions source=limen/backtest/observed_costs.py region=signature:CostConditions role=signature -->
```python
class CostConditions(TypedDict):
    venue: str
    environment: Literal['mainnet']
    product_kind: Literal['linear_perpetual']
    instrument: str
    base_currency: str
    quote_currency: str
    reference_id: str
    reference_convention: Literal['minute_midpoint_label']
    clock: CostClock
```

### `limen.backtest.observed_costs.CostDisposition`

<!-- reference: public-limen-backtest-observed-costs-costdisposition source=limen/backtest/observed_costs.py region=definition:CostDisposition role=definition -->
```python
CostDisposition = Literal['original', 'retrospective_cost_scenario', 'retrospective_low_effective_support', 'unavailable', 'economic_failure']
```

### `limen.backtest.trade_contract.BPS`

<!-- reference: public-limen-backtest-trade-contract-bps source=limen/backtest/trade_contract.py region=definition:BPS role=definition -->
```python
BPS = 10000.0
```

### `limen.backtest.trade_contract.ExecutionEvent`

<!-- reference: public-limen-backtest-trade-contract-executionevent source=limen/backtest/trade_contract.py region=signature:ExecutionEvent role=signature -->
```python
@dataclass(frozen=True)
class ExecutionEvent:
    event_id: str
    time_ns: int
    kind: Literal['observation', 'funding', 'timer', 'signal']
    source_row_id: str | None
    available_at_ns: int
    observation_phase: Literal['open', 'close'] | None = None
```

### `limen.backtest.trade_contract.JsonScalar`

<!-- reference: public-limen-backtest-trade-contract-jsonscalar source=limen/backtest/trade_contract.py region=definition:JsonScalar role=definition -->
```python
JsonScalar: TypeAlias = str | int | float | bool | None
```

### `limen.backtest.trade_contract.JsonValue`

<!-- reference: public-limen-backtest-trade-contract-jsonvalue source=limen/backtest/trade_contract.py region=definition:JsonValue role=definition -->
```python
JsonValue: TypeAlias = JsonScalar | list['JsonValue'] | dict[str, 'JsonValue']
```

### `limen.backtest.trade_contract.NANOSECONDS`

<!-- reference: public-limen-backtest-trade-contract-nanoseconds source=limen/backtest/trade_contract.py region=definition:NANOSECONDS role=definition -->
```python
NANOSECONDS = 1000000000
```

### `limen.backtest.trade_contract.PredictionMode`

<!-- reference: public-limen-backtest-trade-contract-predictionmode source=limen/backtest/trade_contract.py region=definition:PredictionMode role=definition -->
```python
PredictionMode: TypeAlias = Literal['binary', 'target_exposure']
```

### `limen.backtest.trade_contract.RULE_VERSION`

<!-- reference: public-limen-backtest-trade-contract-rule-version source=limen/backtest/trade_contract.py region=definition:RULE_VERSION role=definition -->
```python
RULE_VERSION = 'limen-trade-v1'
```

### `limen.backtest.trade_contract.SourceBinding`

<!-- reference: public-limen-backtest-trade-contract-sourcebinding source=limen/backtest/trade_contract.py region=signature:SourceBinding role=signature -->
```python
@dataclass(frozen=True)
class SourceBinding:
    identity: str
    checksum: str
    start_ns: int
    end_ns: int
    precision_ns: int
    interpretation: str
```

### `limen.backtest.trade_contract.TradeLedger`

<!-- reference: public-limen-backtest-trade-contract-tradeledger source=limen/backtest/trade_contract.py region=signature:TradeLedger role=signature -->
```python
@dataclass(frozen=True)
class TradeLedger:
    states: pl.DataFrame
    intents: pl.DataFrame
    fills: pl.DataFrame
    episodes: pl.DataFrame
    funding: pl.DataFrame
    metrics: Mapping[str, float]
    contract_digest: str
```

### `limen.backtest.trade_contract.contract_digest`

<!-- reference: public-limen-backtest-trade-contract-contract-digest source=limen/backtest/trade_contract.py region=signature:contract_digest role=signature -->
```python
def contract_digest(contract: Mapping[str, JsonValue]) -> str:
    ...
```

### `limen.backtest.trade_contract.export_trade_contract`

<!-- reference: public-limen-backtest-trade-contract-export-trade-contract source=limen/backtest/trade_contract.py region=signature:export_trade_contract role=signature -->
```python
def export_trade_contract(policy: TradePolicy, inputs: TradeInputs) -> dict[str, JsonValue]:
    ...
```

### `limen.backtest.trade_contract.finite_number`

<!-- reference: public-limen-backtest-trade-contract-finite-number source=limen/backtest/trade_contract.py region=signature:finite_number role=signature -->
```python
def finite_number(value: object, name: str) -> float:
    ...
```

### `limen.backtest.trade_contract.json_value`

<!-- reference: public-limen-backtest-trade-contract-json-value source=limen/backtest/trade_contract.py region=signature:json_value role=signature -->
```python
def json_value(value: object) -> JsonValue:
    ...
```

### `limen.backtest.trade_contract.source_binding`

<!-- reference: public-limen-backtest-trade-contract-source-binding source=limen/backtest/trade_contract.py region=signature:source_binding role=signature -->
```python
def source_binding(frame: pl.DataFrame, identity: str, start_ns: int, end_ns: int, precision_ns: int, interpretation: str) -> SourceBinding:
    ...
```

### `limen.backtest.trade_execution`

<!-- reference: public-limen-backtest-trade-execution source=limen/backtest/trade_execution.py region=signature:trade_execution role=signature -->
```python
def trade_execution(inputs: TradeInputs, policy: TradePolicy) -> TradeLedger:
    ...
```

### `limen.calibration.CalibratorProtocol`

<!-- reference: public-limen-calibration-calibratorprotocol source=limen/calibration/pipeline.py region=signature:CalibratorProtocol role=signature -->
```python
class CalibratorProtocol(Protocol):
    ...
```

Protocol for probability calibration functions.

### `limen.calibration.ThresholdOptimizerProtocol`

<!-- reference: public-limen-calibration-thresholdoptimizerprotocol source=limen/calibration/pipeline.py region=signature:ThresholdOptimizerProtocol role=signature -->
```python
class ThresholdOptimizerProtocol(Protocol):
    ...
```

Protocol for threshold optimisation functions.

### `limen.calibration.apply_calibrated_predict`

<!-- reference: public-limen-calibration-apply-calibrated-predict source=limen/calibration/pipeline.py region=signature:apply_calibrated_predict role=signature -->
```python
def apply_calibrated_predict(model: Any, config: CalibrationConfigProtocol, data: dict[str, Any]) -> dict[str, Any]:
    ...
```

Apply calibration and threshold optimisation to a fitted model's predictions.

Args:

model: Fitted classifier with predict_proba method

config (CalibrationConfigProtocol): Resolved calibration configuration

data (dict): Data dictionary with x_val, y_val, x_test keys

Returns:

dict: Results with '_preds', '_probs', 'optimal_threshold' and 'val_score'

(val_score is None when no threshold_func is configured)

### `limen.calibration.fit_calibrator`

<!-- reference: public-limen-calibration-fit-calibrator source=limen/calibration/pipeline.py region=signature:fit_calibrator role=signature -->
```python
def fit_calibrator(model: Any, config: CalibrationConfigProtocol, x_val: npt.NDArray[Any] | pl.DataFrame, y_val: npt.NDArray[Any] | pl.Series) -> tuple[Any, float, float | None]:
    ...
```

Fit calibrator on validation data.

Args:

model (Any): Fitted classifier with predict_proba method

config (CalibrationConfigProtocol): Resolved calibration configuration

x_val (np.ndarray or pl.DataFrame): Validation features

y_val (np.ndarray or pl.Series): Validation labels

Returns:

tuple: (fitted_calibrator, optimal_threshold, val_score)

val_score is None when no threshold_func is configured

### `limen.cli.cli`

<!-- reference: public-limen-cli-cli source=limen/cli/main.py region=signature:cli role=signature -->
```python
@click.group()
@click.version_option(version=__version__, prog_name='limen')
def cli() -> None:
    ...
```

Limen — declarative ML experiment runner.



Define experiments in YAML; validate, compile, search and save with the CLI.



Quick start:

limen validate experiment.yaml       Check YAML; profile it with limen profile

limen run experiment.yaml            Run the experiment

limen run --dry-run experiment.yaml  Validate + compile only, no execution



YAML structure:

schema_version: "1.0"

metadata:

name: my_experiment

mode: development          # development \| production

sfd:

manifest:

type: ml                 # ml \| rule_based

...

params:

lookback: [12, 24, 48]

...

uel:

n_permutations: 100

search_strategy:

type: random             # random \| grid

output_format: csv         # csv \| parquet

Template discovery: limen list-templates

### `limen.cohort.sfc.Selector`

<!-- reference: public-limen-cohort-sfc-selector source=limen/cohort/sfc/__init__.py region=definition:Selector role=definition -->
```python
Selector = Callable[..., list[int | str]]
```

### `limen.cohort.sfc.SelectorContext`

<!-- reference: public-limen-cohort-sfc-selectorcontext source=limen/cohort/sfc/__init__.py region=definition:SelectorContext role=definition -->
```python
SelectorContext = dict[str, Any]
```

### `limen.data.bars.liquidity_bars`

<!-- reference: public-limen-data-bars-liquidity-bars source=limen/data/bars/standard_bars.py region=signature:liquidity_bars role=signature -->
```python
def liquidity_bars(data: pl.DataFrame, liquidity_threshold: float) -> pl.DataFrame:
    ...
```

Compute liquidity bars with fixed liquidity sampling.

Args:

data (pl.DataFrame): Klines data

liquidity_threshold (float): Liquidity threshold per bar

Returns:

pl.DataFrame: Standard liquidity data bars

### `limen.data.bars.trade_bars`

<!-- reference: public-limen-data-bars-trade-bars source=limen/data/bars/standard_bars.py region=signature:trade_bars role=signature -->
```python
def trade_bars(data: pl.DataFrame, trade_threshold: int) -> pl.DataFrame:
    ...
```

Compute trade bars with fixed trade count sampling.

Args:

data (pl.DataFrame): Klines data

trade_threshold (int): Number of trades per bar

Returns:

pl.DataFrame: Standard trade count data bars

### `limen.data.bars.volume_bars`

<!-- reference: public-limen-data-bars-volume-bars source=limen/data/bars/standard_bars.py region=signature:volume_bars role=signature -->
```python
def volume_bars(data: pl.DataFrame, volume_threshold: float) -> pl.DataFrame:
    ...
```

Compute volume bars with fixed volume size sampling.

Args:

data (pl.DataFrame): Klines data

volume_threshold (float): Volume threshold per bar

Returns:

pl.DataFrame: Standard volume data bars

### `limen.data.compute_data_bars`

<!-- reference: public-limen-data-compute-data-bars source=limen/data/utils/compute_data_bars.py region=signature:compute_data_bars role=signature -->
```python
def compute_data_bars(data: pl.DataFrame, **params: Any) -> pl.DataFrame:
    ...
```

Compute adaptive bar formation based on bar_type parameter.

Args:

data (pl.DataFrame): Klines dataset with 'datetime', 'open', 'high', 'low', 'close', 'volume', 'no_of_trades', 'liquidity_sum', 'maker_ratio', 'maker_volume', 'maker_liquidity' columns

**params: Bar formation parameters including bar_type and threshold parameters

Returns:

pl.DataFrame: Data bars with columns 'datetime', 'open', 'high', 'low', 'close', 'volume', 'no_of_trades', 'liquidity_sum', 'maker_ratio', 'maker_volume', 'maker_liquidity', 'mean', 'bar_count', 'base_interval'

### `limen.data.utils.random_slice`

<!-- reference: public-limen-data-utils-random-slice source=limen/data/utils/random_slice.py region=signature:random_slice role=signature -->
```python
def random_slice(df: pl.DataFrame, rows: int, *, safe_range_low: float=0.25, safe_range_high: float=0.75, seed: int | None=None) -> pl.DataFrame:
    ...
```

Compute contiguous slice from DataFrame within specified safe range.

Args:

df (pl.DataFrame): Input DataFrame to slice from

rows (int): Positive, non-boolean integer number of rows to include

safe_range_low (float): Lower bound of safe range as fraction of total rows

safe_range_high (float): Upper bound of safe range as fraction of total rows

seed (int \| None): Random seed for reproducible results

Returns:

pl.DataFrame: Contiguous slice of the original DataFrame maintaining row order

### `limen.data.utils.split_data_to_prep_output`

<!-- reference: public-limen-data-utils-split-data-to-prep-output source=limen/data/utils/splits.py region=signature:split_data_to_prep_output role=signature -->
```python
def split_data_to_prep_output(split_data: list[pl.DataFrame], cols: list[str], all_datetimes: list[datetime] | list[int]) -> dict[str, Any]:
    ...
```

Compute data preparation output dictionary from split data and column names.

Args:

split_data (list): List of three DataFrames representing train, validation, and test splits

cols (list): Column names where the last column is the target variable

all_datetimes (list): List of all datetimes

Returns:

dict: Dictionary with train, validation, and test features and targets

### `limen.data.utils.split_data_to_rule_based_prep_output`

<!-- reference: public-limen-data-utils-split-data-to-rule-based-prep-output source=limen/data/utils/splits.py region=signature:split_data_to_rule_based_prep_output role=signature -->
```python
def split_data_to_rule_based_prep_output(split_data: list[pl.DataFrame], all_datetimes: list[datetime] | list[int]) -> dict[str, Any]:
    ...
```

Compute data preparation output dictionary for rule-based strategies.

Args:

split_data (list): List of three DataFrames representing train, validation, and test splits

all_datetimes (list): List of all datetimes

Returns:

dict: Dictionary with train, val, test full DataFrames and _alignment metadata

### `limen.data.utils.split_random`

<!-- reference: public-limen-data-utils-split-random source=limen/data/utils/splits.py region=signature:split_random role=signature -->
```python
def split_random(data: pl.DataFrame, ratios: Sequence[int], seed: int | None=None) -> list[pl.DataFrame]:
    ...
```

Compute random data splits with proportional lengths based on ratios.

Args:

data (pl.DataFrame): Polars DataFrame to split randomly

ratios (Sequence[int]): Sequence of positive integers defining split proportions

seed (int \| None): Random seed; None draws a fresh, disjoint partition each call

Returns:

List[pl.DataFrame]: Disjoint, randomly shuffled partitions containing every input row

### `limen.data.utils.split_sequential`

<!-- reference: public-limen-data-utils-split-sequential source=limen/data/utils/splits.py region=signature:split_sequential role=signature -->
```python
def split_sequential(data: pl.DataFrame, ratios: Sequence[int]) -> list[pl.DataFrame]:
    ...
```

Compute sequential data splits with proportional lengths based on ratios.

Args:

data (pl.DataFrame): Polars DataFrame to split sequentially

ratios (Sequence[int]): Sequence of positive integers defining split proportions

Returns:

List[pl.DataFrame]: List of DataFrames partitioned sequentially without losing or duplicating rows

### `limen.data.utils.split_walk_forward`

<!-- reference: public-limen-data-utils-split-walk-forward source=limen/data/utils/split_walk_forward.py region=signature:split_walk_forward role=signature -->
```python
def split_walk_forward(data: pl.DataFrame, *, n_folds: int, test_bars: int, purge_bars: int, embargo_bars: int, anchored: bool) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    ...
```

Split an already time-ordered frame by row position.

The final ``n_folds * test_bars`` rows form contiguous test windows.

Each candidate train ends ``purge_bars`` before its test start and

excludes ``embargo_bars`` rows following every earlier test window.

Anchored trains start at row zero; rolling candidate trains retain

the first fold's width before embargo removal.

### `limen.experiment.CalibrationConfig`

<!-- reference: public-limen-experiment-calibrationconfig source=limen/experiment/manifest_core.py region=signature:CalibrationConfig role=signature -->
```python
@dataclass
class CalibrationConfig:
    calibration_func: CalibratorProtocol | None = None
    calibration_params: dict[str, Any] = field(default_factory=dict[str, Any])
    threshold_func: ThresholdOptimizerProtocol | None = None
    threshold_params: dict[str, Any] = field(default_factory=dict[str, Any])
```

Stores probability calibration and threshold function references with their params.

### `limen.experiment.CalibrationConfig.resolve`

<!-- reference: public-limen-experiment-calibrationconfig-resolve source=limen/experiment/manifest_core.py region=signature:CalibrationConfig.resolve role=signature -->
```python
def resolve(self, round_params: dict[str, Any]) -> 'CalibrationConfig':
    ...
```

Return a new config with string params resolved from round_params.

### `limen.experiment.GridStrategy`

<!-- reference: public-limen-experiment-gridstrategy source=limen/experiment/param_search/grid_strategy.py region=signature:GridStrategy role=signature -->
```python
class GridStrategy(SearchStrategy):
    ...
```

Finite mixed-radix traversal; seeded shuffle uses a versioned permutation.

### `limen.experiment.GridStrategy.__init__`

<!-- reference: public-limen-experiment-gridstrategy---init-- source=limen/experiment/param_search/grid_strategy.py region=signature:GridStrategy.__init__ role=signature -->
```python
def __init__(self, domain: ParamDomain, *, seed: int | None=None, shuffle: bool=False) -> None:
    ...
```

### `limen.experiment.GridStrategy.get_state`

<!-- reference: public-limen-experiment-gridstrategy-get-state source=limen/experiment/param_search/grid_strategy.py region=signature:GridStrategy.get_state role=signature -->
```python
@override
def get_state(self) -> dict[str, Any]:
    ...
```

### `limen.experiment.GridStrategy.is_finite`

<!-- reference: public-limen-experiment-gridstrategy-is-finite source=limen/experiment/param_search/grid_strategy.py region=signature:GridStrategy.is_finite role=signature -->
```python
@property
@override
def is_finite(self) -> bool:
    ...
```

### `limen.experiment.GridStrategy.on_domain_changed`

<!-- reference: public-limen-experiment-gridstrategy-on-domain-changed source=limen/experiment/param_search/grid_strategy.py region=signature:GridStrategy.on_domain_changed role=signature -->
```python
@override
def on_domain_changed(self, _domain: ParamDomain, _changed_params: list[str]) -> None:
    ...
```

### `limen.experiment.GridStrategy.set_state`

<!-- reference: public-limen-experiment-gridstrategy-set-state source=limen/experiment/param_search/grid_strategy.py region=signature:GridStrategy.set_state role=signature -->
```python
@override
def set_state(self, state: dict[str, Any]) -> None:
    ...
```

### `limen.experiment.RandomStrategy`

<!-- reference: public-limen-experiment-randomstrategy source=limen/experiment/param_search/random_strategy.py region=signature:RandomStrategy role=signature -->
```python
class RandomStrategy(SearchStrategy):
    ...
```

Lazy random sampling from the parameter domain.

Samples each parameter independently via uniform random choice.

Infinite — never raises StopIteration. Uses _is_unseen for

dedup to avoid repeating previously seen combinations.

### `limen.experiment.RandomStrategy.__init__`

<!-- reference: public-limen-experiment-randomstrategy---init-- source=limen/experiment/param_search/random_strategy.py region=signature:RandomStrategy.__init__ role=signature -->
```python
def __init__(self, domain: ParamDomain, *, seed: int | None=None) -> None:
    ...
```

### `limen.experiment.RandomStrategy.get_state`

<!-- reference: public-limen-experiment-randomstrategy-get-state source=limen/experiment/param_search/random_strategy.py region=signature:RandomStrategy.get_state role=signature -->
```python
@override
def get_state(self) -> dict[str, Any]:
    ...
```

### `limen.experiment.RandomStrategy.on_domain_changed`

<!-- reference: public-limen-experiment-randomstrategy-on-domain-changed source=limen/experiment/param_search/random_strategy.py region=signature:RandomStrategy.on_domain_changed role=signature -->
```python
@override
def on_domain_changed(self, _domain: ParamDomain, _changed_params: list[str]) -> None:
    ...
```

### `limen.experiment.RandomStrategy.set_state`

<!-- reference: public-limen-experiment-randomstrategy-set-state source=limen/experiment/param_search/random_strategy.py region=signature:RandomStrategy.set_state role=signature -->
```python
@override
def set_state(self, state: dict[str, Any]) -> None:
    ...
```

### `limen.experiment.STRATEGY_REGISTRY`

<!-- reference: public-limen-experiment-strategy-registry source=limen/experiment/param_search/registry.py region=definition:STRATEGY_REGISTRY role=definition -->
```python
STRATEGY_REGISTRY: dict[str, type[SearchStrategy]] = {'random': RandomStrategy, 'grid': GridStrategy, 'sobol': SobolStrategy}
```

### `limen.experiment.acceptance_report.acceptance_report`

<!-- reference: public-limen-experiment-acceptance-report-acceptance-report source=limen/experiment/acceptance_report.py region=signature:acceptance_report role=signature -->
```python
def acceptance_report(directory: Path, *, acceptance: Mapping[str, float] | None=None) -> dict[str, object] | None:
    ...
```

### `limen.experiment.acceptance_report.read_acceptance`

<!-- reference: public-limen-experiment-acceptance-report-read-acceptance source=limen/experiment/_acceptance_settings.py region=signature:read_acceptance role=signature -->
```python
def read_acceptance(value: object) -> dict[str, float]:
    ...
```

### `limen.experiment.convergence_report.convergence_header`

<!-- reference: public-limen-experiment-convergence-report-convergence-header source=limen/experiment/convergence_report.py region=signature:convergence_header role=signature -->
```python
def convergence_header(path: Path, header: list[str]) -> list[str]:
    ...
```

Extend older result headers without inferring past warning evidence.

### `limen.experiment.convergence_report.convergence_report`

<!-- reference: public-limen-experiment-convergence-report-convergence-report source=limen/experiment/convergence_report.py region=signature:convergence_report role=signature -->
```python
def convergence_report(results: pl.DataFrame, *, parameter_columns: Sequence[str]) -> dict[str, object]:
    ...
```

Describe recorded warning frequency, without inferring missing evidence.

### `limen.experiment.convergence_report.convergence_warning`

<!-- reference: public-limen-experiment-convergence-report-convergence-warning source=limen/experiment/convergence_report.py region=signature:convergence_warning role=signature -->
```python
def convergence_warning(caught: Sequence[warnings.WarningMessage], succeeded: bool) -> bool | None:
    ...
```

Record category evidence only for completed rounds.

### `limen.experiment.convergence_report.convergence_warnings`

<!-- reference: public-limen-experiment-convergence-report-convergence-warnings source=limen/experiment/convergence_report.py region=signature:convergence_warnings role=signature -->
```python
@contextmanager
def convergence_warnings() -> Iterator[list[warnings.WarningMessage]]:
    ...
```

Observe filtered warnings while preserving errors and external display.

### `limen.experiment.holdout.FrozenHoldoutSelection`

<!-- reference: public-limen-experiment-holdout-frozenholdoutselection source=limen/experiment/_holdout_freeze.py region=signature:FrozenHoldoutSelection role=signature -->
```python
class FrozenHoldoutSelection:
    ...
```

### `limen.experiment.holdout.FrozenHoldoutSelection.__init__`

<!-- reference: public-limen-experiment-holdout-frozenholdoutselection---init-- source=limen/experiment/_holdout_freeze.py region=signature:FrozenHoldoutSelection.__init__ role=signature -->
```python
def __init__(self, token: object, /) -> None:
    ...
```

### `limen.experiment.holdout.final_holdout_report`

<!-- reference: public-limen-experiment-holdout-final-holdout-report source=limen/experiment/_holdout_assessment.py region=signature:final_holdout_report role=signature -->
```python
def final_holdout_report(frozen: FrozenHoldoutSelection, *, project_root: str | Path, reservation_id: str, start: str | None=None, end: str | None=None) -> dict[str, object]:
    ...
```

Allocate durable exposure, then assess a causal tape from an owned fitted copy.

### `limen.experiment.holdout.freeze_holdout_selection`

<!-- reference: public-limen-experiment-holdout-freeze-holdout-selection source=limen/experiment/_holdout_freeze.py region=signature:freeze_holdout_selection role=signature -->
```python
def freeze_holdout_selection(selection: Sensor | Cohort, *, context_start: str, scenarios: list[Mapping[str, object]], evaluation_target: NextBarUpTarget | NextBarDownTarget | ThresholdBinaryTarget | None=None, round_locator: Mapping[str, object] | None=None) -> FrozenHoldoutSelection:
    ...
```

Own the fitted selection and preregister its final-assessment specification.

### `limen.experiment.holdout.read_holdout_report`

<!-- reference: public-limen-experiment-holdout-read-holdout-report source=limen/experiment/_holdout_report.py region=signature:read_holdout_report role=signature -->
```python
def read_holdout_report(project_root: str | Path, attempt_id: str) -> dict[str, object]:
    ...
```

Read completed evidence, verifying hashes and schemas without evaluating.

### `limen.experiment.holdout.release_holdout`

<!-- reference: public-limen-experiment-holdout-release-holdout source=limen/experiment/_holdout_history.py region=signature:release_holdout role=signature -->
```python
def release_holdout(project_root: str | Path, reservation_id: str) -> None:
    ...
```

Lift research exclusion while preserving the post-release exposure record.

### `limen.experiment.holdout.reserve_holdout`

<!-- reference: public-limen-experiment-holdout-reserve-holdout source=limen/experiment/_holdout_history.py region=signature:reserve_holdout role=signature -->
```python
def reserve_holdout(project_root: str | Path, start: str, end: str, *, interval_seconds: int) -> str:
    ...
```

Exclude the continuous suffix from the earliest active reservation onward.

### `limen.experiment.holdout.withdraw_holdout`

<!-- reference: public-limen-experiment-holdout-withdraw-holdout source=limen/experiment/_holdout_history.py region=signature:withdraw_holdout role=signature -->
```python
def withdraw_holdout(project_root: str | Path, reservation_id: str) -> None:
    ...
```

Retire an unused reservation without deleting its identity or history.

### `limen.experiment.param_search.SearchStrategy`

<!-- reference: public-limen-experiment-param-search-searchstrategy source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy role=signature -->
```python
class SearchStrategy(ABC):
    ...
```

Abstract base for all search strategies.

Strategies are iterators that yield dict[str, Any] combination of parameters.

They hold a reference to a ParamDomain and register as observers

so they can react to domain mutations.

### `limen.experiment.param_search.SearchStrategy.__init__`

<!-- reference: public-limen-experiment-param-search-searchstrategy---init-- source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy.__init__ role=signature -->
```python
def __init__(self, domain: ParamDomain, *, seed: int | None=None) -> None:
    ...
```

Initialize the SearchStrategy.

Args:

domain (ParamDomain): ParamDomain to generate combinations from

seed (int \| None): Optional random seed for reproducibility

### `limen.experiment.param_search.SearchStrategy.compute_param_hash`

<!-- reference: public-limen-experiment-param-search-searchstrategy-compute-param-hash source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy.compute_param_hash role=signature -->
```python
def compute_param_hash(self, combo: dict[str, Any]) -> str:
    ...
```

Compute deterministic hash of a parameter combination.

Returns the full 64 hex characters (256 bits) of the SHA-256

digest.

NOTE:

Strips ``_``-prefixed keys because domain params never

start with underscore, but log rows include MSQ metadata

columns (``_id``, ``_round_index``, etc.) that must not

affect the hash.

Args:

combo (dict[str, Any]): Parameter combination or log row

Returns:

str: 64-character hex digest

### `limen.experiment.param_search.SearchStrategy.domain`

<!-- reference: public-limen-experiment-param-search-searchstrategy-domain source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy.domain role=signature -->
```python
@property
def domain(self) -> ParamDomain:
    ...
```

### `limen.experiment.param_search.SearchStrategy.generated_count`

<!-- reference: public-limen-experiment-param-search-searchstrategy-generated-count source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy.generated_count role=signature -->
```python
@property
def generated_count(self) -> int:
    ...
```

### `limen.experiment.param_search.SearchStrategy.get_state`

<!-- reference: public-limen-experiment-param-search-searchstrategy-get-state source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy.get_state role=signature -->
```python
@abstractmethod
def get_state(self) -> dict[str, Any]:
    ...
```

Export state for checkpointing.

### `limen.experiment.param_search.SearchStrategy.is_finite`

<!-- reference: public-limen-experiment-param-search-searchstrategy-is-finite source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy.is_finite role=signature -->
```python
@property
def is_finite(self) -> bool:
    ...
```

Whether this strategy has a finite number of combinations.

Override to return True for exhaustive strategies (e.g. Grid).

### `limen.experiment.param_search.SearchStrategy.mark_seen`

<!-- reference: public-limen-experiment-param-search-searchstrategy-mark-seen source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy.mark_seen role=signature -->
```python
def mark_seen(self, combo: dict[str, Any]) -> str:
    ...
```

Register a combination as seen and return its hash.

Called by MSQ after a combo passes filters and is yielded.

Reuses ``_id`` if already attached by ``_is_unseen``.

Args:

combo (dict[str, Any]): Parameter combination

Returns:

str: The param hash

### `limen.experiment.param_search.SearchStrategy.on_domain_changed`

<!-- reference: public-limen-experiment-param-search-searchstrategy-on-domain-changed source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy.on_domain_changed role=signature -->
```python
def on_domain_changed(self, _domain: ParamDomain, _changed_params: list[str]) -> None:
    ...
```

Called when ParamDomain is mutated. Override if strategy

maintains state dependent on the domain.

### `limen.experiment.param_search.SearchStrategy.rebuild_seen_from_log`

<!-- reference: public-limen-experiment-param-search-searchstrategy-rebuild-seen-from-log source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy.rebuild_seen_from_log role=signature -->
```python
def rebuild_seen_from_log(self, hashes: Iterable[str]) -> None:
    ...
```

Populate _seen from an iterable of hashes. Used on resume

to reconstruct dedup state from the experiment log.

Args:

hashes (Iterable[str]): Hash strings from previous runs

### `limen.experiment.param_search.SearchStrategy.set_state`

<!-- reference: public-limen-experiment-param-search-searchstrategy-set-state source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy.set_state role=signature -->
```python
@abstractmethod
def set_state(self, state: dict[str, Any]) -> None:
    ...
```

Restore state from checkpoint.

### `limen.experiment.param_search.SearchStrategy.update_from_feedback`

<!-- reference: public-limen-experiment-param-search-searchstrategy-update-from-feedback source=limen/experiment/param_search/search_strategy.py region=signature:SearchStrategy.update_from_feedback role=signature -->
```python
def update_from_feedback(self, _log: Any, _interventions: list[dict[str, Any]]) -> None:
    ...
```

Hook for strategies that adapt based on experiment feedback.

Default is no-op. Stateful strategies (e.g. TPE) override this.

### `limen.experiment.param_search.SobolStrategy`

<!-- reference: public-limen-experiment-param-search-sobolstrategy source=limen/experiment/param_search/sobol_strategy.py region=signature:SobolStrategy role=signature -->
```python
class SobolStrategy(SearchStrategy):
    ...
```

Scrambled Sobol points projected onto the current discrete domain.

### `limen.experiment.param_search.SobolStrategy.__init__`

<!-- reference: public-limen-experiment-param-search-sobolstrategy---init-- source=limen/experiment/param_search/sobol_strategy.py region=signature:SobolStrategy.__init__ role=signature -->
```python
def __init__(self, domain: ParamDomain, *, seed: int) -> None:
    ...
```

### `limen.experiment.param_search.SobolStrategy.get_state`

<!-- reference: public-limen-experiment-param-search-sobolstrategy-get-state source=limen/experiment/param_search/sobol_strategy.py region=signature:SobolStrategy.get_state role=signature -->
```python
@override
def get_state(self) -> dict[str, object]:
    ...
```

### `limen.experiment.param_search.SobolStrategy.on_domain_changed`

<!-- reference: public-limen-experiment-param-search-sobolstrategy-on-domain-changed source=limen/experiment/param_search/sobol_strategy.py region=signature:SobolStrategy.on_domain_changed role=signature -->
```python
@override
def on_domain_changed(self, _domain: ParamDomain, _changed_params: list[str]) -> None:
    ...
```

### `limen.experiment.param_search.SobolStrategy.set_state`

<!-- reference: public-limen-experiment-param-search-sobolstrategy-set-state source=limen/experiment/param_search/sobol_strategy.py region=signature:SobolStrategy.set_state role=signature -->
```python
@override
def set_state(self, state: dict[str, object]) -> None:
    ...
```

### `limen.experiment.reducer.ACTION_SUGGEST`

<!-- reference: public-limen-experiment-reducer-action-suggest source=limen/experiment/reducer/pruning_strategy.py region=definition:ACTION_SUGGEST role=definition -->
```python
ACTION_SUGGEST = 'suggest'
```

### `limen.experiment.reducer.BudgetReducer`

<!-- reference: public-limen-experiment-reducer-budgetreducer source=limen/experiment/reducer/budget_reducer.py region=signature:BudgetReducer role=signature -->
```python
class BudgetReducer(PruningStrategy):
    ...
```

Trim the search queue when projected completion exceeds resource

budgets (walltime or permutation count).

Purely resource-driven — no statistical analysis of results.

Supports two trim strategies:

- random: uniformly downsample permutations to the target count

- worst_first: rank parameter values by mean metric performance

and remove combinations containing the worst-performing values

A random trim is always emitted as a fallback to ensure the queue

is reduced to the target count if value-based pruning alone is

insufficient.

NOTE: Fires at most once per experiment — once trimming occurs,

further calls to analyze_and_intervene return empty. The

check_after_pct gate prevents premature trimming when throughput

estimates are unreliable early in the experiment.

### `limen.experiment.reducer.BudgetReducer.analyze_and_intervene`

<!-- reference: public-limen-experiment-reducer-budgetreducer-analyze-and-intervene source=limen/experiment/reducer/budget_reducer.py region=signature:BudgetReducer.analyze_and_intervene role=signature -->
```python
@override
def analyze_and_intervene(self, log: pl.DataFrame, msq: Any) -> list[dict[str, Any]]:
    ...
```

Check resource budgets and trim the queue if over budget.

Computes throughput from elapsed time and completed rounds,

then projects whether the remaining queue can finish within

the configured budget.

### `limen.experiment.reducer.BudgetReducer.get_state`

<!-- reference: public-limen-experiment-reducer-budgetreducer-get-state source=limen/experiment/reducer/budget_reducer.py region=signature:BudgetReducer.get_state role=signature -->
```python
@override
def get_state(self) -> dict[str, Any]:
    ...
```

Export state for checkpointing.

### `limen.experiment.reducer.BudgetReducer.set_state`

<!-- reference: public-limen-experiment-reducer-budgetreducer-set-state source=limen/experiment/reducer/budget_reducer.py region=signature:BudgetReducer.set_state role=signature -->
```python
@override
def set_state(self, state: dict[str, Any]) -> None:
    ...
```

Restore state from checkpoint.

### `limen.experiment.reducer.CorrelationReducer`

<!-- reference: public-limen-experiment-reducer-correlationreducer source=limen/experiment/reducer/correlation_reducer.py region=signature:CorrelationReducer role=signature -->
```python
class CorrelationReducer(PruningStrategy):
    ...
```

Prune parameter values based on bootstrap correlation analysis.

Two pruning modes:

- Wrong-direction: parameters with strong negative correlation

(for maximization) get their worst-performing value removed

- Low-impact: parameters with negligible \|correlation\| and high

sign stability get a keep_is suggestion for the best value

NOTE: Only acts on numeric parameters. Categorical/string parameters

are coerced to numeric internally and silently skipped when coercion

produces all-NaN or constant columns.

### `limen.experiment.reducer.CorrelationReducer.analyze_and_intervene`

<!-- reference: public-limen-experiment-reducer-correlationreducer-analyze-and-intervene source=limen/experiment/reducer/correlation_reducer.py region=signature:CorrelationReducer.analyze_and_intervene role=signature -->
```python
@override
def analyze_and_intervene(self, log: pl.DataFrame, msq: Any) -> list[dict[str, Any]]:
    ...
```

Analyze parameter correlations and return interventions.

Computes bootstrap correlations between each parameter and the

target metric. Emits remove_is for wrong-direction parameters

and keep_is suggestions for low-impact parameters.

### `limen.experiment.reducer.CorrelationReducer.get_state`

<!-- reference: public-limen-experiment-reducer-correlationreducer-get-state source=limen/experiment/reducer/correlation_reducer.py region=signature:CorrelationReducer.get_state role=signature -->
```python
@override
def get_state(self) -> dict[str, Any]:
    ...
```

Export state for checkpointing.

### `limen.experiment.reducer.CorrelationReducer.set_state`

<!-- reference: public-limen-experiment-reducer-correlationreducer-set-state source=limen/experiment/reducer/correlation_reducer.py region=signature:CorrelationReducer.set_state role=signature -->
```python
@override
def set_state(self, state: dict[str, Any]) -> None:
    ...
```

Restore state from checkpoint.

### `limen.experiment.reducer.FILTER_EXCLUDE_VALUE`

<!-- reference: public-limen-experiment-reducer-filter-exclude-value source=limen/experiment/reducer/filter_types.py region=definition:FILTER_EXCLUDE_VALUE role=definition -->
```python
FILTER_EXCLUDE_VALUE = 'exclude_value'
```

### `limen.experiment.reducer.FILTER_KEEP_BETWEEN`

<!-- reference: public-limen-experiment-reducer-filter-keep-between source=limen/experiment/reducer/filter_types.py region=definition:FILTER_KEEP_BETWEEN role=definition -->
```python
FILTER_KEEP_BETWEEN = 'keep_between'
```

### `limen.experiment.reducer.FILTER_KEEP_VALUES`

<!-- reference: public-limen-experiment-reducer-filter-keep-values source=limen/experiment/reducer/filter_types.py region=definition:FILTER_KEEP_VALUES role=definition -->
```python
FILTER_KEEP_VALUES = 'keep_values'
```

### `limen.experiment.reducer.FILTER_SAMPLE`

<!-- reference: public-limen-experiment-reducer-filter-sample source=limen/experiment/reducer/filter_types.py region=definition:FILTER_SAMPLE role=definition -->
```python
FILTER_SAMPLE = 'sample'
```

### `limen.experiment.reducer.FocusReducer`

<!-- reference: public-limen-experiment-reducer-focusreducer source=limen/experiment/reducer/focus_reducer.py region=signature:FocusReducer role=signature -->
```python
class FocusReducer(PruningStrategy):
    ...
```

Detect breakthrough results and narrow the parameter space for

exploitation using reversible named filters.

Monitors a target metric for values crossing a configurable

threshold. On breakthrough, narrows each parameter around the

winning combination and injects fine-grained linspace variations

for numeric parameters. Snaps back to full exploration after

focus_timeout rounds without improvement.

NOTE: Configuring multiple FocusReducer instances that operate on

overlapping parameters may cause conflicting filters. Use one

FocusReducer per experiment, or ensure parameter spaces are disjoint.

### `limen.experiment.reducer.FocusReducer.analyze_and_intervene`

<!-- reference: public-limen-experiment-reducer-focusreducer-analyze-and-intervene source=limen/experiment/reducer/focus_reducer.py region=signature:FocusReducer.analyze_and_intervene role=signature -->
```python
@override
def analyze_and_intervene(self, log: pl.DataFrame, msq: Any) -> list[dict[str, Any]]:
    ...
```

Analyze log for breakthrough results and return interventions.

When OFF, scans for metric crossing the threshold. When ON,

checks for improvement or timeout snap-back.

### `limen.experiment.reducer.FocusReducer.get_state`

<!-- reference: public-limen-experiment-reducer-focusreducer-get-state source=limen/experiment/reducer/focus_reducer.py region=signature:FocusReducer.get_state role=signature -->
```python
@override
def get_state(self) -> dict[str, Any]:
    ...
```

Export state for checkpointing.

### `limen.experiment.reducer.FocusReducer.set_state`

<!-- reference: public-limen-experiment-reducer-focusreducer-set-state source=limen/experiment/reducer/focus_reducer.py region=signature:FocusReducer.set_state role=signature -->
```python
@override
def set_state(self, state: dict[str, Any]) -> None:
    ...
```

Restore state from checkpoint.

### `limen.experiment.reducer.PruningStrategy`

<!-- reference: public-limen-experiment-reducer-pruningstrategy source=limen/experiment/reducer/pruning_strategy.py region=signature:PruningStrategy role=signature -->
```python
class PruningStrategy(ABC):
    ...
```

Abstract base for all pruning strategies (reducers).

Reducers are pure analysis components. They receive an experiment log

and an MSQ reference (for observability only), analyze the data, and

return a list of interventions communicated to MSQ for modifying the

param space.

### `limen.experiment.reducer.PruningStrategy.__init__`

<!-- reference: public-limen-experiment-reducer-pruningstrategy---init-- source=limen/experiment/reducer/pruning_strategy.py region=signature:PruningStrategy.__init__ role=signature -->
```python
def __init__(self, *, active: bool=True) -> None:
    ...
```

Initialize the PruningStrategy.

Args:

active (bool): Whether this reducer is enabled. Inactive reducers

return empty intervention lists from analyze_and_intervene

### `limen.experiment.reducer.PruningStrategy.active`

<!-- reference: public-limen-experiment-reducer-pruningstrategy-active source=limen/experiment/reducer/pruning_strategy.py region=signature:PruningStrategy.active role=signature -->
```python
@property
def active(self) -> bool:
    ...

@active.setter
def active(self, value: bool) -> None:
    ...
```

### `limen.experiment.reducer.REDUCER_REGISTRY`

<!-- reference: public-limen-experiment-reducer-reducer-registry source=limen/experiment/reducer/registry.py region=definition:REDUCER_REGISTRY role=definition -->
```python
REDUCER_REGISTRY: dict[str, type[PruningStrategy]] = {'budget': BudgetReducer, 'correlation': CorrelationReducer, 'focus': FocusReducer, 'sanity': SanityReducer, 'saturation': SaturationReducer}
```

### `limen.experiment.reducer.SanityReducer`

<!-- reference: public-limen-experiment-reducer-sanityreducer source=limen/experiment/reducer/sanity_reducer.py region=signature:SanityReducer role=signature -->
```python
class SanityReducer(PruningStrategy):
    ...
```

Remove parameter values whose trials produce null or NaN in the

target metric above a configurable threshold.

Counts both null and NaN (for float columns) in the metric column.

Values are pruned when missing_count / total > nan_threshold and

total >= min_observations.

Optionally emits suggestion interventions (logged but not dispatched)

for zero-metric, execution timeout, and warning rate signals.

### `limen.experiment.reducer.SanityReducer.analyze_and_intervene`

<!-- reference: public-limen-experiment-reducer-sanityreducer-analyze-and-intervene source=limen/experiment/reducer/sanity_reducer.py region=signature:SanityReducer.analyze_and_intervene role=signature -->
```python
@override
def analyze_and_intervene(self, log: pl.DataFrame, msq: Any) -> list[dict[str, Any]]:
    ...
```

Analyze experiment log and return interventions.

Returns remove_is interventions for NaN rate exceeding the

threshold, plus suggestion interventions for zero-metric,

execution timeout, and warning rate signals.

### `limen.experiment.reducer.SanityReducer.get_state`

<!-- reference: public-limen-experiment-reducer-sanityreducer-get-state source=limen/experiment/reducer/sanity_reducer.py region=signature:SanityReducer.get_state role=signature -->
```python
@override
def get_state(self) -> dict[str, Any]:
    ...
```

Export state for checkpointing.

### `limen.experiment.reducer.SanityReducer.set_state`

<!-- reference: public-limen-experiment-reducer-sanityreducer-set-state source=limen/experiment/reducer/sanity_reducer.py region=signature:SanityReducer.set_state role=signature -->
```python
@override
def set_state(self, state: dict[str, Any]) -> None:
    ...
```

Restore state from checkpoint.

### `limen.experiment.reducer.SaturationReducer`

<!-- reference: public-limen-experiment-reducer-saturationreducer source=limen/experiment/reducer/saturation_reducer.py region=signature:SaturationReducer role=signature -->
```python
class SaturationReducer(PruningStrategy):
    ...
```

Detect parameter values whose metric variance has collapsed and

apply partial pruning via named filters.

Uses Coefficient of Variation (CV = std / \|mean\|) over a rolling

window of recent observations. Saturated values get a sample filter

that retains retain_fraction of combinations for continued monitoring.

### `limen.experiment.reducer.SaturationReducer.analyze_and_intervene`

<!-- reference: public-limen-experiment-reducer-saturationreducer-analyze-and-intervene source=limen/experiment/reducer/saturation_reducer.py region=signature:SaturationReducer.analyze_and_intervene role=signature -->
```python
@override
def analyze_and_intervene(self, log: pl.DataFrame, msq: Any) -> list[dict[str, Any]]:
    ...
```

Analyze metric variance per parameter value and return interventions.

Computes CV for each parameter value over a rolling window.

Emits set_filter for saturated values and clear_filter for

values that recover from saturation.

### `limen.experiment.reducer.SaturationReducer.get_state`

<!-- reference: public-limen-experiment-reducer-saturationreducer-get-state source=limen/experiment/reducer/saturation_reducer.py region=signature:SaturationReducer.get_state role=signature -->
```python
@override
def get_state(self) -> dict[str, Any]:
    ...
```

Export state for checkpointing.

### `limen.experiment.reducer.SaturationReducer.set_state`

<!-- reference: public-limen-experiment-reducer-saturationreducer-set-state source=limen/experiment/reducer/saturation_reducer.py region=signature:SaturationReducer.set_state role=signature -->
```python
@override
def set_state(self, state: dict[str, Any]) -> None:
    ...
```

Restore state from checkpoint.

### `limen.experiment.reducer.TRIM_RANDOM`

<!-- reference: public-limen-experiment-reducer-trim-random source=limen/experiment/reducer/budget_reducer.py region=definition:TRIM_RANDOM role=definition -->
```python
TRIM_RANDOM = 'random'
```

### `limen.experiment.reducer.TRIM_WORST_FIRST`

<!-- reference: public-limen-experiment-reducer-trim-worst-first source=limen/experiment/reducer/budget_reducer.py region=definition:TRIM_WORST_FIRST role=definition -->
```python
TRIM_WORST_FIRST = 'worst_first'
```

### `limen.features.common_strategy_positions.POSITION_COLUMN`

<!-- reference: public-limen-features-common-strategy-positions-position-column source=limen/features/common_strategy_positions.py region=definition:POSITION_COLUMN role=definition -->
```python
POSITION_COLUMN = 'common_strategy_position'
```

### `limen.features.common_strategy_positions.common_strategy_positions`

<!-- reference: public-limen-features-common-strategy-positions-common-strategy-positions source=limen/features/common_strategy_positions.py region=signature:common_strategy_positions role=signature -->
```python
def common_strategy_positions(data: pl.DataFrame | pl.LazyFrame, strategy: str) -> pl.DataFrame | pl.LazyFrame:
    ...
```

Append one common-strategy position column without dropping warm-up rows.

Lazy execution evaluates the complete input frame once, so optimizations

cannot move a filter or slice ahead of the strategy's indicator history.

### `limen.features.price_lines.line_active_count`

<!-- reference: public-limen-features-price-lines-line-active-count source=limen/features/price_lines.py region=signature:line_active_count role=signature -->
```python
def line_active_count(long: LineArrays, short: LineArrays, n_rows: int) -> npt.NDArray[np.intp]:
    ...
```

### `limen.features.price_lines.line_recency`

<!-- reference: public-limen-features-price-lines-line-recency source=limen/features/price_lines.py region=signature:line_recency role=signature -->
```python
def line_recency(ends: npt.NDArray[np.intp], n_rows: int, lookback: int) -> npt.NDArray[np.float64]:
    ...
```

### `limen.features.volatility_barriers.volatility_barriers`

<!-- reference: public-limen-features-volatility-barriers-volatility-barriers source=limen/features/volatility_barriers.py region=signature:volatility_barriers role=signature -->
```python
def volatility_barriers(data: pl.DataFrame, *, span: int=24, regime_span: int=168, min_periods: int=24, take_profit_multiple: float=2.0, stop_loss_multiple: float=1.0, minimum_bps: float=10.0, maximum_bps: float=2000.0, low_regime_ratio: float=0.75, high_regime_ratio: float=1.25, low_multiplier: float=0.8, normal_multiplier: float=1.0, high_multiplier: float=1.2) -> pl.DataFrame:
    ...
```

Produce entry distances from strictly prior simple-return EWMA volatility.

### `limen.indicators.cdladvancedblock.candle_totals`

<!-- reference: public-limen-indicators-cdladvancedblock-candle-totals source=limen/indicators/cdladvancedblock.py region=signature:candle_totals role=signature -->
```python
def candle_totals(values: npt.NDArray[np.float64], period: int, start: int, count: int, initial: float=0.0) -> npt.NDArray[np.float64]:
    ...
```

### `limen.indicators.linearreg.linearreg_components`

<!-- reference: public-limen-indicators-linearreg-linearreg-components source=limen/indicators/linearreg.py region=signature:linearreg_components role=signature -->
```python
def linearreg_components(values: npt.NDArray[np.float64], period: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ...
```

### `limen.indicators.midprice.window_extrema`

<!-- reference: public-limen-indicators-midprice-window-extrema source=limen/indicators/midprice.py region=signature:window_extrema role=signature -->
```python
def window_extrema(high: npt.NDArray[np.float64], low: npt.NDArray[np.float64], period: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ...
```

### `limen.inference.sensor.PredictionReason`

<!-- reference: public-limen-inference-sensor-predictionreason source=limen/inference/_python_sensor.py region=definition:PredictionReason role=definition -->
```python
PredictionReason = Literal['warm-up', 'inside-training-window', 'null-features', 'sensor-error']
```

### `limen.log.ConfusionNotApplicableError`

<!-- reference: public-limen-log-confusionnotapplicableerror source=limen/log/_permutation_confusion_metrics.py region=signature:ConfusionNotApplicableError role=signature -->
```python
class ConfusionNotApplicableError(ValueError):
    ...
```

Recorded outputs carry no binary classification interpretation.

### `limen.log.permutation_prediction_performance`

<!-- reference: public-limen-log-permutation-prediction-performance source=limen/log/_permutation_prediction_performance.py region=signature:permutation_prediction_performance role=signature -->
```python
def permutation_prediction_performance(self: Any, round_id: int) -> pl.DataFrame:
    ...
```

Create prediction performance table based on round id.

Preparation failures propagate; retained round parameters are never replaced with defaults.

Args:

round_id (int): Round ID (i.e. nth permutation in an experiment)

Returns:

pl.DataFrame: Table with predictions, actuals, hit/miss, datetime, open, close, and price_change

### `limen.sfd.foundational_sfd.ridge_regressor.manifest`

<!-- reference: public-limen-sfd-foundational-sfd-ridge-regressor-manifest source=limen/sfd/foundational_sfd/ridge_regressor.py region=signature:manifest role=signature -->
```python
def manifest() -> MLManifest:
    ...
```

Build a next-return SFD with lagged returns and train-fitted scaling.

### `limen.sfd.foundational_sfd.ridge_regressor.params`

<!-- reference: public-limen-sfd-foundational-sfd-ridge-regressor-params source=limen/sfd/foundational_sfd/ridge_regressor.py region=signature:params role=signature -->
```python
def params() -> dict[str, list[object]]:
    ...
```

Return a small search space over native sklearn Ridge parameters.

### `limen.sfd.foundational_sfd.xgboost_regressor.manifest`

<!-- reference: public-limen-sfd-foundational-sfd-xgboost-regressor-manifest source=limen/sfd/foundational_sfd/xgboost_regressor.py region=signature:manifest role=signature -->
```python
def manifest() -> MLManifest:
    ...
```

Build independently selectable research groups with per-round preparation.

### `limen.sfd.foundational_sfd.xgboost_regressor.params`

<!-- reference: public-limen-sfd-foundational-sfd-xgboost-regressor-params source=limen/sfd/foundational_sfd/xgboost_regressor.py region=signature:params role=signature -->
```python
def params() -> dict[str, list[object]]:
    ...
```

Expose model defaults and a discrete research space over existing features.

### `limen.sfd.reference_architecture.DLinearRegressor`

<!-- reference: public-limen-sfd-reference-architecture-dlinearregressor source=limen/sfd/reference_architecture/dlinear_regressor.py region=signature:DLinearRegressor role=signature -->
```python
class DLinearRegressor(ReferenceModel):
    ...
```

DLinear regression model with train/evaluate interface.

Implements canonical DLinear semantics (Zeng et al. 2023): the lookback

window is decomposed by a centered moving average with replicate edge

padding into trend and remainder components, each component gets its own

linear head, and the head outputs are summed. The fit is the exact

minimizer of the ridge-regularized DLinear MSE objective, computed in

closed form by SVD, so training is deterministic and needs no seed.

The lookback window is read from the feature columns named

'ret_1_lag_{i}', ordered by lag descending so each row is the

window in time order (oldest bar first). All other feature columns are

ignored.

### `limen.sfd.reference_architecture.DLinearRegressor.evaluate`

<!-- reference: public-limen-sfd-reference-architecture-dlinearregressor-evaluate source=limen/sfd/reference_architecture/dlinear_regressor.py region=signature:DLinearRegressor.evaluate role=signature -->
```python
@override
def evaluate(self, data: dict[str, Any], inline_metrics: bool=True) -> dict[str, Any]:
    ...
```

Evaluate trained model on test data.

Args:

data (dict): Data dictionary with x_test, y_test, and optionally price_data_for_backtest

inline_metrics (bool): Whether to include `confusion_*` and `backtest_*` keys

Returns:

dict: Metrics dict, optionally with flattened `confusion_*` and `backtest_*` keys

### `limen.sfd.reference_architecture.DLinearRegressor.predict`

<!-- reference: public-limen-sfd-reference-architecture-dlinearregressor-predict source=limen/sfd/reference_architecture/dlinear_regressor.py region=signature:DLinearRegressor.predict role=signature -->
```python
@override
def predict(self, data: dict[str, Any]) -> dict[str, Any]:
    ...
```

Compute continuous predictions from feature data.

Args:

data (dict): Data dictionary with x_test holding the window columns

Returns:

dict: Prediction results with '_preds' key

### `limen.sfd.reference_architecture.DLinearRegressor.train`

<!-- reference: public-limen-sfd-reference-architecture-dlinearregressor-train source=limen/sfd/reference_architecture/dlinear_regressor.py region=signature:DLinearRegressor.train role=signature -->
```python
@override
def train(self, data: dict[str, Any], **params: Any) -> 'DLinearRegressor':
    ...
```

Fit the DLinear heads in closed form on provided data.

Args:

data (dict): Data dictionary with x_train (polars DataFrame holding

the lookback window columns) and y_train

**params: kernel_size (odd int) and alpha (non-negative float)

Returns:

DLinearRegressor: Self with fitted weights stored

### `limen.sfd.reference_architecture.DirectionSizingModel`

<!-- reference: public-limen-sfd-reference-architecture-directionsizingmodel source=limen/sfd/reference_architecture/direction_sizing.py region=signature:DirectionSizingModel role=signature -->
```python
class DirectionSizingModel(ReferenceModel):
    prediction_mode: ClassVar[Literal['binary', 'target_exposure']] = 'target_exposure'
    direction_model: ComponentEstimator
    sizing_model: ComponentEstimator
```

### `limen.sfd.reference_architecture.DirectionSizingModel.evaluate`

<!-- reference: public-limen-sfd-reference-architecture-directionsizingmodel-evaluate source=limen/sfd/reference_architecture/direction_sizing.py region=signature:DirectionSizingModel.evaluate role=signature -->
```python
@override
def evaluate(self, data: dict[str, object], inline_metrics: bool=True) -> dict[str, object]:
    ...
```

### `limen.sfd.reference_architecture.DirectionSizingModel.predict`

<!-- reference: public-limen-sfd-reference-architecture-directionsizingmodel-predict source=limen/sfd/reference_architecture/direction_sizing.py region=signature:DirectionSizingModel.predict role=signature -->
```python
@override
def predict(self, data: dict[str, object]) -> dict[str, object]:
    ...
```

### `limen.sfd.reference_architecture.DirectionSizingModel.train`

<!-- reference: public-limen-sfd-reference-architecture-directionsizingmodel-train source=limen/sfd/reference_architecture/direction_sizing.py region=signature:DirectionSizingModel.train role=signature -->
```python
@override
def train(self, data: dict[str, object], **params: object) -> DirectionSizingModel:
    ...
```

### `limen.sfd.reference_architecture.LightGBMBinary`

<!-- reference: public-limen-sfd-reference-architecture-lightgbmbinary source=limen/sfd/reference_architecture/lightgbm_binary.py region=signature:LightGBMBinary role=signature -->
```python
class LightGBMBinary(ReferenceModel):
    ...
```

LightGBM binary classifier with train/evaluate interface.

### `limen.sfd.reference_architecture.LightGBMBinary.__init__`

<!-- reference: public-limen-sfd-reference-architecture-lightgbmbinary---init-- source=limen/sfd/reference_architecture/lightgbm_binary.py region=signature:LightGBMBinary.__init__ role=signature -->
```python
def __init__(self, prediction_calibration_config: 'CalibrationConfig | None'=None) -> None:
    ...
```

### `limen.sfd.reference_architecture.LightGBMBinary.evaluate`

<!-- reference: public-limen-sfd-reference-architecture-lightgbmbinary-evaluate source=limen/sfd/reference_architecture/lightgbm_binary.py region=signature:LightGBMBinary.evaluate role=signature -->
```python
@override
def evaluate(self, data: dict[str, Any], inline_metrics: bool=True) -> dict[str, Any]:
    ...
```

Evaluate test predictions and preserve configured model output records.

### `limen.sfd.reference_architecture.LightGBMBinary.predict`

<!-- reference: public-limen-sfd-reference-architecture-lightgbmbinary-predict source=limen/sfd/reference_architecture/lightgbm_binary.py region=signature:LightGBMBinary.predict role=signature -->
```python
@override
def predict(self, data: dict[str, Any]) -> dict[str, Any]:
    ...
```

Compute binary predictions from feature data.

On the first call with calibration configured, fits the calibrator on

validation data and stores it for reuse. Subsequent calls (including

inference without val data) use the stored calibrator directly.

Args:

data (dict): Data dictionary. Training call requires x_val, y_val, x_test;

inference call requires only x_test

Returns:

dict: Prediction results with '_preds' and '_probs' keys; calibrated path

also includes 'optimal_threshold' and 'val_score'

### `limen.sfd.reference_architecture.LightGBMBinary.train`

<!-- reference: public-limen-sfd-reference-architecture-lightgbmbinary-train source=limen/sfd/reference_architecture/lightgbm_binary.py region=signature:LightGBMBinary.train role=signature -->
```python
@override
def train(self, data: dict[str, Any], **params: Any) -> 'LightGBMBinary':
    ...
```

Train LightGBM binary classifier on provided data.

Fits with early stopping against the validation split when one is

present and early_stopping_rounds is set; otherwise fits plainly on

the training split.

Args:

data (dict): Data dictionary with x_train, y_train, and optionally x_val, y_val

**params: LGBMClassifier hyperparameters plus early_stopping_rounds

Returns:

LightGBMBinary: Self with fitted model stored

### `limen.sfd.reference_architecture.LogRegBinary`

<!-- reference: public-limen-sfd-reference-architecture-logregbinary source=limen/sfd/reference_architecture/logreg_binary.py region=signature:LogRegBinary role=signature -->
```python
class LogRegBinary(ReferenceModel):
    ...
```

Logistic regression binary classifier with train/evaluate interface.

Declared non-deterministic because sklearn solver refits are not

bit-reproducible across BLAS builds and thread counts even with a fixed

random_state, so reconstruction validation must use relative tolerance.

### `limen.sfd.reference_architecture.LogRegBinary.__init__`

<!-- reference: public-limen-sfd-reference-architecture-logregbinary---init-- source=limen/sfd/reference_architecture/logreg_binary.py region=signature:LogRegBinary.__init__ role=signature -->
```python
def __init__(self, prediction_calibration_config: 'CalibrationConfig | None'=None) -> None:
    ...
```

### `limen.sfd.reference_architecture.LogRegBinary.evaluate`

<!-- reference: public-limen-sfd-reference-architecture-logregbinary-evaluate source=limen/sfd/reference_architecture/logreg_binary.py region=signature:LogRegBinary.evaluate role=signature -->
```python
@override
def evaluate(self, data: dict[str, Any], inline_metrics: bool=True) -> dict[str, Any]:
    ...
```

Evaluate trained model on test data.

Args:

data (dict): Data dictionary with x_test, y_test, and optionally price_data_for_backtest

inline_metrics (bool): Whether to include `confusion_*` and `backtest_*` keys

Returns:

dict: Metrics dict, optionally with flattened `confusion_*` and `backtest_*` keys

### `limen.sfd.reference_architecture.LogRegBinary.predict`

<!-- reference: public-limen-sfd-reference-architecture-logregbinary-predict source=limen/sfd/reference_architecture/logreg_binary.py region=signature:LogRegBinary.predict role=signature -->
```python
@override
def predict(self, data: dict[str, Any]) -> dict[str, Any]:
    ...
```

Compute binary predictions from feature data.

On the first call with calibration configured, fits the calibrator on

validation data and stores it for reuse. Subsequent calls (including

inference without val data) use the stored calibrator directly.

Args:

data (dict): Data dictionary. Training call requires x_val, y_val, x_test;

inference call requires only x_test

Returns:

dict: Prediction results with '_preds' and '_probs' keys; calibrated path

also includes 'optimal_threshold' and 'val_score'

### `limen.sfd.reference_architecture.LogRegBinary.train`

<!-- reference: public-limen-sfd-reference-architecture-logregbinary-train source=limen/sfd/reference_architecture/logreg_binary.py region=signature:LogRegBinary.train role=signature -->
```python
@override
def train(self, data: dict[str, Any], **params: Any) -> 'LogRegBinary':
    ...
```

Train logistic regression classifier on provided data.

Args:

data (dict): Data dictionary with x_train, y_train

**params: LogisticRegression hyperparameters

### `limen.sfd.reference_architecture.RandomBinary`

<!-- reference: public-limen-sfd-reference-architecture-randombinary source=limen/sfd/reference_architecture/random_binary.py region=signature:RandomBinary role=signature -->
```python
class RandomBinary(ReferenceModel):
    ...
```

Random binary classifier with train/evaluate interface.

### `limen.sfd.reference_architecture.RandomBinary.__init__`

<!-- reference: public-limen-sfd-reference-architecture-randombinary---init-- source=limen/sfd/reference_architecture/random_binary.py region=signature:RandomBinary.__init__ role=signature -->
```python
def __init__(self) -> None:
    ...
```

### `limen.sfd.reference_architecture.RandomBinary.evaluate`

<!-- reference: public-limen-sfd-reference-architecture-randombinary-evaluate source=limen/sfd/reference_architecture/random_binary.py region=signature:RandomBinary.evaluate role=signature -->
```python
@override
def evaluate(self, data: dict[str, Any], inline_metrics: bool=True) -> dict[str, Any]:
    ...
```

Evaluate random classifier on test data.

Args:

data (dict): Data dictionary with x_test, y_test, and optionally price_data_for_backtest

inline_metrics (bool): Whether to include `confusion_*` and `backtest_*` keys

Returns:

dict: Metrics dict, optionally with flattened `confusion_*` and `backtest_*` keys

### `limen.sfd.reference_architecture.RandomBinary.predict`

<!-- reference: public-limen-sfd-reference-architecture-randombinary-predict source=limen/sfd/reference_architecture/random_binary.py region=signature:RandomBinary.predict role=signature -->
```python
@override
def predict(self, data: dict[str, Any]) -> dict[str, Any]:
    ...
```

Compute random binary predictions.

Args:

data (dict): Data dictionary with x_test

Returns:

dict: Prediction results with '_preds' and '_probs' keys

### `limen.sfd.reference_architecture.RandomBinary.train`

<!-- reference: public-limen-sfd-reference-architecture-randombinary-train source=limen/sfd/reference_architecture/random_binary.py region=signature:RandomBinary.train role=signature -->
```python
@override
def train(self, data: dict[str, Any], **params: Any) -> 'RandomBinary':
    ...
```

Store random weights configuration. No actual training.

Args:

data (dict): Data dictionary (unused for random classifier)

**params: Must include 'random_weights' if overriding default

Returns:

RandomBinary: Self with configuration stored

### `limen.sfd.reference_architecture.ReferenceModel`

<!-- reference: public-limen-sfd-reference-architecture-referencemodel source=limen/sfd/reference_architecture/base.py region=signature:ReferenceModel role=signature -->
```python
class ReferenceModel(ABC):
    deterministic: bool = False
    prediction_mode: ClassVar[Literal['binary', 'target_exposure']] = 'binary'
```

Base class for class-based reference architecture models.

### `limen.sfd.reference_architecture.ReferenceModel.__init__`

<!-- reference: public-limen-sfd-reference-architecture-referencemodel---init-- source=limen/sfd/reference_architecture/base.py region=signature:ReferenceModel.__init__ role=signature -->
```python
def __init__(self) -> None:
    ...
```

### `limen.sfd.reference_architecture.RidgeRegressor`

<!-- reference: public-limen-sfd-reference-architecture-ridgeregressor source=limen/sfd/reference_architecture/ridge_regressor.py region=signature:RidgeRegressor role=signature -->
```python
class RidgeRegressor(ReferenceModel):
    ...
```

Sklearn ridge regression with Limen's train/predict/evaluate contract.

Each train call fits a fresh estimator on the training split only.

Solver behavior, validation, and warnings belong to sklearn. The

deterministic flag is conservative because stochastic solvers are allowed.

### `limen.sfd.reference_architecture.RidgeRegressor.__init__`

<!-- reference: public-limen-sfd-reference-architecture-ridgeregressor---init-- source=limen/sfd/reference_architecture/ridge_regressor.py region=signature:RidgeRegressor.__init__ role=signature -->
```python
def __init__(self) -> None:
    ...
```

### `limen.sfd.reference_architecture.RidgeRegressor.evaluate`

<!-- reference: public-limen-sfd-reference-architecture-ridgeregressor-evaluate source=limen/sfd/reference_architecture/ridge_regressor.py region=signature:RidgeRegressor.evaluate role=signature -->
```python
@override
def evaluate(self, data: dict[str, object], inline_metrics: bool=True) -> dict[str, object]:
    ...
```

Score continuous predictions, with optional standard directional metrics.

### `limen.sfd.reference_architecture.RidgeRegressor.predict`

<!-- reference: public-limen-sfd-reference-architecture-ridgeregressor-predict source=limen/sfd/reference_architecture/ridge_regressor.py region=signature:RidgeRegressor.predict role=signature -->
```python
@override
def predict(self, data: dict[str, object]) -> dict[str, _Predictions]:
    ...
```

Return continuous predictions; only x_test is required.

### `limen.sfd.reference_architecture.RidgeRegressor.train`

<!-- reference: public-limen-sfd-reference-architecture-ridgeregressor-train source=limen/sfd/reference_architecture/ridge_regressor.py region=signature:RidgeRegressor.train role=signature -->
```python
@override
def train(self, data: dict[str, object], **params: object) -> 'RidgeRegressor':
    ...
```

Fit sklearn Ridge using native constructor parameters.

### `limen.sfd.reference_architecture.RuleBasedStrategy`

<!-- reference: public-limen-sfd-reference-architecture-rulebasedstrategy source=limen/sfd/reference_architecture/rule_based.py region=signature:RuleBasedStrategy role=signature -->
```python
class RuleBasedStrategy(ReferenceModel):
    ...
```

Rule-based strategy that applies boolean predicate logic per bar to produce long/flat positions.

### `limen.sfd.reference_architecture.RuleBasedStrategy.__init__`

<!-- reference: public-limen-sfd-reference-architecture-rulebasedstrategy---init-- source=limen/sfd/reference_architecture/rule_based.py region=signature:RuleBasedStrategy.__init__ role=signature -->
```python
def __init__(self, sharpe_std_threshold: float=0.5, sharpe_degradation_threshold: float=0.3) -> None:
    ...
```

### `limen.sfd.reference_architecture.RuleBasedStrategy.evaluate`

<!-- reference: public-limen-sfd-reference-architecture-rulebasedstrategy-evaluate source=limen/sfd/reference_architecture/rule_based.py region=signature:RuleBasedStrategy.evaluate role=signature -->
```python
@override
def evaluate(self, data: dict[str, Any], inline_metrics: bool=True) -> dict[str, Any]:
    ...
```

Evaluate strategy across all splits and return rule-based metrics.

Args:

data (dict): Data dict with 'train', 'val', 'test' DataFrames and 'strategy' config

inline_metrics (bool): Unused — included for interface compatibility

Returns:

dict: Tier 1 position stats, Tier 2 backtest metrics per split,

Tier 3 stability metrics, and '_preds' key

### `limen.sfd.reference_architecture.RuleBasedStrategy.predict`

<!-- reference: public-limen-sfd-reference-architecture-rulebasedstrategy-predict source=limen/sfd/reference_architecture/rule_based.py region=signature:RuleBasedStrategy.predict role=signature -->
```python
@override
def predict(self, data: dict[str, Any]) -> dict[str, Any]:
    ...
```

Apply boolean logic tree to test split and return per-bar position signals.

Args:

data (dict): Data dict with 'test' DataFrame and 'strategy' config

Returns:

dict: {'_preds': np.ndarray} of 0/1 integer positions

### `limen.sfd.reference_architecture.RuleBasedStrategy.train`

<!-- reference: public-limen-sfd-reference-architecture-rulebasedstrategy-train source=limen/sfd/reference_architecture/rule_based.py region=signature:RuleBasedStrategy.train role=signature -->
```python
@override
def train(self, data: dict[str, Any], **params: Any) -> 'RuleBasedStrategy':
    ...
```

Own the selected rule configuration without learned fitting.

### `limen.sfd.reference_architecture.TabPFNBinary`

<!-- reference: public-limen-sfd-reference-architecture-tabpfnbinary source=limen/sfd/reference_architecture/tabpfn_binary.py region=signature:TabPFNBinary role=signature -->
```python
class TabPFNBinary(ReferenceModel):
    ...
```

TabPFN binary classifier with train/evaluate interface.

### `limen.sfd.reference_architecture.TabPFNBinary.__init__`

<!-- reference: public-limen-sfd-reference-architecture-tabpfnbinary---init-- source=limen/sfd/reference_architecture/tabpfn_binary.py region=signature:TabPFNBinary.__init__ role=signature -->
```python
def __init__(self, prediction_calibration_config: 'CalibrationConfig | None'=None) -> None:
    ...
```

### `limen.sfd.reference_architecture.TabPFNBinary.evaluate`

<!-- reference: public-limen-sfd-reference-architecture-tabpfnbinary-evaluate source=limen/sfd/reference_architecture/tabpfn_binary.py region=signature:TabPFNBinary.evaluate role=signature -->
```python
@override
def evaluate(self, data: dict[str, Any], inline_metrics: bool=True) -> dict[str, Any]:
    ...
```

Evaluate trained model on test data.

Args:

data (dict): Data dictionary with x_val, y_val, x_test, y_test,

and optionally price_data_for_backtest

inline_metrics (bool): Whether to include `confusion_*` and `backtest_*` keys

Returns:

dict: Metrics dict, optionally with flattened `confusion_*` and `backtest_*` keys

### `limen.sfd.reference_architecture.TabPFNBinary.predict`

<!-- reference: public-limen-sfd-reference-architecture-tabpfnbinary-predict source=limen/sfd/reference_architecture/tabpfn_binary.py region=signature:TabPFNBinary.predict role=signature -->
```python
@override
def predict(self, data: dict[str, Any]) -> dict[str, Any]:
    ...
```

Compute binary predictions with optional calibration and threshold tuning.

On the first call with calibration configured, fits the calibrator on

validation data and stores it for reuse. Subsequent calls (including

inference without val data) use the stored calibrator directly.

Args:

data (dict): Data dictionary. Training call requires x_val, y_val, x_test;

inference call requires only x_test

Returns:

dict: Prediction results with '_preds' and '_probs' keys; calibrated path

also includes 'optimal_threshold' and 'val_score'

### `limen.sfd.reference_architecture.TabPFNBinary.train`

<!-- reference: public-limen-sfd-reference-architecture-tabpfnbinary-train source=limen/sfd/reference_architecture/tabpfn_binary.py region=signature:TabPFNBinary.train role=signature -->
```python
@override
def train(self, data: dict[str, Any], **params: Any) -> 'TabPFNBinary':
    ...
```

Train TabPFN classifier on provided data.

Args:

data (dict): Data dictionary with x_train, y_train

**params: TabPFN hyperparameters including n_ensemble_configurations, device

Returns:

TabPFNBinary: Self with fitted model stored

### `limen.sfd.reference_architecture.XGBoostRegressor`

<!-- reference: public-limen-sfd-reference-architecture-xgboostregressor source=limen/sfd/reference_architecture/xgboost_regressor.py region=signature:XGBoostRegressor role=signature -->
```python
class XGBoostRegressor(ReferenceModel):
    ...
```

XGBoost regression model with train/evaluate interface.

### `limen.sfd.reference_architecture.XGBoostRegressor.evaluate`

<!-- reference: public-limen-sfd-reference-architecture-xgboostregressor-evaluate source=limen/sfd/reference_architecture/xgboost_regressor.py region=signature:XGBoostRegressor.evaluate role=signature -->
```python
@override
def evaluate(self, data: dict[str, Any], inline_metrics: bool=True) -> dict[str, Any]:
    ...
```

Evaluate test predictions and preserve the configured boosting count.

### `limen.sfd.reference_architecture.XGBoostRegressor.predict`

<!-- reference: public-limen-sfd-reference-architecture-xgboostregressor-predict source=limen/sfd/reference_architecture/xgboost_regressor.py region=signature:XGBoostRegressor.predict role=signature -->
```python
@override
def predict(self, data: dict[str, Any]) -> dict[str, Any]:
    ...
```

Predict continuous returns for the test features.

### `limen.sfd.reference_architecture.XGBoostRegressor.train`

<!-- reference: public-limen-sfd-reference-architecture-xgboostregressor-train source=limen/sfd/reference_architecture/xgboost_regressor.py region=signature:XGBoostRegressor.train role=signature -->
```python
@override
def train(self, data: dict[str, Any], **params: Any) -> 'XGBoostRegressor':
    ...
```

Fit the training split; early stopping uses only nonempty validation data.

### `limen.sfd.reference_architecture.direction_sizing`

<!-- reference: public-limen-sfd-reference-architecture-direction-sizing source=limen/sfd/reference_architecture/direction_sizing.py region=signature:direction_sizing role=signature -->
```python
def direction_sizing(data: dict[str, object], *, direction_factory: ComponentFactory=_DIRECTION_FACTORY, sizing_factory: ComponentFactory=_SIZING_FACTORY, direction_params: Mapping[str, JsonValue] | None=None, sizing_params: Mapping[str, JsonValue] | None=None, direction_features: Sequence[str] | None=None, sizing_features: Sequence[str] | None=None, conditional_size: bool=False, folds: int=5, min_train_samples: int=20, return_scale: float=0.01, max_size: float=1.0, seed: int=42, inline_metrics: bool=True) -> dict[str, object]:
    ...
```

### `limen.sfd.reference_architecture.direction_sizing.ComponentEstimator`

<!-- reference: public-limen-sfd-reference-architecture-direction-sizing-componentestimator source=limen/sfd/reference_architecture/direction_sizing.py region=signature:ComponentEstimator role=signature -->
```python
class ComponentEstimator(Protocol):
    ...
```

### `limen.sfd.reference_architecture.direction_sizing.ComponentEstimator.fit`

<!-- reference: public-limen-sfd-reference-architecture-direction-sizing-componentestimator-fit source=limen/sfd/reference_architecture/direction_sizing.py region=signature:ComponentEstimator.fit role=signature -->
```python
def fit(self, x: Array, y: Array) -> object:
    ...
```

### `limen.sfd.reference_architecture.direction_sizing.ComponentEstimator.predict`

<!-- reference: public-limen-sfd-reference-architecture-direction-sizing-componentestimator-predict source=limen/sfd/reference_architecture/direction_sizing.py region=signature:ComponentEstimator.predict role=signature -->
```python
def predict(self, x: Array) -> Array:
    ...
```

### `limen.sfd.reference_architecture.direction_sizing.ComponentFactory`

<!-- reference: public-limen-sfd-reference-architecture-direction-sizing-componentfactory source=limen/sfd/reference_architecture/direction_sizing.py region=signature:ComponentFactory role=signature -->
```python
class ComponentFactory(Protocol):
    deterministic: bool
```

### `limen.sfd.reference_architecture.direction_sizing.ComponentFactory.__call__`

<!-- reference: public-limen-sfd-reference-architecture-direction-sizing-componentfactory---call-- source=limen/sfd/reference_architecture/direction_sizing.py region=signature:ComponentFactory.__call__ role=signature -->
```python
def __call__(self, *, seed: int, **params: JsonValue) -> ComponentEstimator:
    ...
```

### `limen.sfd.reference_architecture.direction_sizing.FoldFeatures`

<!-- reference: public-limen-sfd-reference-architecture-direction-sizing-foldfeatures source=limen/experiment/_fold_preparation.py region=signature:FoldFeatures role=signature -->
```python
@dataclass(frozen=True)
class FoldFeatures:
    train: pl.DataFrame
    predict: pl.DataFrame
    train_rows: tuple[str, ...]
    predict_rows: tuple[str, ...]
```

### `limen.sfd.reference_architecture.direction_sizing.FoldPreparation`

<!-- reference: public-limen-sfd-reference-architecture-direction-sizing-foldpreparation source=limen/experiment/_fold_preparation.py region=signature:FoldPreparation role=signature -->
```python
@runtime_checkable
class FoldPreparation(Protocol):
    raw_features: pl.DataFrame
    row_ids: tuple[str, ...]
    identities: pl.DataFrame
    deterministic: bool
```

### `limen.sfd.reference_architecture.direction_sizing.FoldPreparation.fit_transform`

<!-- reference: public-limen-sfd-reference-architecture-direction-sizing-foldpreparation-fit-transform source=limen/experiment/_fold_preparation.py region=signature:FoldPreparation.fit_transform role=signature -->
```python
def fit_transform(self, train_rows: Sequence[str], predict_rows: Sequence[str]) -> FoldFeatures:
    ...
```

### `limen.sfd.reference_architecture.dlinear_regressor`

<!-- reference: public-limen-sfd-reference-architecture-dlinear-regressor source=limen/sfd/reference_architecture/dlinear_regressor.py region=signature:dlinear_regressor role=signature -->
```python
def dlinear_regressor(data: dict[str, Any], kernel_size: int=DEFAULT_KERNEL_SIZE, alpha: float=DEFAULT_ALPHA) -> dict[str, Any]:
    ...
```

Compute DLinear regression predictions and evaluation metrics.

Args:

data (dict): Data dictionary with x_train, y_train, x_test, y_test

kernel_size (int): Odd moving-average kernel size for the decomposition

alpha (float): Ridge regularization strength on the component heads

Returns:

dict: Results with continuous metrics, predictions, inline confusion metrics,

and backtest metrics when price_data_for_backtest is in data

### `limen.sfd.reference_architecture.lightgbm_binary`

<!-- reference: public-limen-sfd-reference-architecture-lightgbm-binary source=limen/sfd/reference_architecture/lightgbm_binary.py region=signature:lightgbm_binary role=signature -->
```python
def lightgbm_binary(data: dict[str, Any], objective: str | None=None, boosting_type: str='gbdt', num_leaves: int=31, max_depth: int=-1, learning_rate: float=0.1, n_estimators: int=100, subsample_for_bin: int=200000, min_split_gain: float=0.0, min_child_weight: float=0.001, min_child_samples: int=20, subsample: float=1.0, subsample_freq: int=0, colsample_bytree: float=1.0, reg_alpha: float=0.0, reg_lambda: float=0.0, class_weight: float | str | dict[int, float] | None=None, random_state: int=42, n_jobs: int | None=None, importance_type: str='split', early_stopping_rounds: int | None=DEFAULT_EARLY_STOPPING_ROUNDS, deterministic: bool=True, force_row_wise: bool=True, verbosity: int=-1, prediction_calibration_config: 'CalibrationConfig | None'=None) -> dict[str, Any]:
    ...
```

Compute LightGBM binary predictions and evaluation metrics.

Args:

data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test

objective (str \| None): LightGBM objective; None lets LightGBM select the

binary default. Must be a binary objective ('binary', 'cross_entropy')

boosting_type (str): Boosting algorithm ('gbdt', 'dart', 'rf')

num_leaves (int): Maximum tree leaves per booster

max_depth (int): Maximum tree depth, -1 for no limit

learning_rate (float): Boosting learning rate

n_estimators (int): Number of boosting rounds (upper bound under early stopping)

subsample_for_bin (int): Number of samples for histogram bin construction

min_split_gain (float): Minimum loss reduction to split a leaf

min_child_weight (float): Minimum sum of instance Hessian in a leaf

min_child_samples (int): Minimum samples in a leaf

subsample (float): Row subsampling ratio per boosting round

subsample_freq (int): Subsampling frequency, 0 disables

colsample_bytree (float): Feature subsampling ratio per tree

reg_alpha (float): L1 regularization weight

reg_lambda (float): L2 regularization weight

class_weight (float, str, or dict): Class weights. Numeric values keep the

legacy `{0: value, 1: 1}` shorthand; 'balanced'/dicts pass through to LightGBM

random_state (int): Random seed

n_jobs (int \| None): Number of parallel threads, None for LightGBM default

importance_type (str): Feature importance type ('split' or 'gain')

early_stopping_rounds (int \| None): Early-stopping patience against the

validation split; None or 0 disables early stopping

deterministic (bool): Enforce reproducible LightGBM training; pair with

force_row_wise and a fixed random_state

force_row_wise (bool): Force row-wise histogram building, required for

deterministic training

verbosity (int): LightGBM verbosity, -1 silences

prediction_calibration_config (CalibrationConfig \| None): Optional calibration config

Returns:

dict: Results with binary metrics, predictions, inline confusion metrics,

and backtest metrics when price_data_for_backtest is in data

### `limen.sfd.reference_architecture.meta_allocation.MetaAllocationModel`

<!-- reference: public-limen-sfd-reference-architecture-meta-allocation-metaallocationmodel source=limen/sfd/reference_architecture/meta_allocation.py region=signature:MetaAllocationModel role=signature -->
```python
class MetaAllocationModel(ReferenceModel):
    prediction_mode: ClassVar[Literal['binary', 'target_exposure']] = 'target_exposure'
```

### `limen.sfd.reference_architecture.meta_allocation.MetaAllocationModel.evaluate`

<!-- reference: public-limen-sfd-reference-architecture-meta-allocation-metaallocationmodel-evaluate source=limen/sfd/reference_architecture/meta_allocation.py region=signature:MetaAllocationModel.evaluate role=signature -->
```python
@override
def evaluate(self, data: dict[str, object], inline_metrics: bool=True) -> dict[str, object]:
    ...
```

### `limen.sfd.reference_architecture.meta_allocation.MetaAllocationModel.predict`

<!-- reference: public-limen-sfd-reference-architecture-meta-allocation-metaallocationmodel-predict source=limen/sfd/reference_architecture/meta_allocation.py region=signature:MetaAllocationModel.predict role=signature -->
```python
@override
def predict(self, data: dict[str, object]) -> dict[str, object]:
    ...
```

### `limen.sfd.reference_architecture.meta_allocation.MetaAllocationModel.train`

<!-- reference: public-limen-sfd-reference-architecture-meta-allocation-metaallocationmodel-train source=limen/sfd/reference_architecture/meta_allocation.py region=signature:MetaAllocationModel.train role=signature -->
```python
@override
def train(self, data: dict[str, object], **params: object) -> MetaAllocationModel:
    ...
```

### `limen.sfd.reference_architecture.meta_allocation.meta_allocation`

<!-- reference: public-limen-sfd-reference-architecture-meta-allocation-meta-allocation source=limen/sfd/reference_architecture/meta_allocation.py region=signature:meta_allocation role=signature -->
```python
def meta_allocation(data: dict[str, object], *, context: AllocationContext, folds: int=5, internal_validation_fraction: float=0.2, min_train_samples: int=20, features: Sequence[str] | None=None, include_primary_proba: bool=False, probability_factory: ComponentFactory=DIRECTION_FACTORY, sizing_factory: ComponentFactory=SIZING_FACTORY, probability_params: Mapping[str, JsonValue] | None=None, sizing_params: Mapping[str, JsonValue] | None=None, probability_threshold: float=0.5, return_scale: float=0.01, max_size: float=1.0, risk_budget_fraction: float | None=None, seed: int=42) -> dict[str, object]:
    ...
```

### `limen.sfd.reference_architecture.random_binary`

<!-- reference: public-limen-sfd-reference-architecture-random-binary source=limen/sfd/reference_architecture/random_binary.py region=signature:random_binary role=signature -->
```python
def random_binary(data: dict[str, Any], random_weights: float=0.5) -> dict[str, Any]:
    ...
```

Random binary classifier for testing and demonstration purposes.

Args:

data (dict): Data dictionary with x_test, y_test

random_weights (float): Probability weight for class 1 (0.0 to 1.0)

Returns:

dict: Results with binary metrics, predictions, inline confusion metrics,

and backtest metrics when price_data_for_backtest is in data

### `limen.sfd.reference_architecture.rule_based`

<!-- reference: public-limen-sfd-reference-architecture-rule-based source=limen/sfd/reference_architecture/rule_based.py region=signature:rule_based role=signature -->
```python
def rule_based(data: dict[str, Any], sharpe_std_threshold: float=0.5, sharpe_degradation_threshold: float=0.3) -> dict[str, Any]:
    ...
```

Apply a rule-based strategy to the given data and return evaluation metrics.

Args:

data (dict): Data dict with 'train', 'val', 'test' DataFrames and 'strategy' config

sharpe_std_threshold (float): Max sharpe_std for is_stable to be True

sharpe_degradation_threshold (float): Max sharpe_degradation for is_stable to be True

Returns:

dict: Rule-based metrics with Tier 1, Tier 2, Tier 3 keys and '_preds'

### `limen.sfd.reference_architecture.tabpfn_binary`

<!-- reference: public-limen-sfd-reference-architecture-tabpfn-binary source=limen/sfd/reference_architecture/tabpfn_binary.py region=signature:tabpfn_binary role=signature -->
```python
def tabpfn_binary(data: dict[str, Any], n_ensemble_configurations: int=4, device: str='cpu', prediction_calibration_config: 'CalibrationConfig | None'=None) -> dict[str, Any]:
    ...
```

Compute TabPFN binary classification with optional calibration and threshold tuning.

Args:

data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test

n_ensemble_configurations (int): Number of ensemble configurations for TabPFN

device (str): Device to run on ('cpu' or 'cuda')

prediction_calibration_config (CalibrationConfig \| None): Optional calibration config

Returns:

dict: Results with binary metrics, predictions, inline confusion metrics,

backtest metrics when price_data_for_backtest is in data,

and optionally 'optimal_threshold', 'val_score' when calibration is configured

### `limen.sfd.reference_architecture.xgboost_regressor`

<!-- reference: public-limen-sfd-reference-architecture-xgboost-regressor source=limen/sfd/reference_architecture/xgboost_regressor.py region=signature:xgboost_regressor role=signature -->
```python
def xgboost_regressor(data: dict[str, Any], learning_rate: float=0.01, max_depth: int=3, n_estimators: int=500, min_child_weight: int=10, subsample: float=0.6, colsample_bytree: float=0.6, gamma: float=0.5, reg_alpha: float=0.5, reg_lambda: float=5.0, objective: str='reg:squarederror', booster: str='gbtree', early_stopping_rounds: int | None=50, random_state: int=42, *, eval_metric: str | list[str] | None=None, max_leaves: int | None=None, max_bin: int | None=None, grow_policy: str | None=None, sampling_method: str | None=None, colsample_bylevel: float | None=None, colsample_bynode: float | None=None, max_delta_step: float | None=None, tree_method: str | None=None, num_parallel_tree: int | None=None, base_score: float | None=None, scale_pos_weight: float | None=None, n_jobs: int | None=None, verbosity: int | None=None, device: str | None=None, enable_categorical: bool | None=None, max_cat_to_onehot: int | None=None, max_cat_threshold: int | None=None, importance_type: str | None=None, interaction_constraints: str | list[list[int | str]] | None=None, monotone_constraints: str | tuple[int, ...] | dict[str, int] | None=None, multi_strategy: str | None=None, validate_parameters: bool | None=None) -> dict[str, Any]:
    ...
```

Train and evaluate a regressor, forwarding optional XGBoost controls.

### `limen.sfd.rule_based.RuleBasedConfig`

<!-- reference: public-limen-sfd-rule-based-rulebasedconfig source=limen/sfd/rule_based/config.py region=signature:RuleBasedConfig role=signature -->
```python
@dataclass
class RuleBasedConfig:
    conditions: list[dict[str, Any]]
    entry: str
```

Rule-based strategy configuration: boolean predicate conditions and entry signal id.

### `limen.sfd.rule_based.build_predicate`

<!-- reference: public-limen-sfd-rule-based-build-predicate source=limen/sfd/rule_based/predicates.py region=signature:build_predicate role=signature -->
```python
def build_predicate(condition: dict[str, Any], round_params: dict[str, Any]) -> pl.Expr:
    ...
```

Route a condition config dict to the appropriate predicate function.

Args:

condition (dict): Condition config with 'type' key and type-specific fields.

Optional 'persistence_n' or 'recency_n' fields wrap the result with

the corresponding temporal modifier.

round_params (dict): Parameter values for template substitution

Returns:

pl.Expr: Boolean predicate expression

### `limen.sfd.rule_based.common_strategies.STRATEGY_NAMES`

<!-- reference: public-limen-sfd-rule-based-common-strategies-strategy-names source=limen/sfd/rule_based/common_strategies.py region=definition:STRATEGY_NAMES role=definition -->
```python
STRATEGY_NAMES = tuple(STRATEGY_PARAMETERS)
```

### `limen.sfd.rule_based.common_strategies.STRATEGY_PARAMETERS`

<!-- reference: public-limen-sfd-rule-based-common-strategies-strategy-parameters source=limen/sfd/rule_based/common_strategies.py region=definition:STRATEGY_PARAMETERS role=definition -->
```python
STRATEGY_PARAMETERS: Mapping[str, Mapping[str, int | float]] = MappingProxyType({'buy_and_hold': MappingProxyType({}), 'sma_cross': MappingProxyType({'fast_period': 50, 'slow_period': 200}), 'ema_cross': MappingProxyType({'fast_period': 12, 'slow_period': 26}), 'macd': MappingProxyType({'fast_period': 12, 'slow_period': 26, 'signal_period': 9}), 'rsi_mean_reversion': MappingProxyType({'period': 14, 'entry_threshold': 30.0, 'exit_threshold': 70.0}), 'bollinger_mean_reversion': MappingProxyType({'period': 20, 'standard_deviations': 2.0, 'ddof': 0}), 'donchian_breakout': MappingProxyType({'entry_period': 20, 'exit_period': 10}), 'time_series_momentum': MappingProxyType({'period': 20})})
```

### `limen.sfd.rule_based.common_strategies.strategy_signals`

<!-- reference: public-limen-sfd-rule-based-common-strategies-strategy-signals source=limen/sfd/rule_based/common_strategies.py region=signature:strategy_signals role=signature -->
```python
def strategy_signals(data: pl.DataFrame, strategy: str) -> npt.NDArray[np.int8]:
    ...
```

Compute one fixed common-strategy baseline as completed-bar long/flat intent.

Rows retain input order. Prices must be finite and positive; when supplied,

datetime must be strictly increasing. Donchian also requires high and low.

Each call starts flat with no indicator history. Warm-up rows remain flat,

and stateful strategies hold until their own exit rule occurs. The returned

array has one 0/1 value per input row, including warm-up and the final row.

Execution lag, fills, fees and final liquidation belong to the backtester.

See docs/Common-Strategies.md for the fixed parameters and exact rules.

### `limen.utils.find_price_lines.LineArrays`

<!-- reference: public-limen-utils-find-price-lines-linearrays source=limen/utils/_computation_cache.py region=definition:LineArrays role=definition -->
```python
LineArrays = tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.float64]]
```

### `limen.utils.find_price_lines.price_line_arrays`

<!-- reference: public-limen-utils-find-price-lines-price-line-arrays source=limen/utils/find_price_lines.py region=signature:price_line_arrays role=signature -->
```python
def price_line_arrays(close: npt.ArrayLike, max_duration_hours: int, min_height_pct: float) -> tuple[LineArrays, LineArrays]:
    ...
```

### `limen.yaml.CompiledSFD`

<!-- reference: public-limen-yaml-compiledsfd source=limen/yaml/compiler.py region=signature:CompiledSFD role=signature -->
```python
class CompiledSFD:
    ...
```

SFD-compatible object built from a validated YAML experiment dict.

### `limen.yaml.CompiledSFD.__init__`

<!-- reference: public-limen-yaml-compiledsfd---init-- source=limen/yaml/compiler.py region=signature:CompiledSFD.__init__ role=signature -->
```python
def __init__(self, yaml_dict: dict[str, Any], *, project_root: str | Path | None=None) -> None:
    ...
```

### `limen.yaml.CompiledSFD.manifest`

<!-- reference: public-limen-yaml-compiledsfd-manifest source=limen/yaml/compiler.py region=signature:CompiledSFD.manifest role=signature -->
```python
def manifest(self) -> Manifest:
    ...
```

Return the compiled Manifest, built once and cached.

### `limen.yaml.CompiledSFD.params`

<!-- reference: public-limen-yaml-compiledsfd-params source=limen/yaml/compiler.py region=signature:CompiledSFD.params role=signature -->
```python
def params(self) -> dict[str, list[Any]]:
    ...
```

Return the parameter search space.

### `limen.yaml.GitError`

<!-- reference: public-limen-yaml-giterror source=limen/yaml/errors.py region=signature:GitError role=signature -->
```python
class GitError(Exception):
    ...
```

Raised when Git operations fail or the YAML file is not in a repository.

### `limen.yaml.GitError.__init__`

<!-- reference: public-limen-yaml-giterror---init-- source=limen/yaml/errors.py region=signature:GitError.__init__ role=signature -->
```python
def __init__(self, path: str, message: str) -> None:
    ...
```

### `limen.yaml.ProfileResult`

<!-- reference: public-limen-yaml-profileresult source=limen/yaml/profiler.py region=signature:ProfileResult role=signature -->
```python
@dataclass
class ProfileResult:
    total_permutations: int
    param_cardinalities: dict[str, int]
    complexity_rating: str
    sample_permutations_attempted: int = 0
    sample_permutations_completed: int = 0
    sample_time_seconds_per_permutation: float | None = None
    warnings: list[str] = field(default_factory=list[str])
    errors: list[str] = field(default_factory=list[str])
    data_quality_warnings: list[str] = field(default_factory=list[str])
```

Summary of a YAML experiment profile run.

### `limen.yaml.ResolutionError`

<!-- reference: public-limen-yaml-resolutionerror source=limen/yaml/errors.py region=signature:ResolutionError role=signature -->
```python
class ResolutionError(Exception):
    ...
```

Raised when a namespaced path cannot be resolved to a Python object.

### `limen.yaml.ResolutionError.__init__`

<!-- reference: public-limen-yaml-resolutionerror---init-- source=limen/yaml/errors.py region=signature:ResolutionError.__init__ role=signature -->
```python
def __init__(self, path: str, allowed_namespaces: list[str]) -> None:
    ...
```

### `limen.yaml.VERSION`

<!-- reference: public-limen-yaml-version source=limen/yaml/schema.py region=definition:VERSION role=definition -->
```python
VERSION = '1.0'
```

### `limen.yaml.ValidationError`

<!-- reference: public-limen-yaml-validationerror source=limen/yaml/errors.py region=signature:ValidationError role=signature -->
```python
class ValidationError(Exception):
    ...
```

Raised when YAML validation fails; carries all collected errors.

### `limen.yaml.ValidationError.__init__`

<!-- reference: public-limen-yaml-validationerror---init-- source=limen/yaml/errors.py region=signature:ValidationError.__init__ role=signature -->
```python
def __init__(self, errors: list['YAMLError']) -> None:
    ...
```

### `limen.yaml.ValidationResult`

<!-- reference: public-limen-yaml-validationresult source=limen/yaml/validator.py region=signature:ValidationResult role=signature -->
```python
@dataclass
class ValidationResult:
    valid: bool
    errors: list[YAMLError] = field(default_factory=list[YAMLError])
    warnings: list[YAMLError] = field(default_factory=list[YAMLError])
    mode: str = 'development'
```

Result of a YAML validation pass.

### `limen.yaml.YAMLError`

<!-- reference: public-limen-yaml-yamlerror source=limen/yaml/errors.py region=signature:YAMLError role=signature -->
```python
@dataclass
class YAMLError:
    message: str
    path: str = ''
    line: int | None = None
    column: int | None = None
    suggestion: str | None = None
```

A single validation error or warning with location and suggestion.

### `limen.yaml.build_manifest`

<!-- reference: public-limen-yaml-build-manifest source=limen/yaml/compiler.py region=signature:build_manifest role=signature -->
```python
def build_manifest(yaml_dict: dict[str, Any]) -> Manifest:
    ...
```

Build the declared ML or rule-based manifest.

### `limen.yaml.build_search_strategy`

<!-- reference: public-limen-yaml-build-search-strategy source=limen/yaml/compiler.py region=signature:build_search_strategy role=signature -->
```python
def build_search_strategy(yaml_dict: dict[str, Any]) -> RandomStrategy | GridStrategy | SobolStrategy:
    ...
```

Build the declared search, requiring a literal seed for shuffled YAML grids.

### `limen.yaml.find_project_root`

<!-- reference: public-limen-yaml-find-project-root source=limen/yaml/config.py region=signature:find_project_root role=signature -->
```python
def find_project_root(start: Path) -> Path | None:
    ...
```

Walk up the directory tree looking for limen.toml.

Args:

start (Path): Directory to start searching from

Returns:

Path \| None: The directory containing limen.toml, or None if not found

### `limen.yaml.get_store_path`

<!-- reference: public-limen-yaml-get-store-path source=limen/yaml/config.py region=signature:get_store_path role=signature -->
```python
def get_store_path(start: Path) -> Path:
    ...
```

Locate the manifest store by finding the project root.

Args:

start (Path): Directory to start searching from

Returns:

Path: Path to manifests/committed/

Raises:

FileNotFoundError: If no limen.toml is found in the directory tree

### `limen.yaml.make_covering_array`

<!-- reference: public-limen-yaml-make-covering-array source=limen/yaml/profiler.py region=signature:make_covering_array role=signature -->
```python
def make_covering_array(params: dict[str, list[Any]], seed: int=42) -> list[dict[str, Any]]:
    ...
```

Build a randomised strength-1 covering array for the given parameter space.

Every value of every parameter appears in at least one permutation.

N = max cardinality across all params. Each parameter's column is filled

with round-robin values then shuffled independently, removing systematic

correlation between parameter assignments.

Args:

params (dict): Parameter name to list of values

seed (int): Random seed for reproducibility

Returns:

list[dict]: N permutations, each a dict of param name to sampled value

### `limen.yaml.parse`

<!-- reference: public-limen-yaml-parse source=limen/yaml/parser.py region=signature:parse role=signature -->
```python
def parse(source: str | Path) -> tuple[dict[str, Any], list[YAMLError]]:
    ...
```

Parse a YAML experiment file or string into a dict.

Mapping-valued keys produce a located parse error before construction.

Args:

source (str \| Path): Path instance to read from disk, or raw YAML string

Returns:

tuple[dict, list[YAMLError]]: Parsed dict and list of parse errors (empty on success)

### `limen.yaml.profile`

<!-- reference: public-limen-yaml-profile source=limen/yaml/profiler.py region=signature:profile role=signature -->
```python
def profile(compiled_sfd: 'CompiledSFD') -> ProfileResult:
    ...
```

Profile a compiled YAML experiment.

Computes static fields (permutation count, complexity) always. Fetches

data via the experiment's own data_source and runs a randomised strength-1

covering array of permutations to record timing and surface data quality

warnings.

Args:

compiled_sfd (CompiledSFD): Compiled SFD built from a validated yaml_dict

Returns:

ProfileResult: Static fields always populated; runtime fields populated

when data loads successfully and at least one permutation completes

### `limen.yaml.read_limen_toml`

<!-- reference: public-limen-yaml-read-limen-toml source=limen/yaml/config.py region=signature:read_limen_toml role=signature -->
```python
def read_limen_toml(project_root: Path) -> dict[str, Any]:
    ...
```

Read limen.toml from the project root.

Args:

project_root (Path): Directory containing limen.toml

Returns:

dict: Parsed contents of limen.toml

### `limen.yaml.resolve`

<!-- reference: public-limen-yaml-resolve source=limen/yaml/resolver.py region=signature:resolve role=signature -->
```python
def resolve(path: str) -> Any:
    ...
```

Resolve a dotted path without leaving the allowed namespaces.

### `limen.yaml.resolve_manifest_uri`

<!-- reference: public-limen-yaml-resolve-manifest-uri source=limen/yaml/store.py region=signature:resolve_manifest_uri role=signature -->
```python
def resolve_manifest_uri(uri: str, start: Path) -> tuple[Path, Path]:
    ...
```

Return (manifest_path, project_root) for a full or unique short URI.

Search from start and verify stored lineage and canonical content against

the complete filename ID. Malformed input, missing project/files and

ambiguity raise ValueError. Integrity failures use "Integrity check failed",

chaining stored canonicalization TypeError/ValueError.

Verification covers this read, not later changes to the returned path.

### `limen.yaml.validate`

<!-- reference: public-limen-yaml-validate source=limen/yaml/validator.py region=signature:validate role=signature -->
```python
def validate(yaml_dict: dict[str, Any]) -> ValidationResult:
    ...
```

Collect YAML errors and warnings; validity requires no errors.
