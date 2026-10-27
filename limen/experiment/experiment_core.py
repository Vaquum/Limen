import limen.experiment._holdout_research as _holdout
import limen.experiment._holdout_checkpoint as _checkpoint_proof
import limen.experiment._round_storage as _storage
import limen.experiment._serial_resume as _serial
import limen.experiment._result_fields as _fields
import limen.experiment._round_errors as _errors
import csv
import importlib.metadata
import json
import logging
import signal
import time
import warnings
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import polars as pl
from tqdm import tqdm

from limen.experiment._walk_forward_run import WalkForwardRun, validate_walk_forward_resume
from limen.experiment._fold_evidence import add_metadata as add_fold_metadata
from limen.experiment._fixed_trade_sources import fixed_trade_sources
from limen.experiment.acceptance_report import acceptance_report
from limen.experiment._acceptance_settings import pbo_blocks
from limen.experiment.convergence_report import convergence_report, convergence_warning, convergence_header, convergence_warnings
from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.errors import StrictModeError
from limen.experiment._grid_resume import add_grid_metadata, complete_failed_header, validate_factorized_grid, validate_grid_resume
from limen.experiment._recording import prepare_recording, project_record
from limen.experiment._objective_run import add_objective_metadata, finalize_objective_result, objective_frame, validate_objective_header, validate_objective_reducers, validate_objective_resume
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.msq import MSQ
from limen.experiment._filter_resume import validate_filter_resume
from limen.experiment.param_domain import ParamDomain
from limen.experiment.reducer.pruning_strategy import PruningStrategy
from limen.experiment.param_search.search_strategy import SearchStrategy
from limen.utils.param_space import ParamSpace
from limen.utils._computation_cache import computation_reuse
from limen.log import ConfusionNotApplicableError
from limen.experiment.manifest_core import Manifest, MLManifest, RuleBasedManifest
from limen.sfd.rule_based._temporal import validate_temporal_conditions
from limen.yaml.store import canonical_manifest_id
import limen.experiment._parallel_rounds as _parallel

import limen.log.log as _log_module

logger = logging.getLogger(__name__)


def _validate_prep_each_round_arg(value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError('UniversalExperimentLoop prep_each_round must be a bool.')

STANDARD_RUN_LOG_BATCH_SIZE = 1000


class UniversalExperimentLoop:

    def __init__(self,
                 *,
                 data: pl.DataFrame | None = None,
                 sfd: Any = None,
                 search_strategy: SearchStrategy | None = None,
                 pruning_strategies: list[PruningStrategy] | None = None,
                 feedback_interval: int = 100,
                 checkpoint_interval: int = 1000,
                 experiment_dir: str | Path | None = None,
                 intra_callback: Callable[[Any, MSQ], None] | None = None,
                 test_mode: bool = False,
                 yaml_reference: dict[str, Any] | None = None,
                 project_root: str | Path | None = None, on_round_error: str = 'abort', max_round_errors: int | None = None) -> None:

        '''Configure the manifest or custom SFD and optional MSQ persistence.'''

        super().__init__()
        self._round_error_policy = _errors.RoundErrorPolicy(on_round_error, max_round_errors)

        if sfd is None:
            raise ValueError('UniversalExperimentLoop sfd is required')

        self._sfd_module_name = getattr(sfd, '__name__', None)
        self.params = sfd.params()
        self.manifest = None
        self.prep: Callable[..., dict[str, Any]] | None = None
        self.model: Callable[..., dict[str, Any]] | None = None

        bound_manifest, self.data, self.prep, self.model = _holdout.initialize_loop(self, sfd, data, test_mode=test_mode,
                                 project_root=project_root, experiment_dir=experiment_dir)
        self.manifest = cast(Manifest | None, bound_manifest)

        self._manifest_functions = (self.prep, self.model)
        self.extras: list[Any] = []
        self.models: list[Any] = []
        self._shutdown_requested: bool = False
        self._pause_requested: bool = False
        self._search_strategy = search_strategy
        self._pruning_strategies = pruning_strategies or []
        self._feedback_interval = feedback_interval
        self._checkpoint_interval = checkpoint_interval
        self._experiment_dir = Path(experiment_dir) if experiment_dir else None
        self._intra_callback = intra_callback
        self._yaml_reference, self._compiled_yaml, self._workers, self._parallel_sources = _parallel.configuration(sfd, yaml_reference)
        self._walk_forward: WalkForwardRun | None = None
        self._record_execution = False
        self._record_model_outputs = False
        self._recording: dict[str, bool] | None = None
        self._factorize = False
        self.round_params: list[dict[str, Any]] = []
        self.preds: list[Any] = []
        self.scalers: list[Any] = []
        self._alignment: list[Any] = []
        self.experiment_log: pl.DataFrame | None = None
        self.convergence_report: dict[str, object] | None = None
        self._convergence_caught: list[warnings.WarningMessage] = []
        self.param_space: ParamSpace | None = None
        self._log: _log_module.Log | None = None
        self.experiment_confusion_metrics: pl.DataFrame | None = None
        self.experiment_backtest_results: pl.DataFrame | None = None
        self.experiment_parameter_correlation: Callable[..., pl.DataFrame] | None = None
        self._clear_post_processing_outputs()

    @property
    def fold_results(self) -> pl.DataFrame:
        return self._walk_forward.results if self._walk_forward is not None else pl.DataFrame()

    check_parallel_admission, _admit_round_errors = _parallel.admit_parallel, _errors.admit

    @property
    def interrupted(self) -> bool:
        return self._shutdown_requested

    @_holdout.run_guard
    @_errors.guarded
    def run(self, experiment_name: str, n_permutations: int = 10000,
            prep_each_round: bool = False, random_search: bool = True, maintain_details_in_params: bool = False,
            context_params: dict[str, object] | None = None, params: Callable[[], dict[str, object]] | None = None,
            prep: Callable[..., dict[str, object]] | None = None, model: Callable[..., dict[str, object]] | None = None,
            resume: bool = False, post_processing: bool = False,
            progress_bar: bool = True, record_execution: bool = False,
            record_model_outputs: bool = False, factorize: bool = False, workers: int | None = None,
            recording: Mapping[str, bool] | None = None) -> None:
        applicable = ('predictions', 'alignment')
        if resume and self.manifest is not None:
            domains = self._search_strategy.domain.params if self._search_strategy is not None else self.params
            point = {key: values[0] for key, values in domains.items()} | (context_params or {})
            if self.manifest.resolve_trade_policy(point) is not None:
                applicable += ('trade_ledger',)
            if self.manifest.split_walk_forward is not None:
                applicable += ('fold_details',)
        self._recording = prepare_recording(recording, reference=self._yaml_reference,
                                            compiled_reference=self._compiled_yaml,
                                            directory=self._experiment_dir, has_search=self._search_strategy is not None,
                                            resume=resume, acceptance=getattr(self.manifest, 'acceptance', None), applicable=applicable)
        if self._recording is not None and self._yaml_reference is None:
            self._yaml_reference = self._compiled_yaml
        original_prep, original_model = _parallel.begin_run(self, workers, prep_each_round, resume, post_processing, factorize, context_params, (params, prep, model))
        self._walk_forward = None
        self.convergence_report = None
        try:
            parameter_names = self._search_strategy.domain.keys if self._search_strategy is not None else self.params
            reuse_sources = self._workers == 1 and all(value is None for value in (params, prep, model)) and (self.prep, self.model) == self._manifest_functions and any(isinstance(self.manifest, builtin) and all(getattr(method, '__self__', None) is self.manifest and getattr(method, '__func__', None) is original for method, original in ((self.manifest.prepare_data, builtin.prepare_data), (self.manifest.run_model, builtin.run_model))) for builtin in (MLManifest, RuleBasedManifest))
            source_context = fixed_trade_sources(self.manifest, [*parameter_names, *(context_params or {})]) if reuse_sources else nullcontext()
            with computation_reuse(), source_context, convergence_warnings() if self._search_strategy is None else nullcontext(self._convergence_caught) as self._convergence_caught:
                self._run(experiment_name, n_permutations, prep_each_round,
                          random_search, maintain_details_in_params, context_params,
                          params, prep, model, resume, post_processing, progress_bar,
                          record_execution, record_model_outputs, factorize)
        finally:
            self._parallel_sources, self._convergence_caught = None, []
            if self._walk_forward is not None:
                self._walk_forward.finish()
                self.prep, self.model = original_prep, original_model
        if self._walk_forward is not None:
            _ = acceptance_report(self._walk_forward.writer.path.parent,
                                  acceptance=self._walk_forward.manifest.acceptance)
        parameter_columns = self._search_strategy.domain.keys if self._search_strategy is not None else list(self.params)
        self.convergence_report = convergence_report(self.experiment_log if self.experiment_log is not None else pl.DataFrame(), parameter_columns=list(dict.fromkeys([*parameter_columns, *(context_params or {})])))
        if self._experiment_dir is not None:
            _ = (self._experiment_dir / 'convergence_report.json').write_text(json.dumps(self.convergence_report, indent=2, allow_nan=False))
        logger.info('Convergence-warning percentage: %s; observed=%s; unavailable=%s', self.convergence_report['convergence_warning_pct'], self.convergence_report['observed_rounds'], self.convergence_report['unavailable_rounds'])

    def _run(self,
            experiment_name: str,
            n_permutations: int = 10000,
            prep_each_round: bool = False,
            random_search: bool = True,
            maintain_details_in_params: bool = False,
            context_params: dict[str, Any] | None = None,
            params: Callable[[], dict[str, Any]] | None = None,
            prep: Callable[..., dict[str, Any]] | None = None,
            model: Callable[..., dict[str, Any]] | None = None,
            resume: bool = False,
            post_processing: bool = False,
            progress_bar: bool = True,
            record_execution: bool = False,
            record_model_outputs: bool = False, factorize: bool = False) -> None:

        '''Run manifest/custom rounds; MSQ owns sampling; manifests require prep_each_round=True.'''

        if self._search_strategy is not None and (overrides := [name for name, value in (('params', params), ('prep', prep), ('model', model)) if value is not None]):
            raise ValueError(f'MSQ runs reject run overrides {overrides}; configure custom functions on the SFD')
        if params is not None and self._search_strategy is None:
            self.params = params()
        domains = self._search_strategy.domain.params if self._search_strategy is not None else self.params
        _fields.validate_parameters(domains, context_params, self.manifest, legacy_details=maintain_details_in_params is True and self._search_strategy is None)
        validate_objective_reducers(self.manifest, self._pruning_strategies)
        if isinstance(self.manifest, RuleBasedManifest) and self.manifest.strategy is not None:
            validate_temporal_conditions(self.manifest.strategy.conditions, domains | {key: [value] for key, value in (context_params or {}).items()})
        if type(record_execution) is not bool:
            raise ValueError('record_execution must be a bool')
        if record_execution and (self._search_strategy is None or self._experiment_dir is None):
            raise ValueError('record_execution=True requires search_strategy and experiment_dir')
        self._record_execution = record_execution
        if type(record_model_outputs) is not bool:
            raise TypeError('record_model_outputs must be a bool')
        if record_model_outputs and (self._search_strategy is None or self._experiment_dir is None):
            raise ValueError('record_model_outputs=True requires search_strategy and experiment_dir')
        self._record_model_outputs = record_model_outputs
        if type(factorize) is not bool:
            raise ValueError('factorize must be a bool')
        if factorize and self._search_strategy is None:
            raise ValueError('factorize requires a search_strategy')
        self._factorize = factorize

        self.round_params = []
        self.models = []
        self.extras = []
        self.preds = []
        self.scalers = []
        self._alignment = []
        self.experiment_log = None
        self._clear_post_processing_outputs()

        if getattr(self.manifest, 'split_walk_forward', None) is not None:
            if not isinstance(self.manifest, Manifest):
                raise ValueError('split_walk_forward requires a Manifest')
            self._walk_forward = WalkForwardRun(
                self.manifest, self.data,
                self._experiment_dir or Path(experiment_name).parent, promotion=bool(self._recording and self._recording['fold_details']),
            )
            if not resume and self._walk_forward.writer.path.exists():
                raise FileExistsError('split_walk_forward return track already exists; resume or choose a fresh output directory')
            self.prep, self.model = self._walk_forward.prepare, self._walk_forward.evaluate

        if resume and self._search_strategy is None:
            raise ValueError('UniversalExperimentLoop resume=True is only supported with a search_strategy.')

        if self._search_strategy is not None:
            self._run_with_msq(
                experiment_name=experiment_name,
                n_permutations=n_permutations,
                context_params=context_params,
                resume=resume,
                post_processing=post_processing,
                progress_bar=progress_bar,
            )
            return

        _validate_prep_each_round_arg(prep_each_round)

        if self.manifest is not None:
            if prep is not None or model is not None:
                raise ValueError(
                    'UniversalExperimentLoop Cannot override prep/model when SFD has manifest.'
                )
            if not prep_each_round:
                raise ValueError(
                    'UniversalExperimentLoop prep_each_round must be True for manifest-driven SFDs.'
                )

        if prep is not None:
            self.prep = prep

        if model is not None:
            self.model = model

        if self.prep is None or self.model is None:
            raise ValueError('UniversalExperimentLoop prep and model functions must be configured')

        self.param_space = ParamSpace(params=self.params, n_permutations=n_permutations, sample=random_search)

        if self._experiment_dir is not None:
            self._experiment_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self._experiment_dir / f'{Path(experiment_name).name}.csv'
        else:
            csv_path = Path(f'{experiment_name}.csv')

        retain_round_artifacts = post_processing
        results_accumulator: list[dict[str, Any]] = []
        log_batches: list[pl.DataFrame] = []

        csv_header: list[str] | None = None
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with csv_path.open('r', newline='') as f:
                csv_header = next(csv.reader(f), None)
            if not csv_header or not any(col.strip() for col in csv_header):
                raise ValueError(
                    f'UniversalExperimentLoop Existing results CSV has no header: {csv_path}'
                )
            if getattr(self.manifest, 'ablation_config', None) is not None and '_dropped_features' not in csv_header:
                raise ValueError('UniversalExperimentLoop Cannot append ablation results without _dropped_features; start a new results file.')
            validate_objective_header(self.manifest, csv_header)
            csv_header = convergence_header(csv_path, csv_header)

        data_dict: dict[str, Any] = {}
        _pending_csv_rows: list[dict[str, Any]] = []

        for i in tqdm(range(n_permutations), disable=not progress_bar):

            start_time = time.time()

            round_params = self.param_space.generate(random_search=random_search)
            if round_params is None:
                raise ValueError('UniversalExperimentLoop parameter space exhausted before completing all rounds')

            if context_params is not None:
                round_params.update(context_params)

            if maintain_details_in_params is True:
                round_params['_experiment_details'] = {
                    'current_index': i,
                }

            warning_start = len(self._convergence_caught)
            try:
                if prep_each_round is True or i == 0:
                    data_dict = self.prep(self.data, round_params=round_params)
                if self._walk_forward is not None:
                    data_dict['_record_execution'] = self._record_execution
                    data_dict['_record_model_outputs'] = self._record_model_outputs
                round_results = self.model(data=data_dict, round_params=round_params)
                round_succeeded = True
            except StrictModeError as exc:
                logger.error('Round %d failed strict mode check: %s', i, exc)
                round_results: dict[str, Any] = {'strict_mode_error': str(exc) or type(exc).__name__}
                round_succeeded = False

            if maintain_details_in_params is True:
                round_params.pop('_experiment_details')

            _fields.validate_result(round_params, round_results, succeeded=round_succeeded)
            if round_succeeded:
                if retain_round_artifacts:
                    self._alignment.append(data_dict['_alignment'])

                if 'extras' in round_results:
                    if retain_round_artifacts:
                        self.extras.append(round_results['extras'])
                    round_results.pop('extras')

                if 'models' in round_results:
                    if retain_round_artifacts:
                        self.models.append(round_results['models'])
                    round_results.pop('models')

                if '_preds' in round_results:
                    if retain_round_artifacts:
                        self.preds.append(round_results['_preds'])
                    round_results.pop('_preds')

                if '_model' in round_results:
                    if retain_round_artifacts:
                        self.models.append(round_results['_model'])
                    round_results.pop('_model')

                if retain_round_artifacts and '_scaler' in data_dict:
                    self.scalers.append(data_dict['_scaler'])

            _: Any = round_results.setdefault('strict_mode_error', None)
            round_results.update(id=i, execution_time=time.time() - start_time)

            if retain_round_artifacts and round_succeeded:
                self.round_params.append(round_params)

            round_results.update(round_params)
            if getattr(self.manifest, 'ablation_config', None) is not None:
                round_results['_dropped_features'] = json.dumps(round_params.get('_dropped_features', []))

            finalize_objective_result(self.manifest, round_results, round_succeeded)
            round_results['_convergence_warning'] = convergence_warning(self._convergence_caught[warning_start:], round_succeeded)
            if self._walk_forward is not None and round_succeeded:
                self._walk_forward.accept(round_results['id'], data_dict)
            results_accumulator.append(dict(round_results))
            if len(results_accumulator) >= STANDARD_RUN_LOG_BATCH_SIZE:
                log_batches.append(objective_frame(self.manifest, results_accumulator))
                results_accumulator = []

            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            if write_header and not round_succeeded:
                _pending_csv_rows.append(dict(round_results))
            else:
                with csv_path.open('a', newline='') as f:
                    writer = csv.writer(f)
                    if write_header:
                        csv_header = list(round_results.keys())
                        writer.writerow(csv_header)
                        for pending in _pending_csv_rows:
                            writer.writerow([
                                '' if (v := pending.get(col)) is None else v
                                for col in csv_header
                            ])
                        _pending_csv_rows = []
                    if csv_header is None:
                        raise ValueError('UniversalExperimentLoop CSV header is missing for append')
                    writer.writerow([
                        '' if (v := round_results.get(col)) is None else v
                        for col in csv_header
                    ])
                    extra_keys = set(round_results) - set(csv_header)
                    if extra_keys:
                        logger.warning(
                            'Round %d result keys not in CSV header — values dropped: %s',
                            i, sorted(extra_keys),
                        )

        if _pending_csv_rows:
            with csv_path.open('a', newline='') as f:
                writer = csv.writer(f)
                if csv_header is None:
                    csv_header = list(_pending_csv_rows[0].keys())
                    writer.writerow(csv_header)
                for pending in _pending_csv_rows:
                    writer.writerow([
                        '' if (v := pending.get(col)) is None else v
                        for col in csv_header
                    ])

        if results_accumulator:
            log_batches.append(objective_frame(self.manifest, results_accumulator))
        if log_batches:
            self.experiment_log = pl.concat(log_batches, how='vertical_relaxed')

        if post_processing:
            self._finalize()

    def _clear_post_processing_outputs(self) -> None:

        '''Clear terminal post-processing outputs before a run.'''

        self._log = None
        self.experiment_confusion_metrics = None
        self.experiment_backtest_results = None
        self.experiment_parameter_correlation = None

    def _finalize(self) -> None:

        '''Compute post-experiment Log, metrics, and backtest results.'''

        if self.experiment_log is None:
            return

        cols_to_multilabel = self.experiment_log.select(pl.col(pl.Utf8).exclude('_dropped_features')).columns

        self._log = _log_module.Log(uel_object=self, cols_to_multilabel=cols_to_multilabel)

        if self._walk_forward is not None:
            self.experiment_backtest_results = self._log.experiment_backtest_results()
            self.experiment_parameter_correlation = self._log.experiment_parameter_correlation
            return

        is_rule_based = isinstance(self.manifest, RuleBasedManifest)
        try:
            self.experiment_confusion_metrics = (
                None if is_rule_based
                else self._log.experiment_confusion_metrics('price_change')
            )
        except ConfusionNotApplicableError:
            self.experiment_confusion_metrics = None
        self.experiment_backtest_results = (
            None if is_rule_based
            else self._log.experiment_backtest_results()
        )
        self.experiment_parameter_correlation = self._log.experiment_parameter_correlation


    def _trigger_feedback(self,
                          msq: Any,
                          strategy: Any,
                          feedback_controller: Any,
                          current_round: int) -> list[dict[str, Any]]:

        '''Execute a feedback cycle at the current round.'''

        return feedback_controller.trigger(
            self.experiment_log, msq, strategy, current_round,
        )


    def _run_with_msq(self,
                      *,
                      experiment_name: str,
                      n_permutations: int,
                      context_params: dict[str, Any] | None,
                      resume: bool,
                      post_processing: bool = False,
                      progress_bar: bool = True) -> None:

        '''Run MSQ with per-round preparation, feedback and checkpoints.'''

        self._validate_msq_preconditions(resume=resume)

        components = self._setup_msq_components(
            experiment_name=experiment_name,
            n_permutations=n_permutations,
        )
        domain = components['domain']
        msq = components['msq']
        feedback_controller = components['feedback_controller']
        checkpoint_manager = components['checkpoint_manager']
        content_hash = components['content_hash']
        strategy_type = components['strategy_type']
        csv_path = components['csv_path']
        round_data_path = components['round_data_path']

        if resume and self._experiment_dir is not None and round_data_path is not None and csv_path.exists():
            checkpoint = checkpoint_manager.validate(self._experiment_dir, content_hash=content_hash, strategy_type=strategy_type)
            validate_filter_resume(checkpoint, cast(SearchStrategy, self._search_strategy), csv_path, parallel=self._workers > 1)
            metadata_path = self._experiment_dir / 'metadata.json'
            validate_walk_forward_resume(self.manifest, json.loads(metadata_path.read_text()) if metadata_path.exists() else {})
            expected = _storage.csv_prefix(round_data_path, checkpoint['metadata']['experiment_round'] + 1)
            _fields.validate_saved_successes(round_data_path, expected, checkpoint['metadata']['experiment_round'] + 1)
            if expected:
                with _storage.open_text(round_data_path, limit=len(expected)) as stream:
                    for line in stream:
                        self._validate_resume_payloads(json.loads(line))
        if _parallel.dispatch(self, components, n_permutations, context_params, resume, progress_bar):
            return

        self._register_shutdown_handler()

        self.round_params = []
        self.models = []
        self.extras = []
        self.preds = []
        self.scalers = []
        self._alignment = []
        self.experiment_log = None
        self._clear_post_processing_outputs()

        if self.prep is None or self.model is None:
            raise ValueError('UniversalExperimentLoop prep and model functions must be configured')

        factorized = None
        if self._factorize:
            if not isinstance(self.manifest, Manifest):
                raise ValueError('factorize requires a manifest')
            if (self.prep, self.model) != self._manifest_functions:
                raise ValueError('factorize requires the original manifest prep and model')
            from limen.experiment._factorize import FactorizedRounds
            factorized = FactorizedRounds(
                manifest=self.manifest, strategy=self._search_strategy, domain=domain.params,
                pruning=bool(self._pruning_strategies), callback=self._intra_callback is not None,
                context=context_params, prep=self.manifest.prepare_data, model=self.manifest.run_model, data=self.data,
                record_execution=self._record_execution, record_model_outputs=self._record_model_outputs,
                intervention_path=self._experiment_dir / 'interventions.json' if self._experiment_dir else None,
            )

        start_round = 0
        if resume and self._experiment_dir is not None and self._experiment_dir.exists():
            start_round = self._restore_checkpoint_state(
                msq, domain, feedback_controller, checkpoint_manager,
                content_hash=content_hash, strategy_type=strategy_type,
                csv_path=csv_path, round_data_path=round_data_path,
                retain_round_artifacts=post_processing, context_params=context_params,
            )
        elif self._experiment_dir:
            self._guard_stale_artifacts()
            _ = self._initialize_fresh(self._experiment_dir, checkpoint_manager)
            self._write_metadata(self._experiment_dir)

        last_msq_state = msq.get_state()
        last_completed_round = None
        results_accumulator: list[dict[str, Any]] = []

        csv_header: list[str] | None = None
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with csv_path.open('r', newline='') as f:
                csv_header = next(csv.reader(f), None)
            if getattr(self.manifest, 'ablation_config', None) is not None and '_dropped_features' not in (csv_header or []):
                raise ValueError('UniversalExperimentLoop Cannot append ablation results without _dropped_features; start a new results file.')
            validate_objective_header(self.manifest, csv_header)
            csv_header = convergence_header(csv_path, csv_header) if csv_header is not None else None

        _pending_csv_rows: list[dict[str, Any]] = []

        for round_params in tqdm(msq, initial=start_round, desc=experiment_name,
                                 disable=not progress_bar):
            current_round = round_params['_round_index']
            current_hash = round_params['_id']

            if self._shutdown_requested:
                logger.info(
                    'Experiment stopped by shutdown signal at round %d',
                    current_round,
                )
                if self._experiment_dir and last_completed_round is not None:
                    self._checkpoint(
                        last_msq_state, domain, self._experiment_dir, checkpoint_manager,
                        last_completed_round, n_permutations,
                        strategy_type=strategy_type, content_hash=content_hash,
                        feedback_controller=feedback_controller,
                        pruning_strategies=self._pruning_strategies,
                    )
                break

            start_time = time.time()

            sfd_params = {k: v for k, v in round_params.items() if not k.startswith('_')}

            if context_params is not None:
                sfd_params.update(context_params)

            data_dict: dict[str, Any] = {}

            round_succeeded = False
            caught: list[warnings.WarningMessage] = []
            replayed_warnings: list[str] = []
            replayed_convergence = False
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if factorized is not None:
                        data_dict, round_results, replayed_warnings, replayed_convergence = factorized.evaluate(sfd_params)
                    else:
                        data_dict = self.prep(self.data, round_params=sfd_params)
                        data_dict['_record_execution'] = self._record_execution
                        data_dict['_record_model_outputs'] = self._record_model_outputs
                        if '_alignment' in data_dict:
                            for key in ('execution', 'market', 'model_outputs'):
                                data_dict['_alignment'].pop(key, None)
                        round_results = self.model(
                            data=data_dict, round_params=sfd_params,
                        )
                round_succeeded = True
            except Exception as exc:
                logger.exception('Round %d failed', current_round)
                round_results = self._round_error_policy.failure(self, exc, current_round)
            except KeyboardInterrupt:
                if self._shutdown_requested:
                    logger.info(
                        'Round %d interrupted by shutdown — checkpointing at round %d',
                        current_round,
                        last_completed_round if last_completed_round is not None else -1,
                    )
                    if self._experiment_dir and last_completed_round is not None:
                        self._checkpoint(
                            last_msq_state, domain, self._experiment_dir, checkpoint_manager,
                            last_completed_round, n_permutations,
                            strategy_type=strategy_type, content_hash=content_hash,
                            feedback_controller=feedback_controller,
                            pruning_strategies=self._pruning_strategies,
                        )
                    break
                raise

            _fields.validate_result(round_params | (context_params or {}), round_results, succeeded=round_succeeded)
            round_results['_warnings'] = (
                json.dumps(list(dict.fromkeys([*replayed_warnings, *(str(w.message) for w in caught)])))
            )

            if 'extras' in round_results:
                extras = round_results.pop('extras')
                if post_processing:
                    self.extras.append(extras)
            if 'models' in round_results:
                models = round_results.pop('models')
                if post_processing:
                    self.models.append(models)
            if '_model' in round_results:
                model = round_results.pop('_model')
                if post_processing:
                    self.models.append(model)
            current_preds = round_results.pop('_preds', None)
            if current_preds is None:
                current_preds = []
            if post_processing and round_succeeded:
                self.preds.append(current_preds)
            if post_processing and round_succeeded and '_scaler' in data_dict:
                self.scalers.append(data_dict['_scaler'])

            if post_processing and round_succeeded:
                self._alignment.append(data_dict['_alignment'])

            round_results.update(id=current_hash, execution_time=time.time() - start_time)

            if post_processing and round_succeeded:
                self.round_params.append(sfd_params)
            round_results.update(round_params | (context_params or {}))
            if getattr(self.manifest, 'ablation_config', None) is not None:
                round_results['_dropped_features'] = json.dumps(sfd_params.get('_dropped_features', []))

            _: Any = round_results.setdefault('strict_mode_error', None)

            finalize_objective_result(self.manifest, round_results, round_succeeded)
            round_results['_convergence_warning'] = replayed_convergence or convergence_warning(caught, round_succeeded)
            if self._walk_forward is not None and round_succeeded:
                self._walk_forward.accept(round_results['id'], data_dict)
            results_accumulator.append(round_results)

            if self._walk_forward is not None and round_succeeded and csv_header:
                csv_header = self._walk_forward.complete_failed_header(csv_path, csv_header, list(round_results))

            if round_succeeded:
                csv_header = complete_failed_header(csv_path, csv_header, list(round_results), self.experiment_log)
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            if write_header and not round_succeeded and _holdout.binding_of(self.manifest) is not None:
                self._flush_results(results_accumulator)
                results_accumulator = []
            if write_header and not round_succeeded and _holdout.binding_of(self.manifest) is None:
                _pending_csv_rows.append(dict(round_results))
            else:
                with csv_path.open('a', newline='') as f:
                    writer = csv.writer(f)
                    if write_header:
                        csv_header = list(round_results.keys())
                        writer.writerow(csv_header)
                        for pending in _pending_csv_rows:
                            writer.writerow([
                                '' if (v := pending.get(col)) is None else v
                                for col in csv_header
                            ])
                        _pending_csv_rows = []
                    if csv_header is None:
                        raise ValueError('UniversalExperimentLoop CSV header is missing for append')
                    writer.writerow([
                        '' if (v := round_results.get(col)) is None else v
                        for col in csv_header
                    ])
                    extra_keys = set(round_results) - set(csv_header)
                    if extra_keys:
                        logger.warning(
                            'Round %d result keys not in CSV header — values dropped: %s',
                            current_round, sorted(extra_keys),
                        )

            if round_data_path and round_succeeded:
                self._append_round_data(
                    round_data_path, current_hash, sfd_params,
                    current_preds,
                    data_dict['_alignment'],
                    round_index=current_round,
                )

            checkpoint_due = (
                self._experiment_dir
                and (checkpoint_manager.should_checkpoint(current_round) or self._round_error_policy.exhausted)
            )

            if feedback_controller.should_trigger(current_round):
                self._flush_results(results_accumulator)
                results_accumulator = []
                interventions = self._trigger_feedback(
                    msq, self._search_strategy,
                    feedback_controller, current_round,
                )
                if interventions and self._experiment_dir:
                    checkpoint_due = True

            if checkpoint_due and self._experiment_dir is not None:
                self._checkpoint(
                    msq, domain, self._experiment_dir, checkpoint_manager,
                    current_round, n_permutations,
                    strategy_type=strategy_type, content_hash=content_hash,
                    feedback_controller=feedback_controller,
                    pruning_strategies=self._pruning_strategies,
                )

            last_msq_state = msq.get_state()
            last_completed_round = current_round

        if _pending_csv_rows:
            with csv_path.open('a', newline='') as f:
                writer = csv.writer(f)
                if csv_header is None:
                    csv_header = list(_pending_csv_rows[0].keys())
                    writer.writerow(csv_header)
                for pending in _pending_csv_rows:
                    writer.writerow([
                        '' if (v := pending.get(col)) is None else v
                        for col in csv_header
                    ])

        if not self._shutdown_requested:
            logger.info(
                'Experiment completed naturally after %d rounds',
                msq.yielded_count,
            )

        self._flush_results(results_accumulator)
        if post_processing:
            self._finalize()


    def _validate_msq_preconditions(self,
                                     *,
                                     resume: bool) -> None:


        if resume:
            if not self._experiment_dir:
                raise ValueError(
                    'UniversalExperimentLoop resume=True requires experiment_dir to be set.'
                )
            if not self._experiment_dir.exists():
                raise FileNotFoundError(
                    f"UniversalExperimentLoop Cannot resume: experiment directory {self._experiment_dir} does not exist."
                )


    def _setup_msq_components(self,
                               *,
                               experiment_name: str,
                               n_permutations: int) -> dict[str, Any]:


        if self._search_strategy is None:
            raise ValueError('UniversalExperimentLoop MSQ execution requires a search_strategy')

        domain = self._search_strategy.domain
        msq = MSQ(
            self._search_strategy, domain, n_permutations=n_permutations,
        )

        intervention_path = None
        audit_log_path = None
        if self._experiment_dir is not None:
            intervention_path = self._experiment_dir / 'interventions.json'
            audit_log_path = self._experiment_dir / 'audit.jsonl'

        feedback_controller = FeedbackController(
            feedback_interval=self._feedback_interval,
            pruning_strategies=self._pruning_strategies,
            intra_callback=self._intra_callback,
            intervention_path=intervention_path,
            audit_log_path=audit_log_path,
        )

        checkpoint_manager = CheckpointManager(
            checkpoint_interval=self._checkpoint_interval,
        )

        csv_path = self._experiment_dir / 'results.csv' if self._experiment_dir else Path(f"{experiment_name}.csv")
        round_data_path = _storage.round_path(self._experiment_dir, fresh=True) if self._experiment_dir else None

        return {
            'domain': domain,
            'msq': msq,
            'feedback_controller': feedback_controller,
            'checkpoint_manager': checkpoint_manager,
            'content_hash': CheckpointManager.compute_content_hash(self.params) if self._experiment_dir is not None else None,
            'strategy_type': type(self._search_strategy).__name__,
            'csv_path': csv_path,
            'round_data_path': round_data_path,
        }


    def _validate_resume_payloads(self, entry: Mapping[str, object]) -> None:
        forward = None if self.manifest is None else self.manifest.split_walk_forward
        message = 'Cannot resume a lighter recording policy: required recorded recovery evidence is missing' + (' for split_walk_forward' if forward is not None else '')
        leaves = entry.get('folds') if forward is not None else [entry]
        if not isinstance(leaves, list) or (forward is not None and len(cast(list[object], leaves)) != forward.n_folds):
            raise ValueError(message)
        parameters = cast(Mapping[str, object], entry.get('round_params', {}))
        event = self.manifest is not None and self.manifest.resolve_trade_policy(parameters) is not None
        for leaf in cast(list[object], leaves):
            if not isinstance(leaf, Mapping):
                raise ValueError(message)
            record = cast(Mapping[str, object], leaf)
            alignment, ledger = record.get('alignment'), record.get('trade_ledger')
            if not isinstance(record.get('preds'), list) or not isinstance(alignment, Mapping) or not isinstance(cast(Mapping[str, object], alignment).get('missing_datetimes'), list):
                raise ValueError(message)
            if (event or 'trade_contract' in record) and (not self._recording or not self._recording['trade_ledger'] or not isinstance(ledger, Mapping) or not all(isinstance(cast(Mapping[str, object], ledger).get(key), list) for key in ('states', 'intents', 'fills', 'episodes', 'funding'))):
                raise ValueError(message)

    def _restore_checkpoint_state(self, msq: MSQ, domain: ParamDomain,
                                   feedback_controller: FeedbackController, checkpoint_manager: CheckpointManager,
                                   *, content_hash: str, strategy_type: str,
                                   csv_path: Path, round_data_path: Path | None,
                                   retain_round_artifacts: bool, context_params: dict[str, object] | None) -> int:


        if self._experiment_dir is None or self._search_strategy is None:
            raise ValueError('UniversalExperimentLoop checkpoint restore requires experiment_dir and search_strategy')

        checkpoint_data = self._resume_from_checkpoint(self._experiment_dir, checkpoint_manager,
                                                      content_hash=content_hash, strategy_type=strategy_type)
        metadata_path = self._experiment_dir / 'metadata.json'
        metadata: dict[str, object] = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        _checkpoint_proof.validate_checkpoint(self._experiment_dir, checkpoint_data, metadata, _holdout.binding_of(self.manifest),
                                                getattr(self.manifest, '_holdout_source', None), self.manifest, self.data)
        if metadata.get('record_execution', False) != self._record_execution:
            raise ValueError('Cannot resume with a different record_execution setting')
        if metadata.get('record_model_outputs', False) != self._record_model_outputs:
            raise ValueError('Cannot resume with a different record_model_outputs setting')
        if metadata.get('factorize', False) != self._factorize:
            raise ValueError('Cannot resume with a different factorize setting')
        validate_objective_resume(self.manifest, metadata, csv_path)
        validate_walk_forward_resume(self.manifest, metadata)
        validate_grid_resume(self._search_strategy, domain, checkpoint_data, metadata, factorize=self._factorize)
        domain.set_state(domain.get_state() if self._factorize else checkpoint_data['domain_state'])
        msq.set_state(checkpoint_data['msq_state'])
        if self._factorize:
            validate_factorized_grid(self._search_strategy)

        if 'feedback_controller_state' in checkpoint_data:
            feedback_controller.set_state(checkpoint_data['feedback_controller_state'])

        if 'pruning_strategy_states' in checkpoint_data:
            states = checkpoint_data['pruning_strategy_states']
            if len(self._pruning_strategies) != len(states):
                raise ValueError(
                    f"UniversalExperimentLoop Pruning strategy count mismatch: checkpoint has {len(states)} but {len(self._pruning_strategies)} configured. Use the same strategies to resume or delete the checkpoint to start fresh."
                )
            for ps, state in zip(self._pruning_strategies, states, strict=True):
                ps.set_state(state)

        start_round = checkpoint_data['metadata']['experiment_round'] + 1
        logger.info('Resuming from round %d', start_round)

        if not round_data_path or (not round_data_path.exists() and not csv_path.exists()):
            raise ValueError(
                f"UniversalExperimentLoop Cannot resume: round_data.jsonl not found in {self._experiment_dir}. Checkpoint indicates {start_round} rounds completed but no round data exists."
            )
        if not csv_path.exists():
            raise ValueError(
                f"UniversalExperimentLoop Cannot resume: results.csv not found in {self._experiment_dir}. Checkpoint indicates {start_round} rounds completed but no results log exists."
            )
        experiment_log, expected = _serial.restore_log(csv_path, round_data_path, start_round, domain.params, context_params, factorize=self._factorize)
        if getattr(self.manifest, 'ablation_config', None) is not None and '_dropped_features' not in experiment_log.columns:
            raise ValueError('UniversalExperimentLoop Cannot resume ablation results without _dropped_features; start a new experiment directory.')
        if self._walk_forward is not None:
            if experiment_log.height != start_round:
                raise ValueError('Cannot resume split_walk_forward without every completed trial result')
            expected = [(index, str(row['id'])) for index, row in enumerate(experiment_log.iter_rows(named=True)) if row['strict_mode_error'] is None]
            if expected or round_data_path.exists():
                _ = self._walk_forward.restore(round_data_path, start_round, expected)
            if not expected:
                self._walk_forward.writer.clear()
        else:
            loaded_rounds = self._load_round_data(round_data_path, up_to_round=start_round, retain_round_artifacts=retain_round_artifacts)
            if loaded_rounds < len(expected):
                raise ValueError(
                    f"UniversalExperimentLoop Cannot resume: round_data.jsonl has {loaded_rounds} entries but checkpoint indicates {start_round} rounds completed."
                )

        self.experiment_log = experiment_log

        col = next((c for c in ('_param_hash', '_id') if c in experiment_log.columns), None)
        if col is not None:
            self._search_strategy.rebuild_seen_from_log(experiment_log[col].drop_nulls().to_list())

        if round_data_path.exists():
            self._truncate_round_data(round_data_path, start_round)
        pl.read_csv(csv_path, n_rows=start_round, infer_schema=False).write_csv(csv_path)

        return start_round


    @staticmethod
    def _truncate_round_data(round_data_path: Path,
                              start_round: int) -> None:


        _storage.truncate_rounds(round_data_path, start_round)


    def _flush_results(self,
                       accumulator: list[dict[str, Any]]) -> None:


        if not accumulator:
            return

        batch = objective_frame(self.manifest, accumulator)
        if self.experiment_log is not None:
            self.experiment_log = pl.concat(
                [self.experiment_log, batch], how='diagonal_relaxed',
            )
        else:
            self.experiment_log = batch


    def _guard_stale_artifacts(self) -> None:


        if not self._experiment_dir or not self._experiment_dir.exists():
            return

        artifact_files = [
            'results.csv', 'round_data.jsonl', 'round_data.jsonl.gz', 'round_market', 'round_signals', 'checkpoint.json',
            'audit.jsonl', 'interventions.json', 'metadata.json',
        ]
        existing = [
            f for f in artifact_files
            if (self._experiment_dir / f).exists()
        ]
        if existing:
            resume_required = ['checkpoint.json', 'round_data.jsonl' if 'round_data.jsonl' in existing else 'round_data.jsonl.gz', 'results.csv']
            missing_for_resume = [f for f in resume_required if f not in existing]
            if not missing_for_resume:
                raise FileExistsError(
                    f"UniversalExperimentLoop Experiment directory {self._experiment_dir} already contains artifacts: {', '.join(existing)}. Set resume=True to continue only when the saved recording policy permits recovery, or choose a different experiment_dir."
                )
            raise FileExistsError(
                f"UniversalExperimentLoop Experiment directory {self._experiment_dir} contains partial artifacts that resume=True cannot continue: {', '.join(existing)} (missing {', '.join(missing_for_resume)}). The previous run crashed before a complete checkpoint was written. Delete these files or choose a different experiment_dir."
            )


    check_fresh_directory = _guard_stale_artifacts


    def _initialize_fresh(self,
                          checkpoint_dir: Path,
                          checkpoint_manager: CheckpointManager) -> Path:

        '''Create a fresh checkpoint directory.'''

        return checkpoint_manager.initialize_fresh(checkpoint_dir)


    def _write_metadata(self, experiment_dir: Path, *, context_parameter_keys: list[str] | None = None) -> None:

        if self._sfd_module_name is None:
            raise ValueError(
                'UniversalExperimentLoop Cannot write metadata: SFD module has no __name__ attribute. Trainer requires a reimportable SFD module.'
            )

        metadata: dict[str, object] = _parallel.worker_metadata(self._workers, {
            'sfd_module': self._sfd_module_name, **self._round_error_policy.metadata,
            'limen_version': self._get_limen_version(),
            'created_at': datetime.now(timezone.utc).isoformat(), 'round_storage': _storage.FORMAT,
        }, context_parameter_keys=context_parameter_keys)
        if self._record_execution:
            metadata['record_execution'] = True
        if self._record_model_outputs:
            metadata['record_model_outputs'] = True
        if self._recording is not None:
            metadata['recording'] = self._recording
        if self._factorize:
            metadata['factorize'] = True
        add_grid_metadata(self._search_strategy, metadata)
        add_objective_metadata(self.manifest, metadata)
        if self._walk_forward is not None:
            add_fold_metadata(metadata, self._walk_forward, self._workers)
            metadata.update(split_walk_forward=self._walk_forward.config.as_dict(), walk_forward_validation_ratio=list(self._walk_forward.manifest.split_config[:2]), pbo_blocks=pbo_blocks(self._walk_forward.manifest.acceptance))
        if self._yaml_reference is not None:
            data_no_lineage = {k: v for k, v in self._yaml_reference.items() if k != 'lineage'}
            metadata['yaml_reference'] = data_no_lineage
            metadata['manifest_id'] = canonical_manifest_id(self._yaml_reference)

        if holdout := _holdout.metadata(self.manifest):
            metadata['holdout'] = holdout
        metadata.update(_checkpoint_proof.metadata_binding(_holdout.binding_of(self.manifest)))
        with (experiment_dir / 'metadata.json').open('w') as f:
            json.dump(metadata, f, indent=2)


    @staticmethod
    def _get_limen_version() -> str:

        try:
            return importlib.metadata.version('vaquum-limen')
        except importlib.metadata.PackageNotFoundError:
            return 'dev'


    def _checkpoint(self, msq: MSQ | dict[str, object], domain: ParamDomain,
                    checkpoint_dir: Path, checkpoint_manager: CheckpointManager,
                    current_round: int, target_permutations: int, *,
                    strategy_type: str, content_hash: str,
                    feedback_controller: FeedbackController | None = None,
                    pruning_strategies: list[PruningStrategy] | None = None,
                    failed_round_params: list[dict[str, object]] | None = None) -> None:

        manifest = getattr(self, 'manifest', None)
        checkpoint_manager.save(
            checkpoint_dir, msq, domain, current_round, target_permutations,
            strategy_type=strategy_type, content_hash=content_hash,
            feedback_controller=feedback_controller, pruning_strategies=pruning_strategies,
            failed_round_params=failed_round_params,
            holdout_binding=_checkpoint_proof.checkpoint_state(checkpoint_dir, current_round, _holdout.binding_of(manifest),
                                                                 getattr(manifest, '_holdout_source', None)),
        )


    def _append_round_data(self,
                           round_data_path: Path,
                           round_id: str,
                           round_params: dict[str, Any],
                           preds: Any,
                           alignment: dict[str, Any],
                           *,
                           round_index: int) -> None:

        '''Append one round's data to the JSONL file.'''

        if self._walk_forward is not None:
            _storage.append_record(round_data_path, project_record(_holdout.fold_trial(self,
                {'round_id': round_id, '_round_index': round_index,
                 'round_params': round_params, 'folds': alignment['folds'], **{key: alignment[key] for key in ('fold_promotion', 'holdout') if key in alignment}}), self._recording))
            return

        entry = {
            'round_id': round_id,
            '_round_index': round_index,
            'round_params': round_params,
            'preds': preds.tolist() if hasattr(preds, 'tolist') else list(preds),
            **{key: alignment[key] for key in ('trade_contract', 'trade_contract_digest', 'trade_ledger', 'learning_binding', 'holdout') if key in alignment},
            'alignment': {
                'missing_datetimes': [
                    dt.isoformat()
                    for dt in alignment.get('missing_datetimes', [])
                ],
                'first_test_datetime': (
                    alignment['first_test_datetime'].isoformat()
                ),
                'last_test_datetime': (
                    alignment['last_test_datetime'].isoformat()
                ),
            },
        }

        if 'holdout' in entry:
            entry['holdout'] = _holdout.bind_round(self.manifest, cast(Mapping[str, object], entry['holdout']), round_id)
        if 'learning_binding' in alignment:
            entry['learning_binding'] = {'round_id': round_id, 'manifest_id': canonical_manifest_id(self._yaml_reference) if self._yaml_reference is not None else None, 'model': alignment['learning_binding']}

        if self._record_execution:
            entry.update(execution=alignment.get('execution'), market=alignment.get('market'))
        if self._record_model_outputs:
            entry.update(alignment.get('model_outputs', {'probs': None}))

        _storage.append_record(round_data_path, project_record(entry, self._recording))


    def _load_round_data(self,
                         round_data_path: Path,
                         up_to_round: int | None = None,
                         *,
                         retain_round_artifacts: bool = True) -> int:

        '''Load accumulated round data from JSONL into instance lists.'''

        if not round_data_path.exists():
            return 0

        if self._walk_forward is not None:
            return self._walk_forward.restore(round_data_path, up_to_round)

        loaded_rounds = 0
        limit = len(_storage.csv_prefix(round_data_path, up_to_round)) if up_to_round is not None else None
        with _storage.open_text(round_data_path, limit=limit) as f:
            for raw_line in f:
                entry = json.loads(raw_line)
                if up_to_round is not None and entry.get('_round_index', 0) >= up_to_round:
                    break

                loaded_rounds += 1
                if retain_round_artifacts:
                    self.round_params.append(entry['round_params'])
                    self.preds.append(entry['preds'])
                    self._alignment.append({
                        **{key: entry[key] for key in ('trade_contract', 'trade_contract_digest', 'trade_ledger', 'learning_binding', 'holdout') if key in entry},
                        'missing_datetimes': [
                            datetime.fromisoformat(dt)
                            for dt in entry['alignment']['missing_datetimes']
                        ],
                        'first_test_datetime': datetime.fromisoformat(
                            entry['alignment']['first_test_datetime'],
                        ),
                        'last_test_datetime': datetime.fromisoformat(
                            entry['alignment']['last_test_datetime'],
                        ),
                    })

        return loaded_rounds


    def _resume_from_checkpoint(self,
                                 checkpoint_dir: Path,
                                 checkpoint_manager: CheckpointManager,
                                 *,
                                 strategy_type: str,
                                 content_hash: str) -> dict[str, Any]:

        '''Validate and load the existing checkpoint state.'''

        return checkpoint_manager.validate(
            checkpoint_dir,
            content_hash=content_hash,
            strategy_type=strategy_type,
        )


    def _register_shutdown_handler(self) -> None:

        '''Register SIGTERM and SIGINT handlers that set _shutdown_requested.'''

        def _handler(signum: int, _frame: Any) -> None:
            if self._shutdown_requested:
                raise KeyboardInterrupt
            logger.warning('Signal %d received — shutdown requested.', signum)
            self._shutdown_requested = True

        try:
            _ = signal.signal(signal.SIGTERM, _handler)
            _ = signal.signal(signal.SIGINT, _handler)
        except ValueError:
            logger.warning(
                'Cannot install signal handlers outside the main thread. Graceful shutdown via signals will not be available.'
            )
