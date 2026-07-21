import copy
import csv
import importlib.metadata
import json
import logging
import signal
import time
import warnings
from collections.abc import Callable
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from tqdm import tqdm

from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.errors import StrictModeError
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.reducer.pruning_strategy import PruningStrategy
from limen.experiment.param_search.search_strategy import SearchStrategy
from limen.utils.param_space import ParamSpace
from limen.log.log import Log
from limen.experiment.manifest_core import RuleBasedManifest
from limen.yaml.store import canonical_manifest_id

logger = logging.getLogger(__name__)


def _validate_prep_each_round_arg(value: object) -> None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(
            'UniversalExperimentLoop prep_each_round must be a bool or None.'
        )

STANDARD_RUN_LOG_BATCH_SIZE = 1000


class UniversalExperimentLoop:

    '''UniversalExperimentLoop class for running experiments.'''

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
                 yaml_reference: dict[str, Any] | None = None) -> None:

        '''
        Initialize the UniversalExperimentLoop.

        NOTE: Automatically detects SFD structure and configures prep/model.
        Manifest-based SFDs auto-generate prep/model from manifest.
        If manifest has data_source_config and no data provided, auto-fetches data.
        Custom SFDs using custom functions approach require explicit data parameter.
        When experiment_dir is provided, all experiment artifacts are stored
        under that directory: checkpoint.json, audit.jsonl, round_data.jsonl,
        interventions.json, and results.csv.

        Args:
            data (pl.DataFrame, optional): The data to use for the experiment
            sfd (SingleFileDecoder, optional): The single file decoder to use for the experiment
            search_strategy (SearchStrategy | None): Search strategy for MSQ-based execution
            pruning_strategies (list[PruningStrategy] | None): Reducers for feedback-driven pruning
            feedback_interval (int): Trigger feedback every N rounds
            checkpoint_interval (int): Save checkpoint every N rounds
            experiment_dir (str | Path | None): Directory for all experiment artifacts
            intra_callback (Callable | None): Python callback receiving (log, msq)
            test_mode (bool): When True and data is None, fetch from test_data_source_config instead of production
            yaml_reference (dict | None): Parsed YAML experiment dict — stored in metadata.json for reproducibility

        '''

        super().__init__()

        if sfd is None:
            raise ValueError('UniversalExperimentLoop sfd is required')

        self._sfd_module_name = getattr(sfd, '__name__', None)
        self.params = sfd.params()
        self.manifest = None
        self.prep: Callable[..., dict[str, Any]] | None = None
        self.model: Callable[..., dict[str, Any]] | None = None

        if hasattr(sfd, 'manifest'):
            manifest = sfd.manifest()
            self.manifest = manifest

            if data is None:
                if self.manifest.data_source_config is None:
                    raise ValueError(
                        'UniversalExperimentLoop No data source configured in manifest. Add .set_data_source(method=HistoricalData.get_spot_klines, params={...}) to manifest or pass data explicitly.'
                    )

                if test_mode and self.manifest.test_data_source_config is not None:
                    self.data = self.manifest.fetch_test_data()
                else:
                    self.data = self.manifest.fetch_data()
            else:
                self.data = data

            if getattr(self.manifest, 'architecture_function', None) is not None:
                def _manifest_prep(data: Any, round_params: dict[str, Any] | None = None) -> dict[str, Any]:
                    '''Prepare data through the manifest pipeline.'''
                    return manifest.prepare_data(data, round_params or {})

                def _manifest_model(data: Any, round_params: dict[str, Any] | None) -> dict[str, Any]:
                    '''Run the manifest reference architecture.'''
                    return manifest.run_model(data, round_params or {})

                self.prep = _manifest_prep
                self.model = _manifest_model
            else:
                raise ValueError(
                    'UniversalExperimentLoop Manifest without architecture_function is not supported. Use .with_reference_architecture(func) in your manifest.'
                )
        else:
            if data is None:
                raise ValueError('UniversalExperimentLoop data parameter required for custom SFDs using custom functions approach')
            self.data = data
            self.prep = getattr(sfd, 'prep', None)
            self.model = getattr(sfd, 'model', None)

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
        self._yaml_reference = copy.deepcopy(yaml_reference)
        self.round_params: list[dict[str, Any]] = []
        self.preds: list[Any] = []
        self.scalers: list[Any] = []
        self._alignment: list[Any] = []
        self.experiment_log: pl.DataFrame | None = None
        self.param_space: ParamSpace | None = None
        self._log: Log | None = None
        self.experiment_confusion_metrics: pd.DataFrame | None = None
        self.experiment_backtest_results: pd.DataFrame | None = None
        self.experiment_parameter_correlation: Callable[..., pd.DataFrame] | None = None
        self._clear_post_processing_outputs()

    def run(self,
            experiment_name: str,
            n_permutations: int = 10000,
            prep_each_round: bool | None = None,
            random_search: bool = True,
            maintain_details_in_params: bool = False,
            context_params: dict[str, Any] | None = None,
            params: Callable[[], dict[str, Any]] | None = None,
            prep: Callable[..., dict[str, Any]] | None = None,
            model: Callable[..., dict[str, Any]] | None = None,
            resume: bool = False,
            post_processing: bool = False,
            progress_bar: bool = True) -> None:

        '''
        Run the experiment `n_permutations` times.

        NOTE: When search_strategy was provided to __init__, dispatches to
        _run_with_msq for MSQ-based execution. Legacy parameters
        (random_search, maintain_details_in_params, params,
        prep, model) are ignored in that path.

        Args:
            experiment_name (str): The name of the experiment
            n_permutations (int): The number of permutations to run
            prep_each_round (bool | None): Whether to use `prep` for each round or just first; None auto-resolves to True for manifest-driven SFDs and False otherwise
            random_search (bool): Whether to use random search or not
            maintain_details_in_params (bool): Whether to maintain experiment details in params
            context_params (dict): The context parameters to use for the experiment
            params (Callable | None): Callable that returns the parameters dict
            prep (Callable | None): Callable to prepare the data
            model (Callable | None): Callable to run the model
            resume (bool): Whether to resume from an existing checkpoint
            post_processing (bool): Whether to compute terminal post-run metrics
            progress_bar (bool): Whether to render the experiment progress bar

        '''

        self.round_params = []
        self.models = []
        self.extras = []
        self.preds = []
        self.scalers = []
        self._alignment = []
        self.experiment_log = None
        self._clear_post_processing_outputs()

        if resume and self._search_strategy is None:
            raise ValueError(
                'UniversalExperimentLoop resume=True is only supported with a search_strategy.'
            )

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
            if prep_each_round is False:
                raise ValueError(
                    'UniversalExperimentLoop prep_each_round must be True for manifest-driven SFDs.'
                )

        if prep_each_round is None:
            prep_each_round = self.manifest is not None

        if params is not None:
            self.params = params()

        if prep is not None:
            self.prep = prep

        if model is not None:
            self.model = model

        if self.prep is None or self.model is None:
            raise ValueError('UniversalExperimentLoop prep and model functions must be configured')

        self.param_space = ParamSpace(params=self.params,
                                      n_permutations=n_permutations)

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

        data_dict: dict[str, Any] = {}
        _pending_csv_rows: list[dict[str, Any]] = []

        for i in tqdm(range(n_permutations), disable=not progress_bar):

            # Start counting execution_time
            start_time = time.time()

            # Generate the parameter values for the current round
            round_params = self.param_space.generate(random_search=random_search)
            if round_params is None:
                raise ValueError('UniversalExperimentLoop parameter space exhausted before completing all rounds')

            # Add context parameters to round_params
            if context_params is not None:
                round_params.update(context_params)

            # Add experiment details to round_params
            if maintain_details_in_params is True:
                round_params['_experiment_details'] = {
                    'current_index': i,
                }

            try:
                if prep_each_round is True or i == 0:
                    data_dict = self.prep(self.data, round_params=round_params)
                round_results = self.model(data=data_dict, round_params=round_params)
                round_succeeded = True
            except StrictModeError as exc:
                logger.error('Round %d failed strict mode check: %s', i, exc)
                round_results: dict[str, Any] = {'strict_mode_error': str(exc)}
                round_succeeded = False

            # Remove the experiment details from the results
            if maintain_details_in_params is True:
                round_params.pop('_experiment_details')

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
            round_results['id'] = i
            round_results['execution_time'] = round(time.time() - start_time, 2)

            if retain_round_artifacts and round_succeeded:
                self.round_params.append(round_params)

            for key in round_params:
                round_results[key] = round_params[key]

            results_accumulator.append(dict(round_results))
            if len(results_accumulator) >= STANDARD_RUN_LOG_BATCH_SIZE:
                log_batches.append(pl.DataFrame(results_accumulator))
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
            log_batches.append(pl.DataFrame(results_accumulator))
        if log_batches:
            self.experiment_log = pl.concat(
                log_batches, how='vertical_relaxed',
            )

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

        cols_to_multilabel = self.experiment_log.select(pl.col(pl.Utf8)).columns

        self._log = Log(uel_object=self, cols_to_multilabel=cols_to_multilabel)

        is_rule_based = isinstance(self.manifest, RuleBasedManifest)
        self.experiment_confusion_metrics = (
            None if is_rule_based
            else self._log.experiment_confusion_metrics('price_change')
        )
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

        '''
        Execute a feedback cycle at the current round.

        Passes the polars experiment log directly to FeedbackController,
        avoiding the overhead of constructing a full Log object.

        Args:
            msq (Any): The mutable search queue
            strategy (Any): The current search strategy
            feedback_controller (Any): The feedback controller
            current_round (int): Current round number

        Returns:
            list[dict]: Interventions applied during this trigger

        '''

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

        '''
        Run the experiment using the Mutable-Search-Queue based execution flow.

        NOTE: Called by run() when search_strategy is configured. Sets up
        MSQ, FeedbackController, and CheckpointManager, then iterates
        over parameter combinations with feedback and checkpoint triggers.
        Data is always prepared each round.

        Args:
            experiment_name (str): The name of the experiment
            n_permutations (int): Maximum number of combinations to run
            context_params (dict | None): Static parameters merged into each round
            resume (bool): Whether to resume from an existing checkpoint
            post_processing (bool): Whether to compute terminal post-run metrics
            progress_bar (bool): Whether to render the experiment progress bar

        '''

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

        start_round = 0
        if resume and self._experiment_dir is not None and self._experiment_dir.exists():
            start_round = self._restore_checkpoint_state(
                msq, domain, feedback_controller, checkpoint_manager,
                content_hash=content_hash, strategy_type=strategy_type,
                csv_path=csv_path, round_data_path=round_data_path,
                retain_round_artifacts=post_processing,
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
                    msq.set_state(last_msq_state)
                    self._checkpoint(
                        msq, domain, self._experiment_dir, checkpoint_manager,
                        last_completed_round, n_permutations,
                        strategy_type=strategy_type, content_hash=content_hash,
                        feedback_controller=feedback_controller,
                        pruning_strategies=self._pruning_strategies,
                    )
                break

            start_time = time.time()

            sfd_params = {
                k: v for k, v in round_params.items()
                if not k.startswith('_')
            }

            if context_params is not None:
                sfd_params.update(context_params)

            data_dict: dict[str, Any] = {}

            round_succeeded = False
            caught: list[warnings.WarningMessage] = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    data_dict = self.prep(self.data, round_params=sfd_params)
                    round_results = self.model(
                        data=data_dict, round_params=sfd_params,
                    )
                round_succeeded = True
            except StrictModeError as exc:
                logger.error('Round %d failed strict mode check: %s', current_round, exc)
                round_results: dict[str, Any] = {'strict_mode_error': str(exc)}
            except KeyboardInterrupt:
                if self._shutdown_requested:
                    logger.info(
                        'Round %d interrupted by shutdown — checkpointing at round %d',
                        current_round,
                        last_completed_round if last_completed_round is not None else -1,
                    )
                    if self._experiment_dir and last_completed_round is not None:
                        msq.set_state(last_msq_state)
                        self._checkpoint(
                            msq, domain, self._experiment_dir, checkpoint_manager,
                            last_completed_round, n_permutations,
                            strategy_type=strategy_type, content_hash=content_hash,
                            feedback_controller=feedback_controller,
                            pruning_strategies=self._pruning_strategies,
                        )
                    break
                raise

            round_results['_warnings'] = (
                json.dumps(list(dict.fromkeys(str(w.message) for w in caught)))
                if caught else '[]'
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

            round_results['id'] = current_hash
            round_results['execution_time'] = round(
                time.time() - start_time, 2,
            )

            if post_processing and round_succeeded:
                self.round_params.append(sfd_params)
            for key, value in round_params.items():
                round_results[key] = value
            if context_params is not None:
                for key, value in context_params.items():
                    round_results[key] = value

            _: Any = round_results.setdefault('strict_mode_error', None)

            results_accumulator.append(round_results)

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
                and checkpoint_manager.should_checkpoint(current_round)
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

        '''Validate preconditions for MSQ-based execution.'''

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

        '''Create MSQ, FeedbackController, CheckpointManager, and file paths.'''

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

        csv_path = (
            self._experiment_dir / 'results.csv'
            if self._experiment_dir
            else Path(f"{experiment_name}.csv")
        )
        round_data_path = (
            self._experiment_dir / 'round_data.jsonl'
            if self._experiment_dir
            else None
        )

        return {
            'domain': domain,
            'msq': msq,
            'feedback_controller': feedback_controller,
            'checkpoint_manager': checkpoint_manager,
            'content_hash': CheckpointManager.compute_content_hash(self.params),
            'strategy_type': type(self._search_strategy).__name__,
            'csv_path': csv_path,
            'round_data_path': round_data_path,
        }


    def _restore_checkpoint_state(self,
                                   msq: MSQ,
                                   domain: ParamDomain,
                                   feedback_controller: FeedbackController,
                                   checkpoint_manager: CheckpointManager,
                                   *,
                                   content_hash: str,
                                   strategy_type: str,
                                   csv_path: Path,
                                   round_data_path: Path | None,
                                   retain_round_artifacts: bool) -> int:

        '''
        Restore experiment state from checkpoint and data files.

        Args:
            msq (MSQ): MSQ instance to restore state into
            domain (ParamDomain): ParamDomain instance to restore state into
            feedback_controller (FeedbackController): FeedbackController to restore
            checkpoint_manager (CheckpointManager): CheckpointManager for loading
            content_hash (str): Expected content hash for validation
            strategy_type (str): Expected strategy type for validation
            csv_path (Path): Path to results CSV
            round_data_path (Path | None): Path to round_data.jsonl
            retain_round_artifacts (bool): Whether to hydrate round data
                into instance artifact lists, or only count entries

        Returns:
            int: The round number to resume from

        '''

        if self._experiment_dir is None or self._search_strategy is None:
            raise ValueError('UniversalExperimentLoop checkpoint restore requires experiment_dir and search_strategy')

        checkpoint_data = self._resume_from_checkpoint(
            self._experiment_dir, checkpoint_manager,
            content_hash=content_hash, strategy_type=strategy_type,
        )
        domain.set_state(checkpoint_data['domain_state'])
        msq.set_state(checkpoint_data['msq_state'])

        if 'feedback_controller_state' in checkpoint_data:
            feedback_controller.set_state(
                checkpoint_data['feedback_controller_state'],
            )

        if 'pruning_strategy_states' in checkpoint_data:
            states = checkpoint_data['pruning_strategy_states']
            if len(self._pruning_strategies) != len(states):
                raise ValueError(
                    f"UniversalExperimentLoop Pruning strategy count mismatch: checkpoint has {len(states)} but {len(self._pruning_strategies)} configured. Use the same strategies to resume or delete the checkpoint to start fresh."
                )
            for ps, state in zip(self._pruning_strategies, states, strict=True):
                ps.set_state(state)

        start_round = (
            checkpoint_data['metadata']['experiment_round'] + 1
        )
        logger.info('Resuming from round %d', start_round)

        if not round_data_path or not round_data_path.exists():
            raise ValueError(
                f"UniversalExperimentLoop Cannot resume: round_data.jsonl not found in {self._experiment_dir}. Checkpoint indicates {start_round} rounds completed but no round data exists."
            )
        loaded_rounds = self._load_round_data(
            round_data_path,
            up_to_round=start_round,
            retain_round_artifacts=retain_round_artifacts,
        )
        if loaded_rounds < start_round:
            raise ValueError(
                f"UniversalExperimentLoop Cannot resume: round_data.jsonl has {loaded_rounds} entries but checkpoint indicates {start_round} rounds completed."
            )

        if not csv_path.exists():
            raise ValueError(
                f"UniversalExperimentLoop Cannot resume: results.csv not found in {self._experiment_dir}. Checkpoint indicates {start_round} rounds completed but no results log exists."
            )
        experiment_log = pl.read_csv(csv_path, n_rows=start_round)
        self.experiment_log = experiment_log

        col = next((c for c in ('_param_hash', '_id') if c in experiment_log.columns), None)
        if col is not None:
            self._search_strategy.rebuild_seen_from_log(
                experiment_log[col].drop_nulls().to_list()
            )

        self._truncate_round_data(round_data_path, start_round)
        experiment_log.write_csv(csv_path)

        return start_round


    @staticmethod
    def _truncate_round_data(round_data_path: Path,
                              start_round: int) -> None:

        '''Truncate round_data.jsonl to only contain rounds before start_round.'''

        valid_lines: list[str] = []
        with round_data_path.open('r') as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    break
                if entry.get('_round_index', 0) >= start_round:
                    break
                valid_lines.append(stripped)

        with round_data_path.open('w') as f:
            for line in valid_lines:
                _ = f.write(line + '\n')


    def _flush_results(self,
                       accumulator: list[dict[str, Any]]) -> None:

        '''Flush accumulated round results into experiment_log.'''

        if not accumulator:
            return

        batch = pl.DataFrame(accumulator)
        if self.experiment_log is not None:
            self.experiment_log = pl.concat(
                [self.experiment_log, batch], how='vertical_relaxed',
            )
        else:
            self.experiment_log = batch


    def _guard_stale_artifacts(self) -> None:

        '''Raise FileExistsError if experiment_dir has leftover artifacts.'''

        if not self._experiment_dir or not self._experiment_dir.exists():
            return

        artifact_files = [
            'results.csv', 'round_data.jsonl', 'checkpoint.json',
            'audit.jsonl', 'interventions.json', 'metadata.json',
        ]
        existing = [
            f for f in artifact_files
            if (self._experiment_dir / f).exists()
        ]
        if existing:
            resume_required = ['checkpoint.json', 'round_data.jsonl', 'results.csv']
            missing_for_resume = [f for f in resume_required if f not in existing]
            if not missing_for_resume:
                raise FileExistsError(
                    f"UniversalExperimentLoop Experiment directory {self._experiment_dir} already contains artifacts: {', '.join(existing)}. Set resume=True to continue or choose a different experiment_dir."
                )
            raise FileExistsError(
                f"UniversalExperimentLoop Experiment directory {self._experiment_dir} contains partial artifacts that resume=True cannot continue: {', '.join(existing)} (missing {', '.join(missing_for_resume)}). The previous run crashed before a complete checkpoint was written. Delete these files or choose a different experiment_dir."
            )


    def _initialize_fresh(self,
                          checkpoint_dir: Path,
                          checkpoint_manager: CheckpointManager) -> Path:

        '''
        Create a fresh checkpoint directory.

        Args:
            checkpoint_dir (Path): Path to create
            checkpoint_manager (CheckpointManager): CheckpointManager instance

        Returns:
            Path: Created directory path

        '''

        return checkpoint_manager.initialize_fresh(checkpoint_dir)


    def _write_metadata(self, experiment_dir: Path) -> None:

        '''
        Write metadata.json to experiment directory.

        Args:
            experiment_dir (Path): Directory to write metadata into

        '''

        if self._sfd_module_name is None:
            raise ValueError(
                'UniversalExperimentLoop Cannot write metadata: SFD module has no __name__ attribute. Trainer requires a reimportable SFD module.'
            )

        metadata = {
            'sfd_module': self._sfd_module_name,
            'limen_version': self._get_limen_version(),
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        if self._yaml_reference is not None:
            data_no_lineage = {k: v for k, v in self._yaml_reference.items() if k != 'lineage'}
            metadata['yaml_reference'] = data_no_lineage
            metadata['manifest_id'] = canonical_manifest_id(self._yaml_reference)

        with (experiment_dir / 'metadata.json').open('w') as f:
            json.dump(metadata, f, indent=2)


    @staticmethod
    def _get_limen_version() -> str:

        try:
            return importlib.metadata.version('vaquum-limen')
        except importlib.metadata.PackageNotFoundError:
            return 'dev'


    def _checkpoint(self,
                    msq: MSQ,
                    domain: ParamDomain,
                    checkpoint_dir: Path,
                    checkpoint_manager: CheckpointManager,
                    current_round: int,
                    target_permutations: int,
                    *,
                    strategy_type: str,
                    content_hash: str,
                    feedback_controller: FeedbackController | None = None,
                    pruning_strategies: list[PruningStrategy] | None = None) -> None:

        '''
        Save a checkpoint at the current experiment state.

        Args:
            msq (MSQ): MSQ instance to checkpoint
            domain (ParamDomain): ParamDomain instance to checkpoint
            checkpoint_dir (Path): Directory to write checkpoint files
            checkpoint_manager (CheckpointManager): CheckpointManager instance
            current_round (int): Current round number
            target_permutations (int): Total rounds planned
            strategy_type (str): Class name of the search strategy
            content_hash (str): SHA-256 digest of the experiment content
            feedback_controller (FeedbackController | None): FeedbackController to checkpoint
            pruning_strategies (list[PruningStrategy] | None): PruningStrategy instances to checkpoint

        '''

        checkpoint_manager.save(
            checkpoint_dir,
            msq,
            domain,
            current_round,
            target_permutations,
            strategy_type=strategy_type,
            content_hash=content_hash,
            feedback_controller=feedback_controller,
            pruning_strategies=pruning_strategies,
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

        entry = {
            'round_id': round_id,
            '_round_index': round_index,
            'round_params': round_params,
            'preds': preds.tolist() if hasattr(preds, 'tolist') else list(preds),
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

        with round_data_path.open('a') as f:
            _ = f.write(json.dumps(entry) + '\n')


    def _load_round_data(self,
                         round_data_path: Path,
                         up_to_round: int | None = None,
                         *,
                         retain_round_artifacts: bool = True) -> int:

        '''
        Load accumulated round data from JSONL into instance lists.

        Args:
            round_data_path (Path): Path to the round_data.jsonl file
            up_to_round (int | None): If set, only load entries with
                round_id < up_to_round (for crash recovery consistency)
            retain_round_artifacts (bool): Whether to hydrate entries into
                instance artifact lists, or only validate/count them

        Returns:
            int: The number of round data entries loaded or counted

        '''

        if not round_data_path.exists():
            return 0

        loaded_rounds = 0
        with round_data_path.open('r') as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    break

                if up_to_round is not None and entry.get('_round_index', 0) >= up_to_round:
                    break

                loaded_rounds += 1
                if retain_round_artifacts:
                    self.round_params.append(entry['round_params'])
                    self.preds.append(entry['preds'])
                    self._alignment.append({
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

        '''
        Validate and load state from an existing checkpoint directory.

        Args:
            checkpoint_dir (Path): Directory containing checkpoint files
            checkpoint_manager (CheckpointManager): CheckpointManager instance
            strategy_type (str): Expected strategy class name for validation
            content_hash (str): Expected SHA-256 digest for validation

        Returns:
            dict: Keys 'metadata', 'msq_state', 'domain_state', and
                optionally 'feedback_controller_state', 'pruning_strategy_states'

        Raises:
            ValueError: If validation fails

        '''

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
