import csv
import importlib.metadata
import json
import logging
import signal
import sqlite3
import time
import warnings
from collections.abc import Callable
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.reducer.pruning_strategy import PruningStrategy
from limen.experiment.param_search.search_strategy import SearchStrategy
from limen.utils.param_space import ParamSpace
from limen.log.log import Log

logger = logging.getLogger(__name__)


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
                 intra_callback: Callable[[Any, MSQ], None] | None = None) -> None:

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

        '''

        if sfd is None:
            raise ValueError('sfd is required')

        self._sfd_module_name = getattr(sfd, '__name__', None)
        self.params = sfd.params()
        self.manifest = None

        if hasattr(sfd, 'manifest'):
            self.manifest = sfd.manifest()

            if data is None:
                if self.manifest.data_source_config is None:
                    raise ValueError(
                        'No data source configured in manifest. '
                        'Add .set_data_source(method=HistoricalData.get_spot_klines, params={...}) '
                        'to manifest or pass data explicitly.'
                    )

                self.data = self.manifest.fetch_data_for_env()
            else:
                self.data = data

            if hasattr(self.manifest, 'model_function') and self.manifest.model_function:
                self.prep = lambda data, round_params=None: self.manifest.prepare_data(data, round_params or {})
                self.model = lambda data, round_params: self.manifest.run_model(data, round_params or {})
            else:
                raise ValueError(
                    'Manifest without model_function is not supported. '
                    'Use .with_model(model_func) in your manifest.'
                )
        else:
            if data is None:
                raise ValueError('data parameter required for custom SFDs using custom functions approach')
            self.data = data
            self.prep = getattr(sfd, 'prep', None)
            self.model = getattr(sfd, 'model', None)

        self.extras = []
        self.models = []
        self._shutdown_requested: bool = False
        self._pause_requested: bool = False
        self._search_strategy = search_strategy
        self._pruning_strategies = pruning_strategies or []
        self._feedback_interval = feedback_interval
        self._checkpoint_interval = checkpoint_interval
        self._experiment_dir = Path(experiment_dir) if experiment_dir else None
        self._intra_callback = intra_callback

    def run(self,
            experiment_name: str,
            n_permutations: int = 10000,
            prep_each_round: bool = False,
            random_search: bool = True,
            maintain_details_in_params: bool = False,
            context_params: dict | None = None,
            params: Callable | None = None,
            prep: Callable | None = None,
            model: Callable | None = None,
            resume: bool = False) -> None:

        '''
        Run the experiment `n_permutations` times.

        NOTE: When search_strategy was provided to __init__, dispatches to
        _run_with_msq for MSQ-based execution. Legacy parameters
        (random_search, maintain_details_in_params, params,
        prep, model) are ignored in that path.

        Args:
            experiment_name (str): The name of the experiment
            n_permutations (int): The number of permutations to run
            prep_each_round (bool): Whether to use `prep` for each round or just first
            random_search (bool): Whether to use random search or not
            maintain_details_in_params (bool): Whether to maintain experiment details in params
            context_params (dict): The context parameters to use for the experiment
            params (Callable | None): Callable that returns the parameters dict
            prep (Callable | None): Callable to prepare the data
            model (Callable | None): Callable to run the model
            resume (bool): Whether to resume from an existing checkpoint

        '''

        self.round_params = []
        self.models = []
        self.preds = []
        self.scalers = []
        self._alignment = []

        if resume and self._search_strategy is None:
            raise ValueError(
                'resume=True is only supported with a search_strategy.'
            )

        if self._search_strategy is not None:
            self._run_with_msq(
                experiment_name=experiment_name,
                n_permutations=n_permutations,
                context_params=context_params,
                resume=resume,
            )
            return

        if self.manifest is not None:
            if prep is not None or model is not None:
                raise ValueError(
                    'Cannot override prep/model when SFM has manifest.'
                )
            if not prep_each_round:
                raise ValueError(
                    'prep_each_round must be True for manifest-driven SFMs.'
                )

        if params is not None:
            self.params = params()

        if prep is not None:
            self.prep = prep

        if model is not None:
            self.model = model

        self.param_space = ParamSpace(params=self.params,
                                      n_permutations=n_permutations)

        for i in tqdm(range(n_permutations)):

            # Start counting execution_time
            start_time = time.time()

            # Generate the parameter values for the current round
            round_params = self.param_space.generate(random_search=random_search)

            # Add context parameters to round_params
            if context_params is not None:
                round_params.update(context_params)

            # Add experiment details to round_params
            if maintain_details_in_params is True:
                round_params['_experiment_details'] = {
                    'current_index': i,
                }

            if prep_each_round is True or i == 0:
                data_dict = self.prep(self.data, round_params=round_params)

            # Perform the model training and evaluation
            round_results = self.model(data=data_dict, round_params=round_params)

            # Remove the experiment details from the results
            if maintain_details_in_params is True:
                round_params.pop('_experiment_details')

            # Add alignment details
            self._alignment.append(data_dict['_alignment'])

            # Handle any extra results that are returned from the model
            if 'extras' in round_results:
                self.extras.append(round_results['extras'])
                round_results.pop('extras')

            # Handle any models that are returned from the model
            if 'models' in round_results:
                self.models.append(round_results['models'])
                round_results.pop('models')

            if '_preds' in round_results:
                self.preds.append(round_results['_preds'])
                round_results.pop('_preds')

            if '_scaler' in data_dict:
                self.scalers.append(data_dict['_scaler'])

            # Add the round number and execution time to the results
            round_results['id'] = i
            round_results['execution_time'] = round(time.time() - start_time, 2)

            self.round_params.append(round_params)

            for key in round_params:
                round_results[key] = round_params[key]

            # Handle writing to the DataFrame
            if i == 0:
                self.experiment_log = pl.DataFrame([round_results])
            else:
                self.experiment_log = self.experiment_log.vstack(pl.DataFrame([round_results]))

            # Handle writing to the file
            if i == 0:
                header_colnames = ','.join(list(round_results.keys()))
                with Path(experiment_name + '.csv').open('a') as f:
                    f.write(f"{header_colnames}\n")

            log_string = f"{', '.join(map(str, self.experiment_log.row(i)))}\n"
            with Path(experiment_name + '.csv').open('a') as f:
                f.write(log_string)

        self._finalize()


    def _finalize(self) -> None:

        '''Compute post-experiment Log, metrics, and backtest results.'''

        if self.experiment_log is None:
            return

        cols_to_multilabel = self.experiment_log.select(pl.col(pl.Utf8)).columns

        self._log = Log(uel_object=self, cols_to_multilabel=cols_to_multilabel)

        self.experiment_confusion_metrics = self._log.experiment_confusion_metrics('price_change')
        self.experiment_backtest_results = self._log.experiment_backtest_results()
        self.experiment_parameter_correlation = self._log.experiment_parameter_correlation


    def _trigger_feedback(self,
                          msq: Any,
                          strategy: Any,
                          feedback_controller: Any,
                          current_round: int) -> list[dict]:

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
                      context_params: dict | None,
                      resume: bool) -> None:

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

        start_round = 0
        if resume and self._experiment_dir.exists():
            start_round = self._restore_checkpoint_state(
                msq, domain, feedback_controller, checkpoint_manager,
                content_hash=content_hash, strategy_type=strategy_type,
                csv_path=csv_path, round_data_path=round_data_path,
            )
        elif self._experiment_dir:
            self._guard_stale_artifacts()
            self._initialize_fresh(self._experiment_dir, checkpoint_manager)
            self._write_metadata(self._experiment_dir)

        last_msq_state = msq.get_state()
        last_completed_round = None
        results_accumulator: list[dict[str, Any]] = []

        for round_params in tqdm(msq, initial=start_round, desc=experiment_name):
            current_round = round_params['_id']

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

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                data_dict = self.prep(self.data, round_params=sfd_params)
                round_results = self.model(
                    data=data_dict, round_params=sfd_params,
                )

            round_results['_warnings'] = (
                json.dumps([str(w.message) for w in caught])
                if caught else '[]'
            )

            if 'extras' in round_results:
                self.extras.append(round_results.pop('extras'))
            if 'models' in round_results:
                self.models.append(round_results.pop('models'))
            current_preds = round_results.pop('_preds', None)
            if current_preds is None:
                current_preds = []
            self.preds.append(current_preds)
            if '_scaler' in data_dict:
                self.scalers.append(data_dict['_scaler'])

            self._alignment.append(data_dict['_alignment'])

            round_results['id'] = current_round
            round_results['execution_time'] = round(
                time.time() - start_time, 2,
            )

            self.round_params.append(sfd_params)
            for key, value in round_params.items():
                round_results[key] = value
            if context_params is not None:
                for key, value in context_params.items():
                    round_results[key] = value

            results_accumulator.append(round_results)

            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open('a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(round_results.keys())
                writer.writerow(round_results.values())

            if round_data_path:
                self._append_round_data(
                    round_data_path, current_round, sfd_params,
                    current_preds,
                    data_dict['_alignment'],
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

            if checkpoint_due:
                self._checkpoint(
                    msq, domain, self._experiment_dir, checkpoint_manager,
                    current_round, n_permutations,
                    strategy_type=strategy_type, content_hash=content_hash,
                    feedback_controller=feedback_controller,
                    pruning_strategies=self._pruning_strategies,
                )

            last_msq_state = msq.get_state()
            last_completed_round = current_round

        if not self._shutdown_requested:
            logger.info(
                'Experiment completed naturally after %d rounds',
                msq.yielded_count,
            )

        self._flush_results(results_accumulator)
        self._finalize()


    def _validate_msq_preconditions(self,
                                     *,
                                     resume: bool) -> None:

        '''Validate preconditions for MSQ-based execution.'''

        if resume:
            if not self._experiment_dir:
                raise ValueError(
                    'resume=True requires experiment_dir to be set.'
                )
            if not self._experiment_dir.exists():
                raise FileNotFoundError(
                    f"Cannot resume: experiment directory "
                    f"{self._experiment_dir} does not exist."
                )


    def _setup_msq_components(self,
                               *,
                               experiment_name: str,
                               n_permutations: int) -> dict[str, Any]:

        '''Create MSQ, FeedbackController, CheckpointManager, and file paths.'''

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
                                   round_data_path: Path | None) -> int:

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

        Returns:
            int: The round number to resume from

        '''

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
                    f"Pruning strategy count mismatch: checkpoint "
                    f"has {len(states)} but "
                    f"{len(self._pruning_strategies)} configured. "
                    f"Use the same strategies to resume or delete "
                    f"the checkpoint to start fresh."
                )
            for ps, state in zip(self._pruning_strategies, states, strict=True):
                ps.set_state(state)

        start_round = (
            checkpoint_data['metadata']['experiment_round'] + 1
        )
        logger.info('Resuming from round %d', start_round)

        if not round_data_path or not round_data_path.exists():
            raise ValueError(
                f"Cannot resume: round_data.jsonl not found in "
                f"{self._experiment_dir}. Checkpoint indicates "
                f"{start_round} rounds completed but no round "
                f"data exists."
            )
        self._load_round_data(round_data_path, up_to_round=start_round)
        if len(self.round_params) < start_round:
            raise ValueError(
                f"Cannot resume: round_data.jsonl has "
                f"{len(self.round_params)} entries but checkpoint "
                f"indicates {start_round} rounds completed."
            )

        if not csv_path.exists():
            raise ValueError(
                f"Cannot resume: results.csv not found in "
                f"{self._experiment_dir}. Checkpoint indicates "
                f"{start_round} rounds completed but no results "
                f"log exists."
            )
        self.experiment_log = pl.read_csv(csv_path, n_rows=start_round)

        if '_param_hash' in self.experiment_log.columns:
            hashes = self.experiment_log['_param_hash'].drop_nulls().to_list()
            self._search_strategy.rebuild_seen_from_log(hashes)

        self._truncate_round_data(round_data_path, start_round)
        self.experiment_log.write_csv(csv_path)

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
                if entry['round_id'] >= start_round:
                    break
                valid_lines.append(stripped)

        with round_data_path.open('w') as f:
            for line in valid_lines:
                f.write(line + '\n')


    def _flush_results(self,
                       accumulator: list[dict[str, Any]]) -> None:

        '''Flush accumulated round results into experiment_log.'''

        if not accumulator:
            return

        batch = pl.DataFrame(accumulator)
        if self.experiment_log is not None:
            self.experiment_log = self.experiment_log.vstack(batch)
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
            raise FileExistsError(
                f"Experiment directory {self._experiment_dir} "
                f"already contains artifacts: "
                f"{', '.join(existing)}. "
                f"Set resume=True to continue or choose a "
                f"different experiment_dir."
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
                'Cannot write metadata: SFD module has no __name__ attribute. '
                'Trainer requires a reimportable SFD module.'
            )

        metadata = {
            'sfd_module': self._sfd_module_name,
            'limen_version': self._get_limen_version(),
            'created_at': datetime.now(timezone.utc).isoformat(),
        }

        with (experiment_dir / 'metadata.json').open('w') as f:
            json.dump(metadata, f, indent=2)


    @staticmethod
    def _get_limen_version() -> str:

        try:
            return importlib.metadata.version('vaquum_limen')
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
                           round_id: int,
                           round_params: dict,
                           preds: Any,
                           alignment: dict) -> None:

        '''Append one round's data to the JSONL file.'''

        entry = {
            'round_id': round_id,
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
            f.write(json.dumps(entry) + '\n')


    def _load_round_data(self,
                         round_data_path: Path,
                         up_to_round: int | None = None) -> None:

        '''
        Load accumulated round data from JSONL into instance lists.

        Args:
            round_data_path (Path): Path to the round_data.jsonl file
            up_to_round (int | None): If set, only load entries with
                round_id < up_to_round (for crash recovery consistency)

        '''

        if not round_data_path.exists():
            return

        with round_data_path.open('r') as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    break

                if (
                    up_to_round is not None
                    and entry['round_id'] >= up_to_round
                ):
                    break

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
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except ValueError:
            logger.warning(
                'Cannot install signal handlers outside the main thread. '
                'Graceful shutdown via signals will not be available.'
            )
