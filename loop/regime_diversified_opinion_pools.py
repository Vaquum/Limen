'''
RegimeDiversifiedOpinionPools - Regime-Based Dynamic Optimization Pipeline.

Compute model selection and prediction aggregation using regime-based clustering.
'''

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl
from functools import reduce
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from loop import UniversalExperimentLoop
from loop.tests.utils.cleanup import cleanup_csv_files
from loop.sfm.reference import empty

DEFAULT_PERF_COLS = [
    'pred_pos_rate_pct', 'actual_pos_rate_pct',
    'precision_pct', 'recall_pct',
    'tp_x_mean', 'fp_x_mean', 'tp_x_median', 'fp_x_median',
    'pred_pos_count', 'pred_pos_x_mean', 'pred_pos_x_median',
    'tp_count', 'fp_count', 'tp_fp_cohen_d', 'tp_fp_ks'
]


class OfflineFilter:

    '''Compute data filtering and validation for offline pipeline.'''

    def __init__(self, perf_cols: List[str] = None, iqr_multiplier: float = 3.0):

        self.perf_cols = perf_cols or DEFAULT_PERF_COLS
        self.iqr_multiplier = iqr_multiplier

    def sanity_filter(self, df: pl.DataFrame) -> pl.DataFrame:

        return self._drop_nulls(df, self.perf_cols)

    def outlier_filter(self, df: pl.DataFrame) -> pl.DataFrame:

        return self._remove_outliers_iqr(df, self.perf_cols)

    def _drop_nulls(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:

        filters = [pl.col(col).is_not_null() for col in columns]

        return df.filter(pl.all_horizontal(filters))

    def _remove_outliers_iqr(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:

        for col in columns:
            bounds = df.select([
                pl.col(col).quantile(0.25).alias('q1'),
                pl.col(col).quantile(0.75).alias('q3')
            ]).row(0, named=True)

            q1, q3 = bounds.get('q1'), bounds.get('q3')

            # Skip filtering if quantiles are None or all values identical
            if q1 is None or q3 is None or q1 == q3:
                continue

            # Ensure we're working with floats
            q1, q3 = float(q1), float(q3)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr

            df = df.filter((pl.col(col) >= lower_bound) &
                           (pl.col(col) <= upper_bound))

        return df


class OfflineRegime:

    '''Compute model regime detection for offline pipeline.'''

    def __init__(self, random_state: int):

        self.random_state = random_state

    def cluster_models(self, df: pl.DataFrame, k: int, perf_cols: List[str] = None) -> np.ndarray:

        cluster_cols = perf_cols or DEFAULT_PERF_COLS

        metrics_matrix = df.select(cluster_cols).to_numpy()
        n_samples = len(metrics_matrix)

        # Adjust k to be at most the number of samples
        actual_k = min(k, n_samples)

        # If only 1 sample, all get label 0
        if actual_k == 1:
            return np.zeros(n_samples, dtype=int)

        scaler = StandardScaler()
        metrics_scaled = scaler.fit_transform(metrics_matrix)

        kmeans = KMeans(n_clusters=actual_k, random_state=self.random_state, n_init='auto')
        cluster_labels = kmeans.fit_predict(metrics_scaled)

        return cluster_labels


class OfflineDiversification:

    '''Compute model diversification and selection within regimes for offline pipeline.'''

    def pca_performance_selection(self,
                                  df: pl.DataFrame,
                                  target_count: int,
                                  perf_cols: Optional[List[str]] = None,
                                  n_components: Optional[int] = None,
                                  n_clusters: int = 8,
                                  random_state: int = 42) -> pl.DataFrame:

        if len(df) <= target_count:
            return df

        perf_cols = perf_cols or DEFAULT_PERF_COLS

        X = df.select(perf_cols).to_numpy().astype(float)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        Xp = pca.fit_transform(Xs)

        # KMeans in PCA space
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=random_state, n_init='auto')
        labels = kmeans.fit_predict(Xp)
        centers = kmeans.cluster_centers_

        # Select medoid from each cluster
        selected_indices: List[int] = []
        for cid in range(n_clusters):
            mask = labels == cid
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue
            cluster_pts = Xp[mask]
            center = centers[cid]
            dists = np.linalg.norm(cluster_pts - center, axis=1)
            local_best = np.argmin(dists)
            selected_indices.append(int(idxs[local_best]))

        # Fill to target_count if needed
        if len(selected_indices) < target_count:
            normed = []
            for c in perf_cols:
                col_vals = df[c].to_numpy()
                mn = np.min(col_vals)
                mx = np.max(col_vals)
                if mx > mn:
                    normalized = (col_vals - mn) / (mx - mn)
                    normed.append(normalized)
                else:
                    normed.append(np.ones(len(df)))
            comp_arr = np.vstack(normed).mean(axis=0)
            order = np.argsort(-comp_arr)

            for idx in order:
                if len(selected_indices) >= target_count:
                    break
                if int(idx) in selected_indices:
                    continue
                selected_indices.append(int(idx))

        selected_indices = list(dict.fromkeys(selected_indices))[:target_count]

        return df.with_row_index().filter(pl.col('index').is_in(selected_indices)).drop('index')


class OnlineModelLoader:

    '''Compute model training and prediction extraction for online pipeline.'''

    def __init__(self, sfm, manifest=None):

        self.sfm = sfm
        self.manifest = manifest
        self.trained_models = {}
        self.exclude_perf_cols = DEFAULT_PERF_COLS + \
            ['x_name', 'n_kept', 'id', 'cluster']

    def extract_model_params(self, regime_df: pl.DataFrame) -> List[Dict]:

        param_cols = [
            col for col in regime_df.columns if col not in self.exclude_perf_cols]
        return [
            {col: [row[col]] for col in param_cols}
            for row in regime_df.iter_rows(named=True)
        ]

    def run_single_model_experiment(self,
                                    data: pl.DataFrame,
                                    params: Dict,
                                    regime_id: int,
                                    model_id: int):

        if self.manifest is not None:
            uel = UniversalExperimentLoop(data=data, single_file_model=empty)
            uel.run(
                experiment_name=f"rdop_regime_{regime_id}_model_{model_id}",
                n_permutations=1,
                prep_each_round=False,
                params=lambda: params,
                manifest=self.sfm.manifest()
            )
        else:
            uel = UniversalExperimentLoop(
                data=data, single_file_model=empty)
            uel.run(
                experiment_name=f"rdop_regime_{regime_id}_model_{model_id}",
                n_permutations=1,
                prep_each_round=False,
                params=lambda: params,
                prep=self.sfm.prep,
                model=self.sfm.model
            )

        # Extract prediction performance
        round_id = 0
        perf_df = uel._log.permutation_prediction_performance(round_id).copy()
        perf_df = perf_df.drop(columns=['actuals', 'hit', 'miss'])

        return perf_df

    def merge_prediction_dataframes(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:

        merge_keys = ['open', 'close', 'price_change']
        return reduce(
            lambda left, right: pd.merge(
                left, right, on=merge_keys, how='inner'),
            dfs
        )


class AggregationStrategy:

    '''Compute prediction aggregation strategies for multiple models.'''

    def __init__(self, threshold: float = 0.5):

        self.threshold = threshold

    def mean_aggregation(self, pred_arrays: np.ndarray) -> pd.Series:

        avg = np.mean(pred_arrays, axis=0)
        return pd.Series((avg >= self.threshold).astype(int))

    def median_aggregation(self, pred_arrays: np.ndarray) -> pd.Series:

        median = np.median(pred_arrays, axis=0)
        return pd.Series((median >= self.threshold).astype(int))

    def majority_vote_aggregation(self, pred_arrays):

        votes = pred_arrays.astype(int)
        frac = (votes == votes.max(axis=0)).sum(axis=0) / votes.shape[0]
        return pd.Series(frac)

    def aggregate(self, pred_arrays: np.ndarray, method: str) -> pd.Series:

        if method == 'mean':
            return self.mean_aggregation(pred_arrays)
        elif method == 'median':
            return self.median_aggregation(pred_arrays)
        elif method == 'majority_vote':
            return self.majority_vote_aggregation(pred_arrays)
        else:
            return self.mean_aggregation(pred_arrays)


class OnlineAggregation:

    '''Compute prediction aggregation and market direction analysis for online pipeline.'''

    def __init__(self, sfm, manifest: Dict = None, aggregation_threshold: float = 0.5):

        self.sfm = sfm
        self.manifest = manifest
        self.aggregation_strategy = AggregationStrategy(threshold=aggregation_threshold)
        self.model_loader = OnlineModelLoader(sfm, manifest)

    def aggregate_predictions(self, predictions_df: pd.DataFrame, method: str = 'mean') -> pd.Series:

        pred_arrays = predictions_df.values.T
        return self.aggregation_strategy.aggregate(pred_arrays, method)

    def run_regime_experiments(self, data: pl.DataFrame, regime_id: int, regime_df: pl.DataFrame,
                               aggregation_method: str) -> pl.DataFrame:

        # Extract model parameters
        model_params = self.model_loader.extract_model_params(regime_df)

        # Run experiments for all models
        experiment_results = [
            self.model_loader.run_single_model_experiment(
                data, params, regime_id, i)
            for i, params in enumerate(model_params)
        ]

        # Filter successful experiments
        successful_experiments = [(i, perf_df) for i, result in enumerate(experiment_results)
                                  if result is not None for perf_df in [result]]

        processed_dfs = []

        for model_idx, perf_df in successful_experiments:
            processed_df = perf_df.rename(
                columns={'predictions': f"predictions_{model_idx}"})
            processed_dfs.append(processed_df)

        merged_df = self.model_loader.merge_prediction_dataframes(
            processed_dfs)
        merged_df = pl.from_pandas(merged_df)

        pred_cols = [
            col for col in merged_df.columns if col.startswith('predictions_')]
        if pred_cols:
            # Aggregate predictions
            agg_series = self.aggregate_predictions(
                merged_df[pred_cols].to_pandas(), aggregation_method)
            merged_df = merged_df.with_columns([
                pl.Series('aggregated_prediction', agg_series),
                pl.Series('pred_direction', agg_series)
            ])

            # Compute correct prediction directly from price_change
            merged_df = merged_df.with_columns(
                (pl.col('pred_direction') == pl.when(pl.col('price_change') > 0).then(1).otherwise(
                    pl.when(pl.col('price_change') < 0).then(-1).otherwise(0)
                )).cast(pl.Int64).alias('correct_prediction')
            )

            # Add regime identifier
            merged_df = merged_df.with_columns([
                pl.lit(regime_id).alias('regime')
            ])

            return merged_df

        return pl.DataFrame()


class RegimeDiversifiedOpinionPools:

    '''Defines Regime Diversified Opinion Pools for Loop experiments.'''

    def __init__(self, sfm, random_state: Optional[int] = 42):
        
        '''
        Create RegimeDiversifiedOpinionPools instance with core SFM dependency.

        Args:
            sfm: Single File Model for experiments
            random_state (int, optional): Random state for reproducible results
        '''

        # Initialize core state
        self.regime_pools = {}
        self.sfm = sfm
        self.manifest = sfm.manifest() if hasattr(sfm, 'manifest') and callable(getattr(sfm, 'manifest')) else None
        self.n_regimes = 0
        self.trained_models = {}
        self.random_state = random_state

    def offline_pipeline(self,
                         confusion_metrics,
                         perf_cols: List[str] = None,
                         iqr_multiplier: float = 3.0,
                         target_count: int = 100,
                         n_components: Optional[int] = None,
                         n_clusters: int = 8,
                         k: int = 5) -> pl.DataFrame:
        
        '''
        Compute offline pipeline for model selection and regime detection.

        Args:
            confusion_metrics: Pandas/polars dataframe with experiment confusion metrics
            perf_cols (List[str], optional): Performance columns for filtering
            iqr_multiplier (float): Multiplier for IQR outlier detection
            target_count (int): Target number of models to select per regime
            k (int): Number of clusters for regime detection

        Returns:
            pl.DataFrame: All regime pools combined with regime labels
        '''

        # Initialize components with provided parameters
        offline_filter = OfflineFilter(
            perf_cols=perf_cols or DEFAULT_PERF_COLS, iqr_multiplier=iqr_multiplier)
        offline_regime = OfflineRegime(random_state=self.random_state)
        offline_diversification = OfflineDiversification()

        confusion_metrics = pl.from_pandas(confusion_metrics)

        # Sanity filtering (null checking)
        df_filtered = offline_filter.sanity_filter(confusion_metrics)

        if len(df_filtered) == 0:
            print('WARNING: All models failed sanity check (contained nulls). Using original metrics.')
            # Return original metrics with added regime column for consistency
            df_filtered = confusion_metrics.with_columns(pl.lit(0).alias('regime'))
            self.n_regimes = 1
            self.regime_pools = {0: df_filtered}
            return df_filtered

        # Outlier filtering
        df_filtered = offline_filter.outlier_filter(df_filtered)

        if len(df_filtered) == 0:
            print('WARNING: All models removed by outlier filtering. Using sanity-filtered metrics.')
            df_filtered = offline_filter.sanity_filter(confusion_metrics)

        # Regime clustering
        cluster_labels = offline_regime.cluster_models(
            df_filtered, k, perf_cols)
        self.n_regimes = k

        df_filtered = df_filtered.with_columns(
            pl.Series('regime', cluster_labels)
        )

        # Diversification
        selected_models_list = []

        for cluster_id in range(k):
            regime_df = df_filtered.filter(pl.col('regime') == cluster_id)

            if len(regime_df) == 0:
                continue

            selected_df = offline_diversification.pca_performance_selection(
                regime_df, target_count, perf_cols, 
                n_components=n_components, 
                n_clusters=n_clusters, 
                random_state=self.random_state
                )
            self.regime_pools[cluster_id] = selected_df
            selected_models_list.append(selected_df)

        return pl.concat(selected_models_list, how='vertical')

    def online_pipeline(self,
                        data: pl.DataFrame,
                        aggregation_method: str = 'mean',
                        aggregation_threshold: float = 0.5) -> pl.DataFrame:
        
        '''
        Compute online pipeline for regime-based prediction aggregation.

        Args:
            data (pl.DataFrame): Klines dataset with 'open', 'close', 'price_change' columns
            aggregation_method (str): Method for aggregating predictions across models in each regime
            aggregation_threshold (float): Threshold for aggregation decision

        Returns:
            pl.DataFrame: Combined predictions with regime identifiers and performance data
        '''

        online_aggregation = OnlineAggregation(
            self.sfm,
            manifest=self.manifest,
            aggregation_threshold=aggregation_threshold
        )

        # Run UEL experiments for each regime
        prediction_dataframes = []

        for regime_id, regime_df in self.regime_pools.items():
            regime_prediction_df = online_aggregation.run_regime_experiments(
                data, regime_id, regime_df, aggregation_method
            )
            if len(regime_prediction_df) > 0:
                prediction_dataframes.append(regime_prediction_df)

        # Aggregate predictions
        return pl.concat(prediction_dataframes, how='vertical')
