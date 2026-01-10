'''
RegimeDiversifiedOpinionPools - Regime-Based Dynamic Optimization Pipeline.

Compute model selection and prediction aggregation using regime-based clustering.
'''

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl
from functools import reduce
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from limen.experiment import UniversalExperimentLoop

DEFAULT_PERF_COLS = [
    'pred_pos_rate_pct', 'actual_pos_rate_pct',
    'precision_pct', 'recall_pct',
    'tp_x_mean', 'fp_x_mean', 'tp_x_median', 'fp_x_median',
    'pred_pos_count', 'pred_pos_x_mean', 'pred_pos_x_median',
    'tp_count', 'fp_count', 'tp_fp_cohen_d', 'tp_fp_ks'
]


class OfflineFilter:

    '''Compute data filtering and validation for offline pipeline.'''

    def __init__(self, perf_cols: List[str] | None = None, iqr_multiplier: float = 3.0):

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

            if q1 is None or q3 is None or q1 == q3:
                continue

            q1, q3 = float(q1), float(q3)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr

            df = df.filter((pl.col(col) >= lower_bound) & (pl.col(col) <= upper_bound))

        return df


class OfflineRegime:

    '''Compute model regime detection for offline pipeline.'''

    def __init__(self, random_state: int):

        self.random_state = random_state

    def cluster_models(self, df: pl.DataFrame, k: int, perf_cols: List[str] = None) -> np.ndarray:

        cluster_cols = perf_cols or DEFAULT_PERF_COLS

        metrics_matrix = df.select(cluster_cols).to_numpy()
        n_samples = len(metrics_matrix)

        actual_k = min(k, n_samples)
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
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
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

    def __init__(self, sfd, manifest=None):

        self.sfd = sfd
        self.manifest = manifest
        self.trained_models = {}
        self.exclude_perf_cols = DEFAULT_PERF_COLS + ['x_name', 'n_kept', 'id', 'cluster']

    def extract_model_params(self, regime_df: pl.DataFrame) -> List[Dict]:

        param_cols = [col for col in regime_df.columns if col not in self.exclude_perf_cols]
        return [
            {col: [row[col]] for col in param_cols}
            for row in regime_df.iter_rows(named=True)
        ]

    def run_single_model_experiment(self,
                                    data: pd.DataFrame,
                                    params: Dict,
                                    regime_id: int,
                                    model_id: int):

        if self.manifest is not None:
            uel = UniversalExperimentLoop(data=data, sfd=self.sfd)
            uel.run(
                experiment_name=f"rdop_regime_{regime_id}_model_{model_id}",
                n_permutations=1,
                prep_each_round=True,
                params=lambda: params
            )
        else:
            uel = UniversalExperimentLoop(
                data=data, sfd=self.sfd)
            uel.run(
                experiment_name=f"rdop_regime_{regime_id}_model_{model_id}",
                n_permutations=1,
                prep_each_round=False,
                params=lambda: params,
                prep=self.sfd.prep,
                model=self.sfd.model
            )

        round_id = 0
        perf_df = uel._log.permutation_prediction_performance(round_id).copy()
        perf_df = perf_df.drop(columns=['actuals', 'hit', 'miss'])

        return perf_df

    def merge_prediction_dataframes(self, dfs: List[pd.DataFrame]) -> pl.DataFrame:

        merge_keys = ['open', 'close', 'price_change']
        pl_dfs = [pl.from_pandas(df) for df in dfs]
        return reduce(
            lambda left, right: left.join(right, on=merge_keys, how='inner'),
            pl_dfs
        )


class AggregationStrategy:

    '''Compute prediction aggregation strategies for multiple models.'''

    def __init__(self, threshold: Optional[float] = None):

        self.threshold = threshold

    def mean_aggregation(self, pred_arrays: np.ndarray) -> np.ndarray:

        avg = np.mean(pred_arrays, axis=0)

        if self.threshold is None:
            return avg

        return (avg >= self.threshold).astype(float)

    def median_aggregation(self, pred_arrays: np.ndarray) -> np.ndarray:

        median = np.median(pred_arrays, axis=0)

        if self.threshold is None:
            return median

        return (median >= self.threshold).astype(float)

    def majority_vote_aggregation(self, pred_arrays):

        results = []

        for col_preds in pred_arrays.T:

            counts = Counter(col_preds)
            most_common_val, count = counts.most_common(1)[0]

            n = len(col_preds)
            percentage = count / n

            if self.threshold is None or percentage >= self.threshold:
                results.append(float(most_common_val))
            else:
                results.append(0.0)

        return np.array(results)

    def aggregate(self, pred_arrays: np.ndarray, method: str) -> np.ndarray:

        if method == 'mean':
            return self.mean_aggregation(pred_arrays)
        elif method == 'median':
            return self.median_aggregation(pred_arrays)
        elif method == 'majority_vote':
            return self.majority_vote_aggregation(pred_arrays)
        else:
            return self.mean_aggregation(pred_arrays)


class OnlineAggregation:

    def __init__(self, sfd, manifest: Dict = None, aggregation_threshold: Optional[float] = None):

        self.sfd = sfd
        self.manifest = manifest
        self.aggregation_strategy = AggregationStrategy(threshold=aggregation_threshold)
        self.model_loader = OnlineModelLoader(sfd, manifest)

    def aggregate_predictions(self, predictions_df: pd.DataFrame, method: str = 'mean') -> np.ndarray:

        pred_arrays = predictions_df.values.T
        return self.aggregation_strategy.aggregate(pred_arrays, method)

    def run_regime_experiments(self, data: pd.DataFrame, regime_id: int, regime_df: pl.DataFrame,
                               aggregation_method: str) -> tuple[np.ndarray, pl.DataFrame]:

        model_params = self.model_loader.extract_model_params(regime_df)

        experiment_results = [
            self.model_loader.run_single_model_experiment(data, params, regime_id, i)
            for i, params in enumerate(model_params)
        ]

        successful_experiments = [(i, perf_df) for i, result in enumerate(experiment_results)
                                  if result is not None for perf_df in [result]]

        processed_dfs = []

        for model_idx, perf_df in successful_experiments:
            processed_df = perf_df.rename(columns={'predictions': f"predictions_{model_idx}"})
            processed_dfs.append(processed_df)

        merged_df = self.model_loader.merge_prediction_dataframes(processed_dfs)

        pred_cols = [col for col in merged_df.columns if col.startswith('predictions_')]
        
        agg_series = self.aggregate_predictions(merged_df[pred_cols].to_pandas(), aggregation_method)
        
        return agg_series, merged_df.drop(pred_cols)

class RegimeDiversifiedOpinionPools:

    '''Defines Regime Diversified Opinion Pools for Loop experiments.'''

    def __init__(self, sfd, random_state: Optional[int] = 42):

        '''
        Create RegimeDiversifiedOpinionPools instance with core SFD dependency.

        Args:
            sfd: Single File Decoder for experiments
            random_state (int, optional): Random state for reproducible results
        '''

        self.regime_pools = {}
        self.sfd = sfd
        self.manifest = sfd.manifest() if hasattr(sfd, 'manifest') and callable(getattr(sfd, 'manifest')) else None
        self.n_regimes = 0
        self.trained_models = {}
        self.random_state = random_state

    def offline_pipeline(self,
                         confusion_metrics,
                         perf_cols: Optional[List[str]] = None,
                         target_count: int = 100,
                         k_regimes: int = 6,
                         iqr_multiplier: float = 3.0,
                         n_pca_components: Optional[int] = None,
                         n_pca_clusters: int = 8
        ) -> Dict[int, pl.DataFrame]:
        
        '''
        Compute offline pipeline for model selection and regime detection.

        Args:
            confusion_metrics (pd.DataFrame): Pandas dataframe with experiment confusion metrics
            perf_cols (List[str], optional): Performance columns for filtering
            target_count (int): Target number of models to select per regime
            k_regimes (int): Number of regime clusters
            iqr_multiplier (float): Multiplier for IQR outlier detection
            n_pca_components (int, optional): Number of PCA dimensions to keep
            n_pca_clusters (int): Number of clusters used inside PCA space

        Returns:
            Dict[int, pl.DataFrame]: Dictionary mapping regime IDs to their respective filtered model dataframes
        '''

        offline_filter = OfflineFilter(perf_cols=perf_cols, iqr_multiplier=iqr_multiplier)
        offline_regime = OfflineRegime(random_state=self.random_state)
        offline_diversification = OfflineDiversification()

        confusion_metrics = pl.from_pandas(confusion_metrics)

        # Sanity filtering
        df_filtered = offline_filter.sanity_filter(confusion_metrics)

        if len(df_filtered) == 0:
            print('WARNING: All models failed sanity check (contained nulls). Using original metrics.')
            df_filtered = confusion_metrics.with_columns(pl.lit(0).alias('regime'))
            self.n_regimes = 1
            self.regime_pools = {0: df_filtered}
            return {0: df_filtered}

        # Outlier filtering
        df_filtered = offline_filter.outlier_filter(df_filtered)

        if len(df_filtered) == 0:
            print('WARNING: All models removed by outlier filtering. Using sanity-filtered metrics.')
            df_filtered = offline_filter.sanity_filter(confusion_metrics)

        # Regime clustering
        cluster_labels = offline_regime.cluster_models(df_filtered, k_regimes, perf_cols)
        self.n_regimes = k_regimes

        df_filtered = df_filtered.with_columns(pl.Series('regime', cluster_labels))

        # Diversification and return regime dictionary
        regime_results = {}

        for cluster_id in range(k_regimes):
            regime_df = df_filtered.filter(pl.col('regime') == cluster_id)

            if len(regime_df) == 0:
                continue

            selected_df = offline_diversification.pca_performance_selection(
                regime_df, target_count, perf_cols,
                n_components=n_pca_components,
                n_clusters=n_pca_clusters,
                random_state=self.random_state
            )
            self.regime_pools[cluster_id] = selected_df
            regime_results[cluster_id] = selected_df

        return regime_results if regime_results else {}

    def online_pipeline(self,
                        data: pd.DataFrame,
                        aggregation_method: str = 'mean',
                        aggregation_threshold: Optional[float] = None) -> pl.DataFrame:

        '''
        Compute online pipeline for regime-based prediction aggregation.

        Args:
            data (pd.DataFrame): The data to use for the experiment
            aggregation_method (str): Method for aggregating predictions across models in each regime
            aggregation_threshold (float, optional): Threshold for aggregation decision

        Returns:
            pl.DataFrame: Combined predictions with regime identifiers.
        '''

        online_aggregation = OnlineAggregation(
            self.sfd,
            manifest=self.manifest,
            aggregation_threshold=aggregation_threshold
        )

        regime_predictions = {}
        result_df = None

        # Run UEL experiments for each regime and aggregation
        for regime_id, regime_df in self.regime_pools.items():
            regime_series, regime_data = online_aggregation.run_regime_experiments(
                data, regime_id, regime_df, aggregation_method
            )
            if len(regime_series) > 0:
                regime_predictions[regime_id] = regime_series
                if result_df is None and len(regime_data) > 0:
                    result_df = regime_data

        if not regime_predictions or result_df is None:
            return pl.DataFrame()

        for regime_id, prediction_series in regime_predictions.items():
            prediction_col = pl.Series(f'regime_{regime_id}_prediction', prediction_series)
            result_df = result_df.with_columns([prediction_col])

        return result_df
