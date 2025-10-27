from __future__ import annotations

"""Granular EDA runtime package."""

import copy
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import (
    bivariate,
    categorical,
    data_quality,
    geospatial,
    marketing,
    numeric,
    outliers,
    relationships,
    target,
    text,
    time_series,
)
from .preprocessing import PreprocessingOptions, PreprocessingReport, apply_preprocessing
from .common import (
    AnalysisChart,
    AnalysisContext,
    AnalysisInsight,
    AnalysisMetric,
    AnalysisResult,
    AnalysisTable,
    dataframe_to_table,
    fig_to_base64,
)

AnalysisHandler = Callable[[pd.DataFrame, AnalysisContext], AnalysisResult]


ANALYSIS_DISPATCH: Dict[str, AnalysisHandler] = {
    # Data quality
    "dataset_shape_analysis": data_quality.dataset_shape_analysis,
    "data_range_validation": data_quality.data_range_validation,
    "data_types_validation": data_quality.data_types_validation,
    "missing_value_analysis": data_quality.missing_value_analysis,
    "duplicate_detection": data_quality.duplicate_detection,
    # Numeric / Univariate
    "summary_statistics": numeric.summary_statistics,
    "numeric_frequency_analysis": numeric.numeric_frequency_analysis,
    "skewness_analysis": numeric.skewness_analysis,
    "skewness_statistics": numeric.skewness_statistics,
    "skewness_visualization": numeric.skewness_visualization,
    "normality_test": numeric.normality_test,
    "distribution_plots": numeric.distribution_plots,
    "histogram_plots": numeric.histogram_plots,
    "box_plots": numeric.box_plots,
    "violin_plots": numeric.violin_plots,
    "kde_plots": numeric.kde_plots,
    # Categorical
    "categorical_frequency_analysis": categorical.categorical_frequency_analysis,
    "categorical_visualization": categorical.categorical_visualization,
    "categorical_bar_charts": categorical.categorical_bar_charts,
    "categorical_pie_charts": categorical.categorical_pie_charts,
    "categorical_cardinality_profile": categorical.categorical_cardinality_profile,
    "rare_category_detection": categorical.rare_category_detection,
    # Bivariate / Multivariate
    "correlation_analysis": bivariate.correlation_analysis,
    "pearson_correlation": bivariate.pearson_correlation,
    "spearman_correlation": bivariate.spearman_correlation,
    "scatter_plot_analysis": bivariate.scatter_plot_analysis,
    "cross_tabulation_analysis": bivariate.cross_tabulation_analysis,
    "categorical_numeric_relationships": bivariate.categorical_numeric_relationships,
    # Outliers
    "iqr_outlier_detection": outliers.iqr_outlier_detection,
    "zscore_outlier_detection": outliers.zscore_outlier_detection,
    "visual_outlier_inspection": outliers.visual_outlier_inspection,
    "outlier_distribution_visualization": outliers.outlier_distribution_visualization,
    "outlier_scatter_matrix": outliers.outlier_scatter_matrix,
    # Time series
    "temporal_trend_analysis": time_series.temporal_trend_analysis,
    "seasonality_detection": time_series.seasonality_detection,
    "datetime_feature_extraction": time_series.datetime_feature_extraction,
    # Text analysis
    "text_length_distribution": text.text_length_distribution,
    "text_token_frequency": text.text_token_frequency,
    "text_vocabulary_summary": text.text_vocabulary_summary,
    "text_feature_engineering_profile": text.text_feature_engineering_profile,
    "text_nlp_profile": text.text_nlp_profile,
    # Relationships & PCA
    "multicollinearity_analysis": relationships.multicollinearity_analysis,
    "pca_dimensionality_reduction": relationships.pca_dimensionality_reduction,
    "pca_scree_plot": relationships.pca_scree_plot,
    "pca_cumulative_variance": relationships.pca_cumulative_variance,
    "pca_visualization": relationships.pca_visualization,
    "pca_biplot": relationships.pca_biplot,
    "pca_heatmap": relationships.pca_heatmap,
    "cluster_tendency_analysis": relationships.cluster_tendency_analysis,
    "cluster_segmentation_analysis": relationships.cluster_segmentation_analysis,
    "network_analysis": relationships.network_analysis,
    "entity_relationship_network": relationships.entity_relationship_network,
    # Geospatial
    "spatial_distribution_analysis": geospatial.spatial_distribution_analysis,
    "coordinate_system_projection_check": geospatial.coordinate_system_projection_check,
    "spatial_relationships_analysis": geospatial.spatial_relationships_analysis,
    "spatial_data_quality_analysis": geospatial.spatial_data_quality_analysis,
    "geospatial_proximity_analysis": geospatial.geospatial_proximity_analysis,
    # Target variable focus
    "target_variable_analysis": target.target_variable_analysis,
    # Marketing analytics
    "campaign_metrics_analysis": marketing.campaign_metrics_analysis,
    "conversion_funnel_analysis": marketing.conversion_funnel_analysis,
    "engagement_analysis": marketing.engagement_analysis,
    "channel_performance_analysis": marketing.channel_performance_analysis,
    "audience_segmentation_analysis": marketing.audience_segmentation_analysis,
    "roi_analysis": marketing.roi_analysis,
    "attribution_analysis": marketing.attribution_analysis,
    "cohort_analysis": marketing.cohort_analysis,
}


class GranularAnalysisRuntime:
    """Execute granular EDA analyses using registered runtime handlers."""

    def __init__(self) -> None:
        self.dispatch_map = ANALYSIS_DISPATCH
        self._settings = self._load_settings()
        self._defaults = self._derive_defaults(self._settings)
        self._cache_enabled = self._defaults["cache_enabled"]
        self._cache_max_items = self._defaults["cache_max_items"]
        self._cache: Dict[Any, Dict[str, Any]] = {}

    def run_analysis(self, analysis_id: str, df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
        context = self._apply_context_defaults(context)
        cache_key = self._build_cache_key(analysis_id, context)
        if cache_key:
            cached = self._cache.get(cache_key)
            if cached:
                ttl = context.cache_ttl_seconds
                if ttl is None or time.monotonic() - cached["timestamp"] <= ttl:
                    return copy.deepcopy(cached["result"])
                self._cache.pop(cache_key, None)

        handler = self.dispatch_map.get(analysis_id)
        if not handler:
            return AnalysisResult(
                analysis_id=analysis_id,
                title=analysis_id.replace("_", " ").title(),
                summary="Analysis not yet implemented.",
                status="warning",
                insights=[
                    AnalysisInsight(
                        level="warning",
                        text="This analysis type is not implemented in the runtime engine yet.",
                    )
                ],
            )

        working_df = df.copy()
        if context.selected_columns:
            available = [col for col in context.selected_columns if col in working_df.columns]
            if available:
                working_df = working_df[available].copy()

        working_df, sampling_meta = self._apply_sampling(working_df, context)

        try:
            result = handler(working_df, context)
        except Exception as exc:  # pragma: no cover - runtime safety
            column_dtypes = {col: str(working_df[col].dtype) for col in working_df.columns[:25]}
            insights = [
                AnalysisInsight(level="danger", text=str(exc) or exc.__class__.__name__),
            ]
            if working_df.empty:
                insights.append(
                    AnalysisInsight(
                        level="info",
                        text="The dataframe contained no rows after preprocessing/selection.",
                    )
                )
            elif column_dtypes:
                preview = ", ".join(f"{col} ({dtype})" for col, dtype in list(column_dtypes.items())[:5])
                insights.append(
                    AnalysisInsight(
                        level="info",
                        text=f"Column snapshot at failure: {preview}",
                    )
                )
            insights.extend(self._build_error_suggestions(working_df))

            error_details: Dict[str, Any] = {
                "exception": repr(exc),
                "row_count": len(working_df),
                "column_dtypes": column_dtypes,
            }
            if sampling_meta:
                error_details["sampling"] = sampling_meta

            return AnalysisResult(
                analysis_id=analysis_id,
                title=analysis_id.replace("_", " ").title(),
                summary="Analysis failed due to an unexpected error.",
                status="error",
                insights=insights,
                details={"error": error_details},
            )

        if sampling_meta:
            sampling_message = sampling_meta.get("message")
            if sampling_message:
                result.insights.append(AnalysisInsight(level="info", text=sampling_message))
            runtime_details = result.details.setdefault("runtime", {})
            runtime_details["sampling"] = sampling_meta

        if cache_key and self._cache_enabled:
            self._cache[cache_key] = {
                "timestamp": time.monotonic(),
                "result": copy.deepcopy(result),
            }
            self._enforce_cache_budget()

        return result

    def _build_cache_key(self, analysis_id: str, context: AnalysisContext) -> Optional[Tuple[Any, ...]]:
        if not self._cache_enabled:
            return None

        if not context.is_caching_enabled():
            return None
        selected = tuple(context.selected_columns or ())
        mapping_items = tuple(sorted((context.column_mapping or {}).items()))
        return (analysis_id, context.cache_key, selected, mapping_items)

    def _apply_context_defaults(self, context: AnalysisContext) -> AnalysisContext:
        updates: Dict[str, Any] = {}

        if context.max_rows_for_analysis is None:
            updates["max_rows_for_analysis"] = self._defaults["max_rows_for_analysis"]
        if not context.sampling_strategy:
            updates["sampling_strategy"] = self._defaults["sampling_strategy"]
        if context.random_state is None:
            updates["random_state"] = self._defaults["random_state"]
        if context.cache_ttl_seconds is None and context.cache_key is not None:
            updates["cache_ttl_seconds"] = self._defaults["cache_ttl_seconds"]

        if not updates:
            return context

        return replace(context, **updates)

    def _load_settings(self) -> Optional[Any]:  # pragma: no cover - accessor
        try:
            from config import get_settings

            return get_settings()
        except Exception:
            return None

    def _derive_defaults(self, settings: Optional[Any]) -> Dict[str, Any]:
        fallback = {
            "max_rows_for_analysis": 50_000,
            "sampling_strategy": "random",
            "random_state": 42,
            "cache_ttl_seconds": 300,
            "cache_enabled": True,
            "cache_max_items": 128,
        }

        if not settings:
            return fallback

        has_helper = hasattr(settings, "is_field_set")
        global_limit = getattr(settings, "EDA_GLOBAL_SAMPLE_LIMIT", None)
        global_limit_set = has_helper and settings.is_field_set("EDA_GLOBAL_SAMPLE_LIMIT")

        runtime_max_rows = getattr(settings, "GRANULAR_RUNTIME_MAX_ROWS", None)
        runtime_max_rows_set = has_helper and settings.is_field_set("GRANULAR_RUNTIME_MAX_ROWS")
        if runtime_max_rows is None:
            if runtime_max_rows_set:
                pass  # Explicitly configured as unlimited
            elif global_limit_set:
                runtime_max_rows = global_limit
            else:
                runtime_max_rows = fallback["max_rows_for_analysis"]

        return {
            "max_rows_for_analysis": runtime_max_rows,
            "sampling_strategy": getattr(settings, "GRANULAR_RUNTIME_SAMPLING_STRATEGY", fallback["sampling_strategy"]),
            "random_state": getattr(settings, "GRANULAR_RUNTIME_RANDOM_STATE", fallback["random_state"]),
            "cache_ttl_seconds": getattr(settings, "GRANULAR_RUNTIME_CACHE_TTL_SECONDS", fallback["cache_ttl_seconds"]),
            "cache_enabled": getattr(settings, "GRANULAR_RUNTIME_CACHE_ENABLED", fallback["cache_enabled"]),
            "cache_max_items": getattr(settings, "GRANULAR_RUNTIME_CACHE_MAX_ITEMS", fallback["cache_max_items"]),
        }

    def _enforce_cache_budget(self) -> None:
        max_items = self._cache_max_items
        if not self._cache_enabled or max_items is None or max_items <= 0:
            return

        while len(self._cache) > max_items:
            oldest_key = min(self._cache.items(), key=lambda item: item[1]["timestamp"])[0]
            self._cache.pop(oldest_key, None)

    def _apply_sampling(
        self, df: pd.DataFrame, context: AnalysisContext
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        if not context.should_sample(len(df)):
            return df, None

        max_rows = context.max_rows_for_analysis or len(df)
        strategy = (context.sampling_strategy or "random").lower()
        valid_strategies = {"random", "head", "stratified"}
        if strategy not in valid_strategies:
            strategy = "random"

        sampled_df: pd.DataFrame
        if strategy == "head":
            sampled_df = df.head(max_rows).copy()
        elif strategy == "stratified":
            stratify_cols = context.stratify_by()
            if stratify_cols and all(col in df.columns for col in stratify_cols):
                sampled_df = self._stratified_sample(df, stratify_cols, max_rows, context.random_state)
            else:
                strategy = "random"
                sampled_df = df.sample(n=max_rows, random_state=context.random_state)
        else:
            sampled_df = df.sample(n=max_rows, random_state=context.random_state)

        sampling_meta = {
            "message": (
                f"Sampled {len(sampled_df):,} of {len(df):,} rows using '{strategy}' strategy "
                f"to respect max_rows_for_analysis={max_rows:,}."
            ),
            "strategy": strategy,
            "original_rows": len(df),
            "sampled_rows": len(sampled_df),
            "max_rows": max_rows,
        }

        return sampled_df, sampling_meta

    def _stratified_sample(
        self,
        df: pd.DataFrame,
        stratify_cols: List[str],
        max_rows: int,
        random_state: Optional[int],
    ) -> pd.DataFrame:
        grouped = df.groupby(stratify_cols, dropna=False, group_keys=False)
        target_ratio = min(1.0, max_rows / max(len(df), 1))
        samples = []
        remaining = max_rows
        for _, group in grouped:
            if remaining <= 0:
                break
            group_size = len(group)
            if group_size == 0:
                continue
            take = max(1, int(round(group_size * target_ratio)))
            take = min(take, group_size, remaining)
            samples.append(group.sample(n=take, random_state=random_state))
            remaining -= take

        if not samples:
            return df.sample(n=max_rows, random_state=random_state)

        sampled_df = pd.concat(samples, ignore_index=False)
        if len(sampled_df) > max_rows:
            sampled_df = sampled_df.sample(n=max_rows, random_state=random_state)
        return sampled_df

    def _build_error_suggestions(self, df: pd.DataFrame) -> List[AnalysisInsight]:
        suggestions: List[AnalysisInsight] = []
        if df.empty:
            suggestions.append(
                AnalysisInsight(
                    level="warning",
                    text="No rows remained after filtering. Review preprocessing steps or selected columns.",
                )
            )
            return suggestions

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            suggestions.append(
                AnalysisInsight(
                    level="info",
                    text="No numeric columns detected. Convert relevant fields using pd.to_numeric where applicable.",
                )
            )

        if df.isnull().all(axis=None):
            suggestions.append(
                AnalysisInsight(
                    level="info",
                    text="All values are null. Validate data ingestion or adjust missing value handling.",
                )
            )

        return suggestions


__all__ = [
    "GranularAnalysisRuntime",
    "AnalysisContext",
    "AnalysisChart",
    "AnalysisInsight",
    "AnalysisMetric",
    "AnalysisResult",
    "AnalysisTable",
    "dataframe_to_table",
    "fig_to_base64",
    "PreprocessingOptions",
    "PreprocessingReport",
    "apply_preprocessing",
]
