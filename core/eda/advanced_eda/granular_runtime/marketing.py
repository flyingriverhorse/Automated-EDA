from __future__ import annotations

"""Marketing analytics runtime analyses."""

from typing import Dict, List

import numpy as np
import pandas as pd

from .common import (
    AnalysisContext,
    AnalysisMetric,
    AnalysisResult,
    AnalysisTable,
    dataframe_to_table,
    insight,
    metric,
)


def _require_mapping(context: AnalysisContext) -> Dict[str, str]:
    return context.column_mapping or {}


def campaign_metrics_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    mapping = _require_mapping(context)
    required = ["impressions", "clicks", "conversions", "spend", "revenue"]
    available = {metric: mapping.get(metric) for metric in required if mapping.get(metric) in df.columns}

    if len(available) < 2:
        return AnalysisResult(
            analysis_id="campaign_metrics_analysis",
            title="Campaign Metrics Analysis",
            summary="Provide column mapping for impressions, clicks, conversions, spend, or revenue.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Map marketing metrics to dataset columns to compute campaign KPIs.",
                )
            ],
        )

    total_rows = len(df)
    metrics_list: List[AnalysisMetric] = [metric("Rows", total_rows)]

    clicks_col = available.get("clicks")
    impressions_col = available.get("impressions")
    conversions_col = available.get("conversions")
    spend_col = available.get("spend")
    revenue_col = available.get("revenue")

    if impressions_col and clicks_col:
        ctr = (df[clicks_col].sum() / df[impressions_col].sum() * 100) if df[impressions_col].sum() else 0
        metrics_list.append(metric("CTR", f"{ctr:.2f}", unit="%", description="Click-through rate"))

    if spend_col and clicks_col:
        cpc = (df[spend_col].sum() / df[clicks_col].sum()) if df[clicks_col].sum() else 0
        metrics_list.append(metric("CPC", f"{cpc:.2f}", description="Cost per click"))

    if spend_col and revenue_col:
        roas = (df[revenue_col].sum() / df[spend_col].sum()) if df[spend_col].sum() else 0
        metrics_list.append(metric("ROAS", f"{roas:.2f}", description="Return on ad spend"))

    grouping_cols = [col for col in [mapping.get("campaign"), mapping.get("channel")] if col in df.columns]
    if grouping_cols:
        aggregated = df.groupby(grouping_cols).agg({val: "sum" for val in available.values() if val in df.columns}).reset_index()
        table_data = aggregated
    else:
        table_data = df[list({col for col in available.values() if col})]

    table = dataframe_to_table(
        table_data,
        title="Campaign Performance",
        description="Aggregated metrics by mapped campaign/channel columns (if provided).",
    )

    return AnalysisResult(
        analysis_id="campaign_metrics_analysis",
        title="Campaign Metrics Analysis",
        summary="Core paid media KPIs from mapped marketing columns.",
        metrics=metrics_list,
        tables=[table],
        insights=[insight("info", "Review CTR, CPC, and ROAS to benchmark campaign efficiency." )],
    )


def conversion_funnel_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    mapping = _require_mapping(context)
    funnel_steps = [
        mapping.get("landed"),
        mapping.get("viewed_product"),
        mapping.get("added_to_cart"),
        mapping.get("started_checkout"),
        mapping.get("completed_purchase"),
    ]
    steps = [step for step in funnel_steps if step in df.columns]
    if not steps:
        return AnalysisResult(
            analysis_id="conversion_funnel_analysis",
            title="Conversion Funnel Analysis",
            summary="Provide boolean columns representing funnel steps (landing, view, cart, checkout, purchase).",
            status="warning",
            insights=[insight("warning", "Map dataset columns to funnel steps to compute drop-off rates.")],
        )

    counts = []
    for step in steps:
        counts.append({"Step": step, "Visitors": int(df[step].sum())})
    funnel_df = pd.DataFrame(counts)
    funnel_df["Drop-Off %"] = funnel_df["Visitors"].pct_change().fillna(0).abs() * 100

    table = dataframe_to_table(
        funnel_df,
        title="Funnel Drop-Off",
        description="Visitors per funnel stage with sequential drop-off percentage.",
        round_decimals=2,
    )

    return AnalysisResult(
        analysis_id="conversion_funnel_analysis",
        title="Conversion Funnel Analysis",
        summary="Tracks drop-off through the mapped conversion funnel steps.",
        tables=[table],
        insights=[insight("info", "Target stages with highest drop-off for optimization." )],
    )


def engagement_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    mapping = _require_mapping(context)
    metric_cols = [mapping.get("session_duration"), mapping.get("pages_viewed"), mapping.get("interactions")]
    metric_cols = [col for col in metric_cols if col in df.columns]
    if not metric_cols:
        return AnalysisResult(
            analysis_id="engagement_analysis",
            title="Engagement Analysis",
            summary="Provide mappings for session duration, pages viewed, or interactions.",
            status="warning",
            insights=[insight("warning", "Map engagement metrics to analyze session quality.")],
        )

    summary = df[metric_cols].describe().transpose()
    table = dataframe_to_table(
        summary.reset_index().rename(columns={"index": "Metric"}),
        title="Engagement Metrics",
        description="Descriptive statistics for engagement measures.",
        round_decimals=2,
    )

    return AnalysisResult(
        analysis_id="engagement_analysis",
        title="Engagement Analysis",
        summary="Descriptive statistics for engagement-focused columns.",
        tables=[table],
        insights=[insight("info", "Higher medians signal stronger engagement." )],
    )


def channel_performance_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    mapping = _require_mapping(context)
    channel_col = mapping.get("channel")
    if channel_col not in df.columns:
        return AnalysisResult(
            analysis_id="channel_performance_analysis",
            title="Channel Performance Analysis",
            summary="Provide a channel/source column mapping to aggregate performance.",
            status="warning",
            insights=[insight("warning", "Map a channel column to evaluate performance per acquisition source." )],
        )

    metric_cols = {metric: mapping.get(metric) for metric in ["sessions", "conversions", "revenue", "spend"] if mapping.get(metric) in df.columns}
    if not metric_cols:
        return AnalysisResult(
            analysis_id="channel_performance_analysis",
            title="Channel Performance Analysis",
            summary="Provide mappings for sessions, conversions, revenue, or spend.",
            status="warning",
            insights=[insight("warning", "Map numerical channel metrics to compute KPIs." )],
        )

    grouped = df.groupby(channel_col).agg({col: "sum" for col in metric_cols.values()}).reset_index()
    table = dataframe_to_table(
        grouped,
        title="Channel Performance",
        description="Aggregated metrics per marketing channel.",
        round_decimals=2,
    )

    return AnalysisResult(
        analysis_id="channel_performance_analysis",
        title="Channel Performance Analysis",
        summary="Aggregated performance by marketing channel.",
        tables=[table],
        insights=[insight("info", "Identify top-performing channels and scale accordingly." )],
    )


def audience_segmentation_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    mapping = _require_mapping(context)
    segmentation_cols = [mapping.get("age"), mapping.get("gender"), mapping.get("location")]
    segmentation_cols = [col for col in segmentation_cols if col in df.columns]
    if not segmentation_cols:
        return AnalysisResult(
            analysis_id="audience_segmentation_analysis",
            title="Audience Segmentation Analysis",
            summary="Provide demographic or behavioral columns (age, gender, location).",
            status="warning",
            insights=[insight("warning", "Map demographic columns to analyze audience segmentation." )],
        )

    tables: List[AnalysisTable] = []
    for col in segmentation_cols:
        counts = df[col].value_counts(dropna=False).reset_index()
        counts.columns = [col, "Count"]
        tables.append(
            dataframe_to_table(
                counts,
                title=f"Segments: {col}",
                description="Audience segments by count.",
            )
        )

    return AnalysisResult(
        analysis_id="audience_segmentation_analysis",
        title="Audience Segmentation Analysis",
        summary="Distribution of mapped audience segmentation attributes.",
        tables=tables,
        insights=[insight("info", "Use segmentation to tailor marketing messaging." )],
    )


def roi_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    mapping = _require_mapping(context)
    spend_col = mapping.get("cost") or mapping.get("spend")
    revenue_col = mapping.get("revenue")
    if spend_col not in df.columns or revenue_col not in df.columns:
        return AnalysisResult(
            analysis_id="roi_analysis",
            title="ROI Analysis",
            summary="Provide mappings for spend/cost and revenue columns.",
            status="warning",
            insights=[insight("warning", "Map spend and revenue columns to compute ROI." )],
        )

    total_spend = df[spend_col].sum()
    total_revenue = df[revenue_col].sum()
    roi = (total_revenue - total_spend) / total_spend * 100 if total_spend else np.nan

    metrics_list = [
        metric("Spend", round(total_spend, 2)),
        metric("Revenue", round(total_revenue, 2)),
        metric("ROI", f"{roi:.2f}", unit="%" if not np.isnan(roi) else None),
    ]

    return AnalysisResult(
        analysis_id="roi_analysis",
        title="ROI Analysis",
        summary="Return on investment computed from mapped spend and revenue columns.",
        metrics=metrics_list,
        insights=[insight("info", "Positive ROI indicates profitable marketing investment." )],
    )


def attribution_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    mapping = _require_mapping(context)
    touchpoint_col = mapping.get("touchpoint")
    conversion_col = mapping.get("conversion")
    if touchpoint_col not in df.columns or conversion_col not in df.columns:
        return AnalysisResult(
            analysis_id="attribution_analysis",
            title="Attribution Analysis",
            summary="Provide touchpoint and conversion columns to compute attribution.",
            status="warning",
            insights=[insight("warning", "Map touchpoint and conversion columns for attribution analysis." )],
        )

    attribution = df.groupby(touchpoint_col)[conversion_col].sum().reset_index()
    attribution.columns = ["Touchpoint", "Conversions"]
    table = dataframe_to_table(
        attribution.sort_values("Conversions", ascending=False),
        title="Touchpoint Attribution",
        description="Total conversions attributed to each touchpoint.",
    )

    return AnalysisResult(
        analysis_id="attribution_analysis",
        title="Attribution Analysis",
        summary="Conversions aggregated by marketing touchpoint.",
        tables=[table],
        insights=[insight("info", "High-converting touchpoints deserve more investment." )],
    )


def cohort_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    mapping = _require_mapping(context)
    signup_col = mapping.get("signup_date")
    activity_col = mapping.get("activity_date")
    if signup_col not in df.columns or activity_col not in df.columns:
        return AnalysisResult(
            analysis_id="cohort_analysis",
            title="Cohort Analysis",
            summary="Provide signup and activity date columns for cohort analysis.",
            status="warning",
            insights=[insight("warning", "Map signup and activity date columns to compute retention cohorts." )],
        )

    cohort_df = df[[signup_col, activity_col]].dropna()
    cohort_df[signup_col] = pd.to_datetime(cohort_df[signup_col])
    cohort_df[activity_col] = pd.to_datetime(cohort_df[activity_col])
    cohort_df["Cohort"] = cohort_df[signup_col].dt.to_period("M")
    cohort_df["Activity Month"] = cohort_df[activity_col].dt.to_period("M")
    cohort_df["Period"] = (cohort_df["Activity Month"] - cohort_df["Cohort"]).apply(lambda x: x.n)

    cohort_pivot = cohort_df.groupby(["Cohort", "Period"]).size().reset_index(name="Users")
    cohort_size = cohort_pivot[cohort_pivot["Period"] == 0][["Cohort", "Users"]].rename(columns={"Users": "Cohort Size"})
    pivot_table = cohort_pivot.pivot(index="Cohort", columns="Period", values="Users").fillna(0)
    retention = pivot_table.divide(cohort_size.set_index("Cohort")["Cohort Size"], axis=0) * 100

    table = dataframe_to_table(
        retention.reset_index(),
        title="Cohort Retention (%)",
        description="Retention percentage by cohort over activity periods (months).",
        round_decimals=1,
    )

    return AnalysisResult(
        analysis_id="cohort_analysis",
        title="Cohort Analysis",
        summary="Monthly cohort retention based on signup and activity dates.",
        tables=[table],
        insights=[insight("info", "Monitor retention decay to improve lifecycle marketing." )],
    )


__all__ = [
    "campaign_metrics_analysis",
    "conversion_funnel_analysis",
    "engagement_analysis",
    "channel_performance_analysis",
    "audience_segmentation_analysis",
    "roi_analysis",
    "attribution_analysis",
    "cohort_analysis",
]
