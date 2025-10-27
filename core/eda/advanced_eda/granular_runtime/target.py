"""Runtime handler for target variable analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api import types as ptypes

from .common import (
    AnalysisChart,
    AnalysisContext,
    AnalysisResult,
    dataframe_to_table,
    fig_to_base64,
    insight,
    metric,
)

TARGET_NAME_HINTS = {
    "target",
    "label",
    "outcome",
    "response",
    "class",
    "status",
    "result",
    "churn",
    "default",
    "fraud",
    "success",
    "converted",
    "y",
}

TARGET_MAX_COLUMNS = 3


@dataclass
class TargetCandidate:
    column: str
    score: float
    role: str
    unique_count: int
    unique_ratio: float
    missing_pct: float


def _score_target_column(series: pd.Series, column: str) -> TargetCandidate:
    total_count = int(series.size)
    non_null = series.dropna()
    non_null_count = int(non_null.size)
    unique_count = int(non_null.nunique()) if non_null_count else 0
    unique_ratio = (unique_count / non_null_count) if non_null_count else 0.0
    missing_pct = (total_count - non_null_count) / total_count if total_count else 0.0

    lower_name = column.lower()
    score = 0.0
    role = "regression"

    is_numeric = ptypes.is_numeric_dtype(series)
    is_boolean = ptypes.is_bool_dtype(series)
    is_categorical = (
        ptypes.is_categorical_dtype(series)
        or ptypes.is_object_dtype(series)
        or is_boolean
    )

    if lower_name in TARGET_NAME_HINTS:
        score += 4.0
    else:
        for hint in TARGET_NAME_HINTS:
            if hint in lower_name:
                score += 2.5
                break

    if unique_count <= 1 or total_count <= 1:
        score -= 4.0

    # Determine likely role
    if is_categorical or unique_count <= max(20, total_count // 5):
        role = "classification"
        score += 2.0
        if unique_count in {2, 3}:
            score += 1.5
        if unique_ratio <= 0.25:
            score += 0.5
    elif is_numeric:
        role = "regression"
        score += 1.5
        if unique_ratio >= 0.4:
            score += 1.0
    else:
        role = "classification" if unique_count <= 20 else "regression"
        score += 0.5

    if unique_ratio > 0.95 and unique_count > 20:
        score -= 3.0  # Likely identifier

    if missing_pct > 0.4:
        score -= 1.0
    elif missing_pct > 0.2:
        score -= 0.5

    return TargetCandidate(
        column=column,
        score=score,
        role=role,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        missing_pct=missing_pct * 100,
    )


def _detect_target_columns(
    df: pd.DataFrame,
    explicit: Optional[Sequence[str]] = None,
    max_candidates: int = TARGET_MAX_COLUMNS,
) -> Tuple[List[str], Dict[str, TargetCandidate], str]:
    metadata: Dict[str, TargetCandidate] = {}
    source = "detected"

    if explicit:
        columns = [col for col in explicit if col in df.columns]
        for col in columns:
            metadata[col] = _score_target_column(df[col], col)
        return columns[:max_candidates], metadata, "manual"

    candidates: List[TargetCandidate] = []
    for column in df.columns:
        candidate = _score_target_column(df[column], column)
        candidates.append(candidate)

    # Sort by score descending, fall back to unique count to break ties
    candidates.sort(key=lambda c: (c.score, -c.unique_count), reverse=True)

    selected: List[str] = []
    for candidate in candidates:
        if len(selected) >= max_candidates:
            break
        if candidate.score <= 0 and selected:
            continue
        selected.append(candidate.column)
        metadata[candidate.column] = candidate

    if not selected and candidates:
        best = max(candidates, key=lambda c: c.score)
        selected = [best.column]
        metadata[best.column] = best

    return selected, metadata, source


def target_variable_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    selected_columns = context.selected_columns or []
    target_columns, column_candidates, selection_source = _detect_target_columns(df, selected_columns)

    if not target_columns:
        return AnalysisResult(
            analysis_id="target_variable_analysis",
            title="Target Variable Analysis",
            summary="No suitable target columns were identified for analysis.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Select one or more target columns or ensure the dataset contains label-like fields.",
                )
            ],
        )

    metrics = [metric("Target columns analysed", len(target_columns))]
    insights: List = []
    tables: List = []
    charts: List[AnalysisChart] = []

    classification_count = 0
    regression_count = 0
    analysed_columns = 0

    selection_details: List[Dict[str, float]] = []

    for column in target_columns:
        series = df[column]
        total_count = int(series.size)
        missing_count = int(series.isna().sum())
        non_null = series.dropna()
        non_null_count = int(non_null.size)

        candidate_info = column_candidates.get(column) or _score_target_column(series, column)
        selection_details.append(
            {
                "column": column,
                "score": round(candidate_info.score, 3),
                "role": candidate_info.role,
                "unique_count": candidate_info.unique_count,
                "unique_ratio": round(candidate_info.unique_ratio, 4),
                "missing_pct": round(candidate_info.missing_pct, 2),
            }
        )

        if non_null_count == 0:
            insights.append(
                insight(
                    "warning",
                    f"Column '{column}' contains only missing values and was skipped.",
                )
            )
            continue

        analysed_columns += 1
        missing_pct = (missing_count / total_count * 100) if total_count else 0.0

        if candidate_info.role == "classification":
            classification_count += 1
            counts = non_null.value_counts().sort_values(ascending=False)
            percent = (counts / total_count * 100) if total_count else counts * 0

            class_table = pd.DataFrame(
                {
                    "Column": column,
                    "Class": counts.index.astype(str),
                    "Count": counts.values.astype(int),
                    "Percent": percent.round(2),
                }
            )
            tables.append(
                dataframe_to_table(
                    class_table,
                    title=f"Class balance • {column}",
                    description="Class distribution with share of total rows.",
                )
            )

            majority_share = float(percent.iloc[0]) if not percent.empty else 0.0
            minority_share = float(percent.iloc[-1]) if percent.size > 1 else majority_share

            metrics.extend(
                [
                    metric(f"{column} • Classes", counts.size),
                    metric(f"{column} • Majority share", round(majority_share, 2), unit="%"),
                    metric(f"{column} • Minority share", round(minority_share, 2), unit="%"),
                    metric(f"{column} • Missing", round(missing_pct, 2), unit="%"),
                ]
            )

            if majority_share >= 75:
                insights.append(
                    insight(
                        "warning",
                        f"{column}: dominant class holds {majority_share:.1f}% of records. Consider resampling strategies.",
                    )
                )
            elif majority_share >= 60:
                insights.append(
                    insight(
                        "info",
                        f"{column}: moderate imbalance with top class at {majority_share:.1f}% of rows.",
                    )
                )
            else:
                insights.append(
                    insight("success", f"{column}: class distribution appears reasonably balanced."),
                )

            top_counts = counts.head(15)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=top_counts.index.astype(str), y=top_counts.values, ax=ax, palette="Blues_d")
            ax.set_title(f"Class Distribution: {column}")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            charts.append(
                AnalysisChart(
                    title=f"Class Distribution: {column}",
                    image=fig_to_base64(fig),
                    description="Bar chart illustrating relative class sizes.",
                )
            )

        else:
            regression_count += 1
            desc = non_null.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            range_value = (float(desc.loc["max"]) - float(desc.loc["min"])) if "max" in desc and "min" in desc else None
            iqr_value = (float(desc.loc["75%"] - desc.loc["25%"])) if {"75%", "25%"}.issubset(desc.index) else None

            summary_row = pd.DataFrame(
                {
                    "Column": [column],
                    "Mean": [float(desc.get("mean", np.nan))],
                    "Median": [float(desc.get("50%", np.nan))],
                    "Std": [float(desc.get("std", np.nan))],
                    "Min": [float(desc.get("min", np.nan))],
                    "P10": [float(desc.get("10%", np.nan))],
                    "P90": [float(desc.get("90%", np.nan))],
                    "Max": [float(desc.get("max", np.nan))],
                    "Range": [range_value],
                    "IQR": [iqr_value],
                    "Missing %": [round(missing_pct, 2)],
                }
            )
            tables.append(
                dataframe_to_table(
                    summary_row,
                    title=f"Regression summary • {column}",
                    description="Summary statistics with tail percentiles for the target variable.",
                )
            )

            metrics.extend(
                [
                    metric(f"{column} • Mean", round(float(desc.get("mean", 0.0)), 3)),
                    metric(f"{column} • Std", round(float(desc.get("std", 0.0)), 3)),
                    metric(f"{column} • Range", round(range_value, 3) if range_value is not None else "n/a"),
                    metric(f"{column} • Missing", round(missing_pct, 2), unit="%"),
                ]
            )

            if range_value is not None and iqr_value is not None:
                spread_ratio = iqr_value / range_value if range_value else np.nan
                if not np.isnan(spread_ratio) and spread_ratio < 0.2:
                    insights.append(
                        insight(
                            "warning",
                            f"{column}: values are heavily concentrated in a narrow band ({spread_ratio:.2f} of total range).",
                        )
                    )
                else:
                    insights.append(
                        insight(
                            "info",
                            f"{column}: central 50% of values span {iqr_value:.3f} within a total range of {range_value:.3f}.",
                        )
                    )
            else:
                insights.append(
                    insight("info", f"{column}: inspected distribution for regression readiness."),
                )

            bin_count = min(40, max(12, int(np.sqrt(non_null_count))))
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(non_null, bins=bin_count, kde=True, ax=ax, color="#F6AD55")
            ax.set_title(f"Distribution: {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            charts.append(
                AnalysisChart(
                    title=f"Distribution: {column}",
                    image=fig_to_base64(fig),
                    description="Histogram with KDE overlay to review regression target spread.",
                )
            )

    if analysed_columns == 0:
        return AnalysisResult(
            analysis_id="target_variable_analysis",
            title="Target Variable Analysis",
            summary="Selected target columns did not contain analyzable data.",
            status="warning",
            insights=insights,
        )

    summary_parts = []
    if classification_count:
        summary_parts.append(f"{classification_count} classification")
    if regression_count:
        summary_parts.append(f"{regression_count} regression")

    summary_text = "Analysed " + " and ".join(summary_parts) + " target column(s)." if summary_parts else "Target variable analysis completed."
    details = {
        "selection_source": selection_source,
        "columns": selection_details,
    }

    status = "success" if tables or charts else "info"

    return AnalysisResult(
        analysis_id="target_variable_analysis",
        title="Target Variable Analysis",
        summary=summary_text,
        status=status,
        metrics=metrics,
        insights=insights,
        tables=tables,
        charts=charts,
        details=details,
    )
