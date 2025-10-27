from __future__ import annotations

"""Time-series focused analyses."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import warnings

from .common import (
    AnalysisChart,
    AnalysisContext,
    AnalysisResult,
    dataframe_to_table,
    fig_to_base64,
    insight,
    datetime_columns,
    metric,
    numeric_columns,
)

_COMMON_DATETIME_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%Y%m%d",
    "%d.%m.%Y",
    "%Y.%m.%d",
)


def _coerce_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    if series.empty:
        return pd.to_datetime(series, errors="coerce")

    clean_series = series
    if not isinstance(series, pd.Series):
        clean_series = pd.Series(series)

    for fmt in _COMMON_DATETIME_FORMATS:
        try:
            return pd.to_datetime(clean_series, format=fmt, errors="raise")
        except (ValueError, TypeError):
            continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(clean_series, errors="coerce")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def _strip_timezone(series: pd.Series) -> pd.Series:
    try:
        tz = getattr(series.dt, "tz", None)
        if tz is not None:
            return series.dt.tz_localize(None)
    except (AttributeError, TypeError):  # pragma: no cover - defensive guard
        return series
    return series


_FREQUENCY_LABELS = {
    "D": "Daily",
    "W": "Weekly",
    "M": "Monthly",
    "Q": "Quarterly",
}


def _resample_with_fallback(
    series: pd.Series,
    frequencies: List[str],
    min_points: int = 2,
) -> Tuple[Optional[pd.Series], Optional[str], Optional[str]]:
    errors: List[str] = []
    for freq in frequencies:
        try:
            resampled = series.resample(freq).mean().dropna()
        except Exception as exc:  # pragma: no cover - defensive guard
            errors.append(f"{freq}: {exc}")
            continue

        if resampled.size >= min_points:
            return resampled, freq, None

        if resampled.empty:
            errors.append(f"{freq}: no data remained after aggregation")
        else:
            errors.append(f"{freq}: only {resampled.size} aggregated point(s)")

    return None, None, "; ".join(errors)


def _prepare_time_series_inputs(
    df: pd.DataFrame, context: AnalysisContext
) -> Tuple[List[str], Dict[str, pd.Series], Dict[str, pd.Series]]:
    selected = list(context.selected_columns or [])

    dt_series_map: Dict[str, pd.Series] = {}
    dt_order: List[str] = []

    def add_datetime_column(name: str) -> None:
        if name in dt_series_map or name not in df.columns:
            return
        original = df[name]
        if not pd.api.types.is_datetime64_any_dtype(original):
            if not (
                pd.api.types.is_object_dtype(original)
                or pd.api.types.is_string_dtype(original)
                or pd.api.types.is_categorical_dtype(original)
            ):
                return

        series = _coerce_datetime(original)
        if series.notna().sum() == 0:
            return
        dt_series_map[name] = series
        dt_order.append(name)

    for name in selected:
        add_datetime_column(name)

    if not dt_order:
        for name in datetime_columns(df):
            add_datetime_column(name)

    numeric_series_map: Dict[str, pd.Series] = {}

    def add_numeric_column(name: str) -> None:
        if name in numeric_series_map or name in dt_series_map or name not in df.columns:
            return
        series = _coerce_numeric(df[name])
        if series.notna().sum() == 0:
            return
        numeric_series_map[name] = series

    for name in selected:
        add_numeric_column(name)

    if not numeric_series_map:
        for name in numeric_columns(df):
            add_numeric_column(name)

    if not numeric_series_map:
        for name in df.columns:
            add_numeric_column(name)

    return dt_order, dt_series_map, numeric_series_map


def temporal_trend_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    dt_cols, dt_series_map, numeric_series_map = _prepare_time_series_inputs(df, context)
    if not dt_cols or not numeric_series_map:
        selected = list(context.selected_columns or [])
        selected_hint = f" Selections considered: {', '.join(selected)}." if selected else ""
        return AnalysisResult(
            analysis_id="temporal_trend_analysis",
            title="Temporal Trend Analysis",
            summary="Requires both datetime and numeric columns to analyze trends.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Select at least one datetime column and one numeric column to analyze temporal trends." + selected_hint,
                )
            ],
        )

    date_col = dt_cols[0]
    date_series = _strip_timezone(dt_series_map[date_col])

    charts: List[AnalysisChart] = []
    metrics: List = []
    warnings: List = []

    for value_col, numeric_series in numeric_series_map.items():
        ts_df = pd.DataFrame({date_col: date_series, value_col: numeric_series})
        ts_df = ts_df.dropna(subset=[date_col, value_col])
        if ts_df[value_col].notna().sum() < 2:
            warnings.append(
                insight(
                    "warning",
                    f"{value_col}: not enough valid numeric points after alignment with {date_col}.",
                )
            )
            continue

        ts_df = ts_df.sort_values(date_col)

        resampled, freq_used, error = _resample_with_fallback(
            ts_df.set_index(date_col)[value_col],
            frequencies=["W", "D"],
            min_points=2,
        )

        if resampled is None or freq_used is None:
            details = f" ({error})" if error else ""
            warnings.append(
                insight(
                    "warning",
                    f"{value_col}: insufficient data after aggregation{details}.",
                )
            )
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        resampled.plot(ax=ax, color="#2C7A7B")
        freq_label = _FREQUENCY_LABELS.get(freq_used, freq_used)
        ax.set_title(f"{freq_label} Trend: {value_col}")
        ax.set_xlabel(freq_label)
        ax.set_ylabel(value_col)
        fig.autofmt_xdate()

        charts.append(
            AnalysisChart(
                title=f"{freq_label} Trend: {value_col}",
                image=fig_to_base64(fig),
                description=f"{freq_label} averaged trend of the numeric column.",
            )
        )

        range_start = resampled.index.min()
        range_end = resampled.index.max()
        if range_start is not None and range_end is not None:
            metrics.append(
                metric(
                    f"{value_col} Date Range",
                    f"{range_start.date()} → {range_end.date()}",
                )
            )
        else:
            warnings.append(
                insight(
                    "info",
                    f"{value_col}: unable to determine date range for the aggregated series.",
                )
            )

    if not charts:
        return AnalysisResult(
            analysis_id="temporal_trend_analysis",
            title="Temporal Trend Analysis",
            summary="No valid datetime and numeric pair produced a trend chart.",
            status="warning",
            insights=warnings
            or [
                insight(
                    "warning",
                    "Ensure your datetime and numeric selections contain parseable values (NaNs are skipped).",
                )
            ],
        )

    return AnalysisResult(
        analysis_id="temporal_trend_analysis",
        title="Temporal Trend Analysis",
        summary=f"Temporal trend over time using {date_col} across {len(charts)} numeric column(s).",
        metrics=metrics,
        charts=charts,
        insights=warnings
        + [
            insight(
                "info",
                "Aggregation frequency adapts between daily and weekly to keep at least two points for trend lines.",
            )
        ],
    )


def seasonality_detection(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    dt_cols, dt_series_map, numeric_series_map = _prepare_time_series_inputs(df, context)
    if not dt_cols or not numeric_series_map:
        selected = list(context.selected_columns or [])
        selected_hint = f" Selections considered: {', '.join(selected)}." if selected else ""
        return AnalysisResult(
            analysis_id="seasonality_detection",
            title="Seasonality Detection",
            summary="Requires both datetime and numeric columns to detect seasonality.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Select datetime and numeric columns to detect seasonality." + selected_hint,
                )
            ],
        )

    date_col = dt_cols[0]
    date_series = _strip_timezone(dt_series_map[date_col])

    charts: List[AnalysisChart] = []
    warnings: List = []

    for value_col, numeric_series in numeric_series_map.items():
        ts_df = pd.DataFrame({date_col: date_series, value_col: numeric_series})
        ts_df = ts_df.dropna(subset=[date_col, value_col])
        if ts_df[value_col].notna().sum() < 2:
            warnings.append(
                insight(
                    "warning",
                    f"{value_col}: not enough valid numeric points after alignment with {date_col}.",
                )
            )
            continue

        ts_df = ts_df.sort_values(date_col)

        resampled, freq_used, error = _resample_with_fallback(
            ts_df.set_index(date_col)[value_col],
            frequencies=["M", "W"],
            min_points=2,
        )

        if resampled is None or freq_used is None:
            details = f" ({error})" if error else ""
            warnings.append(
                insight(
                    "warning",
                    f"{value_col}: insufficient data after aggregation{details}.",
                )
            )
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        resampled.plot(ax=ax, marker="o", color="#D53F8C")
        freq_label = _FREQUENCY_LABELS.get(freq_used, freq_used)
        ax.set_xlabel(freq_label)
        ax.set_ylabel(value_col)
        ax.set_title(f"{freq_label} Seasonality: {value_col}")
        fig.autofmt_xdate()
        charts.append(
            AnalysisChart(
                title=f"{freq_label} Seasonality: {value_col}",
                image=fig_to_base64(fig),
                description=f"{freq_label} averaged trend revealing seasonal patterns.",
            )
        )

    if not charts:
        return AnalysisResult(
            analysis_id="seasonality_detection",
            title="Seasonality Detection",
            summary="No valid datetime and numeric pair produced a seasonality chart.",
            status="warning",
            insights=warnings
            or [
                insight(
                    "warning",
                    "Ensure your datetime and numeric selections contain parseable values (NaNs are skipped).",
                )
            ],
        )

    return AnalysisResult(
        analysis_id="seasonality_detection",
        title="Seasonality Detection",
        summary=f"Seasonality insights using {date_col} across {len(charts)} numeric column(s).",
        charts=charts,
        insights=warnings
        + [
            insight(
                "info",
                "Aggregation frequency adapts between monthly and weekly to retain enough observations for seasonality detection.",
            )
        ],
    )


def datetime_feature_extraction(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    dt_cols, dt_series_map, _ = _prepare_time_series_inputs(df, context)
    if not dt_cols:
        return AnalysisResult(
            analysis_id="datetime_feature_extraction",
            title="Datetime Feature Extraction",
            summary="No datetime-like columns available to derive calendar features.",
            status="warning",
            insights=[
                insight("warning", "Add or convert a column to datetime to extract calendar features."),
            ],
        )

    rows = []
    seasonal_columns: List[str] = []
    intraday_columns: List[str] = []
    coverage_alerts: List[str] = []

    for col in dt_cols:
        raw_series = df[col] if col in df.columns else pd.Series(dtype="datetime64[ns]")
        normalized = dt_series_map.get(col, _coerce_datetime(raw_series))
        valid = normalized.dropna()
        total_count = int(raw_series.size)
        valid_count = int(valid.size)
        if valid_count == 0:
            coverage_alerts.append(col)
            continue

        coverage_pct = valid_count / total_count * 100 if total_count else 0.0
        missing_pct = 100 - coverage_pct

        features = [
            ("Year", valid.dt.year, "Use for long-term trend splits"),
            ("Quarter", valid.dt.quarter, "Helpful for quarterly reporting"),
            ("Month", valid.dt.month, "Captures seasonal cycles"),
            ("Week of Year", valid.dt.isocalendar().week, "Highlights weekly operational cadence"),
            ("Weekday", valid.dt.weekday, "Detect weekday vs weekend patterns"),
            ("Day", valid.dt.day, "Use for monthly seasonality evaluation"),
            ("Hour", valid.dt.hour, "Model intra-day behaviour"),
        ]

        month_unique = valid.dt.month.nunique()
        hour_unique = valid.dt.hour.nunique()
        if month_unique > 1:
            seasonal_columns.append(col)
        if hour_unique > 1:
            intraday_columns.append(col)

        for feature_name, series_slice, note in features:
            try:
                unique_values = int(series_slice.nunique())
            except Exception:  # pragma: no cover - defensive guard for legacy pandas
                unique_values = 0

            rows.append(
                {
                    "Column": col,
                    "Feature": feature_name,
                    "Unique Values": unique_values,
                    "Coverage %": round(coverage_pct, 2),
                    "Notes": note if unique_values > 1 else "Low variance – may be optional",
                }
            )

        rows.append(
            {
                "Column": col,
                "Feature": "Missing",
                "Unique Values": missing_pct,
                "Coverage %": round(coverage_pct, 2),
                "Notes": "Percentage of records without a valid timestamp.",
            }
        )

    if not rows:
        return AnalysisResult(
            analysis_id="datetime_feature_extraction",
            title="Datetime Feature Extraction",
            summary="Datetime parsing failed for the detected columns.",
            status="warning",
            insights=[
                insight(
                    "danger",
                    "Parsing produced no valid timestamps. Check date formats or timezone offsets.",
                ),
            ],
        )

    table = dataframe_to_table(
        pd.DataFrame(rows),
        title="Calendar Feature Candidates",
        description="Suggested datetime-derived features with variance and coverage context.",
    )

    insights = []
    if seasonal_columns:
        sample = ", ".join(seasonal_columns[:5])
        more = "…" if len(seasonal_columns) > 5 else ""
        insights.append(
            insight(
                "info",
                f"Monthly or quarterly seasonality detected in: {sample}{more}. Consider adding month/quarter encodings.",
            )
        )
    if intraday_columns:
        sample = ", ".join(intraday_columns[:5])
        more = "…" if len(intraday_columns) > 5 else ""
        insights.append(
            insight(
                "info",
                f"Intraday variation present in: {sample}{more}. Include hour-of-day or shift indicators.",
            )
        )
    if coverage_alerts:
        sample = ", ".join(coverage_alerts[:5])
        more = "…" if len(coverage_alerts) > 5 else ""
        insights.append(
            insight(
                "warning",
                f"Datetime coverage below 5% for: {sample}{more}. You may need to clean or backfill timestamps.",
            )
        )
    if not insights:
        insights.append(
            insight("success", "Datetime columns parsed cleanly with multiple feature options."),
        )

    metrics = [
        metric("Datetime Columns", len(dt_cols)),
        metric("Rows Analysed", len(df)),
        metric("Feature Rows", len(rows)),
    ]

    return AnalysisResult(
        analysis_id="datetime_feature_extraction",
        title="Datetime Feature Extraction",
        summary="Evaluates candidate calendar features and timestamp coverage for datetime columns.",
        metrics=metrics,
        tables=[table],
        insights=insights,
    )


__all__ = [
    "temporal_trend_analysis",
    "seasonality_detection",
    "datetime_feature_extraction",
]
