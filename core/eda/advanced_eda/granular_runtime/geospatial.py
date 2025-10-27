from __future__ import annotations

"""Geospatial analysis runtime handlers."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd

from .common import (
    AnalysisChart,
    AnalysisContext,
    AnalysisInsight,
    AnalysisMetric,
    AnalysisResult,
    AnalysisTable,
    dataframe_to_table,
    fig_to_base64,
    insight,
    metric,
)


@dataclass
class _GeospatialInputs:
    latitude: str
    longitude: str
    label: Optional[str]
    frame: pd.DataFrame


_LATITUDE_HINTS = ("latitude", "lat", "y", "latitud", "lat_deg")
_LONGITUDE_HINTS = ("longitude", "lon", "lng", "long", "x", "longitud", "lon_deg")
_LABEL_HINTS = ("name", "city", "region", "area", "location", "place", "label", "site")


def _select_by_hints(columns: Iterable[str], hints: Tuple[str, ...]) -> Optional[str]:
    lower_lookup = {col.lower(): col for col in columns}
    for hint in hints:
        for candidate in lower_lookup:
            if hint in candidate and lower_lookup[candidate] != "":
                return lower_lookup[candidate]
    return None


def _detect_geospatial_inputs(
    df: pd.DataFrame,
    context: AnalysisContext,
    *,
    include_label: bool = True,
) -> Optional[_GeospatialInputs]:
    mapping = context.column_mapping or {}
    selected = list(context.selected_columns or [])

    def resolve(preferred: Optional[str], hints: Tuple[str, ...]) -> Optional[str]:
        if preferred and preferred in df.columns:
            return preferred
        for name in selected:
            if name in df.columns and any(h in name.lower() for h in hints):
                return name
        detected = _select_by_hints(df.columns, hints)
        if detected:
            return detected
        return None

    latitude = resolve(mapping.get("latitude"), _LATITUDE_HINTS)
    longitude = resolve(mapping.get("longitude"), _LONGITUDE_HINTS)

    if latitude is None or longitude is None:
        return None

    # Optional label/tooltip column
    label: Optional[str] = None
    if include_label:
        label = resolve(mapping.get("location_label") or mapping.get("label"), _LABEL_HINTS)

    working = df[[latitude, longitude]].copy()
    working.columns = ["latitude", "longitude"]
    working = working.dropna(subset=["latitude", "longitude"])

    if working.empty:
        return None

    # Ensure numeric
    for col in ("latitude", "longitude"):
        if not pd.api.types.is_numeric_dtype(working[col]):
            working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.dropna(subset=["latitude", "longitude"])

    if working.empty:
        return None

    if include_label and label and label in df.columns:
        working[label] = df.loc[working.index, label]

    return _GeospatialInputs(
        latitude=latitude,
        longitude=longitude,
        label=label if include_label else None,
        frame=working,
    )


def _haversine(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Return pairwise great-circle distances in kilometres."""

    # Convert to radians
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)

    diff_lat = lat_rad[:, None] - lat_rad[None, :]
    diff_lon = lon_rad[:, None] - lon_rad[None, :]

    a = np.sin(diff_lat / 2.0) ** 2 + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * np.sin(diff_lon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 0)))
    earth_radius_km = 6371.0
    return earth_radius_km * c


def _haversine_pairwise(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Return haversine distance between matching coordinate arrays in kilometres."""

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    diff_lat = lat2_rad - lat1_rad
    diff_lon = lon2_rad - lon1_rad

    a = np.sin(diff_lat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(diff_lon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 0)))
    earth_radius_km = 6371.0
    return earth_radius_km * c


def _prepare_sample(inputs: _GeospatialInputs, sample_size: int = 500, random_state: Optional[int] = None) -> pd.DataFrame:
    frame = inputs.frame
    if len(frame) <= sample_size:
        return frame
    return frame.sample(n=sample_size, random_state=random_state)


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return None
            return float(stripped)
        return float(value)
    except (TypeError, ValueError):
        return None


def spatial_data_quality_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    inputs = _detect_geospatial_inputs(df, context, include_label=False)
    if not inputs:
        return AnalysisResult(
            analysis_id="spatial_data_quality_analysis",
            title="Spatial Data Quality Analysis",
            summary="Latitude and longitude columns are required to audit spatial data quality.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Select or map latitude and longitude columns to evaluate spatial data integrity.",
                )
            ],
        )

    latitude_col = inputs.latitude
    longitude_col = inputs.longitude

    lat_raw = df[latitude_col]
    lon_raw = df[longitude_col]

    lat_numeric = _coerce_numeric_series(lat_raw)
    lon_numeric = _coerce_numeric_series(lon_raw)

    total_rows = len(df)
    valid_mask = lat_numeric.notna() & lon_numeric.notna()
    valid_pairs = int(valid_mask.sum())

    missing_lat = int(lat_raw.isna().sum())
    missing_lon = int(lon_raw.isna().sum())
    missing_both = int((lat_raw.isna() & lon_raw.isna()).sum())

    coerced_lat_missing = int(lat_raw.notna().sum() - lat_numeric.notna().sum())
    coerced_lon_missing = int(lon_raw.notna().sum() - lon_numeric.notna().sum())

    out_of_range_lat_mask = lat_numeric.abs() > 90
    out_of_range_lon_mask = lon_numeric.abs() > 180
    out_of_range_lat = int(out_of_range_lat_mask.sum())
    out_of_range_lon = int(out_of_range_lon_mask.sum())

    zero_coordinate_pairs = int(((lat_numeric == 0) & (lon_numeric == 0)).sum())

    rounded_pairs = pd.DataFrame({
        "latitude": lat_numeric.round(6),
        "longitude": lon_numeric.round(6),
    })
    duplicate_pairs = int(rounded_pairs[valid_mask].duplicated().sum())

    issue_rows_mask = (~valid_mask) | out_of_range_lat_mask | out_of_range_lon_mask

    metrics = [
        metric("Rows analysed", total_rows),
        metric("Valid coordinate pairs", valid_pairs),
        metric(
            "Valid pair coverage (%)",
            round((valid_pairs / total_rows) * 100, 2) if total_rows else 0,
        ),
        metric("Missing latitude", missing_lat),
        metric("Missing longitude", missing_lon),
        metric("Non-numeric latitude", coerced_lat_missing),
        metric("Non-numeric longitude", coerced_lon_missing),
        metric("Out-of-range latitude", out_of_range_lat),
        metric("Out-of-range longitude", out_of_range_lon),
        metric("Zero coordinate pairs", zero_coordinate_pairs),
        metric("Duplicate coordinate pairs", duplicate_pairs),
    ]

    issue_records = [
        ("Missing latitude", missing_lat),
        ("Missing longitude", missing_lon),
        ("Non-numeric latitude", coerced_lat_missing),
        ("Non-numeric longitude", coerced_lon_missing),
        ("Out-of-range latitude", out_of_range_lat),
        ("Out-of-range longitude", out_of_range_lon),
        ("Zero coordinate pair", zero_coordinate_pairs),
        ("Duplicate coordinate pair", duplicate_pairs),
    ]

    issue_df = pd.DataFrame(
        [
            {
                "Issue": label,
                "Count": count,
                "Percent": round((count / total_rows) * 100, 2) if total_rows else 0.0,
            }
            for label, count in issue_records
            if count > 0
        ]
    )

    tables = []
    if not issue_df.empty:
        tables.append(
            dataframe_to_table(
                issue_df,
                title="Spatial Data Quality Findings",
                description="Counts and proportions of detected coordinate quality issues.",
                round_decimals=2,
            )
        )

    if duplicate_pairs:
        duplicate_counts = (
            rounded_pairs[valid_mask]
            .value_counts()
            .reset_index(name="Occurrences")
            .sort_values("Occurrences", ascending=False)
        )
        duplicate_counts = duplicate_counts.rename(columns={"latitude": "Latitude", "longitude": "Longitude"})
        duplicate_counts = duplicate_counts[duplicate_counts["Occurrences"] > 1].copy()
        if not duplicate_counts.empty:
            duplicate_counts.loc[:, "Latitude"] = duplicate_counts["Latitude"].astype(float).round(6)
            duplicate_counts.loc[:, "Longitude"] = duplicate_counts["Longitude"].astype(float).round(6)
            tables.append(
                dataframe_to_table(
                    duplicate_counts.head(20),
                    title="Duplicate Coordinate Pairs",
                    description="Top duplicate latitude/longitude pairs (rounded to 6 decimal places).",
                    round_decimals=6,
                )
            )

    if issue_rows_mask.any():
        sample_columns = [latitude_col, longitude_col]
        fallback_columns = [inputs.label] if inputs.label and inputs.label in df.columns else []
        preview_columns = sample_columns + fallback_columns
        preview_df = df.loc[issue_rows_mask, preview_columns].copy()
        preview_df = preview_df.head(10)
        if not preview_df.empty:
            tables.append(
                dataframe_to_table(
                    preview_df,
                    title="Sample of Problematic Coordinates",
                    description="First 10 rows exhibiting missing, non-numeric, or out-of-range coordinates.",
                    round_decimals=6,
                )
            )

    charts: List[AnalysisChart] = []
    if valid_pairs > 0:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        valid_latitudes = lat_numeric[valid_mask]
        valid_longitudes = lon_numeric[valid_mask]
        axes[0].hist(valid_latitudes, bins=30, color="#2D9CDB", alpha=0.8)
        axes[0].set_title("Latitude Distribution")
        axes[0].set_xlabel("Latitude")
        axes[0].set_ylabel("Count")
        axes[1].hist(valid_longitudes, bins=30, color="#6C5CE7", alpha=0.8)
        axes[1].set_title("Longitude Distribution")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Count")
        fig.tight_layout()
        charts.append(
            AnalysisChart(
                title="Coordinate Distributions",
                image=fig_to_base64(fig),
                description="Histogram overview of valid latitude and longitude values.",
            )
        )
        plt.close(fig)

        density_counts = (
            rounded_pairs[valid_mask]
            .value_counts()
            .reset_index(name="Count")
        )
        fig_map, ax_map = plt.subplots(figsize=(6, 6))
        counts = density_counts["Count"].to_numpy()
        scatter = ax_map.scatter(
            density_counts["longitude"],
            density_counts["latitude"],
            c=counts,
            cmap="plasma",
            s=30 + counts * 10,
            alpha=0.85,
            edgecolor="#1b1f2326",
            linewidth=0.4,
        )
        if counts.size:
            min_count = counts.min()
            max_count = counts.max()
            if max_count == min_count:
                norm = plt.Normalize(vmin=min_count - 0.5, vmax=max_count + 0.5)
            else:
                norm = plt.Normalize(vmin=min_count, vmax=max_count)
            scatter.set_norm(norm)
            colors = plt.cm.plasma(norm(counts))
            colors[:, 3] = np.where(counts > 1, 0.85, 0.25)
            scatter.set_facecolors(colors)
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        ax_map.set_title("Spatial Distribution with Duplicate Density")
        colorbar = fig_map.colorbar(scatter, ax=ax_map, pad=0.01)
        colorbar.set_label("Duplicate frequency")
        fig_map.tight_layout()
        charts.append(
            AnalysisChart(
                title="Duplicate Density Scatter",
                image=fig_to_base64(fig_map),
                description="Scatter plot of valid coordinates where marker size and colour intensify with duplicate frequency.",
            )
        )
        plt.close(fig_map)

    status = "success"
    insights = [
        insight(
            "info",
            f"Analysed coordinates from latitude column '{latitude_col}' and longitude column '{longitude_col}'.",
        )
    ]

    if valid_pairs == 0:
        status = "warning"
        insights.append(
            insight(
                "danger",
                "No valid coordinate pairs remained after converting to numeric values.",
            )
        )
    else:
        valid_share = (valid_pairs / max(total_rows, 1))
        if valid_share < 0.7:
            status = "warning"
            insights.append(
                insight(
                    "warning",
                    f"Only {valid_share:.0%} of rows contained valid coordinate pairs after cleaning.",
                )
            )
        else:
            insights.append(
                insight(
                    "success",
                    f"{valid_share:.0%} of rows retained valid coordinates after cleaning.",
                )
            )

    if out_of_range_lat or out_of_range_lon:
        status = "warning"
        if out_of_range_lat:
            insights.append(
                insight(
                    "warning",
                    f"Detected {out_of_range_lat} latitude values outside the ±90° bounds.",
                )
            )
        if out_of_range_lon:
            insights.append(
                insight(
                    "warning",
                    f"Detected {out_of_range_lon} longitude values outside the ±180° bounds.",
                )
            )

    if duplicate_pairs:
        insights.append(
            insight(
                "info",
                f"Found {duplicate_pairs} duplicated coordinate pairs (rounded to 6 decimal places).",
            )
        )

    if zero_coordinate_pairs:
        insights.append(
            insight(
                "warning",
                f"{zero_coordinate_pairs} coordinate pairs were exactly (0, 0), which may indicate missing geocoding results.",
            )
        )

    summary = "Spatial data quality audit summarising missing, invalid, and duplicate coordinate pairs."

    details: Dict[str, Any] = {
        "columns": {"latitude": latitude_col, "longitude": longitude_col},
        "counts": {
            "total_rows": total_rows,
            "valid_pairs": valid_pairs,
            "missing_latitude": missing_lat,
            "missing_longitude": missing_lon,
            "non_numeric_latitude": coerced_lat_missing,
            "non_numeric_longitude": coerced_lon_missing,
            "out_of_range_latitude": out_of_range_lat,
            "out_of_range_longitude": out_of_range_lon,
            "zero_coordinate_pairs": zero_coordinate_pairs,
            "duplicate_pairs": duplicate_pairs,
        },
    }

    return AnalysisResult(
        analysis_id="spatial_data_quality_analysis",
        title="Spatial Data Quality Analysis",
        summary=summary,
        status=status,
        metrics=metrics,
        charts=charts,
        tables=tables,
        insights=insights,
        details=details,
    )


def geospatial_proximity_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    inputs = _detect_geospatial_inputs(df, context)
    if not inputs:
        return AnalysisResult(
            analysis_id="geospatial_proximity_analysis",
            title="Geospatial Proximity Analysis",
            summary="Latitude and longitude columns are required to measure proximity.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Select or map latitude and longitude columns to evaluate proximity metrics.",
                )
            ],
        )

    frame = inputs.frame
    if frame.empty:
        return AnalysisResult(
            analysis_id="geospatial_proximity_analysis",
            title="Geospatial Proximity Analysis",
            summary="No usable latitude/longitude pairs were found after preprocessing.",
            status="warning",
            insights=[
                insight("warning", "Add more observations with valid coordinate data to run this analysis."),
            ],
        )

    metadata = context.metadata or {}

    optional_mode_raw = metadata.get("proximity_optional_mode", "none")
    if isinstance(optional_mode_raw, str):
        optional_mode = optional_mode_raw.strip().lower()
    else:
        optional_mode = str(optional_mode_raw).strip().lower()
    if optional_mode not in {"radius", "comparison"}:
        optional_mode = "none"

    radius_km = _safe_float(metadata.get("proximity_radius_km"))
    invalid_radius = False
    if radius_km is not None and radius_km <= 0:
        radius_km = None
        invalid_radius = True

    reference_lat = _safe_float(metadata.get("reference_latitude"))
    reference_lon = _safe_float(metadata.get("reference_longitude"))
    manual_reference = reference_lat is not None and reference_lon is not None

    inconsistent_reference = (reference_lat is None) ^ (reference_lon is None)
    if inconsistent_reference:
        reference_lat = reference_lon = None

    reference_source = "dataset centroid"
    if manual_reference:
        reference_source = "user provided"
    else:
        reference_lat = float(frame["latitude"].mean())
        reference_lon = float(frame["longitude"].mean())

    latitudes = frame["latitude"].to_numpy()
    longitudes = frame["longitude"].to_numpy()

    ref_lat_rad = math.radians(reference_lat)
    ref_lon_rad = math.radians(reference_lon)
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)

    diff_lat = lat_rad - ref_lat_rad
    diff_lon = lon_rad - ref_lon_rad
    a = np.sin(diff_lat / 2.0) ** 2 + np.cos(ref_lat_rad) * np.cos(lat_rad) * np.sin(diff_lon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 0)))
    distances_km = 6371.0 * c

    mean_distance = float(np.mean(distances_km)) if len(distances_km) else float("nan")
    median_distance = float(np.median(distances_km)) if len(distances_km) else float("nan")
    percentile_90 = float(np.percentile(distances_km, 90)) if len(distances_km) else float("nan")

    within_radius_count: Optional[int] = None
    within_radius_share: Optional[float] = None
    if radius_km is not None:
        within_radius_mask = distances_km <= radius_km
        within_radius_count = int(within_radius_mask.sum())
        within_radius_share = (within_radius_count / len(distances_km)) * 100 if len(distances_km) else 0.0

    comparison_metrics: List[AnalysisMetric] = []
    comparison_insights: List[AnalysisInsight] = []
    comparison_details: Optional[Dict[str, Any]] = None
    comparison_table_df: Optional[pd.DataFrame] = None
    comparison_charts: List[AnalysisChart] = []
    comparison_warning = False

    comparison_lat_column = metadata.get("comparison_latitude_column")
    comparison_lon_column = metadata.get("comparison_longitude_column")

    if comparison_lat_column or comparison_lon_column:
        if comparison_lat_column and comparison_lon_column:
            missing_columns = [
                col
                for col in (comparison_lat_column, comparison_lon_column)
                if col not in df.columns
            ]
            if missing_columns:
                comparison_warning = True
                comparison_insights.append(
                    insight(
                        "warning",
                        f"Comparison columns {missing_columns} were not found in the dataset.",
                    )
                )
            else:
                comparison_lat_series = _coerce_numeric_series(df[comparison_lat_column]).reindex(frame.index)
                comparison_lon_series = _coerce_numeric_series(df[comparison_lon_column]).reindex(frame.index)

                comparison_frame = frame[["latitude", "longitude"]].copy()
                comparison_frame["comparison_latitude"] = comparison_lat_series
                comparison_frame["comparison_longitude"] = comparison_lon_series
                if inputs.label and inputs.label in df.columns and inputs.label not in comparison_frame.columns:
                    comparison_frame[inputs.label] = df.loc[comparison_frame.index, inputs.label]

                comparison_frame = comparison_frame.dropna(subset=["comparison_latitude", "comparison_longitude"])

                if comparison_frame.empty:
                    comparison_warning = True
                    comparison_insights.append(
                        insight(
                            "warning",
                            "No overlapping rows contained valid comparison coordinates to calculate row-level differences.",
                        )
                    )
                else:
                    deltas = _haversine_pairwise(
                        comparison_frame["latitude"].to_numpy(),
                        comparison_frame["longitude"].to_numpy(),
                        comparison_frame["comparison_latitude"].to_numpy(),
                        comparison_frame["comparison_longitude"].to_numpy(),
                    )

                    comparison_frame = comparison_frame.assign(**{"Delta (km)": deltas})
                    comparison_count = len(comparison_frame)
                    mean_delta_comp = float(np.mean(deltas))
                    median_delta_comp = float(np.median(deltas))
                    percentile90_comp = float(np.percentile(deltas, 90))
                    max_delta_comp = float(np.max(deltas))

                    comparison_metrics.extend(
                        [
                            metric("Comparison rows analysed", comparison_count),
                            metric("Mean comparison delta (km)", round(mean_delta_comp, 3)),
                            metric("Median comparison delta (km)", round(median_delta_comp, 3)),
                            metric("90th percentile comparison delta (km)", round(percentile90_comp, 3)),
                            metric("Maximum comparison delta (km)", round(max_delta_comp, 3)),
                        ]
                    )

                    fig_delta_hist, ax_delta_hist = plt.subplots(figsize=(7, 4))
                    ax_delta_hist.hist(deltas, bins=25, color="#6C5CE7", alpha=0.85)
                    ax_delta_hist.set_xlabel("Haversine delta (km)")
                    ax_delta_hist.set_ylabel("Count")
                    ax_delta_hist.set_title("Comparison Delta Distribution")
                    fig_delta_hist.tight_layout()
                    comparison_charts.append(
                        AnalysisChart(
                            title="Comparison Delta Distribution",
                            image=fig_to_base64(fig_delta_hist),
                            description="Histogram of haversine distance differences between the primary and comparison coordinates.",
                        )
                    )
                    plt.close(fig_delta_hist)

                    top_n = min(15, comparison_count)
                    if top_n > 0:
                        ranked = comparison_frame.nlargest(top_n, "Delta (km)").copy()
                        if inputs.label and inputs.label in ranked.columns:
                            y_labels = ranked[inputs.label].astype(str).fillna("(missing)").tolist()
                        else:
                            y_labels = [f"Row {idx}" for idx in ranked.index]
                        fig_delta_rank, ax_delta_rank = plt.subplots(figsize=(7, 5))
                        ax_delta_rank.barh(range(top_n), ranked["Delta (km)"], color="#E17055", alpha=0.85)
                        ax_delta_rank.set_yticks(range(top_n))
                        ax_delta_rank.set_yticklabels(y_labels)
                        ax_delta_rank.invert_yaxis()
                        ax_delta_rank.set_xlabel("Haversine delta (km)")
                        ax_delta_rank.set_title("Largest Coordinate Differences")
                        fig_delta_rank.tight_layout()
                        comparison_charts.append(
                            AnalysisChart(
                                title="Largest Coordinate Differences",
                                image=fig_to_base64(fig_delta_rank),
                                description="Top coordinate pairs ranked by haversine distance between the primary and comparison selections.",
                            )
                        )
                        plt.close(fig_delta_rank)

                    comparison_table_df = comparison_frame.copy()
                    comparison_details = {
                        "columns": {
                            "latitude": comparison_lat_column,
                            "longitude": comparison_lon_column,
                        },
                        "rows_compared": comparison_count,
                        "mean_delta_km": mean_delta_comp,
                        "median_delta_km": median_delta_comp,
                        "percentile_90_delta_km": percentile90_comp,
                        "max_delta_km": max_delta_comp,
                    }

                    comparison_insights.append(
                        insight(
                            "info",
                            "Calculated per-row coordinate differences using the haversine formula between the selected comparison columns and primary coordinates.",
                        )
                    )
        else:
            comparison_warning = True
            comparison_insights.append(
                insight(
                    "warning",
                    "Provide both comparison latitude and longitude columns to calculate per-row coordinate differences.",
                )
            )

    base_charts: List[AnalysisChart] = []
    if len(distances_km) > 0 and optional_mode != "comparison":
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(distances_km, bins=25, color="#0984E3", alpha=0.8)
        ax.set_xlabel("Distance to reference (km)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of distances to reference point")
        if radius_km is not None:
            ax.axvline(radius_km, color="#E17055", linestyle="--", linewidth=2, label=f"Radius {radius_km:.2f} km")
            ax.legend()
        base_charts.append(
            AnalysisChart(
                title="Proximity Distance Distribution",
                image=fig_to_base64(fig),
                description="Histogram of distances from each observation to the reference point.",
            )
        )
        plt.close(fig)

        fig_map, ax_map = plt.subplots(figsize=(6.5, 6))
        scatter = ax_map.scatter(
            longitudes,
            latitudes,
            c=distances_km,
            cmap="viridis",
            s=30 + np.clip(distances_km, 0, None) * 2.5,
            alpha=0.85,
            edgecolor="#1b1f2333",
            linewidth=0.4,
        )
        ax_map.scatter([reference_lon], [reference_lat], marker="*", color="#d63031", s=180, label="Reference")

        sample_count = min(len(longitudes), 200)
        if sample_count > 0:
            sample_indices = np.linspace(0, len(longitudes) - 1, sample_count, dtype=int)
            ax_map.quiver(
                np.full(sample_count, reference_lon),
                np.full(sample_count, reference_lat),
                longitudes[sample_indices] - reference_lon,
                latitudes[sample_indices] - reference_lat,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.0025,
                color="#636e72",
                alpha=0.25,
            )

        if radius_km is not None:
            approx_radius_deg = radius_km / 111.0
            coverage_circle = patches.Circle(
                (reference_lon, reference_lat),
                approx_radius_deg,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                edgecolor="#e17055",
                alpha=0.8,
            )
            ax_map.add_patch(coverage_circle)
            ax_map.legend(loc="upper right")

        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        ax_map.set_title("Spatial proximity to reference")
        colorbar = fig_map.colorbar(scatter, ax=ax_map, pad=0.01)
        colorbar.set_label("Distance (km)")
        fig_map.tight_layout()
        base_charts.append(
            AnalysisChart(
                title="Spatial Proximity Map",
                image=fig_to_base64(fig_map),
                description="Scatter plot of observations coloured by distance, with arrows pointing from the reference point.",
            )
        )
        plt.close(fig_map)

    charts = comparison_charts if optional_mode == "comparison" else base_charts + comparison_charts

    tables: List[AnalysisTable] = []
    if optional_mode != "comparison":
        ranking_df = frame.copy()
        ranking_df["Distance (km)"] = distances_km
        if inputs.label and inputs.label in ranking_df.columns:
            ranking_columns = [inputs.label, "latitude", "longitude", "Distance (km)"]
        else:
            ranking_columns = ["latitude", "longitude", "Distance (km)"]
        nearest_table_df = ranking_df[ranking_columns].sort_values("Distance (km)").head(10)
        tables.append(
            dataframe_to_table(
                nearest_table_df,
                title="Nearest Observations",
                description="Top 10 closest observations to the reference point (after cleaning).",
                round_decimals=3,
            )
        )

    if comparison_table_df is not None and not comparison_table_df.empty:
        comparison_display = comparison_table_df.copy()
        rename_map = {
            "latitude": "Primary latitude",
            "longitude": "Primary longitude",
            "comparison_latitude": "Comparison latitude",
            "comparison_longitude": "Comparison longitude",
        }
        comparison_display = comparison_display.rename(columns=rename_map)
        if inputs.label and inputs.label in comparison_display.columns:
            ordered_columns = [inputs.label] + [col for col in comparison_display.columns if col != inputs.label]
            comparison_display = comparison_display[ordered_columns]
        tables.append(
            dataframe_to_table(
                comparison_display.sort_values("Delta (km)", ascending=False).head(10),
                title="Coordinate Comparison Differences",
                description="Largest per-row distance differences between the primary and comparison coordinate columns (Haversine distance).",
                round_decimals=3,
            )
        )

    metrics = [
        metric("Rows analysed", len(frame)),
        metric("Reference latitude", round(reference_lat, 6)),
        metric("Reference longitude", round(reference_lon, 6)),
        metric("Mean distance (km)", round(mean_distance, 3) if not math.isnan(mean_distance) else "N/A"),
        metric("Median distance (km)", round(median_distance, 3) if not math.isnan(median_distance) else "N/A"),
        metric("90th percentile distance (km)", round(percentile_90, 3) if not math.isnan(percentile_90) else "N/A"),
    ]

    if within_radius_count is not None and within_radius_share is not None:
        metrics.append(metric("Within radius", within_radius_count))
        metrics.append(metric("Within radius (%)", round(within_radius_share, 2)))

    if comparison_metrics:
        metrics.extend(comparison_metrics)

    insights = [
        insight(
            "info",
            f"Computed proximity metrics using latitude '{inputs.latitude}' and longitude '{inputs.longitude}'.",
        ),
        insight("info", f"Reference point source: {reference_source}."),
    ]

    if optional_mode == "comparison":
        insights.append(
            insight(
                "info",
                "Comparison-only mode enabled. Radius/reference visuals are hidden to spotlight coordinate deltas.",
            )
        )

    if comparison_insights:
        insights.extend(comparison_insights)

    status = "success"

    if invalid_radius:
        insights.append(
            insight(
                "warning",
                "Ignored proximity radius because the supplied value was not a positive number.",
            )
        )

    if inconsistent_reference:
        insights.append(
            insight(
                "warning",
                "Reference latitude and longitude were not both provided. Falling back to dataset centroid.",
            )
        )

    if within_radius_count is not None:
        if within_radius_count == 0:
            status = "warning"
            insights.append(
                insight(
                    "warning",
                    f"No observations fall within the {radius_km:.2f} km radius around the reference point.",
                )
            )
        else:
            insights.append(
                insight(
                    "success",
                    f"{within_radius_count} observations ({within_radius_share:.1f}%) lie within {radius_km:.2f} km of the reference point.",
                )
            )
    elif optional_mode != "comparison":
        insights.append(
            insight(
                "info",
                "Configure an optional radius (km) to highlight coverage around the reference point.",
            )
        )

    if comparison_warning and status != "warning":
        status = "warning"

    if optional_mode == "comparison":
        summary = "Coordinate comparison metrics highlighting haversine deltas between the selected coordinate sets."
    else:
        summary = "Proximity metrics summarising distances to a reference point with optional radius coverage."

    details: Dict[str, Any] = {
        "columns": {"latitude": inputs.latitude, "longitude": inputs.longitude, "label": inputs.label},
        "reference": {
            "latitude": reference_lat,
            "longitude": reference_lon,
            "source": reference_source,
            "manual": manual_reference,
        },
        "radius_km": radius_km,
        "metadata": metadata,
        "optional_mode": optional_mode,
    }

    if comparison_details:
        details["comparison"] = comparison_details

    return AnalysisResult(
        analysis_id="geospatial_proximity_analysis",
        title="Geospatial Proximity Analysis",
        summary=summary,
        status=status,
        metrics=metrics,
        charts=charts,
        tables=tables,
        insights=insights,
        details=details,
    )


def coordinate_system_projection_check(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    inputs = _detect_geospatial_inputs(df, context, include_label=False)
    if not inputs:
        return AnalysisResult(
            analysis_id="coordinate_system_projection_check",
            title="Coordinate System & Projection Check",
            summary="Latitude and longitude columns are required for projection checks.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Map or select latitude/longitude columns to run the coordinate system validation.",
                )
            ],
        )

    frame = inputs.frame
    lat_series = frame["latitude"]
    lon_series = frame["longitude"]

    lat_min = float(lat_series.min())
    lat_max = float(lat_series.max())
    lon_min = float(lon_series.min())
    lon_max = float(lon_series.max())

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    issues: List[str] = []
    status = "success"

    if lat_series.abs().max() > 90.0 + 0.0001:
        issues.append(
            "Latitude values exceed ±90°. Data may be stored in a projected coordinate system (e.g., Web Mercator)."
        )
    if lon_series.abs().max() > 180.0 + 0.0001:
        issues.append(
            "Longitude values exceed ±180°. Confirm that longitudes are expressed in decimal degrees."
        )
    if lat_range == 0 or lon_range == 0:
        issues.append("Latitude and longitude show zero spread. Confirm that coordinates vary across records.")
    if issues:
        status = "warning"

    metrics = [
        metric("Rows analysed", len(frame)),
        metric("Latitude range", f"{lat_min:.4f} → {lat_max:.4f}"),
        metric("Longitude range", f"{lon_min:.4f} → {lon_max:.4f}"),
        metric("Lat span (°)", round(lat_range, 4)),
        metric("Lon span (°)", round(lon_range, 4)),
    ]

    coverage_table = dataframe_to_table(
        pd.DataFrame(
            {
                "Quantile": ["5%", "25%", "50%", "75%", "95%"],
                "Latitude": np.percentile(lat_series, [5, 25, 50, 75, 95]),
                "Longitude": np.percentile(lon_series, [5, 25, 50, 75, 95]),
            }
        ),
        title="Coordinate Quantiles",
        description="Distribution summary across latitude and longitude to spot anomalies.",
        round_decimals=4,
    )

    insights = [
        insight(
            "info",
            f"Detected latitude column '{inputs.latitude}' and longitude column '{inputs.longitude}'.",
        )
    ]
    insights.extend(insight("warning", message) for message in issues)

    if not issues:
        insights.append(
            insight(
                "success",
                "Coordinate ranges fall within expected decimal-degree bounds (EPSG:4326).",
            )
        )

    return AnalysisResult(
        analysis_id="coordinate_system_projection_check",
        title="Coordinate System & Projection Check",
        summary="Validates latitude/longitude ranges and highlights potential projection issues.",
        status=status,
        metrics=metrics,
        tables=[coverage_table],
        insights=insights,
    )


def spatial_distribution_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    inputs = _detect_geospatial_inputs(df, context, include_label=False)
    if not inputs:
        return AnalysisResult(
            analysis_id="spatial_distribution_analysis",
            title="Spatial Distribution Analysis",
            summary="Latitude and longitude columns are required to visualize spatial distribution.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Select or map latitude and longitude columns to explore spatial distribution.",
                )
            ],
        )

    sample = _prepare_sample(inputs, sample_size=500, random_state=context.random_state)
    total_points = len(inputs.frame)
    sample_points = len(sample)

    charts: List[AnalysisChart] = []

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sample["longitude"], sample["latitude"], s=12, alpha=0.65, color="#2D9CDB")
    ax.set_title("Geographic Scatter (sampled)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.2)
    charts.append(
        AnalysisChart(
            title="Spatial Scatter",
            image=fig_to_base64(fig),
            description="Sampled latitude/longitude points to visualise geographic coverage.",
        )
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(sample["longitude"], sample["latitude"], gridsize=30, cmap="viridis", mincnt=1)
    ax.set_title("Spatial Density (hexbin)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Point count")
    charts.append(
        AnalysisChart(
            title="Spatial Density",
            image=fig_to_base64(fig),
            description="Hexbin heatmap indicating concentration of observations.",
        )
    )
    plt.close(fig)

    metrics = [
        metric("Rows analysed", total_points),
        metric("Visualised sample", sample_points),
        metric("Latitude mean", round(sample["latitude"].mean(), 4)),
        metric("Longitude mean", round(sample["longitude"].mean(), 4)),
    ]

    if sample_points < total_points:
        metrics.append(
            metric(
                "Sampling note",
                f"Visualisations use {sample_points} of {total_points} points to remain responsive.",
            )
        )

    summary = "Geospatial scatterplots highlighting observed coverage and density." if charts else "Insufficient data to build geospatial charts."

    return AnalysisResult(
        analysis_id="spatial_distribution_analysis",
        title="Spatial Distribution Analysis",
        summary=summary,
        charts=charts,
        metrics=metrics,
        insights=[
            insight(
                "info",
                f"Using latitude '{inputs.latitude}' and longitude '{inputs.longitude}'."
            )
        ],
    )


def spatial_relationships_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    inputs = _detect_geospatial_inputs(df, context)
    if not inputs:
        return AnalysisResult(
            analysis_id="spatial_relationships_analysis",
            title="Spatial Relationships Analysis",
            summary="Latitude and longitude columns are required to analyse spatial proximity.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Select or map latitude and longitude columns to compute spatial relationships.",
                )
            ],
        )

    sample = _prepare_sample(inputs, sample_size=250, random_state=context.random_state)
    if len(sample) < 2:
        return AnalysisResult(
            analysis_id="spatial_relationships_analysis",
            title="Spatial Relationships Analysis",
            summary="Need at least two coordinate pairs to compute spatial relationships.",
            status="warning",
            insights=[insight("warning", "Add more observations with latitude/longitude values.")],
        )

    coords = sample[["latitude", "longitude"]].to_numpy()
    distances = _haversine(coords[:, 0], coords[:, 1])
    np.fill_diagonal(distances, np.nan)

    nearest_distances = np.nanmin(distances, axis=1)
    mean_distance = float(np.nanmean(nearest_distances)) if np.isfinite(nearest_distances).any() else float("nan")
    median_distance = float(np.nanmedian(nearest_distances)) if np.isfinite(nearest_distances).any() else float("nan")

    metrics = [
        metric("Rows analysed", len(inputs.frame)),
        metric("Sample used for distances", len(sample)),
        metric("Mean nearest neighbour (km)", round(mean_distance, 3) if not math.isnan(mean_distance) else "N/A"),
        metric(
            "Median nearest neighbour (km)",
            round(median_distance, 3) if not math.isnan(median_distance) else "N/A",
        ),
    ]

    # Build table of the tightest spatial clusters
    flat_distances = []
    labels = inputs.label
    for idx, distance in enumerate(nearest_distances):
        if math.isnan(distance):
            continue
        row = {
            "Latitude": sample.iloc[idx]["latitude"],
            "Longitude": sample.iloc[idx]["longitude"],
            "Nearest Distance (km)": round(float(distance), 3),
        }
        if labels and labels in sample.columns:
            row["Label"] = sample.iloc[idx][labels]
        flat_distances.append(row)

    nearest_table = dataframe_to_table(
        pd.DataFrame(sorted(flat_distances, key=lambda item: item["Nearest Distance (km)"])[:10]),
        title="Nearest Neighbour Summary",
        description="Shortest distances between nearby observations (top 10).",
        round_decimals=3,
    ) if flat_distances else None

    fig, ax = plt.subplots(figsize=(7, 4))
    valid_distances = nearest_distances[np.isfinite(nearest_distances)]
    if valid_distances.size > 0:
        ax.hist(valid_distances, bins=20, color="#6C5CE7", alpha=0.85)
        ax.set_xlabel("Nearest neighbour distance (km)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of nearest neighbour distances")
        chart = AnalysisChart(
            title="Nearest Neighbour Distribution",
            image=fig_to_base64(fig),
            description="Histogram of nearest neighbour distances across sampled points.",
        )
        charts = [chart]
    else:
        ax.text(0.5, 0.5, "Not enough unique points for distance histogram", ha="center", va="center")
        chart = AnalysisChart(
            title="Nearest Neighbour Distribution",
            image=fig_to_base64(fig),
            description="Not enough unique points to compute distances.",
        )
        charts = [chart]
    plt.close(fig)

    insights = [
        insight(
            "info",
            f"Computed spatial relationships using latitude '{inputs.latitude}' and longitude '{inputs.longitude}'.",
        )
    ]

    if labels and labels in sample.columns:
        insights.append(
            insight(
                "info",
                f"Nearest-neighbour summary includes label column '{labels}'.",
            )
        )

    tables = [nearest_table] if nearest_table is not None else []

    return AnalysisResult(
        analysis_id="spatial_relationships_analysis",
        title="Spatial Relationships Analysis",
        summary="Nearest-neighbour measurements highlight spatial proximity patterns.",
        metrics=metrics,
        charts=charts,
        tables=tables,
        insights=insights,
    )


__all__ = [
    "coordinate_system_projection_check",
    "spatial_distribution_analysis",
    "spatial_relationships_analysis",
    "spatial_data_quality_analysis",
    "geospatial_proximity_analysis",
]
