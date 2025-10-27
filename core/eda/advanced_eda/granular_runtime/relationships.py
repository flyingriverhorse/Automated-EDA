from __future__ import annotations

"""Relationship and dimensionality reduction analyses."""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .common import (
    AnalysisChart,
    AnalysisContext,
    AnalysisMetric,
    AnalysisResult,
    AnalysisTable,
    dataframe_to_table,
    fig_to_base64,
    insight,
    numeric_columns,
)

try:  # pragma: no cover - optional dependency
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    PCA = None  # type: ignore
    StandardScaler = None  # type: ignore
    KMeans = None  # type: ignore
    silhouette_score = None  # type: ignore
    NearestNeighbors = None  # type: ignore


def _compute_hopkins_statistic(data: np.ndarray, random_state: int = 42) -> float:
    """Estimate the Hopkins statistic to gauge clustering tendency."""

    n_samples = data.shape[0]
    if n_samples < 5:
        raise ValueError("Hopkins statistic requires at least 5 samples.")

    rng = np.random.default_rng(random_state)
    sample_size = int(max(0.1 * n_samples, min(25, n_samples - 1)))
    sample_size = max(1, min(sample_size, n_samples - 1))

    sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
    sample_points = data[sample_indices]

    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    uniform_points = rng.uniform(low=data_min, high=data_max, size=(sample_size, data.shape[1]))

    if NearestNeighbors is None:
        raise RuntimeError("scikit-learn is required to compute the Hopkins statistic")

    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(data)
    u_dist = nbrs.kneighbors(uniform_points, return_distance=True)[0]
    w_dist = nbrs.kneighbors(sample_points, return_distance=True)[0]

    hopkins = float(u_dist.sum() / (u_dist.sum() + w_dist.sum()))
    return hopkins


def _prepare_numeric_matrix(
    df: pd.DataFrame,
    numeric_cols: List[str],
    minimum_samples: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Sanitize numeric frame for clustering, with simple imputation fallback."""

    info: Dict[str, Any] = {
        "initial_columns": numeric_cols,
        "used_imputation": False,
        "dropped_all_null_columns": [],
        "initial_rows": 0,
        "rows_after_dropna": 0,
        "final_rows": 0,
        "retained_columns": [],
    }

    working = df[numeric_cols].copy()
    working = working.replace([np.inf, -np.inf], np.nan)
    info["initial_rows"] = int(len(working))

    null_only_columns = [col for col in numeric_cols if working[col].isna().all()]
    if null_only_columns:
        working = working.drop(columns=null_only_columns)
        info["dropped_all_null_columns"] = null_only_columns

    complete_cases = working.dropna()
    info["rows_after_dropna"] = int(len(complete_cases))

    if len(complete_cases) >= minimum_samples and complete_cases.shape[1] >= 1:
        info["final_rows"] = int(len(complete_cases))
        info["retained_columns"] = complete_cases.columns.tolist()
        info["sufficient"] = True
        return complete_cases, info

    # Attempt lightweight imputation (median/mean fallback)
    imputed = working.copy()
    for col in imputed.columns.tolist():
        series = imputed[col]
        if series.notna().sum() == 0:
            # Already accounted for null-only columns, but guard anyway
            imputed = imputed.drop(columns=[col])
            continue

        median = series.median()
        if pd.isna(median):
            median = series.mean()
        if pd.isna(median):
            median = 0.0
        imputed[col] = series.fillna(median)

    imputed = imputed.dropna()

    info["used_imputation"] = True if len(imputed) > len(complete_cases) else False
    info["final_rows"] = int(len(imputed))
    info["retained_columns"] = imputed.columns.tolist()
    info["sufficient"] = len(imputed) >= minimum_samples and imputed.shape[1] >= 1

    return imputed, info


def multicollinearity_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="multicollinearity_analysis",
            title="Multicollinearity Analysis",
            summary="At least two numeric columns are required to assess multicollinearity.",
            status="warning",
            insights=[insight("warning", "Select multiple numeric columns to analyze multicollinearity.")],
        )

    try:
        corr = df[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr_pairs = (
            upper.stack()
            .reset_index()
            .rename(columns={"level_0": "Feature A", "level_1": "Feature B", 0: "Correlation"})
        )
        high_corr_pairs = high_corr_pairs[high_corr_pairs["Correlation"] > 0.75].sort_values(
            "Correlation", ascending=False
        )

        vif_data: List[Dict[str, Any]] = []
        numeric_df = df[numeric_cols].dropna()
        if not numeric_df.empty and len(numeric_cols) >= 2:
            try:
                for col in numeric_cols:
                    others = [c for c in numeric_cols if c != col]
                    if not others:
                        continue
                    max_corr = corr.loc[col, others].max()
                    vif = 1 / (1 - max_corr**2) if max_corr < 0.99 else float("inf")
                    vif_data.append({"Feature": col, "VIF": vif})
            except Exception:
                vif_data = []

        tables = []
        corr_table = dataframe_to_table(
            high_corr_pairs,
            title="High Correlation Pairs (>0.75)",
            description="Feature pairs with strong linear dependency.",
            round_decimals=3,
        )
        tables.append(corr_table)

        if vif_data:
            vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
            tables.append(
                dataframe_to_table(
                    vif_df,
                    title="Variance Inflation Factors (VIF)",
                    description="VIF > 5 indicates potential multicollinearity.",
                    round_decimals=2,
                )
            )

        insights: List = []
        if high_corr_pairs.empty:
            insights.append(insight("success", "No high correlations detected above 0.75 threshold."))
        else:
            insights.append(
                insight(
                    "warning",
                    f"Found {len(high_corr_pairs)} feature pairs with high correlation (>0.75). Consider removing or combining correlated features.",
                )
            )

        if vif_data:
            high_vif = [row for row in vif_data if row["VIF"] > 5]
            if high_vif:
                insights.append(
                    insight(
                        "warning",
                        f"Found {len(high_vif)} features with VIF > 5, indicating potential multicollinearity.",
                    )
                )
            else:
                insights.append(insight("success", "All VIF values are below 5, indicating low multicollinearity."))
        else:
            insights.append(insight("info", "VIF calculation not available for this dataset configuration."))

        return AnalysisResult(
            analysis_id="multicollinearity_analysis",
            title="Multicollinearity Analysis",
            summary="Identifies strongly correlated numeric feature pairs and calculates VIF scores.",
            tables=tables,
            insights=insights,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return AnalysisResult(
            analysis_id="multicollinearity_analysis",
            title="Multicollinearity Analysis",
            summary="Failed to complete multicollinearity analysis.",
            status="error",
            insights=[insight("danger", f"Error in multicollinearity analysis: {str(exc)}")],
        )


def _run_pca(df: pd.DataFrame, numeric_cols: List[str], n_components: Optional[int] = None):
    if PCA is None or StandardScaler is None:
        raise RuntimeError("scikit-learn is required for PCA analyses")

    data = df[numeric_cols].dropna()
    if data.empty:
        raise ValueError("No complete rows available after dropping NaNs for PCA.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)
    return pca, components


def pca_dimensionality_reduction(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="pca_dimensionality_reduction",
            title="PCA Analysis",
            summary="At least two numeric columns are required for PCA.",
            status="warning",
            insights=[insight("warning", "Select multiple numeric columns to perform PCA.")],
        )

    try:
        pca, components = _run_pca(df, numeric_cols)
    except RuntimeError:
        return AnalysisResult(
            analysis_id="pca_dimensionality_reduction",
            title="PCA Analysis",
            summary="scikit-learn dependency not available for PCA computation.",
            status="error",
            insights=[insight("danger", "Install scikit-learn to enable PCA analyses.")],
        )
    except ValueError as exc:
        return AnalysisResult(
            analysis_id="pca_dimensionality_reduction",
            title="PCA Analysis",
            summary=str(exc),
            status="warning",
            insights=[insight("warning", "Impute or drop NaNs to run PCA.")],
        )

    explained = pd.DataFrame(
        {
            "Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "Explained Variance %": pca.explained_variance_ratio_ * 100,
            "Cumulative %": np.cumsum(pca.explained_variance_ratio_ * 100),
        }
    )

    table = dataframe_to_table(
        explained,
        title="PCA Explained Variance",
        description="Variance explained by each principal component.",
        round_decimals=2,
    )

    message = (
        f"Top 2 components explain {explained['Cumulative %'].iloc[1]:.2f}% of variance"
        if len(explained) >= 2
        else "PCA computed successfully."
    )

    return AnalysisResult(
        analysis_id="pca_dimensionality_reduction",
        title="PCA Analysis",
        summary="Principal component analysis capturing variance explained by components.",
        tables=[table],
        insights=[insight("info", message)],
        details={
            "components": components[:, : min(3, components.shape[1])].tolist(),
            "feature_names": numeric_cols,
            "explained_variance": explained.to_dict(orient="records"),
        },
    )


def pca_scree_plot(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    base = pca_dimensionality_reduction(df, context)
    if base.status != "success":
        return base

    variance_rows = base.details.get("explained_variance", [])
    variance_values = [row["Explained Variance %"] for row in variance_rows]

    base.tables = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        range(1, len(variance_values) + 1),
        variance_values,
        marker="o",
        color="#ED8936",
        label="Explained Variance",
    )
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance %")
    ax.set_title("PCA Scree Plot")

    if variance_values:
        kaiser_threshold = 100 / len(variance_values)
        ax.axhline(
            kaiser_threshold,
            color="#4A5568",
            linestyle="--",
            label="Kaiser (λ = 1)",
        )

        components_above = [
            row["Component"]
            for row in variance_rows
            if row["Explained Variance %"] >= kaiser_threshold
        ]

        if components_above:
            base.insights.append(
                insight(
                    "info",
                    "Kaiser criterion satisfied by " + ", ".join(components_above),
                )
            )
        else:
            base.insights.append(
                insight("info", "No components exceed the Kaiser criterion (λ = 1)."),
            )

    ax.legend()

    chart = AnalysisChart(
        title="PCA Scree Plot",
        image=fig_to_base64(fig),
        description="Explained variance percentage by component.",
    )

    base.analysis_id = "pca_scree_plot"
    base.title = "PCA Scree Plot"
    base.summary = "Explained variance per principal component."
    base.charts = [chart]
    return base


def pca_cumulative_variance(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    base = pca_dimensionality_reduction(df, context)
    if base.status != "success":
        return base

    variance_rows = base.details.get("explained_variance", [])
    cumulative_values = [row["Cumulative %"] for row in variance_rows]

    base.tables = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(cumulative_values) + 1), cumulative_values, marker="o", color="#38B2AC")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance %")
    ax.axhline(90, color="red", linestyle="--", label="90%")
    ax.legend()
    ax.set_title("PCA Cumulative Variance")

    chart = AnalysisChart(
        title="PCA Cumulative Variance",
        image=fig_to_base64(fig),
        description="Cumulative explained variance by number of components.",
    )

    base.analysis_id = "pca_cumulative_variance"
    base.title = "PCA Cumulative Variance"
    base.summary = "Cumulative explained variance across principal components."
    base.charts = [chart]
    return base


def pca_visualization(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="pca_visualization",
            title="PCA Visualization",
            summary="At least two numeric columns are required for PCA.",
            status="warning",
            insights=[insight("warning", "Select multiple numeric columns to perform PCA visualization.")],
        )

    try:
        pca, components = _run_pca(df, numeric_cols, n_components=2)
    except RuntimeError:
        return AnalysisResult(
            analysis_id="pca_visualization",
            title="PCA Visualization",
            summary="scikit-learn dependency not available for PCA computation.",
            status="error",
            insights=[insight("danger", "Install scikit-learn to enable PCA analyses.")],
        )
    except ValueError as exc:
        return AnalysisResult(
            analysis_id="pca_visualization",
            title="PCA Visualization",
            summary=str(exc),
            status="warning",
            insights=[insight("warning", "Impute or drop NaNs to run PCA.")],
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(components[:, 0], components[:, 1], alpha=0.7, color="#4C51BF")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection (PC1 vs PC2)")

    chart = AnalysisChart(
        title="PCA Projection",
        image=fig_to_base64(fig),
        description="Scatter of first two principal components.",
    )

    return AnalysisResult(
        analysis_id="pca_visualization",
        title="PCA Visualization",
        summary="Projection of data onto the first two principal components.",
        charts=[chart],
        insights=[insight("info", "Clusters in the PCA scatter may indicate latent structure." )],
    )


def pca_biplot(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="pca_biplot",
            title="PCA Biplot",
            summary="At least two numeric columns are required for PCA.",
            status="warning",
            insights=[insight("warning", "Select multiple numeric columns to perform PCA biplot.")],
        )

    try:
        pca, components = _run_pca(df, numeric_cols, n_components=2)
    except RuntimeError:
        return AnalysisResult(
            analysis_id="pca_biplot",
            title="PCA Biplot",
            summary="scikit-learn dependency not available for PCA computation.",
            status="error",
            insights=[insight("danger", "Install scikit-learn to enable PCA analyses.")],
        )
    except ValueError as exc:
        return AnalysisResult(
            analysis_id="pca_biplot",
            title="PCA Biplot",
            summary=str(exc),
            status="warning",
            insights=[insight("warning", "Impute or drop NaNs to run PCA.")],
        )

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(components[:, 0], components[:, 1], alpha=0.6, color="#2C5282")
    for i, feature in enumerate(numeric_cols):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], color="red", alpha=0.5)
        ax.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, feature, color="red")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Biplot")

    chart = AnalysisChart(
        title="PCA Biplot",
        image=fig_to_base64(fig),
        description="Biplot showing PCA scores and loadings.",
    )

    return AnalysisResult(
        analysis_id="pca_biplot",
        title="PCA Biplot",
        summary="Combines PCA scores and feature loadings for interpretation.",
        charts=[chart],
        insights=[insight("info", "Arrows indicate contribution of original features to components.")],
    )


def pca_heatmap(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="pca_heatmap",
            title="PCA Loadings Heatmap",
            summary="At least two numeric columns are required for PCA.",
            status="warning",
            insights=[insight("warning", "Select multiple numeric columns to compute PCA loadings.")],
        )

    try:
        pca, _ = _run_pca(df, numeric_cols)
    except RuntimeError:
        return AnalysisResult(
            analysis_id="pca_heatmap",
            title="PCA Loadings Heatmap",
            summary="scikit-learn dependency not available for PCA computation.",
            status="error",
            insights=[insight("danger", "Install scikit-learn to enable PCA analyses.")],
        )
    except ValueError as exc:
        return AnalysisResult(
            analysis_id="pca_heatmap",
            title="PCA Loadings Heatmap",
            summary=str(exc),
            status="warning",
            insights=[insight("warning", "Impute or drop NaNs to run PCA.")],
        )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=numeric_cols,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        loadings.iloc[:, : min(5, loadings.shape[1])],
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("PCA Loadings (First 5 Components)")

    chart = AnalysisChart(
        title="PCA Loadings Heatmap",
        image=fig_to_base64(fig),
        description="Contribution of variables to principal components.",
    )

    return AnalysisResult(
        analysis_id="pca_heatmap",
        title="PCA Loadings Heatmap",
        summary="Heatmap of PCA loadings for top components.",
        charts=[chart],
        insights=[insight("info", "High absolute loadings indicate strong influence on components." )],
    )


def cluster_tendency_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="cluster_tendency_analysis",
            title="Cluster Tendency Analysis",
            summary="At least two numeric columns are required to evaluate clustering tendency.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Select multiple numeric columns to assess whether the dataset forms natural clusters.",
                )
            ],
        )

    if StandardScaler is None or NearestNeighbors is None:
        return AnalysisResult(
            analysis_id="cluster_tendency_analysis",
            title="Cluster Tendency Analysis",
            summary="scikit-learn dependency not available for clustering metrics.",
            status="error",
            insights=[insight("danger", "Install scikit-learn to enable clustering analyses.")],
        )

    prepared_df, prep_info = _prepare_numeric_matrix(df, numeric_cols, minimum_samples=5)

    details: Dict[str, Any] = {
        "features": prep_info.get("retained_columns", numeric_cols),
        "sample_size": int(prep_info.get("final_rows", 0)),
        "preprocessing": prep_info,
    }

    insights: List = []

    if not prep_info.get("sufficient", False):
        message = (
            "At least 5 usable rows are required after handling missing values to evaluate clustering tendency."
        )
        if prep_info.get("dropped_all_null_columns"):
            dropped = ", ".join(prep_info["dropped_all_null_columns"])
            insights.append(
                insight(
                    "info",
                    f"Columns removed because they contained only null values: {dropped}.",
                )
            )
        return AnalysisResult(
            analysis_id="cluster_tendency_analysis",
            title="Cluster Tendency Analysis",
            summary=message,
            status="warning",
            insights=insights
            + [
                insight(
                    "warning",
                    "Provide additional data or address missing values to compute clustering metrics.",
                )
            ],
            details=details,
        )

    if prep_info.get("used_imputation"):
        insights.append(
            insight(
                "info",
                "Missing values were imputed with column medians before computing clustering metrics.",
            )
        )

    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(prepared_df)
    except Exception as exc:
        return AnalysisResult(
            analysis_id="cluster_tendency_analysis",
            title="Cluster Tendency Analysis",
            summary="Failed to scale numeric data for clustering evaluation.",
            status="error",
            insights=[insight("danger", f"Scaling error: {str(exc)}")],
            details=details,
        )

    metrics: List[AnalysisMetric] = []
    details["scaler_mean"] = scaler.mean_.tolist() if hasattr(scaler, "mean_") else None
    details["scaler_scale"] = scaler.scale_.tolist() if hasattr(scaler, "scale_") else None

    hopkins_value: Optional[float] = None
    try:
        hopkins_value = _compute_hopkins_statistic(scaled)
        metrics.append(
            AnalysisMetric(
                label="Hopkins Statistic",
                value=round(hopkins_value, 3),
                description="Values > 0.5 suggest meaningful cluster structure.",
            )
        )
        details["hopkins_statistic"] = hopkins_value
        if hopkins_value >= 0.6:
            insights.append(
                insight("success", f"Hopkins statistic of {hopkins_value:.2f} indicates strong clustering tendency."),
            )
        elif hopkins_value >= 0.45:
            insights.append(
                insight("info", f"Hopkins statistic of {hopkins_value:.2f} suggests moderate clustering structure."),
            )
        else:
            insights.append(
                insight("warning", f"Hopkins statistic of {hopkins_value:.2f} indicates weak clustering tendency."),
            )
    except Exception as exc:
        insights.append(insight("warning", f"Unable to compute Hopkins statistic: {str(exc)}"))

    try:
        neighbor_k = max(2, min(5, scaled.shape[0]))
        nbrs = NearestNeighbors(n_neighbors=neighbor_k)
        nbrs.fit(scaled)
        distances, _ = nbrs.kneighbors(scaled)
        # Skip the zero distance to itself
        avg_distance = float(np.mean(distances[:, 1]))
        metrics.append(
            AnalysisMetric(
                label="Avg. Nearest Neighbor Distance",
                value=round(avg_distance, 3),
                description="Lower values (after scaling) indicate tighter clusters.",
            )
        )
        details["average_nearest_neighbor_distance"] = avg_distance
    except Exception as exc:
        insights.append(insight("info", f"Nearest-neighbor distance unavailable: {str(exc)}"))

    silhouette_value: Optional[float] = None
    if KMeans is not None and silhouette_score is not None:
        try:
            if scaled.shape[0] > 2:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = kmeans.fit_predict(scaled)
                if np.unique(labels).size > 1:
                    silhouette_value = float(silhouette_score(scaled, labels))
                    metrics.append(
                        AnalysisMetric(
                            label="Silhouette (k=2)",
                            value=round(silhouette_value, 3),
                            description="Higher values (0-1) indicate better separation for a 2-cluster split.",
                        )
                    )
                    details["silhouette_two_clusters"] = silhouette_value
                    if silhouette_value >= 0.5:
                        insights.append(
                            insight(
                                "success",
                                f"Two-cluster silhouette of {silhouette_value:.2f} indicates well-separated groups.",
                            )
                        )
                    elif silhouette_value >= 0.3:
                        insights.append(
                            insight(
                                "info",
                                f"Two-cluster silhouette of {silhouette_value:.2f} shows moderate separation.",
                            )
                        )
                    else:
                        insights.append(
                            insight(
                                "warning",
                                f"Two-cluster silhouette of {silhouette_value:.2f} indicates overlapping clusters.",
                            )
                        )
                else:
                    insights.append(
                        insight(
                            "info",
                            "Two-cluster sample collapsed into a single group, suggesting minimal separation.",
                        )
                    )
        except Exception as exc:
            insights.append(insight("info", f"Silhouette sample unavailable: {str(exc)}"))

    summary = "Cluster tendency metrics evaluate whether the dataset contains discoverable segments."

    return AnalysisResult(
        analysis_id="cluster_tendency_analysis",
        title="Cluster Tendency Analysis",
        summary=summary,
        metrics=metrics,
        insights=insights,
        details=details,
    )


def cluster_segmentation_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="cluster_segmentation_analysis",
            title="Cluster Segmentation",
            summary="At least two numeric columns are required for clustering.",
            status="warning",
            insights=[insight("warning", "Select multiple numeric columns to perform segmentation.")],
        )

    if StandardScaler is None or KMeans is None:
        return AnalysisResult(
            analysis_id="cluster_segmentation_analysis",
            title="Cluster Segmentation",
            summary="scikit-learn dependency not available for clustering.",
            status="error",
            insights=[insight("danger", "Install scikit-learn to enable clustering analyses.")],
        )

    prepared_df, prep_info = _prepare_numeric_matrix(df, numeric_cols, minimum_samples=6)

    insights: List = []
    if not prep_info.get("sufficient", False):
        warning_summary = (
            "At least 6 usable rows are required after handling missing values to perform clustering."
        )
        if prep_info.get("dropped_all_null_columns"):
            dropped = ", ".join(prep_info["dropped_all_null_columns"])
            insights.append(
                insight(
                    "info",
                    f"Columns removed because they contained only null values: {dropped}.",
                )
            )
        return AnalysisResult(
            analysis_id="cluster_segmentation_analysis",
            title="Cluster Segmentation",
            summary=warning_summary,
            status="warning",
            insights=insights
            + [
                insight(
                    "warning",
                    "Provide additional data or address missing values to perform clustering.",
                )
            ],
            details={
                "preprocessing": prep_info,
            },
        )

    if prep_info.get("used_imputation"):
        insights.append(
            insight(
                "info",
                "Missing values were imputed with column medians before clustering.",
            )
        )

    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(prepared_df)
    except Exception as exc:
        return AnalysisResult(
            analysis_id="cluster_segmentation_analysis",
            title="Cluster Segmentation",
            summary="Failed to scale numeric data for clustering.",
            status="error",
            insights=[insight("danger", f"Scaling error: {str(exc)}")],
            details={
                "preprocessing": prep_info,
            },
        )

    max_k = min(6, scaled.shape[0] - 1)
    if max_k < 2:
        return AnalysisResult(
            analysis_id="cluster_segmentation_analysis",
            title="Cluster Segmentation",
            summary="Not enough samples to form multiple clusters.",
            status="warning",
            insights=[insight("warning", "Collect more data points to segment into clusters.")],
        )

    candidates: List[Dict[str, Any]] = []
    for k in range(2, max_k + 1):
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(scaled)
            if np.unique(labels).size < 2:
                continue
            inertia = float(model.inertia_)
            silhouette_val: Optional[float] = None
            if silhouette_score is not None and scaled.shape[0] > k:
                silhouette_val = float(silhouette_score(scaled, labels))
            candidates.append(
                {
                    "k": k,
                    "labels": labels,
                    "model": model,
                    "silhouette": silhouette_val,
                    "inertia": inertia,
                }
            )
        except Exception:
            continue

    if not candidates:
        return AnalysisResult(
            analysis_id="cluster_segmentation_analysis",
            title="Cluster Segmentation",
            summary="Unable to fit clustering models with the available data.",
            status="warning",
            insights=[insight("warning", "Clustering failed. Consider scaling features or adding more data.")],
        )

    def _rank(candidate: Dict[str, Any]) -> tuple:
        silhouette_val = candidate.get("silhouette")
        if silhouette_val is not None:
            return (2, silhouette_val)
        return (1, -candidate["inertia"])

    best = max(candidates, key=_rank)
    best_k = int(best["k"])
    best_labels = best["labels"]
    best_silhouette = best.get("silhouette")
    best_inertia = best.get("inertia", 0.0)

    centers_scaled = best["model"].cluster_centers_
    scale_values = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    centers_original = centers_scaled * scale_values + scaler.mean_

    cluster_counts = pd.Series(best_labels).value_counts().sort_index()
    feature_list = prepared_df.columns.tolist()
    cluster_records: List[Dict[str, Any]] = []
    for cluster_idx, count in cluster_counts.items():
        centroid = centers_original[int(cluster_idx)]
        row: Dict[str, Any] = {
            "Cluster": f"Cluster {int(cluster_idx) + 1}",
            "Size": int(count),
            "Share %": float((count / len(prepared_df)) * 100),
        }
        for feature, value in zip(feature_list, centroid):
            row[feature] = float(value)
        cluster_records.append(row)

    centroid_table = dataframe_to_table(
        pd.DataFrame(cluster_records),
        title="Cluster Centroids & Sizes",
        description="Cluster sizes, shares, and centroid values in original feature units.",
        round_decimals=2,
    )

    evaluation_table = dataframe_to_table(
        pd.DataFrame(
            [
                {
                    "Clusters": cand["k"],
                    "Silhouette": cand["silhouette"],
                    "Inertia": cand["inertia"],
                }
                for cand in candidates
            ]
        ),
        title="Cluster Evaluation Summary",
        description="Comparison of candidate cluster counts using silhouette score and inertia.",
        round_decimals=3,
    )

    overall_mean = prepared_df.mean()
    overall_std = prepared_df.std(ddof=0).replace(0, np.nan)

    discriminator_records: List[Dict[str, Any]] = []
    per_cluster_insights: Dict[int, Dict[str, Any]] = {}
    for cluster_idx in cluster_counts.index:
        cluster_mask = best_labels == cluster_idx
        cluster_df = prepared_df.iloc[cluster_mask]
        cluster_mean = cluster_df.mean()

        feature_deltas: List[Tuple[str, float, float, float, Optional[float]]] = []
        for feature in feature_list:
            cluster_value = float(cluster_mean.get(feature, np.nan))
            overall_value = float(overall_mean.get(feature, np.nan))
            if not np.isfinite(cluster_value) or not np.isfinite(overall_value):
                continue
            delta = cluster_value - overall_value
            std = overall_std.get(feature)
            z_score = float(delta / std) if std is not None and np.isfinite(std) else None
            feature_deltas.append((feature, cluster_value, overall_value, delta, z_score))

        if not feature_deltas:
            continue

        feature_deltas.sort(key=lambda item: abs(item[4]) if item[4] is not None else abs(item[3]), reverse=True)
        top_features = feature_deltas[: min(3, len(feature_deltas))]

        if top_features:
            top_feature_name, _, _, top_delta, top_z = top_features[0]
            per_cluster_insights[int(cluster_idx)] = {
                "feature": top_feature_name,
                "z_score": float(top_z) if top_z is not None else None,
                "delta": float(top_delta),
            }

        for feature, cluster_value, overall_value, delta, z_score in top_features:
            relative_delta = None
            if overall_value != 0:
                relative_delta = float((delta / overall_value) * 100.0)
            discriminator_records.append(
                {
                    "Cluster": f"Cluster {int(cluster_idx) + 1}",
                    "Feature": feature,
                    "Cluster Mean": cluster_value,
                    "Overall Mean": overall_value,
                    "Delta": delta,
                    "Relative Δ %": relative_delta,
                    "Std Z-Score": z_score,
                }
            )

    top_discriminator_table: Optional[AnalysisTable] = None
    if discriminator_records:
        top_discriminator_table = dataframe_to_table(
            pd.DataFrame(discriminator_records),
            title="Top Feature Discriminators",
            description="Leading features that differentiate each cluster compared to the overall dataset mean.",
            round_decimals=2,
        )

    centers_df = pd.DataFrame(centers_original, columns=feature_list)
    feature_spread = centers_df.max(axis=0) - centers_df.min(axis=0)
    feature_variability = prepared_df.var(ddof=0).replace(0, np.nan)
    discrimination_scores = []
    for feature in feature_list:
        spread = float(feature_spread.get(feature, 0.0))
        variability = float(feature_variability.get(feature, np.nan))
        score = None
        if variability and np.isfinite(variability):
            score = spread / variability if variability != 0 else None
        discrimination_scores.append(
            {
                "Feature": feature,
                "Cluster Mean Range": spread,
                "Variance": variability,
                "Discrimination Score": score,
            }
        )

    discrimination_scores.sort(
        key=lambda item: item["Discrimination Score"] if item["Discrimination Score"] is not None else item["Cluster Mean Range"],
        reverse=True,
    )

    discrimination_table: Optional[AnalysisTable] = None
    top_global_features: List[str] = []
    if discrimination_scores:
        top_global_features = [row["Feature"] for row in discrimination_scores[: min(5, len(discrimination_scores))]]
        discrimination_table = dataframe_to_table(
            pd.DataFrame(discrimination_scores[: min(10, len(discrimination_scores))]),
            title="Feature Discrimination Summary",
            description="Features ranked by how much their cluster means diverge relative to overall variance.",
            round_decimals=2,
        )

    metrics = [
        AnalysisMetric(
            label="Optimal Clusters",
            value=best_k,
            description="Selected using silhouette score when available, otherwise inertia.",
        ),
        AnalysisMetric(
            label="Inertia",
            value=round(best_inertia, 2),
            description="Lower inertia indicates tighter clusters.",
        ),
    ]
    if best_silhouette is not None:
        metrics.append(
            AnalysisMetric(
                label="Best Silhouette Score",
                value=round(best_silhouette, 3),
                description="Closer to 1 indicates better-defined clusters.",
            )
        )

    details = {
        "preprocessing": prep_info,
    }

    details["scaler_mean"] = scaler.mean_.tolist() if hasattr(scaler, "mean_") else None
    details["scaler_scale"] = scaler.scale_.tolist() if hasattr(scaler, "scale_") else None
    details["sample_size"] = int(prepared_df.shape[0])
    details["features"] = feature_list

    insights = insights or []
    for cluster_idx, info_dict in per_cluster_insights.items():
        feat = info_dict.get("feature")
        if not feat:
            continue
        z_val = info_dict.get("z_score")
        delta_val = info_dict.get("delta")
        if z_val is not None and np.isfinite(z_val):
            insights.append(
                insight(
                    "info",
                    f"Cluster {int(cluster_idx) + 1} is most differentiated by {feat} (z-score {z_val:.2f} vs overall mean).",
                )
            )
        elif delta_val is not None and np.isfinite(delta_val):
            insights.append(
                insight(
                    "info",
                    f"Cluster {int(cluster_idx) + 1} shows strong lift on {feat} (delta {delta_val:.2f} from mean).",
                )
            )
    if best_silhouette is not None:
        if best_silhouette >= 0.5:
            insights.append(
                insight("success", f"Silhouette score of {best_silhouette:.2f} indicates well separated clusters."),
            )
        elif best_silhouette >= 0.3:
            insights.append(
                insight("info", f"Silhouette score of {best_silhouette:.2f} suggests moderate separation."),
            )
        else:
            insights.append(
                insight("warning", f"Silhouette score of {best_silhouette:.2f} indicates overlapping clusters."),
            )
    else:
        insights.append(
            insight(
                "info",
                "Selected cluster configuration minimizes inertia; silhouette score unavailable for evaluation.",
            )
        )

    min_share = float(cluster_counts.min() / len(prepared_df))
    max_share = float(cluster_counts.max() / len(prepared_df))
    if min_share < 0.1:
        insights.append(
            insight(
                "warning",
                "One or more clusters contain fewer than 10% of samples. Consider rebalancing or adjusting features.",
            )
        )
    else:
        insights.append(
            insight("success", "Clusters are relatively balanced across the dataset."),
        )

    charts: List[AnalysisChart] = []
    if PCA is not None and scaled.shape[1] >= 2:
        try:
            pca = PCA(n_components=2)
            components = pca.fit_transform(scaled)
            fig, ax = plt.subplots(figsize=(6, 4))
            palette = sns.color_palette("husl", best_k)
            for cluster_idx in range(best_k):
                mask = best_labels == cluster_idx
                ax.scatter(
                    components[mask, 0],
                    components[mask, 1],
                    s=45,
                    alpha=0.75,
                    color=palette[cluster_idx % len(palette)],
                    label=f"Cluster {cluster_idx + 1}",
                )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Cluster Segmentation (PCA Projection)")
            ax.legend(loc="best", frameon=True)
            charts.append(
                AnalysisChart(
                    title="Cluster Segmentation",
                    image=fig_to_base64(fig),
                    description="Clusters projected onto the first two principal components.",
                )
            )
        except Exception:
            pass

    if discriminator_records and sns is not None and top_global_features:
        try:
            plot_features = top_global_features[: min(3, len(top_global_features))]
            melted = (
                prepared_df.assign(cluster=(best_labels + 1))[plot_features + ["cluster"]]
                .melt(id_vars="cluster", var_name="Feature", value_name="Value")
            )
            fig, ax = plt.subplots(figsize=(5 * max(1, len(plot_features)), 4))
            sns.boxplot(data=melted, x="Feature", y="Value", hue="cluster", ax=ax)
            ax.set_title("Top Discriminators by Cluster")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Value")
            ax.legend(title="Cluster", loc="best", frameon=True)
            charts.append(
                AnalysisChart(
                    title="Top Feature Distributions",
                    image=fig_to_base64(fig),
                    description="Boxplots showing how key differentiating features vary across clusters.",
                )
            )
        except Exception:
            pass

    assignments_preview = prepared_df.assign(cluster=(best_labels + 1)).head(50)
    details["cluster_assignments_preview"] = assignments_preview.to_dict(orient="records")
    details["evaluation_summary"] = [
        {
            "clusters": cand["k"],
            "silhouette": cand["silhouette"],
            "inertia": cand["inertia"],
        }
        for cand in candidates
    ]
    details["cluster_centers"] = [
        {
            "cluster": int(idx) + 1,
            "centroid": {
                feature: float(value)
                for feature, value in zip(feature_list, centers_original[int(idx)])
            },
            "size": int(cluster_counts.loc[idx]),
        }
        for idx in cluster_counts.index
    ]
    if discriminator_records:
        details["top_feature_discriminators"] = discriminator_records
    if discrimination_scores:
        details["feature_discrimination_scores"] = discrimination_scores

    summary = f"K-means segmentation produced {best_k} clusters across {len(feature_list)} numeric features."

    return AnalysisResult(
        analysis_id="cluster_segmentation_analysis",
        title="Cluster Segmentation",
        summary=summary,
        metrics=metrics,
        insights=insights,
        tables=[
            table
            for table in [centroid_table, evaluation_table, top_discriminator_table, discrimination_table]
            if table is not None
        ],
        charts=charts,
        details=details,
    )


def network_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="network_analysis",
            title="Network Analysis",
            summary="At least two numeric columns are required to form a correlation network.",
            status="warning",
            insights=[insight("warning", "Select multiple numeric columns to build a correlation network.")],
        )

    corr = df[numeric_cols].corr().abs()
    edges = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .rename(columns={"level_0": "Source", "level_1": "Target", 0: "Weight"})
    )
    edges = edges[edges["Weight"] > 0.6]

    table = dataframe_to_table(
        edges,
        title="Correlation Network Edges",
        description="Pairs of features with correlation weight > 0.6.",
        round_decimals=2,
    )

    insight_text = (
        f"Network contains {len(edges)} strong relationships" if not edges.empty else "No strong correlations above 0.6"
    )

    return AnalysisResult(
        analysis_id="network_analysis",
        title="Correlation Network Analysis",
        summary="Builds network edges based on strong correlations (>|0.6|).",
        tables=[table],
        insights=[insight("info", insight_text)],
    )


def entity_relationship_network(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    categorical_cols = [col for col in df.columns if df[col].dtype == "object" or str(df[col].dtype).startswith("category")]
    if len(categorical_cols) < 2:
        return AnalysisResult(
            analysis_id="entity_relationship_network",
            title="Entity Relationship Network",
            summary="At least two categorical columns are required to derive entity relationships.",
            status="warning",
            insights=[insight("warning", "Select multiple categorical columns to build entity networks.")],
        )

    col_a, col_b = categorical_cols[:2]
    co_occurrence = (
        df.groupby([col_a, col_b]).size().reset_index(name="Count").sort_values("Count", ascending=False)
    )

    table = dataframe_to_table(
        co_occurrence,
        title=f"Co-occurrence of {col_a} and {col_b}",
        description="Co-occurring categorical pairs ranked by frequency.",
    )

    return AnalysisResult(
        analysis_id="entity_relationship_network",
        title="Entity Relationship Network",
        summary=f"Co-occurrence analysis between '{col_a}' and '{col_b}'.",
        tables=[table],
        insights=[insight("info", "High co-occurrence counts indicate strong entity relationships." )],
    )


__all__ = [
    "multicollinearity_analysis",
    "pca_dimensionality_reduction",
    "pca_scree_plot",
    "pca_cumulative_variance",
    "pca_visualization",
    "pca_biplot",
    "pca_heatmap",
    "cluster_tendency_analysis",
    "cluster_segmentation_analysis",
    "network_analysis",
    "entity_relationship_network",
]
