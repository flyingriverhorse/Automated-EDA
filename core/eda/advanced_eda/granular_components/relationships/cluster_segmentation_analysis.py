"""Cluster segmentation analysis component metadata."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ClusterSegmentationAnalysis:
    """Metadata for cluster segmentation workflows."""

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "name": "cluster_segmentation_analysis",
            "display_name": "Cluster Segmentation",
            "description": "Create K-means segments, evaluate silhouettes, and summarise centroids across numeric features.",
            "category": "relationships",
            "complexity": "advanced",
            "tags": ["clustering", "segmentation", "kmeans", "unsupervised"],
            "estimated_runtime": "5-12 seconds",
            "icon": "🧩",
        }

    @staticmethod
    def validate_data_compatibility(data_preview: Optional[Dict[str, Any]] = None) -> bool:
        if not data_preview:
            return True

        numeric_cols = data_preview.get("numeric_columns", []) or []
        row_count = (
            data_preview.get("row_count")
            or data_preview.get("total_rows")
            or (data_preview.get("shape", [0])[0] if data_preview.get("shape") else 0)
            or 0
        )
        return len(numeric_cols) >= 2 and row_count >= 6

    @staticmethod
    def generate_code(data_preview: Optional[Dict[str, Any]] = None) -> str:
        return (
            "# Cluster segmentation is handled by the granular runtime.\n"
            "# Use `cluster_segmentation_analysis` from the relationships runtime module to run the analysis.\n"
        )
