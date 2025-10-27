"""Cluster tendency analysis component metadata."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ClusterTendencyAnalysis:
    """Describe cluster tendency evaluation for relationship exploration."""

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "name": "cluster_tendency_analysis",
            "display_name": "Cluster Tendency Assessment",
            "description": "Estimate whether numeric features form natural clusters using Hopkins statistics and silhouette sampling.",
            "category": "relationships",
            "complexity": "intermediate",
            "tags": ["clustering", "unsupervised", "hopkins", "silhouette"],
            "estimated_runtime": "3-8 seconds",
            "icon": "🧭",
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
        return len(numeric_cols) >= 2 and row_count >= 5

    @staticmethod
    def generate_code(data_preview: Optional[Dict[str, Any]] = None) -> str:
        return (
            "# Cluster tendency analysis is executed via the granular runtime.\n"
            "# Configure numeric columns and invoke `cluster_tendency_analysis` "
            "from the relationships runtime module.\n"
        )
