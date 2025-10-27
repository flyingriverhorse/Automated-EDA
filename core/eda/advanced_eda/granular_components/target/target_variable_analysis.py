"""Target Variable Analysis granular component."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

TARGET_NAME_HINTS = {
    "target",
    "label",
    "outcome",
    "response",
    "y",
    "class",
    "status",
    "churn",
    "default",
    "fraud",
    "success",
    "converted",
}


class TargetVariableAnalysis:
    """Metadata and notebook code generator for target variable analysis."""

    MAX_SUGGESTED_COLUMNS = 3

    def __init__(self) -> None:
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "target_variable_analysis",
            "display_name": "Target Variable Analysis",
            "description": (
                "Assess target columns for modelling readiness, including class balance "
                "for classification tasks and range diagnostics for regression targets."
            ),
            "category": "target",
            "complexity": "basic",
            "tags": ["target", "modelling", "distribution", "class-balance", "regression"],
            "estimated_runtime": "1-3 seconds",
            "icon": "🎯",
            "column_requirements": {
                "min": 1,
                "max": self.MAX_SUGGESTED_COLUMNS,
                "description": "Select up to three candidate target columns to profile."
            },
        }

    def validate_data_compatibility(self, data_preview: Optional[Dict[str, Any]] = None) -> bool:
        if not data_preview:
            return True

        columns = data_preview.get("columns")
        if isinstance(columns, list) and columns:
            return True

        all_columns = data_preview.get("all_columns") or data_preview.get("original_columns")
        if isinstance(all_columns, list) and all_columns:
            return True

        return False

    def generate_code(self, data_preview: Optional[Dict[str, Any]] = None) -> str:
        selected_columns: List[str] = []
        if data_preview:
            if isinstance(data_preview.get("columns"), list):
                selected_columns = list(dict.fromkeys(data_preview["columns"]))
            elif isinstance(data_preview.get("selected_columns"), list):
                selected_columns = list(dict.fromkeys(data_preview["selected_columns"]))

        hints: List[str] = []
        if not selected_columns and data_preview:
            all_columns = data_preview.get("all_columns") or data_preview.get("original_columns") or []
            for column in all_columns:
                normalized = str(column).lower()
                if normalized in TARGET_NAME_HINTS or any(hint in normalized for hint in TARGET_NAME_HINTS):
                    hints.append(str(column))
                    if len(hints) >= self.MAX_SUGGESTED_COLUMNS:
                        break

        column_list_repr = selected_columns or hints
        column_list_repr = column_list_repr[: self.MAX_SUGGESTED_COLUMNS]

        columns_literal = repr(column_list_repr)

        return f"""
# ===== TARGET VARIABLE ANALYSIS =====
# Profiles candidate target columns for modelling readiness.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

candidate_targets = {columns_literal}
if not candidate_targets:
    # Fallback: use low-cardinality columns as potential classification targets
    value_counts = df.nunique().sort_values()
    candidate_targets = value_counts[value_counts <= 20].index.tolist()[:3]
    print(f"⚠️ No explicit targets supplied, falling back to {{len(candidate_targets)}} heuristic candidate(s).")

if not candidate_targets:
    raise ValueError("No suitable target columns were supplied or detected.")

for column in candidate_targets:
    if column not in df.columns:
        print(f"⚠️ Skipping '{{column}}' because the column is not present in the dataframe.")
        continue

    series = df[column]
    total_count = len(series)
    missing = series.isna().sum()
    non_null = series.dropna()
    unique_count = non_null.nunique()
    unique_ratio = unique_count / len(non_null) if len(non_null) else 0

    print("\n" + "=" * 72)
    print(f"🎯 Target column: {{column}}")
    print("=" * 72)
    print(f"Total observations: {{total_count:,}}")
    print(f"Missing values: {{missing:,}} ({{missing / total_count * 100 if total_count else 0:.1f}}%)")
    print(f"Unique (non-null): {{unique_count:,}}")

    likely_classification = (
        series.dtype == object
        or str(series.dtype).startswith("category")
        or unique_count <= min(20, max(5, total_count // 10))
    )

    if likely_classification:
        print("📊 Detected as classification target")
        counts = non_null.value_counts().sort_values(ascending=False)
        percent = counts / total_count * 100 if total_count else counts * 0
        summary = pd.DataFrame(
            {
                "Class": counts.index.astype(str),
                "Count": counts.values.astype(int),
                "Percent": percent.round(2),
            }
        )
        print(summary.head(15).to_string(index=False))

        majority_share = percent.iloc[0] if not percent.empty else 0
        if majority_share >= 75:
            print(f"⚠️  Imbalance warning: top class covers {{majority_share:.1f}}% of rows.")
        elif majority_share >= 60:
            print(f"ℹ️  Moderate imbalance: top class covers {{majority_share:.1f}}% of rows.")
        else:
            print("✅ Classes appear reasonably balanced.")

        plt.figure(figsize=(6, 4))
        sns.barplot(x=counts.index.astype(str)[:15], y=counts.values[:15], palette="Blues_d")
        plt.title(f"Class Distribution: {{column}}")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("📈 Detected as regression target")
        stats = non_null.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        print(stats.to_frame(name="Value").to_string())

        spread = stats.loc["max"] - stats.loc["min"] if "max" in stats and "min" in stats else None
        if spread is not None:
            print(f"Range: {{spread:.3f}}")

        plt.figure(figsize=(6, 4))
        sns.histplot(non_null, bins=min(30, max(10, int(np.sqrt(len(non_null))))) or 10, kde=True, color="#F6AD55")
        plt.title(f"Distribution: {{column}}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

print("\n🎉 Target variable analysis complete!\n")
"""
