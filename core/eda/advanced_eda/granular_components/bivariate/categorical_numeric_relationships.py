"""Categorical vs Numeric Relationships Analysis Component.

Generates grouped box/violin visualizations and aggregation statistics for categorical-numeric pairs.
"""


class CategoricalNumericRelationshipAnalysis:
    """Explore numeric distributions across categorical segments."""

    @staticmethod
    def get_metadata():
        return {
            "name": "categorical_numeric_relationships",
            "display_name": "Categorical vs Numeric Explorer",
            "description": "Grouped box and violin plots with segment statistics for categorical vs numeric pairs",
            "category": "bivariate",
            "complexity": "intermediate",
            "tags": ["categorical", "numeric", "boxplot", "violin", "group statistics"],
            "estimated_runtime": "3-8 seconds",
            "icon": "🧮",
        }

    @staticmethod
    def validate_data_compatibility(data_preview=None):
        if not data_preview:
            return True
        categorical_cols = data_preview.get("categorical_columns", []) + data_preview.get("object_columns", [])
        numeric_cols = data_preview.get("numeric_columns", [])
        return bool(categorical_cols and numeric_cols)

    @staticmethod
    def generate_code(data_preview=None):
        return '''
# ===== CATEGORICAL VS NUMERIC RELATIONSHIPS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

print("="*70)
print("🧮 CATEGORICAL VS NUMERIC RELATIONSHIPS")
print("="*70)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n📊 Categorical columns: {len(categorical_cols)}")
print(f"📈 Numeric columns: {len(numeric_cols)}")

if not categorical_cols or not numeric_cols:
    print("❌ Need at least one categorical and one numeric column")
else:
    max_pairs = 8
    max_categories = 12
    analyzed_pairs = []
    summary_rows = []

    for cat_col in categorical_cols:
        unique_vals = df[cat_col].dropna().unique()
        if len(unique_vals) < 2:
            print(f"⚠️  Skipping '{cat_col}' — fewer than 2 categories")
            continue
        if len(unique_vals) > max_categories:
            print(f"⚠️  Skipping '{cat_col}' — too many categories ({len(unique_vals)})")
            continue

        for num_col in numeric_cols:
            if len(analyzed_pairs) >= max_pairs:
                break

            pair_df = df[[cat_col, num_col]].dropna()
            if pair_df.empty or pair_df[num_col].nunique() <= 1:
                continue

            pair_df = pair_df.copy()
            pair_df[cat_col] = pair_df[cat_col].astype(str)

            grouped = (
                pair_df.groupby(cat_col)[num_col]
                .agg(['mean', 'median', 'std', 'count', 'min', 'max'])
                .reset_index()
            )

            grouped.rename(columns={
                cat_col: 'Category',
                'mean': 'Mean',
                'median': 'Median',
                'std': 'Std',
                'count': 'Count',
                'min': 'Min',
                'max': 'Max'
            }, inplace=True)

            grouped.insert(0, 'Categorical Column', cat_col)
            grouped.insert(2, 'Numeric Column', num_col)
            summary_rows.extend(grouped.to_dict(orient='records'))

            mean_range = grouped['Mean'].max() - grouped['Mean'].min()
            pair_info = {
                'categorical_column': cat_col,
                'numeric_column': num_col,
                'categories_analyzed': int(grouped.shape[0]),
                'mean_range': float(mean_range),
                'min_mean': float(grouped['Mean'].min()),
                'max_mean': float(grouped['Mean'].max()),
            }

            groups_for_anova = [vals[num_col].values for _, vals in pair_df.groupby(cat_col) if len(vals) > 1]
            if len(groups_for_anova) >= 2:
                try:
                    stat, p_value = f_oneway(*groups_for_anova)
                    pair_info['anova_statistic'] = float(stat)
                    pair_info['anova_p_value'] = float(p_value)
                except Exception as err:
                    pair_info['anova_error'] = str(err)

            analyzed_pairs.append(pair_info)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.boxplot(data=pair_df, x=cat_col, y=num_col, ax=axes[0], palette='Set2')
            axes[0].set_title(f"Boxplot: {num_col} by {cat_col}")
            axes[0].set_xlabel(cat_col)
            axes[0].set_ylabel(num_col)
            axes[0].tick_params(axis='x', rotation=30)

            try:
                sns.violinplot(data=pair_df, x=cat_col, y=num_col, ax=axes[1], palette='muted', cut=0)
                axes[1].set_title(f"Violin: {num_col} by {cat_col}")
                axes[1].set_xlabel(cat_col)
                axes[1].set_ylabel(num_col)
                axes[1].tick_params(axis='x', rotation=30)
            except Exception:
                axes[1].axis('off')
                axes[1].text(0.5, 0.5, 'Violin plot unavailable', ha='center', va='center')

            plt.tight_layout()
            plt.show()

        if len(analyzed_pairs) >= max_pairs:
            print("\n⚠️  Pair limit reached — showing the first {max_pairs} combinations")
            break

    if not analyzed_pairs:
        print("❌ No suitable categorical/numeric pairs were found")
    else:
        print("\n📋 GROUP STATISTICS")
        summary_df = pd.DataFrame(summary_rows)
        display(summary_df.head(25))

        pair_df = pd.DataFrame(analyzed_pairs)
        pair_df_sorted = pair_df.sort_values('mean_range', ascending=False)
        print("\n🔍 PAIR SUMMARY (sorted by mean range)")
        display(pair_df_sorted)

        significant = pair_df.dropna(subset=['anova_p_value']) if 'anova_p_value' in pair_df else pd.DataFrame()
        significant = significant[significant['anova_p_value'] < 0.05]
        if not significant.empty:
            print("\n🚨 Significant differences detected (ANOVA p < 0.05):")
            for _, row in significant.sort_values('anova_p_value').iterrows():
                print(f"   • {row['numeric_column']} by {row['categorical_column']} — p = {row['anova_p_value']:.4f}")
        else:
            print("\nℹ️  No ANOVA p-values below 0.05 detected across analyzed pairs")

print("\n" + "="*70)
print("✅ Categorical vs numeric exploration complete!")
'''


def get_component():
    return CategoricalNumericRelationshipAnalysis
