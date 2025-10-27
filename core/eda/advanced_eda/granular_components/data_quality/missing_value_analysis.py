"""Missing Value Analysis Component.

Focused analysis component for analyzing missing values in the dataset.
"""
from typing import Dict, Any


class MissingValueAnalysis:
    """Focused component for missing value analysis"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "missing_value_analysis",
            "display_name": "Missing Value Analysis",
            "description": "Analyze patterns and distribution of missing values in the dataset",
            "category": "data_quality",
            "complexity": "basic",
            "required_data_types": ["any"],
            "estimated_runtime": "5-15 seconds",
            "icon": "question-circle",
            "tags": ["missing-values", "data-quality", "null-values"]
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview: Dict[str, Any] = None) -> bool:
        """This analysis works with any dataset"""
        return True
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        """Generate focused missing value analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== MISSING VALUE ANALYSIS ===")
print("Comprehensive analysis of missing values in the dataset")
print()

# Basic missing value statistics
total_cells = df.size
total_missing = df.isnull().sum().sum()
missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0

print(f"📊 MISSING VALUE SUMMARY")
print(f"   Total data points: {total_cells:,}")
print(f"   Missing values: {total_missing:,}")
print(f"   Missing percentage: {missing_percentage:.2f}%")
print()

# Column-wise missing value analysis
missing_by_column = df.isnull().sum()
missing_columns = missing_by_column[missing_by_column > 0]

if len(missing_columns) > 0:
    print("🔍 MISSING VALUES BY COLUMN")
    missing_df = pd.DataFrame({
        'Column': missing_columns.index,
        'Missing Count': missing_columns.values,
        'Missing %': (missing_columns.values / len(df) * 100).round(2),
        'Data Type': [str(df[col].dtype) for col in missing_columns.index]
    }).sort_values('Missing Count', ascending=False)
    
    print(missing_df.to_string(index=False))
    print()
    
    # Visualization of missing values
    print("📈 MISSING VALUE VISUALIZATION")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Missing Value Analysis', fontsize=16, fontweight='bold')
    
    # 1. Missing values heatmap
    if len(df) <= 1000:  # Only for reasonably sized datasets
        sns.heatmap(df.isnull(), cmap='viridis', cbar=True, ax=axes[0,0])
        axes[0,0].set_title('Missing Values Heatmap')
        axes[0,0].set_xlabel('Columns')
        axes[0,0].set_ylabel('Rows')
    else:
        # Sample for large datasets
        sample_df = df.sample(min(1000, len(df)))
        sns.heatmap(sample_df.isnull(), cmap='viridis', cbar=True, ax=axes[0,0])
        axes[0,0].set_title('Missing Values Heatmap (Sample)')
    
    # 2. Missing values bar chart
    missing_counts = df.isnull().sum().sort_values(ascending=True)
    missing_counts = missing_counts[missing_counts > 0]
    if len(missing_counts) > 0:
        missing_counts.plot(kind='barh', ax=axes[0,1], color='coral')
        axes[0,1].set_title('Missing Values by Column')
        axes[0,1].set_xlabel('Number of Missing Values')
    
    # 3. Missing value percentage
    missing_percentages = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    missing_percentages = missing_percentages[missing_percentages > 0]
    if len(missing_percentages) > 0:
        missing_percentages.plot(kind='barh', ax=axes[1,0], color='lightblue')
        axes[1,0].set_title('Missing Value Percentage by Column')
        axes[1,0].set_xlabel('Percentage Missing (%)')
    
    # 4. Missing value patterns
    if len(missing_columns) > 1:
        # Matrix showing missing value combinations
        missing_matrix = df[missing_columns.index].isnull().astype(int)
        pattern_counts = missing_matrix.value_counts().head(10)
        
        if len(pattern_counts) > 1:
            pattern_counts.plot(kind='bar', ax=axes[1,1], color='lightgreen')
            axes[1,1].set_title('Top Missing Value Patterns')
            axes[1,1].set_xlabel('Pattern Index')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    print()
    
    # Row-wise missing value analysis
    rows_with_missing = df.isnull().any(axis=1).sum()
    rows_missing_percent = (rows_with_missing / len(df) * 100) if len(df) > 0 else 0
    
    print("📋 ROW-WISE MISSING VALUE ANALYSIS")
    print(f"   Rows with missing values: {rows_with_missing:,} ({rows_missing_percent:.2f}%)")
    print(f"   Complete rows (no missing): {len(df) - rows_with_missing:,}")
    
    # Missing value distribution by row
    missing_per_row = df.isnull().sum(axis=1)
    if missing_per_row.max() > 0:
        print()
        print("   Missing values per row distribution:")
        row_missing_dist = missing_per_row.value_counts().sort_index()
        for missing_count, row_count in row_missing_dist.items():
            if missing_count > 0:
                print(f"     {missing_count} missing: {row_count} rows")
    
    print()
    
    # Data type specific missing value analysis
    print("🎯 DATA TYPE SPECIFIC ANALYSIS")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime']).columns
    
    for col_type, cols in [('Numeric', numeric_cols), ('Categorical', categorical_cols), ('DateTime', datetime_cols)]:
        if len(cols) > 0:
            missing_in_type = df[cols].isnull().sum().sum()
            total_in_type = df[cols].size
            percent_in_type = (missing_in_type / total_in_type * 100) if total_in_type > 0 else 0
            print(f"   {col_type} columns: {missing_in_type:,} missing ({percent_in_type:.2f}%)")
    
    print()
    print("💡 HANDLING RECOMMENDATIONS")
    
    # Provide recommendations based on missing patterns
    high_missing_cols = missing_df[missing_df['Missing %'] > 50]['Column'].tolist()
    medium_missing_cols = missing_df[(missing_df['Missing %'] > 10) & (missing_df['Missing %'] <= 50)]['Column'].tolist()
    low_missing_cols = missing_df[missing_df['Missing %'] <= 10]['Column'].tolist()
    
    if high_missing_cols:
        print(f"   🔴 High missing (>50%): {high_missing_cols}")
        print("      → Consider dropping these columns or investigating why so much data is missing")
    
    if medium_missing_cols:
        print(f"   🟡 Medium missing (10-50%): {medium_missing_cols}")
        print("      → Consider imputation strategies or flagging missing values")
    
    if low_missing_cols:
        print(f"   🟢 Low missing (<10%): {low_missing_cols}")
        print("      → Simple imputation (mean/mode/median) or row removal may work")
    
    print()
    print("   Suggested approaches:")
    print("   • Numeric: Mean/median imputation, forward/backward fill")
    print("   • Categorical: Mode imputation, 'Unknown' category")
    print("   • Time series: Forward fill, interpolation")
    print("   • Advanced: KNN imputation, predictive modeling")

else:
    print("✅ NO MISSING VALUES FOUND")
    print("   Your dataset is complete with no missing values!")
    print("   This is excellent for analysis and modeling.")

print("\\n" + "="*50)
'''