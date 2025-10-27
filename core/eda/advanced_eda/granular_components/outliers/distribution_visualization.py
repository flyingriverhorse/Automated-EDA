"""Outlier Distribution Visualization Component.

Focused analysis component for visualizing outliers in distribution plots only.
"""
from typing import Dict, Any


class OutlierDistributionVisualization:
    """Focused component for outlier visualization in distributions"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "outlier_distribution_visualization",
            "display_name": "Outlier Distribution Plots",
            "description": "Distribution plots highlighting outliers using statistical methods",
            "category": "outliers",
            "subcategory": "visualization",
            "complexity": "basic",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-15 seconds",
            "icon": "chart-outliers",
            "tags": ["visualization", "outliers", "distribution", "iqr", "zscore"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if dataset has numeric columns"""
        if not data_preview:
            return True
        
        data = data_preview.get("data", [])
        if not data:
            return True
            
        # Check if any columns might be numeric
        for row in data[:5]:
            for value in row:
                try:
                    float(str(value))
                    return True
                except (ValueError, TypeError):
                    continue
        return False
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        """Generate focused outlier distribution visualization code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=== OUTLIER DISTRIBUTION VISUALIZATION ===")
print("Distribution plots with outlier highlighting")
print()

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    print("❌ NO NUMERIC COLUMNS FOUND")
    print("   This analysis requires numeric data.")
else:
    print(f"🔍 ANALYZING OUTLIERS IN {len(numeric_cols)} NUMERIC COLUMNS")
    print()
    
    # Select up to 12 columns for visualization
    display_cols = numeric_cols[:12]
    n_cols = len(display_cols)
    n_cols_dist = min(3, n_cols)
    n_rows_dist = (n_cols + n_cols_dist - 1) // n_cols_dist
    
    fig, axes = plt.subplots(n_rows_dist, n_cols_dist, figsize=(5*n_cols_dist, 4*n_rows_dist))
    fig.suptitle('Distribution Analysis with Outlier Detection', fontsize=16, fontweight='bold')
    
    # Ensure axes is always a 2D array for consistent indexing
    if n_rows_dist == 1 and n_cols_dist == 1:
        axes = np.array([[axes]])
    elif n_rows_dist == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_dist == 1:
        axes = axes.reshape(-1, 1)
    
    for i, col in enumerate(display_cols):
        row = i // n_cols_dist
        col_idx = i % n_cols_dist
        
        ax = axes[row, col_idx]
        col_data = df[col].dropna()
        
        if len(col_data) < 4:
            ax.text(0.5, 0.5, f'Insufficient data\\nfor {col}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{col} - Insufficient Data')
            continue
        
        # Calculate outliers using IQR method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers_iqr = (col_data < lower_bound) | (col_data > upper_bound)
        n_outliers_iqr = outliers_iqr.sum()
        
        # Calculate outliers using Z-score method
        z_scores = np.abs(stats.zscore(col_data))
        outliers_zscore = z_scores > 3
        n_outliers_zscore = outliers_zscore.sum()
        
        # Create histogram
        ax.hist(col_data, bins=30, alpha=0.7, color='lightblue', label='Normal Data')
        
        # Highlight IQR outliers
        if n_outliers_iqr > 0:
            outlier_data = col_data[outliers_iqr]
            ax.hist(outlier_data, bins=30, alpha=0.8, color='red', 
                   label=f'IQR Outliers ({n_outliers_iqr})')
        
        # Add vertical lines for thresholds
        ax.axvline(lower_bound, color='red', linestyle='--', alpha=0.8, label=f'IQR Lower: {lower_bound:.2f}')
        ax.axvline(upper_bound, color='red', linestyle='--', alpha=0.8, label=f'IQR Upper: {upper_bound:.2f}')
        
        # Add mean and median lines
        mean_val = col_data.mean()
        median_val = col_data.median()
        ax.axvline(mean_val, color='green', linestyle='-', alpha=0.8, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='-', alpha=0.8, label=f'Median: {median_val:.2f}')
        
        ax.set_title(f'{col}\\nIQR: {n_outliers_iqr} | Z-score: {n_outliers_zscore}')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Remove empty subplots
    total_subplots = n_rows_dist * n_cols_dist
    if n_cols < total_subplots:
        for i in range(n_cols, total_subplots):
            row = i // n_cols_dist
            col_idx = i % n_cols_dist
            ax_to_remove = axes[row, col_idx]
            fig.delaxes(ax_to_remove)
    
    plt.tight_layout()
    plt.show()
    print()
    
    # Summary table of outliers
    print("📊 OUTLIER DETECTION SUMMARY:")
    summary_data = []
    
    for col in display_cols:
        col_data = df[col].dropna()
        
        if len(col_data) >= 4:
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(col_data))
            outliers_zscore = (z_scores > 3).sum()
            
            # Modified Z-score method
            median_val = col_data.median()
            mad = np.median(np.abs(col_data - median_val))
            modified_z_scores = 0.6745 * (col_data - median_val) / mad
            outliers_modified_z = (np.abs(modified_z_scores) > 3.5).sum()
            
            summary_data.append({
                'Column': col,
                'Total_Points': len(col_data),
                'IQR_Outliers': outliers_iqr,
                'ZScore_Outliers': outliers_zscore,
                'Modified_Z_Outliers': outliers_modified_z,
                'IQR_Percentage': f"{outliers_iqr/len(col_data)*100:.1f}%"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print()
    
    print("🔍 OUTLIER DETECTION METHODS:")
    print("   • IQR Method: Points beyond Q1-1.5*IQR or Q3+1.5*IQR")
    print("   • Z-Score Method: Points with |z-score| > 3")
    print("   • Modified Z-Score: More robust, uses median and MAD")
    print("   • Visual inspection recommended for final decision")

print("\\n" + "="*50)
print("✅ OUTLIER DISTRIBUTION VISUALIZATION COMPLETE")
print("="*50)
'''