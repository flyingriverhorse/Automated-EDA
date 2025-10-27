"""Summary Statistics Analysis Component.

Focused analysis component for basic summary statistics of numeric variables.
"""
from typing import Dict, Any


class SummaryStatisticsAnalysis:
    """Focused component for summary statistics analysis"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "summary_statistics",
            "display_name": "Summary Statistics",
            "description": "Basic summary statistics: mean, median, min, max, std for numeric variables",
            "category": "univariate",
            "complexity": "basic",
            "required_data_types": ["numeric"],
            "estimated_runtime": "5-10 seconds",
            "icon": "calculator",
            "tags": ["statistics", "numeric", "descriptive"]
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
        """Generate focused summary statistics analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== SUMMARY STATISTICS ANALYSIS ===")
print("Basic descriptive statistics for numeric variables")
print()

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    print("❌ NO NUMERIC COLUMNS FOUND")
    print("   This analysis requires numeric data.")
    print("   Check if columns need type conversion.")
else:
    print(f"📊 ANALYZING {len(numeric_cols)} NUMERIC COLUMNS")
    print(f"   Columns: {numeric_cols}")
    print()
    
    # Basic summary statistics
    summary_stats = df[numeric_cols].describe()
    
    print("📈 BASIC SUMMARY STATISTICS")
    print(summary_stats.round(3))
    print()
    
    # Enhanced statistics table
    enhanced_stats = pd.DataFrame({
        'Column': numeric_cols,
        'Count': [df[col].count() for col in numeric_cols],
        'Missing': [df[col].isnull().sum() for col in numeric_cols],
        'Mean': [df[col].mean() for col in numeric_cols],
        'Median': [df[col].median() for col in numeric_cols],
        'Mode': [df[col].mode().iloc[0] if len(df[col].mode()) > 0 else np.nan for col in numeric_cols],
        'Min': [df[col].min() for col in numeric_cols],
        'Max': [df[col].max() for col in numeric_cols],
        'Range': [df[col].max() - df[col].min() for col in numeric_cols],
        'Std': [df[col].std() for col in numeric_cols],
        'Var': [df[col].var() for col in numeric_cols],
    })
    
    # Round numeric values for better display
    numeric_columns_for_rounding = ['Mean', 'Median', 'Mode', 'Min', 'Max', 'Range', 'Std', 'Var']
    for col in numeric_columns_for_rounding:
        enhanced_stats[col] = enhanced_stats[col].round(3)
    
    print("📋 ENHANCED STATISTICS TABLE")
    print(enhanced_stats.to_string(index=False))
    print()
    
    # Statistical insights for each column
    print("🎯 COLUMN-SPECIFIC INSIGHTS")
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            mean_val = col_data.mean()
            median_val = col_data.median()
            std_val = col_data.std()
            
            print(f"\\n   📊 {col.upper()}:")
            print(f"      • Range: {col_data.min():.3f} to {col_data.max():.3f}")
            print(f"      • Central tendency: Mean={mean_val:.3f}, Median={median_val:.3f}")
            
            # Compare mean vs median
            if abs(mean_val - median_val) < 0.1 * std_val:
                print(f"      • Distribution: Likely symmetric (mean ≈ median)")
            elif mean_val > median_val:
                print(f"      • Distribution: Right-skewed (mean > median)")
            else:
                print(f"      • Distribution: Left-skewed (mean < median)")
            
            # Coefficient of variation
            if mean_val != 0:
                cv = (std_val / abs(mean_val)) * 100
                print(f"      • Variability: CV = {cv:.1f}% ", end="")
                if cv < 15:
                    print("(Low variability)")
                elif cv < 35:
                    print("(Moderate variability)")
                else:
                    print("(High variability)")
            
            # Outlier indication using IQR
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            print(f"      • Potential outliers: {len(outliers)} values ({len(outliers)/len(col_data)*100:.1f}%)")
    
    print()
    
    # Correlation between statistics
    if len(numeric_cols) > 1:
        print("🔗 RELATIONSHIPS BETWEEN STATISTICS")
        stats_for_corr = enhanced_stats[['Column', 'Mean', 'Std', 'Range', 'Var']].set_index('Column')
        
        # Find columns with similar scales
        means = enhanced_stats.set_index('Column')['Mean']
        similar_scale_pairs = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                ratio = max(means[col1], means[col2]) / (min(means[col1], means[col2]) + 1e-10)
                if ratio < 2:  # Similar order of magnitude
                    similar_scale_pairs.append((col1, col2))
        
        if similar_scale_pairs:
            print(f"   📏 Similar scale columns (good for comparison): {similar_scale_pairs}")
        
        # Highest and lowest variability
        std_values = enhanced_stats.set_index('Column')['Std'].sort_values()
        print(f"   📉 Lowest variability: {std_values.index[0]} (std={std_values.iloc[0]:.3f})")
        print(f"   📈 Highest variability: {std_values.index[-1]} (std={std_values.iloc[-1]:.3f})")
    
    print()
    
    # Data quality indicators from statistics
    print("✅ DATA QUALITY INDICATORS")
    quality_issues = []
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        
        # Check for constant values
        if col_data.std() == 0:
            quality_issues.append(f"{col}: Constant values (no variation)")
        
        # Check for extreme outliers (beyond 3 standard deviations)
        if len(col_data) > 0:
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            extreme_outliers = (z_scores > 3).sum()
            if extreme_outliers > len(col_data) * 0.1:  # More than 10% extreme outliers
                quality_issues.append(f"{col}: Many extreme outliers ({extreme_outliers} values)")
        
        # Check for suspicious ranges (negative values where they shouldn't be)
        if col.lower() in ['age', 'price', 'count', 'quantity', 'amount'] and col_data.min() < 0:
            quality_issues.append(f"{col}: Negative values in column that should be positive")
    
    if quality_issues:
        print("   ⚠️ Potential issues found:")
        for issue in quality_issues:
            print(f"      • {issue}")
    else:
        print("   ✅ No obvious statistical anomalies detected")
    
    print()
    print("💡 RECOMMENDATIONS")
    print("   • Use median instead of mean for skewed distributions")
    print("   • Consider log transformation for high-variability columns")
    print("   • Investigate outliers before analysis or modeling")
    print("   • Check if missing values follow any patterns")
    
    if len(numeric_cols) > 1:
        print("   • Compare scales before correlation analysis")
        print("   • Consider standardization for machine learning")

print("\\n" + "="*50)
'''