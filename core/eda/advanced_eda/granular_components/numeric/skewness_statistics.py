"""Skewness Statistics Analysis Component.

Focused component specifically for skewness statistical analysis without visualizations.
"""
from typing import Dict, Any


class SkewnessStatisticsAnalysis:
    """Focused component for skewness statistical analysis"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "skewness_statistics",
            "display_name": "Skewness Statistics",
            "description": "Statistical analysis of distribution skewness with interpretations and tests",
            "category": "univariate",
            "subcategory": "skewness",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "5-10 seconds",
            "icon": "📊",
            "tags": ["skewness", "statistics", "normality", "analysis"]
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
        """Generate focused skewness statistics analysis code"""
        return '''import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("📊 SKEWNESS STATISTICS ANALYSIS")
print("="*60)
print("Statistical analysis of distribution skewness")
print()

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    print("❌ NO NUMERIC COLUMNS FOUND")
    print("   This analysis requires numeric data.")
else:
    print(f"📊 ANALYZING SKEWNESS STATISTICS FOR {len(numeric_cols)} NUMERIC COLUMNS")
    print()
    
    # Calculate skewness for all numeric columns
    skewness_data = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 1:
            skew_value = stats.skew(col_data)
            
            # Additional statistics
            kurtosis_value = stats.kurtosis(col_data)
            mean_val = col_data.mean()
            median_val = col_data.median()
            
            skewness_data.append({
                'Column': col,
                'Skewness': round(skew_value, 4),
                'Kurtosis': round(kurtosis_value, 4),
                'Mean': round(mean_val, 4),
                'Median': round(median_val, 4),
                'Count': len(col_data),
                'Missing': df[col].isnull().sum()
            })
    
    # Create skewness summary table
    skew_df = pd.DataFrame(skewness_data)
    
    if len(skew_df) == 0:
        print("❌ No valid numeric columns found for skewness analysis")
    else:
        # Add interpretation functions
        def interpret_skewness(skew):
            if abs(skew) < 0.5:
                return "Approximately Symmetric"
            elif abs(skew) < 1:
                return "Moderately Skewed"
            else:
                return "Highly Skewed"
        
        def skew_direction(skew):
            if skew > 0.5:
                return "Right (Positive)"
            elif skew < -0.5:
                return "Left (Negative)"
            else:
                return "Approximately Symmetric"
        
        def interpret_kurtosis(kurt):
            if kurt > 3:
                return "Heavy-tailed (Leptokurtic)"
            elif kurt < 3:
                return "Light-tailed (Platykurtic)"
            else:
                return "Normal-tailed (Mesokurtic)"
        
        # Add interpretation columns
        skew_df['Skewness_Interpretation'] = skew_df['Skewness'].apply(interpret_skewness)
        skew_df['Skewness_Direction'] = skew_df['Skewness'].apply(skew_direction)
        skew_df['Kurtosis_Interpretation'] = skew_df['Kurtosis'].apply(interpret_kurtosis)
        skew_df['Mean_Median_Diff'] = skew_df['Mean'] - skew_df['Median']
        
        # Display comprehensive skewness table
        print("📋 COMPREHENSIVE SKEWNESS STATISTICS TABLE:")
        print("-" * 100)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(skew_df.to_string(index=False))
        print()
        
        # Detailed analysis by skewness categories
        print("📊 SKEWNESS ANALYSIS BY CATEGORIES:")
        print("-" * 60)
        
        # Categorize variables by skewness
        symmetric_vars = skew_df[skew_df['Skewness'].abs() < 0.5]
        mod_skewed_vars = skew_df[(skew_df['Skewness'].abs() >= 0.5) & (skew_df['Skewness'].abs() < 1)]
        high_skewed_vars = skew_df[skew_df['Skewness'].abs() >= 1]
        
        print(f"\\n🟢 APPROXIMATELY SYMMETRIC VARIABLES ({len(symmetric_vars)}):")
        if len(symmetric_vars) > 0:
            for _, row in symmetric_vars.iterrows():
                print(f"   • {row['Column']}: skewness = {row['Skewness']:.3f}")
        else:
            print("   None found")
        
        print(f"\\n🟡 MODERATELY SKEWED VARIABLES ({len(mod_skewed_vars)}):")
        if len(mod_skewed_vars) > 0:
            for _, row in mod_skewed_vars.iterrows():
                direction = "right" if row['Skewness'] > 0 else "left"
                print(f"   • {row['Column']}: skewness = {row['Skewness']:.3f} ({direction})")
        else:
            print("   None found")
        
        print(f"\\n🔴 HIGHLY SKEWED VARIABLES ({len(high_skewed_vars)}):")
        if len(high_skewed_vars) > 0:
            for _, row in high_skewed_vars.iterrows():
                direction = "right" if row['Skewness'] > 0 else "left"
                print(f"   • {row['Column']}: skewness = {row['Skewness']:.3f} ({direction})")
        else:
            print("   None found")
        
        # Statistical tests for normality
        print("\\n🧪 NORMALITY TESTS (Shapiro-Wilk Test):")
        print("-" * 60)
        print("Testing null hypothesis: Data comes from normal distribution")
        print("p < 0.05 → Reject normality (data is NOT normal)")
        print()
        
        normality_results = []
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) >= 3 and len(col_data) <= 5000:  # Shapiro-Wilk limitations
                try:
                    stat, p_value = stats.shapiro(col_data)
                    is_normal = p_value > 0.05
                    normality_results.append({
                        'Column': col,
                        'Statistic': round(stat, 4),
                        'P_Value': p_value,
                        'Is_Normal': is_normal,
                        'Sample_Size': len(col_data)
                    })
                except:
                    continue
        
        if normality_results:
            norm_df = pd.DataFrame(normality_results)
            for _, row in norm_df.iterrows():
                status = "✅ Normal" if row['Is_Normal'] else "❌ Not Normal"
                p_str = f"{row['P_Value']:.2e}" if row['P_Value'] < 0.001 else f"{row['P_Value']:.4f}"
                print(f"   {row['Column']:20s}: {status:12s} (p = {p_str}, n = {row['Sample_Size']})")
        
        # Mean vs Median analysis
        print("\\n📊 MEAN VS MEDIAN ANALYSIS:")
        print("-" * 60)
        print("Mean ≈ Median → Symmetric distribution")
        print("Mean > Median → Right-skewed (positive skew)")
        print("Mean < Median → Left-skewed (negative skew)")
        print()
        
        for _, row in skew_df.iterrows():
            mean_median_diff = row['Mean_Median_Diff']
            if abs(mean_median_diff) < 0.1 * abs(row['Mean']):  # Less than 10% difference
                relationship = "Mean ≈ Median (symmetric)"
            elif mean_median_diff > 0:
                relationship = "Mean > Median (right-skewed)"
            else:
                relationship = "Mean < Median (left-skewed)"
            
            print(f"   {row['Column']:20s}: {relationship}")
            print(f"      Mean: {row['Mean']:10.3f}, Median: {row['Median']:10.3f}, Diff: {mean_median_diff:8.3f}")
        
        # Recommendations for transformation
        print("\\n💡 TRANSFORMATION RECOMMENDATIONS:")
        print("-" * 60)
        
        high_right_skew = skew_df[skew_df['Skewness'] > 1]
        high_left_skew = skew_df[skew_df['Skewness'] < -1]
        
        if len(high_right_skew) > 0:
            print("\\n🔧 For RIGHT-SKEWED variables, consider:")
            for _, row in high_right_skew.iterrows():
                print(f"   • {row['Column']} (skew={row['Skewness']:.3f}):")
                print(f"      - Log transformation: log(x)")
                print(f"      - Square root transformation: sqrt(x)")
                print(f"      - Box-Cox transformation")
        
        if len(high_left_skew) > 0:
            print("\\n🔧 For LEFT-SKEWED variables, consider:")
            for _, row in high_left_skew.iterrows():
                print(f"   • {row['Column']} (skew={row['Skewness']:.3f}):")
                print(f"      - Reflect then log: log(max(x) - x + 1)")
                print(f"      - Square transformation: x^2")
                print(f"      - Exponential transformation")
        
        # Summary statistics
        print("\\n📋 OVERALL SKEWNESS SUMMARY:")
        print("-" * 60)
        
        total_vars = len(skew_df)
        symmetric_count = len(symmetric_vars)
        mod_skewed_count = len(mod_skewed_vars)
        high_skewed_count = len(high_skewed_vars)
        
        print(f"   Total numeric variables analyzed: {total_vars}")
        print(f"   Approximately symmetric (|skew| < 0.5): {symmetric_count} ({symmetric_count/total_vars*100:.1f}%)")
        print(f"   Moderately skewed (0.5 ≤ |skew| < 1.0): {mod_skewed_count} ({mod_skewed_count/total_vars*100:.1f}%)")
        print(f"   Highly skewed (|skew| ≥ 1.0): {high_skewed_count} ({high_skewed_count/total_vars*100:.1f}%)")
        
        avg_abs_skew = skew_df['Skewness'].abs().mean()
        print(f"   Average absolute skewness: {avg_abs_skew:.3f}")
        
        most_skewed = skew_df.loc[skew_df['Skewness'].abs().idxmax()]
        print(f"   Most skewed variable: {most_skewed['Column']} (skew = {most_skewed['Skewness']:.3f})")

print("\\n" + "="*60)
print("✅ Skewness statistics analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return SkewnessStatisticsAnalysis