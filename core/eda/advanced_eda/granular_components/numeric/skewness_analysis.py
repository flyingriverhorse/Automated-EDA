"""Skewness Analysis Component.

Focused analysis component for analyzing skewness of numeric variables with visualizations.
"""
from typing import Dict, Any


class SkewnessAnalysis:
    """Focused component for skewness analysis with graphs"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "skewness_analysis",
            "display_name": "Skewness Analysis",
            "description": "Analyze skewness of distributions with visual representations and interpretations",
            "category": "univariate",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-20 seconds",
            "icon": "trending-up",
            "tags": ["skewness", "distribution", "statistics", "visualization"]
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
        """Generate focused skewness analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=== SKEWNESS ANALYSIS ===")
print("Comprehensive analysis of distribution skewness with visualizations")
print()

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    print("❌ NO NUMERIC COLUMNS FOUND")
    print("   This analysis requires numeric data.")
else:
    print(f"📊 ANALYZING SKEWNESS FOR {len(numeric_cols)} NUMERIC COLUMNS")
    print()
    
    # Calculate skewness for all numeric columns
    skewness_data = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 1:
            skew_value = stats.skew(col_data)
            skewness_data.append({
                'Column': col,
                'Skewness': round(skew_value, 4),
                'Count': len(col_data),
                'Missing': df[col].isnull().sum()
            })
    
    # Create skewness summary table
    skew_df = pd.DataFrame(skewness_data)
    
    # Add interpretation
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
            return "Symmetric"
    
    skew_df['Interpretation'] = skew_df['Skewness'].apply(interpret_skewness)
    skew_df['Direction'] = skew_df['Skewness'].apply(skew_direction)
    
    # Sort by absolute skewness value
    skew_df = skew_df.reindex(skew_df['Skewness'].abs().sort_values(ascending=False).index)
    
    print("📈 SKEWNESS SUMMARY TABLE")
    print(skew_df.to_string(index=False))
    print()
    
    # Skewness interpretation guide
    print("📚 SKEWNESS INTERPRETATION GUIDE")
    print("   📊 Skewness Value Ranges:")
    print("      • -0.5 to 0.5: Approximately symmetric")
    print("      • 0.5 to 1.0 or -1.0 to -0.5: Moderately skewed")
    print("      • > 1.0 or < -1.0: Highly skewed")
    print()
    print("   📈 Direction Meaning:")
    print("      • Positive (Right) skew: Long tail extends to the right")
    print("      • Negative (Left) skew: Long tail extends to the left")
    print("      • Zero skew: Symmetric distribution")
    print()
    
    # Create comprehensive visualizations
    n_cols = len(numeric_cols)
    if n_cols > 0:
        # Calculate subplot layout
        n_rows = (n_cols + 2) // 3  # Max 3 columns per row
        n_plot_cols = min(3, n_cols)
        
        fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(5*n_plot_cols, 4*n_rows))
        fig.suptitle('Skewness Analysis: Distribution Shapes', fontsize=16, fontweight='bold')
        
        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            row = i // 3
            col_idx = i % 3
            
            if n_rows > 1:
                ax = axes[row, col_idx] if n_plot_cols > 1 else axes[row]
            else:
                ax = axes[col_idx] if n_plot_cols > 1 else axes[0]
            
            col_data = df[col].dropna()
            
            if len(col_data) > 1:
                # Plot histogram with density curve
                ax.hist(col_data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
                
                # Add density curve
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(col_data)
                    x_range = np.linspace(col_data.min(), col_data.max(), 200)
                    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Density')
                except:
                    pass
                
                # Add mean and median lines
                mean_val = col_data.mean()
                median_val = col_data.median()
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
                
                # Get skewness for this column
                skew_val = stats.skew(col_data)
                
                ax.set_title(f'{col}\\nSkewness: {skew_val:.3f} ({skew_direction(skew_val)})')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend(fontsize='small')
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        if n_cols < n_rows * n_plot_cols:
            for i in range(n_cols, n_rows * n_plot_cols):
                row = i // 3
                col_idx = i % 3
                if n_rows > 1:
                    fig.delaxes(axes[row, col_idx] if n_plot_cols > 1 else axes[row])
                else:
                    if n_plot_cols > 1:
                        fig.delaxes(axes[col_idx])
        
        plt.tight_layout()
        plt.show()
        print()
    
    # Detailed analysis for most skewed columns
    most_skewed = skew_df.head(min(3, len(skew_df)))
    
    if len(most_skewed) > 0:
        print("🎯 DETAILED ANALYSIS OF MOST SKEWED COLUMNS")
        
        for _, row in most_skewed.iterrows():
            col_name = row['Column']
            skew_val = row['Skewness']
            
            print(f"\\n   📊 {col_name.upper()} (Skewness: {skew_val})")
            
            col_data = df[col_name].dropna()
            mean_val = col_data.mean()
            median_val = col_data.median()
            
            print(f"      • Mean: {mean_val:.3f}")
            print(f"      • Median: {median_val:.3f}")
            print(f"      • Mean - Median: {mean_val - median_val:.3f}")
            
            # Explain the skewness
            if skew_val > 0.5:
                print(f"      • Right-skewed: Most values cluster on the left, with a long tail extending right")
                print(f"      • Mean > Median indicates the tail is pulling the average higher")
                print(f"      • Common in: Income, prices, reaction times")
            elif skew_val < -0.5:
                print(f"      • Left-skewed: Most values cluster on the right, with a long tail extending left")
                print(f"      • Mean < Median indicates the tail is pulling the average lower") 
                print(f"      • Common in: Test scores (ceiling effect), age at retirement")
            else:
                print(f"      • Approximately symmetric: Values are balanced around the center")
                print(f"      • Mean ≈ Median indicates balanced distribution")
            
            # Percentile analysis
            percentiles = col_data.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            print(f"      • Percentiles: 10%={percentiles.iloc[0]:.2f}, 25%={percentiles.iloc[1]:.2f}, 75%={percentiles.iloc[3]:.2f}, 90%={percentiles.iloc[4]:.2f}")
    
    # Summary statistics by skewness category
    symmetric_cols = skew_df[abs(skew_df['Skewness']) < 0.5]['Column'].tolist()
    moderate_skew_cols = skew_df[(abs(skew_df['Skewness']) >= 0.5) & (abs(skew_df['Skewness']) < 1.0)]['Column'].tolist()
    high_skew_cols = skew_df[abs(skew_df['Skewness']) >= 1.0]['Column'].tolist()
    
    print()
    print("📋 SKEWNESS CATEGORIES SUMMARY")
    print(f"   🟢 Symmetric columns ({len(symmetric_cols)}): {symmetric_cols}")
    print(f"   🟡 Moderately skewed ({len(moderate_skew_cols)}): {moderate_skew_cols}")
    print(f"   🔴 Highly skewed ({len(high_skew_cols)}): {high_skew_cols}")
    
    print()
    print("💡 RECOMMENDATIONS")
    
    if high_skew_cols:
        print("   🔴 For highly skewed columns:")
        print("      • Consider log transformation for right-skewed data")
        print("      • Consider square root transformation for moderate right skew")
        print("      • Consider Box-Cox transformation for general skewness")
        print("      • Use median instead of mean for central tendency")
        print("      • Be cautious with parametric statistical tests")
    
    if moderate_skew_cols:
        print("   🟡 For moderately skewed columns:")
        print("      • May still be acceptable for many analyses")
        print("      • Consider transformation if normality is required")
        print("      • Use robust statistics (median, IQR)")
    
    if symmetric_cols:
        print("   🟢 For symmetric columns:")
        print("      • Good candidates for parametric tests")
        print("      • Mean and median are both representative")
        print("      • Standard statistical methods apply")
    
    print()
    print("   📊 Transformation examples:")
    print("      • Right skew: np.log1p(df['column']) or np.sqrt(df['column'])")
    print("      • Left skew: np.power(df['column'], 2) or np.exp(df['column'])")
    print("      • Test transformations before applying to entire dataset")

print("\\n" + "="*50)
'''