"""Z-Score Outlier Detection Component.

Provides Z-score based outlier detection with statistical analysis.
"""


class ZScoreOutlierDetection:
    """Detect outliers using Z-score method with statistical thresholds."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "zscore_outlier_detection",
            "display_name": "Z-Score Outlier Detection",
            "description": "Identify outliers using Z-score method with customizable thresholds",
            "category": "outlier_detection",
            "complexity": "intermediate",
            "tags": ["outliers", "z-score", "statistical", "standardized"],
            "estimated_runtime": "2-4 seconds",
            "icon": "📈"
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview=None):
        """Check if analysis can be performed on the data."""
        if not data_preview:
            return True
        numeric_cols = data_preview.get('numeric_columns', [])
        return len(numeric_cols) > 0
    
    @staticmethod
    def generate_code(data_preview=None):
        """Generate code for Z-score outlier detection."""
        
        return '''
# ===== Z-SCORE OUTLIER DETECTION =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("="*60)
print("📈 Z-SCORE OUTLIER DETECTION ANALYSIS")
print("="*60)

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\\n📊 Found {len(numeric_cols)} numeric columns for Z-score analysis")

if len(numeric_cols) == 0:
    print("❌ No numeric columns found for outlier detection")
else:
    # Remove columns with all NaN or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and not df[col].isna().all():
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns for Z-score analysis")
    
    if len(valid_cols) == 0:
        print("❌ No valid columns after removing constant/all-NaN columns")
    else:
        # Define different Z-score thresholds
        thresholds = {
            'Conservative (|z| > 3)': 3.0,
            'Moderate (|z| > 2.5)': 2.5,
            'Liberal (|z| > 2)': 2.0
        }
        
        # Store results for each threshold
        results_by_threshold = {}
        
        print("\\n🔍 Z-SCORE ANALYSIS BY THRESHOLD:")
        print("-" * 60)
        
        for threshold_name, threshold_value in thresholds.items():
            print(f"\\n📊 {threshold_name}:")
            print("-" * 40)
            
            outlier_summary = []
            
            for col in valid_cols:
                # Calculate Z-scores
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                    
                z_scores = np.abs(stats.zscore(col_data))
                outliers = col_data[z_scores > threshold_value]
                outlier_count = len(outliers)
                total_count = len(col_data)
                outlier_percentage = (outlier_count / total_count * 100) if total_count > 0 else 0
                
                if outlier_count > 0:
                    print(f"   {col:<25} | {outlier_count:>4} ({outlier_percentage:>5.1f}%)")
                    outlier_summary.append({
                        'column': col,
                        'count': outlier_count,
                        'percentage': outlier_percentage,
                        'outliers': outliers,
                        'z_scores': z_scores[z_scores > threshold_value]
                    })
                else:
                    print(f"   {col:<25} | {outlier_count:>4} ({outlier_percentage:>5.1f}%)")
            
            results_by_threshold[threshold_name] = outlier_summary
        
        # Detailed analysis for the moderate threshold (2.5)
        print(f"\\n{'='*60}")
        print(f"📊 DETAILED ANALYSIS (Moderate Threshold |z| > 2.5)")
        print('='*60)
        
        moderate_results = results_by_threshold.get('Moderate (|z| > 2.5)', [])
        
        for result in moderate_results:
            col = result['column']
            outliers = result['outliers']
            outlier_count = result['count']
            
            if outlier_count > 0:
                print(f"\\n📈 Column: {col}")
                
                # Basic statistics
                col_data = df[col].dropna()
                mean_val = col_data.mean()
                std_val = col_data.std()
                
                print(f"   • Mean: {mean_val:.3f}")
                print(f"   • Standard deviation: {std_val:.3f}")
                print(f"   • Outliers: {outlier_count} ({outlier_count/len(col_data)*100:.1f}%)")
                
                # Outlier statistics
                min_outlier = outliers.min()
                max_outlier = outliers.max()
                mean_outlier = outliers.mean()
                
                print(f"   • Outlier range: [{min_outlier:.3f}, {max_outlier:.3f}]")
                print(f"   • Mean of outliers: {mean_outlier:.3f}")
                
                # Z-score analysis
                max_z = np.abs(stats.zscore(col_data)).max()
                print(f"   • Maximum |Z-score|: {max_z:.3f}")
                
                # Direction of outliers
                positive_outliers = (outliers > mean_val + 2.5 * std_val).sum()
                negative_outliers = (outliers < mean_val - 2.5 * std_val).sum()
                
                print(f"   • Positive outliers (high): {positive_outliers}")
                print(f"   • Negative outliers (low): {negative_outliers}")
        
        # Create visualization
        if len(valid_cols) > 0:
            print(f"\\n📊 Generating Z-score visualization...")
            
            # Select up to 9 columns for visualization
            viz_cols = valid_cols[:9]
            n_cols_plot = min(3, len(viz_cols))
            n_rows_plot = (len(viz_cols) + n_cols_plot - 1) // n_cols_plot
            
            fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6*n_cols_plot, 4*n_rows_plot))
            fig.suptitle('📈 Z-Score Distribution Analysis', fontsize=16, fontweight='bold')
            
            if len(viz_cols) == 1:
                axes = [axes]
            elif n_rows_plot == 1 or n_cols_plot == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for idx, col in enumerate(viz_cols):
                ax = axes[idx]
                
                # Calculate Z-scores
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{col}\\n(No data)')
                    continue
                
                z_scores = stats.zscore(col_data)
                
                # Create histogram of Z-scores
                ax.hist(z_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                
                # Add threshold lines
                for threshold in [-3, -2.5, -2, 2, 2.5, 3]:
                    color = 'red' if abs(threshold) == 3 else 'orange' if abs(threshold) == 2.5 else 'yellow'
                    ax.axvline(threshold, color=color, linestyle='--', alpha=0.8)
                
                # Count outliers for different thresholds
                outliers_3 = (np.abs(z_scores) > 3).sum()
                outliers_25 = (np.abs(z_scores) > 2.5).sum()
                outliers_2 = (np.abs(z_scores) > 2).sum()
                
                ax.set_title(f'{col}\\n|z|>3: {outliers_3}, |z|>2.5: {outliers_25}, |z|>2: {outliers_2}', 
                           fontweight='bold')
                ax.set_xlabel('Z-Score', fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(viz_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        
        # Comparative summary
        print("\\n" + "="*60)
        print("📊 Z-SCORE THRESHOLD COMPARISON")
        print("="*60)
        
        print(f"\\n{'Threshold':<20} | {'Total Outliers':<15} | {'Avg per Column':<15} | {'% of Data':<10}")
        print("-" * 70)
        
        total_data_points = sum([len(df[col].dropna()) for col in valid_cols])
        
        for threshold_name, results in results_by_threshold.items():
            total_outliers = sum([r['count'] for r in results])
            avg_per_column = total_outliers / len(valid_cols) if valid_cols else 0
            percentage = (total_outliers / total_data_points * 100) if total_data_points > 0 else 0
            
            print(f"{threshold_name:<20} | {total_outliers:<15,} | {avg_per_column:<15.1f} | {percentage:<10.2f}%")
        
        print(f"\\n💡 Z-SCORE INTERPRETATION:")
        print(f"   📈 Z-score measures how many standard deviations a value is from the mean")
        print(f"   🟡 |z| > 2: Moderate outlier (outside ~95% of normal distribution)")
        print(f"   🟠 |z| > 2.5: Strong outlier (outside ~99% of normal distribution)")
        print(f"   🔴 |z| > 3: Extreme outlier (outside ~99.7% of normal distribution)")
        print(f"   ⚠️  Z-score assumes normal distribution - check distribution first!")

print("\\n" + "="*60)
print("✅ Z-score outlier detection complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return ZScoreOutlierDetection