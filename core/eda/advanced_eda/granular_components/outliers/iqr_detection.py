"""IQR Outlier Detection Component.

Provides IQR-based outlier detection with boxplots and detailed analysis.
"""


class IQROutlierDetection:
    """Detect outliers using the Interquartile Range (IQR) method."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "iqr_outlier_detection",
            "display_name": "IQR Outlier Detection",
            "description": "Identify outliers using IQR method with boxplot visualization", 
            "category": "outlier_detection",
            "complexity": "intermediate",
            "tags": ["outliers", "IQR", "boxplot", "interquartile"],
            "estimated_runtime": "2-5 seconds",
            "icon": "📦"
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
        """Generate code for IQR outlier detection."""
        
        return '''
# ===== IQR OUTLIER DETECTION =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("📦 IQR OUTLIER DETECTION ANALYSIS")
print("="*60)

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\\n📊 Found {len(numeric_cols)} numeric columns for outlier detection")

if len(numeric_cols) == 0:
    print("❌ No numeric columns found for outlier detection")
else:
    # Remove columns with all NaN or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and not df[col].isna().all():
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns for IQR analysis")
    
    if len(valid_cols) == 0:
        print("❌ No valid columns after removing constant/all-NaN columns")
    else:
        # Store outlier information
        outlier_summary = []
        all_outliers_data = {}
        
        # IQR Analysis for each column
        print("\\n🔍 IQR OUTLIER ANALYSIS BY COLUMN:")
        print("-" * 60)
        
        for col in valid_cols:
            print(f"\\n📊 Column: {col}")
            
            # Calculate IQR statistics
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            total_count = df[col].count()  # Non-null count
            outlier_percentage = (outlier_count / total_count * 100) if total_count > 0 else 0
            
            print(f"   • Q1 (25th percentile): {Q1:.3f}")
            print(f"   • Q3 (75th percentile): {Q3:.3f}")
            print(f"   • IQR: {IQR:.3f}")
            print(f"   • Lower bound: {lower_bound:.3f}")
            print(f"   • Upper bound: {upper_bound:.3f}")
            print(f"   • Outliers detected: {outlier_count:,} ({outlier_percentage:.1f}%)")
            
            if outlier_count > 0:
                # Outlier statistics
                min_outlier = outliers.min()
                max_outlier = outliers.max()
                extreme_low = (outliers < lower_bound).sum()
                extreme_high = (outliers > upper_bound).sum()
                
                print(f"   • Extreme low outliers: {extreme_low} (< {lower_bound:.3f})")
                print(f"   • Extreme high outliers: {extreme_high} (> {upper_bound:.3f})")
                print(f"   • Range of outliers: [{min_outlier:.3f}, {max_outlier:.3f}]")
                
                # Store for summary
                outlier_summary.append({
                    'column': col,
                    'count': outlier_count,
                    'percentage': outlier_percentage,
                    'extreme_low': extreme_low,
                    'extreme_high': extreme_high,
                    'min_outlier': min_outlier,
                    'max_outlier': max_outlier
                })
                
                all_outliers_data[col] = outliers
            else:
                print(f"   ✅ No outliers detected")
        
        # Create boxplot visualizations
        if len(valid_cols) > 0:
            print(f"\\n📈 Generating boxplot visualizations...")
            
            # Calculate grid size for boxplots
            n_cols_plot = min(4, len(valid_cols))
            n_rows_plot = (len(valid_cols) + n_cols_plot - 1) // n_cols_plot
            
            fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(5*n_cols_plot, 4*n_rows_plot))
            fig.suptitle('📦 IQR Outlier Detection - Boxplots', fontsize=16, fontweight='bold')
            
            if len(valid_cols) == 1:
                axes = [axes]
            elif n_rows_plot == 1 or n_cols_plot == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for idx, col in enumerate(valid_cols):
                ax = axes[idx]
                
                # Create boxplot
                box_data = df[col].dropna()
                bp = ax.boxplot(box_data, patch_artist=True, notch=True)
                
                # Customize boxplot
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][0].set_alpha(0.7)
                bp['medians'][0].set_color('red')
                bp['medians'][0].set_linewidth(2)
                
                # Calculate outlier information for this column
                Q1 = box_data.quantile(0.25)
                Q3 = box_data.quantile(0.75) 
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = len(box_data[(box_data < lower_bound) | (box_data > upper_bound)])
                
                # Set title with outlier count
                title_color = 'red' if outlier_count > len(box_data) * 0.05 else 'green' if outlier_count == 0 else 'orange'
                ax.set_title(f'{col}\\n{outlier_count} outliers ({outlier_count/len(box_data)*100:.1f}%)', 
                           fontweight='bold', color=title_color)
                
                ax.set_ylabel('Value', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add IQR boundaries as horizontal lines
                ax.axhline(y=lower_bound, color='red', linestyle='--', alpha=0.7, label=f'Lower bound: {lower_bound:.2f}')
                ax.axhline(y=upper_bound, color='red', linestyle='--', alpha=0.7, label=f'Upper bound: {upper_bound:.2f}')
                ax.legend(fontsize=8)
            
            # Hide unused subplots
            for idx in range(len(valid_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        
        # Overall summary
        print("\\n" + "="*60)
        print("📊 IQR OUTLIER DETECTION SUMMARY")
        print("="*60)
        
        print(f"\\n📈 Overall Statistics:")
        total_outliers = sum([summary['count'] for summary in outlier_summary])
        total_data_points = len(df) * len(valid_cols)  # Total possible data points
        
        print(f"   • Columns analyzed: {len(valid_cols)}")
        print(f"   • Total outliers detected: {total_outliers:,}")
        print(f"   • Overall outlier rate: {(total_outliers / total_data_points * 100):.2f}%")
        print(f"   • Columns with outliers: {len(outlier_summary)}")
        print(f"   • Clean columns: {len(valid_cols) - len(outlier_summary)}")
        
        if outlier_summary:
            print(f"\\n🔍 COLUMNS WITH MOST OUTLIERS:")
            print("-" * 40)
            
            # Sort by outlier count
            sorted_outliers = sorted(outlier_summary, key=lambda x: x['count'], reverse=True)
            
            for i, summary in enumerate(sorted_outliers[:10], 1):  # Top 10
                col = summary['column']
                count = summary['count']
                percentage = summary['percentage']
                extreme_low = summary['extreme_low']
                extreme_high = summary['extreme_high']
                
                print(f"   {i:2d}. {col:<20} | {count:>6,} ({percentage:>5.1f}%) | Low: {extreme_low}, High: {extreme_high}")
        
        print(f"\\n💡 OUTLIER INTERPRETATION:")
        print(f"   📦 IQR Method: Outliers are values outside Q1 - 1.5×IQR or Q3 + 1.5×IQR")
        print(f"   🟢 Green titles: No outliers detected")
        print(f"   🟠 Orange titles: Moderate outliers (< 5% of data)")  
        print(f"   🔴 Red titles: High outlier percentage (≥ 5% of data)")
        print(f"   📊 Box elements: Box = IQR, Red line = Median, Whiskers = 1.5×IQR")

print("\\n" + "="*60)
print("✅ IQR outlier detection complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return IQROutlierDetection