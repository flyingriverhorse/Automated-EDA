"""Box Plots Analysis Component.

Focused component specifically for box plot visualizations.
"""
from typing import Dict, Any


class BoxPlotsAnalysis:
    """Focused component for box plot visualization"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "box_plots",
            "display_name": "Box Plots",
            "description": "Create comprehensive box plots for outlier detection and distribution comparison",
            "category": "univariate",
            "subcategory": "distribution_plots",
            "complexity": "basic",
            "required_data_types": ["numeric"],
            "estimated_runtime": "8-12 seconds",
            "icon": "📦",
            "tags": ["visualization", "boxplot", "outliers", "quartiles"]
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
        """Generate focused box plots analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("📦 BOX PLOTS ANALYSIS")
print("="*60)
print("Comprehensive box plot visualizations for distribution analysis")
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
    print(f"📦 CREATING BOX PLOTS FOR {len(numeric_cols)} NUMERIC COLUMNS")
    print()
    
    # 1. INDIVIDUAL BOX PLOTS
    print("📊 1. INDIVIDUAL BOX PLOTS WITH STATISTICS")
    
    n_cols = len(numeric_cols)
    n_plot_cols = min(4, n_cols)
    n_rows = (n_cols + 3) // 4
    
    fig1, axes1 = plt.subplots(n_rows, n_plot_cols, figsize=(5*n_plot_cols, 4*n_rows))
    fig1.suptitle('Individual Box Plots with Statistics', fontsize=16, fontweight='bold')
    
    if n_cols == 1:
        axes1 = [axes1]
    elif n_rows == 1:
        axes1 = axes1.reshape(1, -1) if n_plot_cols > 1 else [axes1]
    
    outlier_summary = {}
    
    for i, col in enumerate(numeric_cols):
        row = i // 4
        col_idx = i % 4
        
        if n_rows > 1:
            ax = axes1[row, col_idx] if n_plot_cols > 1 else axes1[row]
        else:
            ax = axes1[col_idx] if n_plot_cols > 1 else axes1[0]
        
        col_data = df[col].dropna()
        
        if len(col_data) > 0:
            # Create box plot
            bp = ax.boxplot([col_data], labels=[''], patch_artist=True, 
                          showmeans=True, meanline=True)
            
            # Customize box plot appearance
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            bp['medians'][0].set_color('red')
            bp['medians'][0].set_linewidth(2)
            bp['means'][0].set_color('green')
            bp['means'][0].set_linewidth(2)
            
            # Calculate statistics
            q1 = np.percentile(col_data, 25)
            q2 = np.percentile(col_data, 50)  # median
            q3 = np.percentile(col_data, 75)
            iqr = q3 - q1
            mean_val = col_data.mean()
            
            # Calculate outliers
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            outliers = col_data[(col_data < lower_fence) | (col_data > upper_fence)]
            outlier_summary[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(col_data)) * 100,
                'lower_fence': lower_fence,
                'upper_fence': upper_fence
            }
            
            # Add statistics text
            stats_text = f'n = {len(col_data)}\\nQ1 = {q1:.2f}\\nQ2 = {q2:.2f}\\nQ3 = {q3:.2f}\\nIQR = {iqr:.2f}\\nMean = {mean_val:.2f}\\nOutliers = {len(outliers)} ({len(outliers)/len(col_data)*100:.1f}%)'
            
            ax.text(1.05, 0.5, stats_text, transform=ax.transAxes, 
                   verticalalignment='center', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_title(col)
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{col}\\nNo valid data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(col)
    
    # Hide unused subplots
    total_subplots = n_rows * n_plot_cols
    for i in range(n_cols, total_subplots):
        row = i // n_plot_cols
        col_idx = i % n_plot_cols
        if n_rows > 1:
            axes1[row, col_idx].set_visible(False)
        else:
            axes1[col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # 2. COMPARATIVE BOX PLOTS (SIDE-BY-SIDE)
    print("\\n📊 2. COMPARATIVE BOX PLOTS (STANDARDIZED)")
    
    # Create standardized data for comparison
    valid_data = []
    valid_labels = []
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 1:
            # Standardize for comparison
            standardized = (col_data - col_data.mean()) / col_data.std()
            valid_data.append(standardized)
            valid_labels.append(col[:12] + '...' if len(col) > 12 else col)
    
    if len(valid_data) > 0:
        plt.figure(figsize=(max(10, len(valid_data) * 0.8), 8))
        
        bp = plt.boxplot(valid_data, labels=valid_labels, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # Color code boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(valid_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize appearance
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)
        
        for mean in bp['means']:
            mean.set_color('green')
            mean.set_linewidth(2)
        
        plt.title('Comparative Box Plots (Standardized)', fontsize=14, fontweight='bold')
        plt.ylabel('Standardized Value')
        plt.xlabel('Variables')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.figtext(0.02, 0.02, 'Red line: Median, Green line: Mean', 
                   fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
    
    # 3. OUTLIER ANALYSIS BOX PLOTS
    print("\\n📊 3. OUTLIER ANALYSIS BOX PLOTS")
    
    # Focus on variables with outliers
    variables_with_outliers = [(col, info) for col, info in outlier_summary.items() 
                              if info['count'] > 0]
    
    if len(variables_with_outliers) > 0:
        n_outlier_vars = min(6, len(variables_with_outliers))  # Limit to 6 variables
        n_outlier_cols = min(3, n_outlier_vars)
        n_outlier_rows = (n_outlier_vars + 2) // 3
        
        fig2, axes2 = plt.subplots(n_outlier_rows, n_outlier_cols, 
                                  figsize=(6*n_outlier_cols, 5*n_outlier_rows))
        fig2.suptitle('Detailed Outlier Analysis', fontsize=16, fontweight='bold')
        
        if n_outlier_vars == 1:
            axes2 = [axes2]
        elif n_outlier_rows == 1:
            axes2 = axes2.reshape(1, -1) if n_outlier_cols > 1 else [axes2]
        
        for i, (col, outlier_info) in enumerate(variables_with_outliers[:n_outlier_vars]):
            row = i // n_outlier_cols
            col_idx = i % n_outlier_cols
            
            if n_outlier_rows > 1:
                ax = axes2[row, col_idx] if n_outlier_cols > 1 else axes2[row]
            else:
                ax = axes2[col_idx] if n_outlier_cols > 1 else axes2[0]
            
            col_data = df[col].dropna()
            
            # Create detailed box plot
            bp = ax.boxplot([col_data], labels=[''], patch_artist=True,
                          showmeans=True, meanline=True, showfliers=True)
            
            # Highlight outliers in red
            bp['boxes'][0].set_facecolor('lightcoral')
            bp['boxes'][0].set_alpha(0.7)
            bp['fliers'][0].set_markerfacecolor('red')
            bp['fliers'][0].set_markeredgecolor('darkred')
            bp['fliers'][0].set_markersize(6)
            
            # Add fence lines
            ax.axhline(outlier_info['lower_fence'], color='orange', linestyle='--', 
                      alpha=0.7, label=f"Lower fence: {outlier_info['lower_fence']:.2f}")
            ax.axhline(outlier_info['upper_fence'], color='orange', linestyle='--', 
                      alpha=0.7, label=f"Upper fence: {outlier_info['upper_fence']:.2f}")
            
            ax.set_title(f'{col}\\n{outlier_info["count"]} outliers ({outlier_info["percentage"]:.1f}%)')
            ax.set_ylabel('Value')
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        total_outlier_subplots = n_outlier_rows * n_outlier_cols
        for i in range(n_outlier_vars, total_outlier_subplots):
            row = i // n_outlier_cols
            col_idx = i % n_outlier_cols
            if n_outlier_rows > 1:
                axes2[row, col_idx].set_visible(False)
            else:
                axes2[col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Outlier summary table
        print("\\n📋 OUTLIER SUMMARY TABLE:")
        print("-" * 80)
        outlier_df = pd.DataFrame([
            {
                'Variable': col,
                'Total_Count': len(df[col].dropna()),
                'Outlier_Count': info['count'],
                'Outlier_Percentage': f"{info['percentage']:.1f}%",
                'Lower_Fence': f"{info['lower_fence']:.2f}",
                'Upper_Fence': f"{info['upper_fence']:.2f}"
            }
            for col, info in variables_with_outliers
        ])
        print(outlier_df.to_string(index=False))
        
    else:
        print("   ✅ No outliers detected in any variables using IQR method")
    
    # 4. BOX PLOTS WITH VIOLIN OVERLAY
    print("\\n📊 4. BOX PLOTS WITH VIOLIN OVERLAY (DISTRIBUTION SHAPE)")
    
    # Select top 4 variables for detailed shape analysis
    detail_vars = numeric_cols[:min(4, len(numeric_cols))]
    
    if len(detail_vars) > 0:
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
        fig3.suptitle('Box Plots with Distribution Shape (Violin Overlay)', 
                     fontsize=16, fontweight='bold')
        
        axes3 = axes3.flatten() if len(detail_vars) > 1 else [axes3]
        
        for i, col in enumerate(detail_vars):
            if i < len(axes3):
                ax = axes3[i]
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    # Create violin plot
                    parts = ax.violinplot([col_data], positions=[1], showmeans=True, 
                                        showmedians=True, showextrema=False)
                    
                    # Customize violin plot
                    parts['bodies'][0].set_facecolor('lightblue')
                    parts['bodies'][0].set_alpha(0.6)
                    
                    # Overlay box plot
                    bp = ax.boxplot([col_data], positions=[1], patch_artist=True, 
                                  widths=0.3, showfliers=True)
                    bp['boxes'][0].set_facecolor('white')
                    bp['boxes'][0].set_alpha(0.8)
                    bp['boxes'][0].set_edgecolor('black')
                    bp['boxes'][0].set_linewidth(2)
                    
                    # Add statistical annotations
                    q1, median, q3 = np.percentile(col_data, [25, 50, 75])
                    mean_val = col_data.mean()
                    
                    ax.text(1.3, median, f'Median: {median:.2f}', 
                           verticalalignment='center', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                    ax.text(1.3, mean_val, f'Mean: {mean_val:.2f}', 
                           verticalalignment='center', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
                    
                    ax.set_title(f'{col}\\nShape + Quartiles Analysis')
                    ax.set_ylabel('Value')
                    ax.set_xticks([])
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'{col}\\nNo valid data', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(col)
        
        # Hide unused subplots
        for i in range(len(detail_vars), len(axes3)):
            axes3[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    # 5. BOX PLOT COMPARISON SUMMARY
    print("\\n📊 5. BOX PLOT ANALYSIS SUMMARY")
    print("-" * 60)
    
    summary_stats = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            q1, q2, q3 = np.percentile(col_data, [25, 50, 75])
            iqr = q3 - q1
            mean_val = col_data.mean()
            
            # Measure of asymmetry (mean vs median)
            skew_indicator = abs(mean_val - q2) / col_data.std() if col_data.std() > 0 else 0
            
            summary_stats.append({
                'Variable': col,
                'Q1': q1,
                'Median': q2,
                'Q3': q3,
                'IQR': iqr,
                'Mean': mean_val,
                'Skew_Indicator': skew_indicator,
                'Outliers': outlier_summary.get(col, {}).get('count', 0)
            })
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        
        print("\\nBox Plot Statistics Summary:")
        print(summary_df.round(3).to_string(index=False))
        
        print("\\n📊 Key Insights:")
        
        # Variables with highest IQR (most spread)
        highest_iqr = summary_df.loc[summary_df['IQR'].idxmax()]
        print(f"   • Highest variability: {highest_iqr['Variable']} (IQR = {highest_iqr['IQR']:.2f})")
        
        # Variables with most outliers
        most_outliers = summary_df.loc[summary_df['Outliers'].idxmax()]
        if most_outliers['Outliers'] > 0:
            print(f"   • Most outliers: {most_outliers['Variable']} ({most_outliers['Outliers']} outliers)")
        
        # Variables with highest asymmetry
        most_skewed = summary_df.loc[summary_df['Skew_Indicator'].idxmax()]
        print(f"   • Most asymmetric: {most_skewed['Variable']} (skew indicator = {most_skewed['Skew_Indicator']:.2f})")
        
        # Overall outlier statistics
        total_outliers = summary_df['Outliers'].sum()
        total_observations = sum(len(df[col].dropna()) for col in numeric_cols)
        print(f"   • Total outliers detected: {total_outliers} ({total_outliers/total_observations*100:.2f}% of all observations)")

print("\\n" + "="*60)
print("✅ Box plots analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return BoxPlotsAnalysis