"""Distribution Plots Analysis Component.

Focused analysis component for creating various distribution plots: histograms, density plots, box plots.
"""
from typing import Dict, Any


class DistributionPlotsAnalysis:
    """Focused component for distribution visualization"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "distribution_plots",
            "display_name": "Distribution Plots",
            "description": "Comprehensive distribution visualizations: histograms, density plots, box plots, violin plots",
            "category": "univariate",
            "complexity": "basic",
            "required_data_types": ["numeric"],
            "estimated_runtime": "15-30 seconds",
            "icon": "chart-bar",
            "tags": ["visualization", "distribution", "plots", "histogram"]
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
        """Generate focused distribution plots analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=== DISTRIBUTION PLOTS ANALYSIS ===")
print("Comprehensive visualization of data distributions")
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
    print(f"📊 CREATING DISTRIBUTION PLOTS FOR {len(numeric_cols)} NUMERIC COLUMNS")
    print()
    
    # 1. HISTOGRAMS WITH STATISTICS
    print("📈 1. HISTOGRAMS WITH STATISTICAL OVERLAYS")
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    n_plot_cols = min(3, n_cols)
    
    fig1, axes1 = plt.subplots(n_rows, n_plot_cols, figsize=(6*n_plot_cols, 4*n_rows))
    fig1.suptitle('Distribution Histograms with Statistics', fontsize=16, fontweight='bold')
    
    if n_cols == 1:
        axes1 = [axes1]
    elif n_rows == 1:
        axes1 = axes1.reshape(1, -1) if n_cols > 1 else [axes1]
    
    for i, col in enumerate(numeric_cols):
        row = i // 3
        col_idx = i % 3
        
        if n_rows > 1:
            ax = axes1[row, col_idx] if n_plot_cols > 1 else axes1[row]
        else:
            ax = axes1[col_idx] if n_plot_cols > 1 else axes1[0]
        
        col_data = df[col].dropna()
        
        if len(col_data) > 0:
            # Plot histogram with statistical overlays
            n_bins = min(50, max(10, int(np.sqrt(len(col_data)))))
            
            # Create histogram
            counts, bins, patches = ax.hist(col_data, bins=n_bins, alpha=0.7, 
                                          color='skyblue', edgecolor='black', 
                                          density=True, label='Data')
            
            # Add KDE if enough data
            if len(col_data) > 10:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(col_data)
                x_range = np.linspace(col_data.min(), col_data.max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
            
            # Add mean and median lines
            mean_val = col_data.mean()
            median_val = col_data.median()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='-', alpha=0.8, label=f'Median: {median_val:.2f}')
            
            # Add normal curve overlay
            x_range = np.linspace(col_data.min(), col_data.max(), 200)
            normal_curve = stats.norm.pdf(x_range, mean_val, col_data.std())
            ax.plot(x_range, normal_curve, 'purple', linewidth=2, alpha=0.7, 
                   label='Normal fit')
            
            ax.set_title(f'{col}\\n(n={len(col_data)}, μ={mean_val:.2f}, σ={col_data.std():.2f})')
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
                fig1.delaxes(axes1[row, col_idx] if n_plot_cols > 1 else axes1[row])
            else:
                if n_plot_cols > 1:
                    fig1.delaxes(axes1[col_idx])
    
    plt.tight_layout()
    plt.show()
    print()
    
    # 2. DENSITY PLOTS (KDE)
    print("🌊 2. KERNEL DENSITY ESTIMATION (KDE) PLOTS")
    
    fig2, axes2 = plt.subplots(n_rows, n_plot_cols, figsize=(6*n_plot_cols, 4*n_rows))
    fig2.suptitle('Kernel Density Estimation Plots', fontsize=16, fontweight='bold')
    
    if n_cols == 1:
        axes2 = [axes2]
    elif n_rows == 1:
        axes2 = axes2.reshape(1, -1) if n_cols > 1 else [axes2]
    
    for i, col in enumerate(numeric_cols):
        row = i // 3
        col_idx = i % 3
        
        if n_rows > 1:
            ax = axes2[row, col_idx] if n_plot_cols > 1 else axes2[row]
        else:
            ax = axes2[col_idx] if n_plot_cols > 1 else axes2[0]
        
        col_data = df[col].dropna()
        
        if len(col_data) > 1:
            # KDE plot with fill
            sns.kdeplot(data=col_data, ax=ax, fill=True, alpha=0.6, color='lightblue')
            
            # Add rug plot (data points on x-axis)
            if len(col_data) <= 1000:  # Only for reasonable sample sizes
                sns.rugplot(data=col_data, ax=ax, alpha=0.5, color='darkblue')
            
            # Add percentile lines
            percentiles = col_data.quantile([0.25, 0.5, 0.75])
            colors = ['orange', 'red', 'purple']
            labels = ['Q1 (25%)', 'Median (50%)', 'Q3 (75%)']
            
            for percentile, color, label in zip(percentiles, colors, labels):
                ax.axvline(percentile, color=color, linestyle=':', linewidth=2, 
                          label=f'{label}: {percentile:.2f}')
            
            ax.set_title(f'{col} - Density Distribution')
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
                fig2.delaxes(axes2[row, col_idx] if n_plot_cols > 1 else axes2[row])
            else:
                if n_plot_cols > 1:
                    fig2.delaxes(axes2[col_idx])
    
    plt.tight_layout()
    plt.show()
    print()
    
    # 3. BOX PLOTS
    print("📦 3. BOX PLOTS (OUTLIER DETECTION)")
    
    fig3, axes3 = plt.subplots(n_rows, n_plot_cols, figsize=(6*n_plot_cols, 4*n_rows))
    fig3.suptitle('Box Plots with Outlier Detection', fontsize=16, fontweight='bold')
    
    if n_cols == 1:
        axes3 = [axes3]
    elif n_rows == 1:
        axes3 = axes3.reshape(1, -1) if n_cols > 1 else [axes3]
    
    for i, col in enumerate(numeric_cols):
        row = i // 3
        col_idx = i % 3
        
        if n_rows > 1:
            ax = axes3[row, col_idx] if n_plot_cols > 1 else axes3[row]
        else:
            ax = axes3[col_idx] if n_plot_cols > 1 else axes3[0]
        
        col_data = df[col].dropna()
        
        if len(col_data) > 0:
            # Box plot
            box_plot = ax.boxplot(col_data, patch_artist=True, 
                                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                                 medianprops=dict(color='red', linewidth=2),
                                 flierprops=dict(marker='o', markerfacecolor='red', 
                                               markersize=4, alpha=0.6))
            
            # Add statistics text
            q1 = col_data.quantile(0.25)
            q2 = col_data.quantile(0.5)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            
            # Count outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            stats_text = f'Q1: {q1:.2f}\\nMedian: {q2:.2f}\\nQ3: {q3:.2f}\\nIQR: {iqr:.2f}\\nOutliers: {len(outliers)}'
            ax.text(1.1, 0.5, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='center', fontsize=9)
            
            ax.set_title(f'{col}\\nBox Plot with Outliers')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, axis='y')
    
    # Remove empty subplots
    if n_cols < n_rows * n_plot_cols:
        for i in range(n_cols, n_rows * n_plot_cols):
            row = i // 3
            col_idx = i % 3
            if n_rows > 1:
                fig3.delaxes(axes3[row, col_idx] if n_plot_cols > 1 else axes3[row])
            else:
                if n_plot_cols > 1:
                    fig3.delaxes(axes3[col_idx])
    
    plt.tight_layout()
    plt.show()
    print()
    
    # 4. VIOLIN PLOTS (if enough unique values)
    violin_cols = [col for col in numeric_cols if df[col].nunique() > 10]
    
    if violin_cols:
        print("🎻 4. VIOLIN PLOTS (DISTRIBUTION SHAPE)")
        
        n_violin_cols = len(violin_cols)
        n_violin_rows = (n_violin_cols + 2) // 3
        n_violin_plot_cols = min(3, n_violin_cols)
        
        fig4, axes4 = plt.subplots(n_violin_rows, n_violin_plot_cols, 
                                  figsize=(6*n_violin_plot_cols, 4*n_violin_rows))
        fig4.suptitle('Violin Plots - Distribution Shape Analysis', fontsize=16, fontweight='bold')
        
        if n_violin_cols == 1:
            axes4 = [axes4]
        elif n_violin_rows == 1:
            axes4 = axes4.reshape(1, -1) if n_violin_cols > 1 else [axes4]
        
        for i, col in enumerate(violin_cols):
            row = i // 3
            col_idx = i % 3
            
            if n_violin_rows > 1:
                ax = axes4[row, col_idx] if n_violin_plot_cols > 1 else axes4[row]
            else:
                ax = axes4[col_idx] if n_violin_plot_cols > 1 else axes4[0]
            
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                # Violin plot
                parts = ax.violinplot([col_data], positions=[1], showmeans=True, 
                                    showmedians=True, showextrema=True)
                
                # Customize colors
                for pc in parts['bodies']:
                    pc.set_facecolor('lightcoral')
                    pc.set_alpha(0.7)
                
                ax.set_title(f'{col}\\nViolin Plot')
                ax.set_ylabel('Value')
                ax.set_xticks([1])
                ax.set_xticklabels([col])
                ax.grid(True, alpha=0.3, axis='y')
        
        # Remove empty subplots
        if n_violin_cols < n_violin_rows * n_violin_plot_cols:
            for i in range(n_violin_cols, n_violin_rows * n_violin_plot_cols):
                row = i // 3
                col_idx = i % 3
                if n_violin_rows > 1:
                    fig4.delaxes(axes4[row, col_idx] if n_violin_plot_cols > 1 else axes4[row])
                else:
                    if n_violin_plot_cols > 1:
                        fig4.delaxes(axes4[col_idx])
        
        plt.tight_layout()
        plt.show()
        print()
    
    # 5. COMPARATIVE ANALYSIS - All distributions in one plot (increased limit)
    if len(numeric_cols) <= 12:  # Increased from 6 to 12 columns
        print("📊 5. COMPARATIVE DISTRIBUTION ANALYSIS")
        
        fig5, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig5.suptitle('Comparative Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Overlapping histograms
        for col in numeric_cols:  # Analyze all available columns (up to 12)
            col_data = df[col].dropna()
            if len(col_data) > 0:
                ax1.hist(col_data, bins=30, alpha=0.5, label=col, density=True)
        ax1.set_title('Overlapping Histograms')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Overlapping KDE plots
        for col in numeric_cols[:6]:
            col_data = df[col].dropna()
            if len(col_data) > 1:
                sns.kdeplot(data=col_data, ax=ax2, label=col, alpha=0.7)
        ax2.set_title('Overlapping Density Plots')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Side-by-side box plots
        box_data = [df[col].dropna().values for col in numeric_cols[:6]]
        ax3.boxplot(box_data, labels=numeric_cols[:6])
        ax3.set_title('Side-by-Side Box Plots')
        ax3.set_ylabel('Value')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Violin plots comparison
        violin_data = [df[col].dropna().values for col in numeric_cols[:6] if df[col].nunique() > 10]
        if violin_data:
            violin_labels = [col for col in numeric_cols[:6] if df[col].nunique() > 10]
            ax4.violinplot(violin_data, positions=range(1, len(violin_data)+1), 
                          showmeans=True, showmedians=True)
            ax4.set_xticks(range(1, len(violin_data)+1))
            ax4.set_xticklabels(violin_labels, rotation=45)
            ax4.set_title('Violin Plots Comparison')
            ax4.set_ylabel('Value')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Not enough\\nvariation for\\nviolin plots', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Violin Plots (Not Available)')
        
        plt.tight_layout()
        plt.show()
        print()
    
    # Summary statistics for distribution analysis
    print("📋 DISTRIBUTION ANALYSIS SUMMARY")
    
    distribution_summary = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            # Calculate distribution characteristics
            skewness = stats.skew(col_data)
            kurtosis = stats.kurtosis(col_data)
            
            # Outlier analysis
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            outliers = col_data[(col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)]
            
            distribution_summary.append({
                'Column': col,
                'Skewness': round(skewness, 3),
                'Kurtosis': round(kurtosis, 3),
                'Outliers': len(outliers),
                'Outlier_%': round(len(outliers)/len(col_data)*100, 1),
                'Range': round(col_data.max() - col_data.min(), 3),
                'IQR': round(iqr, 3)
            })
    
    if distribution_summary:
        summary_df = pd.DataFrame(distribution_summary)
        print(summary_df.to_string(index=False))
        print()
        
        print("💡 INTERPRETATION GUIDE")
        print("   📈 Skewness:")
        print("      • -0.5 to 0.5: Approximately symmetric")
        print("      • > 0.5: Right-skewed (tail extends right)")
        print("      • < -0.5: Left-skewed (tail extends left)")
        print()
        print("   📊 Kurtosis:")
        print("      • > 0: Heavy tails (leptokurtic)")
        print("      • ≈ 0: Normal-like tails (mesokurtic)")  
        print("      • < 0: Light tails (platykurtic)")
        print()
        print("   🎯 Outliers:")
        print("      • < 5%: Few outliers, likely normal variation")
        print("      • 5-10%: Moderate outliers, investigate")
        print("      • > 10%: Many outliers, check data quality")

print("\\n" + "="*50)
'''