"""Histogram Plots Analysis Component.

Focused component specifically for histogram visualizations.
"""
from typing import Dict, Any


class HistogramPlotsAnalysis:
    """Focused component for histogram visualization"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "histogram_plots",
            "display_name": "Histogram Plots",
            "description": "Create comprehensive histogram plots with statistical overlays",
            "category": "univariate",
            "subcategory": "distribution_plots",
            "complexity": "basic",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-15 seconds",
            "icon": "📊",
            "tags": ["visualization", "histogram", "plots", "distribution"]
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
        """Generate focused histogram plots analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("📊 HISTOGRAM PLOTS ANALYSIS")
print("="*60)
print("Comprehensive histogram visualizations with statistical overlays")
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
    print(f"📊 CREATING HISTOGRAM PLOTS FOR {len(numeric_cols)} NUMERIC COLUMNS")
    print()
    
    # 1. BASIC HISTOGRAMS WITH STATISTICS
    print("📈 1. BASIC HISTOGRAMS WITH STATISTICAL OVERLAYS")
    
    n_cols = len(numeric_cols)
    n_plot_cols = min(3, n_cols)
    n_rows = (n_cols + 2) // 3
    
    fig1, axes1 = plt.subplots(n_rows, n_plot_cols, figsize=(6*n_plot_cols, 4*n_rows))
    fig1.suptitle('Basic Histograms with Statistics', fontsize=16, fontweight='bold')
    
    if n_cols == 1:
        axes1 = [axes1]
    elif n_rows == 1:
        axes1 = axes1.reshape(1, -1) if n_plot_cols > 1 else [axes1]
    
    for i, col in enumerate(numeric_cols):
        row = i // 3
        col_idx = i % 3
        
        if n_rows > 1:
            ax = axes1[row, col_idx] if n_plot_cols > 1 else axes1[row]
        else:
            ax = axes1[col_idx] if n_plot_cols > 1 else axes1[0]
        
        col_data = df[col].dropna()
        
        if len(col_data) > 0:
            # Create histogram
            n_bins = min(30, max(10, int(np.sqrt(len(col_data)))))
            n, bins, patches = ax.hist(col_data, bins=n_bins, alpha=0.7, color='skyblue', 
                                      edgecolor='black', density=False)
            
            # Add statistical information
            mean_val = col_data.mean()
            median_val = col_data.median()
            std_val = col_data.std()
            
            # Add vertical lines for mean and median
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, 
                      label=f'Median: {median_val:.2f}')
            
            ax.set_title(f'{col}\\nn={len(col_data)}, σ={std_val:.2f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize='small')
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
    
    # 2. HISTOGRAMS WITH NORMAL DISTRIBUTION OVERLAY
    print("\\n📊 2. HISTOGRAMS WITH NORMAL DISTRIBUTION OVERLAY")
    
    fig2, axes2 = plt.subplots(n_rows, n_plot_cols, figsize=(6*n_plot_cols, 4*n_rows))
    fig2.suptitle('Histograms vs Normal Distribution', fontsize=16, fontweight='bold')
    
    if n_cols == 1:
        axes2 = [axes2]
    elif n_rows == 1:
        axes2 = axes2.reshape(1, -1) if n_plot_cols > 1 else [axes2]
    
    for i, col in enumerate(numeric_cols):
        row = i // 3
        col_idx = i % 3
        
        if n_rows > 1:
            ax = axes2[row, col_idx] if n_plot_cols > 1 else axes2[row]
        else:
            ax = axes2[col_idx] if n_plot_cols > 1 else axes2[0]
        
        col_data = df[col].dropna()
        
        if len(col_data) > 0:
            # Create density histogram
            n_bins = min(30, max(10, int(np.sqrt(len(col_data)))))
            ax.hist(col_data, bins=n_bins, alpha=0.6, color='lightblue', 
                   edgecolor='black', density=True, label='Data')
            
            # Overlay normal distribution
            mean_val = col_data.mean()
            std_val = col_data.std()
            
            x_range = np.linspace(col_data.min(), col_data.max(), 100)
            normal_dist = stats.norm.pdf(x_range, mean_val, std_val)
            ax.plot(x_range, normal_dist, 'r-', linewidth=2, 
                   label=f'Normal(μ={mean_val:.2f}, σ={std_val:.2f})')
            
            # Calculate and display normality test
            if len(col_data) <= 5000:  # Shapiro-Wilk limitation
                try:
                    _, p_value = stats.shapiro(col_data)
                    normality_text = f'Shapiro p={p_value:.3f}'
                    if p_value < 0.05:
                        normality_text += '\\n(Non-normal)'
                    else:
                        normality_text += '\\n(Normal)'
                except:
                    normality_text = 'Normality test failed'
            else:
                normality_text = 'Sample too large\\nfor Shapiro-Wilk'
            
            ax.text(0.02, 0.98, normality_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(col)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{col}\\nNo valid data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(col)
    
    # Hide unused subplots
    for i in range(n_cols, total_subplots):
        row = i // n_plot_cols
        col_idx = i % n_plot_cols
        if n_rows > 1:
            axes2[row, col_idx].set_visible(False)
        else:
            axes2[col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # 3. COMPARATIVE HISTOGRAMS (ALL VARIABLES)
    print("\\n📊 3. COMPARATIVE HISTOGRAMS (STANDARDIZED)")
    
    # Standardize all variables for comparison
    standardized_data = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 1:
            standardized_data[col] = (col_data - col_data.mean()) / col_data.std()
    
    if len(standardized_data) > 0:
        plt.figure(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(standardized_data)))
        
        for i, (col, std_data) in enumerate(standardized_data.items()):
            plt.hist(std_data, bins=30, alpha=0.5, label=col[:15], 
                    color=colors[i], edgecolor='black', density=True)
        
        # Add reference normal distribution
        x_range = np.linspace(-4, 4, 100)
        plt.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'k--', 
                linewidth=2, label='Standard Normal')
        
        plt.title('Comparative Histograms (Standardized)', fontsize=14, fontweight='bold')
        plt.xlabel('Standardized Value')
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 4. HISTOGRAMS WITH PERCENTILE OVERLAYS
    print("\\n📊 4. HISTOGRAMS WITH PERCENTILE INFORMATION")
    
    # Select top 4 variables for detailed analysis
    detail_vars = numeric_cols[:min(4, len(numeric_cols))]
    
    if len(detail_vars) > 0:
        n_detail_cols = min(2, len(detail_vars))
        n_detail_rows = (len(detail_vars) + 1) // 2
        
        fig3, axes3 = plt.subplots(n_detail_rows, n_detail_cols, figsize=(12, 6*n_detail_rows))
        fig3.suptitle('Histograms with Percentile Information', fontsize=16, fontweight='bold')
        
        if len(detail_vars) == 1:
            axes3 = [axes3]
        elif n_detail_rows == 1:
            axes3 = axes3.reshape(1, -1) if n_detail_cols > 1 else [axes3]
        
        for i, col in enumerate(detail_vars):
            row = i // n_detail_cols
            col_idx = i % n_detail_cols
            
            if n_detail_rows > 1:
                ax = axes3[row, col_idx] if n_detail_cols > 1 else axes3[row]
            else:
                ax = axes3[col_idx] if n_detail_cols > 1 else axes3[0]
            
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                # Create histogram
                n_bins = min(30, max(10, int(np.sqrt(len(col_data)))))
                n, bins, patches = ax.hist(col_data, bins=n_bins, alpha=0.7, color='lightgreen', 
                                         edgecolor='black')
                
                # Calculate percentiles
                percentiles = [5, 25, 50, 75, 95]
                percentile_values = np.percentile(col_data, percentiles)
                
                # Add percentile lines
                colors_perc = ['red', 'orange', 'blue', 'orange', 'red']
                labels = ['5th', '25th (Q1)', '50th (Median)', '75th (Q3)', '95th']
                
                for perc, val, color, label in zip(percentiles, percentile_values, colors_perc, labels):
                    ax.axvline(val, color=color, linestyle=':', linewidth=2, alpha=0.8)
                    ax.text(val, ax.get_ylim()[1]*0.8, f'{label}\\n{val:.2f}', 
                           rotation=90, verticalalignment='bottom', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                # Add IQR shading
                q1, q3 = percentile_values[1], percentile_values[3]
                ax.axvspan(q1, q3, alpha=0.2, color='yellow', label='IQR')
                
                ax.set_title(f'{col}\\nIQR: {q3-q1:.2f}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{col}\\nNo valid data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(col)
        
        # Hide unused subplots if any
        total_detail_subplots = n_detail_rows * n_detail_cols
        for i in range(len(detail_vars), total_detail_subplots):
            row = i // n_detail_cols
            col_idx = i % n_detail_cols
            if n_detail_rows > 1:
                axes3[row, col_idx].set_visible(False)
            else:
                axes3[col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    # 5. HISTOGRAM SUMMARY STATISTICS
    print("\\n📊 5. HISTOGRAM ANALYSIS SUMMARY")
    print("-" * 60)
    
    summary_stats = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            # Calculate optimal bin number using different methods
            sturges_bins = int(np.log2(len(col_data)) + 1)
            scott_bins = int((col_data.max() - col_data.min()) / (3.5 * col_data.std() / (len(col_data) ** (1/3))))
            fd_bins = int((col_data.max() - col_data.min()) / (2 * (np.percentile(col_data, 75) - np.percentile(col_data, 25)) / (len(col_data) ** (1/3))))
            
            # Ensure reasonable bounds
            scott_bins = max(5, min(50, scott_bins)) if not np.isnan(scott_bins) else 10
            fd_bins = max(5, min(50, fd_bins)) if not np.isnan(fd_bins) else 10
            
            summary_stats.append({
                'Column': col,
                'Sample_Size': len(col_data),
                'Range': col_data.max() - col_data.min(),
                'Sturges_Bins': sturges_bins,
                'Scott_Bins': scott_bins,
                'FD_Bins': fd_bins,
                'Recommended_Bins': min(30, max(10, int(np.sqrt(len(col_data)))))
            })
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        print("\\nOptimal Bin Number Recommendations:")
        print(summary_df.to_string(index=False, float_format='%.0f'))
        
        print("\\n📋 Bin Selection Guidelines:")
        print("   • Sturges Rule: log₂(n) + 1 (good for normal data)")
        print("   • Scott's Rule: Based on standard deviation (good for continuous data)")
        print("   • Freedman-Diaconis: Based on IQR (robust to outliers)")
        print("   • Square Root: √n (general purpose)")
        
        # Average recommendations
        avg_recommended = summary_df['Recommended_Bins'].mean()
        print(f"\\n📊 Average recommended bins across variables: {avg_recommended:.0f}")

print("\\n" + "="*60)
print("✅ Histogram plots analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return HistogramPlotsAnalysis