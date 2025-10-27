"""KDE (Kernel Density Estimation) Plots Analysis Component.

Focused analysis component for creating KDE/density plots visualization only.
"""
from typing import Dict, Any


class KDEPlotsAnalysis:
    """Focused component for KDE plot visualization"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "kde_plots",
            "display_name": "KDE (Density) Plots",
            "description": "Kernel Density Estimation plots showing smooth distribution curves",
            "category": "numeric",
            "subcategory": "distribution",
            "complexity": "basic",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-15 seconds",
            "icon": "chart-curve",
            "tags": ["visualization", "distribution", "kde", "density", "smooth"]
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
        """Generate focused KDE plots analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

print("=== KDE (KERNEL DENSITY ESTIMATION) PLOTS ANALYSIS ===")
print("Smooth density curves showing distribution shape")
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
    print(f"🌊 CREATING KDE PLOTS FOR {len(numeric_cols)} NUMERIC COLUMNS")
    print()
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    n_plot_cols = min(3, n_cols)
    
    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(6*n_plot_cols, 4*n_rows))
    fig.suptitle('Kernel Density Estimation (KDE) Plots', fontsize=16, fontweight='bold')
    
    # Ensure axes is always a 2D array for consistent indexing
    if n_rows == 1 and n_plot_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_plot_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, col in enumerate(numeric_cols):
        row = i // 3
        col_idx = i % 3
        
        ax = axes[row, col_idx]
        col_data = df[col].dropna()
        
        if len(col_data) < 3:
            ax.text(0.5, 0.5, f'Insufficient data\\nfor {col}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{col} - Insufficient Data')
            continue
            
        # KDE plot with seaborn (filled)
        try:
            sns.kdeplot(data=col_data, ax=ax, fill=True, alpha=0.6, color='skyblue', label='KDE')
            
            # Add mean and median lines
            mean_val = col_data.mean()
            median_val = col_data.median()
            
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
            
            # Add statistics text box
            std_val = col_data.std()
            skew_val = col_data.skew()
            
            stats_text = f'Mean: {mean_val:.2f}\\nMedian: {median_val:.2f}\\nStd: {std_val:.2f}\\nSkew: {skew_val:.2f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{col} - Density Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating KDE\\nfor {col}:\\n{str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{col} - Error')
    
    # Remove empty subplots
    if n_cols < n_rows * n_plot_cols:
        for i in range(n_cols, n_rows * n_plot_cols):
            row = i // 3
            col_idx = i % 3
            ax_to_remove = axes[row, col_idx]
            fig.delaxes(ax_to_remove)
    
    plt.tight_layout()
    plt.show()
    print()
    
    print("🌊 KDE PLOT INTERPRETATION:")
    print("   • Smooth curve showing probability density")
    print("   • Higher peaks → More frequent values")
    print("   • Wider spread → More variability")
    print("   • Multiple peaks → Multi-modal distribution")
    print("   • Skewed curves → Non-symmetric distribution")
    print("   • Red dashed line → Mean value")
    print("   • Orange dashed line → Median value")
    print()
    
    # Create a comparison plot if multiple columns
    if len(numeric_cols) > 1 and len(numeric_cols) <= 8:
        print("📊 COMPARATIVE KDE PLOTS:")
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = sns.color_palette("husl", len(numeric_cols))
        
        for i, col in enumerate(numeric_cols):
            col_data = df[col].dropna()
            if len(col_data) >= 3:
                # Standardize data for comparison
                standardized_data = (col_data - col_data.mean()) / col_data.std()
                
                try:
                    sns.kdeplot(data=standardized_data, ax=ax2, alpha=0.7, 
                              color=colors[i], label=f'{col} (standardized)')
                except:
                    continue
        
        ax2.set_title('Comparative KDE Plots (Standardized)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Standardized Value')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        print()
        
        print("📈 STANDARDIZED COMPARISON INTERPRETATION:")
        print("   • All variables scaled to same range for comparison")
        print("   • Similar shapes → Similar distribution patterns")
        print("   • Different peaks → Different central tendencies")
        print("   • Different spreads → Different variabilities")

print("\\n" + "="*50)
print("✅ KDE PLOTS ANALYSIS COMPLETE")
print("="*50)
'''