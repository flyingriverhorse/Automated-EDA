"""Violin Plots Analysis Component.

Focused analysis component for creating violin plots visualization only.
"""
from typing import Dict, Any


class ViolinPlotsAnalysis:
    """Focused component for violin plot visualization"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "violin_plots",
            "display_name": "Violin Plots",
            "description": "Violin plot visualizations showing distribution shape and density",
            "category": "numeric",
            "subcategory": "distribution",
            "complexity": "basic",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-20 seconds",
            "icon": "chart-violin",
            "tags": ["visualization", "distribution", "violin", "density"]
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
        """Generate focused violin plots analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== VIOLIN PLOTS ANALYSIS ===")
print("Distribution shape and density visualization with violin plots")
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
    print(f"🎻 CREATING VIOLIN PLOTS FOR {len(numeric_cols)} NUMERIC COLUMNS")
    print()
    
    # Filter columns with enough unique values for meaningful violin plots
    violin_cols = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 10 and len(col_data.unique()) > 5:
            violin_cols.append(col)
        else:
            print(f"⚠️  {col}: Skipping (insufficient data or low variability)")
    
    if not violin_cols:
        print("❌ NO SUITABLE COLUMNS FOR VIOLIN PLOTS")
        print("   Violin plots require columns with sufficient data and variability.")
    else:
        print(f"📊 CREATING VIOLIN PLOTS FOR {len(violin_cols)} SUITABLE COLUMNS")
        print()
        
        n_cols = len(violin_cols)
        n_rows = (n_cols + 2) // 3
        n_plot_cols = min(3, n_cols)
        
        fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(6*n_plot_cols, 5*n_rows))
        fig.suptitle('Violin Plots - Distribution Shape and Density', fontsize=16, fontweight='bold')
        
        # Ensure axes is always a 2D array for consistent indexing
        if n_rows == 1 and n_plot_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_plot_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(violin_cols):
            row = i // 3
            col_idx = i % 3
            
            ax = axes[row, col_idx]
            col_data = df[col].dropna()
            
            # Create violin plot
            parts = ax.violinplot([col_data], positions=[0], widths=0.6, showmeans=True, 
                                showmedians=True, showextrema=True)
            
            # Color the violin
            for pc in parts['bodies']:
                pc.set_facecolor('#8da0cb')
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # Color the statistical lines
            parts['cmeans'].set_color('red')
            parts['cmeans'].set_linewidth(2)
            parts['cmedians'].set_color('orange')
            parts['cmedians'].set_linewidth(2)
            parts['cbars'].set_color('black')
            parts['cmins'].set_color('black')
            parts['cmaxes'].set_color('black')
            
            # Add statistics annotations
            mean_val = col_data.mean()
            median_val = col_data.median()
            std_val = col_data.std()
            
            ax.text(0.05, 0.95, f'Mean: {mean_val:.2f}\\nMedian: {median_val:.2f}\\nStd: {std_val:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{col}\\nViolin Plot')
            ax.set_ylabel('Value')
            ax.set_xticks([0])
            ax.set_xticklabels([col])
            ax.grid(True, alpha=0.3)
            
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
        
        print("🎻 VIOLIN PLOT INTERPRETATION:")
        print("   • Width indicates density/frequency at each value")
        print("   • Wider sections → More data points at that value")
        print("   • Narrower sections → Fewer data points at that value")
        print("   • Red line → Mean value")
        print("   • Orange line → Median value")
        print("   • Black lines → Min/Max and quartiles")
        print("   • Symmetric shape → Normal-like distribution")
        print("   • Multiple bulges → Multi-modal distribution")
        print("   • Skewed shape → Non-normal distribution")
        print()
        
        # Summary statistics for violin-suitable columns
        print("📊 SUMMARY STATISTICS FOR VIOLIN PLOT COLUMNS:")
        summary_stats = df[violin_cols].describe()
        print(summary_stats.round(3).to_string())

print("\\n" + "="*50)
print("✅ VIOLIN PLOTS ANALYSIS COMPLETE")
print("="*50)
'''