"""Scatter Plot Analysis Component.

Provides scatter plot visualization for numeric variable relationships.
"""


class ScatterPlotAnalysis:
    """Generate scatter plots for numeric variable pairs."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "scatter_plot_analysis",
            "display_name": "Scatter Plot Analysis",
            "description": "Scatter plots for numeric vs numeric variable pairs",
            "category": "bivariate",
            "complexity": "intermediate",
            "tags": ["scatter", "visualization", "numeric", "relationships"],
            "estimated_runtime": "3-8 seconds",
            "icon": "📈"
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview=None):
        """Check if analysis can be performed on the data."""
        if not data_preview:
            return True
        numeric_cols = data_preview.get('numeric_columns', [])
        return len(numeric_cols) >= 2
    
    @staticmethod
    def generate_code(data_preview=None):
        """Generate code for scatter plot analysis."""
        
        return '''
# ===== SCATTER PLOT ANALYSIS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import pearsonr

print("="*60)
print("📈 SCATTER PLOT ANALYSIS")
print("="*60)

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\\n📊 Found {len(numeric_cols)} numeric columns")

if len(numeric_cols) < 2:
    print("❌ Need at least 2 numeric columns for scatter plot analysis")
else:
    # Remove columns with all NaN or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and not df[col].isna().all():
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns for scatter plots")
    
    if len(valid_cols) < 2:
        print("❌ Not enough valid columns after removing constant/all-NaN columns")
    else:
        # Generate all possible pairs
        all_pairs = list(combinations(valid_cols, 2))
        print(f"📊 Generating scatter plots for {len(all_pairs)} variable pairs")
        
        # Limit number of pairs to avoid overwhelming output
        max_pairs = 20
        if len(all_pairs) > max_pairs:
            print(f"⚠️  Too many pairs ({len(all_pairs)}). Showing top {max_pairs} most correlated pairs.")
            
            # Calculate correlations to select most interesting pairs
            correlations = []
            for col1, col2 in all_pairs:
                try:
                    corr_val, _ = pearsonr(df[col1].dropna(), df[col2].dropna())
                    if not np.isnan(corr_val):
                        correlations.append((col1, col2, abs(corr_val)))
                except:
                    correlations.append((col1, col2, 0))
            
            # Sort by absolute correlation and take top pairs
            correlations.sort(key=lambda x: x[2], reverse=True)
            selected_pairs = [(col1, col2) for col1, col2, _ in correlations[:max_pairs]]
        else:
            selected_pairs = all_pairs
        
        # Calculate grid size
        n_pairs = len(selected_pairs)
        n_cols = min(4, n_pairs)  # Max 4 columns
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        # Create scatter plot grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle('📈 Scatter Plot Analysis - Numeric Variable Pairs', fontsize=16, fontweight='bold')
        
        # Handle single plot case
        if n_pairs == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Generate scatter plots
        strong_relationships = []
        
        for idx, (col1, col2) in enumerate(selected_pairs):
            ax = axes[idx]
            
            # Get data for this pair (remove NaN values)
            pair_data = df[[col1, col2]].dropna()
            
            if len(pair_data) == 0:
                ax.text(0.5, 0.5, 'No valid data\\nfor this pair', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{col1} vs {col2}\\n(No data)')
                continue
            
            x_data = pair_data[col1]
            y_data = pair_data[col2]
            
            # Create scatter plot with trend line
            ax.scatter(x_data, y_data, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            
            # Add trend line
            try:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
                
                # Calculate correlation
                corr_val, p_value = pearsonr(x_data, y_data)
                
                # Determine relationship strength
                if abs(corr_val) >= 0.7:
                    strength = "Strong"
                    strong_relationships.append((col1, col2, corr_val))
                elif abs(corr_val) >= 0.5:
                    strength = "Moderate"
                elif abs(corr_val) >= 0.3:
                    strength = "Weak"
                else:
                    strength = "Very Weak"
                
                # Color code based on significance
                color = 'green' if p_value < 0.05 else 'orange' if p_value < 0.1 else 'red'
                
                title = f'{col1} vs {col2}\\nr = {corr_val:.3f} ({strength})'
                ax.set_title(title, fontweight='bold', color=color)
                
            except:
                ax.set_title(f'{col1} vs {col2}\\n(Unable to compute correlation)')
            
            ax.set_xlabel(col1, fontweight='bold')
            ax.set_ylabel(col2, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add sample size info
            ax.text(0.02, 0.98, f'n = {len(pair_data):,}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Hide unused subplots
        for idx in range(len(selected_pairs), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Summary statistics
        print("\\n📊 SCATTER PLOT SUMMARY:")
        print("-" * 50)
        print(f"   • Total pairs visualized: {len(selected_pairs)}")
        print(f"   • Data points per plot: {len(df):,} (before removing NaN)")
        
        if strong_relationships:
            print("\\n🔍 STRONG RELATIONSHIPS DETECTED:")
            print("-" * 40)
            for col1, col2, corr in sorted(strong_relationships, key=lambda x: abs(x[2]), reverse=True):
                direction = "positive" if corr > 0 else "negative"
                print(f"   • {col1} ↔ {col2}: r = {corr:.3f} ({direction})")
        else:
            print("\\n   ✅ No strong linear relationships (|r| >= 0.7) detected")
        
        # Additional insights
        print("\\n💡 INTERPRETATION GUIDE:")
        print("-" * 30)
        print("   🟢 Green titles: Statistically significant (p < 0.05)")
        print("   🟠 Orange titles: Marginally significant (p < 0.1)")  
        print("   🔴 Red titles: Not significant (p ≥ 0.1)")
        print("   📏 Red dashed line: Linear trend line")
        print("   📊 n = sample size for each pair")

print("\\n" + "="*60)
print("✅ Scatter plot analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return ScatterPlotAnalysis