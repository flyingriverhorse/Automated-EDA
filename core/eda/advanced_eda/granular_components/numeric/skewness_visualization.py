"""Skewness Visualization Analysis Component.

Focused component specifically for skewness visualizations without statistical details.
"""
from typing import Dict, Any


class SkewnessVisualizationAnalysis:
    """Focused component for skewness visualization"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "skewness_visualization",
            "display_name": "Skewness Visualization",
            "description": "Visual analysis of distribution skewness with histograms, Q-Q plots, and box plots",
            "category": "univariate",
            "subcategory": "skewness",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-15 seconds",
            "icon": "📈",
            "tags": ["skewness", "visualization", "distribution", "plots"]
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
        """Generate focused skewness visualization analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("📈 SKEWNESS VISUALIZATION ANALYSIS")
print("="*60)
print("Visual analysis of distribution skewness")
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
    print(f"📊 CREATING SKEWNESS VISUALIZATIONS FOR {len(numeric_cols)} NUMERIC COLUMNS")
    print()
    
    # Calculate skewness for reference
    skewness_values = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 1:
            skewness_values[col] = stats.skew(col_data)
    
    if len(skewness_values) == 0:
        print("❌ No valid numeric columns found for visualization")
    else:
        # 1. SKEWNESS OVERVIEW HISTOGRAM
        print("📊 1. SKEWNESS DISTRIBUTION OVERVIEW")
        
        plt.figure(figsize=(12, 4))
        
        # Subplot 1: Histogram of skewness values
        plt.subplot(1, 2, 1)
        skew_vals = list(skewness_values.values())
        plt.hist(skew_vals, bins=min(10, len(skew_vals)), alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Perfect Symmetry')
        plt.axvline(-0.5, color='orange', linestyle=':', alpha=0.7, label='Moderate Skew Threshold')
        plt.axvline(0.5, color='orange', linestyle=':', alpha=0.7)
        plt.xlabel('Skewness Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Skewness Values\\nAcross All Variables')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Skewness by variable (bar chart)
        plt.subplot(1, 2, 2)
        cols = list(skewness_values.keys())
        skews = list(skewness_values.values())
        
        # Color code by skewness level
        colors = []
        for skew in skews:
            if abs(skew) < 0.5:
                colors.append('green')
            elif abs(skew) < 1:
                colors.append('orange')
            else:
                colors.append('red')
        
        bars = plt.bar(range(len(cols)), skews, color=colors, alpha=0.7)
        plt.axhline(0, color='black', linestyle='-', alpha=0.8)
        plt.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Skew')
        plt.axhline(-0.5, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(1, color='red', linestyle='--', alpha=0.5, label='High Skew')
        plt.axhline(-1, color='red', linestyle='--', alpha=0.5)
        
        plt.xlabel('Variables')
        plt.ylabel('Skewness')
        plt.title('Skewness by Variable')
        plt.xticks(range(len(cols)), [col[:10] + '...' if len(col) > 10 else col for col in cols], 
                   rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 2. DETAILED DISTRIBUTION PLOTS FOR EACH VARIABLE
        print("\\n📈 2. DETAILED DISTRIBUTION ANALYSIS PER VARIABLE")
        
        # Limit to most interesting variables (highly skewed + some normal)
        sorted_by_skew = sorted(skewness_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Select top skewed variables and some symmetric ones
        vars_to_plot = []
        high_skew_count = 0
        low_skew_count = 0
        
        for var, skew in sorted_by_skew:
            if abs(skew) >= 0.5 and high_skew_count < 4:  # Up to 4 high skew vars
                vars_to_plot.append((var, skew))
                high_skew_count += 1
            elif abs(skew) < 0.5 and low_skew_count < 2:  # Up to 2 symmetric vars
                vars_to_plot.append((var, skew))
                low_skew_count += 1
            
            if len(vars_to_plot) >= 6:  # Limit total plots
                break
        
        if len(vars_to_plot) == 0:
            vars_to_plot = sorted_by_skew[:min(4, len(sorted_by_skew))]
        
        n_vars = len(vars_to_plot)
        n_cols = min(2, n_vars)
        n_rows = (n_vars + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(6*n_cols*2, 4*n_rows))
        fig.suptitle('Detailed Skewness Analysis: Distribution Shapes', fontsize=16, fontweight='bold')
        
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (col, skew_val) in enumerate(vars_to_plot):
            row = i // n_cols
            col_start = (i % n_cols) * 2
            
            col_data = df[col].dropna()
            
            # Histogram with KDE
            if n_rows > 1:
                ax1 = axes[row, col_start]
                ax2 = axes[row, col_start + 1]
            else:
                ax1 = axes[col_start]
                ax2 = axes[col_start + 1]
            
            # Plot 1: Histogram + KDE + Normal overlay
            ax1.hist(col_data, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
            
            # KDE
            try:
                sns.kdeplot(data=col_data, ax=ax1, color='red', linewidth=2, label='Actual Distribution')
            except:
                pass
            
            # Normal distribution overlay
            mu, sigma = col_data.mean(), col_data.std()
            x = np.linspace(col_data.min(), col_data.max(), 100)
            normal_curve = stats.norm.pdf(x, mu, sigma)
            ax1.plot(x, normal_curve, 'g--', linewidth=2, label='Normal Distribution', alpha=0.8)
            
            # Add mean and median lines
            ax1.axvline(col_data.mean(), color='red', linestyle='-', alpha=0.8, label=f'Mean: {col_data.mean():.2f}')
            ax1.axvline(col_data.median(), color='blue', linestyle='-', alpha=0.8, label=f'Median: {col_data.median():.2f}')
            
            ax1.set_title(f'{col}\\nSkewness: {skew_val:.3f}')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Density')
            ax1.legend(fontsize='small')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Q-Q Plot
            stats.probplot(col_data, dist="norm", plot=ax2)
            ax2.set_title(f'Q-Q Plot\\n{"Normal-like" if abs(skew_val) < 0.5 else "Non-normal"}')
            ax2.grid(True, alpha=0.3)
        
        # Hide unused subplots
        total_subplots = n_rows * n_cols * 2
        for i in range(len(vars_to_plot) * 2, total_subplots):
            row = i // (n_cols * 2)
            col = i % (n_cols * 2)
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # 3. COMPARATIVE BOX PLOTS
        print("\\n📊 3. COMPARATIVE BOX PLOTS (Skewness Detection)")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Prepare data for box plots
        plot_data = []
        plot_labels = []
        skew_categories = []
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0 and col in skewness_values:
                plot_data.append(col_data)
                plot_labels.append(col[:15] + '...' if len(col) > 15 else col)
                
                skew_val = skewness_values[col]
                if abs(skew_val) < 0.5:
                    skew_categories.append('Symmetric')
                elif abs(skew_val) < 1:
                    skew_categories.append('Moderate')
                else:
                    skew_categories.append('High')
        
        # Box plot
        bp1 = axes[0].boxplot(plot_data, labels=plot_labels, patch_artist=True)
        
        # Color boxes by skewness level
        colors = {'Symmetric': 'lightgreen', 'Moderate': 'orange', 'High': 'red'}
        for patch, category in zip(bp1['boxes'], skew_categories):
            patch.set_facecolor(colors[category])
            patch.set_alpha(0.7)
        
        axes[0].set_title('Box Plots Colored by Skewness Level\\n(Green=Symmetric, Orange=Moderate, Red=High)')
        axes[0].set_ylabel('Value')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Violin plot for better shape visualization
        parts = axes[1].violinplot(plot_data, positions=range(1, len(plot_data) + 1))
        
        for i, (pc, category) in enumerate(zip(parts['bodies'], skew_categories)):
            pc.set_facecolor(colors[category])
            pc.set_alpha(0.7)
        
        axes[1].set_title('Violin Plots: Distribution Shape Visualization')
        axes[1].set_ylabel('Value')
        axes[1].set_xlabel('Variables')
        axes[1].set_xticks(range(1, len(plot_labels) + 1))
        axes[1].set_xticklabels(plot_labels, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 4. SKEWNESS TRANSFORMATION PREVIEW
        print("\\n🔧 4. TRANSFORMATION PREVIEW FOR HIGHLY SKEWED VARIABLES")
        
        high_skew_vars = [(k, v) for k, v in skewness_values.items() if abs(v) > 1]
        
        if len(high_skew_vars) > 0:
            n_transform_vars = min(3, len(high_skew_vars))  # Limit to 3 variables
            
            for i, (col, original_skew) in enumerate(high_skew_vars[:n_transform_vars]):
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f'Transformation Preview: {col} (Original Skewness: {original_skew:.3f})', 
                           fontsize=14, fontweight='bold')
                
                # Original distribution
                axes[0, 0].hist(col_data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                axes[0, 0].set_title(f'Original\\nSkewness: {original_skew:.3f}')
                axes[0, 0].set_ylabel('Frequency')
                
                # Log transformation (if all values > 0)
                if col_data.min() > 0:
                    log_data = np.log(col_data)
                    log_skew = stats.skew(log_data)
                    axes[0, 1].hist(log_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                    axes[0, 1].set_title(f'Log Transform\\nSkewness: {log_skew:.3f}')
                else:
                    axes[0, 1].text(0.5, 0.5, 'Log transform not applicable\\n(negative/zero values)', 
                                   ha='center', va='center', transform=axes[0, 1].transAxes)
                
                # Square root transformation (if all values >= 0)
                if col_data.min() >= 0:
                    sqrt_data = np.sqrt(col_data)
                    sqrt_skew = stats.skew(sqrt_data)
                    axes[1, 0].hist(sqrt_data, bins=30, alpha=0.7, color='orange', edgecolor='black')
                    axes[1, 0].set_title(f'Square Root Transform\\nSkewness: {sqrt_skew:.3f}')
                    axes[1, 0].set_ylabel('Frequency')
                else:
                    axes[1, 0].text(0.5, 0.5, 'Square root transform\\nnot applicable\\n(negative values)', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
                
                # Box-Cox transformation (if all values > 0)
                if col_data.min() > 0:
                    try:
                        boxcox_data, lambda_param = stats.boxcox(col_data)
                        boxcox_skew = stats.skew(boxcox_data)
                        axes[1, 1].hist(boxcox_data, bins=30, alpha=0.7, color='purple', edgecolor='black')
                        axes[1, 1].set_title(f'Box-Cox Transform\\nλ={lambda_param:.3f}, Skewness: {boxcox_skew:.3f}')
                    except:
                        axes[1, 1].text(0.5, 0.5, 'Box-Cox transform\\nfailed', 
                                       ha='center', va='center', transform=axes[1, 1].transAxes)
                else:
                    axes[1, 1].text(0.5, 0.5, 'Box-Cox transform\\nnot applicable\\n(non-positive values)', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                
                for ax in axes.flat:
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        else:
            print("   ℹ️  No highly skewed variables found (|skewness| > 1)")
        
        # Summary visualization
        print("\\n📊 5. SKEWNESS SUMMARY VISUALIZATION")
        
        plt.figure(figsize=(10, 6))
        
        # Create summary categories
        categories = ['Symmetric\\n(|skew| < 0.5)', 'Moderately Skewed\\n(0.5 ≤ |skew| < 1)', 'Highly Skewed\\n(|skew| ≥ 1)']
        counts = [
            sum(1 for s in skewness_values.values() if abs(s) < 0.5),
            sum(1 for s in skewness_values.values() if 0.5 <= abs(s) < 1),
            sum(1 for s in skewness_values.values() if abs(s) >= 1)
        ]
        
        colors = ['green', 'orange', 'red']
        bars = plt.bar(categories, counts, color=colors, alpha=0.7)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{count}\\n({count/len(skewness_values)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('Distribution of Variables by Skewness Level', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Variables')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add total count
        plt.figtext(0.5, 0.02, f'Total Variables Analyzed: {len(skewness_values)}', 
                   ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.show()

print("\\n" + "="*60)
print("✅ Skewness visualization analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return SkewnessVisualizationAnalysis