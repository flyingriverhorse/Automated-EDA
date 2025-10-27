"""Categorical Visualization Component.

Provides bar plots and pie charts for categorical variable distribution.
"""


class CategoricalVisualizationAnalysis:
    """Generate bar plots and pie charts for categorical variables."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "categorical_visualization",
            "display_name": "Categorical Visualization",
            "description": "Bar plots and pie charts for categorical variable distribution",
            "category": "univariate",
            "complexity": "basic", 
            "tags": ["visualization", "categorical", "bar chart", "pie chart"],
            "estimated_runtime": "2-5 seconds",
            "icon": "📊"
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview=None):
        """Check if analysis can be performed on the data."""
        if not data_preview:
            return True
        return len(data_preview.get('categorical_columns', [])) > 0 or len(data_preview.get('object_columns', [])) > 0
    
    @staticmethod
    def generate_code(data_preview=None):
        """Generate code for categorical visualization."""
        
        return '''
# ===== CATEGORICAL VISUALIZATION ANALYSIS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

print("="*60)
print("📊 CATEGORICAL VISUALIZATION ANALYSIS")
print("="*60)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Get categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
print(f"\\n📋 Found {len(categorical_cols)} categorical columns for visualization")

if len(categorical_cols) == 0:
    print("❌ No categorical columns found for visualization")
else:
    # Calculate figure layout
    n_cols = min(len(categorical_cols), 3)  # Max 3 columns
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    # Create subplots for bar charts
    fig_bar, axes_bar = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    fig_bar.suptitle('📊 Categorical Variables - Bar Charts', fontsize=16, fontweight='bold')
    
    if n_rows == 1 and n_cols == 1:
        axes_bar = [axes_bar]
    elif n_rows == 1 or n_cols == 1:
        axes_bar = axes_bar.flatten()
    else:
        axes_bar = axes_bar.flatten()
    
    # Create subplots for pie charts (only for columns with reasonable cardinality)
    pie_candidates = []
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if 2 <= unique_count <= 10:  # Reasonable for pie chart
            pie_candidates.append(col)
    
    if pie_candidates:
        n_pie_cols = min(len(pie_candidates), 3)
        n_pie_rows = (len(pie_candidates) + n_pie_cols - 1) // n_pie_cols
        fig_pie, axes_pie = plt.subplots(n_pie_rows, n_pie_cols, figsize=(6*n_pie_cols, 5*n_pie_rows))
        fig_pie.suptitle('🥧 Categorical Variables - Pie Charts', fontsize=16, fontweight='bold')
        
        if n_pie_rows == 1 and n_pie_cols == 1:
            axes_pie = [axes_pie]
        elif n_pie_rows == 1 or n_pie_cols == 1:
            axes_pie = axes_pie.flatten()
        else:
            axes_pie = axes_pie.flatten()
    
    # Generate visualizations
    for idx, col in enumerate(categorical_cols):
        print(f"\\n📈 Creating visualizations for: {col}")
        
        # Get value counts (limit to top 20 for readability)
        value_counts = df[col].value_counts().head(20)
        
        # Bar Chart
        ax_bar = axes_bar[idx]
        
        # Create bar plot
        bars = ax_bar.bar(range(len(value_counts)), value_counts.values, 
                         color=plt.cm.Set3(np.linspace(0, 1, len(value_counts))))
        
        ax_bar.set_title(f'{col}\\n({df[col].nunique()} unique values)', fontweight='bold', pad=20)
        ax_bar.set_xlabel('Categories', fontweight='bold')
        ax_bar.set_ylabel('Frequency', fontweight='bold')
        
        # Set x-axis labels
        labels = [str(label)[:15] + '...' if len(str(label)) > 15 else str(label) for label in value_counts.index]
        ax_bar.set_xticks(range(len(value_counts)))
        ax_bar.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontweight='bold')
        
        ax_bar.grid(axis='y', alpha=0.3)
        
        # Pie Chart (only if suitable)
        if col in pie_candidates:
            pie_idx = pie_candidates.index(col)
            ax_pie = axes_pie[pie_idx]
            
            # Get top values for pie chart (max 8 slices)
            pie_values = df[col].value_counts().head(8)
            
            # Create pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(pie_values)))
            wedges, texts, autotexts = ax_pie.pie(pie_values.values, 
                                                 labels=pie_values.index,
                                                 autopct='%1.1f%%',
                                                 colors=colors,
                                                 startangle=90)
            
            ax_pie.set_title(f'{col}\\nDistribution', fontweight='bold', pad=20)
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
    
    # Hide unused subplots in bar chart
    for idx in range(len(categorical_cols), len(axes_bar)):
        axes_bar[idx].set_visible(False)
    
    # Hide unused subplots in pie chart
    if pie_candidates:
        for idx in range(len(pie_candidates), len(axes_pie)):
            axes_pie[idx].set_visible(False)
    
    # Adjust layout and display
    fig_bar.tight_layout()
    plt.figure(fig_bar.number)
    plt.show()
    
    if pie_candidates:
        fig_pie.tight_layout()
        plt.figure(fig_pie.number)
        plt.show()
        print(f"\\n🥧 Generated pie charts for {len(pie_candidates)} suitable categorical columns")
    else:
        print(f"\\n⚠️  No columns suitable for pie charts (need 2-10 unique values)")
    
    print(f"\\n📊 Generated bar charts for all {len(categorical_cols)} categorical columns")

print("\\n" + "="*60)
print("✅ Categorical visualization complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return CategoricalVisualizationAnalysis