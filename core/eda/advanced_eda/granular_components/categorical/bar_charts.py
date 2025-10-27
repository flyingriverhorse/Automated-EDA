"""Categorical Bar Charts Component.

Focused component specifically for categorical bar chart visualizations.
"""


class CategoricalBarChartsAnalysis:
    """Generate comprehensive bar charts for categorical variables."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "categorical_bar_charts",
            "display_name": "Categorical Bar Charts",
            "description": "Create detailed bar charts for categorical variable distribution analysis",
            "category": "categorical",
            "subcategory": "visualization",
            "complexity": "basic",
            "tags": ["visualization", "categorical", "bar chart", "frequency"],
            "estimated_runtime": "5-10 seconds",
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
        """Generate code for categorical bar charts."""
        
        return '''
# ===== CATEGORICAL BAR CHARTS ANALYSIS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("📊 CATEGORICAL BAR CHARTS ANALYSIS")
print("="*60)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Get categorical columns (object type and categorical type)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Also include numeric columns with few unique values (likely categorical)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].nunique() <= 10 and df[col].nunique() > 1:
        categorical_cols.append(col)

print(f"\\n📊 Found {len(categorical_cols)} categorical columns for bar chart analysis")

if len(categorical_cols) == 0:
    print("❌ NO CATEGORICAL COLUMNS FOUND")
    print("   This analysis requires categorical data.")
else:
    # 1. INDIVIDUAL BAR CHARTS
    print("\\n📊 1. INDIVIDUAL BAR CHARTS WITH STATISTICS")
    
    n_cols = len(categorical_cols)
    n_plot_cols = min(3, n_cols)
    n_rows = (n_cols + 2) // 3
    
    fig1, axes1 = plt.subplots(n_rows, n_plot_cols, figsize=(6*n_plot_cols, 4*n_rows))
    fig1.suptitle('Individual Category Bar Charts', fontsize=16, fontweight='bold')
    
    if n_cols == 1:
        axes1 = [axes1]
    elif n_rows == 1:
        axes1 = axes1.reshape(1, -1) if n_plot_cols > 1 else [axes1]
    
    category_stats = {}
    
    for i, col in enumerate(categorical_cols):
        row = i // 3
        col_idx = i % 3
        
        if n_rows > 1:
            ax = axes1[row, col_idx] if n_plot_cols > 1 else axes1[row]
        else:
            ax = axes1[col_idx] if n_plot_cols > 1 else axes1[0]
        
        # Calculate value counts
        value_counts = df[col].value_counts()
        total_count = len(df[col].dropna())
        
        if len(value_counts) > 0:
            # Limit to top 15 categories for readability
            if len(value_counts) > 15:
                top_categories = value_counts.head(15)
                others_count = value_counts.tail(len(value_counts) - 15).sum()
                if others_count > 0:
                    top_categories['Others'] = others_count
                value_counts = top_categories
            
            # Create bar chart
            bars = ax.bar(range(len(value_counts)), value_counts.values, 
                         alpha=0.7, edgecolor='black', linewidth=0.7)
            
            # Color bars with gradient
            colors = plt.cm.viridis(np.linspace(0, 1, len(value_counts)))
            for bar, color in zip(bars, colors):
                bar.set_facecolor(color)
            
            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, value_counts.values)):
                height = bar.get_height()
                percentage = (count / total_count) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + max(value_counts) * 0.01,
                       f'{count}\\n({percentage:.1f}%)',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.set_title(f'{col}\\n({len(df[col].unique())} unique values)', fontweight='bold')
            ax.set_xlabel('Categories')
            ax.set_ylabel('Frequency')
            
            # Rotate x-labels if needed
            labels = [str(label)[:15] + '...' if len(str(label)) > 15 else str(label) 
                     for label in value_counts.index]
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Store statistics
            category_stats[col] = {
                'unique_count': df[col].nunique(),
                'most_common': value_counts.index[0],
                'most_common_count': value_counts.iloc[0],
                'most_common_pct': (value_counts.iloc[0] / total_count) * 100,
                'total_observations': total_count,
                'missing_values': df[col].isnull().sum()
            }
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
    
    # 2. HORIZONTAL BAR CHARTS FOR LONG LABELS
    print("\\n📊 2. HORIZONTAL BAR CHARTS (Top Categories)")
    
    # Select top 4 variables with most categories for detailed view
    vars_by_categories = [(col, stats['unique_count']) for col, stats in category_stats.items()]
    vars_by_categories.sort(key=lambda x: x[1], reverse=True)
    top_complex_vars = vars_by_categories[:min(4, len(vars_by_categories))]
    
    if len(top_complex_vars) > 0:
        n_complex_cols = min(2, len(top_complex_vars))
        n_complex_rows = (len(top_complex_vars) + 1) // 2
        
        fig2, axes2 = plt.subplots(n_complex_rows, n_complex_cols, 
                                  figsize=(8*n_complex_cols, 5*n_complex_rows))
        fig2.suptitle('Horizontal Bar Charts - Top Categories', fontsize=16, fontweight='bold')
        
        if len(top_complex_vars) == 1:
            axes2 = [axes2]
        elif n_complex_rows == 1:
            axes2 = axes2.reshape(1, -1) if n_complex_cols > 1 else [axes2]
        
        for i, (col, unique_count) in enumerate(top_complex_vars):
            row = i // n_complex_cols
            col_idx = i % n_complex_cols
            
            if n_complex_rows > 1:
                ax = axes2[row, col_idx] if n_complex_cols > 1 else axes2[row]
            else:
                ax = axes2[col_idx] if n_complex_cols > 1 else axes2[0]
            
            # Get top 12 categories
            value_counts = df[col].value_counts().head(12)
            
            # Create horizontal bar chart
            y_pos = np.arange(len(value_counts))
            bars = ax.barh(y_pos, value_counts.values, alpha=0.7, edgecolor='black')
            
            # Color bars with gradient
            colors = plt.cm.plasma(np.linspace(0, 1, len(value_counts)))
            for bar, color in zip(bars, colors):
                bar.set_facecolor(color)
            
            # Add value labels
            total_count = len(df[col].dropna())
            for i, (bar, count) in enumerate(zip(bars, value_counts.values)):
                width = bar.get_width()
                percentage = (count / total_count) * 100
                ax.text(width + max(value_counts) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{count} ({percentage:.1f}%)',
                       ha='left', va='center', fontsize=9, fontweight='bold')
            
            ax.set_title(f'{col} - Top Categories\\n({unique_count} total unique)', fontweight='bold')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Categories')
            
            # Set category labels
            labels = [str(label)[:25] + '...' if len(str(label)) > 25 else str(label) 
                     for label in value_counts.index]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Invert y-axis to show highest count at top
            ax.invert_yaxis()
        
        # Hide unused subplots
        total_complex_subplots = n_complex_rows * n_complex_cols
        for i in range(len(top_complex_vars), total_complex_subplots):
            row = i // n_complex_cols
            col_idx = i % n_complex_cols
            if n_complex_rows > 1:
                axes2[row, col_idx].set_visible(False)
            else:
                axes2[col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    # 3. COMPARATIVE CATEGORY COUNTS
    print("\\n📊 3. COMPARATIVE CATEGORY COUNTS")
    
    plt.figure(figsize=(12, 6))
    
    # Create comparison of unique category counts
    categories = list(category_stats.keys())
    unique_counts = [stats['unique_count'] for stats in category_stats.values()]
    
    bars = plt.bar(range(len(categories)), unique_counts, alpha=0.7, edgecolor='black')
    
    # Color bars by count level
    colors = []
    for count in unique_counts:
        if count <= 5:
            colors.append('green')  # Low cardinality
        elif count <= 15:
            colors.append('orange')  # Medium cardinality
        else:
            colors.append('red')  # High cardinality
    
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, unique_counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(unique_counts) * 0.01,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Number of Unique Categories per Variable', fontsize=14, fontweight='bold')
    plt.xlabel('Variables')
    plt.ylabel('Number of Unique Categories')
    
    # Set x-labels
    labels = [label[:15] + '...' if len(label) > 15 else label for label in categories]
    plt.xticks(range(len(categories)), labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    plt.figtext(0.02, 0.02, 'Green: Low cardinality (≤5), Orange: Medium (6-15), Red: High (>15)', 
               fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    # 4. CATEGORY STATISTICS SUMMARY
    print("\\n📊 4. CATEGORY STATISTICS SUMMARY")
    print("-" * 80)
    
    # Create summary table
    summary_data = []
    for col, stats in category_stats.items():
        summary_data.append({
            'Variable': col,
            'Unique_Categories': stats['unique_count'],
            'Most_Common_Category': str(stats['most_common'])[:20],
            'Most_Common_Count': stats['most_common_count'],
            'Most_Common_Pct': f"{stats['most_common_pct']:.1f}%",
            'Total_Observations': stats['total_observations'],
            'Missing_Values': stats['missing_values']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\\n📋 Categorical Variables Summary:")
    print(summary_df.to_string(index=False))
    
    # 5. CATEGORY INSIGHTS
    print("\\n💡 CATEGORICAL ANALYSIS INSIGHTS:")
    print("-" * 60)
    
    # Find variables with high cardinality
    high_cardinality = [col for col, stats in category_stats.items() if stats['unique_count'] > 20]
    if high_cardinality:
        print(f"\\n🔴 High Cardinality Variables (>20 categories): {len(high_cardinality)}")
        for col in high_cardinality:
            unique_count = category_stats[col]['unique_count']
            print(f"   • {col}: {unique_count} categories")
        print("   → Consider grouping rare categories or using dimensionality reduction")
    
    # Find variables with dominant categories
    dominant_vars = []
    for col, stats in category_stats.items():
        if stats['most_common_pct'] > 80:
            dominant_vars.append((col, stats['most_common_pct']))
    
    if dominant_vars:
        print(f"\\n🟡 Variables with Dominant Categories (>80%): {len(dominant_vars)}")
        for col, pct in dominant_vars:
            most_common = category_stats[col]['most_common']
            print(f"   • {col}: '{most_common}' represents {pct:.1f}% of data")
        print("   → Check for data imbalance or consider binary encoding")
    
    # Find variables with many missing values
    missing_vars = [(col, stats['missing_values']) for col, stats in category_stats.items() 
                   if stats['missing_values'] > 0]
    
    if missing_vars:
        missing_vars.sort(key=lambda x: x[1], reverse=True)
        print(f"\\n⚠️  Variables with Missing Values: {len(missing_vars)}")
        for col, missing_count in missing_vars:
            total = category_stats[col]['total_observations'] + missing_count
            pct = (missing_count / total) * 100
            print(f"   • {col}: {missing_count} missing ({pct:.1f}%)")
    
    # Overall summary
    print("\\n📋 OVERALL SUMMARY:")
    print("-" * 60)
    total_vars = len(category_stats)
    avg_categories = np.mean([stats['unique_count'] for stats in category_stats.values()])
    print(f"   • Total categorical variables analyzed: {total_vars}")
    print(f"   • Average categories per variable: {avg_categories:.1f}")
    print(f"   • Variables with high cardinality (>20): {len(high_cardinality)}")
    print(f"   • Variables with dominant categories (>80%): {len(dominant_vars)}")
    print(f"   • Variables with missing values: {len(missing_vars)}")

print("\\n" + "="*60)
print("✅ Categorical bar charts analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return CategoricalBarChartsAnalysis