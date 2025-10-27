"""Categorical Pie Charts Component.

Focused component specifically for categorical pie chart visualizations.
"""


class CategoricalPieChartsAnalysis:
    """Generate comprehensive pie charts for categorical variables."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "categorical_pie_charts",
            "display_name": "Categorical Pie Charts",
            "description": "Create detailed pie charts for categorical variable distribution analysis",
            "category": "categorical",
            "subcategory": "visualization",
            "complexity": "basic",
            "tags": ["visualization", "categorical", "pie chart", "proportion"],
            "estimated_runtime": "8-12 seconds",
            "icon": "🥧"
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview=None):
        """Check if analysis can be performed on the data."""
        if not data_preview:
            return True
        return len(data_preview.get('categorical_columns', [])) > 0 or len(data_preview.get('object_columns', [])) > 0
    
    @staticmethod
    def generate_code(data_preview=None):
        """Generate code for categorical pie charts."""
        
        return '''
# ===== CATEGORICAL PIE CHARTS ANALYSIS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("🥧 CATEGORICAL PIE CHARTS ANALYSIS")
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

print(f"\\n🥧 Found {len(categorical_cols)} categorical columns for pie chart analysis")

if len(categorical_cols) == 0:
    print("❌ NO CATEGORICAL COLUMNS FOUND")
    print("   This analysis requires categorical data.")
else:
    # Filter variables suitable for pie charts (not too many categories)
    suitable_vars = []
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if 2 <= unique_count <= 12:  # Good range for pie charts
            suitable_vars.append(col)
        elif unique_count > 12:
            print(f"⚠️  Skipping '{col}' - too many categories ({unique_count}) for pie chart")
    
    print(f"\\n📊 Using {len(suitable_vars)} variables suitable for pie chart visualization")
    
    if len(suitable_vars) == 0:
        print("❌ NO SUITABLE VARIABLES FOUND FOR PIE CHARTS")
        print("   Pie charts work best with 2-12 categories")
    else:
        # 1. INDIVIDUAL PIE CHARTS
        print("\\n🥧 1. INDIVIDUAL PIE CHARTS")
        
        n_cols = len(suitable_vars)
        n_plot_cols = min(3, n_cols)
        n_rows = (n_cols + 2) // 3
        
        fig1, axes1 = plt.subplots(n_rows, n_plot_cols, figsize=(6*n_plot_cols, 5*n_rows))
        fig1.suptitle('Categorical Distribution Pie Charts', fontsize=16, fontweight='bold')
        
        if n_cols == 1:
            axes1 = [axes1]
        elif n_rows == 1:
            axes1 = axes1.reshape(1, -1) if n_plot_cols > 1 else [axes1]
        
        category_proportions = {}
        
        for i, col in enumerate(suitable_vars):
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
                # Create pie chart
                colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
                
                wedges, texts, autotexts = ax.pie(value_counts.values, 
                                                 labels=value_counts.index,
                                                 autopct='%1.1f%%',
                                                 startangle=90,
                                                 colors=colors,
                                                 pctdistance=0.85,
                                                 explode=[0.05] * len(value_counts))  # Small explosion for all slices
                
                # Customize text
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
                
                # Truncate long labels
                for i, text in enumerate(texts):
                    label = str(value_counts.index[i])
                    if len(label) > 15:
                        label = label[:12] + '...'
                    text.set_text(label)
                    text.set_fontsize(8)
                
                ax.set_title(f'{col}\\n({len(value_counts)} categories, n={total_count})', 
                           fontweight='bold', pad=20)
                
                # Store proportions
                category_proportions[col] = {
                    'categories': len(value_counts),
                    'most_common': value_counts.index[0],
                    'most_common_pct': (value_counts.iloc[0] / total_count) * 100,
                    'least_common': value_counts.index[-1],
                    'least_common_pct': (value_counts.iloc[-1] / total_count) * 100,
                    'entropy': -(value_counts / value_counts.sum() * np.log2(value_counts / value_counts.sum())).sum()
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
        
        # 2. DONUT CHARTS (Alternative Style)
        print("\\n🍩 2. DONUT CHARTS (Enhanced Style)")
        
        # Select top 4 variables for detailed donut charts
        detail_vars = suitable_vars[:min(4, len(suitable_vars))]
        
        if len(detail_vars) > 0:
            fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
            fig2.suptitle('Donut Charts with Enhanced Details', fontsize=16, fontweight='bold')
            axes2 = axes2.flatten()
            
            for i, col in enumerate(detail_vars):
                if i < 4:
                    ax = axes2[i]
                    
                    # Calculate value counts
                    value_counts = df[col].value_counts()
                    total_count = len(df[col].dropna())
                    
                    if len(value_counts) > 0:
                        # Create donut chart
                        colors = plt.cm.viridis(np.linspace(0, 1, len(value_counts)))
                        
                        wedges, texts, autotexts = ax.pie(value_counts.values,
                                                         labels=None,  # No labels on the chart
                                                         autopct='%1.1f%%',
                                                         startangle=90,
                                                         colors=colors,
                                                         pctdistance=0.7,
                                                         wedgeprops=dict(width=0.5))  # Donut effect
                        
                        # Center circle for donut effect
                        centre_circle = plt.Circle((0, 0), 0.50, fc='white', linewidth=2, edgecolor='black')
                        ax.add_artist(centre_circle)
                        
                        # Center text
                        ax.text(0, 0, f'{col}\\n{total_count} obs', ha='center', va='center',
                               fontsize=12, fontweight='bold')
                        
                        # Custom legend
                        legend_labels = [f'{cat}: {count} ({count/total_count*100:.1f}%)' 
                                       for cat, count in value_counts.items()]
                        ax.legend(wedges, legend_labels, title=f"{col} Distribution",
                                loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                                fontsize=8)
                        
                        ax.set_title(f'Donut Chart: {col}', fontweight='bold', pad=20)
                    else:
                        ax.text(0.5, 0.5, f'{col}\\nNo valid data', ha='center', va='center',
                               transform=ax.transAxes)
                        ax.set_title(col)
            
            # Hide unused subplots
            for i in range(len(detail_vars), 4):
                axes2[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        
        # 3. COMPARATIVE PIE CHARTS
        print("\\n🥧 3. COMPARATIVE PIE CHARTS (Side-by-Side)")
        
        # Compare up to 3 variables side by side
        compare_vars = suitable_vars[:min(3, len(suitable_vars))]
        
        if len(compare_vars) > 1:
            fig3, axes3 = plt.subplots(1, len(compare_vars), figsize=(6*len(compare_vars), 6))
            fig3.suptitle('Comparative Distribution Analysis', fontsize=16, fontweight='bold')
            
            if len(compare_vars) == 2:
                axes3 = [axes3] if not hasattr(axes3, '__iter__') else axes3
            
            for i, col in enumerate(compare_vars):
                ax = axes3[i] if len(compare_vars) > 1 else axes3
                
                # Calculate value counts
                value_counts = df[col].value_counts()
                total_count = len(df[col].dropna())
                
                if len(value_counts) > 0:
                    # Use consistent colors across charts
                    colors = plt.cm.Pastel1(np.linspace(0, 1, len(value_counts)))
                    
                    wedges, texts, autotexts = ax.pie(value_counts.values,
                                                     labels=value_counts.index,
                                                     autopct='%1.1f%%',
                                                     startangle=90,
                                                     colors=colors,
                                                     pctdistance=0.85)
                    
                    # Enhance text
                    for autotext in autotexts:
                        autotext.set_color('black')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(9)
                    
                    # Truncate labels
                    for text in texts:
                        label = text.get_text()
                        if len(str(label)) > 12:
                            text.set_text(str(label)[:9] + '...')
                        text.set_fontsize(8)
                    
                    ax.set_title(f'{col}\\nEntropy: {category_proportions[col]["entropy"]:.2f}', 
                               fontweight='bold', pad=20)
                else:
                    ax.text(0.5, 0.5, f'{col}\\nNo valid data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(col)
            
            plt.tight_layout()
            plt.show()
        
        # 4. PROPORTION ANALYSIS TABLE
        print("\\n📊 4. PROPORTION ANALYSIS SUMMARY")
        print("-" * 80)
        
        # Create detailed summary
        summary_data = []
        for col, props in category_proportions.items():
            summary_data.append({
                'Variable': col,
                'Categories': props['categories'],
                'Most_Common': str(props['most_common'])[:15],
                'Most_Common_Pct': f"{props['most_common_pct']:.1f}%",
                'Least_Common': str(props['least_common'])[:15],
                'Least_Common_Pct': f"{props['least_common_pct']:.1f}%",
                'Entropy': f"{props['entropy']:.2f}",
                'Distribution_Type': 'Balanced' if props['most_common_pct'] < 50 else 'Skewed'
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\\n📋 Categorical Proportions Summary:")
        print(summary_df.to_string(index=False))
        
        # 5. DISTRIBUTION BALANCE ANALYSIS
        print("\\n⚖️  5. DISTRIBUTION BALANCE ANALYSIS")
        
        plt.figure(figsize=(12, 6))
        
        # Plot entropy vs most common percentage
        entropies = [props['entropy'] for props in category_proportions.values()]
        most_common_pcts = [props['most_common_pct'] for props in category_proportions.values()]
        variable_names = list(category_proportions.keys())
        
        scatter = plt.scatter(most_common_pcts, entropies, 
                            s=100, alpha=0.7, c=range(len(entropies)), cmap='viridis')
        
        # Add variable labels
        for i, var in enumerate(variable_names):
            plt.annotate(var[:10], (most_common_pcts[i], entropies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Most Common Category Percentage (%)')
        plt.ylabel('Entropy (Distribution Balance)')
        plt.title('Distribution Balance Analysis\\nHigher Entropy = More Balanced Distribution', 
                 fontweight='bold')
        
        # Add quadrant lines
        plt.axvline(50, color='red', linestyle='--', alpha=0.5, label='50% Dominance')
        median_entropy = np.median(entropies)
        plt.axhline(median_entropy, color='orange', linestyle='--', alpha=0.5, 
                   label=f'Median Entropy ({median_entropy:.2f})')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 6. PIE CHART INSIGHTS
        print("\\n💡 PIE CHART ANALYSIS INSIGHTS:")
        print("-" * 60)
        
        # Find balanced vs skewed distributions
        balanced_vars = [col for col, props in category_proportions.items() 
                        if props['most_common_pct'] < 50]
        skewed_vars = [col for col, props in category_proportions.items() 
                      if props['most_common_pct'] >= 70]
        
        if balanced_vars:
            print(f"\\n🟢 Balanced Distributions (<50% dominance): {len(balanced_vars)}")
            for col in balanced_vars:
                entropy = category_proportions[col]['entropy']
                print(f"   • {col}: entropy = {entropy:.2f} (well distributed)")
        
        if skewed_vars:
            print(f"\\n🔴 Highly Skewed Distributions (≥70% dominance): {len(skewed_vars)}")
            for col in skewed_vars:
                pct = category_proportions[col]['most_common_pct']
                most_common = category_proportions[col]['most_common']
                print(f"   • {col}: '{most_common}' dominates with {pct:.1f}%")
        
        # Find variables with many small categories
        fragmented_vars = [col for col, props in category_proportions.items()
                          if props['categories'] > 6 and props['least_common_pct'] < 5]
        
        if fragmented_vars:
            print(f"\\n🟡 Fragmented Distributions (many small categories): {len(fragmented_vars)}")
            for col in fragmented_vars:
                cats = category_proportions[col]['categories']
                least_pct = category_proportions[col]['least_common_pct']
                print(f"   • {col}: {cats} categories, smallest = {least_pct:.1f}%")
            print("   → Consider grouping rare categories")
        
        # Overall insights
        print("\\n📊 OVERALL PIE CHART INSIGHTS:")
        print("-" * 60)
        avg_entropy = np.mean(entropies)
        avg_dominance = np.mean(most_common_pcts)
        
        print(f"   • Variables analyzed: {len(category_proportions)}")
        print(f"   • Average entropy (balance): {avg_entropy:.2f}")
        print(f"   • Average dominant category: {avg_dominance:.1f}%")
        print(f"   • Balanced distributions: {len(balanced_vars)}")
        print(f"   • Highly skewed distributions: {len(skewed_vars)}")
        
        if avg_entropy > 2.0:
            print("   ✅ Overall good distribution balance across variables")
        elif avg_entropy < 1.5:
            print("   ⚠️  Many variables show concentration in few categories")
        else:
            print("   ℹ️  Mixed distribution patterns - some balanced, some skewed")

print("\\n" + "="*60)
print("✅ Categorical pie charts analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return CategoricalPieChartsAnalysis