"""Correlation Analysis Component.

Provides comprehensive correlation analysis with multiple correlation methods.
"""


class CorrelationAnalysis:
    """Analyze correlations between numeric variables."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "correlation_analysis",
            "display_name": "Correlation Analysis",
            "description": "Correlation matrix analysis (Pearson/Spearman/Kendall)",
            "category": "bivariate",
            "complexity": "intermediate",
            "tags": ["correlation", "pearson", "spearman", "kendall", "relationships"],
            "estimated_runtime": "2-5 seconds",
            "icon": "🔗"
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
        """Generate code for correlation analysis."""
        
        return '''
# ===== CORRELATION ANALYSIS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau

print("="*60)
print("🔗 CORRELATION ANALYSIS")
print("="*60)

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\\n📊 Found {len(numeric_cols)} numeric columns for correlation analysis")

if len(numeric_cols) < 2:
    print("❌ Need at least 2 numeric columns for correlation analysis")
else:
    # Remove columns with all NaN or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and not df[col].isna().all():
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns for correlation")
    
    if len(valid_cols) < 2:
        print("❌ Not enough valid columns after removing constant/all-NaN columns")
    else:
        # Calculate different types of correlations
        correlations = {}
        
        # Pearson Correlation
        print("\\n🔍 Computing Pearson correlations...")
        pearson_corr = df[valid_cols].corr(method='pearson')
        correlations['Pearson'] = pearson_corr
        
        # Spearman Correlation  
        print("🔍 Computing Spearman correlations...")
        spearman_corr = df[valid_cols].corr(method='spearman')
        correlations['Spearman'] = spearman_corr
        
        # Kendall Correlation (if not too many columns)
        if len(valid_cols) <= 15:  # Kendall is computationally expensive
            print("🔍 Computing Kendall correlations...")
            kendall_corr = df[valid_cols].corr(method='kendall')
            correlations['Kendall'] = kendall_corr
        else:
            print("⚠️  Skipping Kendall correlation (too many columns)")
        
        # Create correlation visualization
        n_methods = len(correlations)
        fig, axes = plt.subplots(1, n_methods, figsize=(8*n_methods, 6))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method, corr_matrix) in enumerate(correlations.items()):
            ax = axes[idx]
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
            sns.heatmap(corr_matrix, 
                       mask=mask,
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       square=True,
                       linewidths=0.5,
                       cbar_kws={"shrink": .8},
                       ax=ax,
                       fmt='.2f')
            
            ax.set_title(f'{method} Correlation Matrix', fontweight='bold', pad=20)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        # Find and report strong correlations
        print("\\n🔍 STRONG CORRELATIONS DETECTED:")
        print("-" * 50)
        
        for method, corr_matrix in correlations.items():
            print(f"\\n📊 {method} Correlations:")
            
            # Get upper triangle of correlation matrix
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find strong correlations (absolute value >= 0.7)
            strong_corrs = []
            for col1 in upper_triangle.columns:
                for col2 in upper_triangle.index:
                    corr_val = upper_triangle.loc[col2, col1]
                    if pd.notna(corr_val) and abs(corr_val) >= 0.7:
                        strong_corrs.append((col1, col2, corr_val))
            
            if strong_corrs:
                strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
                for col1, col2, corr_val in strong_corrs:
                    strength = "🔴 Very Strong" if abs(corr_val) >= 0.9 else "🟠 Strong"
                    direction = "positive" if corr_val > 0 else "negative"
                    print(f"   {strength}: {col1} ↔ {col2} = {corr_val:.3f} ({direction})")
            else:
                print("   ✅ No strong correlations (|r| >= 0.7) found")
        
        # Correlation strength distribution
        print("\\n📊 CORRELATION STRENGTH DISTRIBUTION:")
        print("-" * 50)
        
        for method, corr_matrix in correlations.items():
            print(f"\\n📈 {method} Distribution:")
            
            # Get all correlation values (excluding diagonal)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            all_corrs = corr_matrix.values[mask]
            all_corrs = all_corrs[~np.isnan(all_corrs)]
            
            if len(all_corrs) > 0:
                # Categorize correlations
                very_weak = np.sum(np.abs(all_corrs) < 0.3)
                weak = np.sum((np.abs(all_corrs) >= 0.3) & (np.abs(all_corrs) < 0.5))
                moderate = np.sum((np.abs(all_corrs) >= 0.5) & (np.abs(all_corrs) < 0.7))
                strong = np.sum((np.abs(all_corrs) >= 0.7) & (np.abs(all_corrs) < 0.9))
                very_strong = np.sum(np.abs(all_corrs) >= 0.9)
                total = len(all_corrs)
                
                print(f"   Very Weak (|r| < 0.3):   {very_weak:3d} ({very_weak/total*100:5.1f}%)")
                print(f"   Weak (0.3 ≤ |r| < 0.5):   {weak:3d} ({weak/total*100:5.1f}%)")
                print(f"   Moderate (0.5 ≤ |r| < 0.7): {moderate:3d} ({moderate/total*100:5.1f}%)")
                print(f"   Strong (0.7 ≤ |r| < 0.9):   {strong:3d} ({strong/total*100:5.1f}%)")
                print(f"   Very Strong (|r| ≥ 0.9):   {very_strong:3d} ({very_strong/total*100:5.1f}%)")
        
        # Generate insights and recommendations
        print("\\n💡 INSIGHTS AND RECOMMENDATIONS:")
        print("-" * 50)
        
        # Check for multicollinearity issues
        pearson_matrix = correlations.get('Pearson', correlations[list(correlations.keys())[0]])
        upper_triangle = pearson_matrix.where(
            np.triu(np.ones(pearson_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_pairs = []
        for col1 in upper_triangle.columns:
            for col2 in upper_triangle.index:
                corr_val = upper_triangle.loc[col2, col1]
                if pd.notna(corr_val) and abs(corr_val) >= 0.8:
                    high_corr_pairs.append((col1, col2, corr_val))
        
        if high_corr_pairs:
            print("\\n⚠️  Potential Multicollinearity Issues:")
            print("   Consider removing one variable from highly correlated pairs:")
            for col1, col2, corr_val in high_corr_pairs:
                print(f"   • {col1} and {col2} (r = {corr_val:.3f})")
        
        print("\\n📋 Analysis Summary:")
        print(f"   • Total variable pairs analyzed: {len(all_corrs)}")
        print(f"   • Strongest positive correlation: {np.max(all_corrs):.3f}")  
        print(f"   • Strongest negative correlation: {np.min(all_corrs):.3f}")
        print(f"   • Average absolute correlation: {np.mean(np.abs(all_corrs)):.3f}")

print("\\n" + "="*60)
print("✅ Correlation analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return CorrelationAnalysis