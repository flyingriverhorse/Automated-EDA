"""Pearson Correlation Analysis Component.

Focused component specifically for Pearson correlation analysis.
"""


class PearsonCorrelationAnalysis:
    """Analyze Pearson correlations between numeric variables."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "pearson_correlation",
            "display_name": "Pearson Correlation",
            "description": "Linear correlation analysis using Pearson correlation coefficient",
            "category": "bivariate",
            "subcategory": "correlation",
            "complexity": "intermediate",
            "tags": ["correlation", "pearson", "linear", "relationships"],
            "estimated_runtime": "2-3 seconds",
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
        """Generate code for Pearson correlation analysis."""
        
        return '''
# ===== PEARSON CORRELATION ANALYSIS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

print("="*60)
print("🔗 PEARSON CORRELATION ANALYSIS")
print("="*60)

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\\n📊 Found {len(numeric_cols)} numeric columns for Pearson correlation")

if len(numeric_cols) < 2:
    print("❌ Need at least 2 numeric columns for correlation analysis")
else:
    # Remove columns with all NaN or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and not df[col].isna().all():
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns for Pearson correlation")
    
    if len(valid_cols) < 2:
        print("❌ Not enough valid columns after removing constant/all-NaN columns")
    else:
        # Calculate Pearson correlation
        print("\\n🔍 Computing Pearson correlations...")
        pearson_corr = df[valid_cols].corr(method='pearson')
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))  # Mask upper triangle
        sns.heatmap(pearson_corr, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": .8},
                   fmt='.3f')
        
        plt.title('Pearson Correlation Matrix', fontweight='bold', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Find and report strong Pearson correlations
        print("\\n🔍 STRONG PEARSON CORRELATIONS DETECTED:")
        print("-" * 60)
        
        # Get upper triangle of correlation matrix
        upper_triangle = pearson_corr.where(
            np.triu(np.ones(pearson_corr.shape), k=1).astype(bool)
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
            print("   ✅ No strong Pearson correlations (|r| >= 0.7) found")
        
        # Pearson correlation strength distribution
        print("\\n📊 PEARSON CORRELATION STRENGTH DISTRIBUTION:")
        print("-" * 60)
        
        # Get all correlation values (excluding diagonal)
        mask = ~np.eye(pearson_corr.shape[0], dtype=bool)
        all_corrs = pearson_corr.values[mask]
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
        
        # Statistical significance testing
        print("\\n📊 STATISTICAL SIGNIFICANCE TESTING:")
        print("-" * 60)
        
        significant_pairs = []
        for i, col1 in enumerate(valid_cols):
            for col2 in valid_cols[i+1:]:
                data1 = df[col1].dropna()
                data2 = df[col2].dropna()
                
                # Find common indices
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) > 2:
                    corr_coef, p_value = pearsonr(df.loc[common_idx, col1], df.loc[common_idx, col2])
                    if p_value < 0.05:
                        significant_pairs.append((col1, col2, corr_coef, p_value))
        
        if significant_pairs:
            print("\\n🔍 Statistically Significant Correlations (p < 0.05):")
            significant_pairs.sort(key=lambda x: x[3])  # Sort by p-value
            for col1, col2, corr_val, p_val in significant_pairs:
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                print(f"   {col1} ↔ {col2}: r = {corr_val:.3f}, p = {p_val:.4f} {significance}")
        else:
            print("   ℹ️  No statistically significant correlations found at α = 0.05")
        
        # Generate insights specific to Pearson correlation
        print("\\n💡 PEARSON CORRELATION INSIGHTS:")
        print("-" * 60)
        print("   • Pearson correlation measures LINEAR relationships only")
        print("   • Assumes variables are normally distributed")
        print("   • Sensitive to outliers - consider robust alternatives if needed")
        
        # Check for potential multicollinearity issues
        high_corr_pairs = [pair for pair in strong_corrs if abs(pair[2]) >= 0.8]
        if high_corr_pairs:
            print("\\n⚠️  Potential Multicollinearity Issues:")
            print("   Consider removing one variable from highly correlated pairs:")
            for col1, col2, corr_val in high_corr_pairs:
                print(f"   • {col1} and {col2} (r = {corr_val:.3f})")
        
        print("\\n📋 Pearson Analysis Summary:")
        print(f"   • Total variable pairs analyzed: {len(all_corrs)}")
        print(f"   • Strongest positive correlation: {np.max(all_corrs):.3f}")  
        print(f"   • Strongest negative correlation: {np.min(all_corrs):.3f}")
        print(f"   • Average absolute correlation: {np.mean(np.abs(all_corrs)):.3f}")

print("\\n" + "="*60)
print("✅ Pearson correlation analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return PearsonCorrelationAnalysis