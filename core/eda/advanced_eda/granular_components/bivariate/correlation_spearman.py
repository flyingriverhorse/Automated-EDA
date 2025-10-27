"""Spearman Correlation Analysis Component.

Focused component specifically for Spearman rank correlation analysis.
"""


class SpearmanCorrelationAnalysis:
    """Analyze Spearman rank correlations between numeric variables."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "spearman_correlation",
            "display_name": "Spearman Correlation",
            "description": "Rank-based correlation analysis using Spearman correlation coefficient",
            "category": "bivariate",
            "subcategory": "correlation",
            "complexity": "intermediate",
            "tags": ["correlation", "spearman", "rank", "monotonic", "relationships"],
            "estimated_runtime": "2-3 seconds",
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
        """Generate code for Spearman correlation analysis."""
        
        return '''
# ===== SPEARMAN RANK CORRELATION ANALYSIS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

print("="*60)
print("📈 SPEARMAN RANK CORRELATION ANALYSIS")
print("="*60)

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\\n📊 Found {len(numeric_cols)} numeric columns for Spearman correlation")

if len(numeric_cols) < 2:
    print("❌ Need at least 2 numeric columns for correlation analysis")
else:
    # Remove columns with all NaN or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and not df[col].isna().all():
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns for Spearman correlation")
    
    if len(valid_cols) < 2:
        print("❌ Not enough valid columns after removing constant/all-NaN columns")
    else:
        # Calculate Spearman correlation
        print("\\n🔍 Computing Spearman rank correlations...")
        spearman_corr = df[valid_cols].corr(method='spearman')
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(spearman_corr, dtype=bool))  # Mask upper triangle
        sns.heatmap(spearman_corr, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": .8},
                   fmt='.3f')
        
        plt.title('Spearman Rank Correlation Matrix', fontweight='bold', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Find and report strong Spearman correlations
        print("\\n🔍 STRONG SPEARMAN CORRELATIONS DETECTED:")
        print("-" * 60)
        
        # Get upper triangle of correlation matrix
        upper_triangle = spearman_corr.where(
            np.triu(np.ones(spearman_corr.shape), k=1).astype(bool)
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
                direction = "positive monotonic" if corr_val > 0 else "negative monotonic"
                print(f"   {strength}: {col1} ↔ {col2} = {corr_val:.3f} ({direction})")
        else:
            print("   ✅ No strong Spearman correlations (|ρ| >= 0.7) found")
        
        # Spearman correlation strength distribution
        print("\\n📊 SPEARMAN CORRELATION STRENGTH DISTRIBUTION:")
        print("-" * 60)
        
        # Get all correlation values (excluding diagonal)
        mask = ~np.eye(spearman_corr.shape[0], dtype=bool)
        all_corrs = spearman_corr.values[mask]
        all_corrs = all_corrs[~np.isnan(all_corrs)]
        
        if len(all_corrs) > 0:
            # Categorize correlations
            very_weak = np.sum(np.abs(all_corrs) < 0.3)
            weak = np.sum((np.abs(all_corrs) >= 0.3) & (np.abs(all_corrs) < 0.5))
            moderate = np.sum((np.abs(all_corrs) >= 0.5) & (np.abs(all_corrs) < 0.7))
            strong = np.sum((np.abs(all_corrs) >= 0.7) & (np.abs(all_corrs) < 0.9))
            very_strong = np.sum(np.abs(all_corrs) >= 0.9)
            total = len(all_corrs)
            
            print(f"   Very Weak (|ρ| < 0.3):   {very_weak:3d} ({very_weak/total*100:5.1f}%)")
            print(f"   Weak (0.3 ≤ |ρ| < 0.5):   {weak:3d} ({weak/total*100:5.1f}%)")
            print(f"   Moderate (0.5 ≤ |ρ| < 0.7): {moderate:3d} ({moderate/total*100:5.1f}%)")
            print(f"   Strong (0.7 ≤ |ρ| < 0.9):   {strong:3d} ({strong/total*100:5.1f}%)")
            print(f"   Very Strong (|ρ| ≥ 0.9):   {very_strong:3d} ({very_strong/total*100:5.1f}%)")
        
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
                    corr_coef, p_value = spearmanr(df.loc[common_idx, col1], df.loc[common_idx, col2])
                    if p_value < 0.05:
                        significant_pairs.append((col1, col2, corr_coef, p_value))
        
        if significant_pairs:
            print("\\n🔍 Statistically Significant Spearman Correlations (p < 0.05):")
            significant_pairs.sort(key=lambda x: x[3])  # Sort by p-value
            for col1, col2, corr_val, p_val in significant_pairs:
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                print(f"   {col1} ↔ {col2}: ρ = {corr_val:.3f}, p = {p_val:.4f} {significance}")
        else:
            print("   ℹ️  No statistically significant Spearman correlations found at α = 0.05")
        
        # Compare with Pearson correlation for insights
        print("\\n📊 SPEARMAN VS LINEAR RELATIONSHIP COMPARISON:")
        print("-" * 60)
        
        pearson_corr = df[valid_cols].corr(method='pearson')
        
        # Find cases where Spearman is much stronger than Pearson (non-linear relationships)
        nonlinear_relationships = []
        for i, col1 in enumerate(valid_cols):
            for col2 in valid_cols[i+1:]:
                spear_val = spearman_corr.loc[col1, col2]
                pears_val = pearson_corr.loc[col1, col2]
                
                if pd.notna(spear_val) and pd.notna(pears_val):
                    diff = abs(spear_val) - abs(pears_val)
                    if diff > 0.2 and abs(spear_val) > 0.5:  # Spearman much stronger
                        nonlinear_relationships.append((col1, col2, spear_val, pears_val, diff))
        
        if nonlinear_relationships:
            print("\\n🔍 Potential Non-Linear Relationships Detected:")
            nonlinear_relationships.sort(key=lambda x: x[4], reverse=True)
            for col1, col2, spear, pears, diff in nonlinear_relationships:
                print(f"   {col1} ↔ {col2}: Spearman={spear:.3f}, Pearson={pears:.3f} (diff={diff:.3f})")
                print(f"      → Suggests monotonic but non-linear relationship")
        else:
            print("   ℹ️  No strong evidence of non-linear monotonic relationships found")
        
        # Generate insights specific to Spearman correlation
        print("\\n💡 SPEARMAN CORRELATION INSIGHTS:")
        print("-" * 60)
        print("   • Spearman correlation measures MONOTONIC relationships (not just linear)")
        print("   • Based on ranks, so robust to outliers and non-normal distributions")
        print("   • Better for ordinal data or when relationships are non-linear")
        print("   • Values range from -1 to +1, same interpretation as Pearson")
        
        # Check for potential multicollinearity using Spearman
        high_corr_pairs = [pair for pair in strong_corrs if abs(pair[2]) >= 0.8]
        if high_corr_pairs:
            print("\\n⚠️  Potential Multicollinearity Issues (Monotonic):")
            print("   Consider removing one variable from highly correlated pairs:")
            for col1, col2, corr_val in high_corr_pairs:
                print(f"   • {col1} and {col2} (ρ = {corr_val:.3f})")
        
        print("\\n📋 Spearman Analysis Summary:")
        print(f"   • Total variable pairs analyzed: {len(all_corrs)}")
        print(f"   • Strongest positive correlation: {np.max(all_corrs):.3f}")  
        print(f"   • Strongest negative correlation: {np.min(all_corrs):.3f}")
        print(f"   • Average absolute correlation: {np.mean(np.abs(all_corrs)):.3f}")

print("\\n" + "="*60)
print("✅ Spearman correlation analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return SpearmanCorrelationAnalysis