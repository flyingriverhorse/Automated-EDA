"""Multicollinearity Analysis Component.

Provides comprehensive multicollinearity detection and analysis.
"""


class MulticollinearityAnalysis:
    """Analyze multicollinearity among numeric variables."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "multicollinearity_analysis",
            "display_name": "Multicollinearity Analysis",
            "description": "Detect multicollinearity using correlation heatmaps and VIF analysis",
            "category": "relationships",
            "complexity": "advanced",
            "tags": ["multicollinearity", "VIF", "correlation", "regression"],
            "estimated_runtime": "3-10 seconds",
            "icon": "🔗"
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview=None):
        """Check if analysis can be performed on the data."""
        if not data_preview:
            return True
        numeric_cols = data_preview.get('numeric_columns', [])
        return len(numeric_cols) >= 3
    
    @staticmethod
    def generate_code(data_preview=None):
        """Generate code for multicollinearity analysis."""
        
        return '''
# ===== MULTICOLLINEARITY ANALYSIS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🔗 MULTICOLLINEARITY ANALYSIS")
print("="*60)

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\\n📊 Found {len(numeric_cols)} numeric columns for multicollinearity analysis")

if len(numeric_cols) < 3:
    print("❌ Need at least 3 numeric columns for multicollinearity analysis")
else:
    # Remove columns with all NaN or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and not df[col].isna().all():
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns")
    
    if len(valid_cols) < 3:
        print("❌ Not enough valid columns after removing constant/all-NaN columns")
    else:
        # Limit to reasonable number of columns for analysis
        if len(valid_cols) > 30:  # Increased from 20 to 30 columns
            print(f"⚠️  Too many columns ({len(valid_cols)}). Using first 30 for detailed analysis.")
            analysis_cols = valid_cols[:30]
            print(f"📊 Analyzing columns: {', '.join(analysis_cols[:10])}{'...' if len(analysis_cols) > 10 else ''}")
        else:
            analysis_cols = valid_cols
            print(f"📊 Analyzing all {len(analysis_cols)} valid columns")
        
        # Create clean dataset for analysis
        clean_df = df[analysis_cols].dropna()
        print(f"\\n📊 Analysis Dataset:")
        print(f"   • Variables: {len(analysis_cols)}")
        print(f"   • Observations: {len(clean_df):,}")
        print(f"   • Complete cases: {len(clean_df)/len(df)*100:.1f}% of original data")
        
        if len(clean_df) < 10:
            print("❌ Insufficient complete cases for reliable multicollinearity analysis")
        else:
            # 1. CORRELATION MATRIX ANALYSIS
            print(f"\\n🔍 CORRELATION MATRIX ANALYSIS:")
            print("-" * 50)
            
            # Calculate correlation matrix
            corr_matrix = clean_df.corr()
            
            # Create enhanced correlation heatmap
            plt.figure(figsize=(max(10, len(analysis_cols)*0.8), max(8, len(analysis_cols)*0.7)))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Create heatmap
            sns.heatmap(corr_matrix, 
                       mask=mask,
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       square=True,
                       linewidths=0.5,
                       cbar_kws={"shrink": .8},
                       fmt='.2f',
                       annot_kws={'size': max(8, 12-len(analysis_cols)//2)})
            
            plt.title('🔗 Correlation Matrix (Multicollinearity Detection)', fontsize=14, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()
            
            # Find high correlations
            high_corr_pairs = []
            moderate_corr_pairs = []
            
            for i, col1 in enumerate(analysis_cols):
                for j, col2 in enumerate(analysis_cols[i+1:], i+1):
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) >= 0.9:
                        high_corr_pairs.append((col1, col2, corr_val))
                    elif abs(corr_val) >= 0.7:
                        moderate_corr_pairs.append((col1, col2, corr_val))
            
            print(f"\\n🚨 High Correlations (|r| ≥ 0.9): {len(high_corr_pairs)}")
            if high_corr_pairs:
                for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                    direction = "positive" if corr > 0 else "negative"
                    print(f"   • {col1} ↔ {col2}: r = {corr:.3f} ({direction})")
            else:
                print("   ✅ No extremely high correlations found")
            
            print(f"\\n⚠️  Moderate Correlations (0.7 ≤ |r| < 0.9): {len(moderate_corr_pairs)}")
            if moderate_corr_pairs:
                for col1, col2, corr in sorted(moderate_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
                    direction = "positive" if corr > 0 else "negative"
                    print(f"   • {col1} ↔ {col2}: r = {corr:.3f} ({direction})")
            else:
                print("   ✅ No moderate correlations found")
            
            # 2. VARIANCE INFLATION FACTOR (VIF) ANALYSIS
            print(f"\\n📊 VARIANCE INFLATION FACTOR (VIF) ANALYSIS:")
            print("-" * 50)
            
            try:
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                
                vif_results = []
                
                print(f"   Computing VIF for each variable...")
                
                for i, target_col in enumerate(analysis_cols):
                    try:
                        # Prepare data for regression
                        X = clean_df.drop(columns=[target_col])
                        y = clean_df[target_col]
                        
                        if len(X.columns) == 0:
                            continue
                        
                        # Fit regression model
                        reg = LinearRegression()
                        reg.fit(X, y)
                        y_pred = reg.predict(X)
                        
                        # Calculate R²
                        r2 = r2_score(y, y_pred)
                        
                        # Calculate VIF = 1 / (1 - R²)
                        if r2 < 0.999:  # Avoid division by zero
                            vif = 1 / (1 - r2)
                        else:
                            vif = float('inf')
                        
                        vif_results.append((target_col, vif, r2))
                        
                    except Exception as e:
                        print(f"   ⚠️  Error computing VIF for {target_col}: {str(e)[:50]}")
                
                if vif_results:
                    # Sort by VIF value
                    vif_results.sort(key=lambda x: x[1], reverse=True)
                    
                    print(f"\\n📈 VIF Results (sorted by severity):")
                    print(f"{'Variable':<25} | {'VIF':>8} | {'R²':>6} | {'Interpretation'}")
                    print("-" * 70)
                    
                    severe_vif = []
                    high_vif = []
                    moderate_vif = []
                    
                    for col, vif, r2 in vif_results:
                        if vif >= 10:
                            interpretation = "🔴 Severe"
                            severe_vif.append(col)
                        elif vif >= 5:
                            interpretation = "🟠 High"
                            high_vif.append(col)
                        elif vif >= 2.5:
                            interpretation = "🟡 Moderate"
                            moderate_vif.append(col)
                        else:
                            interpretation = "🟢 Low"
                        
                        vif_display = f"{vif:.2f}" if vif != float('inf') else "∞"
                        print(f"{col:<25} | {vif_display:>8} | {r2:>6.3f} | {interpretation}")
                    
                    # VIF Summary and Recommendations
                    print(f"\\n📊 VIF Summary:")
                    print(f"   • Severe multicollinearity (VIF ≥ 10): {len(severe_vif)} variables")
                    print(f"   • High multicollinearity (VIF 5-10): {len(high_vif)} variables")  
                    print(f"   • Moderate multicollinearity (VIF 2.5-5): {len(moderate_vif)} variables")
                    
                    if severe_vif or high_vif:
                        print(f"\\n🚨 MULTICOLLINEARITY RECOMMENDATIONS:")
                        print(f"   Consider removing or combining variables with high VIF:")
                        
                        if severe_vif:
                            print(f"   🔴 Priority removal candidates: {', '.join(severe_vif[:5])}")
                        
                        if high_vif:
                            print(f"   🟠 Secondary candidates: {', '.join(high_vif[:5])}")
                        
                        print(f"\\n   💡 Strategies:")
                        print(f"   • Remove variables with VIF > 10")
                        print(f"   • Use principal component analysis (PCA)")
                        print(f"   • Apply regularization techniques (Ridge/Lasso)")
                        print(f"   • Combine correlated variables into composite scores")
                    
                else:
                    print("   ❌ Could not compute VIF values")
            
            except ImportError:
                print("   ⚠️  VIF analysis skipped (scikit-learn not available)")
                print("   💡 Install scikit-learn for VIF analysis: pip install scikit-learn")
            
            # 3. CONDITION INDEX ANALYSIS
            print(f"\\n📊 CONDITION INDEX ANALYSIS:")
            print("-" * 50)
            
            try:
                # Standardize the data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                standardized_data = scaler.fit_transform(clean_df)
                
                # Add constant term (intercept)
                X_with_constant = np.column_stack([np.ones(len(standardized_data)), standardized_data])
                
                # Compute SVD
                U, s, Vt = np.linalg.svd(X_with_constant, full_matrices=False)
                
                # Calculate condition indices
                condition_indices = s[0] / s  # Ratio of largest to each singular value
                
                print(f"   Condition Index Analysis:")
                print(f"   • Number of eigenvalues: {len(s)}")
                print(f"   • Condition number: {condition_indices[-1]:.2f}")
                
                if condition_indices[-1] > 30:
                    print(f"   🔴 Severe multicollinearity detected (CN > 30)")
                elif condition_indices[-1] > 15:
                    print(f"   🟠 Moderate multicollinearity detected (CN > 15)")
                else:
                    print(f"   🟢 No serious multicollinearity (CN ≤ 15)")
                
                # Count problematic condition indices
                severe_count = np.sum(condition_indices > 30)
                moderate_count = np.sum((condition_indices > 15) & (condition_indices <= 30))
                
                print(f"   • Severe condition indices (> 30): {severe_count}")
                print(f"   • Moderate condition indices (15-30): {moderate_count}")
            
            except Exception as e:
                print(f"   ⚠️  Condition index analysis error: {str(e)[:50]}")
        
        # Overall Assessment
        print("\\n" + "="*60)
        print("📊 MULTICOLLINEARITY ASSESSMENT SUMMARY")
        print("="*60)
        
        total_high_corr = len(high_corr_pairs) + len(moderate_corr_pairs)
        
        print(f"\\n🔍 Assessment Results:")
        print(f"   • Variables analyzed: {len(analysis_cols)}")
        print(f"   • High correlation pairs: {len(high_corr_pairs)}")
        print(f"   • Moderate correlation pairs: {len(moderate_corr_pairs)}")
        
        if 'vif_results' in locals() and vif_results:
            severe_vif_count = len([vif for _, vif, _ in vif_results if vif >= 10])
            high_vif_count = len([vif for _, vif, _ in vif_results if 5 <= vif < 10])
            print(f"   • Variables with severe VIF: {severe_vif_count}")
            print(f"   • Variables with high VIF: {high_vif_count}")
        
        # Overall multicollinearity assessment
        if (len(high_corr_pairs) > 0 or 
            ('vif_results' in locals() and any(vif >= 10 for _, vif, _ in vif_results))):
            print(f"\\n🚨 OVERALL ASSESSMENT: Severe multicollinearity detected")
            print(f"   Immediate action recommended for reliable modeling")
        elif (len(moderate_corr_pairs) > len(analysis_cols) * 0.1 or
              ('vif_results' in locals() and any(vif >= 5 for _, vif, _ in vif_results))):
            print(f"\\n⚠️  OVERALL ASSESSMENT: Moderate multicollinearity detected")  
            print(f"   Consider addressing before final modeling")
        else:
            print(f"\\n✅ OVERALL ASSESSMENT: Low multicollinearity")
            print(f"   Variables appear suitable for regression modeling")

print("\\n" + "="*60)
print("✅ Multicollinearity analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return MulticollinearityAnalysis