"""Cross-Tabulation Analysis Component.

Provides cross-tabulation analysis for categorical vs categorical variables.
"""


class CrossTabulationAnalysis:
    """Analyze relationships between categorical variables using cross-tabulation."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "cross_tabulation_analysis",
            "display_name": "Cross-Tabulation Analysis",
            "description": "Cross-tabulations and chi-square tests for categorical vs categorical relationships",
            "category": "bivariate",
            "complexity": "intermediate",
            "tags": ["cross-tab", "categorical", "chi-square", "contingency"],
            "estimated_runtime": "2-5 seconds",
            "icon": "📋"
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview=None):
        """Check if analysis can be performed on the data."""
        if not data_preview:
            return True
        categorical_cols = data_preview.get('categorical_columns', []) + data_preview.get('object_columns', [])
        return len(categorical_cols) >= 2
    
    @staticmethod
    def generate_code(data_preview=None):
        """Generate code for cross-tabulation analysis."""
        
        return '''
# ===== CROSS-TABULATION ANALYSIS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from itertools import combinations

print("="*60)
print("📋 CROSS-TABULATION ANALYSIS")
print("="*60)

# Get categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
print(f"\\n📊 Found {len(categorical_cols)} categorical columns")

if len(categorical_cols) < 2:
    print("❌ Need at least 2 categorical columns for cross-tabulation analysis")
else:
    # Filter columns with reasonable cardinality for cross-tabulation
    suitable_cols = []
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if 2 <= unique_count <= 20:  # Reasonable range for cross-tabs
            suitable_cols.append(col)
        elif unique_count > 20:
            print(f"⚠️  Skipping '{col}' - too many categories ({unique_count})")
        elif unique_count < 2:
            print(f"⚠️  Skipping '{col}' - not enough categories ({unique_count})")
    
    print(f"📋 Using {len(suitable_cols)} suitable columns for cross-tabulation")
    
    if len(suitable_cols) < 2:
        print("❌ Not enough suitable columns for cross-tabulation analysis")
    else:
        # Generate all possible pairs
        all_pairs = list(combinations(suitable_cols, 2))
        print(f"📊 Analyzing {len(all_pairs)} categorical variable pairs")
        
        # Limit to reasonable number of pairs
        max_pairs = 10
        if len(all_pairs) > max_pairs:
            print(f"⚠️  Too many pairs. Analyzing first {max_pairs} pairs.")
            selected_pairs = all_pairs[:max_pairs]
        else:
            selected_pairs = all_pairs
        
        # Store results for summary
        significant_associations = []
        
        for pair_idx, (col1, col2) in enumerate(selected_pairs):
            print(f"\\n{'='*60}")
            print(f"📋 CROSS-TABULATION: {col1} vs {col2}")
            print('='*60)
            
            # Create cross-tabulation
            try:
                # Remove NaN values for this analysis
                clean_data = df[[col1, col2]].dropna()
                
                if len(clean_data) == 0:
                    print("❌ No valid data for this pair after removing NaN values")
                    continue
                
                crosstab = pd.crosstab(clean_data[col1], clean_data[col2], margins=True)
                print(f"\\n📊 Cross-tabulation Table:")
                print(crosstab)
                
                # Chi-square test (exclude margins row/column)
                crosstab_no_margins = pd.crosstab(clean_data[col1], clean_data[col2])
                
                if crosstab_no_margins.shape[0] >= 2 and crosstab_no_margins.shape[1] >= 2:
                    chi2, p_value, dof, expected = chi2_contingency(crosstab_no_margins)
                    
                    print(f"\\n🧪 Chi-Square Test Results:")
                    print(f"   • Chi-square statistic: {chi2:.4f}")
                    print(f"   • p-value: {p_value:.6f}")
                    print(f"   • Degrees of freedom: {dof}")
                    
                    # Interpret significance
                    if p_value < 0.001:
                        significance = "Very Strong"
                        emoji = "🔴"
                        significant_associations.append((col1, col2, chi2, p_value, "Very Strong"))
                    elif p_value < 0.01:
                        significance = "Strong"  
                        emoji = "🟠"
                        significant_associations.append((col1, col2, chi2, p_value, "Strong"))
                    elif p_value < 0.05:
                        significance = "Moderate"
                        emoji = "🟡"
                        significant_associations.append((col1, col2, chi2, p_value, "Moderate"))
                    elif p_value < 0.1:
                        significance = "Weak"
                        emoji = "🔵"
                    else:
                        significance = "None"
                        emoji = "⚪"
                    
                    print(f"   • {emoji} Association strength: {significance}")
                    
                    # Cramér's V (effect size)
                    n = crosstab_no_margins.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(crosstab_no_margins.shape) - 1)))
                    print(f"   • Cramér's V (effect size): {cramers_v:.4f}")
                    
                    # Expected vs Observed analysis
                    print(f"\\n🔍 Expected vs Observed Analysis:")
                    max_deviation = 0
                    max_cell = ""
                    
                    for i, row_name in enumerate(crosstab_no_margins.index):
                        for j, col_name in enumerate(crosstab_no_margins.columns):
                            observed = crosstab_no_margins.iloc[i, j]
                            expected_val = expected[i, j]
                            deviation = abs(observed - expected_val) / expected_val if expected_val > 0 else 0
                            
                            if deviation > max_deviation:
                                max_deviation = deviation
                                max_cell = f"{row_name} × {col_name}"
                    
                    if max_deviation > 0:
                        print(f"   • Largest deviation: {max_cell} ({max_deviation:.1%})")
                    
                    # Percentage analysis
                    print(f"\\n📊 Percentage Breakdown:")
                    percentage_crosstab = pd.crosstab(clean_data[col1], clean_data[col2], normalize='index') * 100
                    print(percentage_crosstab.round(1))
                
                else:
                    print("⚠️  Cannot perform chi-square test (insufficient categories)")
                    
            except Exception as e:
                print(f"❌ Error analyzing {col1} vs {col2}: {str(e)}")
        
        # Overall summary
        print("\\n" + "="*60)
        print("📊 CROSS-TABULATION SUMMARY")
        print("="*60)
        
        print(f"\\n📈 Analysis Overview:")
        print(f"   • Pairs analyzed: {len(selected_pairs)}")
        print(f"   • Significant associations found: {len(significant_associations)}")
        
        if significant_associations:
            print(f"\\n🔍 SIGNIFICANT ASSOCIATIONS:")
            print("-" * 50)
            
            # Sort by p-value (most significant first)
            significant_associations.sort(key=lambda x: x[3])
            
            for col1, col2, chi2, p_value, strength in significant_associations:
                print(f"   • {col1} ↔ {col2}:")
                print(f"     - Strength: {strength}")
                print(f"     - Chi² = {chi2:.3f}, p = {p_value:.6f}")
        else:
            print("\\n✅ No significant associations detected at α = 0.05 level")
        
        print(f"\\n💡 INTERPRETATION GUIDE:")
        print(f"   🔴 Very Strong: p < 0.001 (highly significant)")
        print(f"   🟠 Strong: p < 0.01 (very significant)")  
        print(f"   🟡 Moderate: p < 0.05 (significant)")
        print(f"   🔵 Weak: p < 0.1 (marginally significant)")
        print(f"   ⚪ None: p ≥ 0.1 (not significant)")

print("\\n" + "="*60)
print("✅ Cross-tabulation analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return CrossTabulationAnalysis