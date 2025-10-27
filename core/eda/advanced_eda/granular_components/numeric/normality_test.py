"""Normality Test Analysis Component.

Focused analysis component for testing normality of numeric variables with statistical tests.
"""
from typing import Dict, Any


class NormalityTestAnalysis:
    """Focused component for normality testing"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "normality_test",
            "display_name": "Normality Test",
            "description": "Statistical tests for normality including Shapiro-Wilk, Kolmogorov-Smirnov, and Q-Q plots",
            "category": "univariate",
            "complexity": "advanced",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-30 seconds",
            "icon": "chart-line",
            "tags": ["normality", "statistical-tests", "distribution", "hypothesis-testing"]
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
        """Generate focused normality test analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=== NORMALITY TEST ANALYSIS ===")
print("Comprehensive statistical testing for normal distribution")
print()

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    print("❌ NO NUMERIC COLUMNS FOUND")
    print("   This analysis requires numeric data.")
else:
    print(f"📊 TESTING NORMALITY FOR {len(numeric_cols)} NUMERIC COLUMNS")
    print()
    
    # Explanation of normality tests
    print("📚 NORMALITY TESTS OVERVIEW")
    print("   🔬 Tests performed:")
    print("      • Shapiro-Wilk: Most powerful for small samples (n < 50)")
    print("      • D'Agostino-Pearson: Good for medium samples (50 < n < 2000)")
    print("      • Kolmogorov-Smirnov: General goodness-of-fit test")
    print("      • Anderson-Darling: More sensitive to tail differences")
    print()
    print("   📈 Interpretation:")
    print("      • p-value > 0.05: Fail to reject normality (likely normal)")
    print("      • p-value ≤ 0.05: Reject normality (likely not normal)")
    print("      • Visual inspection is also important!")
    print()
    
    # Perform normality tests for each column
    normality_results = []
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        
        if len(col_data) < 3:
            print(f"⚠️  {col}: Too few data points for normality testing")
            continue
        
        print(f"🔍 TESTING: {col.upper()}")
        print(f"   Sample size: {len(col_data)}")
        
        results = {'Column': col, 'Sample_Size': len(col_data)}
        
        # 1. Shapiro-Wilk test (best for small samples)
        if len(col_data) <= 5000:  # Shapiro-Wilk has sample size limit
            try:
                shapiro_stat, shapiro_p = stats.shapiro(col_data)
                results['Shapiro_Stat'] = round(shapiro_stat, 4)
                results['Shapiro_p'] = round(shapiro_p, 6)
                print(f"   📊 Shapiro-Wilk: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.6f}")
                
                if shapiro_p > 0.05:
                    print(f"      ✅ Normal (p > 0.05)")
                else:
                    print(f"      ❌ Not normal (p ≤ 0.05)")
            except Exception as e:
                results['Shapiro_Stat'] = None
                results['Shapiro_p'] = None
                print(f"   ⚠️  Shapiro-Wilk failed: {e}")
        else:
            results['Shapiro_Stat'] = None
            results['Shapiro_p'] = None
            print(f"   ⚠️  Shapiro-Wilk: Sample too large (n > 5000)")
        
        # 2. D'Agostino-Pearson test
        if len(col_data) >= 20:  # Requires at least 20 samples
            try:
                dagostino_stat, dagostino_p = stats.normaltest(col_data)
                results['DAgostino_Stat'] = round(dagostino_stat, 4)
                results['DAgostino_p'] = round(dagostino_p, 6)
                print(f"   📊 D'Agostino-Pearson: statistic={dagostino_stat:.4f}, p-value={dagostino_p:.6f}")
                
                if dagostino_p > 0.05:
                    print(f"      ✅ Normal (p > 0.05)")
                else:
                    print(f"      ❌ Not normal (p ≤ 0.05)")
            except Exception as e:
                results['DAgostino_Stat'] = None
                results['DAgostino_p'] = None
                print(f"   ⚠️  D'Agostino-Pearson failed: {e}")
        else:
            results['DAgostino_Stat'] = None
            results['DAgostino_p'] = None
            print(f"   ⚠️  D'Agostino-Pearson: Sample too small (n < 20)")
        
        # 3. Kolmogorov-Smirnov test
        try:
            # Standardize data for comparison with standard normal
            standardized_data = (col_data - col_data.mean()) / col_data.std()
            ks_stat, ks_p = stats.kstest(standardized_data, 'norm')
            results['KS_Stat'] = round(ks_stat, 4)
            results['KS_p'] = round(ks_p, 6)
            print(f"   📊 Kolmogorov-Smirnov: statistic={ks_stat:.4f}, p-value={ks_p:.6f}")
            
            if ks_p > 0.05:
                print(f"      ✅ Normal (p > 0.05)")
            else:
                print(f"      ❌ Not normal (p ≤ 0.05)")
        except Exception as e:
            results['KS_Stat'] = None
            results['KS_p'] = None
            print(f"   ⚠️  Kolmogorov-Smirnov failed: {e}")
        
        # 4. Anderson-Darling test
        try:
            ad_result = stats.anderson(col_data, dist='norm')
            results['AD_Stat'] = round(ad_result.statistic, 4)
            
            print(f"   📊 Anderson-Darling: statistic={ad_result.statistic:.4f}")
            
            # Check critical values
            significance_levels = [15, 10, 5, 2.5, 1]  # Percentages
            critical_values = ad_result.critical_values
            
            normal_at_levels = []
            for i, (level, critical_val) in enumerate(zip(significance_levels, critical_values)):
                if ad_result.statistic < critical_val:
                    normal_at_levels.append(f"{level}%")
            
            if normal_at_levels:
                results['AD_Normal_Levels'] = ', '.join(normal_at_levels)
                print(f"      ✅ Normal at significance levels: {', '.join(normal_at_levels)}")
            else:
                results['AD_Normal_Levels'] = 'None'
                print(f"      ❌ Not normal at any standard significance level")
                
        except Exception as e:
            results['AD_Stat'] = None
            results['AD_Normal_Levels'] = None
            print(f"   ⚠️  Anderson-Darling failed: {e}")
        
        # Consensus conclusion
        tests_normal = 0
        total_tests = 0
        
        if results.get('Shapiro_p') is not None:
            total_tests += 1
            if results['Shapiro_p'] > 0.05:
                tests_normal += 1
        
        if results.get('DAgostino_p') is not None:
            total_tests += 1
            if results['DAgostino_p'] > 0.05:
                tests_normal += 1
        
        if results.get('KS_p') is not None:
            total_tests += 1
            if results['KS_p'] > 0.05:
                tests_normal += 1
        
        if total_tests > 0:
            consensus_ratio = tests_normal / total_tests
            if consensus_ratio >= 0.7:
                consensus = "Likely Normal"
            elif consensus_ratio >= 0.3:
                consensus = "Mixed Results"
            else:
                consensus = "Likely Not Normal"
        else:
            consensus = "Inconclusive"
        
        results['Consensus'] = consensus
        print(f"   🎯 CONSENSUS: {consensus} ({tests_normal}/{total_tests} tests suggest normality)")
        print()
        
        normality_results.append(results)
    
    # Create summary table
    if normality_results:
        results_df = pd.DataFrame(normality_results)
        
        print("📋 NORMALITY TEST SUMMARY TABLE")
        display_cols = ['Column', 'Sample_Size', 'Shapiro_p', 'DAgostino_p', 'KS_p', 'Consensus']
        available_cols = [col for col in display_cols if col in results_df.columns]
        print(results_df[available_cols].to_string(index=False))
        print()
    
    # Visual normality assessment with Q-Q plots
    if len(numeric_cols) > 0:
        print("📈 VISUAL NORMALITY ASSESSMENT (Q-Q PLOTS)")
        
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        n_plot_cols = min(3, n_cols)
        
        fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(5*n_plot_cols, 4*n_rows))
        fig.suptitle('Q-Q Plots: Visual Normality Assessment', fontsize=16, fontweight='bold')
        
        # Ensure axes is always a 2D array for consistent indexing
        if n_rows == 1 and n_plot_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_plot_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(numeric_cols):
            row = i // 3
            col_idx = i % 3
            
            ax = axes[row, col_idx]
            
            col_data = df[col].dropna()
            
            if len(col_data) > 2:
                # Q-Q plot
                stats.probplot(col_data, dist="norm", plot=ax)
                ax.set_title(f'{col}\\nQ-Q Plot vs Normal Distribution')
                ax.grid(True, alpha=0.3)
                
                # Add correlation coefficient for linearity
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(col_data)))
                sample_quantiles = np.sort(col_data)
                correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
                ax.text(0.05, 0.95, f'R² = {correlation**2:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Remove empty subplots
        if n_cols < n_rows * n_plot_cols:
            for i in range(n_cols, n_rows * n_plot_cols):
                row = i // 3
                col_idx = i % 3
                ax_to_remove = axes[row, col_idx]
                fig.delaxes(ax_to_remove)
        
        plt.tight_layout()
        plt.show()
        print()
        
        print("📖 Q-Q PLOT INTERPRETATION:")
        print("   • Points close to diagonal line → Normal distribution")
        print("   • S-shaped curve → Heavy tails (platykurtic)")
        print("   • Inverted S-shaped curve → Light tails (leptokurtic)")
        print("   • Curved upward → Right skewed")
        print("   • Curved downward → Left skewed")
        print("   • R² > 0.95 → Very close to normal")
        print("   • R² > 0.90 → Reasonably normal")
        print("   • R² < 0.90 → Questionable normality")
    
    # Summary by consensus
    if normality_results:
        consensus_summary = results_df['Consensus'].value_counts()
        
        print()
        print("📊 OVERALL NORMALITY ASSESSMENT")
        for consensus_type, count in consensus_summary.items():
            percentage = (count / len(results_df)) * 100
            print(f"   {consensus_type}: {count} columns ({percentage:.1f}%)")
        
        # Recommendations based on results
        likely_normal = results_df[results_df['Consensus'] == 'Likely Normal']['Column'].tolist()
        likely_not_normal = results_df[results_df['Consensus'] == 'Likely Not Normal']['Column'].tolist()
        mixed_results = results_df[results_df['Consensus'] == 'Mixed Results']['Column'].tolist()
        
        print()
        print("💡 RECOMMENDATIONS")
        
        if likely_normal:
            print(f"   ✅ Normal columns ({len(likely_normal)}): {likely_normal}")
            print("      → Safe to use parametric tests (t-test, ANOVA, linear regression)")
            print("      → Mean and standard deviation are appropriate measures")
        
        if mixed_results:
            print(f"   🟡 Mixed results ({len(mixed_results)}): {mixed_results}")
            print("      → Consider visual inspection and domain knowledge")
            print("      → May be acceptable for robust parametric methods")
            print("      → Consider sample size effects on test sensitivity")
        
        if likely_not_normal:
            print(f"   ❌ Non-normal columns ({len(likely_not_normal)}): {likely_not_normal}")
            print("      → Consider non-parametric tests (Mann-Whitney, Kruskal-Wallis)")
            print("      → Use median and IQR instead of mean and std")
            print("      → Consider data transformation before parametric analysis")
        
        print()
        print("   🔄 Transformation suggestions for non-normal data:")
        print("      • Right-skewed: log, square root, Box-Cox")
        print("      • Left-skewed: square, exponential")
        print("      • Always validate transformation effectiveness")

print("\\n" + "="*50)
'''