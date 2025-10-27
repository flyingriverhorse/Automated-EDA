"""
PCA Dimensionality Reduction Analysis Component

This component provides comprehensive Principal Component Analysis (PCA) for
dimensionality reduction with statistical insights and recommendations.
No visualizations - purely statistical analysis.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

class PCADimensionalityReduction:
    """
    PCA dimensionality reduction analysis with statistical insights.
    
    This component performs PCA to analyze data dimensionality and provides
    recommendations for dimensionality reduction. No graphs - statistical only.
    """
    
    def __init__(self):
        """Initialize the PCA dimensionality reduction component."""
        self.name = "PCA Dimensionality Reduction Analysis"
        self.description = "Statistical PCA analysis for dimensionality reduction"
        
    def get_metadata(self):
        """Return metadata for this analysis component."""
        return {
            "name": "pca_analysis",
            "display_name": "PCA Dimensionality Reduction",
            "description": "Statistical PCA analysis for dimensionality reduction (no graphs)",
            "category": "relationships",
            "subcategory": "pca",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "5-10 seconds",
            "tags": ["pca", "dimensionality", "statistical", "reduction"]
        }
        
    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Perform PCA dimensionality reduction analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            **kwargs: Additional parameters
                - n_components (int): Number of components to compute (default: min(samples, features))
                - standardize (bool): Whether to standardize data (default: True)
                - min_variance_explained (float): Minimum variance to explain (default: 0.8)
        
        Returns:
            Dict[str, Any]: Analysis results with components, variance, and recommendations
        """
        try:
            print("\n" + "="*60)
            print("🔍 PCA DIMENSIONALITY REDUCTION ANALYSIS")
            print("="*60)
            
            # Get analysis parameters
            n_components = kwargs.get('n_components', None)
            standardize = kwargs.get('standardize', True)
            min_variance_explained = kwargs.get('min_variance_explained', 0.8)
            
            # Select only numeric columns and handle missing data
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return {"error": "No numeric columns found for PCA analysis"}
            
            print(f"📊 Dataset Overview:")
            print(f"   • Total features: {len(df.columns)}")
            print(f"   • Numeric features: {len(numeric_cols)}")
            print(f"   • Samples: {len(df)}")
            
            # Handle missing values by imputation
            data_subset = df[numeric_cols].copy()
            missing_before = data_subset.isnull().sum().sum()
            
            if missing_before > 0:
                print(f"   • Missing values found: {missing_before}")
                print(f"   • Applying mean imputation...")
                data_subset = data_subset.fillna(data_subset.mean())
            
            # Remove columns with no variance or all NaN
            valid_cols = []
            for col in numeric_cols:
                if data_subset[col].std() > 1e-10:  # Non-zero variance
                    valid_cols.append(col)
                else:
                    print(f"   ⚠️  Removing '{col}': zero/low variance")
            
            if len(valid_cols) < 2:
                return {"error": f"Need at least 2 valid numeric columns for PCA. Found: {len(valid_cols)}"}
                
            print(f"   • Valid features for PCA: {len(valid_cols)}")
            
            # Prepare data for PCA
            X = data_subset[valid_cols].values
            original_dims = X.shape[1]
            
            # Standardize data if requested
            if standardize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                print(f"   • Data standardized (mean=0, std=1)")
            
            # Determine number of components
            max_components = min(X.shape[0], X.shape[1])
            if n_components is None:
                n_components = max_components
            else:
                n_components = min(n_components, max_components)
                
            print(f"   • Computing {n_components} principal components...")
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            pca_components = pca.fit_transform(X)
            
            # Extract PCA results
            explained_var_ratio = pca.explained_variance_ratio_
            components = pca.components_
            explained_variance = pca.explained_variance_
            cumulative_var_ratio = np.cumsum(explained_var_ratio)
            
            print(f"✅ PCA computation complete!")
            
            # VARIANCE ANALYSIS
            print(f"\n📊 VARIANCE ANALYSIS:")
            print("-" * 50)
            
            print(f"Explained Variance by Component:")
            n_show_components = min(10, n_components)  # Show first 10 components
            for i in range(n_show_components):
                print(f"  PC{i+1}: {explained_var_ratio[i]*100:.2f}% "
                      f"(cumulative: {cumulative_var_ratio[i]*100:.2f}%)")
            
            if n_components > n_show_components:
                print(f"  ... and {n_components - n_show_components} more components")
                
            print(f"\nTotal variance explained by all {n_components} components: {cumulative_var_ratio[-1]*100:.2f}%")
            
            # Kaiser criterion (eigenvalue > 1)
            eigenvalues = explained_variance
            kaiser_components = np.sum(eigenvalues > 1)
            
            print(f"\n🔍 Kaiser Criterion (eigenvalue > 1):")
            print(f"   • Components to retain: {kaiser_components}")
            print(f"   • Variance explained: {cumulative_var_ratio[kaiser_components-1]*100:.2f}%")
            
            # Dimensionality reduction analysis
            components_80 = np.argmax(cumulative_var_ratio >= 0.8) + 1 if np.any(cumulative_var_ratio >= 0.8) else n_components
            components_90 = np.argmax(cumulative_var_ratio >= 0.9) + 1 if np.any(cumulative_var_ratio >= 0.9) else n_components
            components_95 = np.argmax(cumulative_var_ratio >= 0.95) + 1 if np.any(cumulative_var_ratio >= 0.95) else n_components
            
            reduction_80 = (1 - components_80/original_dims) * 100
            reduction_90 = (1 - components_90/original_dims) * 100
            reduction_95 = (1 - components_95/original_dims) * 100
            
            print(f"\n📉 DIMENSIONALITY REDUCTION POTENTIAL:")
            print("-" * 50)
            print(f"To explain 80% variance: {components_80} components needed")
            print(f"   • Dimensionality reduction (80% variance): {reduction_80:.1f}%")
            print(f"     ({original_dims} → {components_80} dimensions)")
            print(f"To explain 90% variance: {components_90} components needed")
            print(f"   • Dimensionality reduction (90% variance): {reduction_90:.1f}%")
            print(f"     ({original_dims} → {components_90} dimensions)")
            print(f"To explain 95% variance: {components_95} components needed")
            print(f"   • Dimensionality reduction (95% variance): {reduction_95:.1f}%")
            print(f"     ({original_dims} → {components_95} dimensions)")
            
            # COMPONENT LOADINGS ANALYSIS
            print(f"\n🔍 PRINCIPAL COMPONENT LOADINGS:")
            print("-" * 50)
            
            # Create loadings dataframe
            loadings_df = pd.DataFrame(
                components[:n_show_components].T,
                columns=[f'PC{i+1}' for i in range(n_show_components)],
                index=valid_cols
            )
            
            print(f"Top variable contributions to each component:")
            
            # Show top loadings for each component
            for i in range(min(5, n_show_components)):  # Show first 5 components
                pc_name = f'PC{i+1}'
                
                # Get absolute loadings and sort
                abs_loadings = loadings_df[pc_name].abs().sort_values(ascending=False)
                top_vars = abs_loadings.head(5).to_dict()  # Top 5 variables
                
                print(f"\n   {pc_name} (explains {explained_var_ratio[i]*100:.1f}% variance):")
                for var, loading_abs in top_vars.items():
                    original_loading = loadings_df.loc[var, pc_name]
                    sign = '+' if original_loading > 0 else '-'
                    print(f"     {sign} {var}: {abs(original_loading):.3f}")
            
            # COMPONENT SCORES ANALYSIS
            print(f"\n📊 PRINCIPAL COMPONENT SCORES:")
            print("-" * 50)
            
            # Basic statistics of PC scores
            pc_scores_df = pd.DataFrame(
                pca_components[:, :n_show_components],
                columns=[f'PC{i+1}' for i in range(n_show_components)]
            )
            
            print(f"\nPC Score Statistics:")
            print(pc_scores_df.describe().round(3))
            
            # RECOMMENDATIONS
            print("\n" + "="*60)
            print("📊 PCA ANALYSIS SUMMARY & RECOMMENDATIONS")
            print("="*60)
            
            print(f"\n🎯 Dimensionality Reduction Recommendations:")
            
            if reduction_80 > 50:
                print(f"   ✅ Excellent reduction potential: {reduction_80:.1f}% reduction")
                print(f"   💡 Strong recommendation: Use PCA for dimensionality reduction")
            elif reduction_80 > 25:
                print(f"   ⚠️  Moderate reduction potential: {reduction_80:.1f}% reduction")
                print(f"   💡 Consider PCA if computational efficiency is important")
            else:
                print(f"   ❌ Limited reduction potential: {reduction_80:.1f}% reduction")
                print(f"   💡 PCA may not provide significant benefits")
            
            print(f"\n📊 Key Insights:")
            print(f"   • {kaiser_components}/{len(valid_cols)} components have eigenvalue > 1")
            print(f"   • First 2 PCs explain {cumulative_var_ratio[1]*100:.1f}% of variance" if n_components >= 2 else f"   • Only 1 PC computed")
            
            if len(valid_cols) > 10:
                print(f"   • High-dimensional dataset - PCA likely beneficial")
            else:
                print(f"   • Moderate-dimensional dataset - evaluate trade-offs")
            
            print(f"\n💡 Usage Recommendations:")
            print(f"   • For visualization: Use first 2-3 components")
            print(f"   • For modeling: Use components explaining ≥80-90% variance")
            print(f"   • For interpretation: Focus on component loadings")
            
            # Prepare return data
            results = {
                'pca_model': pca,
                'components': pca_components,
                'explained_variance_ratio': explained_var_ratio,
                'cumulative_variance_ratio': cumulative_var_ratio,
                'loadings': loadings_df,
                'pc_scores': pc_scores_df,
                'feature_names': valid_cols,
                'n_components': n_components,
                'kaiser_components': kaiser_components,
                'dimensionality_reduction': {
                    '80_percent': {'components': components_80, 'reduction': reduction_80},
                    '90_percent': {'components': components_90, 'reduction': reduction_90},
                    '95_percent': {'components': components_95, 'reduction': reduction_95}
                },
                'recommendations': {
                    'use_pca': reduction_80 > 25,
                    'recommended_components_80': components_80,
                    'recommended_components_90': components_90,
                    'high_dimensional': len(valid_cols) > 10
                }
            }
            
            print("\n" + "="*60)
            print("✅ PCA dimensionality reduction analysis complete!")
            print("="*60)
            
            return results
            
        except Exception as e:
            error_msg = f"Error in PCA dimensionality reduction analysis: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}


def get_component():
    """Return the analysis component."""
    return PCADimensionalityReduction