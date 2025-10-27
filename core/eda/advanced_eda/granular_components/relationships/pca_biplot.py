"""PCA Biplot Component.

Focused analysis component for PCA biplot visualization only.
"""
from typing import Dict, Any


class PCABiplotAnalysis:
    """Focused component for PCA biplot visualization"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "pca_biplot",
            "display_name": "PCA Biplot",
            "description": "PCA biplot showing data points and variable loadings together",
            "category": "relationships",
            "subcategory": "pca",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-15 seconds",
            "icon": "chart-biplot",
            "tags": ["pca", "biplot", "loadings", "visualization", "arrows"]
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview=None):
        """Check if dataset has enough numeric columns for PCA"""
        if not data_preview:
            return True
        
        sample_data = data_preview.get("data", [])
        if not sample_data:
            return True
            
        headers = data_preview.get("columns", [])
        numeric_cols = []
        
        # Check each column for numeric data
        for i, col in enumerate(headers):
            try:
                values = [row[i] for row in sample_data[:5] if len(row) > i]
                numeric_values = 0
                for v in values:
                    try:
                        float(str(v))
                        numeric_values += 1
                    except (ValueError, TypeError):
                        continue
                # If most sampled values are numeric, consider column numeric
                if numeric_values >= len(values) * 0.5:
                    numeric_cols.append(col)
            except:
                continue
        
        # PCA biplot needs at least 2 numeric columns (preferably 3+)
        return len(numeric_cols) >= 2
    
    def analyze(self, df, **kwargs):
        """
        Perform PCA biplot analysis.
        
        Args:
            df: DataFrame to analyze
            **kwargs: Additional parameters
                
        Returns:
            Dict with analysis results
        """
        try:
            print("="*60)
            print("📊 PCA BIPLOT ANALYSIS")  
            print("="*60)
            
            # Execute the generated code logic with proper scope
            exec(self.generate_code(), {'df': df, 'plt': __import__('matplotlib.pyplot'), 'pd': __import__('pandas'), 
                                       'np': __import__('numpy'), 'sns': __import__('seaborn'),
                                       'PCA': __import__('sklearn.decomposition', fromlist=['PCA']).PCA,
                                       'StandardScaler': __import__('sklearn.preprocessing', fromlist=['StandardScaler']).StandardScaler,
                                       'warnings': __import__('warnings')})
            
            return {"status": "completed", "message": "PCA biplot analysis completed successfully"}
            
        except Exception as e:
            error_msg = f"Error in PCA biplot analysis: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    @staticmethod
    def generate_code(data_preview=None):
        """Generate focused PCA biplot code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=== PCA BIPLOT ANALYSIS ===")
print("Combined visualization of data points and variable loadings")
print()

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    print("❌ INSUFFICIENT NUMERIC COLUMNS")
    print("   PCA biplot requires at least 2 numeric variables.")
else:
    print(f"📊 CREATING PCA BIPLOT FOR {len(numeric_cols)} VARIABLES")
    print()
    
    # Remove columns with too many missing values or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and df[col].notna().sum() >= max(10, len(df) * 0.3):
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns (removing columns with >70% missing or constant values)")
    
    # Limit dimensions for better visualization and performance
    max_dimensions = 20  # Reasonable limit for biplot visualization
    if len(valid_cols) > max_dimensions:
        print(f"⚠️  LIMITING DIMENSIONS: Using top {max_dimensions} columns by variance for better visualization")
        # Calculate variance for each column and select top ones
        col_variances = [(col, df[col].var()) for col in valid_cols]
        col_variances.sort(key=lambda x: x[1], reverse=True)
        valid_cols = [col for col, _ in col_variances[:max_dimensions]]
        print(f"📋 Selected columns: {valid_cols[:5]}{'...' if len(valid_cols) > 5 else ''}")
    
    if len(valid_cols) < 2:
        print("❌ INSUFFICIENT VALID COLUMNS")
        print("   PCA biplot requires at least 2 columns with sufficient data.")
    else:
        # Prepare data with imputation for missing values
        X = df[valid_cols].copy()
        
        # Simple imputation: fill missing values with column means
        print(f"\\n🔄 Data preprocessing:")
        total_missing = X.isnull().sum().sum()
        if total_missing > 0:
            print(f"   • Found {total_missing:,} missing values")
            print(f"   • Imputing missing values with column means")
            X = X.fillna(X.mean())
        else:
            print(f"   • No missing values to handle")
        
        print(f"   • Final dataset: {len(X):,} rows × {len(valid_cols)} columns")
        
        if len(X) < 10:
            print("❌ INSUFFICIENT DATA POINTS")
            print("   Need at least 10 data points for reliable PCA biplot.")
        else:
            print(f"   • Using {len(X):,} observations for PCA biplot")
            print()
            
            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA for biplot (2D)
            pca_2d = PCA(n_components=2)
            X_pca = pca_2d.fit_transform(X_scaled)
        
        # Get variance explained by the two components
        var_explained = pca_2d.explained_variance_ratio_
        cumulative_var = np.sum(var_explained)
        
        print(f"🎯 BIPLOT COMPONENTS:")
        print(f"   • PC1 explains: {var_explained[0]*100:.1f}% variance")
        print(f"   • PC2 explains: {var_explained[1]*100:.1f}% variance")
        print(f"   • Total variance shown: {cumulative_var*100:.1f}%")
        print()
        
        # Check if we have categorical columns for coloring
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        color_column = None
        
        if categorical_cols:
            # Find the categorical column with reasonable number of categories
            for col in categorical_cols:
                unique_vals = df[col].nunique()
                if 2 <= unique_vals <= 10:  # Good range for visualization
                    color_column = col
                    break
        
        if color_column:
            # Use original DataFrame for categorical data
            color_data = df[color_column]
            unique_categories = color_data.unique()
            print(f"   Using '{color_column}' for coloring ({len(unique_categories)} categories)")
        else:
            color_data = None
            print("   No suitable categorical column found for coloring")
        
        print()
        
        # Get component loadings for biplot arrows
        loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)
        
        # Create the biplot
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        
        # Plot data points (scores)
        if color_data is not None:
            # Color by categorical variable
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_categories)))
            for i, category in enumerate(unique_categories):
                mask = color_data == category
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          label=f'{category}', alpha=0.7, s=50, color=colors[i])
            ax.legend(title=color_column, bbox_to_anchor=(1.05, 1), loc='upper left', 
                     frameon=True, fancybox=True, shadow=True)
        else:
            # Single color for all points
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50, color='skyblue', 
                      label=f'Data points (n={len(X_pca)})')
            ax.legend()
        
        # Calculate appropriate scale factor for loading arrows
        # Scale so that arrows are visible but not overwhelming
        scores_range = max(np.max(np.abs(X_pca[:, 0])), np.max(np.abs(X_pca[:, 1])))
        loadings_range = max(np.max(np.abs(loadings[:, 0])), np.max(np.abs(loadings[:, 1])))
        scale_factor = scores_range / loadings_range * 0.8
        
        # Add loading vectors (arrows)
        arrow_colors = plt.cm.tab10(np.linspace(0, 1, len(valid_cols)))
        
        for i, var in enumerate(valid_cols):
            # Draw arrow
            arrow_x = loadings[i, 0] * scale_factor
            arrow_y = loadings[i, 1] * scale_factor
            
            ax.arrow(0, 0, arrow_x, arrow_y,
                    head_width=scores_range*0.02, head_length=scores_range*0.03, 
                    fc=arrow_colors[i], ec=arrow_colors[i], alpha=0.8, linewidth=2)
            
            # Add variable label
            label_x = arrow_x * 1.1
            label_y = arrow_y * 1.1
            
            ax.text(label_x, label_y, var, fontsize=11, fontweight='bold',
                   ha='center', va='center', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=arrow_colors[i], 
                            alpha=0.7, edgecolor='black'))
        
        # Customize the plot
        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_title(f'PCA Biplot\\n({cumulative_var*100:.1f}% of total variance explained)', 
                    fontsize=14, fontweight='bold')
        
        # Add grid and origin lines
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
        
        # Set equal aspect ratio for proper interpretation
        ax.set_aspect('equal', adjustable='box')
        
        # Add a subtle background
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        plt.show()
        print()
        
        # Loading values interpretation
        print("🔍 VARIABLE LOADINGS ANALYSIS:")
        print("   (Contribution and direction of original variables)")
        print()
        
        loadings_df = pd.DataFrame(loadings, 
                                 columns=['PC1_Loading', 'PC2_Loading'],
                                 index=valid_cols)
        
        # Calculate loading magnitudes and angles
        loadings_df['Magnitude'] = np.sqrt(loadings_df['PC1_Loading']**2 + loadings_df['PC2_Loading']**2)
        loadings_df['Angle_Degrees'] = np.degrees(np.arctan2(loadings_df['PC2_Loading'], loadings_df['PC1_Loading']))
        
        # Sort by magnitude (most influential variables first)
        loadings_df_sorted = loadings_df.sort_values('Magnitude', ascending=False)
        
        print("📊 VARIABLE LOADINGS TABLE (sorted by influence):")
        print("   " + "="*70)
        print(f"   {'Variable':<20} {'PC1':>8} {'PC2':>8} {'Magnitude':>10} {'Angle':>8}")
        print("   " + "="*70)
        
        for var, row in loadings_df_sorted.iterrows():
            print(f"   {var:<20} {row['PC1_Loading']:>8.3f} {row['PC2_Loading']:>8.3f} " + 
                  f"{row['Magnitude']:>10.3f} {row['Angle_Degrees']:>8.1f}°")
        
        print("   " + "="*70)
        print()
        
        # Identify most influential variables for each PC
        print("🎯 KEY VARIABLE CONTRIBUTIONS:")
        
        # PC1 contributions
        pc1_abs = np.abs(loadings_df['PC1_Loading'])
        top_pc1_idx = pc1_abs.nlargest(3).index
        print(f"\\n   PC1 (most influenced by):")
        for var in top_pc1_idx:
            loading_val = loadings_df.loc[var, 'PC1_Loading']
            contribution = pc1_abs[var] / pc1_abs.sum() * 100
            direction = "positive" if loading_val > 0 else "negative"
            print(f"      • {var}: {loading_val:.3f} ({direction}, {contribution:.1f}% contribution)")
        
        # PC2 contributions  
        pc2_abs = np.abs(loadings_df['PC2_Loading'])
        top_pc2_idx = pc2_abs.nlargest(3).index
        print(f"\\n   PC2 (most influenced by):")
        for var in top_pc2_idx:
            loading_val = loadings_df.loc[var, 'PC2_Loading']
            contribution = pc2_abs[var] / pc2_abs.sum() * 100
            direction = "positive" if loading_val > 0 else "negative"
            print(f"      • {var}: {loading_val:.3f} ({direction}, {contribution:.1f}% contribution)")
        
        print()
        
        # Variable relationships
        print("🔗 VARIABLE RELATIONSHIPS:")
        print()
        
        # Find highly correlated variables (similar loadings)
        correlation_threshold = 0.7
        similar_pairs = []
        
        for i, var1 in enumerate(valid_cols):
            for j, var2 in enumerate(valid_cols[i+1:], i+1):
                # Calculate cosine similarity of loadings
                vec1 = loadings[i, :]
                vec2 = loadings[j, :]
                cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                if abs(cos_sim) > correlation_threshold:
                    similar_pairs.append((var1, var2, cos_sim))
        
        if similar_pairs:
            print("   📈 HIGHLY RELATED VARIABLES (similar loading patterns):")
            for var1, var2, similarity in similar_pairs:
                relationship = "positively" if similarity > 0 else "negatively"
                print(f"      • {var1} ↔ {var2}: {relationship} related (similarity: {similarity:.3f})")
        else:
            print("   • No highly similar variable loading patterns found")
        
        print()
        
        # Quadrant analysis
        print("🧭 BIPLOT QUADRANT ANALYSIS:")
        print("   (Variable positioning interpretation)")
        print()
        
        for i, var in enumerate(valid_cols):
            pc1_load = loadings[i, 0]
            pc2_load = loadings[i, 1]
            
            # Determine quadrant
            if pc1_load > 0 and pc2_load > 0:
                quadrant = "I (top-right)"
            elif pc1_load < 0 and pc2_load > 0:
                quadrant = "II (top-left)"
            elif pc1_load < 0 and pc2_load < 0:
                quadrant = "III (bottom-left)"
            else:
                quadrant = "IV (bottom-right)"
            
            print(f"   • {var}: Quadrant {quadrant}")
        
        print()
        print("📖 BIPLOT INTERPRETATION GUIDE:")
        print("   🎯 DATA POINTS (dots/circles):")
        print("      • Position shows similarity - closer points are more similar")
        print("      • Distance from origin indicates how typical/atypical the observation is")
        print("      • Clusters indicate groups with similar characteristics")
        print()
        print("   🏹 LOADING ARROWS (colored arrows):")
        print("      • Arrow direction shows how variable contributes to each PC")
        print("      • Arrow length shows strength of contribution") 
        print("      • Similar arrow directions → variables are correlated")
        print("      • Opposite arrow directions → variables are negatively correlated")
        print("      • Perpendicular arrows → variables are uncorrelated")
        print()
        print("   🧭 SPATIAL RELATIONSHIPS:")
        print("      • Data points in arrow direction → high values for that variable")
        print("      • Data points opposite arrow → low values for that variable")
        print("      • Points perpendicular to arrow → average values for that variable")
        print()
        print("   📐 ANGLE INTERPRETATION:")
        print("      • 0° (right): Mainly PC1 positive contribution")
        print("      • 90° (up): Mainly PC2 positive contribution") 
        print("      • 180° (left): Mainly PC1 negative contribution")
        print("      • 270° (down): Mainly PC2 negative contribution")

print("\\n" + "="*60)
print("✅ PCA BIPLOT ANALYSIS COMPLETE")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return PCABiplotAnalysis