"""Visual Outlier Inspection Component.

Provides visual outlier detection using scatter plots and histograms.
"""


class VisualOutlierInspection:
    """Visual inspection of outliers using scatter plots and distribution plots."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "visual_outlier_inspection",
            "display_name": "Visual Outlier Inspection",
            "description": "Visual inspection of outliers using scatter plots and histograms",
            "category": "outlier_detection",
            "complexity": "basic",
            "tags": ["outliers", "visualization", "scatter", "histogram", "visual"],
            "estimated_runtime": "3-7 seconds",
            "icon": "👁️"
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview=None):
        """Check if analysis can be performed on the data."""
        if not data_preview:
            return True
        numeric_cols = data_preview.get('numeric_columns', [])
        return len(numeric_cols) > 0
    
    @staticmethod
    def generate_code(data_preview=None):
        """Generate code for visual outlier inspection."""
        
        return '''
# ===== VISUAL OUTLIER INSPECTION =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("="*60)
print("👁️ VISUAL OUTLIER INSPECTION")
print("="*60)

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\\n📊 Found {len(numeric_cols)} numeric columns for visual inspection")

if len(numeric_cols) == 0:
    print("❌ No numeric columns found for visual outlier inspection")
else:
    # Remove columns with all NaN or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and not df[col].isna().all():
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns for visual inspection")
    
    if len(valid_cols) == 0:
        print("❌ No valid columns after removing constant/all-NaN columns")
    else:
        # 1. DISTRIBUTION PLOTS WITH OUTLIER HIGHLIGHTING
        print(f"\\n📊 Creating distribution plots with outlier highlighting...")
        
        # Select up to 12 columns for distribution plots
        dist_cols = valid_cols[:12]
        n_cols_dist = min(4, len(dist_cols))
        n_rows_dist = (len(dist_cols) + n_cols_dist - 1) // n_cols_dist
        
        fig1, axes1 = plt.subplots(n_rows_dist, n_cols_dist, figsize=(5*n_cols_dist, 4*n_rows_dist))
        fig1.suptitle('📊 Distribution Analysis with Outlier Detection', fontsize=16, fontweight='bold')
        
        if len(dist_cols) == 1:
            axes1 = [axes1]
        elif n_rows_dist == 1 or n_cols_dist == 1:
            axes1 = axes1.flatten()
        else:
            axes1 = axes1.flatten()
        
        for idx, col in enumerate(dist_cols):
            ax = axes1[idx]
            
            col_data = df[col].dropna()
            if len(col_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{col}\\n(No data)')
                continue
            
            # Create histogram
            n_bins = min(50, max(10, len(col_data) // 20))
            counts, bins, patches = ax.hist(col_data, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Calculate outlier bounds (IQR method)
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Highlight outlier regions
            ax.axvline(lower_bound, color='red', linestyle='--', alpha=0.8, label=f'Lower bound: {lower_bound:.2f}')
            ax.axvline(upper_bound, color='red', linestyle='--', alpha=0.8, label=f'Upper bound: {upper_bound:.2f}')
            
            # Count outliers
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_count = len(outliers)
            
            # Color bars in outlier regions
            for i, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
                bin_center = (bin_left + bin_right) / 2
                if bin_center < lower_bound or bin_center > upper_bound:
                    patch.set_facecolor('red')
                    patch.set_alpha(0.8)
            
            ax.set_title(f'{col}\\n{outlier_count} outliers ({outlier_count/len(col_data)*100:.1f}%)', 
                        fontweight='bold')
            ax.set_xlabel('Value', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(dist_cols), len(axes1)):
            axes1[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # 2. SCATTER PLOT MATRIX FOR BIVARIATE OUTLIERS
        if len(valid_cols) >= 2:
            print(f"\\n📈 Creating scatter plot matrix for bivariate outlier detection...")
            
            # Select subset for scatter matrix (max 6 variables)
            scatter_cols = valid_cols[:6]
            
            if len(scatter_cols) >= 2:
                # Create pair plot
                scatter_data = df[scatter_cols].dropna()
                
                if len(scatter_data) > 0:
                    # Calculate Mahalanobis distance for multivariate outliers
                    try:
                        from scipy.spatial.distance import mahalanobis
                        
                        # Compute covariance matrix
                        cov_matrix = np.cov(scatter_data.T)
                        cov_inv = np.linalg.pinv(cov_matrix)
                        
                        # Calculate Mahalanobis distances
                        mean_vec = np.mean(scatter_data, axis=0)
                        mahal_distances = []
                        
                        for _, row in scatter_data.iterrows():
                            try:
                                dist = mahalanobis(row, mean_vec, cov_inv)
                                mahal_distances.append(dist)
                            except:
                                mahal_distances.append(0)
                        
                        # Identify multivariate outliers (using chi-square threshold)
                        threshold = stats.chi2.ppf(0.95, df=len(scatter_cols))  # 95% confidence
                        multivariate_outliers = np.array(mahal_distances) > threshold
                        
                        print(f"   📊 Multivariate outliers detected: {multivariate_outliers.sum()} ({multivariate_outliers.sum()/len(scatter_data)*100:.1f}%)")
                        
                    except Exception as e:
                        print(f"   ⚠️  Could not calculate multivariate outliers: {e}")
                        multivariate_outliers = np.zeros(len(scatter_data), dtype=bool)
                    
                    # Create scatter plot matrix
                    n_vars = len(scatter_cols)
                    fig2, axes2 = plt.subplots(n_vars, n_vars, figsize=(3*n_vars, 3*n_vars))
                    fig2.suptitle('📈 Scatter Plot Matrix with Outlier Highlighting', fontsize=16, fontweight='bold')
                    
                    for i in range(n_vars):
                        for j in range(n_vars):
                            ax = axes2[i, j]
                            
                            if i == j:
                                # Diagonal: histogram
                                col_data = scatter_data.iloc[:, i]
                                ax.hist(col_data, bins=20, alpha=0.7, color='skyblue')
                                ax.set_title(scatter_cols[i], fontweight='bold')
                            else:
                                # Off-diagonal: scatter plot
                                x_data = scatter_data.iloc[:, j]
                                y_data = scatter_data.iloc[:, i]
                                
                                # Plot normal points
                                normal_mask = ~multivariate_outliers
                                ax.scatter(x_data[normal_mask], y_data[normal_mask], 
                                         alpha=0.6, s=20, color='blue', label='Normal')
                                
                                # Highlight outliers
                                if multivariate_outliers.sum() > 0:
                                    ax.scatter(x_data[multivariate_outliers], y_data[multivariate_outliers], 
                                             alpha=0.8, s=30, color='red', edgecolors='black', 
                                             label='Outliers')
                                
                                # Add trend line
                                try:
                                    z = np.polyfit(x_data, y_data, 1)
                                    p = np.poly1d(z)
                                    ax.plot(x_data, p(x_data), "gray", linestyle='--', alpha=0.8)
                                except:
                                    pass
                            
                            ax.grid(True, alpha=0.3)
                            
                            # Set labels
                            if i == n_vars - 1:
                                ax.set_xlabel(scatter_cols[j], fontweight='bold')
                            if j == 0 and i != j:
                                ax.set_ylabel(scatter_cols[i], fontweight='bold')
                    
                    plt.tight_layout()
                    plt.show()
        
        # 3. BOX-AND-WHISKER COMPARISON
        print(f"\\n📦 Creating comparative box plots...")
        
        if len(valid_cols) <= 10:
            # Single figure for all columns
            fig3, ax3 = plt.subplots(1, 1, figsize=(max(10, len(valid_cols)*1.2), 6))
            
            box_data = [df[col].dropna() for col in valid_cols]
            bp = ax3.boxplot(box_data, labels=valid_cols, patch_artist=True, notch=True)
            
            # Customize box plots
            colors = plt.cm.Set3(np.linspace(0, 1, len(valid_cols)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.set_title('📦 Comparative Box Plot Analysis', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Variables', fontweight='bold')
            ax3.set_ylabel('Values', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Rotate labels if needed
            if len(max(valid_cols, key=len)) > 10:
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.show()
        
        # Summary insights
        print("\\n" + "="*60)
        print("👁️ VISUAL OUTLIER INSPECTION SUMMARY")
        print("="*60)
        
        print(f"\\n📊 Analysis Overview:")
        print(f"   • Variables visualized: {len(valid_cols)}")
        print(f"   • Distribution plots created: {len(dist_cols)}")
        
        if len(valid_cols) >= 2:
            print(f"   • Scatter matrix variables: {min(6, len(valid_cols))}")
            print(f"   • Multivariate outlier detection: {'✅ Performed' if len(valid_cols) >= 2 else '❌ Skipped'}")
        
        print(f"\\n💡 VISUAL INSPECTION GUIDE:")
        print(f"   📊 Distribution plots: Red dashed lines show IQR outlier bounds")
        print(f"   📈 Scatter matrix: Red points are multivariate outliers") 
        print(f"   📦 Box plots: Points beyond whiskers are outliers")
        print(f"   🔍 Look for: Isolated points, extreme values, unusual patterns")
        print(f"   ⚠️  Context matters: Not all statistical outliers are errors!")

print("\\n" + "="*60)
print("✅ Visual outlier inspection complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return VisualOutlierInspection