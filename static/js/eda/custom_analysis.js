/**
 * Custom Analysis JavaScript
 * Handles custom code execution and analysis functionality
 */

console.log('Custom Analysis JavaScript loading...');

// Global variables for custom analysis
let customCellCounter = 1;
let customMarkdownCounter = 1;
let customExecutionCounter = 1;
let codeEditors = new Map(); // Store CodeMirror instances

// Active execution tracking for monitoring
window.activeExecutions = window.activeExecutions || [];

// Helper function to track execution start
function trackExecutionStart(cellId) {
    window.activeExecutions.push({
        cellId: cellId,
        startTime: Date.now(),
        id: `exec_${Date.now()}_${cellId}`
    });
}

// Helper function to track execution end
function trackExecutionEnd(cellId) {
    window.activeExecutions = window.activeExecutions.filter(exec => exec.cellId !== cellId);
}

function openAskAIForCustomCell(cellId) {
    if (window.EDAChatBridge && typeof window.EDAChatBridge.setActiveCell === 'function') {
        window.EDAChatBridge.setActiveCell(cellId, 'custom');
    }

    if (window.LLMChat && typeof window.LLMChat.showModal === 'function') {
        window.LLMChat.showModal();
    } else {
        showNotification('AI assistant is currently unavailable.', 'warning');
    }
}

// Code Examples
const CODE_EXAMPLES = {
    'basic_info': `# Dataset Basic Information
print("Dataset Shape:", df.shape)
print("\\nDataset Info:")
print(df.info())
print("\\nFirst 5 rows:")
print(df.head())`,

    'missing_values': `# Missing Values Analysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Calculate missing values
missing_data = df.isnull().sum()
missing_percent = 100 * missing_data / len(df)

missing_df = pd.DataFrame({
    'Column': missing_data.index,
    'Missing_Count': missing_data.values,
    'Missing_Percent': missing_percent.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print("Missing Values Summary:")
print(missing_df)

# Visualize missing values
if not missing_df.empty:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(missing_df['Column'], missing_df['Missing_Count'])
    plt.title('Missing Values Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(missing_df['Column'], missing_df['Missing_Percent'])
    plt.title('Missing Values Percentage')
    plt.xticks(rotation=45)
    plt.ylabel('Percentage (%)')
    
    plt.tight_layout()
    plt.show()
else:
    print("No missing values found in the dataset.")`,

    'correlation_heatmap': `# Correlation Heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Print highly correlated pairs
    print("\\nHighly correlated feature pairs (|correlation| > 0.7):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    for pair in high_corr_pairs:
        print(f"{pair[0]} <-> {pair[1]}: {pair[2]:.3f}")
        
    if not high_corr_pairs:
        print("No highly correlated pairs found.")
else:
    print("Need at least 2 numeric columns for correlation analysis.")`,

    'distribution_plots': `# Distribution Plots for Numeric Variables
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if numeric_cols:
    n_cols = min(len(numeric_cols), 4)  # Max 4 columns per row
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        ax = axes[row][col_idx] if n_rows > 1 else axes[col_idx]
        
        # Create histogram with KDE
        df[col].hist(bins=30, alpha=0.7, ax=ax, density=True)
        df[col].plot(kind='kde', ax=ax, color='red')
        
        ax.set_title(f'Distribution of {col}')
        ax.set_ylabel('Density')
        
        # Add statistics
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        if n_rows > 1:
            axes[row][col_idx].set_visible(False)
        else:
            axes[col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Summary Statistics:")
    print(df[numeric_cols].describe())
else:
    print("No numeric columns found for distribution analysis.")`,

    'pca_analysis': `# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) >= 2:
    # Prepare data
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot explained variance
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'bo-')
    plt.title('Explained Variance by Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'ro-')
    plt.axhline(y=0.95, color='k', linestyle='--', alpha=0.7, label='95% Variance')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("PCA Analysis Results:")
    print(f"Total components: {len(pca.explained_variance_ratio_)}")
    print(f"Variance explained by first 3 components: {sum(pca.explained_variance_ratio_[:3]):.3f}")
    
    # Show feature importance in first 2 components
    feature_importance = pd.DataFrame(
        pca.components_[:2].T,
        columns=['PC1', 'PC2'],
        index=numeric_cols
    )
    print("\\nFeature importance in first 2 components:")
    print(feature_importance.round(3))
    
else:
    print("Need at least 2 numeric columns for PCA analysis.")`,

    'outlier_detection': `# Outlier Detection using IQR and Z-Score methods
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if numeric_cols:
    outlier_summary = []
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        # Z-Score method (assuming normal distribution)
        z_scores = np.abs(stats.zscore(data))
        zscore_outliers = data[z_scores > 3]
        
        outlier_summary.append({
            'Column': col,
            'IQR_Outliers': len(iqr_outliers),
            'IQR_Percentage': len(iqr_outliers) / len(data) * 100,
            'ZScore_Outliers': len(zscore_outliers),
            'ZScore_Percentage': len(zscore_outliers) / len(data) * 100
        })
    
    # Create summary DataFrame
    outlier_df = pd.DataFrame(outlier_summary)
    print("Outlier Detection Summary:")
    print(outlier_df.round(2))
    
    # Visualize outliers for top 4 numeric columns
    cols_to_plot = numeric_cols[:4]
    
    fig, axes = plt.subplots(2, len(cols_to_plot), figsize=(4*len(cols_to_plot), 8))
    
    for i, col in enumerate(cols_to_plot):
        # Box plot
        if len(cols_to_plot) == 1:
            ax_box = axes[0]
            ax_hist = axes[1]
        else:
            ax_box = axes[0, i]
            ax_hist = axes[1, i]
            
        df.boxplot(column=col, ax=ax_box)
        ax_box.set_title(f'Box Plot: {col}')
        
        # Histogram with outlier boundaries
        ax_hist.hist(df[col].dropna(), bins=30, alpha=0.7)
        
        # Add IQR boundaries
        data = df[col].dropna()
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        ax_hist.axvline(Q1 - 1.5 * IQR, color='red', linestyle='--', alpha=0.7, label='IQR Bounds')
        ax_hist.axvline(Q3 + 1.5 * IQR, color='red', linestyle='--', alpha=0.7)
        ax_hist.set_title(f'Distribution: {col}')
        ax_hist.legend()
    
    plt.tight_layout()
    plt.show()
    
else:
    print("No numeric columns found for outlier analysis.")`,

    'feature_importance': `# Feature Importance Analysis (using Random Forest)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) >= 2:
    # Assume the last numeric column is the target (you can modify this)
    target_col = numeric_cols[-1]
    feature_cols = numeric_cols[:-1]
    
    print(f"Using '{target_col}' as target variable")
    print(f"Features: {feature_cols}")
    
    # Prepare data
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col].fillna(df[target_col].mean())
    
    # Determine if regression or classification
    unique_targets = len(y.unique())
    is_classification = unique_targets <= 20 and y.dtype == 'object' or unique_targets <= 10
    
    if is_classification:
        # Classification
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model_type = "Classification"
    else:
        # Regression
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_type = "Regression"
    
    # Fit model
    model.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\\n{model_type} Feature Importance:")
    print(importance_df.round(4))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['Importance'])
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance ({model_type})')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(importance_df['Importance']):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Model performance
    train_score = model.score(X, y)
    print(f"\\nModel R² Score (Training): {train_score:.3f}")
    
else:
    print("Need at least 2 numeric columns for feature importance analysis.")`,

    'clustering_analysis': `# Clustering Analysis using K-Means
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) >= 2:
    # Prepare data
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters using elbow method
    inertias = []
    K_range = range(1, min(11, len(df)//2))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    
    # Apply K-means with optimal k (you can adjust this)
    optimal_k = 3  # You might want to choose this based on the elbow curve
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe copy
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Plot clusters (using first 2 principal components for visualization)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    # Plot centroids
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(f'K-Means Clustering (k={optimal_k})')
    plt.colorbar(scatter)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Cluster analysis
    print(f"\\nClustering Results (k={optimal_k}):")
    print("Cluster distribution:")
    print(df_clustered['Cluster'].value_counts().sort_index())
    
    print("\\nCluster centers (original scale):")
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers_original, 
                             columns=numeric_cols, 
                             index=[f'Cluster_{i}' for i in range(optimal_k)])
    print(centers_df.round(2))
    
else:
    print("Need at least 2 numeric columns for clustering analysis.")`,
};

// Add custom CSS styles dynamically
function addCustomAnalysisStyles() {
    if (document.getElementById('customAnalysisStyles')) {
        return; // Styles already added
    }
    
    const style = document.createElement('style');
    style.id = 'customAnalysisStyles';
    style.textContent = `
        /* Custom Analysis Styles */
        .custom-code-cell {
            position: relative;
            margin-bottom: 20px;
            overflow: visible;
        }

        /* Floating cell controls */
        .cell-float-controls {
            position: absolute;
            left: -40px;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            flex-direction: column;
            gap: 6px;
            z-index: 10;
        }

        .cell-float-controls .float-btn {
            width: 28px;
            height: 28px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
            background: var(--bs-light, #f8f9fa);
        }

        .cell-float-controls .float-btn i {
            font-size: 14px;
        }

        /* Better text selection in output */
        .code-output pre::selection {
            background: #316AC5;
            color: white;
        }

        /* Make code cells more balanced */
        .code-editor-container {
            width: 100%;
        }

        .CodeMirror {
            width: 100% !important;
        }

        /* Variables panel styles */
        .variables-panel-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1050;
            display: none;
        }

        .variables-panel-overlay.show {
            display: block;
        }

        .variables-panel {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            z-index: 1051;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            overflow: hidden;
            display: none;
        }

        .variables-panel.show {
            display: block;
        }

        .variables-panel-header {
            background: #007bff;
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .variables-panel-close {
            background: none;
            border: none;
            color: white;
            font-size: 24px;
            cursor: pointer;
            padding: 0;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .variables-panel-content {
            padding: 20px;
            overflow-y: auto;
            max-height: calc(80vh - 70px);
        }

        .variables-category {
            margin-bottom: 20px;
        }

        .variables-category h6 {
            color: #495057;
            margin-bottom: 10px;
            font-weight: 600;
        }

        .variable-item {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 8px 12px;
            margin-bottom: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }

        .variable-item.clickable {
            cursor: pointer;
            transition: background-color 0.2s;
        }

        .variable-item.clickable:hover {
            background: #e9ecef;
            border-color: #007bff;
        }

        /* Plots inline styles */
        .plots-inline-container {
            margin-top: 15px;
        }

        .plot-inline {
            margin-bottom: 15px;
        }

        .plot-wrapper img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        /* Execution status styles */
        .execution-status {
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.85rem;
            margin: 5px 0;
        }

        .execution-status.running {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }

        .execution-status.success {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }

        .execution-status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        /* Responsive styles */
        @media (max-width: 768px) {
            .cell-float-controls {
                left: auto;
                right: 10px;
            }
        }
    `;
    document.head.appendChild(style);
}

function initializeCustomAnalysis() {
    // Add custom styles
    addCustomAnalysisStyles();
    
    // Event listeners for buttons
    const addCellBtn = document.getElementById('addCustomCodeCell');
    if (addCellBtn) {
        addCellBtn.addEventListener('click', addCustomCodeCell);
    }

    updateCustomAnalysisEmptyState();

    // Examples button event listener is handled in custom_analysis_tab.html to avoid conflicts
    // const showExamplesBtn = document.getElementById('showExamplesBtn');
    // if (showExamplesBtn) {
    //     showExamplesBtn.addEventListener('click', toggleExamplesPanel);
    // }

    const closeExamplesBtn = document.getElementById('closeExamplesBtn');
    if (closeExamplesBtn) {
        closeExamplesBtn.addEventListener('click', hideExamplesPanel);
    }

    const importCodeBtn = document.getElementById('importCodeBtn');
    if (importCodeBtn) {
        importCodeBtn.addEventListener('click', importCode);
    }
    
    // Example code listeners
    document.querySelectorAll('[data-code]').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const codeType = this.getAttribute('data-code');
            addCustomCodeCellWithExample(codeType);
            hideExamplesPanel();
        });
    });
    
    console.log('Custom Analysis initialized - ready to add cells');
}

function addCustomCodeCell() {
    const cellId = `custom-cell-${customCellCounter++}`;
    const cellHtml = createCustomCodeCellHtml(cellId);

    const container = document.getElementById('customCodeCells');
    if (!container) {
        console.error('customCodeCells container not found');
        return null;
    }

    // Append new cells at the bottom for natural ordering
    container.insertAdjacentHTML('beforeend', cellHtml);

    // Initialize CodeMirror for this cell
    initializeCodeEditor(cellId);

    const newCell = document.getElementById(cellId);
    if (newCell) {
        newCell.style.opacity = '0';
        newCell.style.transform = 'translateY(10px)'; // Only Y transform for animation

        setTimeout(() => {
            newCell.style.transition = 'all 0.3s ease';
            newCell.style.opacity = '1';
            newCell.style.transform = 'translateY(0)'; // Reset only Y, preserve any CSS X transform
        }, 50);

        setTimeout(() => {
            newCell.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
    }

    const editor = codeEditors.get(cellId);
    if (editor) {
        editor.focus();
        const lastLineIndex = Math.max(editor.lineCount() - 1, 0);
        const currentLineContent = editor.getLine(lastLineIndex) || '';
        editor.setCursor({ line: lastLineIndex, ch: currentLineContent.length });
    }

    updateCustomAnalysisEmptyState();

    return cellId;
}

function updateCustomAnalysisEmptyState() {
    const container = document.getElementById('customCodeCells');
    const emptyState = document.getElementById('customAnalysisEmptyState');

    if (!container || !emptyState) {
        return;
    }

    const hasEntries = Boolean(container.querySelector('.custom-analysis-entry'));
    emptyState.classList.toggle('d-none', hasEntries);
}

function addCustomCodeCellWithExample(exampleType) {
    const cellId = addCustomCodeCell();
    const editor = codeEditors.get(cellId);
    if (editor && CODE_EXAMPLES[exampleType]) {
        editor.setValue(CODE_EXAMPLES[exampleType]);
    }
}

function createCustomCodeCellHtml(cellId) {
    return `
    <div class="custom-code-cell custom-analysis-entry" id="${cellId}">
        <!-- Floating navigation buttons -->
        <div class="cell-float-controls">
            <button class="float-btn float-up-btn" onclick="moveCustomCellUp('${cellId}')" title="Move Up">
                <i class="bi bi-arrow-up"></i>
            </button>
            <button class="float-btn float-down-btn" onclick="moveCustomCellDown('${cellId}')" title="Move Down">
                <i class="bi bi-arrow-down"></i>
            </button>
        </div>
        
        <div class="code-cell-header">
            <h6 class="code-cell-title">
                <i class="bi bi-code-slash"></i> Code Cell #${customCellCounter - 1}
            </h6>
            <div class="code-cell-controls">
                <button class="btn btn-outline-secondary btn-sm collapse-btn" onclick="toggleCellCollapse('${cellId}')" title="Collapse/Expand">
                    <i class="bi bi-chevron-down"></i>
                </button>
                <button class="btn btn-success btn-sm" onclick="executeCustomCode('${cellId}')" title="Run Code">
                    <i class="bi bi-play-fill"></i> Run
                </button>
                <button class="btn btn-outline-primary btn-sm ask-ai-btn" onclick="openAskAIForCustomCell('${cellId}')" title="Ask AI about this cell">
                    <i class="bi bi-stars"></i> Ask AI
                </button>
                <button class="btn btn-outline-secondary btn-sm" onclick="clearOutput('${cellId}')" title="Clear Output">
                    <i class="bi bi-eraser"></i>
                </button>
                <button class="btn btn-outline-danger btn-sm" onclick="deleteCustomCell('${cellId}')" title="Delete Cell">
                    <i class="bi bi-trash"></i>
                </button>
            </div>
        </div>
        
        <div class="code-editor-container">
            <textarea id="editor-${cellId}" class="code-editor code-editor-textarea" placeholder="# Write your Python code here...

# Available: df, numeric_cols, categorical_cols, plt, sns, np, pd, sklearn, scipy, plotly...

print('Hello from custom analysis!')"></textarea>
            <div class="execution-status d-none" id="status-${cellId}"></div>
        </div>
        
        <!-- Output section -->
        <div class="code-output d-none" id="output-${cellId}">
            <pre></pre>
        </div>
    </div>`;
}

function addCustomMarkdownCell(initialContent = '', customHeading = '') {
    const cellId = `custom-markdown-${customMarkdownCounter++}`;
    const container = document.getElementById('customCodeCells');

    if (!container) {
        console.error('customCodeCells container not found');
        return;
    }

    const noteNumber = customMarkdownCounter - 1;
    const displayTitle = customHeading && customHeading.trim()
        ? customHeading.trim()
        : `Markdown note #${noteNumber}`;
    const template = `
    <div class="custom-markdown-card custom-analysis-entry" id="${cellId}">
        <div class="custom-markdown-card__header">
            <div class="custom-markdown-card__title">
                <i class="bi bi-markdown"></i>
                ${displayTitle}
            </div>
            <div class="custom-markdown-card__controls">
                <button class="btn btn-outline-secondary btn-sm" onclick="moveCustomCellUp('${cellId}')" title="Move up">
                    <i class="bi bi-arrow-up"></i>
                </button>
                <button class="btn btn-outline-secondary btn-sm" onclick="moveCustomCellDown('${cellId}')" title="Move down">
                    <i class="bi bi-arrow-down"></i>
                </button>
                <button class="btn btn-outline-secondary btn-sm" onclick="toggleMarkdownPreview('${cellId}')" title="Toggle preview">
                    <i class="bi bi-eye"></i>
                </button>
                <button class="btn btn-outline-danger btn-sm" onclick="deleteCustomMarkdownCell('${cellId}')" title="Delete note">
                    <i class="bi bi-trash"></i>
                </button>
            </div>
        </div>
        <div class="custom-markdown-card__body">
            <textarea class="markdown-note-editor" id="markdown-editor-${cellId}" placeholder="### Notes\nCapture decisions, rationales, and next steps."></textarea>
            <div class="markdown-note-preview d-none" id="markdown-preview-${cellId}"></div>
        </div>
    </div>`;

    container.insertAdjacentHTML('beforeend', template);

    const editor = document.getElementById(`markdown-editor-${cellId}`);
    if (editor) {
        editor.value = initialContent || '';
        editor.focus();
        if (!initialContent) {
            editor.placeholder = '### Notes\nCapture decisions, rationales, and next steps.';
        }
    }

    updateCustomAnalysisEmptyState();
    document.getElementById(cellId)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    showNotification('Markdown note added to workspace.', 'success');
}

function toggleMarkdownPreview(cellId) {
    const editor = document.getElementById(`markdown-editor-${cellId}`);
    const preview = document.getElementById(`markdown-preview-${cellId}`);

    if (!editor || !preview) {
        return;
    }

    const previewIsHidden = preview.classList.contains('d-none');

    if (previewIsHidden) {
        preview.classList.remove('d-none');
        editor.classList.add('d-none');

        const content = editor.value.trim();
        if (content) {
            const escaped = content
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            preview.innerHTML = `<pre>${escaped}</pre>`;
        } else {
            preview.innerHTML = '<p class="text-muted mb-0">No content yet. Add some notes in edit mode.</p>';
        }
    } else {
        preview.classList.add('d-none');
        editor.classList.remove('d-none');
    }
}

function deleteCustomMarkdownCell(cellId) {
    if (!confirm('Delete this markdown note?')) {
        return;
    }

    const element = document.getElementById(cellId);
    if (element) {
        element.remove();
        updateCustomAnalysisEmptyState();
    }
}

function initializeCodeEditor(cellId) {
    const textarea = document.getElementById(`editor-${cellId}`);
    if (textarea && typeof CodeMirror !== 'undefined') {
        const editor = CodeMirror.fromTextArea(textarea, {
            lineNumbers: true,
            mode: 'python',
            theme: 'default',
            autoCloseBrackets: true,
            matchBrackets: true,
            indentUnit: 4,
            lineWrapping: false,
            viewportMargin: Infinity,
            scrollbarStyle: 'null',
            extraKeys: {
                'Ctrl-Enter': function() { executeCustomCode(cellId); },
                'Cmd-Enter': function() { executeCustomCode(cellId); },
                'Alt-Z': function() { toggleWordWrap(cellId); }
            }
        });
        
        // Set initial size
        editor.setSize(null, '150px');
        
        // Auto-expand functionality
        editor.on('changes', function() {
            const lineHeight = 20;
            const lines = editor.lineCount();
            const minHeight = 150;
            const maxHeight = 800;
            const newHeight = Math.min(Math.max(lines * lineHeight + 40, minHeight), maxHeight);
            editor.setSize(null, newHeight + 'px');
        });
        
        const wrapper = editor.getWrapperElement();
        wrapper.style.width = '100%';
        
        codeEditors.set(cellId, editor);
        
        // Apply theme
        applyEditorTheme(editor);
    } else {
        // Fallback for plain textarea if CodeMirror is not available
        const fallbackTextarea = document.getElementById(`editor-${cellId}`);
        if (fallbackTextarea) {
            fallbackTextarea.style.overflow = 'hidden';
            fallbackTextarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(Math.max(this.scrollHeight, 150), 800) + 'px';
            });
        }
    }
}

function executeCustomCode(cellId) {
    const editor = codeEditors.get(cellId);
    if (!editor) {
        console.error('Editor not found for cell:', cellId);
        return;
    }
    
    const code = editor.getValue().trim();
    if (!code) {
        showNotification('Please enter some code to execute', 'warning');
        return;
    }

    // Check rate limits before executing
    if (window.rateLimitMonitor && !window.rateLimitMonitor.canExecute()) {
        const remaining = window.rateLimitMonitor.getRemainingExecutions();
        if (remaining <= 0) {
            showNotification('Rate limit reached. Please wait before executing more code.', 'warning');
            return;
        }
    }
    
    // Show execution status
    showExecutionStatus(cellId, 'running');
    
    // Track execution start for monitoring
    trackExecutionStart(cellId);

    if (window.EDAChatBridge && typeof window.EDAChatBridge.recordCustomPending === 'function') {
        window.EDAChatBridge.recordCustomPending(cellId, {
            code,
            dataset: window.currentDataset || null
        });
    }
    
    // Get source ID (assuming it's available globally)
    const currentSourceId = sourceId || window.sourceId;
    if (!currentSourceId) {
        showExecutionStatus(cellId, 'error');
        showOutput(cellId, 'Error: No dataset loaded. Please load a dataset first.', 'error');
        return;
    }
    
    // Execute code
    fetch(`/advanced-eda/api/execute-custom-code/${currentSourceId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            code: code
        })
    })
    .then(async response => {
        const rawText = await response.text();
        let data = null;
        if (rawText) {
            try {
                data = JSON.parse(rawText);
            } catch (parseError) {
                console.warn('Failed to parse response JSON:', parseError);
            }
        }

        if (response.status === 429) {
            if (window.rateLimitMonitor && data && data.rate_limit_info) {
                window.rateLimitMonitor.handleRateLimitError(data.rate_limit_info);
            }
            const error = new Error((data && data.error) || 'Rate limit exceeded');
            error.isApiError = true;
            error.status = response.status;
            error.data = data;
            error.isRateLimit = true;
            throw error;
        }

        if (!response.ok) {
            const message = (data && data.error) || `Request failed (HTTP ${response.status})`;
            const error = new Error(message);
            error.isApiError = true;
            error.status = response.status;
            error.data = data;
            throw error;
        }

        return data || {};
    })
    .then(data => {
        // Always track execution end
        trackExecutionEnd(cellId);
        
        if (data.success) {
            showExecutionStatus(cellId, 'success');
            
            // Update rate limit monitor after successful execution
            if (window.rateLimitMonitor) {
                setTimeout(() => window.rateLimitMonitor.fetchRateLimitStatus(), 1000);
            }
            
            // Handle output
            let textOutput = '';
            let plots = [];
            
            if (data.output) {
                const output = typeof data.output === 'string' ? data.output : data.output.stdout || '';
                textOutput = output;
            } else {
                textOutput = 'Code executed successfully';
            }
            
            // Handle plots if they exist in the response
            if (data.plots && Array.isArray(data.plots)) {
                plots = data.plots;
            }
            
            // Show text output
            showOutput(cellId, textOutput, 'success');
            
            // Show plots inline if any
            if (plots && plots.length > 0) {
                showPlotsInline(cellId, plots);
            }

            if (window.EDAChatBridge && typeof window.EDAChatBridge.recordCustomSuccess === 'function') {
                window.EDAChatBridge.recordCustomSuccess(cellId, {
                    code,
                    textOutput,
                    plots,
                    rawResponse: data
                });
            }
        } else {
            showExecutionStatus(cellId, 'error');
            showOutput(cellId, data.error || 'Execution failed', 'error');

            if (window.EDAChatBridge && typeof window.EDAChatBridge.recordCustomError === 'function') {
                window.EDAChatBridge.recordCustomError(cellId, {
                    code,
                    errorMessage: data.error || 'Execution failed',
                    rawResponse: data
                });
            }
        }
    })
    .catch(error => {
        // Always track execution end on error
        trackExecutionEnd(cellId);
        
        console.error('Execution error:', error);
        showExecutionStatus(cellId, 'error');
        
        // Update rate limit monitor after failed execution
        if (window.rateLimitMonitor) {
            setTimeout(() => window.rateLimitMonitor.fetchRateLimitStatus(), 1000);
        }
        
        if (error && error.isApiError) {
            const errorData = error.data || {};
            const outputs = Array.isArray(errorData.outputs) ? errorData.outputs : [];
            const errorOutput = outputs.find(output => output && output.type === 'error');
            const fallbackOutput = outputs.length > 0 ? outputs[0] : null;
            const outputText = (errorOutput && errorOutput.text) || (fallbackOutput && fallbackOutput.text) || errorData.error || error.message;
            showOutput(cellId, outputText, 'error');

            if (window.EDAChatBridge && typeof window.EDAChatBridge.recordCustomError === 'function') {
                window.EDAChatBridge.recordCustomError(cellId, {
                    code,
                    errorMessage: outputText,
                    rawResponse: errorData
                });
            }
        } else {
            showOutput(cellId, `Network error: ${error.message}`, 'error');

            if (window.EDAChatBridge && typeof window.EDAChatBridge.recordCustomError === 'function') {
                window.EDAChatBridge.recordCustomError(cellId, {
                    code,
                    errorMessage: error.message
                });
            }
        }
    });
}

function showExecutionStatus(cellId, status) {
    const statusElement = document.getElementById(`status-${cellId}`);
    if (!statusElement) return;
    
    statusElement.className = `execution-status ${status}`;
    statusElement.classList.remove('d-none');
    
    switch (status) {
        case 'running':
            statusElement.innerHTML = '<i class="bi bi-arrow-clockwise"></i> Running...';
            break;
        case 'success':
            statusElement.innerHTML = '<i class="bi bi-check-circle"></i> Success';
            setTimeout(() => statusElement.classList.add('d-none'), 3000);
            break;
        case 'error':
            statusElement.innerHTML = '<i class="bi bi-exclamation-triangle"></i> Error';
            break;
    }
}

function showOutput(cellId, output, type) {
    const outputElement = document.getElementById(`output-${cellId}`);
    if (!outputElement) return;
    
    const preElement = outputElement.querySelector('pre');
    
    // Handle different types of output
    if (typeof output === 'object') {
        if (output.stdout) {
            preElement.textContent = output.stdout;
        } else if (output.result) {
            preElement.textContent = output.result;
        } else {
            preElement.textContent = JSON.stringify(output, null, 2);
        }
    } else if (typeof output === 'string') {
        preElement.textContent = output;
    } else {
        preElement.textContent = String(output);
    }
    
    outputElement.className = `code-output ${type}`;
    outputElement.classList.remove('d-none');

    // Smooth scroll to the output container
    setTimeout(() => {
        outputElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 80);
}

function showPlotsInline(cellId, plots) {
    const outputElement = document.getElementById(`output-${cellId}`);
    if (!outputElement) return;
    
    // Create plots container after text output
    const plotsContainer = document.createElement('div');
    plotsContainer.className = 'plots-inline-container mt-3';
    
    plots.forEach((plotData, index) => {
        const plotDiv = document.createElement('div');
        plotDiv.className = 'plot-inline mb-3';
        plotDiv.innerHTML = `
            <div class="plot-wrapper">
                <img src="data:image/png;base64,${plotData}" 
                     class="plot-image custom-analysis-plot" 
                     alt="Plot ${index + 1}">
            </div>
        `;
        plotsContainer.appendChild(plotDiv);
    });
    
    outputElement.appendChild(plotsContainer);
}

function toggleCellCollapse(cellId) {
    const cell = document.getElementById(cellId);
    const collapseBtn = cell.querySelector('.collapse-btn i');
    const floatControls = cell.querySelector('.cell-float-controls');
    
    if (cell.classList.contains('collapsed')) {
        cell.classList.remove('collapsed');
        collapseBtn.className = 'bi bi-chevron-down';
        if (floatControls) {
            floatControls.style.display = 'flex';
        }
    } else {
        cell.classList.add('collapsed');
        collapseBtn.className = 'bi bi-chevron-right';
        if (floatControls) {
            floatControls.style.display = 'none';
        }
    }
}

function toggleWordWrap(cellId) {
    const editor = codeEditors.get(cellId);
    if (editor) {
        const currentWrap = editor.getOption('lineWrapping');
        editor.setOption('lineWrapping', !currentWrap);
    }
}

function toggleWordWrapAll() {
    const globalBtn = document.getElementById('globalWordWrapBtn');
    const floatingBtn = document.getElementById('toggleWrapFloatingBtn');
    const allCells = document.querySelectorAll('.custom-code-cell');
    
    if (allCells.length === 0) {
        return null;
    }
    
    // Check current state from first cell
    const firstCellId = allCells[0].id;
    const firstEditor = codeEditors.get(firstCellId);
    const currentWrap = firstEditor ? firstEditor.getOption('lineWrapping') : false;
    const newWrap = !currentWrap;
    
    // Apply to all cells
    allCells.forEach(cell => {
        const cellId = cell.id;
        const editor = codeEditors.get(cellId);
        if (editor) {
            editor.setOption('lineWrapping', newWrap);
        }
    });
    
    // Update global button appearance
    if (globalBtn) {
        if (newWrap) {
            globalBtn.classList.remove('btn-outline-info');
            globalBtn.classList.add('btn-info');
            globalBtn.title = 'Disable Word Wrap for All Cells (Alt+Z)';
            globalBtn.setAttribute('aria-pressed', 'true');
        } else {
            globalBtn.classList.remove('btn-info');
            globalBtn.classList.add('btn-outline-info');
            globalBtn.title = 'Enable Word Wrap for All Cells (Alt+Z)';
            globalBtn.setAttribute('aria-pressed', 'false');
        }
    }

    if (floatingBtn) {
        if (newWrap) {
            floatingBtn.classList.remove('btn-outline-info');
            floatingBtn.classList.add('btn-info', 'text-white');
            floatingBtn.setAttribute('aria-pressed', 'true');
            floatingBtn.title = 'Disable word wrap for all cells (Alt+Z)';
        } else {
            floatingBtn.classList.add('btn-outline-info');
            floatingBtn.classList.remove('btn-info', 'text-white');
            floatingBtn.setAttribute('aria-pressed', 'false');
            floatingBtn.title = 'Enable word wrap for all cells (Alt+Z)';
        }
    }

    return newWrap;
}

function showVariablesInfo() {
    // Create or show variables info panel
    let existingPanel = document.getElementById('variablesInfoPanel');
    let existingOverlay = document.getElementById('variablesOverlay');
    
    if (existingPanel) {
        existingPanel.remove();
        if (existingOverlay) existingOverlay.remove();
        return;
    }
    
    // Create overlay for click-outside functionality
    const overlay = document.createElement('div');
    overlay.id = 'variablesOverlay';
    overlay.className = 'variables-panel-overlay show';
    overlay.onclick = function() {
        const panel = document.getElementById('variablesInfoPanel');
        const overlay = document.getElementById('variablesOverlay');
        if (panel) panel.remove();
        if (overlay) overlay.remove();
    };
    
    const panel = document.createElement('div');
    panel.id = 'variablesInfoPanel';
    panel.className = 'variables-panel show';
    
    panel.innerHTML = `
        <div class="variables-panel-header">
            <span><i class="bi bi-code-square"></i> Available Variables</span>
            <button type="button" class="variables-panel-close" onclick="document.getElementById('variablesInfoPanel').remove(); document.getElementById('variablesOverlay').remove();">
                ×
            </button>
        </div>
        <div class="variables-panel-content">
            <div class="variables-category">
                <h6>Data Variables</h6>
                <div class="variable-item clickable" onclick="insertVariableToActiveCell('df')">df</div>
            </div>
            
            <div class="variables-category">
                <h6>Analysis Libraries</h6>
                <div class="variable-item clickable" onclick="insertVariableToActiveCell('pd')">pd (Pandas)</div>
                <div class="variable-item clickable" onclick="insertVariableToActiveCell('np')">np (NumPy)</div>
                <div class="variable-item clickable" onclick="insertVariableToActiveCell('plt')">plt (Matplotlib)</div>
                <div class="variable-item clickable" onclick="insertVariableToActiveCell('sns')">sns (Seaborn)</div>
            </div>
        </div>
    `;
    
    document.body.appendChild(overlay);
    document.body.appendChild(panel);
    
    // Auto-hide after 15 seconds
    setTimeout(() => {
        const panel = document.getElementById('variablesInfoPanel');
        const overlay = document.getElementById('variablesOverlay');
        if (panel) panel.remove();
        if (overlay) overlay.remove();
    }, 15000);
}

function insertVariableToActiveCell(variable) {
    // Find the most recently clicked or focused editor
    let activeEditor = null;
    
    // Try to find an editor that has focus
    codeEditors.forEach((editor, cellId) => {
        if (editor.hasFocus()) {
            activeEditor = editor;
            return;
        }
    });
    
    // If no focused editor, use the first available editor
    if (!activeEditor && codeEditors.size > 0) {
        activeEditor = codeEditors.values().next().value;
    }
    
    if (activeEditor) {
        const cursor = activeEditor.getCursor();
        activeEditor.replaceRange(variable, cursor);
        activeEditor.focus();
        
        // Close the variables panel
        const panel = document.getElementById('variablesInfoPanel');
        const overlay = document.getElementById('variablesOverlay');
        if (panel) panel.remove();
        if (overlay) overlay.remove();
    } else {
        showNotification('No code cell available. Please create a code cell first.', 'warning');
    }
}

function moveCustomCellUp(cellId) {
    console.log('Move custom cell up called for cell:', cellId);
    const cell = document.getElementById(cellId);
    if (!cell) {
        console.error('Custom cell not found with ID:', cellId);
        showNotification('Error: Cell not found', 'error');
        return;
    }
    
    const previousCell = cell.previousElementSibling;
    if (previousCell) {
        cell.parentNode.insertBefore(cell, previousCell);
        showNotification('Cell moved up', 'success');
    } else {
        showNotification('Cell is already at the top', 'info');
    }
}

function moveCustomCellDown(cellId) {
    console.log('Move custom cell down called for cell:', cellId);
    const cell = document.getElementById(cellId);
    if (!cell) {
        console.error('Custom cell not found with ID:', cellId);
        showNotification('Error: Cell not found', 'error');
        return;
    }
    
    const nextCell = cell.nextElementSibling;
    if (nextCell) {
        cell.parentNode.insertBefore(nextCell, cell);
        showNotification('Cell moved down', 'success');
    } else {
        showNotification('Cell is already at the bottom', 'info');
    }
}

function clearOutput(cellId) {
    const outputElement = document.getElementById(`output-${cellId}`);
    const statusElement = document.getElementById(`status-${cellId}`);
    
    if (outputElement) {
        outputElement.classList.add('d-none');
        outputElement.querySelector('pre').textContent = '';
        // Remove any plot containers
        outputElement.querySelectorAll('.plots-inline-container').forEach(el => el.remove());
    }
    
    if (statusElement) {
        statusElement.classList.add('d-none');
    }
}

function deleteCustomCell(cellId) {
    if (confirm('Are you sure you want to delete this code cell?')) {
        const cellElement = document.getElementById(cellId);
        if (cellElement) {
            cellElement.remove();
            codeEditors.delete(cellId);
            updateCustomAnalysisEmptyState();
            if (window.EDAChatBridge && typeof window.EDAChatBridge.removeCustomCell === 'function') {
                window.EDAChatBridge.removeCustomCell(cellId);
            }
        }
    }
}

function runAllCells() {
    const cells = document.querySelectorAll('.custom-code-cell');
    if (cells.length === 0) {
        showNotification('No code cells found to execute', 'info');
        return;
    }
    
    if (confirm(`Are you sure you want to run all ${cells.length} code cells?`)) {
        let cellIndex = 0;
        
        function runNextCell() {
            if (cellIndex < cells.length) {
                const cellId = cells[cellIndex].id;
                const editor = codeEditors.get(cellId);
                
                // Skip empty cells
                if (editor && editor.getValue().trim()) {
                    showNotification(`Running cell ${cellIndex + 1} of ${cells.length}...`, 'info');
                    executeCustomCode(cellId);
                }

                cellIndex++;
                // Run next cell after a short delay to prevent overwhelming the server
                setTimeout(runNextCell, 1500);
            } else {
                showNotification('All cells have been executed!', 'success');
            }
        }
        
        runNextCell();
    }
}

function deleteAllCells() {
    const cells = document.querySelectorAll('.custom-analysis-entry');
    if (cells.length === 0) {
        showNotification('No custom entries found to delete', 'info');
        return;
    }
    
    if (confirm(`Are you sure you want to delete all ${cells.length} custom entries? This action cannot be undone.`)) {
        cells.forEach(cell => {
            const cellId = cell.id;
            cell.remove();
            codeEditors.delete(cellId);
        });
        
        // Reset counters
        customCellCounter = 1;
        customMarkdownCounter = 1;
        customExecutionCounter = 1;

        updateCustomAnalysisEmptyState();

        const currentSourceId = (typeof sourceId !== 'undefined' && sourceId) ? sourceId : window.sourceId;
        if (currentSourceId) {
            fetch(`/advanced-eda/api/reset-custom-workspace/${currentSourceId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json().then(data => ({ ok: response.ok, data })))
            .then(({ ok, data }) => {
                if (ok && data.success) {
                    showNotification('Workspace cleared and runtime reset.', 'success');
                } else {
                    const errorMessage = (data && data.error) ? data.error : 'Sandbox reset failed';
                    showNotification(`Workspace cleared, but runtime reset failed: ${errorMessage}`, 'warning');
                }
            })
            .catch(error => {
                console.error('Failed to reset sandbox:', error);
                showNotification(`Workspace cleared, but runtime reset failed: ${error.message}`, 'warning');
            });
        } else {
            showNotification('Workspace cleared. Load a dataset to start a new session.', 'info');
        }
    }
}

function toggleExamplesPanel() {
    const panel = document.getElementById('examplesPanel');
    const overlay = document.getElementById('examplesPanelOverlay');
    
    if (panel && overlay) {
        // Use the new side panel system with show class and overlay
        if (panel.classList.contains('show')) {
            panel.classList.remove('show');
            overlay.classList.remove('show');
        } else {
            panel.classList.add('show');
            overlay.classList.add('show');
        }
    } else {
        // Fallback for old system with d-none
        if (panel) {
            panel.classList.toggle('d-none');
        }
    }
}

function hideExamplesPanel() {
    const panel = document.getElementById('examplesPanel');
    const overlay = document.getElementById('examplesPanelOverlay');
    
    if (panel && overlay) {
        // Use the new side panel system
        panel.classList.remove('show');
        overlay.classList.remove('show');
    } else if (panel) {
        // Fallback for old system
        panel.classList.add('d-none');
    }
}

function importCode() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.py,.txt';
    
    input.onchange = function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const cellId = addCustomCodeCell();
                const editor = codeEditors.get(cellId);
                if (editor) {
                    editor.setValue(e.target.result);
                }
            };
            reader.readAsText(file);
        }
    };
    
    input.click();
}

function applyEditorTheme(editor) {
    // Apply theme based on current mode
    const isDark = document.body.classList.contains('dark-mode');
    editor.setOption('theme', isDark ? 'material-darker' : 'default');
}

function syncCustomAnalysisDatasetMeta(datasetInfo = {}) {
    const nameEl = document.getElementById('customAnalysisDatasetName');
    const statsEl = document.getElementById('customAnalysisDatasetStats');

    if (!nameEl && !statsEl) {
        return;
    }

    const fallbackInfo = window.currentDataset?.info || window.currentDataset || {};

    const isMeaningful = (value) => {
        if (value === undefined || value === null) return false;
        if (typeof value === 'string') return value.trim() !== '';
        if (typeof value === 'number') return Number.isFinite(value);
        return true;
    };

    const getNested = (source, path) => {
        if (!source) return undefined;
        return path.split('.').reduce((acc, key) => {
            if (acc === undefined || acc === null) {
                return undefined;
            }
            const numericKey = Number(key);
            if (!Number.isNaN(numericKey) && key.trim() !== '') {
                return acc[numericKey];
            }
            return acc[key];
        }, source);
    };

    const pickValue = (...paths) => {
        for (const path of paths) {
            const candidate = getNested(datasetInfo, path);
            if (isMeaningful(candidate)) return candidate;

            const fallbackCandidate = getNested(fallbackInfo, path);
            if (isMeaningful(fallbackCandidate)) return fallbackCandidate;
        }
        return null;
    };

    if (nameEl) {
        const datasetName = pickValue('name', 'dataset_name', 'title', 'metadata.name', 'info.name') || 'Current dataset';
        nameEl.textContent = datasetName;
    }

    if (statsEl) {
        const rowsValue = pickValue('row_count', 'rows', 'total_rows', 'shape.0', 'metadata.row_count', 'metadata.rows', 'info.row_count', 'data.row_count');
        const columnsValue = pickValue('column_count', 'columns', 'total_columns', 'shape.1', 'metadata.column_count', 'metadata.columns', 'info.column_count', 'data.column_count');

        const formatMetric = (value, label) => {
            if (!isMeaningful(value)) return null;
            const numericValue = Number(value);
            const formatted = Number.isFinite(numericValue) ? numericValue.toLocaleString() : value;
            return `${formatted} ${label}`;
        };

        const parts = [];
        const rowsText = formatMetric(rowsValue, rowsValue === 1 ? 'row' : 'rows');
        const columnsText = formatMetric(columnsValue, columnsValue === 1 ? 'column' : 'columns');

        if (rowsText) parts.push(rowsText);
        if (columnsText) parts.push(columnsText);

        statsEl.textContent = parts.length ? parts.join(' · ') : 'Rows & columns pending';
    }
}


// Theme change handler
document.addEventListener('themeChanged', function() {
    codeEditors.forEach(editor => {
        applyEditorTheme(editor);
    });
});

// Export functions to global scope
window.customAnalysis = {
    addCustomCodeCell,
    executeCustomCode,
    clearOutput,
    deleteCustomCell,
    toggleCellCollapse,
    toggleWordWrap,
    toggleWordWrapAll,
    showVariablesInfo,
    insertVariableToActiveCell,
    showPlotsInline,
    runAllCells,
    deleteAllCells,
    moveCustomCellUp,
    moveCustomCellDown,
    toggleExamplesPanel,
    hideExamplesPanel,
    importCode,
    syncCustomAnalysisDatasetMeta
};

// Make functions globally available for onclick handlers
window.addCustomCodeCell = addCustomCodeCell;
window.executeCustomCode = executeCustomCode;
window.clearOutput = clearOutput;
window.deleteCustomCell = deleteCustomCell;
window.toggleCellCollapse = toggleCellCollapse;
window.toggleWordWrap = toggleWordWrap;
window.toggleWordWrapAll = toggleWordWrapAll;
window.showVariablesInfo = showVariablesInfo;
window.insertVariableToActiveCell = insertVariableToActiveCell;
window.runAllCells = runAllCells;
window.deleteAllCells = deleteAllCells;
window.moveCustomCellUp = moveCustomCellUp;
window.moveCustomCellDown = moveCustomCellDown;
window.toggleExamplesPanel = toggleExamplesPanel;
window.hideExamplesPanel = hideExamplesPanel;
window.importCode = importCode;
window.addCustomMarkdownCell = addCustomMarkdownCell;
window.toggleMarkdownPreview = toggleMarkdownPreview;
window.deleteCustomMarkdownCell = deleteCustomMarkdownCell;
window.syncCustomAnalysisDatasetMeta = syncCustomAnalysisDatasetMeta;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeCustomAnalysis();
    
    // Wire global shortcuts and toolbar buttons
    const addCellBtn = document.getElementById('addCellBtn');
    if (addCellBtn) {
        addCellBtn.addEventListener('click', function() {
            addCustomCodeCell();
        });
    }

    const addMarkdownBtn = document.getElementById('addMarkdownBtn');
    if (addMarkdownBtn) {
        addMarkdownBtn.addEventListener('click', function() {
            addCustomMarkdownCell();
        });
    }

    const toggleWrapFloatingBtn = document.getElementById('toggleWrapFloatingBtn');
    if (toggleWrapFloatingBtn) {
        toggleWrapFloatingBtn.addEventListener('click', function() {
            toggleWordWrapAll();
        });
    }

    document.addEventListener('sharedDataUpdated', function(event) {
        if (event?.detail?.key === 'dataset') {
            syncCustomAnalysisDatasetMeta(event.detail.value || {});
        }
    });

    if (window.currentDataset) {
        const initialDatasetInfo = window.currentDataset.info || window.currentDataset;
        syncCustomAnalysisDatasetMeta(initialDatasetInfo);
    }

    document.addEventListener('keydown', function(e) {
        // Alt+Z for word wrap toggle
        if (e.altKey && e.key === 'z') {
            e.preventDefault();
            toggleWordWrapAll();
        }
    });
});

console.log('Custom Analysis JavaScript loaded successfully');