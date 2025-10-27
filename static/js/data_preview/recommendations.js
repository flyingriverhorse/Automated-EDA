/**
 * recommendations.js
 * ==================
 * 
 * Enhanced Recommendations Module for Data Preview Page
 * 
 * Purpose:
 *  - Provide comprehensive, actionable data analysis recommendations
 *  - Offer strategic dataset-level insights and project suggestions
 *  - Integrate data quality issues with specific solutions
 *  - Guide users through data preparation and ML pipeline decisions
 *  - Include interactive filtering and search capabilities
 *  - Smart missing data thresholds: recommend dropping columns with >90% missing data
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    global.DI.recommendations = global.DI.recommendations || {};
    const recommendationsNamespace = global.DI.recommendations;

    const pendingContainerMap = new Map();
    
    // Configuration
    const config = {
        highMissingThreshold: 20,
        criticalMissingThreshold: 70,  // Above this, recommend dropping
        extremeMissingThreshold: 90,   // Almost certainly should drop
        veryHighCardinalityThreshold: 90,
        largeDatasetThreshold: 100 * 1024 * 1024 // 100MB
    };
    
    // Cache DOM elements
    const domCache = {};

    const defaultFilterState = {
        priority: 'all',
        category: 'all',
        focus: 'all',
        search: ''
    };

    let filterState = { ...defaultFilterState };

    function resetFilterState(overrides = {}) {
        filterState = { ...defaultFilterState, ...overrides };
    }

    const SIGNAL_FOCUS_LABELS = {
        missing_data: { label: 'Missing Data', icon: 'fas fa-droplet' },
        empty_column: { label: 'Empty Columns', icon: 'fas fa-ban' },
        low_variance: { label: 'Low Variance', icon: 'fas fa-wave-square' },
        outlier: { label: 'Outliers', icon: 'fas fa-chart-line' },
        skewness: { label: 'Skewness', icon: 'fas fa-chart-area' },
        multicollinearity: { label: 'Multi-collinearity', icon: 'fas fa-project-diagram' },
        nlp: { label: 'NLP', icon: 'fas fa-language' },
        text_quality: { label: 'Text Quality', icon: 'fas fa-paragraph' },
        high_cardinality: { label: 'High Cardinality', icon: 'fas fa-th-large' },
        pattern_detection: { label: 'Pattern Detection', icon: 'fas fa-fingerprint' },
        feature_engineering: { label: 'Feature Engineering', icon: 'fas fa-cogs' },
        privacy: { label: 'Privacy', icon: 'fas fa-user-shield' },
        project_roadmap: { label: 'Project Roadmap', icon: 'fas fa-route' },
        advanced_analysis: { label: 'Advanced Analysis', icon: 'fas fa-flask' },
        class_imbalance: { label: 'Class Imbalance', icon: 'fas fa-balance-scale' },
        data_drift: { label: 'Data Drift', icon: 'fas fa-arrows-rotate' },
        data_quality: { label: 'Data Quality', icon: 'fas fa-triangle-exclamation' }
    };

    const SIGNAL_ALIAS_MAP = {
        missing: 'missing_data',
        'missing-data': 'missing_data',
        'missing-values': 'missing_data',
        nulls: 'missing_data',
        null: 'missing_data',
        sparsity: 'empty_column',
        sparse: 'empty_column',
        constant: 'low_variance',
        'low-variance': 'low_variance',
        variance: 'low_variance',
        outliers: 'outlier',
        'outlier-detection': 'outlier',
    skew: 'skewness',
    skewed: 'skewness',
    'skewed-distribution': 'skewness',
    'distribution-skew': 'skewness',
    'skewness-score': 'skewness',
        kurtosis: 'skewness',
        imbalance: 'class_imbalance',
        'class-imbalance': 'class_imbalance',
        'classimbalance': 'class_imbalance',
        'imbalance-ratio': 'class_imbalance',
        'minority-class': 'class_imbalance',
        multicolinear: 'multicollinearity',
        multicollinearity: 'multicollinearity',
        correlation: 'multicollinearity',
        collinearity: 'multicollinearity',
        nlp_text: 'nlp',
        text: 'nlp',
        sentiment: 'text_quality',
        cleanliness: 'text_quality',
        cardinality: 'high_cardinality',
        hashing: 'high_cardinality',
        pattern: 'pattern_detection',
        leakage: 'privacy',
        pii: 'privacy',
        governance: 'privacy',
        roadmap: 'project_roadmap',
        strategy: 'project_roadmap',
        monitoring: 'advanced_analysis',
        optimization: 'advanced_analysis',
        drift: 'data_drift',
        'data-drift': 'data_drift',
        'concept-drift': 'data_drift',
        'covariate-shift': 'data_drift',
        'drift-detection': 'data_drift',
        quality: 'data_quality',
        'data-quality': 'data_quality',
        'quality-issue': 'data_quality'
    };

    const KNOWN_SIGNAL_KEYS = new Set([...Object.keys(SIGNAL_FOCUS_LABELS)]);

    const TEXT_SIGNAL_PATTERNS = [
        { regex: /\b(skew|skewness|skewed|kurtosis)\b/i, signal: 'skewness' },
        { regex: /\b(imbalance|class imbalance|imbalanced|minority class)\b/i, signal: 'class_imbalance' },
        { regex: /\b(outlier|anomal(y|ies)|z-score)\b/i, signal: 'outlier' },
        { regex: /\bmissing data|nulls?|impute|na\b/i, signal: 'missing_data' },
        { regex: /\bmulticollinearity|collinear|correlation matrix\b/i, signal: 'multicollinearity' },
        { regex: /\bhigh cardinality|unique values|identifier\b/i, signal: 'high_cardinality' },
        { regex: /\btoken|nlp|text processing|language model\b/i, signal: 'nlp' },
        { regex: /\bprivacy|pii|sensitive|mask\b/i, signal: 'privacy' },
        { regex: /\b(drifts?|shift|covariate shift|concept drift)\b/i, signal: 'data_drift' },
        { regex: /\b(data quality|quality issue|profiling|validation)\b/i, signal: 'data_quality' }
    ];

    const PRIORITY_LABELS = {
        critical: { label: 'Critical', icon: 'fas fa-fire' },
        high: { label: 'High', icon: 'fas fa-exclamation-circle' },
        medium: { label: 'Medium', icon: 'fas fa-equals' },
        strategic: { label: 'Strategic', icon: 'fas fa-compass' },
        advanced: { label: 'Advanced', icon: 'fas fa-flask' },
        low: { label: 'Low', icon: 'fas fa-leaf' }
    };

    const CATEGORY_FILTER_CONFIG = [
        { key: 'high-priority-issues', label: 'High Priority', icon: 'fas fa-shield-halved', analysisKey: 'highPriority' },
        { key: 'data-preparation-feature-engineering', label: 'Data Preparation', icon: 'fas fa-screwdriver-wrench', analysisKey: 'dataPreparation' },
        { key: 'analysis-project-ideas', label: 'Project Ideas', icon: 'fas fa-lightbulb', analysisKey: 'projectIdeas' },
        { key: 'advanced-considerations', label: 'Advanced Lens', icon: 'fas fa-flask', analysisKey: 'advancedConsiderations' }
    ];
    
    /**
     * Render comprehensive recommendations in the specified container
     */
    function renderRecommendations(qualityReport, containerId) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.warn('Container not found:', containerId, '- deferring enhanced render');
            queueRecommendationsRender(qualityReport, containerId);
            return;
        }

        recommendationsNamespace.__lastRenderMode = 'enhanced';

        const shell = ensureRecommendationsShell(container);
        if (!shell.content) {
            console.warn('Recommendations shell missing content region');
            return;
        }

        showRecommendationsLoading(shell);

        // Use setTimeout to allow the UI to update before heavy processing
        setTimeout(() => {
            try {
                const analysis = analyzeDataset(qualityReport);
                const quickFilterSummary = buildQuickFilterSummary(analysis);

                resetFilterState();

                // Update dataset stats tiles
                updateDatasetStats(qualityReport, containerId);

                let html = '';

                // High Priority Recommendations (Critical Issues)
                if (analysis.highPriority.length > 0) {
                    html += renderRecommendationCategory('High Priority Issues', analysis.highPriority, 'high', 'fas fa-exclamation-triangle');
                }

                // Data Preparation & Feature Engineering
                if (analysis.dataPreparation.length > 0) {
                    html += renderRecommendationCategory('Data Preparation & Feature Engineering', analysis.dataPreparation, 'medium', 'fas fa-cogs');
                }

                // Analysis & Project Ideas
                if (analysis.projectIdeas.length > 0) {
                    html += renderRecommendationCategory('Analysis & Project Ideas', analysis.projectIdeas, 'strategic', 'fas fa-brain');
                }

                // Advanced Considerations
                if (analysis.advancedConsiderations.length > 0) {
                    html += renderRecommendationCategory('Advanced Considerations', analysis.advancedConsiderations, 'low', 'fas fa-graduation-cap');
                }

                if (!html.trim()) {
                    html = `
                        <div class="empty-recommendations">
                            <i class="fas fa-wand-magic-sparkles"></i>
                            <p>No automated recommendations detected. Try adjusting filters or exploring the data quality report for insights.</p>
                        </div>
                    `;
                }

                const quickFiltersMarkup = renderQuickFilters(quickFilterSummary);
                const finalMarkup = `
                    ${quickFiltersMarkup}
                    <div class="recommendation-categories">
                        ${html}
                    </div>
                `;

                shell.content.innerHTML = finalMarkup;

                if (shell.root) {
                    shell.root.setAttribute('data-hydrated', 'true');
                }

                // Initialize interactive features
                initInteractiveFeatures(shell.root);

            } catch (error) {
                console.error('Error rendering recommendations:', error);
                shell.content.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i>
                        Error generating recommendations: ${error.message}
                    </div>
                `;

                if (shell.root) {
                    shell.root.setAttribute('data-hydrated', 'error');
                }
            }
        }, 100);

        ensureRecommendationsWatchdog(containerId);
    }
    
    
    function getRecommendationsRoot(containerId = 'recommendationsTabContent') {
        const container = document.getElementById(containerId);
        if (!container) {
            return document;
        }
        return container.querySelector('[data-role="recommendations-root"]') || container;
    }

    function ensureRecommendationsShell(container) {
        if (!container) {
            return {};
        }

        let root = container.querySelector('[data-role="recommendations-root"]');
        if (!root) {
            buildRecommendationsShell(container);
            root = container.querySelector('[data-role="recommendations-root"]');
        }

    const content = container.querySelector('[data-role="recommendations-content"]');
    const main = container.querySelector('[data-role="recommendations-main"]');

    return { container, root, content, main };
    }

    function buildRecommendationsShell(container) {
        if (!container) {
            return;
        }

        container.innerHTML = `
            <div class="enhanced-recommendations-container" data-role="recommendations-root">
                <div class="recommendations-dashboard">
                    <div class="recommendations-main-content" data-role="recommendations-main">
                        <div class="recommendations-header" data-role="recommendations-header">
                            <h4><i class="fas fa-lightbulb"></i> Comprehensive Data Analysis Recommendations</h4>
                            <p class="subtitle">Intelligent insights for your dataset</p>
                        </div>

                        <div class="recommendations-search-bar">
                            <label for="search-filter">Search Recommendations</label>
                            <div class="search-input-wrapper">
                                <i class="fas fa-search"></i>
                                <input type="text" id="search-filter" placeholder="Search across titles, descriptions, actions, or code snippets...">
                            </div>
                        </div>

                        <div class="recommendations-content" data-role="recommendations-content">
                            <div class="loading-recommendations">
                                <i class="fas fa-spinner fa-spin"></i>
                                <p>Analyzing your dataset and generating recommendations...</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    function showRecommendationsLoading(shell) {
        if (!shell || !shell.content) {
            return;
        }

        if (shell.root) {
            shell.root.removeAttribute('data-hydrated');
        }

        shell.content.innerHTML = `
            <div class="loading-recommendations">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Analyzing your dataset and generating recommendations...</p>
            </div>
        `;
    }

    /**
     * Update dataset statistics tiles in the header
     */
    function updateDatasetStats(qualityReport, containerId = 'recommendationsTabContent') {
        const metadata = qualityReport.basic_metadata || {};
        const quality = qualityReport.quality_metrics || {};
        const textSummary = qualityReport.text_analysis_summary || {};
        
        const root = getRecommendationsRoot(containerId);
        const totalColsElement = root.querySelector('#total-columns');
        const sampleRowsElement = root.querySelector('#sample-rows');
        const completenessElement = root.querySelector('#completeness');
        const textColumnsElement = root.querySelector('#text-columns');
        
        if (totalColsElement) totalColsElement.textContent = metadata.total_columns || '-';
        if (sampleRowsElement) sampleRowsElement.textContent = metadata.sample_rows ? metadata.sample_rows.toLocaleString() : '-';
        if (completenessElement) completenessElement.textContent = quality.overall_completeness ? `${quality.overall_completeness.toFixed(1)}%` : '-';
        if (textColumnsElement) textColumnsElement.textContent = textSummary.total_text_columns || '-';
    }
    
    /**
     * Initialize interactive features like filtering and search
     */
    function initInteractiveFeatures(root = getRecommendationsRoot()) {
        if (!root) {
            return;
        }

        const searchInput = root.querySelector('#search-filter');

        if (searchInput) {
            if (searchInput.value !== filterState.search) {
                searchInput.value = filterState.search;
            }

            if (!searchInput.dataset.listenerAttached) {
                const handler = debounce(event => {
                    filterState.search = event.target.value || '';
                    applyFilters(root);
                }, 300);

                searchInput.addEventListener('input', handler);
                searchInput.dataset.listenerAttached = 'true';
            }
        }

        // Set up copy buttons within the root context
        root.querySelectorAll('.copy-code-btn').forEach(btn => {
            if (btn.dataset.copyListenerAttached) {
                return;
            }

            btn.addEventListener('click', function() {
                const codeId = this.getAttribute('data-code-id');
                if (codeId) {
                    copyToClipboard(codeId);
                } else {
                    // Fallback for old onclick method
                    const onclick = this.getAttribute('onclick');
                    if (onclick) {
                        const match = onclick.match(/copyToClipboard\('([^']+)'\)/);
                        if (match) {
                            copyToClipboard(match[1]);
                        }
                    }
                }
            });

            btn.dataset.copyListenerAttached = 'true';
        });

        setupQuickFilterChips(root);
    }
    
    /**
     * Update search results summary
     */
    function updateSearchResultsSummary(searchValue, isGlobalSearch, root = getRecommendationsRoot()) {
        // Remove existing search summary
        const existingSummary = root.querySelector('.search-results-summary');
        if (existingSummary) {
            existingSummary.remove();
        }
        
        if (isGlobalSearch && searchValue.length > 0) {
            const visibleItems = root.querySelectorAll('.recommendation-item:not([style*="display: none"])');
            const totalItems = root.querySelectorAll('.recommendation-item').length;
            
            const summaryHtml = `
                <div class="search-results-summary">
                    <i class="fas fa-search"></i>
                    <span>Found ${visibleItems.length} of ${totalItems} recommendations</span>
                    ${visibleItems.length === 0 ? '<span class="no-results">Try different keywords or clear search.</span>' : ''}
                    <button class="clear-search-btn" onclick="clearSearch()">
                        <i class="fas fa-times"></i> Clear
                    </button>
                </div>
            `;
            
            // Place the summary directly beneath the search block
            const searchContainer = root.querySelector('.recommendations-search-bar');
            if (searchContainer) {
                searchContainer.insertAdjacentHTML('afterend', summaryHtml);
            }
        }
    }

    /**
     * Clear search functionality
     */
    function clearSearch(containerId) {
        const root = getRecommendationsRoot(containerId);
        filterState.search = '';
        const searchInput = root.querySelector('#search-filter');
        if (searchInput) {
            searchInput.value = '';
        }
        applyFilters(root); // Re-apply filters without search
    }

    /**
     * Debounce function to limit how often a function can be called
     */
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    /**
     * Apply filters to recommendations based on user selection
     */
    function applyFilters(root = getRecommendationsRoot()) {
        if (!root) {
            return;
        }

        const {
            priorityValue,
            categoryValue,
            focusValue,
            searchValue: rawSearchValue
        } = getCurrentFilterValues();

        const searchValue = (rawSearchValue || '').toLowerCase().trim();
        
        // If there's a search term, prioritize global search over filters
        const isGlobalSearch = searchValue.length > 0;
        
        root.querySelectorAll('.recommendation-item').forEach(item => {
            let shouldShow = true;
            
            if (isGlobalSearch) {
                // Global search across all content
                const title = item.querySelector('.recommendation-title')?.textContent.toLowerCase() || '';
                const description = item.querySelector('.recommendation-description')?.textContent.toLowerCase() || '';
                const action = item.querySelector('.recommendation-action')?.textContent.toLowerCase() || '';
                const code = item.querySelector('pre code')?.textContent.toLowerCase() || '';
                const priority = item.getAttribute('data-priority') || '';
                const categoryHeader = item.closest('.recommendation-category')?.querySelector('.category-header')?.textContent.toLowerCase() || '';
                const tagsAttr = item.getAttribute('data-tags') || '';
                const focusAttr = item.getAttribute('data-focus') || '';
                
                // Search across all content
                const allContent = `${title} ${description} ${action} ${code} ${priority} ${categoryHeader} ${tagsAttr} ${focusAttr}`;
                shouldShow = allContent.includes(searchValue);
            } else {
                // Apply regular filters when no search term
                const priority = item.getAttribute('data-priority');
                const category = item.closest('.recommendation-category')?.getAttribute('data-category');
                const tags = (item.getAttribute('data-tags') || '').split(' ').filter(Boolean);
                
                const priorityMatch = priorityValue === 'all' || priority === priorityValue;
                const categoryMatch = categoryValue === 'all' || category === categoryValue;
                const focusMatch = focusValue === 'all' || tags.includes(focusValue);

                shouldShow = priorityMatch && categoryMatch && focusMatch;
            }
            
            item.style.display = shouldShow ? 'grid' : 'none';
        });
        
        // Hide empty categories
        root.querySelectorAll('.recommendation-category').forEach(category => {
            const visibleItems = Array.from(category.querySelectorAll('.recommendation-item')).filter(item => item.style.display !== 'none');
            if (visibleItems.length === 0) {
                category.style.display = 'none';
            } else {
                category.style.display = 'block';
            }
        });
        
        // Show search results summary if searching
        updateSearchResultsSummary(searchValue, isGlobalSearch, root);

        updateQuickFilterState(root);
    }
    
    /**
     * Comprehensive dataset analysis to generate strategic recommendations
     */
    function analyzeDataset(qualityReport) {
        const metadata = qualityReport.basic_metadata || {};
        const quality = qualityReport.quality_metrics || {};
        const columnDetails = quality.column_details || [];
        const textSummary = qualityReport.text_analysis_summary || {};
        const backendRecommendations = Array.isArray(qualityReport.recommendations) ? qualityReport.recommendations : [];
        
        const analysis = {
            highPriority: [],
            dataPreparation: [],
            projectIdeas: [],
            advancedConsiderations: []
        };
        
        // High Priority Issues (Critical Data Quality Problems)
        analyzeHighPriorityIssues(analysis, quality, columnDetails);
        
        // Data Preparation & Feature Engineering
        analyzeDataPreparation(analysis, columnDetails, quality);
        
        // Analysis & Project Ideas
        analyzeProjectIdeas(analysis, columnDetails, textSummary, metadata);
        
        // Advanced Considerations
        analyzeAdvancedConsiderations(analysis, columnDetails, metadata, quality);

        // Merge backend service recommendations
        mergeBackendRecommendations(analysis, backendRecommendations);
        
        return analysis;
    }

    function mergeBackendRecommendations(analysis, backendRecommendations) {
        if (!Array.isArray(backendRecommendations) || backendRecommendations.length === 0) {
            return;
        }

        backendRecommendations.forEach((rec, index) => {
            if (!rec) {
                return;
            }

            const title = rec.title || `EDA Recommendation ${index + 1}`;
            const description = rec.description || '';
            const action = rec.action || '';
            const code = rec.code || '';
            const normalized = {
                title,
                description,
                action,
                code,
                priority: (rec.priority || '').toString().trim().toLowerCase(),
                category: (rec.category || '').toString().trim().toLowerCase(),
            };

            if (!normalized.priority) {
                normalized.priority = determineBackendPriority(title, description, action);
            }

            if (!normalized.category) {
                normalized.category = inferBackendCategory(normalized.priority, title, description, action);
            }

            if (Array.isArray(rec.columns) && rec.columns.length > 0) {
                normalized.columns = rec.columns.map(col => col && col.toString ? col.toString() : String(col));
            }

            if (Array.isArray(rec.tags) && rec.tags.length > 0) {
                normalized.tags = rec.tags;
            }

            const passthroughFields = [
                'why_it_matters',
                'feature_impact',
                'metrics',
                'focus_areas',
                'signal_type',
                'priority_score',
                'priority_label',
                'category_label',
                'confidence',
                'references',
                'meta'
            ];

            passthroughFields.forEach(field => {
                if (rec[field] !== undefined) {
                    normalized[field] = rec[field];
                }
            });

            if (!normalized.code || !normalized.code.toString().trim()) {
                const normalizedTags = Array.isArray(normalized.tags) ? normalized.tags : [];
                const normalizedFocusAreas = Array.isArray(normalized.focus_areas) ? normalized.focus_areas : [];

                const hasTextOrientation = [
                    normalized.category,
                    normalized.signal_type,
                    ...normalizedTags,
                    ...normalizedFocusAreas
                ].some(value => {
                    if (!value) {
                        return false;
                    }
                    const marker = value.toString().toLowerCase();
                    return marker.includes('text_preprocessing') || marker.includes('text quality') || marker.includes('nlp') || marker.includes('text');
                });

                if (hasTextOrientation) {
                    const candidateColumn = Array.isArray(normalized.columns) && normalized.columns.length > 0
                        ? normalized.columns[0]
                        : 'text_column';
                    const columnLiteral = JSON.stringify(candidateColumn);

                    normalized.code = `# Text preprocessing template\nimport re\nimport nltk\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom nltk.corpus import stopwords\nfrom nltk.stem import WordNetLemmatizer\n\nnltk.download(['punkt', 'stopwords', 'wordnet'])\nSTOP_WORDS = set(stopwords.words('english'))\nLEMMATIZER = WordNetLemmatizer()\n\ndef preprocess_text(text):\n    text = str(text).lower()\n    text = re.sub(r'[^a-z0-9\\s]', ' ', text)\n    tokens = [word for word in text.split() if word and word not in STOP_WORDS]\n    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]\n    return ' '.join(tokens)\n\nTEXT_COLUMN = ${columnLiteral}\ndf[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('').apply(preprocess_text)\n\nvectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))\ntext_features = vectorizer.fit_transform(df[TEXT_COLUMN])\nprint(f"TF-IDF feature matrix shape: {text_features.shape}")`;
                }
            }

            routeBackendRecommendation(analysis, normalized);
        });
    }

    function determineBackendPriority(title, description, action) {
        const text = `${title} ${description} ${action}`.toLowerCase();

        if (/pii|sensitive data|mask|governance|privacy/.test(text)) {
            return 'critical';
        }

        if (/missing data|null|quality flag|warning|issue/.test(text)) {
            return 'high';
        }

        if (/pattern|encoding|preprocessing|nlp/.test(text)) {
            return 'medium';
        }

        if (/strategy|opportunity|roadmap|project/.test(text)) {
            return 'strategic';
        }

        if (/advanced|outlier|investigate/.test(text)) {
            return 'advanced';
        }

        return 'medium';
    }

    function inferBackendCategory(priority, title, description, action) {
        const text = `${title} ${description} ${action}`.toLowerCase();

        if (/pii|sensitive|privacy|governance|compliance/.test(text)) {
            return 'privacy';
        }

        if (/missing|null|quality|impute|clean|data quality/.test(text)) {
            return 'data_quality';
        }

        if (/nlp|text|token|preprocess|language/.test(text)) {
            return 'text_preprocessing';
        }

        if (/categorical|encoding|one-hot|label/.test(text)) {
            return 'categorical_encoding';
        }

        if (/feature engineering|feature-engineering|feature/.test(text)) {
            return 'feature_engineering';
        }

        if (/project|idea|roadmap|strategy/.test(text) || priority === 'strategic') {
            return 'project_idea';
        }

        if (/outlier|advanced|monitor|govern|drift/.test(text) || priority === 'advanced') {
            return 'advanced_analysis';
        }

        return priority === 'high' || priority === 'critical' ? 'data_quality' : 'data_preparation';
    }

    function routeBackendRecommendation(analysis, rec) {
        const priority = (rec.priority || 'medium').toLowerCase();
        const category = (rec.category || '').toLowerCase();

        const highPriorityCategories = new Set(['privacy', 'data_quality', 'risk', 'compliance']);
        const dataPrepCategories = new Set(['data_preparation', 'text_preprocessing', 'categorical_encoding', 'feature_engineering', 'data_cleaning']);
        const projectIdeaCategories = new Set(['project_idea', 'project_roadmap', 'strategy', 'business_value']);
        const advancedCategories = new Set(['advanced_analysis', 'monitoring', 'optimization', 'governance']);

        if (highPriorityCategories.has(category) || priority === 'critical' || priority === 'high') {
            analysis.highPriority.push(rec);
            return;
        }

        if (projectIdeaCategories.has(category) || priority === 'strategic') {
            analysis.projectIdeas.push(rec);
            return;
        }

        if (advancedCategories.has(category) || priority === 'advanced') {
            analysis.advancedConsiderations.push(rec);
            return;
        }

        if (dataPrepCategories.has(category) || priority === 'medium') {
            analysis.dataPreparation.push(rec);
            return;
        }

        analysis.dataPreparation.push(rec);
    }
    
    /**
     * Analyze high priority data quality issues
     */
    function analyzeHighPriorityIssues(analysis, quality, columnDetails) {
        // Group columns by missing data severity for consolidated recommendations
        const extremeMissingCols = columnDetails.filter(col => col.null_percentage >= config.extremeMissingThreshold);
        const criticalMissingCols = columnDetails.filter(col => 
            col.null_percentage >= config.criticalMissingThreshold && 
            col.null_percentage < config.extremeMissingThreshold
        );
        const moderateMissingCols = columnDetails.filter(col => 
            col.null_percentage > config.highMissingThreshold && 
            col.null_percentage < config.criticalMissingThreshold
        );

        // Consolidated recommendation for extreme missing data (>=90%)
        if (extremeMissingCols.length > 0) {
            const columnList = extremeMissingCols.map(col => `"${col.name}" (${col.null_percentage.toFixed(1)}%)`).join(', ');
            const columnNames = extremeMissingCols.map(col => col.name);
            
            analysis.highPriority.push({
                title: extremeMissingCols.length === 1 ? `Drop Column: ${extremeMissingCols[0].name}` : `Drop ${extremeMissingCols.length} Columns with Extreme Missing Data`,
                description: extremeMissingCols.length === 1 
                    ? `The "${extremeMissingCols[0].name}" column has ${extremeMissingCols[0].null_percentage.toFixed(1)}% missing values. With such extreme missing data, this column provides no meaningful information.`
                    : `${extremeMissingCols.length} columns have extreme missing data (≥90%): ${columnList}. These columns are essentially empty and cannot contribute to analysis.`,
                action: `**Recommended Strategy:** Drop ${extremeMissingCols.length === 1 ? 'this column' : 'these columns'} entirely. ${extremeMissingCols.length === 1 ? "It's" : "They're"} essentially empty and cannot contribute to analysis.`,
                code: extremeMissingCols.length === 1 
                    ? `# Drop column with extreme missing data\ndf = df.drop('${extremeMissingCols[0].name}', axis=1)\nprint(f"Dropped column '${extremeMissingCols[0].name}' - {extremeMissingCols[0].null_percentage.toFixed(1)}% missing values")`
                    : `# Drop columns with extreme missing data (≥90%)\ncolumns_to_drop = ${JSON.stringify(columnNames)}\ndf = df.drop(columns=columns_to_drop)\nprint(f"Dropped {extremeMissingCols.length} columns with extreme missing data:")\n${extremeMissingCols.map(col => `print(f"  - ${col.name}: ${col.null_percentage.toFixed(1)}% missing")`).join('\n')}`,
                priority: 'critical'
            });
        }

        // Consolidated recommendation for critical missing data (70-89%)
        if (criticalMissingCols.length > 0) {
            const columnList = criticalMissingCols.map(col => `"${col.name}" (${col.null_percentage.toFixed(1)}%)`).join(', ');
            const columnNames = criticalMissingCols.map(col => col.name);
            
            analysis.highPriority.push({
                title: criticalMissingCols.length === 1 ? `Consider Dropping: ${criticalMissingCols[0].name}` : `Consider Dropping ${criticalMissingCols.length} Columns with Critical Missing Data`,
                description: criticalMissingCols.length === 1
                    ? `The "${criticalMissingCols[0].name}" column has ${criticalMissingCols[0].null_percentage.toFixed(1)}% missing values. This high missing rate seriously limits its analytical value.`
                    : `${criticalMissingCols.length} columns have critical missing data (70-89%): ${columnList}. These high missing rates seriously limit their analytical value.`,
                action: `**Recommended Strategy:** Strongly consider dropping ${criticalMissingCols.length === 1 ? 'this column' : 'these columns'}. If ${criticalMissingCols.length === 1 ? "it's" : "any are"} truly important for your analysis, investigate why so much data is missing before attempting imputation.`,
                code: criticalMissingCols.length === 1
                    ? `# Option 1: Drop the column (recommended)\ndf = df.drop('${criticalMissingCols[0].name}', axis=1)\n\n# Option 2: Investigate missing pattern first\nprint(f"Missing data pattern for ${criticalMissingCols[0].name}:")\nprint(df['${criticalMissingCols[0].name}'].isnull().value_counts())\n\n# Only if investigation shows meaningful pattern:\n# df['${criticalMissingCols[0].name}'].fillna('MISSING_DATA', inplace=True)`
                    : `# Option 1: Drop all columns (recommended)\ncolumns_to_drop = ${JSON.stringify(columnNames)}\ndf = df.drop(columns=columns_to_drop)\nprint(f"Dropped {criticalMissingCols.length} columns with critical missing data")\n\n# Option 2: Investigate missing patterns first\nfor col in ${JSON.stringify(columnNames)}:\n    print(f"\\nMissing data pattern for {col}:")\n    print(df[col].isnull().value_counts())\n    missing_pct = df[col].isnull().mean() * 100\n    print(f"Missing percentage: {missing_pct:.1f}%")\n\n# Only proceed with imputation if investigation shows meaningful patterns`,
                priority: 'critical'
            });
        }

        // Group moderate missing data by similar percentages and data types
        if (moderateMissingCols.length > 0) {
            // Group by data category and similar missing percentages (within 10% range)
            const groupedCols = groupColumnsBySimilarity(moderateMissingCols);
            
            groupedCols.forEach(group => {
                const columnList = group.map(col => `"${col.name}" (${col.null_percentage.toFixed(1)}%)`).join(', ');
                const columnNames = group.map(col => col.name);
                const avgMissingPct = group.reduce((sum, col) => sum + col.null_percentage, 0) / group.length;
                const dataCategory = group[0].data_category;
                
                let strategy = '';
                if (dataCategory === 'text') {
                    strategy = 'Consider creating a "Missing" category or use domain knowledge for imputation. Evaluate if missing data has a pattern.';
                } else if (dataCategory === 'numerical') {
                    strategy = 'Consider median/mean imputation, KNN imputation, or regression-based imputation. Analyze if missingness correlates with other variables.';
                } else {
                    strategy = 'Create a "Missing" category or use mode for imputation, but first investigate why data is missing.';
                }
                
                analysis.highPriority.push({
                    title: group.length === 1 ? `Moderate Missing Data: ${group[0].name}` : `${group.length} Columns with Moderate Missing Data (${dataCategory})`,
                    description: group.length === 1
                        ? `The "${group[0].name}" column has ${group[0].null_percentage.toFixed(1)}% missing values. This requires careful handling strategy.`
                        : `${group.length} ${dataCategory} columns have moderate missing data (avg: ${avgMissingPct.toFixed(1)}%): ${columnList}. These require careful handling strategy.`,
                    action: `**Recommended Strategy:** ${strategy}`,
                    code: group.length === 1
                        ? `# Analyze missing pattern first\nprint(f"Missing data analysis for ${group[0].name}:")\nprint(f"Missing count: ${group[0].null_count}")\nprint(f"Missing percentage: ${group[0].null_percentage.toFixed(1)}%")\n\n# Check correlation with other missing data\nmissing_corr = df.isnull().corr()['${group[0].name}'].sort_values(ascending=False)\nprint("Correlation with other missing data:", missing_corr.head())\n\n# Imputation (only after analysis)\n${dataCategory === 'numerical' ? 
                            `df['${group[0].name}'].fillna(df['${group[0].name}'].median(), inplace=True)` : 
                            `df['${group[0].name}'].fillna('Missing', inplace=True)`}`
                        : `# Analyze missing patterns for all columns\ncolumns_to_analyze = ${JSON.stringify(columnNames)}\nfor col in columns_to_analyze:\n    print(f"\\nMissing data analysis for {col}:")\n    missing_count = df[col].isnull().sum()\n    missing_pct = df[col].isnull().mean() * 100\n    print(f"Missing count: {missing_count}")\n    print(f"Missing percentage: {missing_pct:.1f}%")\n\n# Check correlation between missing data patterns\nmissing_corr = df[columns_to_analyze].isnull().corr()\nprint("\\nCorrelation between missing data patterns:")\nprint(missing_corr)\n\n# Batch imputation (only after analysis)\n${dataCategory === 'numerical' 
                            ? `for col in columns_to_analyze:\n    df[col].fillna(df[col].median(), inplace=True)`
                            : `for col in columns_to_analyze:\n    df[col].fillna('Missing', inplace=True)`}`,
                    priority: 'high'
                });
            });
        }
        
        // Whitespace issues
        const whitespaceCols = columnDetails.filter(col => 
            col.data_category === 'text' && col.text_category === 'whitespace_heavy'
        );
        whitespaceCols.forEach(col => {
            analysis.highPriority.push({
                title: `Text Cleaning Required: ${col.name}`,
                description: `The "${col.name}" column contains inconsistent whitespace that will create false categories and affect analysis accuracy.`,
                action: 'Clean whitespace before any analysis to ensure consistent categorization.',
                code: `# Clean whitespace in ${col.name}\ndf['${col.name}'] = df['${col.name}'].str.strip()`,
                priority: 'high'
            });
        });
        
        // Overall data quality
        if (quality.overall_completeness < 70) {
            analysis.highPriority.push({
                title: 'Overall Data Quality Concern',
                description: `Dataset completeness is only ${quality.overall_completeness.toFixed(1)}%, indicating systemic data quality issues.`,
                action: 'Consider data audit, source validation, and comprehensive cleaning strategy before analysis.',
                priority: 'critical'
            });
        }
    }
    
    /**
     * Analyze data preparation and feature engineering needs
     */
    function analyzeDataPreparation(analysis, columnDetails, quality) {
        // Categorical encoding recommendations
        const categoricalCols = columnDetails.filter(col => 
            col.data_category === 'text' && col.text_category === 'categorical'
        );
        
        if (categoricalCols.length > 0) {
            // Group by cardinality for specific recommendations
            const lowCardinality = categoricalCols.filter(col => col.unique_count <= 5);
            const mediumCardinality = categoricalCols.filter(col => col.unique_count > 5 && col.unique_count <= 20);
            const highCardinality = categoricalCols.filter(col => col.unique_count > 20);
            
            if (lowCardinality.length > 0) {
                analysis.dataPreparation.push({
                    title: 'One-Hot Encoding Recommended',
                    description: `Low cardinality categorical columns: ${lowCardinality.map(col => `"${col.name}" (${col.unique_count} categories)`).join(', ')}.`,
                    action: 'Use one-hot encoding for these columns as they have few categories and no natural ordering. Suitable for most ML algorithms.',
                    code: `# One-hot encode low cardinality columns\nfrom sklearn.preprocessing import OneHotEncoder\npd.get_dummies(df[${JSON.stringify(lowCardinality.map(col => col.name))}], prefix=${JSON.stringify(lowCardinality.map(col => col.name))})`,
                    priority: 'medium'
                });
            }
            
            if (mediumCardinality.length > 0) {
                analysis.dataPreparation.push({
                    title: 'Encoding Strategy for Medium Cardinality',
                    description: `Medium cardinality columns: ${mediumCardinality.map(col => `"${col.name}" (${col.unique_count} categories)`).join(', ')}.`,
                    action: 'Consider one-hot encoding for tree-based models, or target encoding for linear models. Monitor for overfitting with target encoding.',
                    code: `# Option 1: One-hot encoding\npd.get_dummies(df[${JSON.stringify(mediumCardinality.map(col => col.name))}])\n\n# Option 2: Target encoding (if you have a target variable)\nfrom category_encoders import TargetEncoder\nencoder = TargetEncoder()\nencoded = encoder.fit_transform(df['${mediumCardinality[0]?.name}'], df['target'])`,
                    priority: 'medium'
                });
            }
            
            if (highCardinality.length > 0) {
                analysis.dataPreparation.push({
                    title: 'High Cardinality Challenge',
                    description: `High cardinality columns: ${highCardinality.map(col => `"${col.name}" (${col.unique_count} categories)`).join(', ')}.`,
                    action: 'Consider frequency encoding, target encoding, or feature hashing. One-hot encoding will create too many features.',
                    code: `# Frequency encoding for high cardinality\nfreq_map = df['${highCardinality[0]?.name}'].value_counts().to_dict()\ndf['${highCardinality[0]?.name}_freq'] = df['${highCardinality[0]?.name}'].map(freq_map)\n\n# Or use feature hashing\nfrom sklearn.feature_extraction import FeatureHasher\nhasher = FeatureHasher(n_features=10)\nhashed_features = hasher.transform(df[['${highCardinality[0]?.name}']])`,
                    priority: 'medium'
                });
            }
        }
        
        // Date/time parsing
        const dateCols = columnDetails.filter(col => 
            (col.dtype === 'object' && (col.name.toLowerCase().includes('date') || col.name.toLowerCase().includes('time'))) ||
            col.dtype.includes('datetime')
        );
        
        if (dateCols.length > 0) {
            analysis.dataPreparation.push({
                title: 'Date/Time Feature Engineering',
                description: `Date columns detected: ${dateCols.map(col => `"${col.name}"`).join(', ')}.`,
                action: 'Parse dates and extract temporal features (year, month, day of week, hour) for time-based analysis.',
                code: `# Parse and extract date features\ndf['${dateCols[0]?.name}'] = pd.to_datetime(df['${dateCols[0]?.name}'])\ndf['year'] = df['${dateCols[0]?.name}'].dt.year\ndf['month'] = df['${dateCols[0]?.name}'].dt.month\ndf['day_of_week'] = df['${dateCols[0]?.name}'].dt.dayofweek\ndf['hour'] = df['${dateCols[0]?.name}'].dt.hour`,
                priority: 'medium'
            });
        }
        
        // Numerical feature scaling
        const numericalCols = columnDetails.filter(col => 
            col.data_category === 'numerical' || col.dtype.includes('int') || col.dtype.includes('float')
        );
        
        if (numericalCols.length > 1) {
            analysis.dataPreparation.push({
                title: 'Feature Scaling for Numerical Data',
                description: `${numericalCols.length} numerical columns detected. Different scales may affect model performance.`,
                action: 'Apply StandardScaler for linear models, MinMaxScaler for neural networks, or RobustScaler if outliers are present.',
                code: `# Feature scaling\nfrom sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nnumerical_cols = ${JSON.stringify(numericalCols.slice(0, 5).map(col => col.name))}\ndf[numerical_cols] = scaler.fit_transform(df[numerical_cols])`,
                priority: 'medium'
            });
        }

        if (numericalCols.length > 0 && categoricalCols.length > 0) {
            const numericFeatureNames = numericalCols.slice(0, 5).map(col => col.name);
            const categoricalFeatureNames = categoricalCols.slice(0, 5).map(col => col.name);

            analysis.dataPreparation.push({
                title: 'Build a Feature Engineering Pipeline',
                description: `Blend ${numericalCols.length} numerical and ${categoricalCols.length} categorical columns into a reusable pipeline that automates preprocessing and creates richer features.`,
                action: 'Leverage ColumnTransformer with dedicated numeric and categorical sub-pipelines so you can add scaling, encoders, and interaction features in one place.',
                code: `# Feature engineering pipeline for mixed data\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.linear_model import LogisticRegression\n\nTARGET_COLUMN = 'target'  # update with your target column name\nnumeric_features = ${JSON.stringify(numericFeatureNames)}\ncategorical_features = ${JSON.stringify(categoricalFeatureNames)}\n\nif TARGET_COLUMN not in df.columns:\n    raise KeyError(f"Column '{TARGET_COLUMN}' not found. Update TARGET_COLUMN with your target label.")\n\nnumeric_transformer = Pipeline([\n    ('imputer', SimpleImputer(strategy='median')),\n    ('scaler', StandardScaler()),\n    ('poly', PolynomialFeatures(degree=2, include_bias=False))\n])\n\ncategorical_transformer = Pipeline([\n    ('imputer', SimpleImputer(strategy='most_frequent')),\n    ('encoder', OneHotEncoder(handle_unknown='ignore'))\n])\n\npreprocessor = ColumnTransformer(\n    transformers=[\n        ('num', numeric_transformer, numeric_features),\n        ('cat', categorical_transformer, categorical_features)\n    ],\n    remainder='drop'\n)\n\nmodel = Pipeline([\n    ('preprocessor', preprocessor),\n    ('classifier', LogisticRegression(max_iter=1000))\n])\n\nfeature_ready_X = model.named_steps['preprocessor'].fit_transform(\n    df[numeric_features + categorical_features]\n)\nmodel.fit(df[numeric_features + categorical_features], df[TARGET_COLUMN])`,
                priority: 'medium',
                category: 'feature_engineering',
                focus_areas: ['feature_engineering', 'pipelines'],
                tags: ['column transformer', 'polynomial features', 'feature engineering']
            });
        }
    }
    
    
    /**
     * Analyze project ideas and strategic recommendations
     */
    function analyzeProjectIdeas(analysis, columnDetails, textSummary, metadata) {
        const columnNames = columnDetails.map(col => col.name.toLowerCase());
        const hasRating = columnNames.some(name => name.includes('rating') || name.includes('score'));
        const hasTextReviews = textSummary.free_text_columns > 0 && columnNames.some(name => name.includes('review') || name.includes('comment') || name.includes('text'));
        const hasUserData = columnNames.some(name => name.includes('user') || name.includes('customer'));
        const hasDateData = columnDetails.some(col => col.dtype.includes('datetime') || col.name.toLowerCase().includes('date'));
        const numericalCols = columnDetails.filter(col => col.data_category === 'numerical' || col.dtype.includes('int') || col.dtype.includes('float'));
        
        // Primary project recommendation based on data characteristics
        if (hasRating && hasTextReviews) {
            analysis.projectIdeas.push({
                title: 'Primary Project: Sentiment Analysis & Rating Prediction',
                description: 'Your dataset contains both ratings and text reviews, making it ideal for sentiment analysis and rating prediction projects.',
                action: `Use the text review columns to predict the rating column. This is a classic supervised learning problem that combines NLP with traditional ML.`,
                code: `# Sentiment analysis and rating prediction\nfrom textblob import TextBlob\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n\n# Extract sentiment from text\ndf['sentiment'] = df['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)\n\n# Create TF-IDF features\nvectorizer = TfidfVectorizer(max_features=1000)\ntext_features = vectorizer.fit_transform(df['review_text'])\n\n# Combine with other features for rating prediction\nX = np.hstack([text_features.toarray(), df[['sentiment', 'other_features']]])\ny = df['rating']\n\nmodel = RandomForestRegressor()\nmodel.fit(X, y)`,
                priority: 'strategic'
            });
        } else if (hasTextReviews) {
            analysis.projectIdeas.push({
                title: 'Primary Project: Text Classification & Topic Modeling',
                description: 'Rich text data detected. Consider text classification, topic modeling, or content analysis projects.',
                action: 'Use NLP techniques to classify text, extract topics, or perform content analysis.',
                code: `# Text classification & topic modeling starter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation

TEXT_COLUMN = 'review_text'  # replace with your text column name
TARGET_COLUMN = 'target_label'  # optional: replace with your target label column

# Ensure these columns exist before running this cell
if TEXT_COLUMN not in df.columns:
    raise KeyError(f"Column '{TEXT_COLUMN}' not found. Update TEXT_COLUMN with your text feature.")

# Optional supervised text classification (if you have labels)
if TARGET_COLUMN in df.columns:
    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN],
        df[TARGET_COLUMN],
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COLUMN]
    )

    text_classifier = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')),
        ('model', LogisticRegression(max_iter=1000))
    ])

    text_classifier.fit(X_train, y_train)
    print(f"Validation accuracy: {text_classifier.score(X_test, y_test):.2%}")

# Topic modelling to surface dominant themes
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df[TEXT_COLUMN])

lda = LatentDirichletAllocation(n_components=5, learning_method='online', random_state=42)
topic_mix = lda.fit_transform(tfidf_matrix)

feature_names = tfidf.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_terms = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    print(f"Topic {topic_idx + 1}: {', '.join(top_terms)}")`,
                priority: 'strategic'
            });
        } else if (numericalCols.length > 2 && hasRating) {
            analysis.projectIdeas.push({
                title: 'Primary Project: Predictive Modeling',
                description: 'Multiple numerical features with a potential target variable detected.',
                action: `Use ${numericalCols.slice(0, 3).map(col => col.name).join(', ')} to predict the rating/target variable.`,
                priority: 'strategic'
            });
        }
        
        // Secondary project ideas
        if (hasUserData && hasRating) {
            const possibleTarget = columnDetails.find(col => 
                col.name.toLowerCase().includes('rating') || 
                col.name.toLowerCase().includes('score') ||
                col.name.toLowerCase().includes('purchase') ||
                col.name.toLowerCase().includes('verified')
            );
            
            if (possibleTarget) {
                analysis.projectIdeas.push({
                    title: 'Secondary Project: Customer Behavior Analysis',
                    description: `Analyze customer behavior patterns using "${possibleTarget.name}" as the target variable.`,
                    action: `Explore relationships between user characteristics and ${possibleTarget.name}. Identify factors that influence customer satisfaction or behavior.`,
                    code: `# Customer behavior analysis\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\n# Analyze relationships\nsns.boxplot(data=df, x='user_category', y='${possibleTarget.name}')\nplt.title('${possibleTarget.name} by User Category')\nplt.show()\n\n# Statistical analysis\nfrom scipy.stats import chi2_contingency\ncontingency_table = pd.crosstab(df['user_category'], df['${possibleTarget.name}'])\nchi2, p_value, dof, expected = chi2_contingency(contingency_table)`,
                    priority: 'strategic'
                });
            }
        }
        
        // Time series analysis if date data exists
        if (hasDateData && (hasRating || numericalCols.length > 0)) {
            analysis.projectIdeas.push({
                title: 'Time Series Analysis Opportunity',
                description: 'Date columns detected along with measurable metrics.',
                action: 'Analyze trends over time, seasonal patterns, and forecast future values.',
                code: `# Time series analysis\nimport matplotlib.pyplot as plt\n\n# Parse dates and create time series\ndf['date'] = pd.to_datetime(df['date_column'])\ndf_ts = df.groupby('date')['metric'].mean().resample('D').mean()\n\n# Plot trend\ndf_ts.plot(figsize=(12, 6))\nplt.title('Metric Trend Over Time')\nplt.show()\n\n# Seasonal decomposition\nfrom statsmodels.tsa.seasonal import seasonal_decompose\ndecomposition = seasonal_decompose(df_ts, model='additive')\ndecomposition.plot()`,
                priority: 'strategic'
            });
        }
        
        // Clustering analysis for unsupervised learning
        if (numericalCols.length > 2) {
            analysis.projectIdeas.push({
                title: 'Exploratory Data Analysis & Clustering',
                description: `${numericalCols.length} numerical features available for pattern discovery.`,
                action: 'Perform clustering analysis to discover hidden patterns and customer segments.',
                code: `# Clustering analysis\nfrom sklearn.cluster import KMeans\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\n# Prepare data\nfeatures = ${JSON.stringify(numericalCols.slice(0, 5).map(col => col.name))}\nX = df[features].fillna(df[features].mean())\nX_scaled = StandardScaler().fit_transform(X)\n\n# K-means clustering\nkmeans = KMeans(n_clusters=3, random_state=42)\nclusters = kmeans.fit_predict(X_scaled)\ndf['cluster'] = clusters\n\n# Visualize with PCA\npca = PCA(n_components=2)\nX_pca = pca.fit_transform(X_scaled)\nplt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')\nplt.title('Customer Segments')\nplt.show()`,
                priority: 'strategic'
            });
        }
    }
    
    /**
     * Analyze advanced considerations and warnings
     */
    function analyzeAdvancedConsiderations(analysis, columnDetails, metadata, quality) {
        // High cardinality warnings (potential identifier columns)
        const veryHighCardinalityCols = columnDetails.filter(col => 
            col.unique_percentage > config.veryHighCardinalityThreshold && col.unique_count > 100
        );
        
        veryHighCardinalityCols.forEach(col => {
            analysis.advancedConsiderations.push({
                title: `Potential Identifier Column: ${col.name}`,
                description: `"${col.name}" has ${col.unique_count} unique values for ${Math.round(col.unique_count / col.unique_percentage * 100)} records (${col.unique_percentage.toFixed(1)}% unique).`,
                action: 'This column appears to be an identifier and should likely be **dropped** for machine learning to avoid overfitting. Keep it only for data joining or tracking purposes.',
                code: `# Drop identifier column\ndf_model = df.drop(['${col.name}'], axis=1)\n\n# Or keep for tracking\ndf_analysis = df.drop(['${col.name}'], axis=1)\ndf_results = df[['${col.name}']].copy()  # Keep IDs for result tracking`,
                priority: 'advanced'
            });
        });
        
        // NLP preparation for text columns
        const freeTextCols = columnDetails.filter(col => 
            col.data_category === 'text' && col.text_category === 'free_text'
        );
        
        if (freeTextCols.length > 0) {
            analysis.advancedConsiderations.push({
                title: 'NLP Pipeline Development',
                description: `Free text columns: ${freeTextCols.map(col => col.name).join(', ')}.`,
                action: 'Implement comprehensive text preprocessing pipeline including tokenization, stop word removal, and feature extraction.',
                code: `# NLP preprocessing pipeline\nimport nltk\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom nltk.corpus import stopwords\nfrom nltk.tokenize import word_tokenize\nfrom nltk.stem import PorterStemmer\n\nnltk.download(['punkt', 'stopwords'])\n\ndef preprocess_text(text):\n    # Convert to lowercase\n    text = text.lower()\n    # Remove special characters\n    text = re.sub(r'[^a-zA-Z\\s]', '', text)\n    # Tokenize\n    tokens = word_tokenize(text)\n    # Remove stopwords\n    stop_words = set(stopwords.words('english'))\n    tokens = [token for token in tokens if token not in stop_words]\n    # Stem\n    stemmer = PorterStemmer()\n    tokens = [stemmer.stem(token) for token in tokens]\n    return ' '.join(tokens)\n\n# Apply preprocessing\ndf['text_cleaned'] = df['${freeTextCols[0]?.name}'].apply(preprocess_text)\n\n# Create features\nvectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))\ntext_features = vectorizer.fit_transform(df['text_cleaned'])`,
                priority: 'advanced'
            });
        }
        
        // Memory optimization for large datasets
        if (metadata.memory_usage_bytes > config.largeDatasetThreshold) {
            analysis.advancedConsiderations.push({
                title: 'Memory Optimization Strategy',
                description: `Dataset uses ${formatFileSize(metadata.memory_usage_bytes)} of memory. Large datasets require optimization.`,
                action: 'Optimize data types, use chunked processing, and consider sampling for initial analysis.',
                code: `# Memory optimization\n# Optimize data types\nfor col in df.select_dtypes(include=['int64']).columns:\n    df[col] = pd.to_numeric(df[col], downcast='integer')\n\nfor col in df.select_dtypes(include=['float64']).columns:\n    df[col] = pd.to_numeric(df[col], downcast='float')\n\n# Use categories for strings\nfor col in df.select_dtypes(include=['object']).columns:\n    if df[col].nunique() < 0.5 * len(df):\n        df[col] = df[col].astype('category')\n\n# Chunked processing\nchunk_size = 10000\nfor chunk in pd.read_csv('data.csv', chunksize=chunk_size):\n    # Process chunk\n    result = process_chunk(chunk)\n    # Append to results`,
                priority: 'advanced'
            });
        }
        
        // Data leakage warnings
        const suspiciousColumns = columnDetails.filter(col => 
            col.name.toLowerCase().includes('id') && col.unique_percentage > 80
        );
        
        if (suspiciousColumns.length > 0) {
            analysis.advancedConsiderations.push({
                title: 'Data Leakage Prevention',
                description: `Potential leakage risk from columns: ${suspiciousColumns.map(col => col.name).join(', ')}.`,
                action: 'Verify these columns don\'t contain information that wouldn\'t be available at prediction time. Remove if they represent future information or unique identifiers.',
                priority: 'advanced'
            });
        }
    }
    
    function formatLabel(value) {
        if (!value) return '';
        return value
            .toString()
            .replace(/[_-]+/g, ' ')
            .replace(/\b\w/g, char => char.toUpperCase());
    }

    function toSignalKey(value) {
        if (!value) {
            return '';
        }

        const slug = value
            .toString()
            .trim()
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-');

        if (!slug) {
            return '';
        }

        if (SIGNAL_ALIAS_MAP[slug]) {
            return SIGNAL_ALIAS_MAP[slug];
        }

        if (KNOWN_SIGNAL_KEYS.has(slug)) {
            return slug;
        }

        // Try collapsing hyphens for cases like "skewness-score"
        const collapsed = slug.replace(/-/g, '');
        if (SIGNAL_ALIAS_MAP[collapsed]) {
            return SIGNAL_ALIAS_MAP[collapsed];
        }

        if (KNOWN_SIGNAL_KEYS.has(collapsed)) {
            return collapsed;
        }

        return '';
    }

    function collectSignals(rec) {
        if (!rec) {
            return [];
        }

        const candidates = [];

        const pushCandidate = value => {
            if (Array.isArray(value)) {
                value.forEach(pushCandidate);
                return;
            }
            if (value !== undefined && value !== null) {
                candidates.push(value);
            }
        };

        pushCandidate(rec.signal_type);
        pushCandidate(rec.meta?.signal_type);
        pushCandidate(rec.meta?.signals);
        pushCandidate(rec.focus_areas);
        pushCandidate(rec.meta?.focus_areas);
        pushCandidate(rec.tags);
        pushCandidate(rec.meta?.tags);

        const signals = new Set();

        candidates.forEach(candidate => {
            const key = toSignalKey(candidate);
            if (key) {
                signals.add(key);
            }
        });

        const combinedText = [
            rec.title,
            rec.description,
            rec.action,
            rec.why_it_matters,
            rec.feature_impact,
            Array.isArray(rec.focus_areas) ? rec.focus_areas.join(' ') : '',
            Array.isArray(rec.tags) ? rec.tags.join(' ') : '',
            typeof rec.meta === 'object' && rec.meta !== null ? JSON.stringify(rec.meta) : ''
        ].filter(Boolean).join(' ');

        if (combinedText) {
            TEXT_SIGNAL_PATTERNS.forEach(({ regex, signal }) => {
                if (regex.test(combinedText)) {
                    signals.add(signal);
                }
            });
        }

        return Array.from(signals);
    }

    function dedupePreserveOrder(list) {
        if (!Array.isArray(list) || list.length === 0) {
            return [];
        }

        const seen = new Set();
        const result = [];

        list.forEach(item => {
            if (item === null || item === undefined) {
                return;
            }
            const value = item.toString();
            const key = value.toLowerCase();
            if (!seen.has(key)) {
                seen.add(key);
                result.push(value);
            }
        });

        return result;
    }

    function collectAllRecommendations(analysis) {
        if (!analysis) {
            return [];
        }

        return [
            ...(analysis.highPriority || []),
            ...(analysis.dataPreparation || []),
            ...(analysis.projectIdeas || []),
            ...(analysis.advancedConsiderations || [])
        ];
    }

    function buildQuickFilterSummary(analysis) {
        const allRecommendations = collectAllRecommendations(analysis);
        const total = allRecommendations.length;

        const priorityCounts = allRecommendations.reduce((acc, rec) => {
            const priority = (rec.priority || 'medium').toLowerCase();
            acc[priority] = (acc[priority] || 0) + 1;
            return acc;
        }, {});

        const categories = CATEGORY_FILTER_CONFIG.map(configItem => ({
            ...configItem,
            count: Array.isArray(analysis?.[configItem.analysisKey]) ? analysis[configItem.analysisKey].length : 0
        })).filter(item => item.count > 0);

        const signalCounts = allRecommendations.reduce((acc, rec) => {
            const signals = collectSignals(rec);
            signals.forEach(signal => {
                acc[signal] = (acc[signal] || 0) + 1;
            });
            return acc;
        }, {});

        const signals = Object.entries(signalCounts)
            .map(([signal, count]) => ({
                signal,
                count,
                label: SIGNAL_FOCUS_LABELS[signal]?.label || formatLabel(signal),
                icon: SIGNAL_FOCUS_LABELS[signal]?.icon || 'fas fa-filter'
            }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 10);

        return {
            total,
            priorities: priorityCounts,
            categories,
            signals
        };
    }

    function renderQuickFilters(summary) {
        if (!summary) {
            return '';
        }

        const sections = [];

        const totalChip = renderFilterChip({
            label: 'All Insights',
            count: summary.total,
            icon: 'fas fa-layer-group',
            attributes: { 'data-reset': 'true' }
        });

        sections.push(renderQuickFilterGroup({
            title: 'Quick Picks',
            chipsMarkup: totalChip,
            groupKey: 'quick-picks',
            totalCount: summary.total,
            defaultExpanded: true,
            allowMultiple: true
        }));

        const priorityChips = Object.entries(summary.priorities || {})
            .filter(([, count]) => count > 0)
            .map(([priority, count]) => {
                const meta = PRIORITY_LABELS[priority] || { label: formatLabel(priority), icon: 'fas fa-adjust' };
                return renderFilterChip({
                    label: meta.label,
                    count,
                    icon: meta.icon,
                    attributes: { 'data-priority': priority }
                });
            }).join('');

        sections.push(renderQuickFilterGroup({
            title: 'Priority',
            chipsMarkup: priorityChips,
            groupKey: 'priority',
            totalCount: sumObjectValues(summary.priorities),
            defaultExpanded: false
        }));

        const categoryChips = (summary.categories || [])
            .map(({ key, label, icon, count }) => renderFilterChip({
                label,
                count,
                icon,
                attributes: { 'data-category': key }
            })).join('');

        sections.push(renderQuickFilterGroup({
            title: 'Recommendations',
            chipsMarkup: categoryChips,
            groupKey: 'categories',
            totalCount: sumArrayCounts(summary.categories),
            defaultExpanded: false
        }));

        const signalChips = (summary.signals || [])
            .map(({ signal, label, icon, count }) => renderFilterChip({
                label,
                count,
                icon,
                attributes: { 'data-focus': signal }
            })).join('');

        sections.push(renderQuickFilterGroup({
            title: 'Detected Data Patterns',
            chipsMarkup: signalChips,
            groupKey: 'signals',
            totalCount: sumArrayCounts(summary.signals),
            defaultExpanded: false
        }));

        const validSections = sections.filter(Boolean);

        return validSections.length > 0
            ? `<div class="recommendation-quick-filters" data-role="recommendation-quick-filters">${validSections.join('')}</div>`
            : '';
    }

    function renderFilterChip({ label, count, icon, attributes }) {
        if (!label) {
            return '';
        }

        const attrString = Object.entries(attributes || {})
            .map(([key, value]) => `${key}="${escapeAttribute(value)}"`)
            .join(' ');

        return `
            <button type="button" class="quick-filter-chip" ${attrString}>
                <span class="chip-icon"><i class="${icon}"></i></span>
                <span class="chip-label">${escapeHtml(label)}</span>
                ${count !== undefined ? `<span class="chip-count">${formatCount(count)}</span>` : ''}
            </button>
        `;
    }

    function renderQuickFilterGroup({ title, chipsMarkup, groupKey, totalCount, defaultExpanded = false, allowMultiple = false }) {
        if (!chipsMarkup || !chipsMarkup.trim()) {
            return '';
        }

        const safeKey = (groupKey || title || 'group').toString().toLowerCase().replace(/[^a-z0-9]+/g, '-');
        const expandedAttr = defaultExpanded ? 'true' : 'false';
        const allowAttr = allowMultiple ? 'true' : 'false';
        const direction = defaultExpanded ? 'up' : 'down';
        const panelId = `quick-filter-panel-${safeKey}`;
        const countMarkup = totalCount !== undefined && totalCount !== null
            ? `<span class="toggle-count">${formatCount(totalCount)}</span>`
            : '';
        const hiddenAttr = defaultExpanded ? '' : ' hidden';

        return `
            <div class="quick-filter-group" data-group-key="${escapeAttribute(safeKey)}" data-expanded="${expandedAttr}" data-allow-multiple="${allowAttr}">
                <button type="button" class="quick-filter-group-toggle" aria-expanded="${defaultExpanded ? 'true' : 'false'}" aria-controls="${panelId}">
                    <span class="toggle-label">${escapeHtml(title)}</span>
                    <span class="toggle-meta">
                        ${countMarkup}
                        <span class="toggle-icon" aria-hidden="true"><i class="fas fa-chevron-${direction}"></i></span>
                    </span>
                </button>
                <div class="quick-filter-subchips" id="${panelId}"${hiddenAttr}>
                    <div class="quick-filter-chips">
                        ${chipsMarkup}
                    </div>
                </div>
            </div>
        `;
    }

    function sumObjectValues(map) {
        if (!map) {
            return 0;
        }
        return Object.values(map).reduce((total, value) => total + (Number(value) || 0), 0);
    }

    function sumArrayCounts(items) {
        if (!Array.isArray(items)) {
            return 0;
        }
        return items.reduce((total, item) => total + (Number(item?.count) || 0), 0);
    }

    function formatCount(value) {
        if (value === undefined || value === null) {
            return '';
        }
        try {
            return Number(value).toLocaleString();
        } catch (error) {
            return value;
        }
    }

    function setupQuickFilterChips(root = getRecommendationsRoot()) {
        if (!root) {
            return;
        }

        root.querySelectorAll('.quick-filter-chip').forEach(chip => {
            if (chip.dataset.quickFilterListenerAttached) {
                return;
            }

            chip.addEventListener('click', () => handleQuickFilterClick(chip, root));
            chip.dataset.quickFilterListenerAttached = 'true';
        });

        setupQuickFilterGroupToggles(root);

        updateQuickFilterState(root);
    }

    function setupQuickFilterGroupToggles(root = getRecommendationsRoot()) {
        if (!root) {
            return;
        }

        root.querySelectorAll('.quick-filter-group-toggle').forEach(toggle => {
            if (toggle.dataset.quickGroupListenerAttached) {
                return;
            }

            toggle.addEventListener('click', () => {
                const group = toggle.closest('.quick-filter-group');
                if (!group) {
                    return;
                }

                const isExpanded = group.getAttribute('data-expanded') === 'true';
                const allowMultiple = group.dataset.allowMultiple === 'true';

                if (isExpanded) {
                    collapseQuickFilterGroup(group);
                } else {
                    if (!allowMultiple) {
                        collapseSiblingQuickFilterGroups(group);
                    }
                    expandQuickFilterGroup(group);
                }
            });

            toggle.dataset.quickGroupListenerAttached = 'true';
        });

        root.querySelectorAll('.quick-filter-group').forEach(group => {
            const isExpanded = group.getAttribute('data-expanded') === 'true';
            if (isExpanded) {
                expandQuickFilterGroup(group);
            } else {
                collapseQuickFilterGroup(group);
            }
        });
    }

    function collapseSiblingQuickFilterGroups(group) {
        if (!group) {
            return;
        }

        const container = group.parentElement;
        if (!container) {
            return;
        }

        container.querySelectorAll('.quick-filter-group').forEach(sibling => {
            if (sibling === group) {
                return;
            }
            if (sibling.dataset.allowMultiple === 'true') {
                return;
            }
            collapseQuickFilterGroup(sibling);
        });
    }

    function expandQuickFilterGroup(group) {
        if (!group) {
            return;
        }

        group.setAttribute('data-expanded', 'true');

        const toggle = group.querySelector('.quick-filter-group-toggle');
        if (toggle) {
            toggle.setAttribute('aria-expanded', 'true');
        }

        const panel = group.querySelector('.quick-filter-subchips');
        if (panel) {
            panel.removeAttribute('hidden');
        }

        updateQuickFilterToggleIcon(group, true);
    }

    function collapseQuickFilterGroup(group) {
        if (!group) {
            return;
        }

        group.setAttribute('data-expanded', 'false');

        const toggle = group.querySelector('.quick-filter-group-toggle');
        if (toggle) {
            toggle.setAttribute('aria-expanded', 'false');
        }

        const panel = group.querySelector('.quick-filter-subchips');
        if (panel) {
            panel.setAttribute('hidden', '');
        }

        updateQuickFilterToggleIcon(group, false);
    }

    function updateQuickFilterToggleIcon(group, isExpanded) {
        const icon = group?.querySelector('.toggle-icon i');
        if (!icon) {
            return;
        }

        icon.classList.toggle('fa-chevron-up', Boolean(isExpanded));
        icon.classList.toggle('fa-chevron-down', !isExpanded);
    }

    function handleQuickFilterClick(chip, root = getRecommendationsRoot()) {
        if (!root || !chip) {
            return;
        }

        const searchInput = root.querySelector('#search-filter');
        const wasActive = chip.classList.contains('active');

        if (chip.dataset.reset === 'true') {
            resetFilterState();
            if (searchInput && searchInput.value !== filterState.search) {
                searchInput.value = filterState.search;
            }
        } else {
            if (chip.dataset.priority) {
                filterState.priority = wasActive ? 'all' : chip.dataset.priority;
            }

            if (chip.dataset.category) {
                filterState.category = wasActive ? 'all' : chip.dataset.category;
            }

            if (chip.dataset.focus) {
                filterState.focus = wasActive ? 'all' : chip.dataset.focus;
            }

            if (!chip.dataset.priority && !chip.dataset.category && !chip.dataset.focus) {
                // fallback to reset if chip doesn't map to known filter
                resetFilterState();
            }

            if (!chip.dataset.searchPreserve) {
                filterState.search = '';
                if (searchInput) {
                    searchInput.value = '';
                }
            }
        }

        applyFilters(root);
    }

    function updateQuickFilterState(root = getRecommendationsRoot()) {
        if (!root) {
            return;
        }

        const { priorityValue, categoryValue, focusValue, searchValue } = getCurrentFilterValues();
        const normalizedSearch = (searchValue || '').trim();

        root.querySelectorAll('.quick-filter-chip').forEach(chip => {
            let matches = true;

            if (chip.dataset.reset === 'true') {
                matches = priorityValue === 'all' && categoryValue === 'all' && focusValue === 'all' && !normalizedSearch;
            } else {
                if (chip.dataset.priority) {
                    matches = matches && priorityValue === chip.dataset.priority;
                }
                if (chip.dataset.category) {
                    matches = matches && categoryValue === chip.dataset.category;
                }
                if (chip.dataset.focus) {
                    matches = matches && focusValue === chip.dataset.focus;
                }
            }

            chip.classList.toggle('active', matches);
        });
    }

    function getCurrentFilterValues() {
        return {
            priorityValue: filterState.priority || 'all',
            categoryValue: filterState.category || 'all',
            focusValue: filterState.focus || 'all',
            searchValue: filterState.search || ''
        };
    }

    function escapeAttribute(text) {
        if (!text) return '';
        return escapeHtml(text).replace(/"/g, '&quot;');
    }

    function renderWhyItMatters(text) {
        if (!text) return '';
        return `
            <div class="recommendation-why">
                <i class="fas fa-bullseye"></i>
                <strong>Why it matters:</strong> ${escapeHtml(text)}
            </div>
        `;
    }

    function renderFeatureImpact(text) {
        if (!text) return '';
        return `
            <div class="recommendation-impact">
                <i class="fas fa-sparkles"></i>
                <strong>Feature impact:</strong> ${escapeHtml(text)}
            </div>
        `;
    }

    function renderMetrics(metrics) {
        if (!Array.isArray(metrics) || metrics.length === 0) {
            return '';
        }

        const items = metrics.slice(0, 5).map(metric => {
            const name = escapeHtml(formatLabel(metric.name || ''));
            const unit = metric.unit ? ` ${escapeHtml(metric.unit)}` : '';
            const value = metric.value !== undefined && metric.value !== null
                ? `${escapeHtml(String(metric.value))}${unit}`
                : '';
            const description = metric.description ? `<span class="metric-description">${escapeHtml(metric.description)}</span>` : '';

            return `
                <div class="metric-item">
                    <span class="metric-name">${name}</span>
                    ${value ? `<span class="metric-value">${value}</span>` : ''}
                    ${description}
                </div>
            `;
        }).join('');

        return `
            <div class="recommendation-metrics">
                <i class="fas fa-chart-bar"></i>
                <div class="metrics-list">${items}</div>
            </div>
        `;
    }

    function renderTagPills(tags) {
        const uniqueTags = dedupePreserveOrder(tags);
        if (uniqueTags.length === 0) {
            return '';
        }

        const pills = uniqueTags.slice(0, 8).map(tag => `
            <span class="recommendation-tag">${escapeHtml(formatLabel(tag))}</span>
        `).join('');

        return `
            <div class="recommendation-tags">
                <i class="fas fa-tags"></i>
                ${pills}
            </div>
        `;
    }

    /**
     * Render a category of recommendations
     */
    function renderRecommendationCategory(title, recommendations, priority, iconClass) {
        if (!recommendations || recommendations.length === 0) return '';
        
        // Generate a unique category ID for filtering
        const categoryId = title.toLowerCase().replace(/[^a-z0-9]+/g, '-');
        
        let html = `
            <div class="recommendation-category ${priority}-priority" data-category="${categoryId}">
                <h5 class="category-header">
                    <i class="${iconClass}"></i> ${title}
                </h5>
                <div class="category-recommendations">
        `;
        
        recommendations.forEach((rec, index) => {
            const codeId = `code-${categoryId}-${index}`;
            const priority = (rec.priority || 'medium').toLowerCase();
            const priorityClass = `priority-${priority}`;
            const priorityLabel = rec.priority_label || formatLabel(priority);
            const categoryLabel = rec.category_label || formatLabel(rec.category || '');
            const rawTags = Array.isArray(rec.tags) ? rec.tags.filter(Boolean).map(tag => tag.toString()) : [];
            const rawFocusAreas = Array.isArray(rec.focus_areas) ? rec.focus_areas.filter(Boolean).map(area => area.toString()) : [];
            const uniqueTags = dedupePreserveOrder(rawTags);
            const uniqueFocusAreas = dedupePreserveOrder(rawFocusAreas);
            const tagAttr = escapeAttribute(uniqueTags.map(tag => tag.toLowerCase()).join(' '));
            const focusAttr = escapeAttribute(uniqueFocusAreas.map(area => area.toLowerCase()).join(' '));
            const title = escapeHtml(rec.title || `Recommendation ${index + 1}`);
            const description = escapeHtml(rec.description || '');
            const actionMarkup = rec.action ? `<div class="recommendation-action"><strong>Action:</strong> ${escapeHtml(rec.action)}</div>` : '';
            const columnsMarkup = Array.isArray(rec.columns) && rec.columns.length > 0
                ? `
                        <div class="recommendation-columns">
                            <i class="fas fa-table"></i>
                            <span>${rec.columns.map(col => escapeHtml(col)).join(', ')}</span>
                        </div>
                    `
                : '';
            const whyMarkup = renderWhyItMatters(rec.why_it_matters);
            const featureImpactMarkup = renderFeatureImpact(rec.feature_impact);
            const metricsMarkup = renderMetrics(rec.metrics);
            const tagsMarkup = renderTagPills(uniqueTags);
            const badgeHtml = buildRecommendationBadges({
                priorityClass,
                priorityLabel,
                categoryLabel,
                focusAreas: uniqueFocusAreas
            });

            html += `
                <div class="recommendation-item ${priority}-priority" data-priority="${priority}" data-tags="${tagAttr}" data-focus="${focusAttr}">
                    <div class="recommendation-header">
                        <h6 class="recommendation-title">${title}</h6>
                        <div class="recommendation-badges">${badgeHtml}</div>
                    </div>
                    <div class="recommendation-content">
                        <p class="recommendation-description">${description}</p>
                        ${columnsMarkup}
                        ${whyMarkup}
                        ${featureImpactMarkup}
                        ${actionMarkup}
                        ${metricsMarkup}
                        ${tagsMarkup}
                        ${rec.code ? `
                            <div class="recommendation-code">
                                <div class="code-header">
                                    <span>Code Example:</span>
                                    <button class="copy-code-btn" data-code-id="${codeId}">
                                        <i class="fas fa-copy"></i> Copy
                                    </button>
                                </div>
                                <pre id="${codeId}"><code>${escapeHtml(rec.code)}</code></pre>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        });
        
        html += '</div></div>';
        
        return html;
    }

    function buildRecommendationBadges({ priorityClass, priorityLabel, categoryLabel, focusAreas }) {
        const badgeFragments = [];
        const seenByType = new Set();
        const seenLabels = new Set();

        const pushBadge = (type, label, html, dedupeByLabel = false) => {
            if (!label || !html) {
                return;
            }
            const normalizedLabel = label.toString().trim().toLowerCase();
            if (!normalizedLabel) {
                return;
            }
            if (dedupeByLabel && seenLabels.has(normalizedLabel)) {
                return;
            }

            const key = `${type}:${normalizedLabel}`;
            if (seenByType.has(key)) {
                return;
            }

            seenByType.add(key);
            if (dedupeByLabel) {
                seenLabels.add(normalizedLabel);
            }
            badgeFragments.push(html);
        };

        pushBadge('priority', priorityLabel, `<span class="priority-badge ${priorityClass}">${escapeHtml(priorityLabel)}</span>`);

        if (categoryLabel) {
            pushBadge('category', categoryLabel, `<span class="category-badge">${escapeHtml(categoryLabel)}</span>`, true);
        }

        if (Array.isArray(focusAreas) && focusAreas.length > 0) {
            focusAreas.slice(0, 4).forEach(area => {
                if (!area && area !== 0) {
                    return;
                }
                const label = formatLabel(area);
                pushBadge('focus', label, `<span class="focus-badge">${escapeHtml(label)}</span>`, true);
            });
        }

        return badgeFragments.join('');
    }
    
    /**
     * Helper function to group columns by similarity in missing percentages and data types
     */
    function groupColumnsBySimilarity(columns) {
        const groups = [];
        const processed = new Set();
        
        columns.forEach(col => {
            if (processed.has(col.name)) return;
            
            const group = [col];
            processed.add(col.name);
            
            // Find similar columns (same data category and within 15% missing rate difference)
            columns.forEach(otherCol => {
                if (processed.has(otherCol.name)) return;
                
                const missingDiff = Math.abs(col.null_percentage - otherCol.null_percentage);
                const sameDataCategory = col.data_category === otherCol.data_category;
                
                if (sameDataCategory && missingDiff <= 15) {
                    group.push(otherCol);
                    processed.add(otherCol.name);
                }
            });
            
            groups.push(group);
        });
        
        return groups;
    }

    /**
     * Helper function to format file sizes
     */
    function formatFileSize(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    /**
     * Helper function to escape HTML
     */
    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    /**
     * Copy code to clipboard
     */
    function copyToClipboard(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            const text = element.textContent;
            navigator.clipboard.writeText(text).then(() => {
                // Show success feedback
                const button = document.querySelector(`button[data-code-id="${elementId}"]`);
                if (button) {
                    const originalText = button.innerHTML;
                    button.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    button.classList.add('copied');
                    
                    setTimeout(() => {
                        button.innerHTML = originalText;
                        button.classList.remove('copied');
                    }, 2000);
                }
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
        }
    }
    
    // Export functions
    recommendationsNamespace.renderRecommendations = renderRecommendations;

    function hydrateExistingRecommendations(containerId = 'recommendationsTabContent') {
        const container = document.getElementById(containerId);
        if (!container) {
            const qualityReport = global.DI && global.DI.previewPage && typeof global.DI.previewPage.getCurrentQualityReport === 'function'
                ? global.DI.previewPage.getCurrentQualityReport()
                : null;

            if (qualityReport) {
                queueRecommendationsRender(qualityReport, containerId);
            }
            return;
        }

        ensureRecommendationsWatchdog(containerId);

        const root = container.querySelector('[data-role="recommendations-root"]');
        if (root && root.getAttribute('data-hydrated') === 'true') {
            return;
        }

        const qualityReport = global.DI && global.DI.previewPage && typeof global.DI.previewPage.getCurrentQualityReport === 'function'
            ? global.DI.previewPage.getCurrentQualityReport()
            : null;

        if (qualityReport) {
            try {
                renderRecommendations(qualityReport, containerId);
            } catch (error) {
                console.warn('Failed to hydrate enhanced recommendations:', error);
            }
        }
    }

    function ensureRecommendationsWatchdog(containerId = 'recommendationsTabContent') {
        const container = document.getElementById(containerId);
        if (!container) {
            return;
        }

        const existingObserver = container.__recommendationsObserver;
        if (existingObserver) {
            return;
        }

        const observer = new MutationObserver(() => {
            const root = container.querySelector('[data-role="recommendations-root"]');
            if (root && root.getAttribute('data-hydrated') === 'true') {
                return;
            }

            observer.disconnect();
            delete container.__recommendationsObserver;

            requestAnimationFrame(() => {
                hydrateExistingRecommendations(containerId);
            });
        });

        observer.observe(container, { childList: true, subtree: true });
        container.__recommendationsObserver = observer;
    }

    function queueRecommendationsRender(qualityReport, containerId) {
        if (!qualityReport) {
            return;
        }

        const existing = pendingContainerMap.get(containerId);
        if (existing && existing.qualityReport === qualityReport) {
            return;
        }

        pendingContainerMap.set(containerId, {
            qualityReport,
            timestamp: Date.now(),
            attempts: 0
        });

        setTimeout(() => {
            if (!pendingContainerMap.has(containerId)) {
                return;
            }

            const target = document.getElementById(containerId);
            if (!target) {
                return;
            }

            pendingContainerMap.delete(containerId);

            requestAnimationFrame(() => {
                try {
                    renderRecommendations(qualityReport, containerId);
                } catch (error) {
                    console.warn('Timeout-based render failed, rescheduling:', error);
                    queueRecommendationsRender(qualityReport, containerId);
                }
            });
        }, 600);

        ensureContainerObserver();
    }

    function ensureContainerObserver() {
        if (recommendationsNamespace.__containerObserver) {
            return;
        }

        if (typeof MutationObserver === 'undefined') {
            return;
        }

        const observer = new MutationObserver(() => {
            if (!pendingContainerMap.size) {
                return;
            }

            pendingContainerMap.forEach((pending, containerId) => {
                const target = document.getElementById(containerId);
                if (!target) {
                    pending.attempts += 1;
                    if (pending.attempts > 40 && Date.now() - pending.timestamp > 15000) {
                        console.warn('Recommendations container still unavailable after multiple checks:', containerId);
                        pendingContainerMap.delete(containerId);
                    }
                    return;
                }

                pendingContainerMap.delete(containerId);

                requestAnimationFrame(() => {
                    try {
                        renderRecommendations(pending.qualityReport, containerId);
                    } catch (error) {
                        console.warn('Deferred render failed, rescheduling:', error);
                        queueRecommendationsRender(pending.qualityReport, containerId);
                    }
                });
            });

            if (!pendingContainerMap.size) {
                observer.disconnect();
                delete recommendationsNamespace.__containerObserver;
            }
        });

        observer.observe(document.body || document.documentElement, {
            childList: true,
            subtree: true
        });

        recommendationsNamespace.__containerObserver = observer;
    }

    recommendationsNamespace.ensureShell = ensureRecommendationsShell;
    recommendationsNamespace.getRecommendationsRoot = getRecommendationsRoot;
    recommendationsNamespace.hydrateExistingRecommendations = hydrateExistingRecommendations;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => hydrateExistingRecommendations());
    } else {
        setTimeout(() => hydrateExistingRecommendations(), 0);
    }
    
    // Export utility functions for global access
    global.copyToClipboard = copyToClipboard;
    global.clearSearch = clearSearch;
    
})(window);
