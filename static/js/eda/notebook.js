/**
 * NOTE: This file is auto-generated.
 * Edit the files in static/js/eda/notebook/modules/ and run
 *   python tools/build_notebook_bundle.py
 * to regenerate the bundled notebook script.
 */

/**
 * Notebook Interface JavaScript
 * EDA Notebook functionality for MLops2 project
 */

console.log('Notebook JavaScript loading...');

// Global variables
let cellCounter = 1;
let executionCounter = 1;
let sourceId = null;
let selectedAnalysisTypes = new Set(); // Changed from single to multiple selections
let dataQualityCompleted = false;
let preprocessingApplied = false;
let preprocessingDirty = true;
let lastPreprocessingReport = null;
const PREPROCESSING_STORAGE_KEY = 'eda-preprocessing-config';
const PREPROCESSING_APPLIED_STORAGE_KEY = 'eda-preprocessing-applied-state';
let preprocessingConfig = null;
let preprocessingEventListenersAttached = false;
let preprocessingControlsBound = false;
let preprocessingInitialized = false;

// Enhanced Analysis Categories Configuration
const STATIC_ANALYSIS_CATEGORIES = {
    'data_quality': {
        title: 'Data Quality',
        subtitle: 'Validate and assess data integrity',
        options: [
            { value: 'dataset_shape_analysis', name: 'Dataset Shape Analysis' },
            { value: 'data_range_validation', name: 'Data Range Validation' },
            { value: 'duplicate_detection', name: 'Duplicate Detection' },
            { value: 'missing_value_analysis', name: 'Missing Value Analysis' },
            { value: 'data_types_validation', name: 'Data Types Validation' }
        ]
    },
    'univariate': {
        title: 'Univariate Analysis',
        subtitle: 'Single variable statistical analysis',
        options: [
            { value: 'summary_statistics', name: 'Summary Statistics' },
            { value: 'numeric_frequency_analysis', name: 'Numeric Frequency Analysis' },
            { 
                value: 'skewness_analysis', 
                name: 'Skewness Analysis',
                subcategory: true,
                suboptions: [
                    { value: 'skewness_statistics', name: 'Skewness Statistics' },
                    { value: 'skewness_visualization', name: 'Skewness Visualization' }
                ]
            },
            { value: 'normality_test', name: 'Normality Test' },
            { 
                value: 'distribution_plots', 
                name: 'Distribution Plots',
                subcategory: true,
                suboptions: [
                    { value: 'histogram_plots', name: 'Histogram Plots' },
                    { value: 'box_plots', name: 'Box Plots' },
                    { value: 'violin_plots', name: 'Violin Plots' },
                    { value: 'kde_plots', name: 'KDE (Density) Plots' }
                ]
            }
        ]
    },
    'categorical': {
        title: 'Categorical Analysis',
        subtitle: 'Analyze categorical variables',
        options: [
            { value: 'categorical_frequency_analysis', name: 'Frequency Analysis' },
            { value: 'categorical_cardinality_profile', name: 'Cardinality Profile' },
            { value: 'rare_category_detection', name: 'Rare Category Scan' },
            { 
                value: 'categorical_visualization', 
                name: 'Categorical Visualization',
                subcategory: true,
                suboptions: [
                    { value: 'categorical_bar_charts', name: 'Bar Charts' },
                    { value: 'categorical_pie_charts', name: 'Pie Charts' }
                ]
            }
        ]
    },
    'bivariate': {
        title: 'Bivariate Analysis',
        subtitle: 'Two-variable relationships',
        options: [
            { 
                value: 'correlation_analysis', 
                name: 'Correlation Analysis',
                subcategory: true,
                suboptions: [
                    { value: 'pearson_correlation', name: 'Pearson Correlation' },
                    { value: 'spearman_correlation', name: 'Spearman Correlation' }
                ]
            },
            { value: 'scatter_plot_analysis', name: 'Scatter Plot Analysis' },
            { value: 'cross_tabulation_analysis', name: 'Cross Tabulation Analysis' },
            { value: 'categorical_numeric_relationships', name: 'Categorical vs Numeric Explorer' }
        ]
    },
    'outliers': {
        title: 'Outlier Detection',
        subtitle: 'Identify anomalies and outliers',
        options: [
            { value: 'iqr_outlier_detection', name: 'IQR Outlier Detection' },
            { value: 'zscore_outlier_detection', name: 'Z-Score Outlier Detection' },
            { 
                value: 'visual_outlier_inspection', 
                name: 'Visual Inspection',
                subcategory: true,
                suboptions: [
                    { value: 'visual_outlier_inspection', name: 'Complete Visual Inspection' },
                    { value: 'outlier_distribution_visualization', name: 'Distribution Plots' },
                    { value: 'outlier_scatter_matrix', name: 'Scatter Matrix' }
                ]
            }
        ]
    },
    'time_series': {
        title: 'Time Series',
        subtitle: 'Temporal data analysis',
        options: [
            { value: 'temporal_trend_analysis', name: 'Temporal Trends' },
            { value: 'seasonality_detection', name: 'Seasonality Detection' },
            { value: 'datetime_feature_extraction', name: 'Datetime Feature Extraction' }
        ]
    },
    'geospatial': {
        title: 'Geospatial Analysis',
        subtitle: 'Spatial coverage and proximity insights',
        options: [
            { value: 'coordinate_system_projection_check', name: 'Coordinate System & Projection Check' },
            { value: 'spatial_data_quality_analysis', name: 'Spatial Data Quality Analysis' },
            { value: 'spatial_distribution_analysis', name: 'Spatial Distribution Analysis' },
            { value: 'spatial_relationships_analysis', name: 'Spatial Relationships Analysis' },
            { value: 'geospatial_proximity_analysis', name: 'Geospatial Proximity Analysis' }
        ]
    },
    'text': {
        title: 'Text Analysis',
        subtitle: 'Tokenisation, vocabulary and lexical health',
        options: [
            { value: 'text_length_distribution', name: 'Text Length Distribution' },
            { value: 'text_token_frequency', name: 'Text Token Frequency' },
            { value: 'text_vocabulary_summary', name: 'Text Vocabulary Summary' },
            { value: 'text_feature_engineering_profile', name: 'Text Feature Engineering Profile' },
            { value: 'text_nlp_profile', name: 'Advanced NLP Profile' }
        ]
    },
    'relationships': {
        title: 'Relationships',
        subtitle: 'Multi-variable relationships',
        options: [
            { value: 'multicollinearity_analysis', name: 'Multicollinearity Analysis' },
            { 
                value: 'pca_analysis', 
                name: 'PCA Analysis',
                subcategory: true,
                suboptions: [
                    { value: 'pca_dimensionality_reduction', name: 'Complete PCA Analysis' },
                    { value: 'pca_scree_plot', name: 'Scree Plot' },
                    { value: 'pca_cumulative_variance', name: 'Cumulative Variance' },
                    { value: 'pca_visualization', name: 'PCA Visualization' },
                    { value: 'pca_biplot', name: 'PCA Biplot' },
                    { value: 'pca_heatmap', name: 'PCA Loadings Heatmap' }
                ]
            },
            { 
                value: 'clustering_analysis', 
                name: 'Clustering Analysis',
                subcategory: true,
                suboptions: [
                    { value: 'cluster_tendency_analysis', name: 'Cluster Tendency Assessment' },
                    { value: 'cluster_segmentation_analysis', name: 'Cluster Segmentation' }
                ]
            },
            { value: 'network_analysis', name: 'Network Analysis' },
            { value: 'entity_relationship_network', name: 'Entity Network Analysis' }
        ]
    },
    'target': {
        title: 'Target & Outcome',
        subtitle: 'Profile modelling targets and readiness',
        options: [
            { value: 'target_variable_analysis', name: 'Target Variable Analysis' }
        ]
    },
    'marketing': {
        title: 'Marketing Analytics',
        subtitle: 'Marketing & social media data analysis',
        options: [
            { value: 'campaign_metrics_analysis', name: 'Campaign Metrics Analysis' },
            { value: 'conversion_funnel_analysis', name: 'Conversion Funnel Analysis' },
            { value: 'engagement_analysis', name: 'Engagement Analysis' },
            { value: 'channel_performance_analysis', name: 'Channel Performance Analysis' },
            { value: 'audience_segmentation_analysis', name: 'Audience Segmentation Analysis' },
            { value: 'roi_analysis', name: 'ROI Analysis' },
            { value: 'attribution_analysis', name: 'Attribution Analysis' },
            { value: 'cohort_analysis', name: 'Cohort Analysis' }
        ]
    }
};

const CATEGORY_DISPLAY_OVERRIDES = {
    'Data Quality & Structure': {
        key: 'data_quality',
        title: 'Data Quality',
        subtitle: 'Validate and assess data integrity',
        icon: 'bi-shield-check',
        order: 10,
    },
    'Univariate Analysis (Numeric)': {
        key: 'univariate',
        title: 'Univariate Analysis',
        subtitle: 'Single variable statistical analysis',
        icon: 'bi-graph-up',
        order: 20,
    },
    'Univariate Analysis (Categorical)': {
        key: 'categorical',
        title: 'Categorical Analysis',
        subtitle: 'Analyze categorical variables',
        icon: 'bi-pie-chart-fill',
        order: 30,
    },
    'Bivariate/Multivariate Analysis': {
        key: 'bivariate',
        title: 'Bivariate Analysis',
        subtitle: 'Two-variable relationships',
        icon: 'bi-diagram-2',
        order: 40,
    },
    'Outlier & Anomaly Detection': {
        key: 'outliers',
        title: 'Outlier Detection',
        subtitle: 'Identify anomalies and outliers',
        icon: 'bi-exclamation-triangle',
        order: 50,
    },
    'Time-Series Exploration': {
        key: 'time_series',
        title: 'Time Series',
        subtitle: 'Temporal data analysis',
        icon: 'bi-clock-history',
        order: 60,
    },
    'Geospatial Analysis': {
        key: 'geospatial',
        title: 'Geospatial Analysis',
        subtitle: 'Map coverage and spatial relationships',
        icon: 'bi-geo-alt',
        order: 65,
    },
    'Text Analysis': {
        key: 'text',
        title: 'Text Analysis',
        subtitle: 'Tokenisation, vocabulary and lexical health',
        icon: 'bi-type',
        order: 70,
    },
    'Relationship Exploration': {
        key: 'relationships',
        title: 'Relationship Exploration',
        subtitle: 'Multi-variable relationships',
        icon: 'bi-diagram-3',
        order: 80,
    },
    'Target & Outcome Analysis': {
        key: 'target',
        title: 'Target & Outcome',
        subtitle: 'Profile modelling targets and readiness',
        icon: 'bi-bullseye',
        order: 90,
    },
    'Marketing Analysis': {
        key: 'marketing',
        title: 'Marketing Analytics',
        subtitle: 'Marketing & social media data analysis',
        icon: 'bi-megaphone',
        order: 100,
    },
};

const LEGACY_CATEGORY_TO_BACKEND = {
    data_quality: 'Data Quality & Structure',
    univariate: 'Univariate Analysis (Numeric)',
    categorical: 'Univariate Analysis (Categorical)',
    bivariate: 'Bivariate/Multivariate Analysis',
    outliers: 'Outlier & Anomaly Detection',
    time_series: 'Time-Series Exploration',
    geospatial: 'Geospatial Analysis',
    text: 'Text Analysis',
    relationships: 'Relationship Exploration',
    target: 'Target & Outcome Analysis',
    marketing: 'Marketing Analysis',
};

let ANALYSIS_CATEGORIES = {};
let ANALYSIS_CATEGORY_LOOKUP = {};
let analysisCatalogueLoaded = false;
let analysisCatalogueLoadingPromise = null;

const DATA_QUALITY_CATEGORY_KEY = 'data_quality';
const PREPROCESSING_MODAL_ID = 'preprocessingConfigModal';

if (!preprocessingConfig) {
    preprocessingConfig = loadStoredPreprocessingConfig() || getDefaultPreprocessingConfig();
}

let selectedCategories = new Set();
let selectedSubOptions = new Set();
let isGridView = true;
let currentCategoricalAnalysisType = '';
let currentCategoricalCellId = '';
let categoricalModalConfirmed = false;
let categoricalModalSelection = new Set();
let categoricalModalColumns = [];
let categoricalModalRecommendedDefaults = [];
let categoricalModalSearchTerm = '';
let currentNumericAnalysisType = '';
let currentNumericCellId = '';
let numericModalConfirmed = false;
let numericModalSelection = new Set();
let numericModalColumns = [];
let numericModalRecommendedDefaults = [];
let numericModalSearchTerm = '';
let categoricalModalIsRerun = false;
let numericModalIsRerun = false;
let currentCategoricalNumericAnalysisType = '';
let currentCategoricalNumericCellId = '';
let categoricalNumericModalConfirmed = false;
let categoricalNumericSelectedCategorical = new Set();
let categoricalNumericSelectedNumeric = new Set();
let categoricalNumericModalCategoricalColumns = [];
let categoricalNumericModalNumericColumns = [];
let categoricalNumericModalSearchTerm = { categorical: '', numeric: '' };
let categoricalNumericRecommendedPairs = [];
let categoricalNumericActivePairs = new Set();
let categoricalNumericModalIsRerun = false;
let currentCrossTabAnalysisType = '';
let currentCrossTabCellId = '';
let crossTabModalConfirmed = false;
let crossTabModalSelection = [];
let crossTabModalColumns = [];
let crossTabModalRecommendedDefaults = [];
let crossTabModalSearchTerm = '';
let crossTabModalIsRerun = false;
let currentTimeSeriesAnalysisType = '';
let currentTimeSeriesCellId = '';
let timeSeriesModalConfirmed = false;
let timeSeriesModalSelectedDates = new Set();
let timeSeriesModalNumericSelection = new Set();
let timeSeriesModalDateColumns = [];
let timeSeriesModalNumericColumns = [];
let timeSeriesModalDateRecommendedDefaults = [];
let timeSeriesModalNumericRecommendedDefaults = [];
let timeSeriesModalDateSearchTerm = '';
let timeSeriesModalNumericSearchTerm = '';
let timeSeriesModalRequiresNumeric = false;
let timeSeriesModalAllowsMultipleDates = false;
let timeSeriesModalIsRerun = false;
let currentTextAnalysisType = '';
let currentTextCellId = '';
let textModalConfirmed = false;
let textModalSelection = new Set();
let textModalColumns = [];
let textModalRecommendedDefaults = [];
let textModalSearchTerm = '';
let textModalIsRerun = false;
let currentTargetAnalysisType = '';
let currentTargetCellId = '';
let targetModalConfirmed = false;
let targetModalIsRerun = false;
let targetModalSelection = new Set();
let targetModalColumns = [];
let targetModalRecommendedDefaults = [];
let targetModalSearchTerm = '';
const rerunModalState = {
    cellId: '',
    analysisType: '',
    modalType: '',
    previousSelection: [],
    previousSelectionDetails: null
};
let analysisCodeModalCurrentCode = '';
let analysisCodeModalCurrentAnalysis = '';

// Metadata cache for granular components to avoid duplicate API requests
const analysisMetadataCache = new Map();

function slugifyCategoryKey(name) {
    return (name || '')
        .toString()
        .trim()
        .toLowerCase()
        .replace(/&/g, 'and')
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '') || 'general';
}

function resolveCategoryOverride(categoryKey, category = {}) {
    if (category.backendCategory && CATEGORY_DISPLAY_OVERRIDES[category.backendCategory]) {
        return CATEGORY_DISPLAY_OVERRIDES[category.backendCategory];
    }

    const legacyKey = LEGACY_CATEGORY_TO_BACKEND[categoryKey];
    if (legacyKey && CATEGORY_DISPLAY_OVERRIDES[legacyKey]) {
        return CATEGORY_DISPLAY_OVERRIDES[legacyKey];
    }

    if (category.title && CATEGORY_DISPLAY_OVERRIDES[category.title]) {
        return CATEGORY_DISPLAY_OVERRIDES[category.title];
    }

    return null;
}

function applyCatalogueDisplayMetadata(categories) {
    if (!categories || typeof categories !== 'object') {
        return {};
    }

    Object.entries(categories).forEach(([key, category]) => {
        if (!category || typeof category !== 'object') {
            categories[key] = {
                title: key.replace(/_/g, ' '),
                subtitle: '',
                options: [],
            };
            category = categories[key];
        }

        if (!Array.isArray(category.options)) {
            category.options = [];
        }

        const override = resolveCategoryOverride(key, category);

        if (override) {
            category.title = override.title || category.title || override.key || key;
            category.subtitle = override.subtitle || category.subtitle || '';
            category.icon = override.icon || category.icon || null;
            category.order = typeof override.order === 'number' ? override.order : category.order;
            category.backendCategory = category.backendCategory || override.backendCategory || override.title || category.title;
        } else {
            category.title = category.title || key.replace(/_/g, ' ');
            category.icon = category.icon || null;
            category.order = typeof category.order === 'number' ? category.order : 999;
            category.backendCategory = category.backendCategory || category.title;
        }

        category.count = category.options.length;
        if (!category.subtitle) {
            const countLabel = category.count === 1 ? 'analysis' : 'analyses';
            category.subtitle = `${category.count} ${countLabel}`;
        }

        category.options = category.options
            .filter(option => option && option.value)
            .map(option => ({
                ...option,
                name: option.name || option.value,
                description: option.description || '',
                tags: Array.isArray(option.tags) ? option.tags : [],
                complexity: option.complexity || 'intermediate',
                estimated_runtime: option.estimated_runtime || '1-5 seconds',
                icon: option.icon || category.icon || '📊',
            }));
    });

    return categories;
}

function mergeAnalysisOptionDetails(baseOption, staticOption) {
    const merged = {
        ...(baseOption || {}),
        ...(staticOption || {}),
    };

    merged.value = (staticOption && staticOption.value) || (baseOption && baseOption.value) || merged.value;
    merged.name = (baseOption && baseOption.name) || (staticOption && staticOption.name) || merged.value;
    merged.description = (baseOption && baseOption.description) || (staticOption && staticOption.description) || '';
    merged.tags = Array.isArray(baseOption?.tags)
        ? baseOption.tags
        : Array.isArray(staticOption?.tags)
            ? staticOption.tags
            : [];
    merged.estimated_runtime =
        (baseOption && baseOption.estimated_runtime) ||
        (staticOption && staticOption.estimated_runtime) ||
        '1-5 seconds';
    merged.icon = (baseOption && baseOption.icon) || (staticOption && staticOption.icon) || merged.icon || null;

    return merged;
}

function applyStaticHierarchyOverrides(categories) {
    if (!categories || typeof categories !== 'object') {
        return categories;
    }

    Object.entries(STATIC_ANALYSIS_CATEGORIES).forEach(([staticKey, staticCategory]) => {
        if (!staticCategory || !Array.isArray(staticCategory.options)) {
            return;
        }

        const potentialKeys = new Set([staticKey]);

        const overrideByTitle = CATEGORY_DISPLAY_OVERRIDES[staticCategory.title];
        if (overrideByTitle?.key) {
            potentialKeys.add(overrideByTitle.key);
        }

        const legacyBackend = LEGACY_CATEGORY_TO_BACKEND[staticKey];
        if (legacyBackend && CATEGORY_DISPLAY_OVERRIDES[legacyBackend]?.key) {
            potentialKeys.add(CATEGORY_DISPLAY_OVERRIDES[legacyBackend].key);
        }

        potentialKeys.add(slugifyCategoryKey(staticCategory.title));
        potentialKeys.add(slugifyCategoryKey(staticKey));

        let targetKey = null;
        for (const key of potentialKeys) {
            if (key && categories[key]) {
                targetKey = key;
                break;
            }
        }

        if (!targetKey) {
            return;
        }

        const targetCategory = categories[targetKey];
        if (!targetCategory || !Array.isArray(targetCategory.options)) {
            return;
        }

        const runtimeOptionMap = new Map();
        targetCategory.options.forEach(option => {
            if (option && option.value) {
                runtimeOptionMap.set(option.value, { ...option });
            }
        });

        const newOptions = [];

        staticCategory.options.forEach(staticOption => {
            if (!staticOption || !staticOption.value) {
                return;
            }

            const baseOption = runtimeOptionMap.get(staticOption.value);
            const baseSubMap = new Map(
                Array.isArray(baseOption?.suboptions)
                    ? baseOption.suboptions
                          .filter(sub => sub && sub.value)
                          .map(sub => [sub.value, { ...sub }])
                    : []
            );

            const mergedParent = mergeAnalysisOptionDetails(baseOption, staticOption);
            runtimeOptionMap.delete(staticOption.value);

            if (staticOption.subcategory && Array.isArray(staticOption.suboptions)) {
                mergedParent.subcategory = true;
                const mergedSubOptions = [];

                staticOption.suboptions.forEach(staticSubOption => {
                    if (!staticSubOption || !staticSubOption.value) {
                        return;
                    }

                    const baseSubOption = baseSubMap.get(staticSubOption.value) || runtimeOptionMap.get(staticSubOption.value);
                    const mergedSub = mergeAnalysisOptionDetails(baseSubOption, staticSubOption);
                    mergedSubOptions.push(mergedSub);
                    runtimeOptionMap.delete(staticSubOption.value);
                });

                mergedParent.suboptions = mergedSubOptions;
            } else if ('suboptions' in mergedParent) {
                delete mergedParent.suboptions;
            }

            newOptions.push(mergedParent);
        });

        runtimeOptionMap.forEach(option => {
            newOptions.push(option);
        });

        targetCategory.options = newOptions;
        targetCategory.count = newOptions.length;
        if (!targetCategory.subtitle) {
            const countLabel = targetCategory.count === 1 ? 'analysis' : 'analyses';
            targetCategory.subtitle = `${targetCategory.count} ${countLabel}`;
        }
    });

    return categories;
}

function primeAnalysisMetadataCacheFromCatalogue(categories) {
    if (!categories || typeof categories !== 'object') {
        return;
    }

    Object.values(categories).forEach(category => {
        if (!category || !Array.isArray(category.options)) {
            return;
        }

        category.options.forEach(option => {
            if (!option || !option.value || analysisMetadataCache.has(option.value)) {
                return;
            }

            analysisMetadataCache.set(option.value, {
                id: option.value,
                name: option.name || option.value,
                description: option.description || '',
                complexity: option.complexity || 'intermediate',
                estimated_runtime: option.estimated_runtime || '1-5 seconds',
                tags: Array.isArray(option.tags) ? option.tags : [],
                icon: option.icon || category.icon || '📊',
                category: category.backendCategory || category.title || 'General',
            });
        });
    });
}

function normalizeCatalogueResponse(components) {
    if (!components || typeof components !== 'object') {
        return {};
    }

    const normalized = {};

    Object.entries(components).forEach(([backendCategory, items]) => {
        if (!Array.isArray(items) || items.length === 0) {
            return;
        }

        const override = CATEGORY_DISPLAY_OVERRIDES[backendCategory] || {};
        const key = override.key || slugifyCategoryKey(backendCategory);

        if (!normalized[key]) {
            normalized[key] = {
                title: override.title || backendCategory,
                subtitle: override.subtitle || '',
                icon: override.icon || null,
                order: typeof override.order === 'number' ? override.order : 999,
                backendCategory,
                options: [],
            };
        }

        const target = normalized[key];
        target.backendCategory = backendCategory;
        target.icon = override.icon || target.icon || null;
        target.order = typeof override.order === 'number' ? override.order : target.order;

        items.forEach(item => {
            if (!item || !item.id) {
                return;
            }

            const exists = target.options.some(existing => existing.value === item.id);
            if (exists) {
                return;
            }

            target.options.push({
                value: item.id,
                name: item.name || item.id,
                description: item.description || '',
                tags: Array.isArray(item.tags) ? item.tags : [],
                complexity: item.complexity || 'intermediate',
                estimated_runtime: item.estimated_runtime || '1-5 seconds',
                icon: item.icon || override.icon || '📊',
            });
        });
    });

    const decorated = applyCatalogueDisplayMetadata(normalized);
    return applyStaticHierarchyOverrides(decorated);
}

function initializeAnalysisCatalogueDefaults() {
    const baseCategories = applyCatalogueDisplayMetadata(JSON.parse(JSON.stringify(STATIC_ANALYSIS_CATEGORIES)));
    ANALYSIS_CATEGORIES = applyStaticHierarchyOverrides(baseCategories);
    ANALYSIS_CATEGORY_LOOKUP = buildAnalysisCategoryLookup(ANALYSIS_CATEGORIES);
    primeAnalysisMetadataCacheFromCatalogue(ANALYSIS_CATEGORIES);
}

function useStaticAnalysisCatalogue() {
    initializeAnalysisCatalogueDefaults();
    analysisCatalogueLoaded = true;
    return ANALYSIS_CATEGORIES;
}

async function loadAnalysisCatalogueFromApi() {
    const response = await fetch('/advanced-eda/components/available', {
        method: 'GET',
        credentials: 'same-origin',
        headers: {
            Accept: 'application/json',
        },
    });

    if (!response.ok) {
        throw new Error(`Catalogue request failed with status ${response.status}`);
    }

    const payload = await response.json();
    if (!payload || payload.success === false || !payload.components) {
        throw new Error('Catalogue response missing components');
    }

    const normalized = normalizeCatalogueResponse(payload.components);
    if (!normalized || Object.keys(normalized).length === 0) {
        throw new Error('Catalogue response did not include any categories');
    }

    return normalized;
}

async function ensureAnalysisCatalogueLoaded({ forceReload = false } = {}) {
    if (analysisCatalogueLoaded && !forceReload) {
        return ANALYSIS_CATEGORIES;
    }

    if (analysisCatalogueLoadingPromise) {
        return analysisCatalogueLoadingPromise;
    }

    analysisCatalogueLoadingPromise = (async () => {
        try {
            const categories = await loadAnalysisCatalogueFromApi();
            ANALYSIS_CATEGORIES = categories;
            ANALYSIS_CATEGORY_LOOKUP = buildAnalysisCategoryLookup(ANALYSIS_CATEGORIES);
            primeAnalysisMetadataCacheFromCatalogue(ANALYSIS_CATEGORIES);
            analysisCatalogueLoaded = true;

            document.dispatchEvent(new CustomEvent('analysis-catalogue:updated', {
                detail: { categories: ANALYSIS_CATEGORIES },
            }));

            return ANALYSIS_CATEGORIES;
        } catch (error) {
            console.warn('Falling back to static analysis catalogue:', error);
            const fallback = useStaticAnalysisCatalogue();
            if (typeof showNotification === 'function') {
                showNotification('Live analysis catalogue is unavailable. Using cached defaults.', 'warning');
            }
            return fallback;
        } finally {
            analysisCatalogueLoadingPromise = null;
        }
    })();

    return analysisCatalogueLoadingPromise;
}

window.ensureAnalysisCatalogueLoaded = ensureAnalysisCatalogueLoaded;

initializeAnalysisCatalogueDefaults();

// Helper constants for run status styling
const RUN_STATUS_CLASSES = ['status-queued', 'status-running', 'status-success', 'status-error'];

function updateRunIndicator(badgeElement, status = 'queued', text = 'Queued') {
    if (!badgeElement) {
        return;
    }

    RUN_STATUS_CLASSES.forEach(cls => badgeElement.classList.remove(cls));
    badgeElement.classList.add(`status-${status}`);
    badgeElement.textContent = text;
    badgeElement.setAttribute('data-status', status);
}

function updateAnalysisMeta(cellId, message) {
    const metaElement = document.getElementById(`meta-${cellId}`);
    if (metaElement) {
        metaElement.textContent = message;
    }
}

function clearAnalysisAlerts(cellId) {
    const alertsContainer = document.getElementById(`alerts-${cellId}`);
    if (alertsContainer) {
        alertsContainer.innerHTML = '';
    }
}

function pushAnalysisAlert(cellId, variant, message) {
    const alertsContainer = document.getElementById(`alerts-${cellId}`);
    if (!alertsContainer) {
        return;
    }

    const variantClass = variant === 'error' ? 'alert-danger' : variant === 'warning' ? 'alert-warning' : 'alert-info';
    alertsContainer.innerHTML += `
        <div class="analysis-alert alert ${variantClass}" role="alert">
            ${message}
        </div>
    `;
}

function formatAnalysisCategory(category) {
    if (!category) {
        return 'General';
    }
    return category
        .split('_')
        .map(part => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
}

function getDefaultPreprocessingConfig() {
    return {
        dropMissingColumns: { enabled: false, threshold: 50 },
        dropMissingRows: { enabled: false, threshold: 60 },
        manualDropColumns: { enabled: false, columns: [] },
        dropDuplicates: false,
        imputation: { strategy: 'none', fillValue: '', neighbors: 5 },
    };
}

function clonePreprocessingConfig(config) {
    try {
        return JSON.parse(JSON.stringify(config || {}));
    } catch (error) {
        console.warn('Unable to deep clone preprocessing config:', error);
        return getDefaultPreprocessingConfig();
    }
}

function normalizeThreshold(value, fallback) {
    const parsed = parseFloat(value);
    if (Number.isFinite(parsed)) {
        return Math.min(100, Math.max(0, parsed));
    }
    return fallback ?? null;
}

function normalizeInteger(value, fallback, min = 1, max = 50) {
    const parsed = parseInt(value, 10);
    if (Number.isFinite(parsed)) {
        return Math.min(max, Math.max(min, parsed));
    }
    return fallback ?? min;
}

function normalizeColumnList(value) {
    if (!value) {
        return [];
    }

    let raw = value;
    if (typeof raw === 'string') {
        raw = [raw];
    } else if (Array.isArray(raw)) {
        raw = [...raw];
    } else if (typeof raw === 'object') {
        if (Array.isArray(raw.columns)) {
            raw = [...raw.columns];
        } else if (Array.isArray(raw.values)) {
            raw = [...raw.values];
        } else if (Array.isArray(raw.list)) {
            raw = [...raw.list];
        } else {
            raw = Object.values(raw)
                .flat()
                .filter(item => typeof item === 'string');
        }
    } else {
        return [];
    }

    const seen = new Set();
    return raw
        .map(item => (item ?? '').toString().trim())
        .filter(name => {
            if (!name || seen.has(name)) {
                return false;
            }
            seen.add(name);
            return true;
        });
}

function syncManualDropInput() {
    const selectEl = document.getElementById('preprocessManualDropColumns');
    if (!selectEl) {
        return;
    }

    const selectedColumns = new Set(preprocessingConfig?.manualDropColumns?.columns || []);
    Array.from(selectEl.options).forEach(option => {
        option.selected = selectedColumns.has(option.value);
    });
}

function populateManualDropOptions() {
    const selectEl = document.getElementById('preprocessManualDropColumns');
    if (!selectEl) {
        return;
    }

    const manualColumns = Array.isArray(preprocessingConfig?.manualDropColumns?.columns)
        ? [...preprocessingConfig.manualDropColumns.columns]
        : [];
    const datasetColumns = Array.isArray(columnInsightsData?.column_insights)
        ? columnInsightsData.column_insights.map(col => col.name)
        : [];

    const seen = new Set();
    const combined = [...datasetColumns, ...manualColumns]
        .map(name => (name ?? '').toString().trim())
        .filter(name => {
            if (!name || seen.has(name)) {
                return false;
            }
            seen.add(name);
            return true;
        })
        .sort((a, b) => a.localeCompare(b));

    const activeSelection = new Set(manualColumns);

    selectEl.innerHTML = '';
    combined.forEach(columnName => {
        const option = document.createElement('option');
        option.value = columnName;
        option.textContent = columnName;
        option.selected = activeSelection.has(columnName);
        selectEl.appendChild(option);
    });
}

function deriveInsightDropSuggestions() {
    if (!Array.isArray(columnInsightsData?.column_insights)) {
        return [];
    }

    const threshold = (() => {
        const configured = preprocessingConfig?.dropMissingColumns?.threshold;
        if (typeof configured === 'number' && !Number.isNaN(configured)) {
            return configured;
        }
        return 80;
    })();

    const suggestions = new Set();

    columnInsightsData.column_insights.forEach(col => {
        if (!col || !col.name) {
            return;
        }

        const nullPct = typeof col.null_percentage === 'number' ? col.null_percentage : parseFloat(col.null_percentage);
        const issueMessages = Array.isArray(col.issue_messages) ? col.issue_messages : [];
        const inferredStats = typeof col.statistics === 'object' && col.statistics !== null ? col.statistics : {};

        const singleValueFlags = [inferredStats.constant_value, inferredStats.single_value, inferredStats.low_variance];
        const hasBooleanLowVariance = singleValueFlags.some(flag => flag === true || flag === 'true');

        const uniqueRatio = typeof inferredStats.unique_ratio === 'number' ? inferredStats.unique_ratio : NaN;
        const hasLowUniqueRatio = Number.isFinite(uniqueRatio) && uniqueRatio <= 0.02;

        const uniqueCount =
            typeof inferredStats.unique_count === 'number'
                ? inferredStats.unique_count
                : typeof inferredStats.distinct_count === 'number'
                ? inferredStats.distinct_count
                : null;
        const hasSingleUnique = uniqueCount !== null && uniqueCount <= 1;

        const hasLowVarianceIssue = issueMessages.some(message =>
            typeof message === 'string' && /constant|single value|low variance|unique value|zero variance/i.test(message)
        );

        const mostlyMissing = typeof nullPct === 'number' && !Number.isNaN(nullPct) && nullPct >= threshold;

        if (mostlyMissing || hasBooleanLowVariance || hasLowVarianceIssue || hasLowUniqueRatio || hasSingleUnique) {
            const normalizedName = (col.name ?? '').toString().trim();
            if (normalizedName) {
                suggestions.add(normalizedName);
            }
        }
    });

    return Array.from(suggestions);
}

function getCurrentDropRecommendations(report = lastPreprocessingReport) {
    const fromReport = Array.isArray(report?.column_drop_recommendations)
        ? report.column_drop_recommendations
        : [];
    const fromInsights = deriveInsightDropSuggestions();

    const seen = new Set();
    const combined = [];

    [...fromReport, ...fromInsights].forEach(columnName => {
        const normalized = (columnName ?? '').toString().trim();
        if (!normalized || seen.has(normalized)) {
            return;
        }
        seen.add(normalized);
        combined.push(normalized);
    });

    return combined;
}

function renderDropRecommendations(report = lastPreprocessingReport) {
    const container = document.getElementById('preprocessDropRecommendations');
    if (!container) {
        return;
    }

    container.innerHTML = '';

    const recommendations = getCurrentDropRecommendations(report);

    if (!recommendations.length) {
        const emptyState = document.createElement('span');
        emptyState.className = 'text-muted small';
        emptyState.textContent =
            'No obvious drop candidates yet. Check column insights for high-missing or low-variance columns, or apply preprocessing to refresh suggestions.';
        container.appendChild(emptyState);
        return;
    }

    const selected = new Set(preprocessingConfig?.manualDropColumns?.columns || []);

    const guidance = document.createElement('span');
    guidance.className = 'text-muted small d-block w-100';
    guidance.textContent = 'Suggested from current column insights (missing % and low-variance indicators).';
    container.appendChild(guidance);

    recommendations.forEach(columnName => {
        const badge = document.createElement('span');
        badge.className = `badge rounded-pill px-3 py-2 preprocess-drop-chip ${
            selected.has(columnName) ? 'text-bg-primary' : 'text-bg-light border border-secondary-subtle'
        }`;
        badge.textContent = columnName;
        container.appendChild(badge);
    });
}

function normalizePreprocessingConfig(rawConfig) {
    const defaults = getDefaultPreprocessingConfig();
    if (!rawConfig || typeof rawConfig !== 'object') {
        return clonePreprocessingConfig(defaults);
    }

    const normalized = clonePreprocessingConfig(defaults);

    if (rawConfig.dropMissingColumns) {
        normalized.dropMissingColumns.enabled = Boolean(rawConfig.dropMissingColumns.enabled);
        normalized.dropMissingColumns.threshold = normalizeThreshold(
            rawConfig.dropMissingColumns.threshold,
            defaults.dropMissingColumns.threshold,
        );
    }

    if (rawConfig.dropMissingRows) {
        normalized.dropMissingRows.enabled = Boolean(rawConfig.dropMissingRows.enabled);
        normalized.dropMissingRows.threshold = normalizeThreshold(
            rawConfig.dropMissingRows.threshold,
            defaults.dropMissingRows.threshold,
        );
    }

    const manualDropSource =
        rawConfig.drop_columns ??
        rawConfig.manualDropColumns ??
        rawConfig.explicit_drop_columns ??
        rawConfig.dropColumns;

    const manualColumns = normalizeColumnList(manualDropSource);
    normalized.manualDropColumns.columns = manualColumns;
    const manualEnabledFlag = Boolean(
        manualDropSource && typeof manualDropSource === 'object' && manualDropSource.enabled
    );
    normalized.manualDropColumns.enabled = manualColumns.length > 0 ? true : manualEnabledFlag;

    const suggestedDropSource =
        rawConfig?.suggested_drop_columns ??
        rawConfig?.suggestedDropColumns ??
        rawConfig?.drop_column_suggestions;
    const suggestedColumns = normalizeColumnList(suggestedDropSource);
    if (suggestedColumns.length) {
        const merged = new Set(normalized.manualDropColumns.columns);
        suggestedColumns.forEach(columnName => {
            if (columnName) {
                merged.add(columnName);
            }
        });
        normalized.manualDropColumns.columns = Array.from(merged);
        if (normalized.manualDropColumns.columns.length > 0) {
            normalized.manualDropColumns.enabled = true;
        }
    }

    normalized.dropDuplicates = Boolean(rawConfig.dropDuplicates);

    if (rawConfig.imputation) {
        const strategy = (rawConfig.imputation.strategy || defaults.imputation.strategy || 'none')
            .toString()
            .toLowerCase();
        normalized.imputation.strategy = strategy;
        normalized.imputation.fillValue =
            rawConfig.imputation.fillValue !== undefined
                ? rawConfig.imputation.fillValue
                : defaults.imputation.fillValue;
        normalized.imputation.neighbors = normalizeInteger(
            rawConfig.imputation.neighbors,
            defaults.imputation.neighbors,
            1,
            50,
        );
    }

    return normalized;
}

function loadStoredPreprocessingConfig() {
    try {
        const raw = sessionStorage.getItem(PREPROCESSING_STORAGE_KEY);
        if (!raw) {
            return null;
        }
        const parsed = JSON.parse(raw);
        return normalizePreprocessingConfig(parsed);
    } catch (error) {
        console.warn('Unable to load stored preprocessing config:', error);
        return null;
    }
}

function savePreprocessingConfig() {
    try {
        sessionStorage.setItem(PREPROCESSING_STORAGE_KEY, JSON.stringify(preprocessingConfig));
    } catch (error) {
        console.warn('Unable to persist preprocessing config:', error);
    }
}

function getPreprocessingAppliedStorageKey() {
    if (typeof sessionStorage === 'undefined') {
        return null;
    }

    const id = sourceId || initSourceId();
    if (!id) {
        return null;
    }

    return `${PREPROCESSING_APPLIED_STORAGE_KEY}:${id}`;
}

function loadStoredPreprocessingAppliedState() {
    const storageKey = getPreprocessingAppliedStorageKey();
    if (!storageKey) {
        return null;
    }

    try {
        const raw = sessionStorage.getItem(storageKey);
        if (raw === null) {
            return null;
        }
        return raw === 'true';
    } catch (error) {
        console.warn('Unable to load stored preprocessing applied state:', error);
        return null;
    }
}

function persistPreprocessingAppliedState(appliedValue) {
    const storageKey = getPreprocessingAppliedStorageKey();
    if (!storageKey) {
        return;
    }

    try {
        const value = typeof appliedValue === 'boolean' ? appliedValue : preprocessingApplied;
        sessionStorage.setItem(storageKey, value ? 'true' : 'false');
    } catch (error) {
        console.warn('Unable to persist preprocessing applied state:', error);
    }
}

function buildAnalysisCategoryLookup(categories = ANALYSIS_CATEGORIES) {
    const lookup = {};
    Object.entries(categories || {}).forEach(([categoryKey, category]) => {
        if (!category || !Array.isArray(category.options)) {
            return;
        }

        category.options.forEach(option => {
            if (option.value) {
                lookup[option.value] = categoryKey;
            }

            if (option.subcategory && Array.isArray(option.suboptions)) {
                option.suboptions.forEach(subopt => {
                    if (subopt.value) {
                        lookup[subopt.value] = categoryKey;
                    }
                });
            }
        });
    });
    return lookup;
}

function getAnalysisCategory(analysisId) {
    return analysisId ? ANALYSIS_CATEGORY_LOOKUP[analysisId] || null : null;
}

function getFriendlyImputationName(strategy) {
    switch ((strategy || 'none').toLowerCase()) {
        case 'mean':
            return 'Mean';
        case 'median':
            return 'Median';
        case 'most_frequent':
            return 'Most frequent';
        case 'constant':
            return 'Constant';
        case 'knn':
            return 'KNN';
        default:
            return 'None';
    }
}

function buildPreprocessingSummary(report) {
    const summaryParts = [];
    const config = report?.options ? normalizePreprocessingConfig(report.options) : preprocessingConfig;

    if (!config) {
        return '';
    }

    if (config.dropMissingColumns?.enabled) {
        summaryParts.push(`Cols ≥ ${config.dropMissingColumns.threshold ?? 0}%`);
    }
    if (config.dropMissingRows?.enabled) {
        summaryParts.push(`Rows ≥ ${config.dropMissingRows.threshold ?? 0}%`);
    }
    if (config.manualDropColumns?.enabled && config.manualDropColumns.columns?.length) {
        summaryParts.push(`Drop cols (${config.manualDropColumns.columns.length})`);
    }
    if (config.dropDuplicates) {
        summaryParts.push('Duplicates removed');
    }
    if (config.imputation?.strategy && config.imputation.strategy !== 'none') {
        summaryParts.push(`Impute: ${getFriendlyImputationName(config.imputation.strategy)}`);
    }

    return summaryParts.join(' • ');
}

function isCategoryLocked(categoryKey) {
    void categoryKey;
    // All analysis categories are now available regardless of preprocessing state.
    return false;
}

function updatePreprocessingStatusBadge(report) {
    const badge = document.getElementById('preprocessingStatusBadge');
    if (!badge) {
        return;
    }

    const variantClasses = [
        'bg-secondary-subtle',
        'bg-warning-subtle',
        'bg-success-subtle',
        'bg-danger-subtle',
        'text-body-secondary',
        'text-warning',
        'text-success',
        'text-danger',
    ];
    variantClasses.forEach(cls => badge.classList.remove(cls));
    badge.removeAttribute('title');

    if (preprocessingApplied) {
        const appliedSummary = buildPreprocessingSummary(report || lastPreprocessingReport);
        const persistenceMessage = appliedSummary
            ? `${appliedSummary} • saved until reset`
            : 'Preprocessing applied • saved until reset';
        badge.textContent = persistenceMessage;
        badge.classList.add('bg-success-subtle', 'text-success');
        badge.setAttribute('title', 'Preprocessing stays active for this dataset after reload until you reset it.');
        return;
    }

    const planSummary = buildPreprocessingSummary(report);

    if (preprocessingDirty) {
        const message = planSummary
            ? `${planSummary} • ready to apply for cleaner analyses`
            : 'Preprocessing plan ready • apply to clean your data before running analyses';
        badge.textContent = message;
        badge.classList.add('bg-warning-subtle', 'text-warning');
        return;
    }

    badge.textContent = planSummary || 'Preprocessing optional • apply to clean your data before running analyses';
    badge.classList.add('bg-secondary-subtle', 'text-body-secondary');
}

function refreshCategoryLocks() {
    document.querySelectorAll('.analysis-category-card').forEach(card => {
        const categoryKey = card?.dataset?.category;
        const locked = isCategoryLocked(categoryKey);
        card.classList.toggle('locked', locked);
        card.setAttribute('aria-disabled', locked ? 'true' : 'false');
    });

    document.querySelectorAll('.analysis-option').forEach(option => {
        const value = option.getAttribute('data-value');
        const categoryKey = getAnalysisCategory(value);
        const locked = isCategoryLocked(categoryKey);
        option.classList.toggle('locked', locked);
        option.setAttribute('aria-disabled', locked ? 'true' : 'false');
    });

    document.querySelectorAll('.selected-category-item').forEach(item => {
        const categoryKey = item?.dataset?.category;
        const locked = isCategoryLocked(categoryKey);
        item.dataset.locked = locked ? 'true' : 'false';
        item.querySelectorAll('.suboption-item, .sub-suboption-item').forEach(sub => {
            sub.classList.toggle('locked', locked);
            sub.setAttribute('data-locked', locked ? 'true' : 'false');
        });
    });
}

function openPreprocessingModal() {
    const modalElement = document.getElementById(PREPROCESSING_MODAL_ID);
    if (!modalElement) {
        console.warn('Preprocessing configuration modal not found in DOM');
        return;
    }

    populatePreprocessingModal();

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
}

function populatePreprocessingModal() {
    const config = preprocessingConfig || getDefaultPreprocessingConfig();

    const dropColumnsToggle = document.getElementById('preprocessDropColumnsToggle');
    const dropColumnsThreshold = document.getElementById('preprocessDropColumnsThreshold');
    if (dropColumnsToggle) {
        dropColumnsToggle.checked = Boolean(config.dropMissingColumns.enabled);
    }
    if (dropColumnsThreshold) {
        dropColumnsThreshold.value = config.dropMissingColumns.threshold ?? '';
        dropColumnsThreshold.disabled = !config.dropMissingColumns.enabled;
    }

    const dropRowsToggle = document.getElementById('preprocessDropRowsToggle');
    const dropRowsThreshold = document.getElementById('preprocessDropRowsThreshold');
    if (dropRowsToggle) {
        dropRowsToggle.checked = Boolean(config.dropMissingRows.enabled);
    }
    if (dropRowsThreshold) {
        dropRowsThreshold.value = config.dropMissingRows.threshold ?? '';
        dropRowsThreshold.disabled = !config.dropMissingRows.enabled;
    }

    const dropDuplicatesToggle = document.getElementById('preprocessDropDuplicatesToggle');
    if (dropDuplicatesToggle) {
        dropDuplicatesToggle.checked = Boolean(config.dropDuplicates);
    }

    const imputationStrategy = document.getElementById('preprocessImputationStrategy');
    if (imputationStrategy) {
        imputationStrategy.value = config.imputation.strategy || 'none';
    }

    const fillValueInput = document.getElementById('preprocessImputationFillValue');
    if (fillValueInput) {
        fillValueInput.value = config.imputation.fillValue ?? '';
    }

    const neighborsInput = document.getElementById('preprocessImputationNeighbors');
    if (neighborsInput) {
        neighborsInput.value = config.imputation.neighbors ?? 5;
    }

    toggleImputationDetails(config.imputation.strategy);

    populateManualDropOptions();
    syncManualDropInput();
    renderDropRecommendations(lastPreprocessingReport);

    attachPreprocessingInputListeners();

    const statusEl = document.getElementById('preprocessingModalStatus');
    if (statusEl) {
        if (preprocessingApplied) {
            statusEl.textContent = 'Preprocessing is active. You can now cap, transform, or remove outliers as needed after reviewing the analysis output.';
        } else if (preprocessingDirty) {
            statusEl.textContent = 'Apply the updated plan so advanced analyses run on the trimmed dataset before you tackle outlier or transformation steps.';
        } else {
            statusEl.textContent = 'Start by dropping obvious problem columns or duplicates, then revisit for outlier handling after your first EDA pass.';
        }
    }

    updatePreprocessingPreview(lastPreprocessingReport);
}

function attachPreprocessingInputListeners() {
    if (preprocessingEventListenersAttached) {
        return;
    }

    preprocessingConfig.manualDropColumns = preprocessingConfig.manualDropColumns || { enabled: false, columns: [] };

    const dropColumnsToggle = document.getElementById('preprocessDropColumnsToggle');
    if (dropColumnsToggle) {
        dropColumnsToggle.addEventListener('change', event => {
            preprocessingConfig.dropMissingColumns.enabled = event.target.checked;
            if (dropColumnsThreshold) {
                dropColumnsThreshold.disabled = !event.target.checked;
            }
            markPreprocessingDirty();
            updatePreprocessingPreview();
            rerenderColumnItems();
        });
    }

    const dropColumnsThreshold = document.getElementById('preprocessDropColumnsThreshold');
    if (dropColumnsThreshold) {
        dropColumnsThreshold.addEventListener('change', event => {
            const sanitized = normalizeThreshold(event.target.value, preprocessingConfig.dropMissingColumns.threshold);
            preprocessingConfig.dropMissingColumns.threshold = sanitized;
            event.target.value = sanitized ?? '';
            markPreprocessingDirty();
            updatePreprocessingPreview();
            rerenderColumnItems();
        });
    }

    const dropRowsToggle = document.getElementById('preprocessDropRowsToggle');
    if (dropRowsToggle) {
        dropRowsToggle.addEventListener('change', event => {
            preprocessingConfig.dropMissingRows.enabled = event.target.checked;
            if (dropRowsThreshold) {
                dropRowsThreshold.disabled = !event.target.checked;
            }
            markPreprocessingDirty();
            updatePreprocessingPreview();
            rerenderColumnItems();
        });
    }

    const dropRowsThreshold = document.getElementById('preprocessDropRowsThreshold');
    if (dropRowsThreshold) {
        dropRowsThreshold.addEventListener('change', event => {
            const sanitized = normalizeThreshold(event.target.value, preprocessingConfig.dropMissingRows.threshold);
            preprocessingConfig.dropMissingRows.threshold = sanitized;
            event.target.value = sanitized ?? '';
            markPreprocessingDirty();
            updatePreprocessingPreview();
            rerenderColumnItems();
        });
    }

    const manualDropSelect = document.getElementById('preprocessManualDropColumns');
    if (manualDropSelect) {
        manualDropSelect.addEventListener('change', () => {
            const selectedValues = Array.from(manualDropSelect.selectedOptions).map(option => option.value);
            preprocessingConfig.manualDropColumns.columns = selectedValues;
            preprocessingConfig.manualDropColumns.enabled = selectedValues.length > 0;
            markPreprocessingDirty();
            updatePreprocessingPreview();
            rerenderColumnItems();
            renderDropRecommendations(lastPreprocessingReport);
        });
    }

    const applyRecommendationsBtn = document.getElementById('applyDropRecommendationsBtn');
    if (applyRecommendationsBtn) {
        applyRecommendationsBtn.addEventListener('click', () => {
            const recommendations = getCurrentDropRecommendations(lastPreprocessingReport);

            if (!recommendations.length) {
                if (typeof showNotification === 'function') {
                    showNotification('No clear drop suggestions right now. Review column insights or adjust thresholds to surface candidates.', 'info');
                }
                return;
            }

            const nextSelection = new Set(preprocessingConfig.manualDropColumns.columns || []);
            recommendations.forEach(columnName => {
                if (columnName) {
                    nextSelection.add(columnName);
                }
            });

            preprocessingConfig.manualDropColumns.columns = Array.from(nextSelection);
            preprocessingConfig.manualDropColumns.enabled = preprocessingConfig.manualDropColumns.columns.length > 0;

            populateManualDropOptions();
            syncManualDropInput();
            markPreprocessingDirty();
            updatePreprocessingPreview();
            rerenderColumnItems();
            renderDropRecommendations(lastPreprocessingReport);
        });
    }

    const removeSelectedBtn = document.getElementById('removeSelectedDropColumnsBtn');
    if (removeSelectedBtn) {
        removeSelectedBtn.addEventListener('click', () => {
            const selectEl = document.getElementById('preprocessManualDropColumns');
            if (!selectEl) {
                return;
            }

            const selectedValues = Array.from(selectEl.selectedOptions).map(option => option.value);
            if (!selectedValues.length) {
                if (typeof showNotification === 'function') {
                    showNotification('Select one or more columns in the list to remove them from the drop plan.', 'info');
                }
                return;
            }

            const current = new Set(preprocessingConfig.manualDropColumns.columns || []);
            selectedValues.forEach(columnName => {
                if (columnName) {
                    current.delete(columnName);
                }
            });

            preprocessingConfig.manualDropColumns.columns = Array.from(current);
            preprocessingConfig.manualDropColumns.enabled = preprocessingConfig.manualDropColumns.columns.length > 0;

            populateManualDropOptions();
            syncManualDropInput();
            markPreprocessingDirty();
            updatePreprocessingPreview();
            rerenderColumnItems();
            renderDropRecommendations(lastPreprocessingReport);

            if (typeof showNotification === 'function') {
                showNotification('Removed selected columns from the drop list.', 'success');
            }
        });
    }

    const dropDuplicatesToggle = document.getElementById('preprocessDropDuplicatesToggle');
    if (dropDuplicatesToggle) {
        dropDuplicatesToggle.addEventListener('change', event => {
            preprocessingConfig.dropDuplicates = event.target.checked;
            markPreprocessingDirty();
            updatePreprocessingPreview();
        });
    }

    const imputationStrategy = document.getElementById('preprocessImputationStrategy');
    if (imputationStrategy) {
        imputationStrategy.addEventListener('change', event => {
            preprocessingConfig.imputation.strategy = event.target.value || 'none';
            toggleImputationDetails(preprocessingConfig.imputation.strategy);
            markPreprocessingDirty();
            updatePreprocessingPreview();
        });
    }

    const fillValueInput = document.getElementById('preprocessImputationFillValue');
    if (fillValueInput) {
        fillValueInput.addEventListener('input', event => {
            preprocessingConfig.imputation.fillValue = event.target.value;
            markPreprocessingDirty();
        });
    }

    const neighborsInput = document.getElementById('preprocessImputationNeighbors');
    if (neighborsInput) {
        neighborsInput.addEventListener('change', event => {
            preprocessingConfig.imputation.neighbors = normalizeInteger(
                event.target.value,
                preprocessingConfig.imputation.neighbors,
                1,
                25,
            );
            neighborsInput.value = preprocessingConfig.imputation.neighbors;
            markPreprocessingDirty();
        });
    }

    const applyButton = document.getElementById('applyPreprocessingBtn');
    if (applyButton) {
        applyButton.addEventListener('click', handleApplyPreprocessing);
    }

    preprocessingEventListenersAttached = true;
}

function toggleImputationDetails(strategy) {
    const normalized = (strategy || 'none').toLowerCase();
    const constantGroup = document.getElementById('imputationConstantGroup');
    const neighborsGroup = document.getElementById('imputationNeighborsGroup');

    if (constantGroup) {
        constantGroup.classList.toggle('d-none', normalized !== 'constant');
    }
    if (neighborsGroup) {
        neighborsGroup.classList.toggle('d-none', normalized !== 'knn');
    }
}

function markPreprocessingDirty() {
    preprocessingDirty = true;
    preprocessingApplied = false;
    updatePreprocessingStatusBadge();
    refreshCategoryLocks();
}

function buildPreprocessingPayload() {
    if (!preprocessingConfig) {
        return null;
    }

    return {
        drop_missing_columns: {
            enabled: preprocessingConfig.dropMissingColumns.enabled,
            threshold: preprocessingConfig.dropMissingColumns.enabled
                ? preprocessingConfig.dropMissingColumns.threshold
                : null,
        },
        drop_missing_rows: {
            enabled: preprocessingConfig.dropMissingRows.enabled,
            threshold: preprocessingConfig.dropMissingRows.enabled
                ? preprocessingConfig.dropMissingRows.threshold
                : null,
        },
        drop_columns: {
            enabled:
                Boolean(preprocessingConfig.manualDropColumns.enabled) &&
                Array.isArray(preprocessingConfig.manualDropColumns.columns) &&
                preprocessingConfig.manualDropColumns.columns.length > 0,
            columns: Array.isArray(preprocessingConfig.manualDropColumns.columns)
                ? preprocessingConfig.manualDropColumns.columns
                : [],
        },
        drop_duplicates: preprocessingConfig.dropDuplicates,
        imputation: {
            strategy: preprocessingConfig.imputation.strategy || 'none',
            fill_value: preprocessingConfig.imputation.fillValue ?? '',
            neighbors: preprocessingConfig.imputation.neighbors,
        },
    };
}

function updatePreprocessingPreview(report) {
    const previewSummaryEl = document.getElementById('preprocessingPreviewSummary');
    const previewListEl = document.getElementById('preprocessingPreviewList');

    if (previewSummaryEl) {
    const summary = buildPreprocessingSummary(report);
    previewSummaryEl.textContent = summary || 'No preprocessing changes yet. Begin by removing noisy columns or duplicates, then explore outlier treatments after initial analyses.';
    }

    if (previewListEl) {
        previewListEl.innerHTML = '';
    }

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        return;
    }

    const config = preprocessingConfig || getDefaultPreprocessingConfig();
    const threshold = config.dropMissingColumns.enabled
        ? config.dropMissingColumns.threshold
        : null;

    const manualDrops = new Set(config.manualDropColumns?.enabled ? config.manualDropColumns.columns || [] : []);

    const appliedDrops = new Set(report?.dropped_columns || lastPreprocessingReport?.dropped_columns || []);
    const plannedDrops = [];

    columnInsightsData.column_insights.forEach(col => {
        if (!Object.prototype.hasOwnProperty.call(col, '_original_status_display')) {
            col._original_status_display = col.status_display;
        }

        const alreadyDropped = appliedDrops.has(col.name) || Boolean(col.dropped);
        const shouldPreviewDrop =
            !alreadyDropped &&
            threshold !== null &&
            typeof col.null_percentage === 'number' &&
            col.null_percentage >= threshold;

    const manuallySelected = !alreadyDropped && manualDrops.has(col.name);

    const willDrop = shouldPreviewDrop || manuallySelected;

        col.drop_preview = willDrop;
        col.preview_threshold = shouldPreviewDrop ? threshold : null;
        if (manuallySelected) {
            col.preview_message = 'Will be dropped (manual selection)';
        } else if (shouldPreviewDrop) {
            col.preview_message = `Will be dropped (≥ ${Math.round(threshold)}% missing)`;
        } else {
            col.preview_message = '';
        }

        if (willDrop) {
            let reason = 'planned';
            if (manuallySelected) {
                reason = 'manual selection';
            } else if (shouldPreviewDrop) {
                reason = `≥ ${Math.round(threshold ?? 0)}% missing`;
            }
            plannedDrops.push({ name: col.name, reason });
        }

        if (!alreadyDropped && !willDrop && col._original_status_display !== undefined) {
            col.status_display = col._original_status_display;
        }
    });

    if (previewListEl) {
        if (appliedDrops.size === 0 && plannedDrops.length === 0) {
            const pill = document.createElement('span');
            pill.className = 'preprocessing-preview-pill preview-plan';
            pill.textContent = 'No columns scheduled for removal';
            previewListEl.appendChild(pill);
        } else {
            appliedDrops.forEach(columnName => {
                const pill = document.createElement('span');
                pill.className = 'preprocessing-preview-pill preview-applied';
                pill.textContent = `${columnName} • dropped`;
                previewListEl.appendChild(pill);
            });
            plannedDrops.forEach(({ name, reason }) => {
                const pill = document.createElement('span');
                pill.className = 'preprocessing-preview-pill preview-plan';
                pill.textContent = `${name} • ${reason}`;
                previewListEl.appendChild(pill);
            });
        }
    }
}

function applyDroppedColumnsToInsights(droppedColumns = []) {
    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        return;
    }

    const droppedSet = new Set(droppedColumns || []);
    const droppedNames = [];
    let activeColumnsCount = 0;
    let numericCount = 0;
    let textCount = 0;
    let datetimeCount = 0;
    let booleanCount = 0;
    let problematicCount = 0;
    let missingCount = 0;

    columnInsightsData.column_insights.forEach(col => {
        if (!Object.prototype.hasOwnProperty.call(col, '_original_status_display')) {
            col._original_status_display = col.status_display;
        }

        const isDropped = droppedSet.has(col.name);
        col.dropped = isDropped;

        if (isDropped) {
            droppedNames.push(col.name);
            col.drop_preview = false;
            col.preview_message = '';
            col.preview_threshold = null;
            col.status_display = 'Dropped from the dataset – not available';
            const issues = Array.isArray(col.issue_messages) ? [...col.issue_messages] : [];
            if (!issues.includes('Dropped from the dataset')) {
                issues.push('Dropped from the dataset');
            }
            col.issue_messages = issues;
            col.has_issues = true;
        } else if (!col.drop_preview && col._original_status_display !== undefined) {
            col.status_display = col._original_status_display;
            activeColumnsCount += 1;
            if (col.data_category === 'numeric') {
                numericCount += 1;
            } else if (col.data_category === 'text') {
                textCount += 1;
            } else if (col.data_category === 'datetime') {
                datetimeCount += 1;
            } else if (col.data_category === 'boolean') {
                booleanCount += 1;
            }
            if (col.has_issues) {
                problematicCount += 1;
            }
            if (typeof col.null_percentage === 'number' && col.null_percentage > 0) {
                missingCount += 1;
            }
        } else {
            // Column scheduled for drop preview but not yet applied
            if (!isDropped) {
                activeColumnsCount += 1;
                if (col.data_category === 'numeric') {
                    numericCount += 1;
                } else if (col.data_category === 'text') {
                    textCount += 1;
                } else if (col.data_category === 'datetime') {
                    datetimeCount += 1;
                } else if (col.data_category === 'boolean') {
                    booleanCount += 1;
                }
                if (col.has_issues) {
                    problematicCount += 1;
                }
                if (typeof col.null_percentage === 'number' && col.null_percentage > 0) {
                    missingCount += 1;
                }
            }
        }
    });

    if (!columnInsightsData.summary_stats || typeof columnInsightsData.summary_stats !== 'object') {
        columnInsightsData.summary_stats = {};
    }

    const summary = columnInsightsData.summary_stats;
    summary.total_columns = activeColumnsCount;
    summary.numeric_columns = numericCount;
    summary.text_columns = textCount;
    summary.datetime_columns = datetimeCount;
    summary.boolean_columns = booleanCount;
    summary.problematic_columns = problematicCount;
    summary.missing_data_columns = missingCount;
    summary.dropped_columns = droppedNames;
    summary.dropped_columns_count = droppedNames.length;
}

function refreshColumnInsightsSummary() {
    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        return;
    }

    displayColumnInsights(columnInsightsData);
}

function updateAvailableColumnCaches(activeColumnNames = []) {
    if (!Array.isArray(activeColumnNames)) {
        return;
    }

    if (typeof sessionStorage !== 'undefined') {
        try {
            sessionStorage.setItem('datasetColumns', JSON.stringify(activeColumnNames));
        } catch (error) {
            console.warn('Unable to persist active column cache to sessionStorage:', error);
        }
    }

    if (typeof window !== 'undefined' && window.currentDataFrame && Array.isArray(window.currentDataFrame.columns)) {
        window.currentDataFrame.columns = [...activeColumnNames];
    }
}

function removeDroppedColumnsFromSelection(columns) {
    if (!Array.isArray(columns) || columns.length === 0) {
        return;
    }

    let removed = false;
    columns.forEach(columnName => {
        if (selectedColumns.has(columnName)) {
            selectedColumns.delete(columnName);
            removed = true;
        }
    });

    if (removed) {
        updateSelectedColumnsBadge();
    }
}

function reportIndicatesActivePreprocessing(report) {
    if (!report || typeof report !== 'object') {
        return false;
    }

    if (Array.isArray(report.applied_operations) && report.applied_operations.length > 0) {
        return true;
    }
    if (Array.isArray(report.dropped_columns) && report.dropped_columns.length > 0) {
        return true;
    }
    if (Array.isArray(report.manual_dropped_columns) && report.manual_dropped_columns.length > 0) {
        return true;
    }
    if (Array.isArray(report.suggested_dropped_columns) && report.suggested_dropped_columns.length > 0) {
        return true;
    }
    if (typeof report.duplicate_rows_removed === 'number' && report.duplicate_rows_removed > 0) {
        return true;
    }
    if (typeof report.dropped_rows === 'number' && report.dropped_rows > 0) {
        return true;
    }
    if (report.imputation_details && typeof report.imputation_details === 'object') {
        const strategy = String(report.imputation_details.strategy || '').toLowerCase();
        if (strategy && strategy !== 'none') {
            return true;
        }
    }

    const options = report.options && typeof report.options === 'object' ? report.options : {};
    const dropMissingColumnsEnabled = Boolean(options.drop_missing_columns?.enabled && options.drop_missing_columns?.threshold !== null && options.drop_missing_columns?.threshold !== undefined);
    const dropMissingRowsEnabled = Boolean(options.drop_missing_rows?.enabled && options.drop_missing_rows?.threshold !== null && options.drop_missing_rows?.threshold !== undefined);
    const dropDuplicatesEnabled = Boolean(options.drop_duplicates);

    const explicitDropColumns = (() => {
        if (options.drop_columns?.enabled && Array.isArray(options.drop_columns?.columns) && options.drop_columns.columns.length > 0) {
            return true;
        }
        if (options.manualDropColumns?.enabled && Array.isArray(options.manualDropColumns?.columns) && options.manualDropColumns.columns.length > 0) {
            return true;
        }
        return false;
    })();

    const suggestedDropColumns = (() => {
        if (options.suggested_drop_columns?.enabled && Array.isArray(options.suggested_drop_columns?.columns) && options.suggested_drop_columns.columns.length > 0) {
            return true;
        }
        if (options.suggestedDropColumns?.enabled && Array.isArray(options.suggestedDropColumns?.columns) && options.suggestedDropColumns.columns.length > 0) {
            return true;
        }
        return false;
    })();

    const imputationStrategy = (() => {
        if (!options.imputation || typeof options.imputation !== 'object') {
            return '';
        }
        const strategy = options.imputation.strategy;
        return typeof strategy === 'string' ? strategy.toLowerCase() : '';
    })();

    const imputationActive = Boolean(imputationStrategy && imputationStrategy !== 'none');

    return (
        dropMissingColumnsEnabled ||
        dropMissingRowsEnabled ||
        dropDuplicatesEnabled ||
        explicitDropColumns ||
        suggestedDropColumns ||
        imputationActive
    );
}

function handlePreprocessingReport(report, options = {}) {
    if (!report || typeof report !== 'object') {
        return;
    }

    const { preserveAppliedState = false } = options;

    lastPreprocessingReport = report;

    if (report.options) {
        preprocessingConfig = normalizePreprocessingConfig(report.options);
        savePreprocessingConfig();
    }

    const previousApplied = preprocessingApplied;
    const previousDirty = preprocessingDirty;
    const appliedFlag = typeof report.applied === 'boolean' ? report.applied : null;
    const activeOperations = reportIndicatesActivePreprocessing(report);

    if (appliedFlag === true || activeOperations) {
        preprocessingApplied = true;
        preprocessingDirty = false;
    } else if (appliedFlag === false && !activeOperations) {
        if (previousApplied || preserveAppliedState) {
            preprocessingApplied = true;
            preprocessingDirty = false;
        } else {
            preprocessingApplied = false;
            preprocessingDirty = true;
        }
    } else {
        preprocessingApplied = previousApplied;
        preprocessingDirty = preprocessingApplied ? false : previousDirty;
    }

    persistPreprocessingAppliedState(preprocessingApplied);

    const droppedColumns = report.dropped_columns || [];
    applyDroppedColumnsToInsights(droppedColumns);
    removeDroppedColumnsFromSelection(droppedColumns);

    if (columnInsightsData && Array.isArray(columnInsightsData.column_insights)) {
        const activeColumns = columnInsightsData.column_insights
            .filter(col => !col.dropped)
            .map(col => col.name);
        updateAvailableColumnCaches(activeColumns);
    }

    refreshColumnInsightsSummary();
    populateManualDropOptions();
    syncManualDropInput();
    renderDropRecommendations(report);
    updatePreprocessingPreview(report);
    updatePreprocessingStatusBadge(report);
    refreshCategoryLocks();
}

async function applyPreprocessingAndRefresh(preprocessingPayload) {
    const sourceId = initSourceId();
    if (!sourceId) {
        throw new Error('Missing source ID for preprocessing operations.');
    }

    try {
        const response = await fetch(`/advanced-eda/api/column-insights/${sourceId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                preprocessing: preprocessingPayload,
                base: preprocessingApplied ? 'current' : 'auto',
            }),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data?.error || `HTTP ${response.status}: ${response.statusText}`);
        }

        return data;
    } catch (error) {
        throw error;
    }
}

function rerenderColumnItems() {
    if (columnInsightsData && Array.isArray(columnInsightsData.column_insights)) {
        renderColumnItems(columnInsightsData.column_insights);
    }
}

async function handleApplyPreprocessing() {
    if (!preprocessingConfig) {
        preprocessingConfig = getDefaultPreprocessingConfig();
    }

    if (preprocessingConfig.dropMissingColumns.enabled && preprocessingConfig.dropMissingColumns.threshold === null) {
        showNotification('Set a threshold for dropping columns.', 'warning');
        return;
    }

    if (preprocessingConfig.dropMissingRows.enabled && preprocessingConfig.dropMissingRows.threshold === null) {
        showNotification('Set a threshold for dropping rows.', 'warning');
        return;
    }

    const preprocessingPayload = buildPreprocessingPayload();
    const modalElement = document.getElementById(PREPROCESSING_MODAL_ID);
    const applyButton = document.getElementById('applyPreprocessingBtn');
    let previousButtonLabel = null;

    if (applyButton) {
        previousButtonLabel = applyButton.innerHTML;
        applyButton.disabled = true;
        applyButton.setAttribute('aria-busy', 'true');
        applyButton.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Applying preprocessing…
        `;
    }

    showColumnInsightsLoading();

    try {
        const appliedData = await applyPreprocessingAndRefresh(preprocessingPayload);

        if (!appliedData || !appliedData.success) {
            throw new Error(appliedData?.error || 'Preprocessing failed.');
        }

        columnInsightsData = appliedData;
        window.columnInsightsData = appliedData;

        selectedColumns.clear();
        appliedData.column_insights.forEach((col, index) => {
            col.__index = index;
            if (col.selected) {
                selectedColumns.add(col.name);
            }
        });

        if (appliedData.preprocessing_report) {
            handlePreprocessingReport(appliedData.preprocessing_report);
        } else {
            preprocessingApplied = true;
            preprocessingDirty = false;
            persistPreprocessingAppliedState(true);
            refreshColumnInsightsSummary();
            updatePreprocessingPreview();
            updatePreprocessingStatusBadge();
            refreshCategoryLocks();
        }

        savePreprocessingConfig();
        showNotification('Preprocessing applied and column insights updated.', 'success');

        const statusEl = document.getElementById('preprocessingModalStatus');
        if (statusEl) {
            statusEl.textContent = 'Preprocessing applied. Column insights refreshed.';
        }
    } catch (error) {
        console.error('Apply preprocessing failed:', error);
        showColumnInsightsError();
        showNotification(error.message || 'Failed to apply preprocessing configuration.', 'error');
        return;
    } finally {
        if (applyButton) {
            applyButton.disabled = false;
            applyButton.removeAttribute('aria-busy');
            if (previousButtonLabel !== null) {
                applyButton.innerHTML = previousButtonLabel;
            }
        }
        if (modalElement) {
            const modalInstance = bootstrap.Modal.getInstance(modalElement);
            if (modalInstance) {
                modalInstance.hide();
            }
        }
    }
}

function initializePreprocessingState() {
    preprocessingConfig = normalizePreprocessingConfig(preprocessingConfig || getDefaultPreprocessingConfig());

    const storedApplied = loadStoredPreprocessingAppliedState();
    if (storedApplied !== null) {
        preprocessingApplied = storedApplied;
        if (storedApplied) {
            preprocessingDirty = false;
        }
    }

    bindPreprocessingControls();

    if (!preprocessingInitialized) {
        attachPreprocessingInputListeners();
        preprocessingInitialized = true;
    }

    updatePreprocessingStatusBadge();
    refreshCategoryLocks();
    updatePreprocessingPreview();
}

function bindPreprocessingControls() {
    if (preprocessingControlsBound) {
        return;
    }

    const openBtn = document.getElementById('openPreprocessingConfigBtn');
    if (openBtn) {
        openBtn.addEventListener('click', event => {
            event.preventDefault();
            openPreprocessingModal();
        });
    }

    const resetBtn = document.getElementById('resetPreprocessingBtn');
    if (resetBtn) {
        resetBtn.addEventListener('click', async event => {
            event.preventDefault();

            if (resetBtn.disabled) {
                return;
            }

            const confirmReset = confirm('Reset preprocessing and restore the original column insights?');
            if (!confirmReset) {
                return;
            }

            const previousLabel = resetBtn.innerHTML;
            resetBtn.disabled = true;
            resetBtn.setAttribute('aria-busy', 'true');
            resetBtn.innerHTML = `
                <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                Resetting…
            `;

            try {
                await resetPreprocessingState({ resetConfig: true, notify: true });
            } catch (error) {
                console.error('Preprocessing reset failed:', error);
            } finally {
                resetBtn.disabled = false;
                resetBtn.removeAttribute('aria-busy');
                resetBtn.innerHTML = previousLabel;
                resetBtn.classList.remove('active');
                resetBtn.blur();
            }
        });
    }

    preprocessingControlsBound = true;
}

async function resetPreprocessingState({ resetConfig = false, notify = true } = {}) {
    if (resetConfig) {
        preprocessingConfig = getDefaultPreprocessingConfig();
    }

    preprocessingApplied = false;
    preprocessingDirty = true;
    lastPreprocessingReport = null;
    persistPreprocessingAppliedState(false);

    applyDroppedColumnsToInsights([]);
    if (columnInsightsData && Array.isArray(columnInsightsData.column_insights)) {
        const activeColumns = columnInsightsData.column_insights.map(col => col.name);
        updateAvailableColumnCaches(activeColumns);
    }

    refreshColumnInsightsSummary();
    updatePreprocessingPreview();

    updatePreprocessingStatusBadge();
    refreshCategoryLocks();
    savePreprocessingConfig();

    const sourceId = initSourceId();
    if (!sourceId) {
        if (notify) {
            showNotification('Preprocessing settings reset locally. Reload the notebook after saving to confirm.', 'info');
        }
        return;
    }

    try {
        const response = await fetch(`/advanced-eda/api/preprocessing/${sourceId}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        if (notify) {
            showNotification('Preprocessing reset. Column insights restored to the original dataset.', 'info');
        }

        showColumnInsightsLoading();
        await loadColumnInsights();
    } catch (error) {
        console.error('Failed to reset preprocessing state:', error);
        if (notify) {
            showNotification(error.message || 'Failed to reset preprocessing state.', 'error');
        }
        throw error;
    }
}

// Initialize source ID from URL parameters
function initSourceId() {
    if (!sourceId) {
        sourceId = new URLSearchParams(window.location.search).get('source_id');
    }
    return sourceId;
}

// Initialize the interactive analysis grid with multi-selection support


function escapeHtml(value) {
    if (value === null || value === undefined) {
        return '';
    }

    return value
        .toString()
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

let analysisCatalogueUpdateListenerAttached = false;

async function initAnalysisGrid() {
    console.log('Initializing analysis grid with multi-selection');
    
    try {
        await ensureAnalysisCatalogueLoaded();
    } catch (error) {
        console.warn('Unable to load analysis catalogue from API. Using fallback data.', error);
    }

    // Check if we have the new grid system
    if (document.getElementById('analysisCardsGrid')) {
        initializeEnhancedAnalysisGrid();
    } else {
        // Fallback to original system with marketing modal support
        const options = document.querySelectorAll('.analysis-option');
        
        options.forEach(option => {
            option.addEventListener('click', function(e) {
                e.preventDefault();
                const analysisType = this.getAttribute('data-value');
                triggerAnalysisRun(analysisType, this);
            });
        });
    }

    // Initialize bulk selection buttons (removed - no longer needed)
}

// Initialize the enhanced analysis grid system
function initializeEnhancedAnalysisGrid() {
    console.log('Initializing enhanced analysis grid system');
    
    // Initialize grid view by default
    renderAnalysisCardsGrid();

    if (!analysisCatalogueUpdateListenerAttached) {
        document.addEventListener('analysis-catalogue:updated', () => {
            renderAnalysisCardsGrid();
            updateSidebar();
        });
        analysisCatalogueUpdateListenerAttached = true;
    }
    
    // Set up event listeners
    setupEnhancedEventListeners();
}

// Render analysis categories as interactive cards
function renderAnalysisCardsGrid() {
    const gridContainer = document.getElementById('analysisCardsGrid');
    if (!gridContainer) return;
    
    gridContainer.innerHTML = '';
    
    const categories = Object.entries(ANALYSIS_CATEGORIES || {})
        .map(([categoryKey, category]) => ({ key: categoryKey, category }))
        .filter(entry => entry.category && Array.isArray(entry.category.options) && entry.category.options.length > 0)
        .sort((a, b) => {
            const orderA = typeof a.category.order === 'number' ? a.category.order : 999;
            const orderB = typeof b.category.order === 'number' ? b.category.order : 999;
            if (orderA !== orderB) {
                return orderA - orderB;
            }
            const titleA = (a.category.title || a.key || '').toString();
            const titleB = (b.category.title || b.key || '').toString();
            return titleA.localeCompare(titleB);
        });

    if (categories.length === 0) {
        gridContainer.innerHTML = '<div class="text-muted small">No analyses are available right now.</div>';
        return;
    }

    categories.forEach(({ key, category }) => {
        const cardElement = createCategoryCard(key, category);
        gridContainer.appendChild(cardElement);
    });

    refreshCategoryLocks();
}

// Create individual category card
function createCategoryCard(categoryKey, category) {
    const isSelected = selectedCategories.has(categoryKey);
    
    const cardDiv = document.createElement('div');
    cardDiv.className = `analysis-category-card ${isSelected ? 'selected' : ''}`;
    cardDiv.dataset.category = categoryKey;
    if (category?.backendCategory) {
        cardDiv.dataset.backendCategory = category.backendCategory;
    }

    const locked = isCategoryLocked(categoryKey);
    cardDiv.classList.toggle('locked', locked);
    cardDiv.setAttribute('aria-disabled', locked ? 'true' : 'false');
    
    const categoryTitle = category?.title ? escapeHtml(category.title) : escapeHtml(categoryKey);
    const categorySubtitle = category?.subtitle ? escapeHtml(category.subtitle) : '';
    const totalOptions = typeof category?.count === 'number'
        ? category.count
        : Array.isArray(category?.options) ? category.options.length : 0;
    const countLabel = totalOptions > 0 ? `${totalOptions} ${totalOptions === 1 ? 'analysis' : 'analyses'}` : '';
    const subtitleHtml = categorySubtitle ? `<p class="category-card-subtitle">${categorySubtitle}</p>` : '';
    const countHtml = countLabel ? `
        <div class="category-card-meta">
            <span class="category-card-count">${escapeHtml(countLabel)}</span>
        </div>
    ` : '';

    cardDiv.innerHTML = `
        <div class="category-card-header">
            <div class="flex-grow-1">
                <h6 class="category-card-title">${categoryTitle}</h6>
                ${subtitleHtml}
            </div>
            ${countHtml}
        </div>
    `;
    
    cardDiv.addEventListener('click', () => handleCategoryClick(categoryKey, category));
    
    return cardDiv;
}

// Handle category card click
function handleCategoryClick(categoryKey, category) {
    console.log('Category clicked:', categoryKey);

    if (isCategoryLocked(categoryKey)) {
        const message = categoryKey === DATA_QUALITY_CATEGORY_KEY
            ? 'Data quality checks are disabled after preprocessing is applied.'
            : 'Apply preprocessing before running this analysis category.';
        showNotification(message, 'warning');
        return;
    }
    
    if (selectedCategories.has(categoryKey)) {
        selectedCategories.delete(categoryKey);
        // Remove all sub-options of this category
        category.options.forEach(opt => selectedSubOptions.delete(opt.value));
    } else {
        selectedCategories.add(categoryKey);
    }
    
    updateCardSelection(categoryKey);
    updateSidebar();
    updateEnhancedAnalysisSelection();
}

// Update card visual selection state
function updateCardSelection(categoryKey) {
    const card = document.querySelector(`[data-category="${categoryKey}"]`);
    if (card) {
        card.classList.toggle('selected', selectedCategories.has(categoryKey));
    }
}

// Update sidebar with selected categories
function updateSidebar() {
    const sidebarElement = document.getElementById('selectedCategoriesSidebar');
    const sidebarContent = document.getElementById('sidebarContent');

    if (!sidebarElement || !sidebarContent) {
        return;
    }

    const emptyStateHTML = `
        <div class="sidebar-empty-state">
            <i class="bi bi-magic"></i>
            <span>Select analyses from the catalogue to build your queue.</span>
        </div>
    `;

    if (selectedCategories.size === 0) {
        sidebarElement.classList.remove('is-active');
        sidebarElement.classList.add('is-empty');
        sidebarElement.setAttribute('aria-hidden', 'true');
        sidebarContent.innerHTML = emptyStateHTML;
        return;
    }

    sidebarElement.classList.add('is-active');
    sidebarElement.classList.remove('is-empty');
    sidebarElement.removeAttribute('aria-hidden');
    sidebarContent.innerHTML = '';

    selectedCategories.forEach(categoryKey => {
        const category = ANALYSIS_CATEGORIES[categoryKey];
        if (!category) {
            selectedCategories.delete(categoryKey);
            return;
        }
        const categoryElement = createSelectedCategoryItem(categoryKey, category);
        sidebarContent.appendChild(categoryElement);
    });
}

// Create selected category item in sidebar
function createSelectedCategoryItem(categoryKey, category) {
    const itemDiv = document.createElement('div');
    itemDiv.className = 'selected-category-item';
    itemDiv.dataset.category = categoryKey;
    if (category?.backendCategory) {
        itemDiv.dataset.backendCategory = category.backendCategory;
    }

    const title = category?.title ? escapeHtml(category.title) : escapeHtml(categoryKey);
    const subtitle = category?.subtitle ? `<small class="text-muted d-block">${escapeHtml(category.subtitle)}</small>` : '';
    
    itemDiv.innerHTML = `
        <div class="selected-category-header" onclick="toggleCategoryExpansion('${categoryKey}')">
            <div class="selected-category-title">
                <span>${title}</span>
                ${subtitle}
            </div>
            <button class="expand-toggle" id="toggle-${categoryKey}">
                <i class="bi bi-chevron-right"></i>
            </button>
        </div>
        <div class="category-suboptions" id="suboptions-${categoryKey}">
            ${category.options.map(option => createSubOptionHTML(option, categoryKey)).join('')}
        </div>
    `;
    
    return itemDiv;
}

// Create HTML for sub-option (with support for hierarchical structure)
function createSubOptionHTML(option, categoryKey) {
    if (option.subcategory && option.suboptions) {
        const groupTitle = escapeHtml(option.name || option.value);
        // This is a hierarchical option with sub-suboptions
        return `
            <div class="suboption-group">
                <div class="suboption-group-header" onclick="toggleSubOptionGroup('${option.value}')">
                    <span>${groupTitle}</span>
                    <i class="bi bi-chevron-right expand-icon" id="expand-${option.value}"></i>
                </div>
                <div class="sub-suboptions" id="subsuboptions-${option.value}">
                    ${option.suboptions.map(subopt => `
                        ${(() => {
                            const subValue = subopt.value;
                            const subName = escapeHtml(subopt.name || subValue);
                            const subDescription = subopt.description ? escapeHtml(subopt.description) : '';
                            const subRuntime = subopt.estimated_runtime ? escapeHtml(subopt.estimated_runtime) : '';
                            const subTags = Array.isArray(subopt.tags) && subopt.tags.length ? escapeHtml(subopt.tags.join(',')) : '';
                            const titleAttr = subDescription ? ` title="${subDescription}"` : '';
                            const descAttr = subDescription ? ` data-description="${subDescription}"` : '';
                            const tagsAttr = subTags ? ` data-tags="${subTags}"` : '';
                            const runtimeHtml = subRuntime ? `<small class="suboption-meta text-muted ms-1">${subRuntime}</small>` : '';
                            return `
                        <div class="sub-suboption-item" 
                             data-value="${subValue}"${descAttr}${tagsAttr}${titleAttr}
                             onclick="toggleSubOption(event, '${subValue}', '${categoryKey}')">
                            <span class="suboption-label">${subName}</span>
                            ${runtimeHtml}
                        </div>
                            `;
                        })()}
                    `).join('')}
                </div>
            </div>
        `;
    } else {
        const optionValue = option.value;
        const optionName = escapeHtml(option.name || optionValue);
        const optionDescription = option.description ? escapeHtml(option.description) : '';
        const optionRuntime = option.estimated_runtime ? escapeHtml(option.estimated_runtime) : '';
        const optionTags = Array.isArray(option.tags) && option.tags.length ? escapeHtml(option.tags.join(',')) : '';
        const titleAttr = optionDescription ? ` title="${optionDescription}"` : '';
        const descAttr = optionDescription ? ` data-description="${optionDescription}"` : '';
        const tagsAttr = optionTags ? ` data-tags="${optionTags}"` : '';
        const runtimeHtml = optionRuntime ? `<small class="suboption-meta text-muted ms-1">${optionRuntime}</small>` : '';
        // Regular single option
        return `
            <div class="suboption-item" 
                 data-value="${optionValue}"${descAttr}${tagsAttr}${titleAttr}
                 onclick="toggleSubOption(event, '${optionValue}', '${categoryKey}')">
                <span class="suboption-label">${optionName}</span>
                ${runtimeHtml}
            </div>
        `;
    }
}

// Toggle sub-option group expansion
function toggleSubOptionGroup(optionValue) {
    const subSubOptionsElement = document.getElementById(`subsuboptions-${optionValue}`);
    const expandIcon = document.getElementById(`expand-${optionValue}`);
    
    if (subSubOptionsElement && expandIcon) {
        const isExpanded = subSubOptionsElement.classList.contains('expanded');
        
        if (isExpanded) {
            subSubOptionsElement.classList.remove('expanded');
            expandIcon.classList.remove('expanded');
        } else {
            subSubOptionsElement.classList.add('expanded');
            expandIcon.classList.add('expanded');
        }
    }
}

// Toggle category expansion in sidebar
function toggleCategoryExpansion(categoryKey) {
    const suboptionsElement = document.getElementById(`suboptions-${categoryKey}`);
    const toggleButton = document.getElementById(`toggle-${categoryKey}`);
    
    if (suboptionsElement && toggleButton) {
        const isExpanded = suboptionsElement.classList.contains('expanded');
        
        if (isExpanded) {
            suboptionsElement.classList.remove('expanded');
            toggleButton.classList.remove('expanded');
        } else {
            suboptionsElement.classList.add('expanded');
            toggleButton.classList.add('expanded');
        }
    }
}

// Toggle sub-option selection
async function toggleSubOption(event, optionValue, categoryKey) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    const triggerElement = event?.currentTarget || document.querySelector(`[data-value="${optionValue}"]`);
    await triggerAnalysisRun(optionValue, triggerElement);
}

// Get count of selected sub-options for a category
function getSelectedSubOptionsCount(categoryKey) {
    const category = ANALYSIS_CATEGORIES[categoryKey];
    let count = 0;
    
    category.options.forEach(option => {
        if (option.subcategory && option.suboptions) {
            // Count selected sub-suboptions
            count += option.suboptions.filter(subopt => selectedSubOptions.has(subopt.value)).length;
        } else {
            // Count regular options
            if (selectedSubOptions.has(option.value)) {
                count++;
            }
        }
    });
    
    return count;
}

// Update analysis selection summary
function updateEnhancedAnalysisSelection() {
    const summaryElement = document.getElementById('analysisSelectionSummary');
    const countElement = document.getElementById('selectionCount');
    const containerElement = document.getElementById('selectedItemsContainer');
    const addButton = document.getElementById('addAnalysisBtn');
    
    const totalSelected = selectedSubOptions.size;
    
    if (totalSelected === 0) {
        if (summaryElement) summaryElement.style.display = 'none';
        return;
    }
    
    if (summaryElement) summaryElement.style.display = 'block';
    if (countElement) countElement.textContent = `${totalSelected} selected`;
    if (addButton) addButton.disabled = totalSelected === 0;
    
    // Update selected items display
    if (containerElement) {
        containerElement.innerHTML = '';
        selectedSubOptions.forEach(optionValue => {
            const optionInfo = findOptionInfo(optionValue);
            if (optionInfo) {
                const badge = document.createElement('span');
                badge.className = 'selected-item-badge';
                badge.textContent = optionInfo.name || optionValue;
                if (optionInfo.description) {
                    badge.title = optionInfo.description;
                }
                containerElement.appendChild(badge);
            }
        });
    }
}

function isCategoricalAnalysisType(analysisType) {
    if (!analysisType) {
        return false;
    }

    return [
        'categorical_frequency_analysis',
        'categorical_cardinality_profile',
        'rare_category_detection',
        'categorical_visualization',
        'categorical_bar_charts',
        'categorical_pie_charts'
    ].includes(analysisType);
}

function isNumericFrequencyAnalysisType(analysisType) {
    return analysisType === 'numeric_frequency_analysis';
}

function isCrossTabAnalysisType(analysisType) {
    return analysisType === 'cross_tabulation_analysis';
}

function isCategoricalNumericAnalysisType(analysisType) {
    return analysisType === 'categorical_numeric_relationships';
}

function isTimeSeriesAnalysisType(analysisType) {
    if (!analysisType) {
        return false;
    }

    return [
        'temporal_trend_analysis',
        'seasonality_detection',
        'datetime_feature_extraction'
    ].includes(analysisType);
}

function isGeospatialAnalysisType(analysisType) {
    if (!analysisType) {
        return false;
    }

    return [
        'coordinate_system_projection_check',
        'spatial_distribution_analysis',
        'spatial_relationships_analysis',
        'spatial_data_quality_analysis',
        'geospatial_proximity_analysis'
    ].includes(analysisType);
}

function isTargetAnalysisType(analysisType) {
    return analysisType === 'target_variable_analysis';
}

function isTextAnalysisType(analysisType) {
    if (!analysisType) {
        return false;
    }

    return [
        'text_length_distribution',
        'text_token_frequency',
        'text_vocabulary_summary',
        'text_feature_engineering_profile',
        'text_nlp_profile'
    ].includes(analysisType);
}

function isNetworkAnalysisType(analysisType) {
    return analysisType === 'network_analysis';
}

function isEntityNetworkAnalysisType(analysisType) {
    return analysisType === 'entity_relationship_network';
}

function isMarketingAnalysisType(analysisType) {
    if (!analysisType || typeof window === 'undefined') {
        return false;
    }

    const config = window.MARKETING_ANALYSIS_CONFIG;
    return Boolean(config && Object.prototype.hasOwnProperty.call(config, analysisType));
}

async function gatherAvailableColumnNames() {
    if (columnInsightsData && Array.isArray(columnInsightsData.column_insights) && columnInsightsData.column_insights.length > 0) {
        return columnInsightsData.column_insights
            .filter(col => !col.dropped)
            .map(col => col.name);
    }

    if (window.currentDataFrame && Array.isArray(window.currentDataFrame.columns) && window.currentDataFrame.columns.length > 0) {
        return [...window.currentDataFrame.columns];
    }

    try {
        const storedColumns = sessionStorage.getItem('datasetColumns');
        if (storedColumns) {
            const parsed = JSON.parse(storedColumns);
            if (Array.isArray(parsed)) {
                return parsed;
            }
        }
    } catch (storageError) {
        console.warn('Unable to parse stored dataset columns for marketing modal:', storageError);
    }

    try {
        await loadColumnInsights();
        if (columnInsightsData && Array.isArray(columnInsightsData.column_insights) && columnInsightsData.column_insights.length > 0) {
            return columnInsightsData.column_insights.map(col => col.name);
        }
    } catch (error) {
        console.warn('Column insight prefetch failed for marketing modal:', error);
    }

    return [];
}

async function prepareMarketingAnalysisConfiguration(analysisType) {
    currentMarketingCellId = '';
    marketingModalConfirmed = false;
    const columnCandidates = await gatherAvailableColumnNames();
    const sanitizedColumns = Array.from(new Set((columnCandidates || []).filter(Boolean)));

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipMarketingModal: true,
        prefetchedColumns: sanitizedColumns
    });

    if (!cellId) {
        showNotification('Unable to prepare marketing analysis cell. Please try again.', 'error');
        return;
    }

    currentMarketingCellId = cellId;

    try {
        showMarketingAnalysisModal(analysisType, sanitizedColumns);
    } catch (error) {
        console.error('Failed to open marketing analysis modal:', error);
        showNotification('Unable to open marketing configuration modal. Please try again.', 'error');
    }
}

function attachMarketingModalLifecycleHandlers() {
    const modalElement = document.getElementById('marketingAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleMarketingModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleMarketingModalHidden() {
    if (marketingModalConfirmed) {
        marketingModalConfirmed = false;
        return;
    }

    if (!currentMarketingCellId) {
        currentAnalysisType = '';
        selectedColumnMapping = {};
        return;
    }

    const pendingCell = document.querySelector(`[data-cell-id="${currentMarketingCellId}"]`);
    if (pendingCell) {
        pendingCell.remove();
        showNotification('Marketing analysis cancelled.', 'info');
        updateAnalysisResultsPlaceholder();
    }

    currentMarketingCellId = '';
    currentAnalysisType = '';
    selectedColumnMapping = {};
}

async function triggerAnalysisRun(analysisType, anchorElement) {
    if (!analysisType) {
        return;
    }

    const element = anchorElement || document.querySelector(`.analysis-option[data-value="${analysisType}"]`);

    const categoryKey = getAnalysisCategory(analysisType);
    if (isCategoryLocked(categoryKey)) {
        const message = categoryKey === DATA_QUALITY_CATEGORY_KEY
            ? 'Data quality checks are disabled after preprocessing is applied.'
            : 'Apply preprocessing before running this analysis.';
        showNotification(message, 'warning');
        if (element) {
            element.classList.remove('is-running');
            element.removeAttribute('aria-busy');
        }
        return;
    }

    if (element) {
        element.classList.add('is-running');
        element.setAttribute('aria-busy', 'true');
    }

    try {
        if (isMarketingAnalysisType(analysisType)) {
            await prepareMarketingAnalysisConfiguration(analysisType);
        } else if (isNumericFrequencyAnalysisType(analysisType)) {
            await prepareNumericFrequencyConfiguration(analysisType);
        } else if (isCategoricalAnalysisType(analysisType)) {
            await prepareCategoricalAnalysisConfiguration(analysisType);
        } else if (isCategoricalNumericAnalysisType(analysisType)) {
            await prepareCategoricalNumericAnalysisConfiguration(analysisType);
        } else if (isCrossTabAnalysisType(analysisType)) {
            await prepareCrossTabAnalysisConfiguration(analysisType);
        } else if (isGeospatialAnalysisType(analysisType)) {
            await prepareGeospatialAnalysisConfiguration(analysisType);
        } else if (isTimeSeriesAnalysisType(analysisType)) {
            await prepareTimeSeriesAnalysisConfiguration(analysisType);
        } else if (isTargetAnalysisType(analysisType)) {
            await prepareTargetAnalysisConfiguration(analysisType);
        } else if (isTextAnalysisType(analysisType)) {
            await prepareTextAnalysisConfiguration(analysisType);
        } else if (isNetworkAnalysisType(analysisType)) {
            await prepareNetworkAnalysisConfiguration(analysisType);
        } else if (isEntityNetworkAnalysisType(analysisType)) {
            await prepareEntityNetworkAnalysisConfiguration(analysisType);
        } else {
            await addSingleAnalysisCell(analysisType);
        }
    } catch (error) {
        console.error(`Failed to trigger analysis for ${analysisType}:`, error);
        const analysisName = typeof getAnalysisTypeName === 'function' ? getAnalysisTypeName(analysisType) : analysisType;
        showNotification(`Unable to start ${analysisName}`, 'error');
    } finally {
        if (element) {
            element.classList.remove('is-running');
            element.removeAttribute('aria-busy');
        }
    }
}


// Find option info by value
function findOptionInfo(optionValue) {
    if (!optionValue) {
        return null;
    }

    for (const [categoryKey, category] of Object.entries(ANALYSIS_CATEGORIES || {})) {
        if (!category || !Array.isArray(category.options)) {
            continue;
        }

        for (const option of category.options) {
            if (option.value === optionValue) {
                return { ...option, categoryKey };
            }

            if (option.subcategory && Array.isArray(option.suboptions)) {
                const subOption = option.suboptions.find(sub => sub.value === optionValue);
                if (subOption) {
                    return {
                        ...subOption,
                        categoryKey,
                        parent: option.value,
                        description: subOption.description || option.description || '',
                        estimated_runtime: subOption.estimated_runtime || option.estimated_runtime || '1-5 seconds',
                        tags: Array.isArray(subOption.tags) ? subOption.tags : option.tags,
                    };
                }
            }
        }
    }
    return null;
}

// Setup enhanced event listeners
function setupEnhancedEventListeners() {
    // Back to grid button
    const clearCategoriesBtn = document.getElementById('clearCategoriesBtn');
    if (clearCategoriesBtn) {
        clearCategoriesBtn.addEventListener('click', () => {
            selectedCategories.clear();
            updateSidebar();
            renderAnalysisCardsGrid();
        });
    }
    
    // Clear all selections
    const clearAllBtn = document.getElementById('clearAllBtn');
    if (clearAllBtn) {
        clearAllBtn.addEventListener('click', () => {
            selectedCategories.clear();
            selectedSubOptions.clear();
            selectedAnalysisTypes.clear();
            updateSidebar();
            updateEnhancedAnalysisSelection();
            renderAnalysisCardsGrid();
        });
    }
    
    // Add analysis button
    const addAnalysisBtn = document.getElementById('addAnalysisBtn');
    if (addAnalysisBtn) {
        addAnalysisBtn.addEventListener('click', addSelectedAnalysisTypes);
    }
}

// Compatibility function for enhanced system
async function addSelectedAnalysisTypes() {
    console.log('addSelectedAnalysisTypes called via enhanced system');
    return await addSelectedAnalysisCells();
}

// Make enhanced functions globally accessible
window.toggleCategoryExpansion = toggleCategoryExpansion;
window.toggleSubOption = toggleSubOption;
window.toggleSubOptionGroup = toggleSubOptionGroup;

// Toggle analysis selection (multi-select)
async function toggleAnalysisSelection(element) {
    if (!element) {
        return;
    }

    const value = element.getAttribute('data-value');
    console.log('Triggering analysis from grid selection:', value);
    await triggerAnalysisRun(value, element);
}

// Update visibility of selected analyses container
function updateSelectedAnalysesVisibility() {
    const container = document.getElementById('selectedAnalysesContainer');
    if (container) {
        container.style.display = selectedAnalysisTypes.size > 0 ? 'block' : 'none';
    }
}

// Add a badge for selected analysis
function addBadge(value, name) {
    const container = document.getElementById('selectedAnalysesDisplay');
    if (!container) return;
    
    const badge = document.createElement('span');
    badge.className = 'badge bg-primary me-1 mb-1';
    badge.setAttribute('data-value', value);
    badge.innerHTML = `
        ${name}
        <button type="button" class="btn-close btn-close-white ms-1 badge-close-btn" onclick="removeBadgeAndDeselect('${value}')"></button>
    `;
    
    container.appendChild(badge);
}

// Remove badge for deselected analysis
function removeBadge(value) {
    const badge = document.querySelector(`[data-value="${value}"].badge`);
    if (badge) {
        badge.remove();
    }
}

// Remove badge and deselect analysis (called from badge X button)
function removeBadgeAndDeselect(value) {
    selectedAnalysisTypes.delete(value);
    removeBadge(value);
    
    // Also remove visual selection from grid
    const option = document.querySelector(`.analysis-option[data-value="${value}"]`);
    if (option) {
        option.classList.remove('selected');
    }
    
    updateAddButton();
    updateSelectedAnalysesVisibility();
}

// Legacy analysis type bulk selection (removed - keeping for compatibility)
function selectCategory(analysisValues) {
    // Bulk selection removed - each card now handles individual selection
    console.log('selectCategory called but bulk selection has been removed');
}

// Legacy select all analyses (removed - keeping for compatibility)  
function selectAllAnalyses() {
    // Bulk selection removed - each card now handles individual selection
    console.log('selectAllAnalyses called but bulk selection has been removed');
}

// Clear all selections
function clearAllSelections() {
    console.log('Clearing all analysis selections');
    
    // Clear all selections
    document.querySelectorAll('.analysis-option').forEach(opt => {
        opt.classList.remove('selected');
    });
    
    // Clear badges
    const container = document.getElementById('selectedAnalysesDisplay');
    if (container) {
        container.innerHTML = '';
    }
    
    // Clear stored selections
    selectedAnalysisTypes.clear();
    
    updateAddButton();
    updateSelectedAnalysesVisibility();
}

// Update the add button state (legacy - button removed)
function updateAddButton() {
    // Button has been removed from UI - this is now a no-op
    // Keeping function for compatibility with existing calls
}

// Legacy functions for compatibility
function selectAnalysisType(target, displayName) {
    if (!target) {
        return;
    }

    if (typeof target === 'string') {
        const element = document.querySelector(`.analysis-option[data-value="${target}"]`);
        triggerAnalysisRun(target, element);
        return;
    }

    // Redirect to immediate execution for DOM elements
    toggleAnalysisSelection(target);
}

function clearSelection() {
    // Redirect to new multi-select system
    clearAllSelections();
}

function onAnalysisTypeChange() {
    // This is now handled by the interactive grid
    console.log('onAnalysisTypeChange called - using interactive grid instead');
}

// Add selected analysis results (supports multiple selections)


async function addSelectedAnalysisCells() {
    console.log('addSelectedAnalysisCells called with selections:', selectedAnalysisTypes);
    
    if (selectedAnalysisTypes.size === 0) {
        showNotification('Please select analysis types from the categories above', 'info');
        return;
    }
    
    // Check if any marketing analyses are selected
    const marketingAnalyses = [];
    const nonMarketingAnalyses = [];
    
    for (const analysisType of selectedAnalysisTypes) {
        if (MARKETING_ANALYSIS_CONFIG[analysisType]) {
            marketingAnalyses.push(analysisType);
        } else {
            nonMarketingAnalyses.push(analysisType);
        }
    }
    
    // If there are marketing analyses, handle them specially
    if (marketingAnalyses.length > 0) {
        if (marketingAnalyses.length === 1 && selectedAnalysisTypes.size === 1) {
            // Single marketing analysis - show modal directly
            await prepareMarketingAnalysisConfiguration(marketingAnalyses[0]);
        } else {
            // Multiple analyses with at least one marketing - show informative message
            showNotification(
                `Marketing analyses (${marketingAnalyses.map(a => getAnalysisTypeName(a)).join(', ')}) require column mapping configuration. Please add them individually for the best experience.`,
                'info',
                8000
            );
            
            // Process only non-marketing analyses for bulk add
            if (nonMarketingAnalyses.length > 0) {
                let addedCount = 0;
                const errors = [];
                
                for (const analysisType of nonMarketingAnalyses) {
                    try {
                        console.log(`Adding analysis result for: ${analysisType}`);
                        await addSingleAnalysisCell(analysisType);
                        addedCount++;
                    } catch (error) {
                        console.error(`Failed to add analysis result for ${analysisType}:`, error);
                        errors.push(`${getAnalysisTypeName(analysisType)}: ${error.message}`);
                    }
                }
                
                if (addedCount > 0) {
                    showNotification(`Added ${addedCount} non-marketing analysis result(s). Add marketing analyses individually.`, 'success');
                }
                
                if (errors.length > 0) {
                    showNotification(`Some analyses failed to add: ${errors.join(', ')}`, 'warning');
                }
            }
        }
        
        // Don't clear selections if marketing analyses need to be added individually
        if (marketingAnalyses.length === 1 && selectedAnalysisTypes.size === 1) {
            clearAllSelections();
        }
        
        return;
    }
    
    // No marketing analyses - process normally
    let addedCount = 0;
    const errors = [];
    
    // Process each selected analysis type
    for (const analysisType of selectedAnalysisTypes) {
        try {
            console.log(`Adding analysis result for: ${analysisType}`);
            await addSingleAnalysisCell(analysisType);
            addedCount++;
        } catch (error) {
            console.error(`Failed to add analysis result for ${analysisType}:`, error);
            errors.push(`${getAnalysisTypeName(analysisType)}: ${error.message}`);
        }
    }
    
    // Show results
    if (addedCount > 0) {
        showNotification(`Added ${addedCount} analysis result(s)`, 'success');
        
        // Clear selections after successful addition
        clearAllSelections();
    }
    
    if (errors.length > 0) {
        showNotification(`Some analyses failed to add: ${errors.join(', ')}`, 'warning');
    }
}

(function initializeEDAChatBridge() {
    const existingBridge = typeof window.EDAChatBridge === 'object' && window.EDAChatBridge !== null
        ? window.EDAChatBridge
        : {};

    const analysisCells = existingBridge.__analysisCells instanceof Map ? existingBridge.__analysisCells : new Map();
    const customCells = existingBridge.__customCells instanceof Map ? existingBridge.__customCells : new Map();

    let activeContext = existingBridge.__activeContext || null;
    let notebookMeta = existingBridge.__notebookMeta || null;
    let lastUpdate = existingBridge.lastUpdate || null;

    const MAX_TEXT_LENGTH = 6000;
    const MAX_PREVIEW_LENGTH = 180;
    const MAX_TABLE_ROWS = 10;
    const MAX_TABLES = 3;
    const MAX_CHARTS = 4;

    function truncateText(value, maxLength = MAX_TEXT_LENGTH) {
        if (typeof value !== 'string') {
            return value;
        }
        if (value.length <= maxLength) {
            return value;
        }
        return `${value.slice(0, maxLength)}…`;
    }

    function buildMetricSummary(structuredResults) {
        if (!Array.isArray(structuredResults) || structuredResults.length === 0) {
            return null;
        }
        const primaryResult = structuredResults[0] || {};
        const metrics = Array.isArray(primaryResult.metrics) ? primaryResult.metrics : [];
        if (metrics.length === 0) {
            return null;
        }
        const metricHighlights = metrics.slice(0, 3).map(metric => {
            const label = metric.label || metric.name || 'Metric';
            const value = metric.value;
            const unit = metric.unit ? ` ${metric.unit}` : '';
            if (typeof value === 'number') {
                return `${label}: ${Number(value.toFixed(4))}${unit}`;
            }
            return `${label}: ${value}${unit}`;
        });
        return metricHighlights.join('; ');
    }

    function buildInsightSummary(structuredResults) {
        if (!Array.isArray(structuredResults) || structuredResults.length === 0) {
            return null;
        }
        const primaryResult = structuredResults[0] || {};
        const insights = Array.isArray(primaryResult.insights) ? primaryResult.insights : [];
        if (insights.length === 0) {
            return null;
        }
        const highlights = insights
            .filter(insight => insight && insight.text)
            .slice(0, 2)
            .map(insight => {
                const level = insight.level ? `${insight.level.toUpperCase()}: ` : '';
                return `${level}${truncateText(insight.text, 300)}`;
            });
        return highlights.length ? highlights.join(' | ') : null;
    }

    function buildAnalysisLLMSummary(entry) {
        if (!entry) {
            return null;
        }
        const parts = [];
        const metaSummary = truncateText(entry.metaSummary, 600);
        if (metaSummary) {
            parts.push(metaSummary);
        }

        const metricSummary = buildMetricSummary(entry.structuredResults);
        if (metricSummary) {
            parts.push(`Key metrics: ${metricSummary}`);
        }

        const insightSummary = buildInsightSummary(entry.structuredResults);
        if (insightSummary) {
            parts.push(`Insights: ${insightSummary}`);
        }

        const request = entry.request || {};
        if (!parts.length && Array.isArray(request.selected_columns) && request.selected_columns.length) {
            const preview = request.selected_columns.slice(0, 5);
            const remainder = request.selected_columns.length - preview.length;
            parts.push(`Columns analysed: ${preview.join(', ')}${remainder > 0 ? ` (+${remainder})` : ''}`);
        }

        const legacyStdout = entry.legacyOutput?.stdout;
        if (legacyStdout) {
            parts.push(`Console: ${truncateText(legacyStdout, 400)}`);
        }

        const warnings = entry.responsePreview?.warnings;
        if (warnings) {
            const warningText = Array.isArray(warnings) ? warnings.join(' | ') : warnings;
            parts.push(`Warnings: ${truncateText(warningText, 400)}`);
        }

        return parts.length ? parts.join('\n') : null;
    }

    function buildCustomLLMSummary(entry) {
        if (!entry) {
            return null;
        }
        const parts = [];
        if (entry.textOutput) {
            parts.push(truncateText(entry.textOutput, 600));
        }
        if (entry.error) {
            parts.push(`Error: ${truncateText(entry.error, 400)}`);
        }
        if (entry.plotCount) {
            parts.push(`Plots generated: ${entry.plotCount}`);
        }
        if (entry.responsePreview?.rate_limit_info) {
            parts.push('Rate limit info available for review.');
        }
        return parts.length ? parts.join('\n') : null;
    }

    function buildNotebookSummary() {
        const dataset = getDatasetInfo();
        const analysisCount = analysisCells.size;
        const customCount = customCells.size;
        const parts = [];

        if (dataset.name) {
            parts.push(`Dataset: ${dataset.name}`);
        }
        if (analysisCount) {
            parts.push(`Analyses run: ${analysisCount}`);
        }
        if (customCount) {
            parts.push(`Custom cells run: ${customCount}`);
        }
        if (activeContext?.cellId) {
            parts.push(`Focus on ${activeContext.scope} cell ${activeContext.cellId}`);
        }
        if (lastUpdate) {
            try {
                const timestamp = new Date(lastUpdate);
                if (!Number.isNaN(timestamp.valueOf())) {
                    parts.push(`Context updated ${timestamp.toLocaleString()}`);
                }
            } catch (e) {
                // ignore timestamp parsing issues
            }
        }

        return parts.join(' • ');
    }

    function buildAnalysisHighlights(limit = 4) {
        const sortedEntries = Array.from(analysisCells.values()).sort((a, b) => {
            const aTime = a.completedAt || a.startedAt || '';
            const bTime = b.completedAt || b.startedAt || '';
            if (aTime === bTime) {
                return 0;
            }
            return aTime > bTime ? -1 : 1;
        });
        return sortedEntries.slice(0, limit).map(entry => ({
            cellId: entry.cellId,
            analysisType: entry.analysisType,
            analysisName: entry.analysisName,
            status: entry.status,
            summary: truncateText(entry.llmSummary || buildAnalysisLLMSummary(entry), 800)
        }));
    }

    function buildCustomHighlights(limit = 3) {
        const sortedEntries = Array.from(customCells.values()).sort((a, b) => {
            const aTime = a.completedAt || a.startedAt || '';
            const bTime = b.completedAt || b.startedAt || '';
            if (aTime === bTime) {
                return 0;
            }
            return aTime > bTime ? -1 : 1;
        });
        return sortedEntries.slice(0, limit).map(entry => ({
            cellId: entry.cellId,
            status: entry.status,
            summary: truncateText(entry.llmSummary || buildCustomLLMSummary(entry), 600)
        }));
    }

    function sanitizeStructuredResults(results) {
        if (!Array.isArray(results)) {
            return [];
        }
        return results.slice(0, 6).map(result => {
            const sanitized = {
                analysis_id: result.analysis_id || result.id || null,
                title: result.title || null,
                summary: truncateText(result.summary, 1000),
                status: result.status || 'success'
            };

            if (Array.isArray(result.metrics)) {
                sanitized.metrics = result.metrics.slice(0, 6).map(metric => ({
                    label: metric.label || metric.name || '',
                    value: typeof metric.value === 'number' ? metric.value : truncateText(String(metric.value), 200),
                    unit: metric.unit || null,
                    description: truncateText(metric.description, 400),
                    trend: metric.trend || null
                }));
            }

            if (Array.isArray(result.insights)) {
                sanitized.insights = result.insights.slice(0, 8).map(insight => ({
                    level: insight.level || 'info',
                    text: truncateText(insight.text, 600)
                }));
            }

            if (Array.isArray(result.tables)) {
                sanitized.tables = result.tables.slice(0, MAX_TABLES).map(table => ({
                    title: table.title || null,
                    description: truncateText(table.description, 400),
                    columns: Array.isArray(table.columns) ? table.columns.slice(0, 12) : [],
                    rows: Array.isArray(table.rows)
                        ? table.rows.slice(0, MAX_TABLE_ROWS).map(row => {
                            const rowCopy = {};
                            Object.entries(row || {}).forEach(([key, value]) => {
                                rowCopy[key] = typeof value === 'number' ? value : truncateText(String(value), 300);
                            });
                            return rowCopy;
                        })
                        : []
                }));
            }

            if (Array.isArray(result.charts)) {
                sanitized.charts = result.charts.slice(0, MAX_CHARTS).map(chart => ({
                    title: chart.title || null,
                    description: truncateText(chart.description, 400),
                    hasImage: Boolean(chart.image),
                    imagePreview: chart.image ? truncateText(chart.image, MAX_PREVIEW_LENGTH) : null
                }));
            }

            if (result.details && typeof result.details === 'object') {
                try {
                    sanitized.details = JSON.parse(JSON.stringify(result.details));
                } catch (detailError) {
                    sanitized.details = result.details;
                }
            }

            return sanitized;
        });
    }

    function sanitizeLegacyOutput(output) {
        if (!output || typeof output !== 'object') {
            return null;
        }

        const sanitized = {};
        if (output.stdout) {
            sanitized.stdout = truncateText(output.stdout, MAX_TEXT_LENGTH);
        }
        if (output.stderr) {
            sanitized.stderr = truncateText(output.stderr, 2000);
        }
        if (Array.isArray(output.data_frames) && output.data_frames.length > 0) {
            sanitized.data_frames = output.data_frames.slice(0, 2).map(frame => ({
                name: frame.name || null,
                preview: frame.html ? truncateText(frame.html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim(), 1000) : null
            }));
        }
        if (Array.isArray(output.plots) && output.plots.length > 0) {
            sanitized.plots = {
                count: output.plots.length,
                preview: truncateText(String(output.plots[0]), MAX_PREVIEW_LENGTH)
            };
        }
        return sanitized;
    }

    function sanitizeRequestPayload(payload) {
        if (!payload || typeof payload !== 'object') {
            return null;
        }
        const sanitized = {
            analysis_ids: Array.isArray(payload.analysis_ids) ? [...payload.analysis_ids] : null,
            source_id: payload.source_id || null
        };
        if (Array.isArray(payload.selected_columns)) {
            sanitized.selected_columns = payload.selected_columns.slice(0, 100);
        }
        if (payload.column_mapping && typeof payload.column_mapping === 'object') {
            sanitized.column_mapping = payload.column_mapping;
        }
        if (payload.preprocessing && typeof payload.preprocessing === 'object') {
            sanitized.preprocessing = payload.preprocessing;
        }
        if (payload.analysis_metadata && typeof payload.analysis_metadata === 'object') {
            sanitized.analysis_metadata = payload.analysis_metadata;
        }
        return sanitized;
    }

    function sanitizePreprocessingReport(report) {
        if (!report || typeof report !== 'object') {
            return null;
        }
        const { applied_steps, dropped_columns, imputations, summary } = report;
        return {
            summary: truncateText(summary || '', 1000),
            applied_steps: Array.isArray(applied_steps) ? applied_steps.slice(0, 10) : null,
            dropped_columns: Array.isArray(dropped_columns) ? dropped_columns.slice(0, 20) : null,
            imputations: Array.isArray(imputations) ? imputations.slice(0, 10) : null
        };
    }

    function sanitizeCustomResponse(response) {
        if (!response || typeof response !== 'object') {
            return null;
        }
        const sanitized = {
            success: Boolean(response.success)
        };
        if (response.error) {
            sanitized.error = truncateText(response.error, 800);
        }
        if (response.output && typeof response.output === 'string') {
            sanitized.stdout = truncateText(response.output, 600);
        } else if (response.output && typeof response.output === 'object') {
            sanitized.stdout = truncateText(response.output.stdout || '', 600);
            if (response.output.stderr) {
                sanitized.stderr = truncateText(response.output.stderr, 400);
            }
        }
        if (Array.isArray(response.plots)) {
            sanitized.plot_count = response.plots.length;
        }
        if (response.rate_limit_info) {
            sanitized.rate_limit_info = response.rate_limit_info;
        }
        return sanitized;
    }

    function buildResponsePreview(response) {
        if (!response || typeof response !== 'object') {
            return null;
        }
        const preview = {
            success: Boolean(response.success)
        };
        if (response.analysis_count !== undefined) {
            preview.analysis_count = response.analysis_count;
        }
        if (response.metadata && typeof response.metadata === 'object') {
            const executionTime = response.metadata.execution_time || response.metadata.runtime_seconds;
            if (executionTime !== undefined) {
                preview.execution_time = executionTime;
            }
        }
        if (response.warnings) {
            preview.warnings = Array.isArray(response.warnings) ? response.warnings.slice(0, 5) : truncateText(String(response.warnings), 500);
        }
        return preview;
    }

    function getSelectedColumnsSnapshot() {
        try {
            if (typeof selectedColumns !== 'undefined' && selectedColumns && typeof selectedColumns.size === 'number') {
                return Array.from(selectedColumns);
            }
        } catch (e) {
            console.warn('Unable to snapshot selected columns:', e);
        }
        return null;
    }

    function getDatasetInfo() {
        const sourceId = typeof initSourceId === 'function' ? initSourceId() : (existingBridge.sourceId || null);
        let name = null;
        const nameEl = document.getElementById('dataset-name') || document.getElementById('datasetName');
        if (nameEl && nameEl.textContent) {
            name = nameEl.textContent.trim();
        } else if (window.currentDataset?.name) {
            name = window.currentDataset.name;
        }
        return {
            sourceId,
            name,
            info: window.currentDataset?.info || null
        };
    }

    function getPreprocessingContext() {
        try {
            const applied = typeof preprocessingApplied !== 'undefined' ? Boolean(preprocessingApplied) : null;
            const dirty = typeof preprocessingDirty !== 'undefined' ? Boolean(preprocessingDirty) : null;
            const summaryFn = typeof buildPreprocessingSummary === 'function' ? buildPreprocessingSummary : null;
            const summary = summaryFn && lastPreprocessingReport ? summaryFn(lastPreprocessingReport) : null;
            return {
                applied,
                pendingChanges: dirty,
                summary,
                lastReport: sanitizePreprocessingReport(lastPreprocessingReport)
            };
        } catch (error) {
            console.warn('Unable to build preprocessing context for LLM chat:', error);
            return null;
        }
    }

    function emitContextUpdate() {
        lastUpdate = new Date().toISOString();
        if (window.EDAChatBridge) {
            window.EDAChatBridge.lastUpdate = lastUpdate;
            window.EDAChatBridge.__analysisCells = analysisCells;
            window.EDAChatBridge.__customCells = customCells;
            window.EDAChatBridge.__activeContext = activeContext;
            window.EDAChatBridge.__notebookMeta = notebookMeta;
        }
        try {
            const context = buildContext();
            window.dispatchEvent(new CustomEvent('eda-llm-context-updated', { detail: context }));
        } catch (eventError) {
            console.warn('Failed to dispatch EDA LLM context event:', eventError);
        }
    }

    function buildContext() {
        return {
            type: 'eda_notebook',
            dataset: getDatasetInfo(),
            preprocessing: getPreprocessingContext(),
            selectedColumns: getSelectedColumnsSnapshot(),
            activeContext,
            lastUpdate,
            analysisCells: Array.from(analysisCells.values()),
            customAnalyses: Array.from(customCells.values()),
            analysisHighlights: buildAnalysisHighlights(),
            customHighlights: buildCustomHighlights(),
            summary: buildNotebookSummary(),
            meta: notebookMeta || {}
        };
    }

    function ensureAnalysisEntry(cellId) {
        const existingEntry = analysisCells.get(cellId);
        if (existingEntry) {
            return existingEntry;
        }
        const entry = {
            cellId,
            scope: 'analysis',
            status: 'queued',
            runCount: 0
        };
        analysisCells.set(cellId, entry);
        return entry;
    }

    function ensureCustomEntry(cellId) {
        const existingEntry = customCells.get(cellId);
        if (existingEntry) {
            return existingEntry;
        }
        const entry = {
            cellId,
            scope: 'custom',
            status: 'queued',
            runCount: 0
        };
        customCells.set(cellId, entry);
        return entry;
    }

    function recordAnalysisPending(cellId, payload = {}) {
        const entry = ensureAnalysisEntry(cellId);
        const now = new Date().toISOString();
        const updated = {
            ...entry,
            analysisType: payload.analysisType || entry.analysisType || null,
            analysisName: payload.analysisName || entry.analysisName || (payload.analysisType ? getAnalysisTypeName(payload.analysisType) : null),
            status: 'running',
            startedAt: now,
            completedAt: null,
            request: sanitizeRequestPayload(payload.requestPayload) || entry.request || null,
            selectedColumnsSnapshot: Array.isArray(payload.selectedColumns) ? payload.selectedColumns : entry.selectedColumnsSnapshot || getSelectedColumnsSnapshot() || null
        };
        if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        analysisCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function recordAnalysisSuccess(cellId, payload = {}) {
        const entry = ensureAnalysisEntry(cellId);
        const updated = {
            ...entry,
            analysisType: payload.analysisType || entry.analysisType || null,
            analysisName: payload.analysisName || entry.analysisName || (payload.analysisType ? getAnalysisTypeName(payload.analysisType) : null),
            status: 'success',
            completedAt: new Date().toISOString(),
            runCount: (entry.runCount || 0) + 1,
            structuredResults: sanitizeStructuredResults(payload.structuredResults),
            legacyOutput: sanitizeLegacyOutput(payload.legacyOutput),
            request: sanitizeRequestPayload(payload.requestPayload) || entry.request || null,
            selectedColumnsSnapshot: Array.isArray(payload.selectedColumns) ? payload.selectedColumns : entry.selectedColumnsSnapshot || getSelectedColumnsSnapshot() || null,
            metaSummary: payload.metaSummary ? truncateText(payload.metaSummary, 1200) : entry.metaSummary || null,
            responsePreview: buildResponsePreview(payload.rawResponse)
        };
        if (!updated.startedAt) {
            updated.startedAt = entry.startedAt || new Date().toISOString();
        }
        const summary = buildAnalysisLLMSummary(updated);
        if (summary) {
            updated.llmSummary = summary;
        } else if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        analysisCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function recordAnalysisError(cellId, payload = {}) {
        const entry = ensureAnalysisEntry(cellId);
        const updated = {
            ...entry,
            analysisType: payload.analysisType || entry.analysisType || null,
            analysisName: payload.analysisName || entry.analysisName || (payload.analysisType ? getAnalysisTypeName(payload.analysisType) : null),
            status: 'error',
            completedAt: new Date().toISOString(),
            error: truncateText(payload.errorMessage || payload.error || 'Analysis failed.', 1200),
            request: sanitizeRequestPayload(payload.requestPayload) || entry.request || null,
            selectedColumnsSnapshot: Array.isArray(payload.selectedColumns) ? payload.selectedColumns : entry.selectedColumnsSnapshot || getSelectedColumnsSnapshot() || null
        };
        if (!updated.startedAt) {
            updated.startedAt = entry.startedAt || new Date().toISOString();
        }
        const errorSummary = truncateText(updated.error, 600);
        if (errorSummary) {
            updated.llmSummary = errorSummary;
        } else if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        analysisCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function removeAnalysisCell(cellId) {
        if (analysisCells.has(cellId)) {
            analysisCells.delete(cellId);
            if (activeContext?.scope === 'analysis' && activeContext.cellId === cellId) {
                activeContext = null;
            }
            emitContextUpdate();
        }
    }

    function recordCustomPending(cellId, payload = {}) {
        const entry = ensureCustomEntry(cellId);
        const now = new Date().toISOString();
        const updated = {
            ...entry,
            status: 'running',
            startedAt: now,
            completedAt: null,
            code: truncateText(payload.code || entry.code || '', MAX_TEXT_LENGTH)
        };
        if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        customCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function recordCustomSuccess(cellId, payload = {}) {
        const entry = ensureCustomEntry(cellId);
        const updated = {
            ...entry,
            status: 'success',
            completedAt: new Date().toISOString(),
            runCount: (entry.runCount || 0) + 1,
            code: truncateText(payload.code || entry.code || '', MAX_TEXT_LENGTH),
            textOutput: truncateText(payload.textOutput || '', MAX_TEXT_LENGTH),
            plotCount: Array.isArray(payload.plots) ? payload.plots.length : entry.plotCount || 0,
            plotsPreview: Array.isArray(payload.plots) && payload.plots.length > 0 ? truncateText(String(payload.plots[0]), MAX_PREVIEW_LENGTH) : entry.plotsPreview || null,
            responsePreview: sanitizeCustomResponse(payload.rawResponse)
        };
        if (!updated.startedAt) {
            updated.startedAt = entry.startedAt || new Date().toISOString();
        }
        const summary = buildCustomLLMSummary(updated);
        if (summary) {
            updated.llmSummary = summary;
        } else if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        customCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function recordCustomError(cellId, payload = {}) {
        const entry = ensureCustomEntry(cellId);
        const updated = {
            ...entry,
            status: 'error',
            completedAt: new Date().toISOString(),
            code: truncateText(payload.code || entry.code || '', MAX_TEXT_LENGTH),
            error: truncateText(payload.errorMessage || payload.error || 'Execution failed.', 2000),
            responsePreview: sanitizeCustomResponse(payload.rawResponse)
        };
        if (!updated.startedAt) {
            updated.startedAt = entry.startedAt || new Date().toISOString();
        }
        const summary = buildCustomLLMSummary(updated);
        if (summary) {
            updated.llmSummary = summary;
        } else if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        customCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function removeCustomCell(cellId) {
        if (customCells.has(cellId)) {
            customCells.delete(cellId);
            if (activeContext?.scope === 'custom' && activeContext.cellId === cellId) {
                activeContext = null;
            }
            emitContextUpdate();
        }
    }

    function setActiveCell(cellId, scope = 'analysis') {
        activeContext = { scope, cellId, timestamp: new Date().toISOString() };
        emitContextUpdate();
    }

    function setNotebookMeta(meta) {
        if (!meta || typeof meta !== 'object') {
            return;
        }
        notebookMeta = { ...(notebookMeta || {}), ...meta };
        emitContextUpdate();
    }

    window.EDAChatBridge = Object.assign(existingBridge, {
        getContext: buildContext,
        setActiveCell,
        recordAnalysisPending,
        recordAnalysisSuccess,
        recordAnalysisError,
        removeAnalysisCell,
        recordCustomPending,
        recordCustomSuccess,
        recordCustomError,
        removeCustomCell,
        setNotebookMeta,
        lastUpdate,
        __analysisCells: analysisCells,
        __customCells: customCells,
        __activeContext: activeContext,
        __notebookMeta: notebookMeta
    });
})();

function openAskAIForCell(cellId) {
    if (window.EDAChatBridge && typeof window.EDAChatBridge.setActiveCell === 'function') {
        window.EDAChatBridge.setActiveCell(cellId, 'analysis');
    }

    if (window.LLMChat && typeof window.LLMChat.showModal === 'function') {
        window.LLMChat.showModal();
    } else {
        showNotification('AI assistant is currently unavailable.', 'warning');
    }
}

// Add a single analysis result card (extracted from original addAnalysisCell function)
async function addSingleAnalysisCell(analysisType, options = {}) {
    console.log('Adding single analysis result for:', analysisType);

    const {
        skipMarketingModal = false,
        skipCategoricalModal = false,
        skipNumericModal = false,
        skipCategoricalNumericModal = false,
        skipCrossTabModal = false,
    skipGeospatialModal = false,
        skipTimeSeriesModal = false,
        skipTextModal = false,
        skipTargetModal = false,
        skipNetworkModal = false,
        skipEntityNetworkModal = false,
        prefetchedColumns = [],
        analysisOptions = {}
    } = options || {};

    const cellId = `cell-${cellCounter++}`;
    console.log('Result card ID:', cellId);

    // Create and add cell immediately (shows loading state)
    const cellHTML = createAnalysisCell(cellId, analysisType);
    console.log('Cell HTML created');

    const cellsContainer = document.getElementById('notebookCells');
    console.log('Cells container:', cellsContainer);

    if (cellsContainer) {
        if (cellsContainer.firstElementChild) {
            cellsContainer.insertAdjacentHTML('afterbegin', cellHTML);
            console.log('Cell HTML inserted at top');
        } else {
            cellsContainer.insertAdjacentHTML('beforeend', cellHTML);
            console.log('Cell HTML inserted as first item');
        }
        updateAnalysisResultsPlaceholder();
    } else {
        console.error('notebookCells container not found!');
        return null;
    }

    // Scroll to new cell
    const newCell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (newCell) {
        newCell.scrollIntoView({ behavior: 'smooth', block: 'center' });
        console.log('Scrolled to new cell');
    } else {
        console.error('New cell not found for scrolling:', cellId);
    }

    // Generate and run analysis
    try {
        if (isMarketingAnalysisType(analysisType)) {
            console.log('Marketing analysis detected, preparing configuration for:', analysisType);

            let columns = Array.isArray(prefetchedColumns) ? [...prefetchedColumns] : [];

            if (columns.length === 0) {
                if (window.currentDataFrame && Array.isArray(window.currentDataFrame.columns) && window.currentDataFrame.columns.length > 0) {
                    columns = [...window.currentDataFrame.columns];
                } else {
                    const storedColumns = sessionStorage.getItem('datasetColumns');
                    if (storedColumns) {
                        try {
                            const parsed = JSON.parse(storedColumns);
                            if (Array.isArray(parsed)) {
                                columns = parsed;
                            }
                        } catch (e) {
                            console.warn('Could not parse stored columns:', e);
                        }
                    }
                }
            }

            const sanitizedColumns = Array.from(new Set((columns || []).filter(Boolean)));

            if (sanitizedColumns.length > 0) {
                try {
                    sessionStorage.setItem('datasetColumns', JSON.stringify(sanitizedColumns));
                } catch (storageError) {
                    console.warn('Failed to persist dataset columns for marketing modal:', storageError);
                }
            }

            currentMarketingCellId = cellId;

            if (skipMarketingModal) {
                return cellId;
            }

            showMarketingAnalysisModal(analysisType, sanitizedColumns);
        } else if (isCategoricalAnalysisType(analysisType) && skipCategoricalModal) {
            return cellId;
        } else if (isNumericFrequencyAnalysisType(analysisType) && skipNumericModal) {
            return cellId;
        } else if (isCategoricalNumericAnalysisType(analysisType) && skipCategoricalNumericModal) {
            return cellId;
        } else if (isCrossTabAnalysisType(analysisType) && skipCrossTabModal) {
            return cellId;
        } else if (isGeospatialAnalysisType(analysisType) && skipGeospatialModal) {
            return cellId;
        } else if (isTimeSeriesAnalysisType(analysisType) && skipTimeSeriesModal) {
            return cellId;
        } else if (isTargetAnalysisType(analysisType) && skipTargetModal) {
            return cellId;
        } else if (isTextAnalysisType(analysisType) && skipTextModal) {
            return cellId;
        } else if (isNetworkAnalysisType(analysisType) && skipNetworkModal) {
            return cellId;
        } else if (isEntityNetworkAnalysisType(analysisType) && skipEntityNetworkModal) {
            return cellId;
        } else {
            await generateAndRunAnalysis(cellId, analysisType, {}, analysisOptions);
        }
    } catch (error) {
        console.error('Analysis execution failed:', error);
        showNotification(`Analysis execution failed for ${getAnalysisTypeName(analysisType)}`, 'error');
    }

    return cellId;
}

// Legacy function for compatibility (redirects to new multi-select system)
async function addAnalysisCell() {
    console.log('addAnalysisCell called - redirecting to multi-select system');
    return await addSelectedAnalysisCells();
}

// Generate and run analysis via backend
async function generateAndRunAnalysis(cellId, analysisType, columnMapping = {}, analysisOptions = {}) {
    // Handle case where cellId is actually analysisType (for marketing modal calls)
    if (typeof cellId === 'string' && !analysisType) {
        analysisType = cellId;
        cellId = `analysis-${Date.now()}`;
        
        // Create a new cell for the analysis
        const newCell = addAnalysisCell();
        cellId = newCell.getAttribute('data-cell-id');
        
        console.log('Created new cell for marketing analysis:', cellId);
    }
    
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    
    console.log('generateAndRunAnalysis called with:', { cellId, analysisType, columnMapping });
    
    if (!analysisType) {
        console.error('No analysis type provided!');
        showNotification('Please select an analysis type first', 'error');
        return;
    }
    const outputArea = document.getElementById(`output-${cellId}`);
    const resultDiv = document.getElementById(`result-${cellId}`);
    const loadingDiv = cell ? cell.querySelector('.loading-indicator') : null;
    const statusBadge = cell ? cell.querySelector('.analysis-run-indicator') : null;
    const optionsForRun = analysisOptions || {};
    const overrideSelection = Array.isArray(optionsForRun.overrideSelectedColumns)
        ? optionsForRun.overrideSelectedColumns.filter(name => name !== null && name !== undefined)
        : null;
    const normalizedOverrideSelection = overrideSelection && overrideSelection.length > 0
        ? Array.from(new Set(
            overrideSelection
                .map(name => (typeof name === 'string' ? name : String(name)).trim())
                .filter(Boolean)
        ))
        : [];
    const selectionPayload = optionsForRun.modalSelectionPayload || null;
    const analysisMetadataOverride = optionsForRun.analysisMetadata;
    const shouldIncludeGlobalSelection = optionsForRun.includeGlobalSelectedColumns !== false;

    let requestPayload = null;
    let runSelectedColumns = null;

    try {
        // Show executing state
        if (cell) {
            cell.classList.add('executing');
            cell.classList.remove('error');
        }
        updateRunIndicator(statusBadge, 'running', 'Running…');
        updateAnalysisMeta(cellId, `Running ${getAnalysisTypeName(analysisType)} • ${new Date().toLocaleTimeString()}`);
        clearAnalysisAlerts(cellId);

        if (loadingDiv) {
            loadingDiv.style.display = 'block';
            loadingDiv.innerHTML = `
                <div class="text-center text-muted py-4">
                    <div class="loading-spinner"></div>
                    <p class="mt-2 mb-0">Generating and running analysis…</p>
                </div>
            `;
        }

        if (outputArea) {
            outputArea.style.display = 'none';
        }

        if (resultDiv) {
            resultDiv.innerHTML = '';
        }
        
        console.log(`Calling granular analysis execution API: /advanced-eda/components/run-analysis`);
        console.log(`Analysis IDs: [${analysisType}]`);
        
        // Call backend API for granular analysis execution
        // Include selected columns if available
        requestPayload = {
            analysis_ids: [analysisType],
            source_id: sourceId
        };
        
        // Add column mapping for marketing analyses
        if (columnMapping && Object.keys(columnMapping).length > 0) {
            requestPayload.column_mapping = columnMapping;
            console.log('Including marketing column mapping:', columnMapping);
        }
        
        if (normalizedOverrideSelection.length > 0) {
            requestPayload.selected_columns = [...normalizedOverrideSelection];
            console.log('Using modal-selected columns in analysis:', requestPayload.selected_columns);
        } else if (shouldIncludeGlobalSelection && selectedColumns.size > 0) {
            requestPayload.selected_columns = Array.from(selectedColumns);
            console.log('Including selected columns in analysis:', requestPayload.selected_columns);
        }

        if (analysisMetadataOverride && typeof analysisMetadataOverride === 'object' && Object.keys(analysisMetadataOverride).length > 0) {
            requestPayload.analysis_metadata = analysisMetadataOverride;
            console.log('Including analysis metadata overrides:', analysisMetadataOverride);
        }

        if (preprocessingApplied) {
            const preprocessingPayload = buildPreprocessingPayload();
            if (preprocessingPayload) {
                requestPayload.preprocessing = preprocessingPayload;
                console.log('Including preprocessing payload:', preprocessingPayload);
            }
        }

        runSelectedColumns = Array.isArray(requestPayload.selected_columns)
            ? [...requestPayload.selected_columns]
            : (selectedColumns && selectedColumns.size > 0 ? Array.from(selectedColumns) : null);

        if (window.EDAChatBridge && typeof window.EDAChatBridge.recordAnalysisPending === 'function') {
            window.EDAChatBridge.recordAnalysisPending(cellId, {
                analysisType,
                analysisName: getAnalysisTypeName(analysisType),
                requestPayload,
                selectedColumns: runSelectedColumns || undefined
            });
        }
        
        const response = await fetch(`/advanced-eda/components/run-analysis`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestPayload)
        });

        console.log(`Response status: ${response.status}`);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('API Error Response:', errorText);
            console.error('Request payload was:', requestPayload);
            console.error('Response status:', response.status);
            console.error('Response headers:', [...response.headers.entries()]);
            
            // Try to parse error as JSON for better debugging
            try {
                const errorJson = JSON.parse(errorText);
                console.error('Parsed error JSON:', errorJson);
                if (errorJson.debug_info) {
                    console.error('Debug info:', errorJson.debug_info);
                }
            } catch (e) {
                console.error('Error response is not JSON:', errorText);
            }
            
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('API Result:', result);

        if (result.success) {
            updateRunIndicator(statusBadge, 'success', `Run #${executionCounter++}`);
            if (cell) {
                cell.classList.remove('executing');
            }

            if (loadingDiv) {
                loadingDiv.style.display = 'none';
            }
            if (outputArea) {
                outputArea.style.display = 'block';
            }

            const structuredResults = getStructuredResultsFromResponse(result);
            const legacyOutput = result.output || result.execution_result;

            const preprocessingReportPayload = result.preprocessing_report || result.metadata?.preprocessing_report;
            if (preprocessingReportPayload) {
                handlePreprocessingReport(preprocessingReportPayload, { preserveAppliedState: true });
            }

            if (resultDiv) {
                if (structuredResults.length > 0) {
                    renderStructuredAnalysis(resultDiv, structuredResults);
                } else if (legacyOutput && (legacyOutput.stdout || legacyOutput.stderr || legacyOutput.plots?.length > 0 || legacyOutput.data_frames?.length > 0)) {
                    renderLegacyCellOutput(resultDiv, legacyOutput);
                } else {
                    resultDiv.innerHTML = '<p class="text-muted">Analysis completed successfully (no output to display).</p>';
                }
            }

            let metaSummary = buildAnalysisMetaText(structuredResults, result.analysis_count);
            updateAnalysisMeta(cellId, metaSummary);

            if (window.EDAChatBridge && typeof window.EDAChatBridge.recordAnalysisSuccess === 'function') {
                window.EDAChatBridge.recordAnalysisSuccess(cellId, {
                    analysisType,
                    analysisName: getAnalysisTypeName(analysisType),
                    structuredResults,
                    legacyOutput,
                    requestPayload,
                    selectedColumns: runSelectedColumns || undefined,
                    metaSummary,
                    rawResponse: result
                });
            }

            if (result.warnings) {
                pushAnalysisAlert(cellId, 'warning', result.warnings);
            }

            if (Array.isArray(result.incompatible_analyses) && result.incompatible_analyses.length > 0) {
                pushAnalysisAlert(
                    cellId,
                    'info',
                    `Skipped incompatible analyses: ${result.incompatible_analyses.join(', ')}`
                );
            }

            const executionWarnings = legacyOutput?.warnings;
            if (executionWarnings) {
                const warningMessage = Array.isArray(executionWarnings)
                    ? executionWarnings.join('<br>')
                    : executionWarnings;
                pushAnalysisAlert(cellId, 'warning', warningMessage);
            }

            // Surface warning or danger insights in alerts area for quick scanning
            if (structuredResults.length > 0) {
                structuredResults.forEach(res => {
                    if (Array.isArray(res.insights)) {
                        res.insights
                            .filter(insight => ['warning', 'danger'].includes((insight.level || '').toLowerCase()))
                            .forEach(insight => pushAnalysisAlert(cellId, insight.level === 'danger' ? 'error' : 'warning', insight.text));
                    }
                });
            }

            if (cell) {
                if (optionsForRun.modalType) {
                    cell.dataset.modalType = optionsForRun.modalType;
                } else {
                    delete cell.dataset.modalType;
                }

                if (normalizedOverrideSelection.length > 0) {
                    cell.dataset.modalSelection = JSON.stringify(normalizedOverrideSelection);
                } else {
                    delete cell.dataset.modalSelection;
                }

                if (selectionPayload) {
                    try {
                        cell.dataset.modalSelectionDetails = JSON.stringify(selectionPayload);
                    } catch (detailError) {
                        console.warn('Unable to persist modal selection details:', detailError);
                    }
                } else {
                    delete cell.dataset.modalSelectionDetails;
                }
            }

            showNotification(`${getAnalysisTypeName(analysisType)} completed!`, 'success');

        } else {
            updateRunIndicator(statusBadge, 'error', 'Failed');
            if (cell) {
                cell.classList.add('error');
                cell.classList.remove('executing');
            }
            const errorMessage = result.error || 'Analysis generation failed';
            if (loadingDiv) {
                loadingDiv.style.display = 'block';
                loadingDiv.innerHTML = `<div class="error-message"><strong>Error:</strong> ${errorMessage}</div>`;
            }
            updateAnalysisMeta(cellId, 'Execution failed. Review the error details below.');
            pushAnalysisAlert(cellId, 'error', errorMessage);
            showNotification(errorMessage, 'error');

            if (window.EDAChatBridge && typeof window.EDAChatBridge.recordAnalysisError === 'function') {
                window.EDAChatBridge.recordAnalysisError(cellId, {
                    analysisType,
                    analysisName: getAnalysisTypeName(analysisType),
                    errorMessage,
                    requestPayload,
                    selectedColumns: runSelectedColumns || undefined
                });
            }
            return;
        }
        
    } catch (error) {
        console.error('Analysis generation error:', error);
        if (cell) {
            cell.classList.remove('executing');
            cell.classList.add('error');
        }
        updateRunIndicator(statusBadge, 'error', 'Failed');
        if (loadingDiv) {
            loadingDiv.style.display = 'block';
            loadingDiv.innerHTML = `<div class="error-message"><strong>Error:</strong> ${error.message}</div>`;
        }
        if (outputArea) {
            outputArea.style.display = 'none';
        }
        updateAnalysisMeta(cellId, 'Execution failed. Check the message above for details.');
        pushAnalysisAlert(cellId, 'error', error.message || 'Analysis failed to complete.');
        showNotification('Analysis failed to complete', 'error');

        if (window.EDAChatBridge && typeof window.EDAChatBridge.recordAnalysisError === 'function') {
            window.EDAChatBridge.recordAnalysisError(cellId, {
                analysisType,
                analysisName: getAnalysisTypeName(analysisType),
                errorMessage: error?.message,
                requestPayload,
                selectedColumns: runSelectedColumns || undefined
            });
        }
    }
}

// Create modern analysis result card
function createAnalysisCell(cellId, analysisType) {
    const analysisName = getAnalysisTypeName(analysisType);
    const defaultDescription = 'Loading component details…';
    
    return `
    <article class="analysis-result-card" data-cell-type="analysis" data-cell-id="${cellId}" data-analysis-type="${analysisType}">
        <header class="analysis-result-header">
            <div class="analysis-header-text">
                <div class="analysis-name-row">
                    <span class="analysis-title">${analysisName}</span>
                </div>
                <p class="analysis-description" id="description-${cellId}">${defaultDescription}</p>
            </div>
            <div class="analysis-actions">
                <span class="analysis-run-indicator status-queued" aria-live="polite">Queued</span>
                <button class="btn btn-sm btn-primary ask-ai-btn ms-2" onclick="openAskAIForCell('${cellId}')" title="Ask AI about this result">
                    <i class="bi bi-stars"></i> Ask AI
                </button>
                <button class="btn btn-sm btn-outline-secondary" onclick="moveCellUp('${cellId}')" title="Move result up">
                    <i class="bi bi-arrow-up"></i>
                </button>
                <button class="btn btn-sm btn-outline-secondary" onclick="moveCellDown('${cellId}')" title="Move result down">
                    <i class="bi bi-arrow-down"></i>
                </button>
                <button class="btn btn-sm btn-outline-secondary" onclick="openAnalysisCodeFromCell('${cellId}', '${analysisType}')" title="View code example">
                    <i class="bi bi-code-slash"></i> Code
                </button>
                <button class="btn btn-sm btn-outline-secondary" onclick="rerunAnalysis('${cellId}', '${analysisType}')" title="Run again">
                    <i class="bi bi-arrow-clockwise"></i> Rerun
                </button>
                <button class="btn btn-sm btn-outline-danger" onclick="deleteCell('${cellId}')" title="Remove result">
                    <i class="bi bi-trash"></i>
                </button>
            </div>
        </header>
        <div class="analysis-meta-row">
            <div class="analysis-meta" id="meta-${cellId}">Ready to execute. Results will appear below.</div>
        </div>
        <div class="analysis-tags" id="tags-${cellId}"></div>
        <div class="analysis-body">
            <div class="loading-indicator">
                <div class="text-center text-muted py-4">
                    <div class="loading-spinner"></div>
                    <p class="mt-2 mb-0">Preparing analysis...</p>
                </div>
            </div>
            <div class="analysis-output cell-output" id="output-${cellId}" style="display: none;">
                <div id="result-${cellId}"></div>
            </div>
        </div>
    </article>`;
}

async function loadAnalysisMetadata(cellId, analysisType) {
    if (!analysisType) {
        return;
    }

    if (analysisMetadataCache.has(analysisType)) {
        applyAnalysisMetadata(cellId, analysisMetadataCache.get(analysisType));
        return;
    }

    try {
        const response = await fetch(`/advanced-eda/components/${analysisType}/info`);
        if (!response.ok) {
            throw new Error(`Metadata request failed with status ${response.status}`);
        }

        const payload = await response.json();
        const metadata = payload.component || payload.metadata || {};
        analysisMetadataCache.set(analysisType, metadata);
        applyAnalysisMetadata(cellId, metadata);
    } catch (error) {
        console.warn(`Failed to load analysis metadata for ${analysisType}:`, error);
        const descriptionElement = document.getElementById(`description-${cellId}`);
        if (descriptionElement && descriptionElement.textContent === 'Loading component details…') {
            descriptionElement.textContent = 'Detailed description is unavailable for this component.';
        }
    }
}

function applyAnalysisMetadata(cellId, metadata = {}) {
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (!cell) {
        return;
    }

    const titleElement = cell.querySelector('.analysis-title');
    if (titleElement && metadata.name) {
        titleElement.textContent = metadata.name;
    }

    const descriptionElement = document.getElementById(`description-${cellId}`);
    if (descriptionElement && metadata.description) {
        descriptionElement.textContent = metadata.description;
    }

    const metaElement = document.getElementById(`meta-${cellId}`);
    if (metaElement) {
        const category = formatAnalysisCategory(metadata.category);
        const complexity = (metadata.complexity || 'intermediate').replace('_', ' ');
        const runtime = metadata.estimated_runtime || '1-5 seconds';
        metaElement.textContent = `${category} • ${complexity.charAt(0).toUpperCase() + complexity.slice(1)} • Est. runtime ${runtime}`;
    }

    const tagsElement = document.getElementById(`tags-${cellId}`);
    if (tagsElement) {
        const tags = Array.isArray(metadata.tags) ? metadata.tags : [];
        if (tags.length) {
            tagsElement.innerHTML = tags
                .slice(0, 8)
                .map(tag => `<span class="analysis-tag">#${tag}</span>`)
                .join('');
        } else {
            tagsElement.innerHTML = '';
        }
    }
}

// Rerun analysis
async function rerunAnalysis(cellId, analysisType) {
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (!cell) {
        showNotification('Analysis cell could not be found for rerun.', 'error');
        return;
    }

    const modalType = cell.dataset.modalType || '';
    let previousSelection = [];
    let previousSelectionDetails = null;

    if (cell.dataset.modalSelection) {
        try {
            const parsed = JSON.parse(cell.dataset.modalSelection);
            if (Array.isArray(parsed)) {
                previousSelection = parsed;
            }
        } catch (parseError) {
            console.warn('Unable to parse stored modal selection for rerun:', parseError);
        }
    }

    if (cell.dataset.modalSelectionDetails) {
        try {
            const parsedDetails = JSON.parse(cell.dataset.modalSelectionDetails);
            if (parsedDetails && typeof parsedDetails === 'object') {
                previousSelectionDetails = parsedDetails;
            }
        } catch (detailError) {
            console.warn('Unable to parse stored modal selection details for rerun:', detailError);
        }
    }

    if (modalType && previousSelection.length > 0) {
        openRerunChoiceModal(cellId, analysisType, modalType, previousSelection, previousSelectionDetails);
        return;
    }

    await executeAnalysisRerun(cellId, analysisType, {});
}

async function executeAnalysisRerun(cellId, analysisType, analysisOptions = {}) {
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (!cell) {
        showNotification('Analysis cell could not be found for rerun.', 'error');
        return;
    }

    const outputArea = document.getElementById(`output-${cellId}`);
    const loadingDiv = cell.querySelector('.loading-indicator');

    cell.classList.remove('error');
    if (outputArea) {
        outputArea.style.display = 'none';
    }
    if (loadingDiv) {
        loadingDiv.style.display = 'block';
        loadingDiv.innerHTML = '<div class="text-center text-muted py-4"><div class="loading-spinner"></div><p class="mt-2 mb-0">Re-running analysis...</p></div>';
    }

    showNotification('Re-running analysis...', 'info');

    if (window.EDAChatBridge && typeof window.EDAChatBridge.recordAnalysisPending === 'function') {
        window.EDAChatBridge.recordAnalysisPending(cellId, {
            analysisType,
            analysisName: getAnalysisTypeName(analysisType),
            requestPayload: analysisOptions?.requestPayload,
            selectedColumns: analysisOptions?.selectedColumns
        });
    }

    try {
        const columnMapping = analysisOptions && typeof analysisOptions.columnMapping === 'object'
            ? { ...analysisOptions.columnMapping }
            : {};
        const optionsForRun = { ...analysisOptions };
        if (optionsForRun.columnMapping) {
            delete optionsForRun.columnMapping;
        }
        await generateAndRunAnalysis(cellId, analysisType, columnMapping, optionsForRun);
    } catch (error) {
        console.error('Analysis rerun failed:', error);
        showNotification('Analysis rerun failed', 'error');
    }
}

function openRerunChoiceModal(cellId, analysisType, modalType, previousSelection, previousDetails = null) {
    const modalElement = document.getElementById('analysisRerunModal');
    if (!modalElement) {
        console.warn('Rerun choice modal is unavailable; defaulting to reuse of previous selection.');
        const fallbackOptions = buildAnalysisOptionsForModal(modalType, previousSelection, previousDetails);
        executeAnalysisRerun(cellId, analysisType, fallbackOptions);
        return;
    }

    rerunModalState.cellId = cellId;
    rerunModalState.analysisType = analysisType;
    rerunModalState.modalType = modalType;
    rerunModalState.previousSelection = Array.isArray(previousSelection) ? [...previousSelection] : [];
    rerunModalState.previousSelectionDetails = previousDetails && typeof previousDetails === 'object'
        ? { ...previousDetails }
        : null;

    const analysisName = getAnalysisTypeName(analysisType);
    const messageElement = document.getElementById('rerunModalMessage');
    if (messageElement) {
        messageElement.textContent = `Would you like to rerun ${analysisName} with the previous column selection or choose a new set?`;
    }

    const previewElement = document.getElementById('rerunModalSelectionPreview');
    if (previewElement) {
        if (rerunModalState.previousSelection.length === 0) {
            previewElement.textContent = 'No previous column selection was stored.';
        } else {
            const preview = rerunModalState.previousSelection.slice(0, 6);
            const remainder = rerunModalState.previousSelection.length - preview.length;
            previewElement.textContent = remainder > 0
                ? `${preview.join(', ')} (+${remainder} more)`
                : preview.join(', ');
        }
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
}

function handleRerunReuse() {
    const { cellId, analysisType, modalType, previousSelection, previousSelectionDetails } = rerunModalState;
    const modalElement = document.getElementById('analysisRerunModal');
    if (modalElement) {
        const instance = bootstrap.Modal.getInstance(modalElement);
        instance?.hide();
    }

    if (!cellId || !analysisType) {
        resetRerunModalState();
        return;
    }

    const selection = Array.isArray(previousSelection) ? [...previousSelection] : [];
    const detailPayload = previousSelectionDetails && typeof previousSelectionDetails === 'object'
        ? { ...previousSelectionDetails }
        : null;
    resetRerunModalState();
    const rerunOptions = buildAnalysisOptionsForModal(modalType, selection, detailPayload);
    executeAnalysisRerun(cellId, analysisType, rerunOptions);
}

function handleRerunReconfigure() {
    const { cellId, analysisType, modalType, previousSelection, previousSelectionDetails } = rerunModalState;
    const modalElement = document.getElementById('analysisRerunModal');
    if (modalElement) {
        const instance = bootstrap.Modal.getInstance(modalElement);
        instance?.hide();
    }

    if (!cellId || !analysisType) {
        resetRerunModalState();
        return;
    }

    const selection = Array.isArray(previousSelection) ? [...previousSelection] : [];
    const detailPayload = previousSelectionDetails && typeof previousSelectionDetails === 'object'
        ? { ...previousSelectionDetails }
        : null;
    resetRerunModalState();

    if (modalType === 'categorical') {
        openCategoricalModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'numeric') {
        openNumericModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'categorical-numeric') {
        openCategoricalNumericModalForRerun(cellId, analysisType, selection, detailPayload);
    } else if (modalType === 'crosstab') {
        openCrossTabModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'time-series') {
        openTimeSeriesModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'text') {
        openTextModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'network') {
        openNetworkModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'entity-network') {
        openEntityNetworkModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'target') {
        openTargetModalForRerun(cellId, analysisType, selection, detailPayload);
    } else if (modalType === 'geospatial') {
        openGeospatialModalForRerun(cellId, analysisType, selection, detailPayload);
    } else {
        const fallbackOptions = buildAnalysisOptionsForModal(modalType, selection, detailPayload);
        executeAnalysisRerun(cellId, analysisType, fallbackOptions);
    }
}

function resetRerunModalState() {
    rerunModalState.cellId = '';
    rerunModalState.analysisType = '';
    rerunModalState.modalType = '';
    rerunModalState.previousSelection = [];
    rerunModalState.previousSelectionDetails = null;
}

function buildAnalysisOptionsForModal(modalType, selection, previousDetails) {
    let normalizedSelection = Array.isArray(selection)
        ? selection
            .map(name => (typeof name === 'string' ? name.trim() : String(name || '')))
            .filter(Boolean)
        : [];

    const options = {
        overrideSelectedColumns: normalizedSelection,
        includeGlobalSelectedColumns: false
    };

    if (modalType) {
        options.modalType = modalType;
    }

    if (previousDetails && typeof previousDetails === 'object') {
        options.modalSelectionPayload = { ...previousDetails };

        if (modalType === 'categorical-numeric' && typeof buildCategoricalNumericAnalysisMetadata === 'function') {
            try {
                options.analysisMetadata = buildCategoricalNumericAnalysisMetadata(previousDetails);
            } catch (metadataError) {
                console.warn('Unable to rebuild categorical vs numeric metadata for rerun fallback:', metadataError);
            }
        } else if (modalType === 'geospatial') {
            const latitude = typeof previousDetails.latitude === 'string' ? previousDetails.latitude.trim() : '';
            const longitude = typeof previousDetails.longitude === 'string' ? previousDetails.longitude.trim() : '';
            const label = typeof previousDetails.label === 'string' ? previousDetails.label.trim() : '';

            if (latitude && longitude) {
                options.columnMapping = {
                    latitude,
                    longitude
                };
                if (label) {
                    options.columnMapping.location_label = label;
                }

                if (normalizedSelection.length === 0) {
                    normalizedSelection = label ? [latitude, longitude, label] : [latitude, longitude];
                    options.overrideSelectedColumns = normalizedSelection;
                }
            }
        }
    }

    options.overrideSelectedColumns = normalizedSelection;

    return options;
}

function openAnalysisCode(event, analysisType) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    if (!analysisType) {
        showNotification('Select an analysis type to view code.', 'warning');
        return;
    }

    const selectedCols = selectedColumns.size > 0 ? Array.from(selectedColumns) : null;
    showAnalysisCodeModal(analysisType, { selectedColumns: selectedCols });
}

function openAnalysisCodeFromCell(cellId, analysisType) {
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    let selectedCols = null;

    if (cell && cell.dataset.modalSelection) {
        try {
            const parsed = JSON.parse(cell.dataset.modalSelection);
            if (Array.isArray(parsed) && parsed.length > 0) {
                selectedCols = parsed;
            }
        } catch (error) {
            console.warn('Unable to parse stored modal selection for code preview:', error);
        }
    }

    if (!selectedCols && selectedColumns.size > 0) {
        selectedCols = Array.from(selectedColumns);
    }

    showAnalysisCodeModal(analysisType, { selectedColumns: selectedCols });
}

async function showAnalysisCodeModal(analysisType, options = {}) {
    const modalElement = document.getElementById('analysisCodeModal');
    if (!modalElement) {
        showNotification('Code preview modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    const titleElement = document.getElementById('analysisCodeModalLabel');
    if (titleElement) {
        titleElement.textContent = `Code snippet • ${getAnalysisTypeName(analysisType)}`;
    }

    const statusElement = document.getElementById('analysisCodeStatus');
    const codeWrapper = document.getElementById('analysisCodeBlock');
    const codeElement = document.getElementById('analysisCodeContent');
    const copyButton = document.getElementById('analysisCodeCopyBtn');

    if (copyButton) {
        copyButton.disabled = true;
    }

    if (statusElement) {
        statusElement.classList.remove('d-none');
        statusElement.textContent = 'Generating notebook-ready code...';
    }
    if (codeWrapper) {
        codeWrapper.classList.add('d-none');
    }
    if (codeElement) {
        codeElement.textContent = '';
    }

    analysisCodeModalCurrentAnalysis = analysisType;
    analysisCodeModalCurrentCode = '';

    const payload = {
        analysis_ids: [analysisType],
        source_id: initSourceId()
    };

    if (!payload.source_id) {
        if (statusElement) {
            statusElement.textContent = 'Source identifier is missing. Load a dataset before generating code.';
        }
        return;
    }

    if (Array.isArray(options.selectedColumns) && options.selectedColumns.length > 0) {
        payload.selected_columns = options.selectedColumns;
    }

    if (options.columnMapping && typeof options.columnMapping === 'object') {
        payload.column_mapping = options.columnMapping;
    }

    try {
        const response = await fetch('/advanced-eda/components/generate-code', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Code generation failed with status ${response.status}`);
        }

        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || 'Code generation failed.');
        }

        analysisCodeModalCurrentCode = result.code || '';

        if (statusElement) {
            statusElement.classList.add('d-none');
        }
        if (codeWrapper) {
            codeWrapper.classList.remove('d-none');
        }
        if (codeElement) {
            codeElement.textContent = analysisCodeModalCurrentCode || '# No code generated.';
        }
        if (copyButton) {
            copyButton.disabled = !analysisCodeModalCurrentCode;
        }
    } catch (error) {
        console.error('Failed to generate analysis code:', error);
        if (statusElement) {
            statusElement.classList.remove('d-none');
            statusElement.textContent = error.message || 'Unable to generate code for this analysis.';
        }
        if (copyButton) {
            copyButton.disabled = true;
        }
    }
}

function copyAnalysisCodeToClipboard() {
    if (!analysisCodeModalCurrentCode) {
        showNotification('No code available to copy.', 'warning');
        return;
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(analysisCodeModalCurrentCode)
            .then(() => showNotification('Analysis code copied to clipboard.', 'success'))
            .catch(error => {
                console.warn('Clipboard copy failed:', error);
                fallbackCopyAnalysisCode();
            });
    } else {
        fallbackCopyAnalysisCode();
    }
}

function fallbackCopyAnalysisCode() {
    const textarea = document.createElement('textarea');
    textarea.value = analysisCodeModalCurrentCode;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'absolute';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();
    try {
        document.execCommand('copy');
        showNotification('Analysis code copied to clipboard.', 'success');
    } catch (error) {
        console.warn('Fallback clipboard copy failed:', error);
        showNotification('Unable to copy code automatically. Select and copy manually.', 'warning');
    } finally {
        document.body.removeChild(textarea);
    }
}

// Structured analysis rendering helpers
function getStructuredResultsFromResponse(result) {
    if (Array.isArray(result.analysis_results) && result.analysis_results.length > 0) {
        return result.analysis_results;
    }

    if (result.output && Array.isArray(result.output.analysis_results) && result.output.analysis_results.length > 0) {
        return result.output.analysis_results;
    }

    return [];
}

function renderStructuredAnalysis(container, analysisResults) {
    if (!Array.isArray(analysisResults) || analysisResults.length === 0) {
        container.innerHTML = '<p class="text-muted">Analysis completed successfully (no structured results to display).</p>';
        return;
    }

    const sections = analysisResults.map(renderStructuredSection).join('');
    container.innerHTML = sections;
}

function buildAnalysisMetaText(structuredResults, analysisCount) {
    const timestamp = new Date().toLocaleTimeString();

    if (!structuredResults || structuredResults.length === 0) {
        const count = analysisCount || 1;
        const suffix = count === 1 ? 'component' : 'components';
        return `Completed ${timestamp} • ${count} ${suffix}`;
    }

    const counts = structuredResults.reduce((acc, result) => {
        const status = (result.status || 'success').toLowerCase();
        acc[status] = (acc[status] || 0) + 1;
        return acc;
    }, {});

    const summary = Object.entries(counts)
        .map(([status, count]) => `${count} ${status}`)
        .join(', ');

    return `Completed ${timestamp} • ${summary}`;
}

function renderStructuredSection(result) {
    const title = escapeHtml(result.title || result.analysis_id || 'Analysis Result');
    const summary = result.summary ? `<p class="analysis-section-summary">${escapeHtml(result.summary)}</p>` : '';
    const metrics = renderAnalysisMetrics(result.metrics);
    const insights = renderAnalysisInsights(result.insights);
    const tables = renderAnalysisTables(result.tables);
    const charts = renderAnalysisCharts(result.charts);
    const details = renderAnalysisDetails(result.details);
    const statusBadge = renderAnalysisStatusBadge(result.status);

    return `
        <section class="analysis-structured-section">
            <header class="structured-section-header">
                <div class="structured-section-heading">
                    <h5>${title}</h5>
                    ${summary}
                </div>
                ${statusBadge}
            </header>
            ${metrics}
            ${insights}
            ${tables}
            ${charts}
            ${details}
        </section>
    `;
}

function renderAnalysisStatusBadge(status) {
    const normalized = (status || 'success').toLowerCase();
    const label = normalized.charAt(0).toUpperCase() + normalized.slice(1);
    return `<span class="analysis-status-badge status-${escapeHtml(normalized)}">${escapeHtml(label)}</span>`;
}

function renderAnalysisMetrics(metrics) {
    if (!Array.isArray(metrics) || metrics.length === 0) {
        return '';
    }

    const cards = metrics
        .map(metric => {
            const label = escapeHtml(metric.label || 'Metric');
            const value = formatMetricValue(metric.value);
            const unit = metric.unit ? `<span class="metric-unit">${escapeHtml(metric.unit)}</span>` : '';
            const description = metric.description ? `<div class="metric-description">${escapeHtml(metric.description)}</div>` : '';
            const trendIcon = metric.trend === 'up' ? '<i class="bi bi-arrow-up-right text-success"></i>' : metric.trend === 'down' ? '<i class="bi bi-arrow-down-right text-danger"></i>' : '';
            return `
                <div class="analysis-metric-card">
                    <div class="metric-header">
                        <span class="metric-label">${label}</span>
                        ${trendIcon}
                    </div>
                    <div class="metric-value">${value}${unit}</div>
                    ${description}
                </div>
            `;
        })
        .join('');

    return `
        <section class="analysis-section analysis-section-metrics">
            <h6>Key metrics</h6>
            <div class="analysis-metrics-grid">
                ${cards}
            </div>
        </section>
    `;
}

function renderAnalysisInsights(insights) {
    if (!Array.isArray(insights) || insights.length === 0) {
        return '';
    }

    const items = insights
        .map(insight => {
            const level = escapeHtml((insight.level || 'info').toLowerCase());
            const text = escapeHtml(insight.text || '');
            const icon = level === 'success' ? 'bi-check-circle' : level === 'warning' ? 'bi-exclamation-triangle' : level === 'danger' ? 'bi-x-circle' : 'bi-info-circle';
            return `
                <li class="analysis-insight-item insight-${level}">
                    <span class="insight-badge">
                        <i class="bi ${icon}"></i>
                        ${level.charAt(0).toUpperCase() + level.slice(1)}
                    </span>
                    <span class="insight-text">${text}</span>
                </li>
            `;
        })
        .join('');

    return `
        <section class="analysis-section analysis-section-insights">
            <h6>Insights</h6>
            <ul class="analysis-insights-list">
                ${items}
            </ul>
        </section>
    `;
}

function renderAnalysisTables(tables) {
    if (!Array.isArray(tables) || tables.length === 0) {
        return '';
    }

    const sections = tables
        .map(table => {
            const title = escapeHtml(table.title || 'Table');
            const description = table.description ? `<p class="analysis-section-note">${escapeHtml(table.description)}</p>` : '';
            const tableHtml = createTableHtml(table.columns, table.rows);
            return `
                <div class="analysis-table-wrapper">
                    <div class="analysis-table-header">
                        <h6>${title}</h6>
                        ${description}
                    </div>
                    <div class="table-responsive analysis-table-responsive">
                        ${tableHtml}
                    </div>
                </div>
            `;
        })
        .join('');

    return `
        <section class="analysis-section analysis-section-tables">
            ${sections}
        </section>
    `;
}

function renderAnalysisCharts(charts) {
    if (!Array.isArray(charts) || charts.length === 0) {
        return '';
    }

    const items = charts
        .map((chart, index) => {
            const rawTitle = chart.title || `Chart ${index + 1}`;
            const rawDescription = chart.description || '';
            const title = escapeHtml(rawTitle);
            const description = rawDescription ? `<p class="analysis-section-note">${escapeHtml(rawDescription)}</p>` : '';
            const imageSrc = chart.image || '';
            const datasetTitle = escapeHtml(rawTitle);
            const datasetDescription = escapeHtml(rawDescription);
            const datasetSrc = escapeHtml(imageSrc);

            return `
                <figure class="analysis-chart" data-chart-src="${datasetSrc}" data-chart-title="${datasetTitle}" data-chart-description="${datasetDescription}">
                    <button type="button" class="chart-expand-button" data-chart-expand-trigger aria-label="Expand chart ${title}">
                        <i class="bi bi-arrows-fullscreen" aria-hidden="true"></i>
                        <span class="visually-hidden">Expand chart</span>
                    </button>
                    <img src="${imageSrc}" alt="${title}" class="analysis-chart-image" data-chart-expand-trigger />
                    <figcaption>
                        <strong>${title}</strong>
                        ${description}
                    </figcaption>
                </figure>
            `;
        })
        .join('');

    return `
        <section class="analysis-section analysis-section-charts">
            <h6>Visualizations</h6>
            <div class="analysis-charts-grid">
                ${items}
            </div>
        </section>
    `;
}

function renderAnalysisDetails(details) {
    if (!details || typeof details !== 'object' || Object.keys(details).length === 0) {
        return '';
    }

    const json = JSON.stringify(details, null, 2);
    return `
        <section class="analysis-section analysis-section-details">
            <h6>Additional details</h6>
            <pre class="analysis-details-pre">${escapeHtml(json)}</pre>
        </section>
    `;
}

function openChartExpandModal(imageSrc, title, description) {
    if (!imageSrc) {
        console.warn('Cannot open chart modal without an image source');
        return;
    }

    const modalElement = document.getElementById('chartExpandModal');
    const modalImage = document.getElementById('chartExpandModalImage');
    const modalTitle = document.getElementById('chartExpandModalTitle');
    const modalDescription = document.getElementById('chartExpandModalDescription');

    if (!modalElement || !modalImage || !modalTitle || !modalDescription) {
        console.warn('Chart expand modal elements not found in the DOM');
        return;
    }

    modalImage.src = imageSrc;
    modalImage.alt = title || 'Expanded chart';
    modalTitle.textContent = title || 'Chart visualization';

    if (description) {
        modalDescription.textContent = description;
        modalDescription.classList.remove('d-none');
    } else {
        modalDescription.textContent = '';
        modalDescription.classList.add('d-none');
    }

    if (typeof bootstrap === 'undefined' || !bootstrap.Modal) {
        console.warn('Bootstrap modal library is not available for chart expansion');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
}

document.addEventListener('click', event => {
    const trigger = event.target.closest('[data-chart-expand-trigger]');
    if (!trigger) {
        return;
    }

    const chartFigure = trigger.closest('.analysis-chart');
    if (!chartFigure) {
        return;
    }

    const { chartSrc, chartTitle, chartDescription } = chartFigure.dataset;
    if (!chartSrc) {
        console.warn('No chart source found for expand interaction');
        return;
    }

    event.preventDefault();
    openChartExpandModal(chartSrc, chartTitle, chartDescription);
});

function createTableHtml(columns = [], rows = []) {
    if (!Array.isArray(columns) || columns.length === 0) {
        return '<p class="text-muted">No table data available.</p>';
    }

    const headerHtml = columns.map(column => `<th>${escapeHtml(column)}</th>`).join('');
    const bodyHtml = rows
        .map(row => {
            const cells = columns
                .map(column => {
                    const cellValue = row[column];
                    if (cellValue === null || cellValue === undefined) {
                        return '<td><span class="text-muted">—</span></td>';
                    }
                    return `<td>${escapeHtml(String(cellValue))}</td>`;
                })
                .join('');
            return `<tr>${cells}</tr>`;
        })
        .join('');

    return `
        <table class="table table-sm table-striped">
            <thead>
                <tr>${headerHtml}</tr>
            </thead>
            <tbody>
                ${bodyHtml}
            </tbody>
        </table>
    `;
}

function formatMetricValue(value) {
    if (typeof value === 'number') {
        return value % 1 === 0 ? value.toString() : value.toFixed(2);
    }
    if (value === null || value === undefined) {
        return '—';
    }
    return escapeHtml(String(value));
}

function escapeHtml(value) {
    if (typeof value !== 'string') {
        return value;
    }
    return value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// Legacy output renderer (fallback)
function renderLegacyCellOutput(container, output) {
    let html = '';

    if (output.stdout) {
        html += `<div class="output-section">
            <pre class="stdout-output">${escapeHtml(output.stdout)}</pre>
        </div>`;
    }

    if (output.stderr) {
        html += `<div class="output-section">
            <div class="alert alert-warning">
                <strong>Warning/Error Output:</strong>
                <pre class="stderr-output">${escapeHtml(output.stderr)}</pre>
            </div>
        </div>`;
    }

    if (output.plots && output.plots.length > 0) {
        html += '<div class="output-section"><h6>Generated Plots:</h6><div class="plots-container">';
        output.plots.forEach((plot, index) => {
            html += `<div class="chart-container mb-3">
                <img src="data:image/png;base64,${plot}" alt="Plot ${index + 1}" class="notebook-plot-image">
            </div>`;
        });
        html += '</div></div>';
    }

    if (output.data_frames && output.data_frames.length > 0) {
        html += '<div class="output-section"><h6>Generated DataFrames:</h6>';
        output.data_frames.forEach(df => {
            html += `<div class="dataframe-container mb-3">
                <div class="table-responsive">
                    ${df.html}
                </div>
            </div>`;
        });
        html += '</div>';
    }

    if (!html) {
        html = '<p class="text-muted">No output generated</p>';
    }

    container.innerHTML = html;
}

// Delete analysis result card
function deleteCell(cellId) {
    if (confirm('Remove this analysis result?')) {
        const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
        if (cell) {
            cell.remove();
            if (window.EDAChatBridge && typeof window.EDAChatBridge.removeAnalysisCell === 'function') {
                window.EDAChatBridge.removeAnalysisCell(cellId);
            }
            showNotification('Analysis result removed.', 'info');
            updateAnalysisResultsPlaceholder();
        }
    }
}

// Get friendly analysis type name
function getAnalysisTypeName(analysisType) {
    const names = {
        // Data Quality & Structure
        'dataset_shape_analysis': 'Dataset Shape Analysis',
        'data_range_validation': 'Data Range Validation',
        'data_types_validation': 'Data Types Validation',
        'missing_value_analysis': 'Missing Value Analysis',
        'duplicate_detection': 'Duplicate Detection',
        
        // Univariate Analysis - Numeric
        'summary_statistics': 'Summary Statistics',
    'numeric_frequency_analysis': 'Numeric Frequency Analysis',
        'distribution_plots': 'Distribution Plots',
        'histogram_plots': 'Histogram Plots',
        'box_plots': 'Box Plots',
        'violin_plots': 'Violin Plots',
        'kde_plots': 'KDE (Density) Plots',
        'skewness_analysis': 'Skewness Analysis',
        'skewness_statistics': 'Skewness Statistics',
        'skewness_visualization': 'Skewness Visualization',
        'normality_test': 'Normality Test',
        
        // Univariate Analysis - Categorical
        'categorical_frequency_analysis': 'Categorical Frequency Analysis',
    'categorical_cardinality_profile': 'Categorical Cardinality Profile',
    'rare_category_detection': 'Rare Category Detection',
        'frequency_analysis': 'Frequency Analysis',
        'categorical_visualization': 'Categorical Visualization',
        'categorical_bar_charts': 'Categorical Bar Charts',
        'categorical_pie_charts': 'Categorical Pie Charts',
        
        // Bivariate/Multivariate Analysis
        'correlation_analysis': 'Correlation Analysis',
        'pearson_correlation': 'Pearson Correlation',
        'spearman_correlation': 'Spearman Correlation',
    'scatter_plot_analysis': 'Scatter Plot Analysis',
    'cross_tabulation_analysis': 'Cross Tabulation Analysis',
    'categorical_numeric_relationships': 'Categorical vs Numeric Explorer',
        
        // Outlier & Anomaly Detection
        'iqr_outlier_detection': 'IQR Outlier Detection',
        'iqr_detection': 'IQR Outlier Detection',
        'zscore_outlier_detection': 'Z-Score Outlier Detection',
        'zscore_detection': 'Z-Score Outlier Detection',
        'visual_outlier_inspection': 'Visual Outlier Inspection',
        'visual_inspection': 'Visual Outlier Inspection',
        'outlier_distribution_visualization': 'Outlier Distribution Plots',
        'outlier_scatter_matrix': 'Outlier Scatter Matrix',
        
        // Time-Series Exploration
        'temporal_trend_analysis': 'Temporal Trend Analysis',
        'temporal_trends': 'Temporal Trends',
        'seasonality_detection': 'Seasonality Detection',
    'datetime_feature_extraction': 'Datetime Feature Extraction',

        // Geospatial Analysis
        'coordinate_system_projection_check': 'Coordinate System & Projection Check',
        'spatial_distribution_analysis': 'Spatial Distribution Analysis',
        'spatial_relationships_analysis': 'Spatial Relationships Analysis',

        // Text Analysis
        'text_length_distribution': 'Text Length Distribution',
        'text_token_frequency': 'Text Token Frequency',
    'text_vocabulary_summary': 'Text Vocabulary Summary',
    'text_feature_engineering_profile': 'Text Feature Engineering Profile',
    'text_nlp_profile': 'Advanced NLP Profile',
        
        // Relationship Exploration
        'multicollinearity_analysis': 'Multicollinearity Analysis',
        'pca_dimensionality_reduction': 'PCA Analysis',
        'network_analysis': 'Network Analysis',
        'entity_relationship_network': 'Entity Network Analysis',
        'pca_analysis': 'PCA Analysis',
        'pca_scree_plot': 'PCA Scree Plot',
        'pca_cumulative_variance': 'PCA Cumulative Variance',
        'pca_visualization': 'PCA Visualization',
        'pca_biplot': 'PCA Biplot',
        'pca_heatmap': 'PCA Loadings Heatmap',
    'cluster_tendency_analysis': 'Cluster Tendency Analysis',
    'cluster_segmentation_analysis': 'Cluster Segmentation',
        
        // Marketing Analysis Components
        'campaign_metrics_analysis': 'Campaign Metrics Analysis',
        'conversion_funnel_analysis': 'Conversion Funnel Analysis',
        'engagement_analysis': 'Engagement Analysis',
        'channel_performance_analysis': 'Channel Performance Analysis',
        'audience_segmentation_analysis': 'Audience Segmentation Analysis',
        'roi_analysis': 'ROI Analysis',
        'attribution_analysis': 'Attribution Analysis',
        'cohort_analysis': 'Cohort Analysis',
        
        // Legacy names for backward compatibility
        'data_quality_structure': 'Data Quality & Structure Check',
        'missing_data_analysis': 'Missing Data Analysis',
        'data_profiling': 'Comprehensive Data Profiling',
        'univariate_numeric': 'Numeric Variables Analysis',
        'univariate_categorical': 'Categorical Variables Analysis',
        'distribution_analysis': 'Distribution Analysis',
        'bivariate_multivariate': 'Relationships & Interactions',
        'relationship_exploration': 'Advanced Relationships',
        'advanced_outlier_detection': 'Advanced Outlier Detection',
        'outlier_detection': 'Basic Outlier Detection',
        'time_series_exploration': 'Time Series Analysis',
        'data_overview': 'Data Overview & Summary',
        'data_types': 'Data Types & Info',
        'missing_values': 'Missing Values Analysis',
        'descriptive_stats': 'Descriptive Statistics',
        'feature_importance': 'Feature Importance',
        'clustering_analysis': 'Clustering Analysis',
        'classification_eda': 'Classification EDA',
        'regression_eda': 'Regression EDA',
        'time_series_eda': 'Time Series EDA',
        'fraud_detection_eda': 'Fraud Detection EDA'
    };
    
    return names[analysisType] || analysisType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Add markdown cell
function addMarkdownCell() {
    console.log('addMarkdownCell called');
    const cellId = `markdown-${cellCounter++}`;
    const markdownHTML = `
    <div class="notebook-cell" data-cell-type="markdown" data-cell-id="${cellId}">
        <div class="cell-container">
            <div class="cell-header">
                <div class="cell-type-indicator markdown-cell">
                    <i class="bi bi-markdown"></i> Markdown
                </div>
                <div class="cell-actions">
                    <button class="btn btn-sm btn-outline-secondary" onclick="moveCellUp('${cellId}')" title="Move Up">
                        <i class="bi bi-arrow-up"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="moveCellDown('${cellId}')" title="Move Down">
                        <i class="bi bi-arrow-down"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="editMarkdownCell('${cellId}')">
                        <i class="bi bi-pencil"></i> Edit
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteCell('${cellId}')">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
            <div class="cell-content">
                <textarea class="markdown-editor markdown-editor-min-height" id="markdown-${cellId}" placeholder="Enter markdown content...">## Analysis Notes

Add your markdown content here...</textarea>
                <div class="markdown-preview d-none" id="preview-${cellId}">
                    <h2>Analysis Notes</h2>
                    <p>Add your markdown content here...</p>
                </div>
            </div>
        </div>
    </div>`;
    
    const cellsContainer = document.getElementById('notebookCells');
    cellsContainer.insertAdjacentHTML('afterbegin', markdownHTML);
    updateAnalysisResultsPlaceholder();
    
    showNotification('Markdown cell added at top!', 'success');
    
    // Scroll to new cell
    const newCell = document.querySelector(`[data-cell-id="${cellId}"]`);
    newCell.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Edit markdown cell
function editMarkdownCell(cellId) {
    const editor = document.getElementById(`markdown-${cellId}`);
    const preview = document.getElementById(`preview-${cellId}`);
    
    if (editor.style.display === 'none') {
        // Switch to edit mode
        editor.style.display = 'block';
        preview.style.display = 'none';
    } else {
        // Switch to preview mode
        editor.style.display = 'none';
        preview.style.display = 'block';
        // Here you would typically convert markdown to HTML
        preview.innerHTML = `<pre>${editor.value}</pre>`;
    }
}

let lastDomainDetectionResult = null;
let isDomainDetectionLoading = false;

// Helper to format domain labels consistently
function formatDomainLabel(domain) {
    if (!domain) {
        return 'Unknown';
    }

    try {
        return domain
            .toString()
            .replace(/[_-]+/g, ' ')
            .replace(/\s+/g, ' ')
            .trim()
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    } catch (error) {
        console.warn('Unable to format domain label:', domain, error);
        return String(domain);
    }
}

function handleDomainButtonClick(event) {
    const wantsRefresh = event && (event.shiftKey || event.metaKey || event.ctrlKey);

    if (isDomainDetectionLoading) {
        console.log('Domain detection is already running, ignoring click');
        return;
    }

    if (wantsRefresh || !lastDomainDetectionResult) {
        loadDomainDetection(true, { forceRefresh: Boolean(wantsRefresh) });
        return;
    }

    openDomainModalWithResult(lastDomainDetectionResult);
}

// Load and display domain detection results
async function loadDomainDetection(manualTrigger = false, options = {}) {
    const { forceRefresh = false } = options;
    console.log('loadDomainDetection called with sourceId:', sourceId, 'manualTrigger:', manualTrigger, 'forceRefresh:', forceRefresh);

    const detectButton = document.getElementById('domainDetectButton');
    const labelEl = document.getElementById('domainButtonLabel');
    const confidenceEl = document.getElementById('domainButtonConfidence');
    const separatorEl = document.getElementById('domainButtonSeparator');
    const spinnerEl = document.getElementById('domainButtonSpinner');

    if (!detectButton || !labelEl) {
        console.warn('Domain detection UI not found');
        return null;
    }

    if (!sourceId) {
        labelEl.textContent = 'No dataset';
        if (confidenceEl) confidenceEl.classList.add('d-none');
        if (separatorEl) separatorEl.classList.add('d-none');
        detectButton.disabled = true;
        detectButton.classList.add('domain-compact-button--error');
        return null;
    }

    isDomainDetectionLoading = true;
    detectButton.disabled = true;
    detectButton.classList.remove('domain-compact-button--error');
    detectButton.classList.add('domain-compact-button--loading');
    detectButton.setAttribute('data-loading', 'true');

    if (spinnerEl) {
        spinnerEl.classList.remove('d-none');
    }

    labelEl.textContent = manualTrigger ? (forceRefresh ? 'Refreshing…' : 'Rechecking…') : 'Detecting…';
    if (confidenceEl) confidenceEl.classList.add('d-none');
    if (separatorEl) separatorEl.classList.add('d-none');

    try {
        console.log(`Calling domain detection API: /advanced-eda/api/detect-domain/${sourceId}`);
        const response = await fetch(`/advanced-eda/api/detect-domain/${sourceId}`);
        console.log('Domain detection response status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Domain detection API error:', errorText);

            if (response.status === 404) {
                throw new Error('Dataset not found. Please verify the data source.');
            }

            throw new Error(`Failed with status ${response.status}`);
        }

        const result = await response.json();
        console.log('Domain detection result:', result);

        if (result.success && (result.primary_domain || result.domain)) {
            const domainNameRaw = result.primary_domain || result.domain;
            const formattedDomain = formatDomainLabel(domainNameRaw);
            const confidenceScore = result.primary_confidence ?? result.confidence ?? 0;
            const confidencePercent = Math.round(confidenceScore * 100);

            labelEl.textContent = formattedDomain;

            if (confidenceEl) {
                confidenceEl.textContent = `${confidencePercent}% confidence`;
                confidenceEl.classList.remove('d-none');
            }

            if (separatorEl) {
                separatorEl.classList.remove('d-none');
            }

            detectButton.dataset.domain = formattedDomain;
            detectButton.dataset.confidence = confidencePercent;
            detectButton.setAttribute('data-has-domain', 'true');
            detectButton.title = `Detected domain: ${formattedDomain}. Click to view details. Hold Shift to refresh.`;

            lastDomainDetectionResult = result;

            if (Array.isArray(result.recommendations)) {
                displayDomainSpecificRecommendations(result.recommendations, formattedDomain);
            }

            if (manualTrigger) {
                openDomainModalWithResult(result);
            }

            return result;
        }

        throw new Error(result.error || 'Domain detection failed');
    } catch (error) {
        console.error('Domain detection error:', error);
        lastDomainDetectionResult = null;
    delete detectButton.dataset.domain;
    delete detectButton.dataset.confidence;
    detectButton.removeAttribute('data-has-domain');
        labelEl.textContent = 'Retry detection';
        if (confidenceEl) {
            confidenceEl.textContent = 'Tap to try again';
            confidenceEl.classList.remove('d-none');
        }
        if (separatorEl) {
            separatorEl.classList.remove('d-none');
        }
        detectButton.classList.add('domain-compact-button--error');
        detectButton.title = `Domain detection failed: ${error.message}`;
        throw error;
    } finally {
        isDomainDetectionLoading = false;
        detectButton.disabled = false;
        detectButton.classList.remove('domain-compact-button--loading');
        detectButton.removeAttribute('data-loading');
        if (spinnerEl) {
            spinnerEl.classList.add('d-none');
        }
    }
}

// Display domain-specific recommendations in the sidebar (compact tracking)
function displayDomainSpecificRecommendations(recommendations, domainName) {
    console.log('Domain-specific recommendations available for', domainName, recommendations);

    const detectButton = document.getElementById('domainDetectButton');
    if (detectButton) {
        detectButton.dataset.recommendationsCount = Array.isArray(recommendations) ? recommendations.length : 0;
    }
}

function cleanupDomainModalArtifacts() {
    try {
        document.body.classList.remove('modal-open');
    } catch (err) {
        // ignore
    }

    try {
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(backdrop => backdrop.remove());
    } catch (err) {
        // ignore
    }
}

function openDomainModalWithResult(result) {
    if (!result) {
        console.warn('Cannot open domain modal without a result');
        return;
    }

    const modalElement = document.getElementById('domainRecommendationsModal');
    const modalContent = document.getElementById('domainRecommendationsContent');

    if (!modalElement || !modalContent) {
        console.warn('Domain recommendations modal elements not found');
        return;
    }

    modalContent.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading domain summary...</span>
            </div>
            <p class="mt-3 text-muted">Preparing domain summary…</p>
        </div>
    `;

    let modalInstance = null;
    try {
        if (typeof bootstrap === 'undefined' || !bootstrap.Modal) {
            throw new Error('Bootstrap modal library not available');
        }

        modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
        modalInstance.show();
    } catch (err) {
        console.error('Failed to open domain recommendations modal:', err);
        cleanupDomainModalArtifacts();
        if (typeof showNotification === 'function') {
            showNotification('Domain recommendations are unavailable. Please try again in a moment.', 'error');
        }
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', () => {
        cleanupDomainModalArtifacts();
    }, { once: true });

    requestAnimationFrame(() => {
        try {
            displayDetailedDomainRecommendations(result);
        } catch (renderError) {
            console.error('Failed to render domain recommendations:', renderError);
            modalContent.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    Unable to render domain recommendations. Please try again.
                </div>
            `;
        }
    });
}

function forceRefreshDomainDetection() {
    if (isDomainDetectionLoading) {
        return;
    }

    loadDomainDetection(true, { forceRefresh: true });
}

// Load and display domain analysis
async function loadDomainAnalysis(templateName = 'data_quality_structure') {
    console.log('loadDomainAnalysis called with sourceId:', sourceId, 'template:', templateName);
    if (!sourceId) {
        alert('No dataset ID available for domain analysis');
        return;
    }

    const domainSection = document.getElementById('domainAnalysisSection');
    if (domainSection) {
        domainSection.classList.remove('d-none');
        domainSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    try {
        console.log(`Calling domain analysis API: /advanced-eda/api/domain-analysis/${sourceId}?template_name=${templateName}`);
        const response = await fetch(`/advanced-eda/api/domain-analysis/${sourceId}?template_name=${templateName}&analysis_depth=intermediate`);
        console.log('Domain analysis response status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Domain analysis API error:', errorText);
            throw new Error(`Failed to load domain analysis: ${response.status}`);
        }

        const result = await response.json();
        console.log('Domain analysis result:', result);

        if (result.success) {
            displayDomainAnalysisResults(result);
        } else {
            throw new Error(result.error || 'Domain analysis failed');
        }
    } catch (error) {
        console.error('Domain analysis error:', error);
        const analysisContent = document.getElementById('domainAnalysisContent');
        if (analysisContent) {
            analysisContent.innerHTML = `<div class="alert alert-danger">Failed to load domain analysis: ${error.message}</div>`;
        }
    }
}

// Load and display domain recommendations
async function loadDomainRecommendations() {
    console.log('loadDomainRecommendations called with sourceId:', sourceId);

    if (isDomainDetectionLoading) {
        console.log('Domain detection currently running; please wait until it finishes.');
        return;
    }

    if (!sourceId) {
        alert('No dataset ID available for domain recommendations');
        return;
    }

    if (lastDomainDetectionResult) {
        openDomainModalWithResult(lastDomainDetectionResult);
        return;
    }

    try {
        const result = await loadDomainDetection(true, { forceRefresh: true });
        if (result) {
            openDomainModalWithResult(result);
        }
    } catch (error) {
        console.error('Unable to load domain recommendations:', error);
        showNotification(`Domain recommendations failed: ${error.message}`, 'error');
    }
}

// Cell movement functions
function moveCellUp(cellId) {
    console.log('Move up called for cell:', cellId);
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (cell) {
        const previousSibling = cell.previousElementSibling;
        if (previousSibling) {
            cell.parentNode.insertBefore(cell, previousSibling);
            showNotification('Cell moved up', 'success');
        } else {
            showNotification('Cell is already at the top', 'info');
        }
    } else {
        console.error('Cell not found with ID:', cellId);
        showNotification('Error: Cell not found', 'error');
    }
}

function moveCellDown(cellId) {
    console.log('Move down called for cell:', cellId);
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (cell) {
        const nextSibling = cell.nextElementSibling;
        if (nextSibling) {
            cell.parentNode.insertBefore(nextSibling, cell);
            showNotification('Cell moved down', 'success');
        } else {
            showNotification('Cell is already at the bottom', 'info');
        }
    } else {
        console.error('Cell not found with ID:', cellId);
        showNotification('Error: Cell not found', 'error');
    }
}

function updateAnalysisResultsPlaceholder() {
    const container = document.getElementById('notebookCells');
    const placeholder = document.getElementById('analysisResultsPlaceholder');

    if (!container || !placeholder) {
        return;
    }

    const hasCells = container.children.length > 0;
    placeholder.classList.toggle('d-none', hasCells);
    placeholder.setAttribute('aria-hidden', hasCells ? 'true' : 'false');
}

function clearAllCells() {
    const cellsContainer = document.getElementById('notebookCells');
    if (cellsContainer) {
        if (confirm('Clear all analysis results? This action cannot be undone.')) {
            cellsContainer.innerHTML = '';
            cellCounter = 1;
            executionCounter = 1;
            showNotification('Analysis results cleared. Dataset and preprocessing stay as-is.', 'success');
            updateAnalysisResultsPlaceholder();
            updatePreprocessingStatusBadge(lastPreprocessingReport);
            refreshCategoryLocks();
        }
    }
}

// Display detailed domain recommendations in modal
function displayDetailedDomainRecommendations(domainResult) {
    const recommendationsContent = document.getElementById('domainRecommendationsContent');
    if (!recommendationsContent) return;
    
    const primaryDomain = domainResult.primary_domain || domainResult.domain || 'general';
    const primaryConfidence = domainResult.primary_confidence || domainResult.confidence || 0;
    const secondaryDomains = domainResult.secondary_domains || [];
    const recommendations = domainResult.recommendations || [];
    const allScores = domainResult.all_scores || domainResult.domain_scores || {};
    const formattedPrimaryDomain = formatDomainLabel(primaryDomain);
    const primaryConfidencePercent = Math.round(primaryConfidence * 100);
    
    let html = '';
    
    html += `
        <div class="d-flex flex-column flex-md-row justify-content-between align-items-start align-items-md-center gap-3 mb-4">
            <div>
                <h5 class="mb-1">Domain overview</h5>
                <p class="text-muted small mb-0">Confidence scores and recommended flows for ${formattedPrimaryDomain} data.</p>
            </div>
            <div class="d-flex gap-2">
                <button class="btn btn-outline-secondary btn-sm" onclick="forceRefreshDomainDetection()">
                    <i class="bi bi-arrow-clockwise"></i> Re-run detection
                </button>
            </div>
        </div>
    `;
    
    // Safety disclaimer with exclamation icon
    html += `
        <div class="alert alert-warning domain-warning-alert d-flex align-items-start mb-4" role="alert">
            <i class="bi bi-exclamation-triangle-fill text-warning me-3 fs-4" style="margin-top: 2px;"></i>
            <div>
                <strong>Important Notice:</strong> Domain predictions are not 100% accurate and should be used as guidance only. 
                Please investigate the results carefully for your safety and verify the appropriateness of the recommended analysis 
                methods for your specific dataset and use case.
            </div>
        </div>
    `;
    
    // Primary and Secondary Domain Detection Summary
    html += `
        <div class="mb-4">
            <div class="card border-primary domain-detection-card">
                <div class="card-header bg-primary text-white py-2 domain-detection-card__header">
                    <h6 class="mb-0">
                        <i class="bi bi-bullseye"></i> Domain Detection Results
                    </h6>
                </div>
                <div class="card-body py-3">
                    <!-- Primary Domain -->
                    <div class="mb-3">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="fw-bold text-primary">Primary Domain: ${formattedPrimaryDomain}</span>
                            <span class="badge bg-primary">${primaryConfidencePercent}% confidence</span>
                        </div>
                        <div class="progress domain-confidence-progress">
                            <div class="progress-bar bg-primary" style="width: ${primaryConfidencePercent}%"></div>
                        </div>
                    </div>
    `;
    
    // Secondary Domains
    if (secondaryDomains.length > 0) {
        html += `
                    <div class="border-top pt-3">
                        <h6 class="small text-muted mb-3">Secondary Domain Predictions:</h6>
        `;
        
        secondaryDomains.forEach((secondary, index) => {
            const nameValue = secondary.domain || secondary.name || secondary;
            const confidence = secondary.confidence || secondary.score || 0;
            const formattedName = formatDomainLabel(nameValue);
            const confidencePercent = Math.round(confidence * 100);
            html += `
                        <div class="mb-2">
                            <div class="d-flex justify-content-between align-items-center mb-1">
                                <span class="small">${formattedName}</span>
                                <span class="small text-muted">${confidencePercent}%</span>
                            </div>
                            <div class="progress secondary-domain-progress">
                                <div class="progress-bar bg-secondary" style="width: ${confidencePercent}%"></div>
                            </div>
                        </div>
            `;
        });
        
        html += `
                    </div>
        `;
    }
    
    html += `
                </div>
            </div>
        </div>
    `;
    
    // Domain-specific recommendations from the analyzer
    if (recommendations.length > 0) {
        html += `
            <div class="mb-4">
                <h6 class="domain-recommendations-title"><i class="bi bi-lightbulb"></i> Domain-Specific Recommendations</h6>
                <div class="card border-success domain-recommendations-card">
                    <div class="card-body py-3">
                        <ul class="list-group list-group-flush domain-recommendations-list">
        `;
        
        recommendations.forEach((rec, index) => {
            html += `
                <li class="list-group-item border-0 py-2 px-0">
                    <div class="d-flex align-items-start">
                        <i class="bi bi-check-circle-fill text-success me-2 mt-1 flex-shrink-0"></i>
                        <span class="small">${rec}</span>
                    </div>
                </li>
            `;
        });
        
        html += `
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Analysis workflow recommendations based on primary domain
    const workflowRecommendations = getDomainSpecificWorkflow(primaryDomain);
    if (workflowRecommendations.length > 0) {
        html += `
            <div class="mb-4">
                <h6><i class="bi bi-diagram-3"></i> Recommended Analysis Workflow</h6>
                <div class="row">
        `;
        
        workflowRecommendations.forEach((step, index) => {
            html += `
                <div class="col-md-6 mb-2">
                    <div class="card border-light h-100 recommendation-card recommendation-card-clickable" onclick="applyRecommendation('${step.template}')">
                        <div class="card-body py-2">
                            <div class="d-flex align-items-center">
                                <div class="badge bg-primary rounded-pill me-2 flex-shrink-0">${index + 1}</div>
                                <div>
                                    <h6 class="card-title small mb-1">${step.title}</h6>
                                    <p class="card-text small text-muted mb-0">${step.description}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += `</div></div>`;
    }
    
    // Show domain scores if available
    if (Object.keys(allScores).length > 0) {
        const topScores = Object.entries(allScores)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5);
        const maxScore = topScores.length ? Math.max(...topScores.map(([, score]) => score)) : 1;
        const safeMaxScore = maxScore === 0 ? 1 : maxScore;
            
        html += `
            <div class="mb-3">
                <h6><i class="bi bi-bar-chart"></i> Domain Detection Scores</h6>
                <div class="small">
        `;
        
        topScores.forEach(([domain, score]) => {
            const percentage = (score / safeMaxScore) * 100;
            const formattedDomain = formatDomainLabel(domain);
            html += `
                <div class="mb-2">
                    <div class="d-flex justify-content-between">
                        <span>${formattedDomain}</span>
                        <span>${score.toFixed(1)}</span>
                    </div>
                    <div class="progress domain-score-progress">
                        <div class="progress-bar ${formatDomainLabel(domain) === formattedPrimaryDomain ? 'bg-primary' : 'bg-secondary'}" style="width: ${Math.min(percentage, 100)}%"></div>
                    </div>
                </div>
            `;
        });
        
        html += `</div></div>`;
    }
    
    recommendationsContent.innerHTML = html;
}

// Get domain-specific workflow recommendations
function getDomainSpecificWorkflow(domainName) {
    const workflows = {
        'fraud': [
            { title: "Data Quality Check", description: "Check for missing values and anomalies", template: "data_quality_structure" },
            { title: "Advanced Outlier Detection", description: "Detect fraudulent transactions using advanced methods", template: "advanced_outlier_detection" },
            { title: "Feature Analysis", description: "Analyze transaction patterns and amounts", template: "univariate_numeric" },
            { title: "Relationship Exploration", description: "Discover hidden fraud patterns using clustering", template: "relationship_exploration" },
            { title: "Correlation Analysis", description: "Find relationships between fraud indicators", template: "correlation_analysis" },
            { title: "Time Series Analysis", description: "Analyze fraud trends over time", template: "time_series_exploration" }
        ],
        'healthcare': [
            { title: "Data Quality Assessment", description: "Analyze healthcare data quality and completeness", template: "data_quality_structure" },
            { title: "Patient Demographics", description: "Analyze patient population characteristics", template: "univariate_categorical" },
            { title: "Treatment Outcomes", description: "Examine treatment effectiveness patterns", template: "distribution_analysis" },
            { title: "Medical Correlations", description: "Find relationships between symptoms and outcomes", template: "bivariate_multivariate" },
            { title: "Missing Data Analysis", description: "Check medical record completeness", template: "missing_data_analysis" },
            { title: "Pattern Discovery", description: "Discover patient clusters and treatment patterns", template: "relationship_exploration" }
        ],
        'finance': [
            { title: "Data Quality Assessment", description: "Comprehensive data quality validation", template: "data_quality_structure" },
            { title: "Financial Metrics Analysis", description: "Comprehensive numeric analysis of financial data", template: "univariate_numeric" },
            { title: "Time Series Analysis", description: "Analyze financial trends and patterns over time", template: "time_series_exploration" },
            { title: "Market Correlations", description: "Find relationships between financial metrics", template: "bivariate_multivariate" },
            { title: "Advanced Outlier Detection", description: "Detect anomalous financial transactions", template: "advanced_outlier_detection" },
            { title: "Pattern Discovery", description: "Discover market patterns using advanced analytics", template: "relationship_exploration" }
        ],
        'retail': [
            { title: "Data Quality Check", description: "Validate retail data integrity", template: "data_quality_structure" },
            { title: "Sales Patterns", description: "Analyze sales and customer behavior", template: "univariate_numeric" },
            { title: "Customer Analysis", description: "Product performance and categorical analysis", template: "univariate_categorical" },
            { title: "Time Series Analysis", description: "Identify seasonal patterns and trends", template: "time_series_exploration" },
            { title: "Customer Relationships", description: "Discover customer segments and buying patterns", template: "relationship_exploration" },
            { title: "Market Correlations", description: "Find relationships between products and sales", template: "bivariate_multivariate" }
        ],
        'marketing': [
            { title: "Data Quality Assessment", description: "Validate marketing data completeness", template: "data_quality_structure" },
            { title: "Campaign Analysis", description: "Analyze campaign performance metrics", template: "univariate_numeric" },
            { title: "Customer Segmentation", description: "Group customers by demographics and behavior", template: "univariate_categorical" },
            { title: "Multi-channel Analysis", description: "Analyze relationships between marketing channels", template: "bivariate_multivariate" },
            { title: "Customer Journey", description: "Track customer interactions over time", template: "time_series_exploration" },
            { title: "Pattern Discovery", description: "Discover hidden customer patterns and clusters", template: "relationship_exploration" }
        ],
        'iot': [
            { title: "Sensor Data Quality", description: "Check sensor data quality and completeness", template: "data_quality_structure" },
            { title: "Time Series Analysis", description: "Analyze sensor readings over time", template: "time_series_exploration" },
            { title: "Anomaly Detection", description: "Detect sensor anomalies and failures", template: "advanced_outlier_detection" },
            { title: "Sensor Correlations", description: "Find relationships between sensor readings", template: "bivariate_multivariate" },
            { title: "Pattern Discovery", description: "Discover operational patterns in IoT data", template: "relationship_exploration" },
            { title: "Missing Data Analysis", description: "Handle sensor data gaps and outages", template: "missing_data_analysis" }
        ]
    };
    
    return workflows[domainName] || [
        { title: "Data Quality Assessment", description: "Start with comprehensive data quality and structure analysis", template: "data_quality_structure" },
        { title: "Numeric Analysis", description: "Comprehensive analysis of numeric variables", template: "univariate_numeric" },
        { title: "Categorical Analysis", description: "Analyze categorical variables and distributions", template: "univariate_categorical" },
        { title: "Relationship Analysis", description: "Discover relationships between variables", template: "bivariate_multivariate" },
        { title: "Pattern Discovery", description: "Advanced pattern discovery and clustering", template: "relationship_exploration" },
        { title: "Outlier Detection", description: "Identify anomalies and outliers", template: "advanced_outlier_detection" },
        { title: "Time Series Analysis", description: "Analyze temporal patterns (if time data exists)", template: "time_series_exploration" }
    ];
}

// Display domain recommendations (legacy function - updated to use new approach)
function displayDomainRecommendations(result) {
    const recommendationsContent = document.getElementById('domainRecommendationsContent');
    if (!recommendationsContent) return;
    
    let html = '';
    
    // Analysis recommendations based on detected domain
    html += `
        <div class="mb-3">
            <h6><i class="bi bi-lightbulb"></i> Recommended Analysis Approaches</h6>
            <p class="text-muted small">Based on your detected domain type and data characteristics</p>
        </div>
    `;
    
    // Template-based recommendations
    if (result.template_name) {
        html += `
            <div class="mb-3">
                <div class="card border-primary">
                    <div class="card-header bg-primary text-white py-2">
                        <h6 class="mb-0">Primary Recommendation: ${result.template_name}</h6>
                    </div>
                    <div class="card-body py-2">
                        <p class="small mb-0">This template is most suitable for your dataset based on domain analysis.</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Generate specific recommendations based on common analysis patterns
    const recommendations = [
        {
            title: "Data Quality Assessment",
            description: "Start with comprehensive data quality and structure analysis",
            template: "data_quality_structure"
        },
        {
            title: "Numeric Variable Analysis", 
            description: "Analyze numeric variables with advanced statistics and distributions",
            template: "univariate_numeric"
        },
        {
            title: "Categorical Analysis",
            description: "Comprehensive analysis of categorical variables and their distributions", 
            template: "univariate_categorical"
        },
        {
            title: "Missing Data Analysis",
            description: "Identify missing values and data quality issues",
            template: "missing_data_analysis"
        },
        {
            title: "Correlation Analysis",
            description: "Discover relationships between variables",
            template: "correlation_analysis"
        },
        {
            title: "Distribution Analysis",
            description: "Analyze the distribution patterns of your variables",
            template: "distribution_analysis"
        }
    ];
    
    html += `
        <div class="mb-3">
            <h6><i class="bi bi-list-check"></i> Suggested Analysis Steps</h6>
            <div class="row">
    `;
    
    recommendations.forEach(rec => {
        html += `
            <div class="col-md-6 mb-2">
                <div class="card border-light h-100 recommendation-card recommendation-card-clickable" onclick="applyRecommendation('${rec.template}')">
                    <div class="card-body py-2">
                        <div class="d-flex align-items-center">
                            <div>
                                <h6 class="card-title small mb-1">${rec.title}</h6>
                                <p class="card-text small text-muted mb-0">${rec.description}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += `</div></div>`;
    
    // Additional domain-specific tips
    html += `
        <div class="mb-3">
            <h6><i class="bi bi-info-circle"></i> Tips for Better Analysis</h6>
            <ul class="small mb-0">
                <li>Start with data exploration to understand your dataset structure</li>
                <li>Check for missing values and outliers before advanced analysis</li>
                <li>Use appropriate visualization techniques for your data types</li>
                <li>Consider domain-specific metrics and KPIs</li>
            </ul>
        </div>
    `;
    
    recommendationsContent.innerHTML = html;
}

// Apply a recommendation (select analysis type and add cell)
function applyRecommendation(templateName) {
    console.log('Applying recommendation:', templateName);
    
    // Find and select the analysis type
    const analysisOption = document.querySelector(`[data-value="${templateName}"]`);
    if (analysisOption) {
        if (typeof selectAnalysisType === 'function') {
            selectAnalysisType(analysisOption);
        } else {
            triggerAnalysisRun(templateName, analysisOption);
        }
        
        // Show success message
        showNotification(`Applied recommendation: ${analysisOption.getAttribute('data-name')}`, 'success');
        
        // Close recommendations modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('domainRecommendationsModal'));
        if (modal) {
            modal.hide();
        }
        
        // Scroll to analysis types section after modal is hidden
        setTimeout(() => {
            const analysisGrid = document.getElementById('analysisGrid');
            if (analysisGrid) {
                analysisGrid.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 300);
    }
}

// Close domain recommendations section
function closeDomainRecommendations() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('domainRecommendationsModal'));
    if (modal) {
        modal.hide();
    }
}

// Export functions for global access
// ============================================================================
// COLUMN INSIGHTS FUNCTIONALITY
// ============================================================================

// Global variables for column insights
let columnInsightsData = null;


window.columnInsightsData = null;
window.columnInsightsLoading = false;
let selectedColumns = new Set();
let columnFilterState = {
    activeKey: 'all',
    type: 'all',
    showIssuesOnly: false,
    showSelectedOnly: false,
    search: ''
};
let columnFiltersInitialized = false;

function formatNumberValue(value) {
    if (value === null || value === undefined) {
        return null;
    }

    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
        return String(value);
    }

    try {
        return numeric.toLocaleString();
    } catch (error) {
        console.warn('Unable to format number value:', value, error);
        return String(value);
    }
}

function formatBytesValue(bytes) {
    if (bytes === null || bytes === undefined) {
        return null;
    }

    const size = Number(bytes);
    if (Number.isNaN(size)) {
        return null;
    }

    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let currentSize = size;
    let unitIndex = 0;

    while (currentSize >= 1024 && unitIndex < units.length - 1) {
        currentSize /= 1024;
        unitIndex += 1;
    }

    const precision = currentSize >= 100 || Number.isInteger(currentSize) ? 0 : currentSize >= 10 ? 1 : 2;
    return `${currentSize.toFixed(precision)} ${units[unitIndex]}`;
}

function formatRelativeTime(timestamp) {
    try {
        const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
        if (Number.isNaN(date.getTime())) {
            return '';
        }

        const diffMs = Date.now() - date.getTime();
        const diffMinutes = Math.round(diffMs / 60000);

        if (diffMinutes < 1) return 'just now';
        if (diffMinutes < 60) return `${diffMinutes} min${diffMinutes === 1 ? '' : 's'} ago`;

        const diffHours = Math.round(diffMinutes / 60);
        if (diffHours < 24) return `${diffHours} hr${diffHours === 1 ? '' : 's'} ago`;

        const diffDays = Math.round(diffHours / 24);
        if (diffDays < 7) return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`;

        const diffWeeks = Math.round(diffDays / 7);
        if (diffWeeks < 5) return `${diffWeeks} week${diffWeeks === 1 ? '' : 's'} ago`;

        const diffMonths = Math.round(diffDays / 30);
        if (diffMonths < 12) return `${diffMonths} month${diffMonths === 1 ? '' : 's'} ago`;

        const diffYears = Math.round(diffDays / 365);
        return `${diffYears} year${diffYears === 1 ? '' : 's'} ago`;
    } catch (error) {
        console.warn('Unable to format relative time:', error);
        return '';
    }
}

function formatTimestampDisplay(timestamp) {
    try {
        const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
        if (Number.isNaN(date.getTime())) {
            return '—';
        }

        const absolute = date.toLocaleString(undefined, {
            dateStyle: 'medium',
            timeStyle: 'short'
        });
        const relative = formatRelativeTime(date);
        return relative ? `${absolute} (${relative})` : absolute;
    } catch (error) {
        console.warn('Unable to format timestamp:', error);
        return '—';
    }
}

function updateSelectedColumnsBadge() {
    const selectedCountEl = document.getElementById('selectedColumns');
    if (!selectedCountEl) {
        return;
    }

    const value = selectedColumns.size;
    try {
        selectedCountEl.textContent = value.toLocaleString();
    } catch (error) {
        selectedCountEl.textContent = value;
    }
}

function initializeColumnFilters() {
    if (columnFiltersInitialized) {
        return;
    }

    const searchInput = document.getElementById('columnSearchInput');
    if (searchInput) {
        searchInput.addEventListener('input', (event) => {
            columnFilterState.search = event.target.value.trim().toLowerCase();
            renderColumnItems(columnInsightsData?.column_insights || []);
        });
    }

    document.querySelectorAll('.column-filter-button').forEach((button) => {
        button.addEventListener('click', () => {
            const filter = button.getAttribute('data-column-filter');
            columnFilterState.activeKey = filter;
            columnFilterState.showIssuesOnly = filter === 'issues';
            columnFilterState.showSelectedOnly = filter === 'selected';

            if (['numeric', 'text', 'datetime', 'boolean'].includes(filter)) {
                columnFilterState.type = filter;
                columnFilterState.showIssuesOnly = false;
                columnFilterState.showSelectedOnly = false;
            } else if (filter === 'issues' || filter === 'selected') {
                columnFilterState.type = 'all';
            } else {
                columnFilterState.type = 'all';
                columnFilterState.showIssuesOnly = false;
                columnFilterState.showSelectedOnly = false;
            }

            updateColumnFilterButtons(filter);
            renderColumnItems(columnInsightsData?.column_insights || []);
        });
    });

    updateColumnFilterButtons(columnFilterState.activeKey);
    columnFiltersInitialized = true;
}

function updateColumnFilterButtons(activeKey) {
    document.querySelectorAll('.column-filter-button').forEach((button) => {
        const filter = button.getAttribute('data-column-filter');
        button.classList.toggle('active', filter === activeKey);
    });
}

function renderColumnItems(columnInsights = []) {
    const columnItemsContainer = document.getElementById('columnItems');
    const emptyState = document.getElementById('columnEmptyState');

    if (!columnItemsContainer || !Array.isArray(columnInsights)) {
        return;
    }

    const filteredColumns = columnInsights.filter((col) => {
        if (col.dropped) {
            return false;
        }

        if (columnFilterState.search && !col.name.toLowerCase().includes(columnFilterState.search)) {
            return false;
        }

        if (columnFilterState.showIssuesOnly && !col.has_issues) {
            return false;
        }

        if (columnFilterState.showSelectedOnly && !selectedColumns.has(col.name)) {
            return false;
        }

        if (['numeric', 'text', 'datetime', 'boolean'].includes(columnFilterState.type)) {
            return col.data_category === columnFilterState.type;
        }

        return true;
    });

    columnItemsContainer.innerHTML = '';

    if (!filteredColumns.length) {
        if (emptyState) {
            emptyState.classList.remove('d-none');
        }
        return;
    }

    if (emptyState) {
        emptyState.classList.add('d-none');
    }

    filteredColumns.forEach((col) => {
        const index = typeof col.__index === 'number' ? col.__index : columnInsights.indexOf(col);
        const columnElement = createColumnItem(col, index >= 0 ? index : 0);
        columnItemsContainer.appendChild(columnElement);
    });
}

function renderColumnRecommendations(recommendations = []) {
    const container = document.getElementById('columnRecommendationList');
    const wrapper = document.getElementById('columnRecommendations');

    if (!container || !wrapper) {
        return;
    }

    if (!recommendations.length) {
        wrapper.classList.add('d-none');
        container.innerHTML = '';
        return;
    }

    const recommendationHtml = recommendations.slice(0, 4).map((rec) => {
        const type = rec.type || 'info';
        const icon = type === 'warning' ? 'bi-exclamation-triangle-fill' : 'bi-info-circle-fill';
        return `
            <div class="column-recommendation column-recommendation--${type}">
                <i class="bi ${icon}"></i>
                <div>
                    <strong>${rec.title || 'Notice'}</strong>
                    <div>${rec.description || rec}</div>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = recommendationHtml;
    wrapper.classList.remove('d-none');
}

// Column insights - expansion functionality removed per user request

function toggleColumnInsightsSize(forceState) {
    // Expansion functionality removed - button will remain for UI consistency but won't expand
    console.log('Column expansion disabled per user request');
}

/**
 * Load column insights data from the API
 */
async function loadColumnInsights() {
    if (window.columnInsightsLoading) {
        console.log('Column insights load already in progress, skipping new request');
        return;
    }

    console.log('Loading column insights...');
    window.columnInsightsLoading = true;

    const sourceId = initSourceId();
    if (!sourceId) {
        console.error('No source ID available for column insights');
        showColumnInsightsError();
        window.columnInsightsLoading = false;
        return;
    }

    try {
        showColumnInsightsLoading();
        
        const response = await fetch(`/advanced-eda/api/column-insights/${sourceId}`);
        const data = await response.json();
        
        if (data.success) {
            columnInsightsData = data;
            window.columnInsightsData = columnInsightsData;
            selectedColumns.clear();
            
            // Initialize selected columns (default to all selected) and preserve ordering
            data.column_insights.forEach((col, index) => {
                col.__index = index;
                if (col.selected) {
                    selectedColumns.add(col.name);
                }
            });
            
            if (data.preprocessing_report) {
                handlePreprocessingReport(data.preprocessing_report);
            } else {
                displayColumnInsights(data);
                updatePreprocessingPreview();
                updatePreprocessingStatusBadge();
                refreshCategoryLocks();
            }
        } else {
            console.error('Failed to load column insights:', data.error);
            showColumnInsightsError();
        }
    } catch (error) {
        console.error('Error loading column insights:', error);
        showColumnInsightsError();
        window.columnInsightsData = null;
    }
    finally {
        window.columnInsightsLoading = false;
    }
}

/**
 * Display column insights data in the UI
 */
function displayColumnInsights(data) {
    console.log('Displaying column insights:', data);
    
    const summaryStats = data.summary_stats || {};
    const columnInsights = data.column_insights || [];

    const summaryLoading = document.getElementById('columnSummaryLoading');
    const summaryContent = document.getElementById('columnSummaryContent');
    const summaryError = document.getElementById('columnSummaryError');
    if (summaryLoading) {
        summaryLoading.classList.add('d-none');
    }
    if (summaryContent) {
        summaryContent.classList.remove('d-none');
    }
    if (summaryError) {
        summaryError.classList.add('d-none');
    }

    const insightsLoading = document.getElementById('columnInsightsLoading');
    const insightsContent = document.getElementById('columnInsightsContent');
    const insightsError = document.getElementById('columnInsightsError');
    if (insightsLoading) {
        insightsLoading.classList.add('d-none');
    }
    if (insightsContent) {
        insightsContent.classList.remove('d-none');
    }
    if (insightsError) {
        insightsError.classList.add('d-none');
    }

    const activeColumns = Array.isArray(columnInsights)
        ? columnInsights.filter(col => !col.dropped)
        : [];

    const missingDataColumns = summaryStats.missing_data_columns ?? activeColumns.filter(col => {
        return typeof col.null_percentage === 'number' && col.null_percentage > 0;
    }).length;

    // Update dataset snapshot information when available
    const datasetName = data.dataset_name || summaryStats.dataset_name || data.source_name || null;
    const datasetNameEl = document.getElementById('dataset-name');
    const datasetHeadingEl = document.getElementById('datasetName');

    if (datasetNameEl && datasetName) {
        datasetNameEl.textContent = datasetName;
    }
    if (datasetHeadingEl && datasetName) {
        datasetHeadingEl.textContent = datasetName;
    }

    const datasetCols = summaryStats.total_columns ?? data.total_columns ?? activeColumns.length;

    const numericColumns = summaryStats.numeric_columns ?? activeColumns.filter(col => col.data_category === 'numeric').length;
    const textColumns = summaryStats.text_columns ?? activeColumns.filter(col => col.data_category === 'text').length;
    const datetimeColumns = summaryStats.datetime_columns ?? activeColumns.filter(col => col.data_category === 'datetime').length;
    const booleanColumns = summaryStats.boolean_columns ?? activeColumns.filter(col => col.data_category === 'boolean').length;
    const problematicColumns = summaryStats.problematic_columns ?? activeColumns.filter(col => col.has_issues).length;
    const missingRatio = datasetCols ? Math.round((missingDataColumns / datasetCols) * 100) : 0;
    const otherColumns = Math.max(datasetCols - (numericColumns + textColumns + datetimeColumns + booleanColumns), 0);

    const totalColumnsEl = document.getElementById('totalColumns');
    if (totalColumnsEl) {
        totalColumnsEl.textContent = formatNumberValue(datasetCols) || 0;
    }

    updateSelectedColumnsBadge();

    const problematicColumnsEl = document.getElementById('problematicColumns');
    if (problematicColumnsEl) {
        problematicColumnsEl.textContent = formatNumberValue(problematicColumns) || 0;
    }

    const missingRatioEl = document.getElementById('missingRatioLabel');
    if (missingRatioEl) {
        missingRatioEl.textContent = datasetCols ? `${missingRatio}%` : '—';
    }

    const tagLabel = (value, label) => `${formatNumberValue(value) || 0} ${label}`;
    const numericTag = document.getElementById('numericColumns');
    if (numericTag) {
        numericTag.textContent = tagLabel(numericColumns, 'numeric');
    }

    const textTag = document.getElementById('textColumns');
    if (textTag) {
        textTag.textContent = tagLabel(textColumns, 'text');
    }

    const datetimeTag = document.getElementById('datetimeColumns');
    if (datetimeTag) {
        datetimeTag.textContent = tagLabel(datetimeColumns, 'datetime');
    }

    const booleanTag = document.getElementById('booleanColumns');
    if (booleanTag) {
        booleanTag.textContent = tagLabel(booleanColumns, 'boolean');
    }

    const otherTag = document.getElementById('otherColumns');
    if (otherTag) {
        otherTag.textContent = tagLabel(otherColumns, 'other');
    }

    renderColumnRecommendations(data.recommendations || []);
    initializeColumnFilters();
    updateColumnFilterButtons(columnFilterState.activeKey || 'all');
    renderColumnItems(columnInsights);
}

/**
 * Create a column item element with enhanced information display
 */
function createColumnItem(col, index) {
    const div = document.createElement('div');
    const isSelected = selectedColumns.has(col.name);
    const issueTypes = Array.isArray(col.issue_types) ? col.issue_types : [];
    const issueMessages = Array.isArray(col.issue_messages) ? col.issue_messages : [];
    const hasIssues = Boolean(col.has_issues || issueTypes.length > 0);
    const nullPercentage = col.null_percentage || 0;
    const isDropPreview = Boolean(col.drop_preview);
    
    // Determine border and background class based on issues and missing data
    let borderClass = 'border';
    let itemClass = 'column-item';
    
    if (hasIssues) {
        if (issueTypes.includes('error')) {
            borderClass = 'border border-danger';
            itemClass += ' column-item-error';
        } else if (issueTypes.includes('warning') || issueTypes.includes('missing_data')) {
            borderClass = 'border border-warning';
            itemClass += ' column-item-warning';
        }
    } else if (nullPercentage > 15) {
        // Highlight columns with moderate missing data even if not flagged as issues
        borderClass = 'border border-info';
        itemClass += ' column-item-info';
    }

    if (isDropPreview) {
        itemClass += ' column-item-drop-preview';
    }
    
    // Determine status icon with more variety
    let statusIcon = '<i class="bi bi-check-circle text-success" title="No issues detected"></i>';
    if (hasIssues) {
        if (issueTypes.includes('error')) {
            statusIcon = '<i class="bi bi-x-circle text-danger" title="Critical issues found"></i>';
        } else if (issueTypes.includes('warning')) {
            statusIcon = '<i class="bi bi-exclamation-triangle text-warning" title="Potential issues found"></i>';
        } else if (issueTypes.includes('missing_data')) {
            statusIcon = '<i class="bi bi-exclamation-circle text-warning" title="High missing data"></i>';
        }
    } else if (nullPercentage > 15) {
        statusIcon = '<i class="bi bi-info-circle text-info" title="Moderate missing data"></i>';
    }
    
    // Create missing data badge if significant
    let missingDataBadge = '';
    if (nullPercentage > 0) {
        let badgeClass = 'badge-light';
        if (nullPercentage > 50) {
            badgeClass = 'badge-danger';
        } else if (nullPercentage > 30) {
            badgeClass = 'badge-warning';
        } else if (nullPercentage > 15) {
            badgeClass = 'badge-info';
        }
        missingDataBadge = `<span class="badge ${badgeClass} ms-1" title="${nullPercentage.toFixed(1)}% missing data">${nullPercentage.toFixed(1)}%</span>`;
    }

    const statusText = (isDropPreview && col.preview_message) ? col.preview_message : (col.status_display || 'No data quality alerts');
    const statusClass = (hasIssues && nullPercentage > 30) || isDropPreview ? 'text-warning' : 'text-muted';
    
    div.className = `${itemClass} ${borderClass} p-3 rounded column-card`;
    div.innerHTML = `
        <div class="form-check">
            <input class="form-check-input" type="checkbox" ${isSelected ? 'checked' : ''} 
                   id="col${index}" onchange="toggleColumnSelection('${col.name}')">
            <label class="form-check-label w-100" for="col${index}">
                <div class="d-flex justify-content-between align-items-start column-card-header">
                    <div class="flex-grow-1">
                        <div class="d-flex align-items-center mb-1">
                            <strong class="column-name">${col.name}</strong>
                            ${missingDataBadge}
                        </div>
                        <small class="d-block text-muted column-type">
                            ${col.type_display}
                        </small>
                        <small class="d-block ${statusClass} column-stats">
                            ${statusText}
                        </small>
                        ${isDropPreview && !issueMessages.length ? `
                            <div class="mt-1">
                                <small class="badge badge-outline-warning me-1">${col.preview_message}</small>
                            </div>
                        ` : ''}
                        ${issueMessages.length > 0 ? `
                            <div class="mt-1">
                                ${issueMessages.slice(0, 3).map(msg => 
                                    `<small class="badge badge-outline-warning me-1">${msg}</small>`
                                ).join('')}
                            </div>
                        ` : ''}
                    </div>
                    <div class="column-status-icon">
                        ${statusIcon}
                    </div>
                </div>
            </label>
        </div>
    `;
    
    return div;
}

/**
 * Toggle column selection
 */
function toggleColumnSelection(columnName) {
    console.log('Toggling column selection:', columnName);
    
    if (selectedColumns.has(columnName)) {
        selectedColumns.delete(columnName);
    } else {
        selectedColumns.add(columnName);
    }
    
    // Update selected count
    updateSelectedColumnsBadge();
    
    console.log('Selected columns:', Array.from(selectedColumns));
}

/**
 * Select all columns
 */
function selectAllColumns() {
    console.log('Selecting all columns');
    
    if (!columnInsightsData) return;
    
    selectedColumns.clear();
    columnInsightsData.column_insights.forEach(col => {
        if (!col.dropped) {
            selectedColumns.add(col.name);
        }
    });
    
    // Update checkboxes
    document.querySelectorAll('#columnItems input[type="checkbox"]').forEach(checkbox => {
        checkbox.checked = true;
    });
    
    // Update count
    updateSelectedColumnsBadge();
}

/**
 * Deselect all columns
 */
function deselectAllColumns() {
    console.log('Deselecting all columns');
    
    selectedColumns.clear();
    
    // Update checkboxes
    document.querySelectorAll('#columnItems input[type="checkbox"]').forEach(checkbox => {
        checkbox.checked = false;
    });
    
    // Update count
    updateSelectedColumnsBadge();
}

/**
 * Select only recommended columns (columns without issues)
 */
function selectRecommendedColumns() {
    console.log('Selecting recommended columns');
    
    if (!columnInsightsData) return;
    
    selectedColumns.clear();
    
    // Select columns without issues
    columnInsightsData.column_insights.forEach(col => {
        if (!col.dropped && !col.has_issues) {
            selectedColumns.add(col.name);
        }
    });
    
    // Update checkboxes
    document.querySelectorAll('#columnItems input[type="checkbox"]').forEach(checkbox => {
        const columnName = checkbox.getAttribute('onchange').match(/'([^']+)'/)[1];
        checkbox.checked = selectedColumns.has(columnName);
    });
    
    // Update count
    updateSelectedColumnsBadge();
}

/**
 * Refresh column insights data
 */
function refreshColumnInsights() {
    console.log('Refreshing column insights');
    loadColumnInsights();
}

/**
 * Get currently selected columns
 */
function getSelectedColumns() {
    return Array.from(selectedColumns);
}

/**
 * Show loading state
 */
function showColumnInsightsLoading() {
    const summaryLoading = document.getElementById('columnSummaryLoading');
    const summaryContent = document.getElementById('columnSummaryContent');
    const summaryError = document.getElementById('columnSummaryError');
    if (summaryLoading) {
        summaryLoading.classList.remove('d-none');
    }
    if (summaryContent) {
        summaryContent.classList.add('d-none');
    }
    if (summaryError) {
        summaryError.classList.add('d-none');
    }

    const loading = document.getElementById('columnInsightsLoading');
    const content = document.getElementById('columnInsightsContent');
    const error = document.getElementById('columnInsightsError');
    if (loading) {
        loading.classList.remove('d-none');
    }
    if (content) {
        content.classList.add('d-none');
    }
    if (error) {
        error.classList.add('d-none');
    }
}

/**
 * Show error state
 */
function showColumnInsightsError() {
    const summaryLoading = document.getElementById('columnSummaryLoading');
    const summaryContent = document.getElementById('columnSummaryContent');
    const summaryError = document.getElementById('columnSummaryError');
    if (summaryLoading) {
        summaryLoading.classList.add('d-none');
    }
    if (summaryContent) {
        summaryContent.classList.add('d-none');
    }
    if (summaryError) {
        summaryError.classList.remove('d-none');
    }

    const loading = document.getElementById('columnInsightsLoading');
    const content = document.getElementById('columnInsightsContent');
    const error = document.getElementById('columnInsightsError');
    if (loading) {
        loading.classList.add('d-none');
    }
    if (content) {
        content.classList.add('d-none');
    }
    if (error) {
        error.classList.remove('d-none');
    }
}

// ============================================================================
// WINDOW EXPORTS AND INITIALIZATION
// ============================================================================

window.onAnalysisTypeChange = onAnalysisTypeChange;
window.addAnalysisCell = addAnalysisCell;
window.addMarkdownCell = addMarkdownCell;
window.clearAllCells = clearAllCells;
window.clearSelection = clearSelection;
window.selectAnalysisType = selectAnalysisType;
window.triggerAnalysisRun = triggerAnalysisRun;
window.addSelectedAnalysisCells = addSelectedAnalysisCells;
window.addSelectedAnalysisTypes = addSelectedAnalysisTypes;
window.toggleAnalysisSelection = toggleAnalysisSelection;
window.updateSelectedAnalysesVisibility = updateSelectedAnalysesVisibility;
window.addBadge = addBadge;
window.removeBadge = removeBadge;
window.removeBadgeAndDeselect = removeBadgeAndDeselect;
window.openAnalysisCode = openAnalysisCode;
window.openAnalysisCodeFromCell = openAnalysisCodeFromCell;
window.copyAnalysisCodeToClipboard = copyAnalysisCodeToClipboard;
window.rerunAnalysis = rerunAnalysis;
window.deleteCell = deleteCell;
window.editMarkdownCell = editMarkdownCell;
window.moveCellUp = moveCellUp;
window.moveCellDown = moveCellDown;
window.handleDomainButtonClick = handleDomainButtonClick;
window.loadDomainDetection = loadDomainDetection;
window.loadDomainRecommendations = loadDomainRecommendations;
window.forceRefreshDomainDetection = forceRefreshDomainDetection;
window.closeDomainRecommendations = closeDomainRecommendations;
window.applyRecommendation = applyRecommendation;

// Column insights exports
window.loadColumnInsights = loadColumnInsights;
window.refreshColumnInsights = refreshColumnInsights;
window.toggleColumnSelection = toggleColumnSelection;
window.selectAllColumns = selectAllColumns;
window.deselectAllColumns = deselectAllColumns;
window.selectRecommendedColumns = selectRecommendedColumns;
window.getSelectedColumns = getSelectedColumns;
// Export functions globally for debugging
// window.toggleColumnInsightsSize removed

// Debug functions removed

// Initialize when DOM is loaded


document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing notebook interface');
    initSourceId();
    initAnalysisGrid();
    initializePreprocessingState();
    initializeCategoricalModal();
    attachCategoricalModalLifecycleHandlers();
    initializeNumericModal();
    attachNumericModalLifecycleHandlers();
    initializeCrossTabModal();
    attachCrossTabModalLifecycleHandlers();
    initializeGeospatialModal();
    attachGeospatialModalLifecycleHandlers();
    initializeTimeSeriesModal();
    attachTimeSeriesModalLifecycleHandlers();
    updateAnalysisResultsPlaceholder();

    if (sourceId) {
        loadDomainDetection().catch(error => {
            console.error('Initial domain detection failed', error);
        });
    } else {
        console.warn('No source_id found in URL parameters');
        showNotification('Warning: No dataset source ID found', 'warning');
    }
    
    // Load column insights automatically
    setTimeout(() => {
        loadColumnInsights();
    }, 500); // Small delay to ensure other components are loaded
});

// Also initialize immediately if DOM is already loaded
if (document.readyState === 'loading') {
    // Do nothing, DOMContentLoaded will fire
} else {
    // DOM is already loaded
    console.log('DOM already loaded, initializing notebook interface immediately');
    initSourceId();
    initAnalysisGrid();
    initializePreprocessingState();
    initializeCategoricalModal();
    attachCategoricalModalLifecycleHandlers();
    initializeNumericModal();
    attachNumericModalLifecycleHandlers();
    initializeCrossTabModal();
    attachCrossTabModalLifecycleHandlers();
    initializeGeospatialModal();
    attachGeospatialModalLifecycleHandlers();
    initializeTimeSeriesModal();
    attachTimeSeriesModalLifecycleHandlers();
    updateAnalysisResultsPlaceholder();

    if (sourceId) {
        loadDomainDetection().catch(error => {
            console.error('Initial domain detection failed', error);
        });
    } else {
        console.warn('No source_id found in URL parameters');
        showNotification('Warning: No dataset source ID found', 'warning');
    }
    
    // Load column insights automatically
    setTimeout(() => {
        loadColumnInsights();
    }, 500);
}

// ============================================
// CATEGORICAL ANALYSIS MODAL FUNCTIONALITY
// ============================================



async function prepareCategoricalAnalysisConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    currentCategoricalAnalysisType = analysisType;
    categoricalModalConfirmed = false;
    categoricalModalSelection = new Set();
    categoricalModalColumns = [];
    categoricalModalRecommendedDefaults = [];
    categoricalModalSearchTerm = '';
    currentCategoricalCellId = '';

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before categorical modal:', error);
        }
    }

    const columnCandidates = getCategoricalColumnCandidates();
    if (!columnCandidates.length) {
        showNotification('No categorical-style columns detected. Running full analysis instead.', 'info');
        await addSingleAnalysisCell(analysisType);
        return;
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipCategoricalModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare categorical analysis cell. Please try again.', 'error');
        return;
    }

    currentCategoricalCellId = cellId;
    categoricalModalColumns = columnCandidates;

    initializeCategoricalModal();
    attachCategoricalModalLifecycleHandlers();
    populateCategoricalModal(analysisType, columnCandidates);

    const modalElement = document.getElementById('categoricalAnalysisModal');
    if (!modalElement) {
        showNotification('Categorical configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
    showNotification(`Select categorical columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
}

function attachCategoricalModalLifecycleHandlers() {
    const modalElement = document.getElementById('categoricalAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleCategoricalModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleCategoricalModalHidden() {
    if (categoricalModalConfirmed) {
        categoricalModalConfirmed = false;
        categoricalModalIsRerun = false;
        return;
    }

    if (categoricalModalIsRerun) {
        categoricalModalIsRerun = false;
        currentCategoricalCellId = '';
        currentCategoricalAnalysisType = '';
        categoricalModalSelection = new Set();
        categoricalModalRecommendedDefaults = [];
        categoricalModalColumns = [];
        categoricalModalSearchTerm = '';
        showNotification('Categorical analysis rerun cancelled.', 'info');
        return;
    }

    if (currentCategoricalCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentCategoricalCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        if (currentCategoricalAnalysisType) {
            showNotification('Categorical analysis cancelled.', 'info');
        }
    }

    currentCategoricalCellId = '';
    currentCategoricalAnalysisType = '';
    categoricalModalSelection = new Set();
    categoricalModalRecommendedDefaults = [];
    categoricalModalColumns = [];
    categoricalModalSearchTerm = '';
}

function initializeCategoricalModal() {
    const modalElement = document.getElementById('categoricalAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const confirmBtn = document.getElementById('categoricalModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchCategoricalAnalysis);
    }

    const selectAllBtn = document.getElementById('categoricalModalSelectAllBtn');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', selectAllCategoricalColumns);
    }

    const clearBtn = document.getElementById('categoricalModalClearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            categoricalModalSelection = new Set();
            renderCategoricalColumnList();
            updateCategoricalSelectionSummary();
            updateCategoricalChipStates();
        });
    }

    const recommendedBtn = document.getElementById('categoricalModalRecommendBtn');
    if (recommendedBtn) {
        recommendedBtn.addEventListener('click', () => {
            applyCategoricalRecommendations();
        });
    }

    const searchInput = document.getElementById('categoricalColumnSearch');
    if (searchInput) {
        searchInput.addEventListener('input', event => {
            categoricalModalSearchTerm = (event.target.value || '').toLowerCase();
            renderCategoricalColumnList();
            updateCategoricalChipStates();
        });
    }

    const columnList = document.getElementById('categoricalColumnList');
    if (columnList) {
        columnList.addEventListener('change', handleCategoricalListChange);
    }

    const chipsContainer = document.getElementById('categoricalRecommendationChips');
    if (chipsContainer) {
        chipsContainer.addEventListener('click', handleCategoricalChipClick);
    }

    modalElement.dataset.initialized = 'true';
}

function toFiniteNumber(value) {
    if (value === null || value === undefined) {
        return null;
    }
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}

function getCategoricalColumnCandidates() {
    if (!Array.isArray(columnInsightsData?.column_insights)) {
        return [];
    }

    const columns = [];

    columnInsightsData.column_insights.forEach(col => {
        if (!col || col.dropped) {
            return;
        }

        const normalizedName = (col.name ?? '').toString().trim();
        if (!normalizedName) {
            return;
        }

        const stats = typeof col.statistics === 'object' && col.statistics !== null ? col.statistics : {};
        const uniqueCount = toFiniteNumber(stats.unique_count ?? stats.distinct_count ?? stats.cardinality);
        const uniqueRatioRaw = toFiniteNumber(stats.unique_ratio ?? stats.distinct_ratio);
        const uniqueRatio = uniqueRatioRaw !== null && uniqueRatioRaw > 1 ? uniqueRatioRaw / 100 : uniqueRatioRaw;
        const missingPct = toFiniteNumber(col.null_percentage);
        const topValues = Array.isArray(stats.top_values) ? stats.top_values : [];
        const dataCategory = (col.data_category || '').toString().toLowerCase();
        const dataType = (col.data_type || '').toString().toLowerCase();

        const categoricalCategories = new Set(['text', 'categorical', 'category', 'bool', 'boolean']);
        const categoricalTypes = new Set(['object', 'category', 'bool', 'boolean']);

        let isCandidate = categoricalCategories.has(dataCategory) || categoricalTypes.has(dataType);

        if (!isCandidate && dataCategory === 'numeric') {
            if (typeof uniqueCount === 'number' && uniqueCount > 0 && uniqueCount <= 15) {
                isCandidate = true;
            } else if (typeof uniqueRatio === 'number' && uniqueRatio > 0 && uniqueRatio <= 0.1) {
                isCandidate = true;
            }
        }

        if (!isCandidate) {
            return;
        }

        const recommended = Boolean(
            (typeof uniqueCount === 'number' && uniqueCount > 0 && uniqueCount <= 12) ||
                (typeof uniqueRatio === 'number' && uniqueRatio > 0 && uniqueRatio <= 0.2) ||
                categoricalCategories.has(dataCategory) && dataCategory !== 'text' ||
                categoricalTypes.has(dataType) && dataType !== 'object' ||
                (topValues.length > 0 && topValues.length <= 12)
        );

        const reasonParts = [];
        if (typeof uniqueCount === 'number') {
            reasonParts.push(`${uniqueCount} ${uniqueCount === 1 ? 'category' : 'categories'}`);
        }
        if (typeof uniqueRatio === 'number') {
            const ratioPct = uniqueRatio <= 1 ? uniqueRatio * 100 : uniqueRatio;
            reasonParts.push(`${ratioPct.toFixed(ratioPct < 5 ? 1 : 0)}% unique`);
        }
        if (typeof missingPct === 'number') {
            reasonParts.push(`${missingPct.toFixed(1)}% missing`);
        }

        columns.push({
            name: normalizedName,
            dataCategory: dataCategory || dataType || 'categorical',
            dataType,
            uniqueCount: typeof uniqueCount === 'number' ? uniqueCount : null,
            uniqueRatio: typeof uniqueRatio === 'number' ? uniqueRatio : null,
            missingPct: typeof missingPct === 'number' ? missingPct : null,
            recommended,
            reason: reasonParts.join(' • ')
        });
    });

    columns.sort((a, b) => {
        if (a.recommended !== b.recommended) {
            return a.recommended ? -1 : 1;
        }
        const aCount = typeof a.uniqueCount === 'number' ? a.uniqueCount : Number.POSITIVE_INFINITY;
        const bCount = typeof b.uniqueCount === 'number' ? b.uniqueCount : Number.POSITIVE_INFINITY;
        if (aCount !== bCount) {
            return aCount - bCount;
        }
        return a.name.localeCompare(b.name);
    });

    return columns;
}

function populateCategoricalModal(analysisType, columns, initialSelection = null) {
    const modalLabel = document.getElementById('categoricalAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.textContent = `Configure ${getAnalysisTypeName(analysisType)}`;
    }

    const subtitle = document.getElementById('categoricalModalSubtitle');
    if (subtitle) {
        subtitle.textContent = 'Pick the categorical columns to analyse. Keeping the list small speeds up frequency tables and charts.';
    }

    const confirmBtn = document.getElementById('categoricalModalConfirmBtn');
    if (confirmBtn) {
        const baseLabel = `Run ${getAnalysisTypeName(analysisType)}`;
        confirmBtn.dataset.baseLabel = baseLabel;
        confirmBtn.textContent = baseLabel;
        confirmBtn.disabled = false;
    }

    const searchInput = document.getElementById('categoricalColumnSearch');
    if (searchInput) {
        searchInput.value = '';
    }

    categoricalModalColumns = Array.isArray(columns) ? [...columns] : [];

    const recommended = categoricalModalColumns.filter(col => col.recommended);
    categoricalModalRecommendedDefaults = recommended.slice(0, 6).map(col => col.name);

    const normalizedInitialSelection = Array.isArray(initialSelection) || initialSelection instanceof Set
        ? Array.from(new Set(initialSelection instanceof Set ? [...initialSelection] : initialSelection)).filter(name =>
            categoricalModalColumns.some(col => col.name === name)
        )
        : [];

    if (normalizedInitialSelection.length > 0) {
        categoricalModalSelection = new Set(normalizedInitialSelection);
    } else if (categoricalModalRecommendedDefaults.length > 0) {
        categoricalModalSelection = new Set(categoricalModalRecommendedDefaults);
    } else {
        const fallbackDefaults = categoricalModalColumns.slice(0, Math.min(5, categoricalModalColumns.length)).map(col => col.name);
        categoricalModalSelection = new Set(fallbackDefaults);
    }

    renderCategoricalRecommendations(recommended);
    renderCategoricalColumnList();
    updateCategoricalChipStates();
    updateCategoricalSelectionSummary();
}

function renderCategoricalRecommendations(recommendedColumns) {
    const container = document.getElementById('categoricalRecommendationChips');
    if (!container) {
        return;
    }

    container.innerHTML = '';

    if (!recommendedColumns.length) {
        container.innerHTML = '<span class="text-muted small">No automatic recommendations yet. Select columns manually below.</span>';
        return;
    }

    recommendedColumns.forEach(col => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-outline-primary btn-sm me-2 mb-2';
        button.dataset.column = col.name;
        if (col.reason) {
            button.title = col.reason;
        }
        button.textContent = col.name;
        container.appendChild(button);
    });
}

function renderCategoricalColumnList() {
    const listElement = document.getElementById('categoricalColumnList');
    if (!listElement) {
        return;
    }

    let filtered = [...categoricalModalColumns];
    if (categoricalModalSearchTerm) {
        filtered = filtered.filter(col => col.name.toLowerCase().includes(categoricalModalSearchTerm));
    }

    if (!filtered.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No columns match your search.</div>';
        return;
    }

    const rows = filtered
        .map(col => {
            const checked = categoricalModalSelection.has(col.name) ? 'checked' : '';
            const detailParts = [];
            if (typeof col.uniqueCount === 'number') {
                detailParts.push(`${col.uniqueCount} ${col.uniqueCount === 1 ? 'category' : 'categories'}`);
            }
            if (typeof col.missingPct === 'number') {
                detailParts.push(`${col.missingPct.toFixed(1)}% missing`);
            }
            const detailText = detailParts.length ? `<small class="text-muted">${detailParts.join(' • ')}</small>` : '';
            const badgeLabel = col.dataCategory || col.dataType || 'categorical';

            return `
                <label class="list-group-item d-flex align-items-start gap-3">
                    <input class="form-check-input mt-1" type="checkbox" value="${escapeHtml(col.name)}" data-column="${escapeHtml(col.name)}" ${checked}>
                    <div class="flex-grow-1">
                        <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                            <strong>${escapeHtml(col.name)}</strong>
                            <span class="badge text-bg-light text-capitalize">${escapeHtml(badgeLabel)}</span>
                        </div>
                        ${detailText}
                    </div>
                </label>
            `;
        })
        .join('');

    listElement.innerHTML = rows;
}

function updateCategoricalSelectionSummary() {
    const summaryElement = document.getElementById('categoricalSelectionSummary');
    const confirmBtn = document.getElementById('categoricalModalConfirmBtn');
    const count = categoricalModalSelection.size;

    if (summaryElement) {
        if (count === 0) {
            summaryElement.textContent = 'Select one or more categorical columns to continue.';
        } else {
            const preview = Array.from(categoricalModalSelection).slice(0, 5);
            const remainder = count - preview.length;
            const suffix = remainder > 0 ? `, +${remainder} more` : '';
            summaryElement.textContent = `${count} column${count === 1 ? '' : 's'} selected: ${preview.join(', ')}${suffix}`;
        }
    }

    if (confirmBtn) {
        const baseLabel = confirmBtn.dataset.baseLabel || confirmBtn.textContent || 'Run analysis';
        confirmBtn.textContent = count > 0 ? `${baseLabel} (${count})` : baseLabel;
        confirmBtn.disabled = count === 0;
    }
}

function updateCategoricalChipStates() {
    const chipsContainer = document.getElementById('categoricalRecommendationChips');
    if (!chipsContainer) {
        return;
    }
    chipsContainer.querySelectorAll('[data-column]').forEach(button => {
        const columnName = button.dataset.column;
        button.classList.toggle('active', categoricalModalSelection.has(columnName));
    });
}

function handleCategoricalListChange(event) {
    const checkbox = event.target;
    if (!checkbox || checkbox.tagName !== 'INPUT' || checkbox.type !== 'checkbox') {
        return;
    }

    const columnName = checkbox.dataset.column;
    if (!columnName) {
        return;
    }

    if (checkbox.checked) {
        categoricalModalSelection.add(columnName);
    } else {
        categoricalModalSelection.delete(columnName);
    }

    updateCategoricalSelectionSummary();
    updateCategoricalChipStates();
}

function handleCategoricalChipClick(event) {
    const button = event.target.closest('[data-column]');
    if (!button) {
        return;
    }
    event.preventDefault();

    const columnName = button.dataset.column;
    if (!columnName) {
        return;
    }

    if (categoricalModalSelection.has(columnName)) {
        categoricalModalSelection.delete(columnName);
    } else {
        categoricalModalSelection.add(columnName);
    }

    renderCategoricalColumnList();
    updateCategoricalSelectionSummary();
    updateCategoricalChipStates();
}

function applyCategoricalRecommendations() {
    if (!categoricalModalRecommendedDefaults.length) {
        showNotification('No recommended categorical columns available yet.', 'info');
        return;
    }

    categoricalModalSelection = new Set(categoricalModalRecommendedDefaults);
    renderCategoricalColumnList();
    updateCategoricalSelectionSummary();
    updateCategoricalChipStates();
}

function selectAllCategoricalColumns() {
    if (!Array.isArray(categoricalModalColumns) || categoricalModalColumns.length === 0) {
        showNotification('No categorical columns available yet.', 'info');
        return;
    }

    categoricalModalSelection = new Set(categoricalModalColumns.map(col => col.name));
    renderCategoricalColumnList();
    updateCategoricalSelectionSummary();
    updateCategoricalChipStates();
}

async function openCategoricalModalForRerun(cellId, analysisType, previousSelection = []) {
    categoricalModalIsRerun = true;
    currentCategoricalCellId = cellId;
    currentCategoricalAnalysisType = analysisType;
    categoricalModalConfirmed = false;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before categorical rerun:', error);
        }
    }

    const columnCandidates = getCategoricalColumnCandidates();
    if (!columnCandidates.length) {
        categoricalModalIsRerun = false;
        showNotification('Categorical columns are unavailable for rerun.', 'warning');
        return;
    }

    categoricalModalColumns = columnCandidates;

    initializeCategoricalModal();
    attachCategoricalModalLifecycleHandlers();
    populateCategoricalModal(analysisType, columnCandidates, previousSelection);

    const modalElement = document.getElementById('categoricalAnalysisModal');
    if (!modalElement) {
        categoricalModalIsRerun = false;
        showNotification('Categorical configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    if (previousSelection && previousSelection.length > 0) {
        showNotification('Review or adjust your previously selected columns, then rerun.', 'info');
    } else {
        showNotification(`Select categorical columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
    }
}

async function launchCategoricalAnalysis() {
    if (!currentCategoricalAnalysisType) {
        showNotification('No categorical analysis selected.', 'error');
        return;
    }

    const selectedList = Array.from(categoricalModalSelection);
    if (!selectedList.length) {
        showNotification('Select at least one categorical column to continue.', 'warning');
        return;
    }

    const confirmBtn = document.getElementById('categoricalModalConfirmBtn');
    let previousLabel = null;
    if (confirmBtn) {
        previousLabel = confirmBtn.innerHTML;
        confirmBtn.disabled = true;
        confirmBtn.setAttribute('aria-busy', 'true');
        confirmBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Running…
        `;
    }

    categoricalModalConfirmed = true;
    const modalElement = document.getElementById('categoricalAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    if (modalInstance) {
        modalInstance.hide();
    }

    try {
        if (!currentCategoricalCellId) {
            const fallbackCellId = await addSingleAnalysisCell(currentCategoricalAnalysisType, {
                skipCategoricalModal: true
            });
            currentCategoricalCellId = fallbackCellId || '';
        }

        if (!currentCategoricalCellId) {
            showNotification('Unable to start categorical analysis: no analysis cell available.', 'error');
            return;
        }

        showNotification(`Running ${getAnalysisTypeName(currentCategoricalAnalysisType)} for ${selectedList.length} column${selectedList.length === 1 ? '' : 's'}.`, 'success');

        await generateAndRunAnalysis(
            currentCategoricalCellId,
            currentCategoricalAnalysisType,
            {},
            {
                overrideSelectedColumns: selectedList,
                includeGlobalSelectedColumns: false,
                modalType: 'categorical'
            }
        );
    } catch (error) {
        console.error('Categorical analysis run failed:', error);
        showNotification(`Categorical analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.removeAttribute('aria-busy');
            if (previousLabel !== null) {
                confirmBtn.innerHTML = previousLabel;
            }
        }
        categoricalModalSelection = new Set();
        categoricalModalRecommendedDefaults = [];
        categoricalModalSearchTerm = '';
        currentCategoricalCellId = '';
        currentCategoricalAnalysisType = '';
        categoricalModalConfirmed = false;
        categoricalModalIsRerun = false;
    }
}

// ============================================
// NUMERIC FREQUENCY MODAL FUNCTIONALITY
// ============================================



async function prepareNumericFrequencyConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    currentNumericAnalysisType = analysisType;
    numericModalConfirmed = false;
    numericModalSelection = new Set();
    numericModalColumns = [];
    numericModalRecommendedDefaults = [];
    numericModalSearchTerm = '';
    currentNumericCellId = '';

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before numeric modal:', error);
        }
    }

    const columnCandidates = getNumericColumnCandidates();
    if (!columnCandidates.length) {
        showNotification('No numeric columns detected. Running full analysis instead.', 'info');
        await addSingleAnalysisCell(analysisType);
        return;
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipNumericModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare numeric analysis cell. Please try again.', 'error');
        return;
    }

    currentNumericCellId = cellId;
    numericModalColumns = columnCandidates;

    initializeNumericModal();
    attachNumericModalLifecycleHandlers();
    populateNumericModal(analysisType, columnCandidates);

    const modalElement = document.getElementById('numericAnalysisModal');
    if (!modalElement) {
        showNotification('Numeric configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
    showNotification(`Select numeric columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
}

function attachNumericModalLifecycleHandlers() {
    const modalElement = document.getElementById('numericAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleNumericModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleNumericModalHidden() {
    if (numericModalConfirmed) {
        numericModalConfirmed = false;
        numericModalIsRerun = false;
        return;
    }

    if (numericModalIsRerun) {
        numericModalIsRerun = false;
        currentNumericCellId = '';
        currentNumericAnalysisType = '';
        numericModalSelection = new Set();
        numericModalRecommendedDefaults = [];
        numericModalColumns = [];
        numericModalSearchTerm = '';
        showNotification('Numeric analysis rerun cancelled.', 'info');
        return;
    }

    if (currentNumericCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentNumericCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        if (currentNumericAnalysisType) {
            showNotification('Numeric frequency analysis cancelled.', 'info');
        }
    }

    currentNumericCellId = '';
    currentNumericAnalysisType = '';
    numericModalSelection = new Set();
    numericModalRecommendedDefaults = [];
    numericModalColumns = [];
    numericModalSearchTerm = '';
}

function initializeNumericModal() {
    const modalElement = document.getElementById('numericAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const confirmBtn = document.getElementById('numericModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchNumericAnalysis);
    }

    const selectAllBtn = document.getElementById('numericModalSelectAllBtn');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', selectAllNumericColumns);
    }

    const clearBtn = document.getElementById('numericModalClearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            numericModalSelection = new Set();
            renderNumericColumnList();
            updateNumericSelectionSummary();
            updateNumericChipStates();
        });
    }

    const recommendedBtn = document.getElementById('numericModalRecommendBtn');
    if (recommendedBtn) {
        recommendedBtn.addEventListener('click', () => {
            applyNumericRecommendations();
        });
    }

    const searchInput = document.getElementById('numericColumnSearch');
    if (searchInput) {
        searchInput.addEventListener('input', event => {
            numericModalSearchTerm = (event.target.value || '').toLowerCase();
            renderNumericColumnList();
            updateNumericChipStates();
        });
    }

    const columnList = document.getElementById('numericColumnList');
    if (columnList) {
        columnList.addEventListener('change', handleNumericListChange);
    }

    const chipsContainer = document.getElementById('numericRecommendationChips');
    if (chipsContainer) {
        chipsContainer.addEventListener('click', handleNumericChipClick);
    }

    modalElement.dataset.initialized = 'true';
}

function getNumericColumnCandidates() {
    if (!Array.isArray(columnInsightsData?.column_insights)) {
        return [];
    }

    const columns = [];
    const numericCategories = new Set(['numeric', 'number', 'continuous', 'integer', 'decimal', 'ratio']);
    const numericTypeTokens = ['int', 'float', 'double', 'decimal', 'number'];

    columnInsightsData.column_insights.forEach(col => {
        if (!col || col.dropped) {
            return;
        }

        const normalizedName = (col.name ?? '').toString().trim();
        if (!normalizedName) {
            return;
        }

        const dataCategory = (col.data_category || '').toString().toLowerCase();
        const dataType = (col.data_type || '').toString().toLowerCase();

        const isNumericCategory = numericCategories.has(dataCategory);
        const isNumericType = numericTypeTokens.some(token => dataType.includes(token));

        if (!isNumericCategory && !isNumericType) {
            return;
        }

        const stats = typeof col.statistics === 'object' && col.statistics !== null ? col.statistics : {};
        const uniqueCount = toFiniteNumber(stats.unique_count ?? stats.distinct_count ?? stats.cardinality);
        const missingPct = toFiniteNumber(col.null_percentage);
        const minValue = toFiniteNumber(stats.min ?? stats.minimum);
        const maxValue = toFiniteNumber(stats.max ?? stats.maximum);
        const stdDev = toFiniteNumber(stats.std_dev ?? stats.std ?? stats.stddev);

        const recommended = Boolean(
            (typeof uniqueCount === 'number' && uniqueCount > 0 && uniqueCount <= 40) ||
                (typeof uniqueCount === 'number' && uniqueCount <= 120 && dataType.includes('int')) ||
                (typeof stdDev === 'number' && stdDev > 0 && stdDev <= (toFiniteNumber(stats.mean) ?? stdDev * 2))
        );

        const reasonParts = [];
        if (typeof uniqueCount === 'number') {
            reasonParts.push(`${uniqueCount} ${uniqueCount === 1 ? 'unique value' : 'unique values'}`);
        }
        if (typeof minValue === 'number' && typeof maxValue === 'number') {
            reasonParts.push(`range ${minValue} – ${maxValue}`);
        }
        if (typeof missingPct === 'number') {
            reasonParts.push(`${missingPct.toFixed(1)}% missing`);
        }

        columns.push({
            name: normalizedName,
            dataCategory: dataCategory || dataType || 'numeric',
            dataType,
            uniqueCount: typeof uniqueCount === 'number' ? uniqueCount : null,
            missingPct: typeof missingPct === 'number' ? missingPct : null,
            minValue,
            maxValue,
            recommended,
            reason: reasonParts.join(' • ')
        });
    });

    columns.sort((a, b) => {
        if (a.recommended !== b.recommended) {
            return a.recommended ? -1 : 1;
        }
        const aUnique = typeof a.uniqueCount === 'number' ? a.uniqueCount : Number.POSITIVE_INFINITY;
        const bUnique = typeof b.uniqueCount === 'number' ? b.uniqueCount : Number.POSITIVE_INFINITY;
        if (aUnique !== bUnique) {
            return aUnique - bUnique;
        }
        return a.name.localeCompare(b.name);
    });

    return columns;
}

function populateNumericModal(analysisType, columns, initialSelection = null) {
    const modalLabel = document.getElementById('numericAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.textContent = `Configure ${getAnalysisTypeName(analysisType)}`;
    }

    const subtitle = document.getElementById('numericModalSubtitle');
    if (subtitle) {
        subtitle.textContent = 'Pick numeric columns to review dominant values or adaptive bins. Smaller sets run faster.';
    }

    const confirmBtn = document.getElementById('numericModalConfirmBtn');
    if (confirmBtn) {
        const baseLabel = `Run ${getAnalysisTypeName(analysisType)}`;
        confirmBtn.dataset.baseLabel = baseLabel;
        confirmBtn.textContent = baseLabel;
        confirmBtn.disabled = false;
    }

    const searchInput = document.getElementById('numericColumnSearch');
    if (searchInput) {
        searchInput.value = '';
    }

    numericModalColumns = Array.isArray(columns) ? [...columns] : [];

    const recommended = numericModalColumns.filter(col => col.recommended);
    numericModalRecommendedDefaults = recommended.slice(0, 6).map(col => col.name);

    const normalizedInitialSelection = Array.isArray(initialSelection) || initialSelection instanceof Set
        ? Array.from(new Set(initialSelection instanceof Set ? [...initialSelection] : initialSelection)).filter(name =>
            numericModalColumns.some(col => col.name === name)
        )
        : [];

    if (normalizedInitialSelection.length > 0) {
        numericModalSelection = new Set(normalizedInitialSelection);
    } else if (numericModalRecommendedDefaults.length > 0) {
        numericModalSelection = new Set(numericModalRecommendedDefaults);
    } else {
        const fallbackDefaults = numericModalColumns.slice(0, Math.min(5, numericModalColumns.length)).map(col => col.name);
        numericModalSelection = new Set(fallbackDefaults);
    }

    renderNumericRecommendations(recommended);
    renderNumericColumnList();
    updateNumericChipStates();
    updateNumericSelectionSummary();
}

function renderNumericRecommendations(recommendedColumns) {
    const container = document.getElementById('numericRecommendationChips');
    if (!container) {
        return;
    }

    container.innerHTML = '';

    if (!recommendedColumns.length) {
        container.innerHTML = '<span class="text-muted small">No automatic recommendations yet. Select columns manually below.</span>';
        return;
    }

    recommendedColumns.forEach(col => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-outline-primary btn-sm me-2 mb-2';
        button.dataset.column = col.name;
        if (col.reason) {
            button.title = col.reason;
        }
        button.textContent = col.name;
        container.appendChild(button);
    });
}

function renderNumericColumnList() {
    const listElement = document.getElementById('numericColumnList');
    if (!listElement) {
        return;
    }

    let filtered = [...numericModalColumns];
    if (numericModalSearchTerm) {
        filtered = filtered.filter(col => col.name.toLowerCase().includes(numericModalSearchTerm));
    }

    if (!filtered.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No columns match your search.</div>';
        return;
    }

    const rows = filtered
        .map(col => {
            const checked = numericModalSelection.has(col.name) ? 'checked' : '';
            const detailParts = [];
            if (typeof col.uniqueCount === 'number') {
                detailParts.push(`${col.uniqueCount} unique`);
            }
            if (typeof col.minValue === 'number' && typeof col.maxValue === 'number') {
                detailParts.push(`range ${col.minValue} – ${col.maxValue}`);
            }
            if (typeof col.missingPct === 'number') {
                detailParts.push(`${col.missingPct.toFixed(1)}% missing`);
            }
            const detailText = detailParts.length ? `<small class="text-muted">${detailParts.join(' • ')}</small>` : '';
            const badgeLabel = col.dataCategory || col.dataType || 'numeric';

            return `
                <label class="list-group-item d-flex align-items-start gap-3">
                    <input class="form-check-input mt-1" type="checkbox" value="${escapeHtml(col.name)}" data-column="${escapeHtml(col.name)}" ${checked}>
                    <div class="flex-grow-1">
                        <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                            <strong>${escapeHtml(col.name)}</strong>
                            <span class="badge text-bg-light text-capitalize">${escapeHtml(badgeLabel)}</span>
                        </div>
                        ${detailText}
                    </div>
                </label>
            `;
        })
        .join('');

    listElement.innerHTML = rows;
}

function updateNumericSelectionSummary() {
    const summaryElement = document.getElementById('numericSelectionSummary');
    const confirmBtn = document.getElementById('numericModalConfirmBtn');
    const count = numericModalSelection.size;

    if (summaryElement) {
        if (count === 0) {
            summaryElement.textContent = 'Select one or more numeric columns to continue.';
        } else {
            const preview = Array.from(numericModalSelection).slice(0, 5);
            const remainder = count - preview.length;
            const suffix = remainder > 0 ? `, +${remainder} more` : '';
            summaryElement.textContent = `${count} column${count === 1 ? '' : 's'} selected: ${preview.join(', ')}${suffix}`;
        }
    }

    if (confirmBtn) {
        const baseLabel = confirmBtn.dataset.baseLabel || confirmBtn.textContent || 'Run analysis';
        confirmBtn.textContent = count > 0 ? `${baseLabel} (${count})` : baseLabel;
        confirmBtn.disabled = count === 0;
    }
}

function updateNumericChipStates() {
    const chipsContainer = document.getElementById('numericRecommendationChips');
    if (!chipsContainer) {
        return;
    }
    chipsContainer.querySelectorAll('[data-column]').forEach(button => {
        const columnName = button.dataset.column;
        button.classList.toggle('active', numericModalSelection.has(columnName));
    });
}

function handleNumericListChange(event) {
    const checkbox = event.target;
    if (!checkbox || checkbox.tagName !== 'INPUT' || checkbox.type !== 'checkbox') {
        return;
    }

    const columnName = checkbox.dataset.column;
    if (!columnName) {
        return;
    }

    if (checkbox.checked) {
        numericModalSelection.add(columnName);
    } else {
        numericModalSelection.delete(columnName);
    }

    updateNumericSelectionSummary();
    updateNumericChipStates();
}

function handleNumericChipClick(event) {
    const button = event.target.closest('[data-column]');
    if (!button) {
        return;
    }
    event.preventDefault();

    const columnName = button.dataset.column;
    if (!columnName) {
        return;
    }

    if (numericModalSelection.has(columnName)) {
        numericModalSelection.delete(columnName);
    } else {
        numericModalSelection.add(columnName);
    }

    renderNumericColumnList();
    updateNumericSelectionSummary();
    updateNumericChipStates();
}

function applyNumericRecommendations() {
    if (!numericModalRecommendedDefaults.length) {
        showNotification('No recommended numeric columns available yet.', 'info');
        return;
    }

    numericModalSelection = new Set(numericModalRecommendedDefaults);
    renderNumericColumnList();
    updateNumericSelectionSummary();
    updateNumericChipStates();
}

function selectAllNumericColumns() {
    if (!Array.isArray(numericModalColumns) || numericModalColumns.length === 0) {
        showNotification('No numeric columns available yet.', 'info');
        return;
    }

    numericModalSelection = new Set(numericModalColumns.map(col => col.name));
    renderNumericColumnList();
    updateNumericSelectionSummary();
    updateNumericChipStates();
}

async function openNumericModalForRerun(cellId, analysisType, previousSelection = []) {
    numericModalIsRerun = true;
    currentNumericCellId = cellId;
    currentNumericAnalysisType = analysisType;
    numericModalConfirmed = false;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before numeric rerun:', error);
        }
    }

    const columnCandidates = getNumericColumnCandidates();
    if (!columnCandidates.length) {
        numericModalIsRerun = false;
        showNotification('Numeric columns are unavailable for rerun.', 'warning');
        return;
    }

    numericModalColumns = columnCandidates;

    initializeNumericModal();
    attachNumericModalLifecycleHandlers();
    populateNumericModal(analysisType, columnCandidates, previousSelection);

    const modalElement = document.getElementById('numericAnalysisModal');
    if (!modalElement) {
        numericModalIsRerun = false;
        showNotification('Numeric configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    if (previousSelection && previousSelection.length > 0) {
        showNotification('Review or adjust your previously selected columns, then rerun.', 'info');
    } else {
        showNotification(`Select numeric columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
    }
}

async function launchNumericAnalysis() {
    if (!currentNumericAnalysisType) {
        showNotification('No numeric analysis selected.', 'error');
        return;
    }

    const selectedList = Array.from(numericModalSelection);
    if (!selectedList.length) {
        showNotification('Select at least one numeric column to continue.', 'warning');
        return;
    }

    const confirmBtn = document.getElementById('numericModalConfirmBtn');
    let previousLabel = null;
    if (confirmBtn) {
        previousLabel = confirmBtn.innerHTML;
        confirmBtn.disabled = true;
        confirmBtn.setAttribute('aria-busy', 'true');
        confirmBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Running…
        `;
    }

    numericModalConfirmed = true;
    const modalElement = document.getElementById('numericAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    if (modalInstance) {
        modalInstance.hide();
    }

    try {
        if (!currentNumericCellId) {
            const fallbackCellId = await addSingleAnalysisCell(currentNumericAnalysisType, {
                skipNumericModal: true
            });
            currentNumericCellId = fallbackCellId || '';
        }

        if (!currentNumericCellId) {
            showNotification('Unable to start numeric analysis: no analysis cell available.', 'error');
            return;
        }

        showNotification(`Running ${getAnalysisTypeName(currentNumericAnalysisType)} for ${selectedList.length} column${selectedList.length === 1 ? '' : 's'}.`, 'success');

        await generateAndRunAnalysis(
            currentNumericCellId,
            currentNumericAnalysisType,
            {},
            {
                overrideSelectedColumns: selectedList,
                includeGlobalSelectedColumns: false,
                modalType: 'numeric'
            }
        );
    } catch (error) {
        console.error('Numeric frequency analysis run failed:', error);
        showNotification(`Numeric analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.removeAttribute('aria-busy');
            if (previousLabel !== null) {
                confirmBtn.innerHTML = previousLabel;
            }
        }
        numericModalSelection = new Set();
        numericModalRecommendedDefaults = [];
        numericModalSearchTerm = '';
        currentNumericCellId = '';
        currentNumericAnalysisType = '';
        numericModalConfirmed = false;
        numericModalIsRerun = false;
    }
}

// ============================================
// CROSS TABULATION MODAL FUNCTIONALITY
// ============================================



function prepareCategoricalNumericAnalysisConfiguration(analysisType) {
    return (async () => {
        if (!analysisType) {
            return;
        }

        currentCategoricalNumericAnalysisType = analysisType;
        categoricalNumericModalConfirmed = false;
        categoricalNumericModalCategoricalColumns = [];
        categoricalNumericModalNumericColumns = [];
        categoricalNumericRecommendedPairs = [];
        categoricalNumericActivePairs = new Set();
        categoricalNumericSelectedCategorical = new Set();
        categoricalNumericSelectedNumeric = new Set();
        categoricalNumericModalSearchTerm = { categorical: '', numeric: '' };
        categoricalNumericModalIsRerun = false;
        currentCategoricalNumericCellId = '';

        if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
            try {
                await loadColumnInsights();
            } catch (error) {
                console.warn('Unable to refresh column insights before categorical vs numeric modal:', error);
            }
        }

        const categoricalCandidates = getCategoricalColumnCandidates();
        const numericCandidates = getNumericColumnCandidates();

        if (!categoricalCandidates.length || !numericCandidates.length) {
            showNotification('Need at least one categorical and one numeric column for this explorer.', 'warning');
            await addSingleAnalysisCell(analysisType);
            return;
        }

        const cellId = await addSingleAnalysisCell(analysisType, {
            skipCategoricalNumericModal: true
        });

        if (!cellId) {
            showNotification('Unable to prepare analysis cell. Please try again.', 'error');
            return;
        }

        currentCategoricalNumericCellId = cellId;
        categoricalNumericModalCategoricalColumns = [...categoricalCandidates];
        categoricalNumericModalNumericColumns = [...numericCandidates];
        categoricalNumericRecommendedPairs = computeCategoricalNumericRecommendations(categoricalCandidates, numericCandidates);

        initializeCategoricalNumericModal();
        attachCategoricalNumericModalLifecycleHandlers();
        populateCategoricalNumericModal(analysisType, categoricalCandidates, numericCandidates);

        const modalElement = document.getElementById('categoricalNumericAnalysisModal');
        if (!modalElement) {
            showNotification('Categorical vs Numeric configuration modal is unavailable.', 'error');
            return;
        }

        const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
        modalInstance.show();
        showNotification(`Select categorical and numeric columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
    })();
}

function attachCategoricalNumericModalLifecycleHandlers() {
    const modalElement = document.getElementById('categoricalNumericAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleCategoricalNumericModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleCategoricalNumericModalHidden() {
    if (categoricalNumericModalConfirmed) {
        categoricalNumericModalConfirmed = false;
        categoricalNumericModalIsRerun = false;
        return;
    }

    if (categoricalNumericModalIsRerun) {
        categoricalNumericModalIsRerun = false;
        resetCategoricalNumericModalState();
        showNotification('Categorical vs numeric analysis rerun cancelled.', 'info');
        return;
    }

    if (currentCategoricalNumericCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentCategoricalNumericCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        if (currentCategoricalNumericAnalysisType) {
            showNotification('Categorical vs numeric analysis cancelled.', 'info');
        }
    }

    resetCategoricalNumericModalState();
}

function initializeCategoricalNumericModal() {
    const modalElement = document.getElementById('categoricalNumericAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const confirmBtn = document.getElementById('categoricalNumericModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchCategoricalNumericAnalysis);
        confirmBtn.dataset.baseLabel = confirmBtn.textContent || 'Run analysis';
    }

    const selectAllCategoricalBtn = document.getElementById('categoricalNumericCategoricalSelectAllBtn');
    if (selectAllCategoricalBtn) {
        selectAllCategoricalBtn.addEventListener('click', () => {
            categoricalNumericSelectedCategorical = new Set(categoricalNumericModalCategoricalColumns.map(col => col.name));
            renderCategoricalNumericColumnLists();
            updateCategoricalNumericSelectionSummary();
            updateCategoricalNumericChipStates();
        });
    }

    const clearCategoricalBtn = document.getElementById('categoricalNumericCategoricalClearBtn');
    if (clearCategoricalBtn) {
        clearCategoricalBtn.addEventListener('click', () => {
            categoricalNumericSelectedCategorical = new Set();
            renderCategoricalNumericColumnLists();
            updateCategoricalNumericSelectionSummary();
            updateCategoricalNumericChipStates();
        });
    }

    const selectAllNumericBtn = document.getElementById('categoricalNumericNumericSelectAllBtn');
    if (selectAllNumericBtn) {
        selectAllNumericBtn.addEventListener('click', () => {
            categoricalNumericSelectedNumeric = new Set(categoricalNumericModalNumericColumns.map(col => col.name));
            renderCategoricalNumericColumnLists();
            updateCategoricalNumericSelectionSummary();
            updateCategoricalNumericChipStates();
        });
    }

    const clearNumericBtn = document.getElementById('categoricalNumericNumericClearBtn');
    if (clearNumericBtn) {
        clearNumericBtn.addEventListener('click', () => {
            categoricalNumericSelectedNumeric = new Set();
            renderCategoricalNumericColumnLists();
            updateCategoricalNumericSelectionSummary();
            updateCategoricalNumericChipStates();
        });
    }

    const recommendedBtn = document.getElementById('categoricalNumericModalRecommendBtn');
    if (recommendedBtn) {
        recommendedBtn.addEventListener('click', applyCategoricalNumericRecommendations);
    }

    const clearAllBtn = document.getElementById('categoricalNumericModalClearAllBtn');
    if (clearAllBtn) {
        clearAllBtn.addEventListener('click', () => {
            categoricalNumericSelectedCategorical = new Set();
            categoricalNumericSelectedNumeric = new Set();
            categoricalNumericActivePairs = new Set();
            renderCategoricalNumericColumnLists();
            updateCategoricalNumericSelectionSummary();
            updateCategoricalNumericChipStates();
        });
    }

    const categoricalSearchInput = document.getElementById('categoricalNumericCategoricalSearch');
    if (categoricalSearchInput) {
        categoricalSearchInput.addEventListener('input', event => {
            categoricalNumericModalSearchTerm.categorical = (event.target.value || '').toLowerCase();
            renderCategoricalNumericColumnLists();
            updateCategoricalNumericChipStates();
        });
    }

    const numericSearchInput = document.getElementById('categoricalNumericNumericSearch');
    if (numericSearchInput) {
        numericSearchInput.addEventListener('input', event => {
            categoricalNumericModalSearchTerm.numeric = (event.target.value || '').toLowerCase();
            renderCategoricalNumericColumnLists();
            updateCategoricalNumericChipStates();
        });
    }

    const categoricalList = document.getElementById('categoricalNumericCategoricalList');
    if (categoricalList) {
        categoricalList.addEventListener('change', event => handleCategoricalNumericListChange(event, 'categorical'));
    }

    const numericList = document.getElementById('categoricalNumericNumericList');
    if (numericList) {
        numericList.addEventListener('change', event => handleCategoricalNumericListChange(event, 'numeric'));
    }

    const chipsContainer = document.getElementById('categoricalNumericRecommendationChips');
    if (chipsContainer) {
        chipsContainer.addEventListener('click', handleCategoricalNumericChipClick);
    }

    modalElement.dataset.initialized = 'true';
}

function populateCategoricalNumericModal(analysisType, categoricalColumns, numericColumns, previousSelection = null, previousDetails = null) {
    const modalLabel = document.getElementById('categoricalNumericModalLabel');
    if (modalLabel) {
        modalLabel.textContent = `Configure ${getAnalysisTypeName(analysisType)}`;
    }

    const subtitle = document.getElementById('categoricalNumericModalSubtitle');
    if (subtitle) {
        subtitle.textContent = 'Pick one or more categorical columns and numeric measures to compare distributions across groups.';
    }

    const summaryCard = document.getElementById('categoricalNumericSelectionSummary');
    if (summaryCard) {
        summaryCard.classList.remove('alert-success');
        summaryCard.classList.add('alert-secondary');
        summaryCard.textContent = 'Select categorical and numeric columns to continue.';
    }

    const confirmBtn = document.getElementById('categoricalNumericModalConfirmBtn');
    if (confirmBtn) {
        const baseLabel = confirmBtn.dataset.baseLabel || `Run ${getAnalysisTypeName(analysisType)}`;
        confirmBtn.textContent = baseLabel;
        confirmBtn.disabled = true;
    }

    const derived = deriveCategoricalNumericSelection(previousDetails, previousSelection);
    const { categorical: priorCategorical, numeric: priorNumeric } = derived;

    categoricalNumericModalCategoricalColumns = Array.isArray(categoricalColumns) ? [...categoricalColumns] : [];
    categoricalNumericModalNumericColumns = Array.isArray(numericColumns) ? [...numericColumns] : [];

    categoricalNumericSelectedCategorical = new Set(priorCategorical.length ? priorCategorical : autoSelectCategoricalDefaults());
    categoricalNumericSelectedNumeric = new Set(priorNumeric.length ? priorNumeric : autoSelectNumericDefaults());

    renderCategoricalNumericRecommendations();
    renderCategoricalNumericColumnLists();
    updateCategoricalNumericSelectionSummary();
    updateCategoricalNumericChipStates();
}

function autoSelectCategoricalDefaults() {
    if (!Array.isArray(categoricalNumericModalCategoricalColumns)) {
        return [];
    }

    const recommended = categoricalNumericModalCategoricalColumns.filter(col => col.recommended).slice(0, 3);
    if (recommended.length) {
        return recommended.map(col => col.name);
    }

    return categoricalNumericModalCategoricalColumns.slice(0, Math.min(3, categoricalNumericModalCategoricalColumns.length)).map(col => col.name);
}

function autoSelectNumericDefaults() {
    if (!Array.isArray(categoricalNumericModalNumericColumns)) {
        return [];
    }

    const recommended = categoricalNumericModalNumericColumns.filter(col => col.recommended).slice(0, 3);
    if (recommended.length) {
        return recommended.map(col => col.name);
    }

    return categoricalNumericModalNumericColumns.slice(0, Math.min(3, categoricalNumericModalNumericColumns.length)).map(col => col.name);
}

function renderCategoricalNumericColumnLists() {
    const categoricalList = document.getElementById('categoricalNumericCategoricalList');
    const numericList = document.getElementById('categoricalNumericNumericList');

    if (categoricalList) {
        let filtered = [...categoricalNumericModalCategoricalColumns];
        if (categoricalNumericModalSearchTerm.categorical) {
            filtered = filtered.filter(col => col.name.toLowerCase().includes(categoricalNumericModalSearchTerm.categorical));
        }

        if (!filtered.length) {
            categoricalList.innerHTML = '<div class="text-muted small px-2 py-3">No categorical columns match your search.</div>';
        } else {
            const rows = filtered.map(col => renderCategoricalNumericListItem(col, 'categorical', categoricalNumericSelectedCategorical.has(col.name))).join('');
            categoricalList.innerHTML = rows;
        }
    }

    if (numericList) {
        let filtered = [...categoricalNumericModalNumericColumns];
        if (categoricalNumericModalSearchTerm.numeric) {
            filtered = filtered.filter(col => col.name.toLowerCase().includes(categoricalNumericModalSearchTerm.numeric));
        }

        if (!filtered.length) {
            numericList.innerHTML = '<div class="text-muted small px-2 py-3">No numeric columns match your search.</div>';
        } else {
            const rows = filtered.map(col => renderCategoricalNumericListItem(col, 'numeric', categoricalNumericSelectedNumeric.has(col.name))).join('');
            numericList.innerHTML = rows;
        }
    }
}

function renderCategoricalNumericListItem(column, columnType, checked) {
    const safeName = typeof escapeHtml === 'function' ? escapeHtml(column.name) : column.name;
    const badgeLabel = column.dataCategory || column.dataType || (columnType === 'categorical' ? 'categorical' : 'numeric');
    const details = [];

    if (typeof column.uniqueCount === 'number') {
        details.push(`${column.uniqueCount} ${column.uniqueCount === 1 ? 'unique value' : 'unique values'}`);
    }
    if (typeof column.missingPct === 'number') {
        details.push(`${column.missingPct.toFixed(1)}% missing`);
    }
    if (column.reason) {
        details.push(column.reason);
    }

    const detailText = details.length ? `<small class="text-muted">${details.join(' • ')}</small>` : '';

    return `
        <label class="list-group-item d-flex align-items-start gap-3">
            <input class="form-check-input mt-1" type="checkbox" value="${safeName}" data-column-type="${columnType}" data-column-name="${safeName}" ${checked ? 'checked' : ''}>
            <div class="flex-grow-1">
                <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                    <strong>${safeName}</strong>
                    <span class="badge text-bg-light text-capitalize">${typeof escapeHtml === 'function' ? escapeHtml(badgeLabel) : badgeLabel}</span>
                </div>
                ${detailText}
            </div>
        </label>
    `;
}

function renderCategoricalNumericRecommendations() {
    const container = document.getElementById('categoricalNumericRecommendationChips');
    if (!container) {
        return;
    }

    if (!Array.isArray(categoricalNumericRecommendedPairs) || !categoricalNumericRecommendedPairs.length) {
        container.innerHTML = '<span class="text-muted small">No smart suggestions yet. Select columns manually.</span>';
        return;
    }

    const chips = categoricalNumericRecommendedPairs.map(pair => {
        const safePair = typeof escapeHtml === 'function'
            ? `${escapeHtml(pair.categorical)} ↔ ${escapeHtml(pair.numeric)}`
            : `${pair.categorical} ↔ ${pair.numeric}`;
        const title = pair.reason ? ` title="${typeof escapeHtml === 'function' ? escapeHtml(pair.reason) : pair.reason}"` : '';
        return `
            <button type="button" class="btn btn-sm btn-outline-primary recommendation-chip" data-pair-id="${pair.id}"${title}>
                <i class="bi bi-diagram-3"></i>
                <span>${safePair}</span>
            </button>
        `;
    }).join('');

    container.innerHTML = chips;
}

function updateCategoricalNumericSelectionSummary() {
    const summaryElement = document.getElementById('categoricalNumericSelectionSummary');
    const confirmBtn = document.getElementById('categoricalNumericModalConfirmBtn');

    const categoricalCount = categoricalNumericSelectedCategorical.size;
    const numericCount = categoricalNumericSelectedNumeric.size;

    if (summaryElement) {
        if (categoricalCount === 0 || numericCount === 0) {
            summaryElement.classList.remove('alert-success');
            summaryElement.classList.add('alert-secondary');
            summaryElement.textContent = 'Select at least one categorical and one numeric column to continue.';
        } else {
            summaryElement.classList.remove('alert-secondary');
            summaryElement.classList.add('alert-success');
            const categoricalPreview = Array.from(categoricalNumericSelectedCategorical).slice(0, 4).join(', ');
            const numericPreview = Array.from(categoricalNumericSelectedNumeric).slice(0, 4).join(', ');
            summaryElement.innerHTML = `
                <strong>${categoricalCount}</strong> categorical column${categoricalCount === 1 ? '' : 's'} selected (${categoricalPreview}${categoricalCount > 4 ? ', …' : ''})<br>
                <strong>${numericCount}</strong> numeric column${numericCount === 1 ? '' : 's'} selected (${numericPreview}${numericCount > 4 ? ', …' : ''})
            `;
        }
    }

    if (confirmBtn) {
        const baseLabel = confirmBtn.dataset.baseLabel || confirmBtn.textContent || 'Run analysis';
        confirmBtn.textContent = categoricalCount && numericCount ? `${baseLabel} (${categoricalCount} × ${numericCount})` : baseLabel;
        confirmBtn.disabled = categoricalCount === 0 || numericCount === 0;
    }
}

function updateCategoricalNumericChipStates() {
    const container = document.getElementById('categoricalNumericRecommendationChips');
    if (!container || !Array.isArray(categoricalNumericRecommendedPairs)) {
        return;
    }

    const selectedCategorical = categoricalNumericSelectedCategorical;
    const selectedNumeric = categoricalNumericSelectedNumeric;

    container.querySelectorAll('[data-pair-id]').forEach(button => {
        const pair = categoricalNumericRecommendedPairs.find(item => item.id === button.dataset.pairId);
        if (!pair) {
            return;
        }
        const isActive = categoricalNumericActivePairs.has(pair.id) || (selectedCategorical.has(pair.categorical) && selectedNumeric.has(pair.numeric));
        button.classList.toggle('active', isActive);
    });
}

function handleCategoricalNumericListChange(event, columnType) {
    const checkbox = event.target;
    if (!checkbox || checkbox.tagName !== 'INPUT' || checkbox.type !== 'checkbox') {
        return;
    }

    const columnName = checkbox.dataset.columnName;
    if (!columnName) {
        return;
    }

    const selectionSet = columnType === 'categorical'
        ? categoricalNumericSelectedCategorical
        : categoricalNumericSelectedNumeric;

    if (checkbox.checked) {
        selectionSet.add(columnName);
    } else {
        selectionSet.delete(columnName);
        if (columnType === 'categorical') {
            categoricalNumericActivePairs = new Set([...categoricalNumericActivePairs].filter(id => {
                const pair = categoricalNumericRecommendedPairs.find(item => item.id === id);
                return pair && pair.categorical !== columnName;
            }));
        } else {
            categoricalNumericActivePairs = new Set([...categoricalNumericActivePairs].filter(id => {
                const pair = categoricalNumericRecommendedPairs.find(item => item.id === id);
                return pair && pair.numeric !== columnName;
            }));
        }
    }

    updateCategoricalNumericSelectionSummary();
    updateCategoricalNumericChipStates();
}

function handleCategoricalNumericChipClick(event) {
    const button = event.target.closest('[data-pair-id]');
    if (!button) {
        return;
    }
    event.preventDefault();

    const pair = categoricalNumericRecommendedPairs.find(item => item.id === button.dataset.pairId);
    if (!pair) {
        return;
    }

    if (categoricalNumericActivePairs.has(pair.id)) {
        categoricalNumericActivePairs.delete(pair.id);
    } else {
        categoricalNumericActivePairs.add(pair.id);
    }

    categoricalNumericSelectedCategorical.add(pair.categorical);
    categoricalNumericSelectedNumeric.add(pair.numeric);

    renderCategoricalNumericColumnLists();
    updateCategoricalNumericSelectionSummary();
    updateCategoricalNumericChipStates();
}

function applyCategoricalNumericRecommendations() {
    if (!Array.isArray(categoricalNumericRecommendedPairs) || !categoricalNumericRecommendedPairs.length) {
        showNotification('No recommended pairs available yet. Select columns manually.', 'info');
        return;
    }

    categoricalNumericSelectedCategorical = new Set(categoricalNumericRecommendedPairs.map(pair => pair.categorical));
    categoricalNumericSelectedNumeric = new Set(categoricalNumericRecommendedPairs.map(pair => pair.numeric));
    categoricalNumericActivePairs = new Set(categoricalNumericRecommendedPairs.map(pair => pair.id));

    renderCategoricalNumericColumnLists();
    updateCategoricalNumericSelectionSummary();
    updateCategoricalNumericChipStates();
}

async function launchCategoricalNumericAnalysis() {
    if (!currentCategoricalNumericAnalysisType) {
        showNotification('No analysis selected for execution.', 'error');
        return;
    }

    const categoricalSelection = Array.from(categoricalNumericSelectedCategorical);
    const numericSelection = Array.from(categoricalNumericSelectedNumeric);

    if (!categoricalSelection.length || !numericSelection.length) {
        showNotification('Select at least one categorical and one numeric column.', 'warning');
        return;
    }

    const confirmBtn = document.getElementById('categoricalNumericModalConfirmBtn');
    let previousLabel = null;
    if (confirmBtn) {
        previousLabel = confirmBtn.innerHTML;
        confirmBtn.disabled = true;
        confirmBtn.setAttribute('aria-busy', 'true');
        confirmBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Preparing…
        `;
    }

    categoricalNumericModalConfirmed = true;
    const modalElement = document.getElementById('categoricalNumericAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    if (modalInstance) {
        modalInstance.hide();
    }

    try {
        if (!currentCategoricalNumericCellId) {
            const fallbackCellId = await addSingleAnalysisCell(currentCategoricalNumericAnalysisType, {
                skipCategoricalNumericModal: true
            });
            currentCategoricalNumericCellId = fallbackCellId || '';
        }

        if (!currentCategoricalNumericCellId) {
            showNotification('Unable to start analysis: no available cell.', 'error');
            return;
        }

        const orderedSelection = [...categoricalSelection, ...numericSelection];
        const activePairs = Array.from(categoricalNumericActivePairs).map(id => {
            const pair = categoricalNumericRecommendedPairs.find(item => item.id === id);
            if (!pair) {
                return null;
            }
            return {
                categorical: pair.categorical,
                numeric: pair.numeric,
                reason: pair.reason || null
            };
        }).filter(Boolean);

        const selectionPayload = {
            categorical: categoricalSelection,
            numeric: numericSelection,
            pairs: activePairs
        };

        const analysisMetadata = buildCategoricalNumericAnalysisMetadata(selectionPayload);

        showNotification(`Running ${getAnalysisTypeName(currentCategoricalNumericAnalysisType)} for ${categoricalSelection.length} categorical × ${numericSelection.length} numeric columns.`, 'success');

        await generateAndRunAnalysis(
            currentCategoricalNumericCellId,
            currentCategoricalNumericAnalysisType,
            {},
            {
                overrideSelectedColumns: orderedSelection,
                includeGlobalSelectedColumns: false,
                modalType: 'categorical-numeric',
                modalSelectionPayload: selectionPayload,
                analysisMetadata
            }
        );
    } catch (error) {
        console.error('Categorical vs numeric analysis run failed:', error);
        showNotification(`Analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.removeAttribute('aria-busy');
            if (previousLabel !== null) {
                confirmBtn.innerHTML = previousLabel;
            }
        }
        resetCategoricalNumericModalState();
    }
}

function resetCategoricalNumericModalState() {
    categoricalNumericModalConfirmed = false;
    categoricalNumericModalIsRerun = false;
    currentCategoricalNumericCellId = '';
    currentCategoricalNumericAnalysisType = '';
    categoricalNumericSelectedCategorical = new Set();
    categoricalNumericSelectedNumeric = new Set();
    categoricalNumericModalCategoricalColumns = [];
    categoricalNumericModalNumericColumns = [];
    categoricalNumericModalSearchTerm = { categorical: '', numeric: '' };
    categoricalNumericRecommendedPairs = [];
    categoricalNumericActivePairs = new Set();
}

async function openCategoricalNumericModalForRerun(cellId, analysisType, previousSelection = [], previousDetails = null) {
    categoricalNumericModalIsRerun = true;
    currentCategoricalNumericCellId = cellId;
    currentCategoricalNumericAnalysisType = analysisType;
    categoricalNumericModalConfirmed = false;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before rerun:', error);
        }
    }

    const categoricalCandidates = getCategoricalColumnCandidates();
    const numericCandidates = getNumericColumnCandidates();

    if (!categoricalCandidates.length || !numericCandidates.length) {
        categoricalNumericModalIsRerun = false;
        showNotification('Required columns are unavailable for rerun.', 'warning');
        return;
    }

    categoricalNumericModalCategoricalColumns = [...categoricalCandidates];
    categoricalNumericModalNumericColumns = [...numericCandidates];
    categoricalNumericRecommendedPairs = computeCategoricalNumericRecommendations(categoricalCandidates, numericCandidates);

    initializeCategoricalNumericModal();
    attachCategoricalNumericModalLifecycleHandlers();
    populateCategoricalNumericModal(analysisType, categoricalCandidates, numericCandidates, previousSelection, previousDetails);

    const modalElement = document.getElementById('categoricalNumericAnalysisModal');
    if (!modalElement) {
        categoricalNumericModalIsRerun = false;
        showNotification('Categorical vs numeric configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    if (previousSelection && previousSelection.length > 0) {
        showNotification('Review or adjust your previously selected categorical and numeric columns.', 'info');
    } else {
        showNotification(`Select categorical and numeric columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
    }
}

function computeCategoricalNumericRecommendations(categoricalColumns, numericColumns) {
    if (!Array.isArray(categoricalColumns) || !Array.isArray(numericColumns)) {
        return [];
    }

    const prioritizedCategorical = [...categoricalColumns].sort((a, b) => {
        if (a.recommended !== b.recommended) {
            return a.recommended ? -1 : 1;
        }
        const aUnique = typeof a.uniqueCount === 'number' ? a.uniqueCount : Number.POSITIVE_INFINITY;
        const bUnique = typeof b.uniqueCount === 'number' ? b.uniqueCount : Number.POSITIVE_INFINITY;
        return aUnique - bUnique;
    }).slice(0, 5);

    const prioritizedNumeric = [...numericColumns].sort((a, b) => {
        if (a.recommended !== b.recommended) {
            return a.recommended ? -1 : 1;
        }
        const aUnique = typeof a.uniqueCount === 'number' ? a.uniqueCount : Number.POSITIVE_INFINITY;
        const bUnique = typeof b.uniqueCount === 'number' ? b.uniqueCount : Number.POSITIVE_INFINITY;
        return aUnique - bUnique;
    }).slice(0, 5);

    const pairs = [];

    prioritizedCategorical.forEach((cat, catIndex) => {
        prioritizedNumeric.forEach((num, numIndex) => {
            const pairId = `${cat.name}:::${num.name}`;
            const categoryCount = typeof cat.uniqueCount === 'number' ? cat.uniqueCount : null;
            const numericUnique = typeof num.uniqueCount === 'number' ? num.uniqueCount : null;
            const score = (categoryCount || 10) * (numericUnique || 15) + catIndex + numIndex / 10;
            const reasonParts = [];
            if (categoryCount !== null) {
                reasonParts.push(`${categoryCount} categories`);
            }
            if (num.reason) {
                reasonParts.push(num.reason);
            }

            pairs.push({
                id: pairId,
                categorical: cat.name,
                numeric: num.name,
                score,
                reason: reasonParts.length ? reasonParts.join(' • ') : ''
            });
        });
    });

    pairs.sort((a, b) => a.score - b.score);
    return pairs.slice(0, 8);
}

function deriveCategoricalNumericSelection(details, fallbackSelection = []) {
    const categorical = new Set();
    const numeric = new Set();

    if (details && typeof details === 'object') {
        const detailCategorical = details.categorical || details.categorical_columns;
        const detailNumeric = details.numeric || details.numeric_columns;

        if (Array.isArray(detailCategorical)) {
            detailCategorical.forEach(name => {
                if (name) {
                    categorical.add(String(name));
                }
            });
        }

        if (Array.isArray(detailNumeric)) {
            detailNumeric.forEach(name => {
                if (name) {
                    numeric.add(String(name));
                }
            });
        }
    }

    if ((!categorical.size || !numeric.size) && Array.isArray(fallbackSelection)) {
        const insightMap = buildColumnInsightLookup();
        fallbackSelection.forEach(name => {
            const info = insightMap[name];
            if (!info) {
                return;
            }
            if (info.kind === 'categorical') {
                categorical.add(name);
            } else if (info.kind === 'numeric') {
                numeric.add(name);
            }
        });
    }

    return {
        categorical: Array.from(categorical),
        numeric: Array.from(numeric)
    };
}

function buildCategoricalNumericAnalysisMetadata(selectionPayload) {
    const metadata = {
        categorical_numeric: {
            categorical_columns: Array.isArray(selectionPayload?.categorical) ? [...selectionPayload.categorical] : [],
            numeric_columns: Array.isArray(selectionPayload?.numeric) ? [...selectionPayload.numeric] : [],
            pairs: Array.isArray(selectionPayload?.pairs) ? [...selectionPayload.pairs] : []
        }
    };

    if (Array.isArray(selectionPayload?.categorical) && Array.isArray(selectionPayload?.numeric)) {
        metadata.categorical_numeric.estimated_pair_count = selectionPayload.categorical.length * selectionPayload.numeric.length;
    }

    return metadata;
}

function buildColumnInsightLookup() {
    const lookup = {};
    if (!Array.isArray(columnInsightsData?.column_insights)) {
        return lookup;
    }

    columnInsightsData.column_insights.forEach(col => {
        if (!col || col.dropped) {
            return;
        }
        const name = (col.name || '').toString();
        if (!name) {
        }
        const category = (col.data_category || '').toLowerCase();
        const dtype = (col.data_type || '').toLowerCase();
        const categoricalTokens = ['text', 'categorical', 'category', 'bool', 'boolean'];
        const numericTokens = ['int', 'float', 'double', 'decimal', 'number'];

        let kind = 'unknown';
        if (categoricalTokens.includes(category) || categoricalTokens.some(token => dtype.includes(token))) {
            kind = 'categorical';
        } else if (numericTokens.includes(category) || numericTokens.some(token => dtype.includes(token))) {
            kind = 'numeric';
        }

        lookup[name] = { kind };
    });

    return lookup;
}


async function prepareCrossTabAnalysisConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    currentCrossTabAnalysisType = analysisType;
    crossTabModalConfirmed = false;
    crossTabModalSelection = [];
    crossTabModalColumns = [];
    crossTabModalRecommendedDefaults = [];
    crossTabModalSearchTerm = '';
    currentCrossTabCellId = '';

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before cross tab modal:', error);
        }
    }

    const columnCandidates = getCategoricalColumnCandidates();
    if (!columnCandidates.length || columnCandidates.length < 2) {
        showNotification('Too few categorical-style columns detected. Running cross tabulation with current defaults.', 'info');
        await addSingleAnalysisCell(analysisType);
        return;
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipCrossTabModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare cross tabulation cell. Please try again.', 'error');
        return;
    }

    currentCrossTabCellId = cellId;
    crossTabModalColumns = columnCandidates;

    initializeCrossTabModal();
    attachCrossTabModalLifecycleHandlers();
    populateCrossTabModal(analysisType, columnCandidates);

    const modalElement = document.getElementById('crossTabAnalysisModal');
    if (!modalElement) {
        showNotification('Cross tab configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
    showNotification(`Pick two categorical columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
}

function attachCrossTabModalLifecycleHandlers() {
    const modalElement = document.getElementById('crossTabAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleCrossTabModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleCrossTabModalHidden() {
    if (crossTabModalConfirmed) {
        crossTabModalConfirmed = false;
        crossTabModalIsRerun = false;
        return;
    }

    if (crossTabModalIsRerun) {
        crossTabModalIsRerun = false;
        resetCrossTabModalState();
        showNotification('Cross tabulation rerun cancelled.', 'info');
        return;
    }

    if (currentCrossTabCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentCrossTabCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        if (currentCrossTabAnalysisType) {
            showNotification('Cross tabulation analysis cancelled.', 'info');
        }
    }

    resetCrossTabModalState();
}

function initializeCrossTabModal() {
    const modalElement = document.getElementById('crossTabAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const confirmBtn = document.getElementById('crossTabModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchCrossTabAnalysis);
    }

    const clearBtn = document.getElementById('crossTabModalClearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            crossTabModalSelection = [];
            renderCrossTabColumnList();
            updateCrossTabSelectionSummary();
            updateCrossTabChipStates();
        });
    }

    const recommendBtn = document.getElementById('crossTabModalRecommendBtn');
    if (recommendBtn) {
        recommendBtn.addEventListener('click', () => {
            applyCrossTabRecommendations();
        });
    }

    const swapBtn = document.getElementById('crossTabModalSwapBtn');
    if (swapBtn) {
        swapBtn.addEventListener('click', () => {
            if (crossTabModalSelection.length === 2) {
                crossTabModalSelection = [crossTabModalSelection[1], crossTabModalSelection[0]];
                renderCrossTabColumnList();
                updateCrossTabSelectionSummary();
                updateCrossTabChipStates();
            }
        });
    }

    const searchInput = document.getElementById('crossTabColumnSearch');
    if (searchInput) {
        searchInput.addEventListener('input', event => {
            crossTabModalSearchTerm = (event.target.value || '').toLowerCase();
            renderCrossTabColumnList();
            updateCrossTabChipStates();
        });
    }

    const columnList = document.getElementById('crossTabColumnList');
    if (columnList) {
        columnList.addEventListener('change', handleCrossTabListChange);
    }

    const chipsContainer = document.getElementById('crossTabRecommendationChips');
    if (chipsContainer) {
        chipsContainer.addEventListener('click', handleCrossTabChipClick);
    }

    modalElement.dataset.initialized = 'true';
}

function populateCrossTabModal(analysisType, columns, initialSelection = null) {
    const modalLabel = document.getElementById('crossTabAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.textContent = `Configure ${getAnalysisTypeName(analysisType)}`;
    }

    const subtitle = document.getElementById('crossTabModalSubtitle');
    if (subtitle) {
        subtitle.textContent = 'Select exactly two categorical columns. The first becomes the row axis, the second becomes the column axis.';
    }

    const confirmBtn = document.getElementById('crossTabModalConfirmBtn');
    if (confirmBtn) {
        const baseLabel = `Run ${getAnalysisTypeName(analysisType)}`;
        confirmBtn.dataset.baseLabel = baseLabel;
        confirmBtn.textContent = baseLabel;
        confirmBtn.disabled = true;
    }

    const searchInput = document.getElementById('crossTabColumnSearch');
    if (searchInput) {
        searchInput.value = '';
    }

    crossTabModalColumns = Array.isArray(columns) ? [...columns] : [];

    const recommended = crossTabModalColumns.filter(col => col.recommended);
    crossTabModalRecommendedDefaults = recommended.slice(0, 4).map(col => col.name);

    const normalizedInitialSelection = Array.isArray(initialSelection)
        ? Array.from(new Set(initialSelection.filter(name => crossTabModalColumns.some(col => col.name === name)))).slice(0, 2)
        : [];

    if (normalizedInitialSelection.length === 2) {
        crossTabModalSelection = [...normalizedInitialSelection];
    } else if (crossTabModalRecommendedDefaults.length >= 2) {
        crossTabModalSelection = crossTabModalRecommendedDefaults.slice(0, 2);
    } else {
        crossTabModalSelection = crossTabModalColumns.slice(0, 2).map(col => col.name);
    }

    renderCrossTabRecommendations(recommended);
    renderCrossTabColumnList();
    updateCrossTabChipStates();
    updateCrossTabSelectionSummary();
}

function renderCrossTabRecommendations(recommendedColumns) {
    const container = document.getElementById('crossTabRecommendationChips');
    if (!container) {
        return;
    }

    container.innerHTML = '';

    if (!recommendedColumns.length) {
        container.innerHTML = '<span class="text-muted small">No automatic recommendations yet. Select columns manually below.</span>';
        return;
    }

    recommendedColumns.forEach(col => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-outline-primary btn-sm me-2 mb-2';
        button.dataset.column = col.name;
        if (col.reason) {
            button.title = col.reason;
        }
        button.textContent = col.name;
        container.appendChild(button);
    });
}

function renderCrossTabColumnList() {
    const listElement = document.getElementById('crossTabColumnList');
    if (!listElement) {
        return;
    }

    let filtered = [...crossTabModalColumns];
    if (crossTabModalSearchTerm) {
        filtered = filtered.filter(col => col.name.toLowerCase().includes(crossTabModalSearchTerm));
    }

    if (!filtered.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No columns match your search.</div>';
        return;
    }

    const selectedSet = new Set(crossTabModalSelection);
    const reachedLimit = crossTabModalSelection.length >= 2;

    const rows = filtered
        .map(col => {
            const checked = selectedSet.has(col.name) ? 'checked' : '';
            const disabled = !selectedSet.has(col.name) && reachedLimit ? 'disabled' : '';
            const positionLabel = (() => {
                if (crossTabModalSelection[0] === col.name) {
                    return '<span class="badge text-bg-info ms-2">Rows</span>';
                }
                if (crossTabModalSelection[1] === col.name) {
                    return '<span class="badge text-bg-primary ms-2">Columns</span>';
                }
                return '';
            })();
            const detailParts = [];
            if (typeof col.uniqueCount === 'number') {
                detailParts.push(`${col.uniqueCount} ${col.uniqueCount === 1 ? 'category' : 'categories'}`);
            }
            if (typeof col.missingPct === 'number') {
                detailParts.push(`${col.missingPct.toFixed(1)}% missing`);
            }
            const detailText = detailParts.length ? `<small class="text-muted">${detailParts.join(' • ')}</small>` : '';
            const badgeLabel = col.dataCategory || col.dataType || 'categorical';

            return `
                <label class="list-group-item d-flex align-items-start gap-3">
                    <input class="form-check-input mt-1" type="checkbox" value="${escapeHtml(col.name)}" data-column="${escapeHtml(col.name)}" ${checked} ${disabled}>
                    <div class="flex-grow-1">
                        <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                            <strong>${escapeHtml(col.name)}</strong>
                            <span>
                                <span class="badge text-bg-light text-capitalize">${escapeHtml(badgeLabel)}</span>
                                ${positionLabel}
                            </span>
                        </div>
                        ${detailText}
                    </div>
                </label>
            `;
        })
        .join('');

    listElement.innerHTML = rows;
}

function updateCrossTabSelectionSummary() {
    const summaryElement = document.getElementById('crossTabSelectionSummary');
    const confirmBtn = document.getElementById('crossTabModalConfirmBtn');
    const count = crossTabModalSelection.length;

    if (summaryElement) {
        if (count === 0) {
            summaryElement.textContent = 'Select two categorical columns to generate a cross tabulation.';
        } else if (count === 1) {
            summaryElement.textContent = `${crossTabModalSelection[0]} selected as rows. Pick one more column to use as columns.`;
        } else if (count === 2) {
            summaryElement.textContent = `${crossTabModalSelection[0]} → rows • ${crossTabModalSelection[1]} → columns. Use swap to flip axes.`;
        } else {
            const preview = crossTabModalSelection.slice(0, 2).join(' → ');
            summaryElement.textContent = `${preview}. Only the first two selections will be used.`;
        }
    }

    if (confirmBtn) {
        const baseLabel = confirmBtn.dataset.baseLabel || confirmBtn.textContent || 'Run analysis';
        confirmBtn.textContent = count === 2 ? `${baseLabel} (${crossTabModalSelection[0]} vs ${crossTabModalSelection[1]})` : baseLabel;
        confirmBtn.disabled = count !== 2;
    }

    const swapBtn = document.getElementById('crossTabModalSwapBtn');
    if (swapBtn) {
        swapBtn.disabled = count !== 2;
    }
}

function updateCrossTabChipStates() {
    const chipsContainer = document.getElementById('crossTabRecommendationChips');
    if (!chipsContainer) {
        return;
    }
    const selectedSet = new Set(crossTabModalSelection);
    chipsContainer.querySelectorAll('[data-column]').forEach(button => {
        const columnName = button.dataset.column;
        button.classList.toggle('active', selectedSet.has(columnName));
    });
}

function handleCrossTabListChange(event) {
    const checkbox = event.target;
    if (!checkbox || checkbox.tagName !== 'INPUT' || checkbox.type !== 'checkbox') {
        return;
    }

    const columnName = checkbox.dataset.column;
    if (!columnName) {
        return;
    }

    if (checkbox.checked) {
        if (!crossTabModalSelection.includes(columnName)) {
            if (crossTabModalSelection.length >= 2) {
                crossTabModalSelection.shift();
            }
            crossTabModalSelection.push(columnName);
        }
    } else {
        crossTabModalSelection = crossTabModalSelection.filter(name => name !== columnName);
    }

    renderCrossTabColumnList();
    updateCrossTabSelectionSummary();
    updateCrossTabChipStates();
}

function handleCrossTabChipClick(event) {
    const button = event.target.closest('[data-column]');
    if (!button) {
        return;
    }
    event.preventDefault();

    const columnName = button.dataset.column;
    if (!columnName) {
        return;
    }

    if (crossTabModalSelection.includes(columnName)) {
        crossTabModalSelection = crossTabModalSelection.filter(name => name !== columnName);
    } else {
        if (crossTabModalSelection.length >= 2) {
            crossTabModalSelection.shift();
        }
        crossTabModalSelection.push(columnName);
    }

    renderCrossTabColumnList();
    updateCrossTabSelectionSummary();
    updateCrossTabChipStates();
}

function applyCrossTabRecommendations() {
    if (crossTabModalRecommendedDefaults.length < 2) {
        if (crossTabModalColumns.length >= 2) {
            crossTabModalSelection = crossTabModalColumns.slice(0, 2).map(col => col.name);
        } else {
            showNotification('Need at least two categorical columns to apply recommendations.', 'warning');
            return;
        }
    } else {
        crossTabModalSelection = crossTabModalRecommendedDefaults.slice(0, 2);
    }

    renderCrossTabColumnList();
    updateCrossTabSelectionSummary();
    updateCrossTabChipStates();
}

async function openCrossTabModalForRerun(cellId, analysisType, previousSelection = []) {
    crossTabModalIsRerun = true;
    currentCrossTabCellId = cellId;
    currentCrossTabAnalysisType = analysisType;
    crossTabModalConfirmed = false;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before cross tab rerun:', error);
        }
    }

    const columnCandidates = getCategoricalColumnCandidates();
    if (!columnCandidates.length || columnCandidates.length < 2) {
        crossTabModalIsRerun = false;
        showNotification('Categorical columns are unavailable for rerun.', 'warning');
        return;
    }

    crossTabModalColumns = columnCandidates;

    initializeCrossTabModal();
    attachCrossTabModalLifecycleHandlers();
    populateCrossTabModal(analysisType, columnCandidates, previousSelection);

    const modalElement = document.getElementById('crossTabAnalysisModal');
    if (!modalElement) {
        crossTabModalIsRerun = false;
        showNotification('Cross tab configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    if (previousSelection && previousSelection.length > 0) {
        showNotification('Review or adjust your previously selected columns, then rerun.', 'info');
    } else {
        showNotification(`Select categorical columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
    }
}

async function launchCrossTabAnalysis() {
    if (!currentCrossTabAnalysisType) {
        showNotification('No cross tabulation analysis selected.', 'error');
        return;
    }

    const selectedList = Array.from(new Set(crossTabModalSelection)).slice(0, 2);
    if (selectedList.length !== 2) {
        showNotification('Select exactly two categorical columns to continue.', 'warning');
        return;
    }

    const confirmBtn = document.getElementById('crossTabModalConfirmBtn');
    let previousLabel = null;
    if (confirmBtn) {
        previousLabel = confirmBtn.innerHTML;
        confirmBtn.disabled = true;
        confirmBtn.setAttribute('aria-busy', 'true');
        confirmBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Running…
        `;
    }

    crossTabModalConfirmed = true;
    const modalElement = document.getElementById('crossTabAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    if (modalInstance) {
        modalInstance.hide();
    }

    try {
        if (!currentCrossTabCellId) {
            const fallbackCellId = await addSingleAnalysisCell(currentCrossTabAnalysisType, {
                skipCrossTabModal: true
            });
            currentCrossTabCellId = fallbackCellId || '';
        }

        if (!currentCrossTabCellId) {
            showNotification('Unable to start cross tabulation: no analysis cell available.', 'error');
            return;
        }

        showNotification(`Running ${getAnalysisTypeName(currentCrossTabAnalysisType)} for ${selectedList[0]} vs ${selectedList[1]}.`, 'success');

        await generateAndRunAnalysis(
            currentCrossTabCellId,
            currentCrossTabAnalysisType,
            {},
            {
                overrideSelectedColumns: selectedList,
                includeGlobalSelectedColumns: false,
                modalType: 'crosstab'
            }
        );
    } catch (error) {
        console.error('Cross tabulation analysis run failed:', error);
        showNotification(`Cross tabulation failed to start: ${error.message || error}`, 'error');
    } finally {
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.removeAttribute('aria-busy');
            if (previousLabel !== null) {
                confirmBtn.innerHTML = previousLabel;
            }
        }
        resetCrossTabModalState();
    }
}

function resetCrossTabModalState() {
    crossTabModalSelection = [];
    crossTabModalRecommendedDefaults = [];
    crossTabModalSearchTerm = '';
    crossTabModalConfirmed = false;
    currentCrossTabCellId = '';
    currentCrossTabAnalysisType = '';
    crossTabModalIsRerun = false;
}

// ============================================
// MARKETING ANALYTICS MODAL FUNCTIONALITY
// ============================================

// Marketing analysis configurations


async function prepareTimeSeriesAnalysisConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    configureTimeSeriesModalFlags(analysisType);

    currentTimeSeriesAnalysisType = analysisType;
    currentTimeSeriesCellId = '';
    timeSeriesModalConfirmed = false;
    timeSeriesModalIsRerun = false;
    timeSeriesModalSelectedDates = new Set();
    timeSeriesModalNumericSelection = new Set();
    timeSeriesModalDateColumns = [];
    timeSeriesModalNumericColumns = [];
    timeSeriesModalDateRecommendedDefaults = [];
    timeSeriesModalNumericRecommendedDefaults = [];
    timeSeriesModalDateSearchTerm = '';
    timeSeriesModalNumericSearchTerm = '';

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before time series modal:', error);
        }
    }

    const datetimeCandidates = getDatetimeColumnCandidates();
    if (!datetimeCandidates.length) {
        showNotification('No datetime-style columns detected. Running analysis with default behaviour instead.', 'info');
        resetTimeSeriesModalState();
        await addSingleAnalysisCell(analysisType);
        return;
    }

    let numericCandidates = [];
    if (timeSeriesModalRequiresNumeric) {
        numericCandidates = getNumericColumnCandidates();
        if (!numericCandidates.length) {
            showNotification('No numeric columns detected. Running analysis with default behaviour instead.', 'info');
            resetTimeSeriesModalState();
            await addSingleAnalysisCell(analysisType);
            return;
        }
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipTimeSeriesModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare time series analysis cell. Please try again.', 'error');
        resetTimeSeriesModalState();
        return;
    }

    currentTimeSeriesCellId = cellId;
    timeSeriesModalDateColumns = datetimeCandidates;
    timeSeriesModalNumericColumns = numericCandidates;

    initializeTimeSeriesModal();
    attachTimeSeriesModalLifecycleHandlers();
    populateTimeSeriesModal(analysisType, datetimeCandidates, numericCandidates);

    const modalElement = document.getElementById('timeSeriesAnalysisModal');
    if (!modalElement) {
        showNotification('Time series configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    const analysisName = getAnalysisTypeName(analysisType);
    if (timeSeriesModalRequiresNumeric) {
        showNotification(`Select a datetime column and numeric measures for ${analysisName}.`, 'info');
    } else {
        showNotification(`Select datetime columns for ${analysisName}.`, 'info');
    }
}

function configureTimeSeriesModalFlags(analysisType) {
    timeSeriesModalRequiresNumeric = analysisType === 'temporal_trend_analysis' || analysisType === 'seasonality_detection';
    timeSeriesModalAllowsMultipleDates = analysisType === 'datetime_feature_extraction';
}

const DATETIME_NAME_HINTS = ['date', 'datetime', 'timestamp', 'time', 'created_at', 'updated_at', 'event_time', 'recorded_at', 'observed_at'];
const DATETIME_SUFFIX_HINTS = ['_at', '_dt'];
const DATETIME_EPOCH_HINTS = ['epoch', 'unix', 'posix'];
const NUMERIC_TYPE_TOKENS = ['int', 'float', 'double', 'decimal', 'number'];

function attachTimeSeriesModalLifecycleHandlers() {
    const modalElement = document.getElementById('timeSeriesAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleTimeSeriesModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleTimeSeriesModalHidden() {
    if (timeSeriesModalConfirmed) {
        timeSeriesModalConfirmed = false;
        timeSeriesModalIsRerun = false;
        return;
    }

    if (timeSeriesModalIsRerun) {
        timeSeriesModalIsRerun = false;
        resetTimeSeriesModalState();
        showNotification('Time series analysis rerun cancelled.', 'info');
        return;
    }

    if (currentTimeSeriesCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentTimeSeriesCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        if (currentTimeSeriesAnalysisType) {
            showNotification('Time series analysis cancelled.', 'info');
        }
    }

    resetTimeSeriesModalState();
}

function initializeTimeSeriesModal() {
    const modalElement = document.getElementById('timeSeriesAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const confirmBtn = document.getElementById('timeSeriesModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchTimeSeriesAnalysis);
    }

    const datetimeAutoBtn = document.getElementById('timeSeriesDatetimeAutoBtn');
    if (datetimeAutoBtn) {
        datetimeAutoBtn.addEventListener('click', autoSelectBestTimeSeriesDatetime);
    }

    const datetimeClearBtn = document.getElementById('timeSeriesDatetimeClearBtn');
    if (datetimeClearBtn) {
        datetimeClearBtn.addEventListener('click', clearTimeSeriesDatetimeSelection);
    }

    const numericSelectAllBtn = document.getElementById('timeSeriesNumericSelectAllBtn');
    if (numericSelectAllBtn) {
        numericSelectAllBtn.addEventListener('click', selectAllTimeSeriesNumericColumns);
    }

    const numericClearBtn = document.getElementById('timeSeriesNumericClearBtn');
    if (numericClearBtn) {
        numericClearBtn.addEventListener('click', clearTimeSeriesNumericSelection);
    }

    const numericRecommendBtn = document.getElementById('timeSeriesNumericRecommendBtn');
    if (numericRecommendBtn) {
        numericRecommendBtn.addEventListener('click', applyTimeSeriesNumericRecommendations);
    }

    const datetimeSearchInput = document.getElementById('timeSeriesDatetimeSearch');
    if (datetimeSearchInput) {
        datetimeSearchInput.addEventListener('input', handleTimeSeriesDatetimeSearch);
    }

    const numericSearchInput = document.getElementById('timeSeriesNumericSearch');
    if (numericSearchInput) {
        numericSearchInput.addEventListener('input', handleTimeSeriesNumericSearch);
    }

    const datetimeList = document.getElementById('timeSeriesDatetimeList');
    if (datetimeList) {
        datetimeList.addEventListener('change', handleTimeSeriesDatetimeListChange);
    }

    const numericList = document.getElementById('timeSeriesNumericList');
    if (numericList) {
        numericList.addEventListener('change', handleTimeSeriesNumericListChange);
    }

    modalElement.dataset.initialized = 'true';
}

function getDatetimeColumnCandidates() {
    if (!Array.isArray(columnInsightsData?.column_insights)) {
        return buildFallbackDatetimeCandidates(new Set());
    }

    const columns = [];
    const seenColumns = new Set();

    columnInsightsData.column_insights.forEach(col => {
        if (!col || col.dropped) {
            return;
        }

        const normalizedName = (col.name ?? '').toString().trim();
        if (!normalizedName) {
            return;
        }

        const normalizedLower = normalizedName.toLowerCase();

        const dataCategory = (col.data_category || '').toString().toLowerCase();
        const dataType = (col.data_type || '').toString().toLowerCase();
        const stats = typeof col.statistics === 'object' && col.statistics !== null ? col.statistics : {};
        const missingPct = toFiniteNumber(col.null_percentage);
        const uniqueCount = toFiniteNumber(stats.unique_count ?? stats.distinct_count ?? stats.cardinality);
        const sampleFormats = Array.isArray(stats.sample_formats) ? stats.sample_formats : [];
        const datetimeHints = Boolean(stats.datetime_detected || stats.date_detected || stats.time_detected);

        const categoryTokens = ['datetime', 'date', 'time'];
        const typeTokens = ['date', 'time', 'timestamp'];
        const isDatetimeCategory = categoryTokens.some(token => dataCategory.includes(token));
        const isDatetimeType = typeTokens.some(token => dataType.includes(token));
        const isNumericType = NUMERIC_TYPE_TOKENS.some(token => dataType.includes(token));
        const nameSuggestsEpoch = DATETIME_EPOCH_HINTS.some(token => normalizedLower.includes(token));
        const nameSuggestsDatetime = DATETIME_NAME_HINTS.some(token => normalizedLower.includes(token)) || DATETIME_SUFFIX_HINTS.some(suffix => normalizedLower.endsWith(suffix));

        if (!isDatetimeCategory && !isDatetimeType && !datetimeHints && !nameSuggestsDatetime && !nameSuggestsEpoch) {
            return;
        }

        if (isNumericType && !isDatetimeCategory && !isDatetimeType && !datetimeHints && !nameSuggestsEpoch) {
            return;
        }

        const reasonParts = [];
        if (typeof uniqueCount === 'number') {
            reasonParts.push(`${uniqueCount} unique`);
        }
        if (typeof missingPct === 'number') {
            reasonParts.push(`${missingPct.toFixed(1)}% missing`);
        }
        if (sampleFormats.length) {
            reasonParts.push(sampleFormats[0]);
        }
        if (nameSuggestsDatetime || nameSuggestsEpoch) {
            reasonParts.push('Name hint');
        }
        if (isNumericType && nameSuggestsEpoch) {
            reasonParts.push('Epoch-style numeric');
        }

        const recommended = Boolean(
            (typeof missingPct !== 'number' || missingPct <= 35) &&
            (isDatetimeCategory || isDatetimeType || sampleFormats.length > 0 || datetimeHints || nameSuggestsDatetime || nameSuggestsEpoch)
        );

        columns.push({
            name: normalizedName,
            dataCategory: dataCategory || dataType || 'datetime',
            dataType,
            missingPct: typeof missingPct === 'number' ? missingPct : null,
            uniqueCount: typeof uniqueCount === 'number' ? uniqueCount : null,
            recommended,
            reason: reasonParts.join(' • ')
        });
        seenColumns.add(normalizedName);
    });

    columns.sort((a, b) => {
        if (a.recommended !== b.recommended) {
            return a.recommended ? -1 : 1;
        }
        const aMissing = typeof a.missingPct === 'number' ? a.missingPct : Number.POSITIVE_INFINITY;
        const bMissing = typeof b.missingPct === 'number' ? b.missingPct : Number.POSITIVE_INFINITY;
        if (aMissing !== bMissing) {
            return aMissing - bMissing;
        }
        return a.name.localeCompare(b.name);
    });

    if (!columns.length) {
        return buildFallbackDatetimeCandidates(seenColumns);
    }

    return columns;
}

function buildFallbackDatetimeCandidates(seenColumns = new Set()) {
    const names = gatherAvailableColumnNames();
    if (!names.length) {
        return [];
    }

    const candidates = [];

    names.forEach(name => {
        if (!name || seenColumns.has(name)) {
            return;
        }

        const lower = name.toLowerCase();
        const nameSuggestsEpoch = DATETIME_EPOCH_HINTS.some(token => lower.includes(token));
        const nameSuggestsDatetime = DATETIME_NAME_HINTS.some(token => lower.includes(token)) || DATETIME_SUFFIX_HINTS.some(suffix => lower.endsWith(suffix));

        if (!nameSuggestsDatetime && !nameSuggestsEpoch) {
            return;
        }

        candidates.push({
            name,
            dataCategory: 'datetime',
            dataType: '',
            missingPct: null,
            uniqueCount: null,
            recommended: true,
            reason: nameSuggestsEpoch ? 'Column name indicates epoch timestamps' : 'Column name suggests datetime'
        });
    });

    return candidates.sort((a, b) => a.name.localeCompare(b.name));
}

function gatherAvailableColumnNames() {
    const names = new Set();

    if (Array.isArray(columnInsightsData?.column_insights)) {
        columnInsightsData.column_insights.forEach(col => {
            const normalizedName = (col?.name ?? '').toString().trim();
            if (normalizedName) {
                names.add(normalizedName);
            }
        });
    }

    if (typeof selectedColumns !== 'undefined' && selectedColumns instanceof Set) {
        selectedColumns.forEach(name => {
            if (typeof name === 'string' && name.trim()) {
                names.add(name.trim());
            }
        });
    }

    if (typeof window !== 'undefined') {
        if (Array.isArray(window.currentDataFrame?.columns)) {
            window.currentDataFrame.columns.forEach(name => {
                if (typeof name === 'string' && name.trim()) {
                    names.add(name.trim());
                }
            });
        }

        try {
            const storedColumns = sessionStorage.getItem('datasetColumns');
            if (storedColumns) {
                const parsed = JSON.parse(storedColumns);
                if (Array.isArray(parsed)) {
                    parsed.forEach(name => {
                        if (typeof name === 'string' && name.trim()) {
                            names.add(name.trim());
                        }
                    });
                }
            }
        } catch (error) {
            console.warn('Unable to read stored dataset columns for datetime fallback:', error);
        }
    }

    return Array.from(names);
}

function populateTimeSeriesModal(analysisType, datetimeColumns, numericColumns, initialSelection = null) {
    const analysisName = getAnalysisTypeName(analysisType);
    const modalLabel = document.getElementById('timeSeriesAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.textContent = `Configure ${analysisName}`;
    }

    const subtitle = document.getElementById('timeSeriesModalSubtitle');
    if (subtitle) {
        subtitle.textContent = timeSeriesModalRequiresNumeric
            ? 'Choose the datetime column that defines your timeline and the numeric measures to trend alongside it.'
            : 'Pick the datetime columns you want to explore. We will derive features and coverage statistics for each one.';
    }

    const numericSection = document.getElementById('timeSeriesNumericSection');
    if (numericSection) {
        numericSection.classList.toggle('d-none', !timeSeriesModalRequiresNumeric);
    }

    const datetimeSearchInput = document.getElementById('timeSeriesDatetimeSearch');
    if (datetimeSearchInput) {
        datetimeSearchInput.value = '';
    }
    const numericSearchInput = document.getElementById('timeSeriesNumericSearch');
    if (numericSearchInput) {
        numericSearchInput.value = '';
    }

    timeSeriesModalDateColumns = Array.isArray(datetimeColumns) ? [...datetimeColumns] : [];
    timeSeriesModalNumericColumns = Array.isArray(numericColumns) ? [...numericColumns] : [];

    timeSeriesModalDateRecommendedDefaults = timeSeriesModalDateColumns
        .filter(col => col.recommended)
        .map(col => col.name);
    timeSeriesModalNumericRecommendedDefaults = timeSeriesModalNumericColumns
        .filter(col => col.recommended)
        .map(col => col.name);

    const normalizedInitialSelection = Array.isArray(initialSelection) || initialSelection instanceof Set
        ? Array.from(new Set(initialSelection instanceof Set ? [...initialSelection] : initialSelection))
        : [];

    const initialDateSelections = normalizedInitialSelection.filter(name =>
        timeSeriesModalDateColumns.some(col => col.name === name)
    );
    const initialNumericSelections = normalizedInitialSelection.filter(name =>
        timeSeriesModalNumericColumns.some(col => col.name === name)
    );

    timeSeriesModalSelectedDates = new Set();
    timeSeriesModalNumericSelection = new Set();

    if (timeSeriesModalAllowsMultipleDates) {
        if (initialDateSelections.length > 0) {
            initialDateSelections.forEach(name => timeSeriesModalSelectedDates.add(name));
        } else {
            const defaults = timeSeriesModalDateRecommendedDefaults.length > 0
                ? timeSeriesModalDateRecommendedDefaults.slice(0, 2)
                : timeSeriesModalDateColumns.slice(0, 1).map(col => col.name);
            defaults.forEach(name => timeSeriesModalSelectedDates.add(name));
        }
    } else {
        const defaultDate = initialDateSelections[0]
            || timeSeriesModalDateRecommendedDefaults[0]
            || (timeSeriesModalDateColumns[0] ? timeSeriesModalDateColumns[0].name : '');
        if (defaultDate) {
            timeSeriesModalSelectedDates.add(defaultDate);
        }
    }

    if (timeSeriesModalRequiresNumeric) {
        if (initialNumericSelections.length > 0) {
            initialNumericSelections.forEach(name => timeSeriesModalNumericSelection.add(name));
        } else if (timeSeriesModalNumericRecommendedDefaults.length > 0) {
            timeSeriesModalNumericRecommendedDefaults.slice(0, 4).forEach(name => timeSeriesModalNumericSelection.add(name));
        } else {
            timeSeriesModalNumericColumns.slice(0, Math.min(3, timeSeriesModalNumericColumns.length))
                .forEach(col => timeSeriesModalNumericSelection.add(col.name));
        }
    }

    timeSeriesModalDateSearchTerm = '';
    timeSeriesModalNumericSearchTerm = '';

    renderTimeSeriesDatetimeList();
    renderTimeSeriesNumericList();
    updateTimeSeriesSelectionSummary();
}

function renderTimeSeriesDatetimeList() {
    const listElement = document.getElementById('timeSeriesDatetimeList');
    if (!listElement) {
        return;
    }

    if (!timeSeriesModalDateColumns.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No datetime columns detected.</div>';
        return;
    }

    const searchTerm = (timeSeriesModalDateSearchTerm || '').toLowerCase();
    let filtered = [...timeSeriesModalDateColumns];
    if (searchTerm) {
        filtered = filtered.filter(col => col.name.toLowerCase().includes(searchTerm));
    }

    if (!filtered.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No datetime columns match your search.</div>';
        return;
    }

    const inputType = timeSeriesModalAllowsMultipleDates ? 'checkbox' : 'radio';
    const inputName = 'time-series-datetime-input';

    const rows = filtered.map(col => {
        const checked = timeSeriesModalSelectedDates.has(col.name) ? 'checked' : '';
        const detailParts = [];
        if (typeof col.missingPct === 'number') {
            detailParts.push(`${col.missingPct.toFixed(1)}% missing`);
        }
        if (typeof col.uniqueCount === 'number') {
            detailParts.push(`${col.uniqueCount} unique`);
        }
        const detailText = detailParts.length ? `<small class="text-muted">${detailParts.join(' • ')}</small>` : '';
        const badgeLabel = col.dataCategory || col.dataType || 'datetime';
        const recommendedBadge = col.recommended
            ? '<span class="badge bg-warning-subtle text-warning-emphasis ms-2">Suggested</span>'
            : '';

        return `
            <label class="list-group-item d-flex align-items-start gap-3">
                <input class="form-check-input mt-1" type="${inputType}" name="${inputName}" value="${escapeHtml(col.name)}" data-column="${escapeHtml(col.name)}" ${checked}>
                <div class="flex-grow-1">
                    <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                        <strong>${escapeHtml(col.name)}</strong>
                        <div class="d-flex align-items-center gap-2">
                            <span class="badge text-bg-light text-capitalize">${escapeHtml(badgeLabel)}</span>
                            ${recommendedBadge}
                        </div>
                    </div>
                    ${detailText}
                    ${col.reason ? `<div class="small text-muted">${escapeHtml(col.reason)}</div>` : ''}
                </div>
            </label>
        `;
    }).join('');

    listElement.innerHTML = rows;
}

function renderTimeSeriesNumericList() {
    const listElement = document.getElementById('timeSeriesNumericList');
    if (!listElement) {
        return;
    }

    if (!timeSeriesModalRequiresNumeric) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">Numeric columns are not required for this analysis.</div>';
        return;
    }

    if (!timeSeriesModalNumericColumns.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No numeric columns detected.</div>';
        return;
    }

    const searchTerm = (timeSeriesModalNumericSearchTerm || '').toLowerCase();
    let filtered = [...timeSeriesModalNumericColumns];
    if (searchTerm) {
        filtered = filtered.filter(col => col.name.toLowerCase().includes(searchTerm));
    }

    if (!filtered.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No numeric columns match your search.</div>';
        return;
    }

    const rows = filtered.map(col => {
        const checked = timeSeriesModalNumericSelection.has(col.name) ? 'checked' : '';
        const detailParts = [];
        if (typeof col.uniqueCount === 'number') {
            detailParts.push(`${col.uniqueCount} unique`);
        }
        if (typeof col.minValue === 'number' && typeof col.maxValue === 'number') {
            detailParts.push(`range ${col.minValue} – ${col.maxValue}`);
        }
        if (typeof col.missingPct === 'number') {
            detailParts.push(`${col.missingPct.toFixed(1)}% missing`);
        }
        const detailText = detailParts.length ? `<small class="text-muted">${detailParts.join(' • ')}</small>` : '';
        const badgeLabel = col.dataCategory || col.dataType || 'numeric';
        const recommendedBadge = col.recommended
            ? '<span class="badge bg-primary-subtle text-primary-emphasis ms-2">Suggested</span>'
            : '';

        return `
            <label class="list-group-item d-flex align-items-start gap-3">
                <input class="form-check-input mt-1" type="checkbox" value="${escapeHtml(col.name)}" data-column="${escapeHtml(col.name)}" ${checked}>
                <div class="flex-grow-1">
                    <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                        <strong>${escapeHtml(col.name)}</strong>
                        <div class="d-flex align-items-center gap-2">
                            <span class="badge text-bg-light text-capitalize">${escapeHtml(badgeLabel)}</span>
                            ${recommendedBadge}
                        </div>
                    </div>
                    ${detailText}
                    ${col.reason ? `<div class="small text-muted">${escapeHtml(col.reason)}</div>` : ''}
                </div>
            </label>
        `;
    }).join('');

    listElement.innerHTML = rows;
}

function handleTimeSeriesDatetimeListChange(event) {
    const target = event.target;
    if (!target || !target.matches('input[data-column]')) {
        return;
    }

    const columnName = target.dataset.column;
    if (!columnName) {
        return;
    }

    if (timeSeriesModalAllowsMultipleDates) {
        if (target.checked) {
            timeSeriesModalSelectedDates.add(columnName);
        } else {
            timeSeriesModalSelectedDates.delete(columnName);
        }
    } else {
        timeSeriesModalSelectedDates = new Set([columnName]);
    }

    renderTimeSeriesDatetimeList();
    updateTimeSeriesSelectionSummary();
}

function handleTimeSeriesNumericListChange(event) {
    const target = event.target;
    if (!target || !target.matches('input[data-column]')) {
        return;
    }

    const columnName = target.dataset.column;
    if (!columnName) {
        return;
    }

    if (target.checked) {
        timeSeriesModalNumericSelection.add(columnName);
    } else {
        timeSeriesModalNumericSelection.delete(columnName);
    }

    renderTimeSeriesNumericList();
    updateTimeSeriesSelectionSummary();
}

function handleTimeSeriesDatetimeSearch(event) {
    timeSeriesModalDateSearchTerm = (event.target.value || '').toLowerCase();
    renderTimeSeriesDatetimeList();
}

function handleTimeSeriesNumericSearch(event) {
    timeSeriesModalNumericSearchTerm = (event.target.value || '').toLowerCase();
    renderTimeSeriesNumericList();
}

function clearTimeSeriesDatetimeSelection() {
    timeSeriesModalSelectedDates = new Set();
    renderTimeSeriesDatetimeList();
    updateTimeSeriesSelectionSummary();
}

function autoSelectBestTimeSeriesDatetime() {
    if (!timeSeriesModalDateColumns.length) {
        showNotification('No datetime columns available to auto-select.', 'warning');
        return;
    }

    const ranked = rankTimeSeriesDatetimeCandidates(timeSeriesModalDateColumns);
    if (!ranked.length) {
        showNotification('No suitable datetime columns found for auto selection.', 'warning');
        return;
    }

    const targetCount = timeSeriesModalAllowsMultipleDates
        ? Math.min(3, ranked.length)
        : 1;
    const selection = ranked.slice(0, targetCount).map(col => col.name);

    timeSeriesModalSelectedDates = new Set(selection);
    renderTimeSeriesDatetimeList();
    updateTimeSeriesSelectionSummary();

    if (selection.length > 0) {
        const preview = selection.join(', ');
        showNotification(`Auto-selected ${preview} for the time axis.`, 'success');
    }
}

function rankTimeSeriesDatetimeCandidates(columns) {
    const recommendedSet = new Set(timeSeriesModalDateRecommendedDefaults);

    return [...columns].sort((a, b) => {
        const aRecommended = recommendedSet.has(a.name);
        const bRecommended = recommendedSet.has(b.name);
        if (aRecommended !== bRecommended) {
            return aRecommended ? -1 : 1;
        }

        const aMissing = typeof a.missingPct === 'number' ? a.missingPct : Number.POSITIVE_INFINITY;
        const bMissing = typeof b.missingPct === 'number' ? b.missingPct : Number.POSITIVE_INFINITY;
        if (aMissing !== bMissing) {
            return aMissing - bMissing;
        }

        const aUnique = typeof a.uniqueCount === 'number' ? a.uniqueCount : -1;
        const bUnique = typeof b.uniqueCount === 'number' ? b.uniqueCount : -1;
        if (aUnique !== bUnique) {
            return bUnique - aUnique;
        }

        return a.name.localeCompare(b.name);
    });
}

function selectAllTimeSeriesNumericColumns() {
    if (!timeSeriesModalRequiresNumeric) {
        return;
    }

    timeSeriesModalNumericSelection = new Set(timeSeriesModalNumericColumns.map(col => col.name));
    renderTimeSeriesNumericList();
    updateTimeSeriesSelectionSummary();
}

function clearTimeSeriesNumericSelection() {
    timeSeriesModalNumericSelection = new Set();
    renderTimeSeriesNumericList();
    updateTimeSeriesSelectionSummary();
}

function applyTimeSeriesNumericRecommendations() {
    if (!timeSeriesModalRequiresNumeric) {
        return;
    }

    if (timeSeriesModalNumericRecommendedDefaults.length > 0) {
        timeSeriesModalNumericSelection = new Set(timeSeriesModalNumericRecommendedDefaults.slice(0, 4));
    } else {
        timeSeriesModalNumericSelection = new Set(
            timeSeriesModalNumericColumns.slice(0, Math.min(3, timeSeriesModalNumericColumns.length)).map(col => col.name)
        );
    }

    renderTimeSeriesNumericList();
    updateTimeSeriesSelectionSummary();
}

function updateTimeSeriesSelectionSummary() {
    const summaryElement = document.getElementById('timeSeriesSelectionSummary');
    const confirmBtn = document.getElementById('timeSeriesModalConfirmBtn');
    const dateCount = timeSeriesModalSelectedDates.size;
    const numericCount = timeSeriesModalNumericSelection.size;

    let message = '';
    let canRun = true;

    if (dateCount === 0) {
        message = 'Select at least one datetime column to continue.';
        canRun = false;
    } else if (timeSeriesModalRequiresNumeric && numericCount === 0) {
        message = 'Select one or more numeric columns to pair with the datetime column.';
        canRun = false;
    } else {
        const datePreview = Array.from(timeSeriesModalSelectedDates).slice(0, 3);
        const extraDates = dateCount - datePreview.length;
        if (timeSeriesModalRequiresNumeric) {
            const numericPreview = Array.from(timeSeriesModalNumericSelection).slice(0, 4);
            const extraNumeric = numericCount - numericPreview.length;
            message = `Datetime: ${datePreview.join(', ')}${extraDates > 0 ? ` (+${extraDates} more)` : ''} • Numeric: ${numericPreview.length ? numericPreview.join(', ') : 'none'}${extraNumeric > 0 ? ` (+${extraNumeric} more)` : ''}`;
        } else {
            message = dateCount === 1
                ? `Datetime column selected: ${datePreview[0]}`
                : `Datetime columns selected: ${datePreview.join(', ')}${extraDates > 0 ? ` (+${extraDates} more)` : ''}`;
        }
    }

    if (summaryElement) {
        summaryElement.textContent = message;
        summaryElement.classList.toggle('alert-warning', !canRun);
        summaryElement.classList.toggle('alert-secondary', canRun);
    }

    if (confirmBtn) {
        const baseLabel = confirmBtn.dataset.baseLabel || `Run ${getAnalysisTypeName(currentTimeSeriesAnalysisType || 'time_series_analysis')}`;
        const label = canRun
            ? timeSeriesModalRequiresNumeric
                ? `${baseLabel} (${dateCount} datetime • ${numericCount} numeric)`
                : `${baseLabel} (${dateCount} datetime)`
            : baseLabel;
        confirmBtn.dataset.baseLabel = baseLabel;
        confirmBtn.textContent = label;
        confirmBtn.disabled = !canRun;
    }
}

async function launchTimeSeriesAnalysis() {
    if (!currentTimeSeriesAnalysisType) {
        showNotification('No time series analysis selected.', 'error');
        return;
    }

    const dateSelection = Array.from(timeSeriesModalSelectedDates);
    if (!dateSelection.length) {
        showNotification('Select at least one datetime column to continue.', 'warning');
        return;
    }

    if (timeSeriesModalRequiresNumeric && timeSeriesModalNumericSelection.size === 0) {
        showNotification('Select one or more numeric columns to continue.', 'warning');
        return;
    }

    const confirmBtn = document.getElementById('timeSeriesModalConfirmBtn');
    let previousLabel = null;
    if (confirmBtn) {
        previousLabel = confirmBtn.innerHTML;
        confirmBtn.disabled = true;
        confirmBtn.setAttribute('aria-busy', 'true');
        confirmBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Running…
        `;
    }

    timeSeriesModalConfirmed = true;
    const modalElement = document.getElementById('timeSeriesAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    modalInstance?.hide();

    try {
        if (!currentTimeSeriesCellId) {
            const fallbackCellId = await addSingleAnalysisCell(currentTimeSeriesAnalysisType, {
                skipTimeSeriesModal: true
            });
            currentTimeSeriesCellId = fallbackCellId || '';
        }

        if (!currentTimeSeriesCellId) {
            showNotification('Unable to start time series analysis: no analysis cell available.', 'error');
            return;
        }

        const numericSelection = timeSeriesModalRequiresNumeric
            ? Array.from(timeSeriesModalNumericSelection)
            : [];

        const finalSelection = [...dateSelection, ...numericSelection];
        const analysisName = getAnalysisTypeName(currentTimeSeriesAnalysisType);
        const numericMessage = timeSeriesModalRequiresNumeric
            ? ` and ${numericSelection.length} numeric column${numericSelection.length === 1 ? '' : 's'}`
            : '';
        showNotification(`Running ${analysisName} with ${dateSelection.length} datetime column${dateSelection.length === 1 ? '' : 's'}${numericMessage}.`, 'success');

        await generateAndRunAnalysis(
            currentTimeSeriesCellId,
            currentTimeSeriesAnalysisType,
            {},
            {
                overrideSelectedColumns: finalSelection,
                includeGlobalSelectedColumns: false,
                modalType: 'time-series'
            }
        );
    } catch (error) {
        console.error('Time series analysis run failed:', error);
        showNotification(`Time series analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.removeAttribute('aria-busy');
            if (previousLabel !== null) {
                confirmBtn.innerHTML = previousLabel;
            }
        }
        resetTimeSeriesModalState();
    }
}

async function openTimeSeriesModalForRerun(cellId, analysisType, previousSelection = []) {
    configureTimeSeriesModalFlags(analysisType);
    timeSeriesModalIsRerun = true;
    timeSeriesModalConfirmed = false;
    currentTimeSeriesAnalysisType = analysisType;
    currentTimeSeriesCellId = cellId;
    timeSeriesModalSelectedDates = new Set();
    timeSeriesModalNumericSelection = new Set();
    timeSeriesModalDateSearchTerm = '';
    timeSeriesModalNumericSearchTerm = '';

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before time series rerun modal:', error);
        }
    }

    const datetimeCandidates = getDatetimeColumnCandidates();
    if (!datetimeCandidates.length) {
        showNotification('No datetime-style columns detected. Unable to reconfigure time series analysis.', 'warning');
        timeSeriesModalIsRerun = false;
        return;
    }

    let numericCandidates = [];
    if (timeSeriesModalRequiresNumeric) {
        numericCandidates = getNumericColumnCandidates();
        if (!numericCandidates.length) {
            showNotification('No numeric columns detected. Unable to reconfigure time series analysis.', 'warning');
            timeSeriesModalIsRerun = false;
            return;
        }
    }

    timeSeriesModalDateColumns = datetimeCandidates;
    timeSeriesModalNumericColumns = numericCandidates;

    initializeTimeSeriesModal();
    attachTimeSeriesModalLifecycleHandlers();
    populateTimeSeriesModal(analysisType, datetimeCandidates, numericCandidates, previousSelection);

    const modalElement = document.getElementById('timeSeriesAnalysisModal');
    if (!modalElement) {
        showNotification('Time series configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
    showNotification('Review or adjust your previously selected columns, then rerun.', 'info');
}

function resetTimeSeriesModalState() {
    currentTimeSeriesAnalysisType = '';
    currentTimeSeriesCellId = '';
    timeSeriesModalConfirmed = false;
    timeSeriesModalIsRerun = false;
    timeSeriesModalSelectedDates = new Set();
    timeSeriesModalNumericSelection = new Set();
    timeSeriesModalDateColumns = [];
    timeSeriesModalNumericColumns = [];
    timeSeriesModalDateRecommendedDefaults = [];
    timeSeriesModalNumericRecommendedDefaults = [];
    timeSeriesModalDateSearchTerm = '';
    timeSeriesModalNumericSearchTerm = '';
    timeSeriesModalRequiresNumeric = false;
    timeSeriesModalAllowsMultipleDates = false;
}


const GEOSPATIAL_LAT_HINTS = ['latitude', 'lat', 'latitud', 'lat_deg', 'y', 'northing'];
const GEOSPATIAL_LON_HINTS = ['longitude', 'lon', 'lng', 'long', 'lon_deg', 'x', 'easting'];
const GEOSPATIAL_LABEL_HINTS = ['name', 'city', 'region', 'area', 'location', 'place', 'label', 'site'];

const GEOSPATIAL_PROXIMITY_ANALYSIS_ID = 'geospatial_proximity_analysis';

const GEOSPATIAL_LABEL_DISABLED_TYPES = new Set([
    'coordinate_system_projection_check',
    'spatial_distribution_analysis',
    'spatial_data_quality_analysis'
]);

let currentGeospatialAnalysisType = '';
let currentGeospatialCellId = '';
let geospatialModalConfirmed = false;
let geospatialModalSelection = {
    latitude: '',
    longitude: '',
    label: '',
    proximityRadiusKm: '',
    referenceLatitude: '',
    referenceLongitude: '',
    comparisonLatitude: '',
    comparisonLongitude: '',
    optionalMode: 'none'
};
let geospatialModalColumns = [];
let geospatialModalIsRerun = false;

function geospatialAnalysisSupportsLabel(analysisType) {
    if (!analysisType) {
        return true;
    }
    return !GEOSPATIAL_LABEL_DISABLED_TYPES.has(analysisType);
}

function geospatialAnalysisUsesProximityOptions(analysisType) {
    return analysisType === GEOSPATIAL_PROXIMITY_ANALYSIS_ID;
}

async function prepareGeospatialAnalysisConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    currentGeospatialAnalysisType = analysisType;
    geospatialModalConfirmed = false;
    geospatialModalIsRerun = false;
    geospatialModalSelection = {
        latitude: '',
        longitude: '',
        label: '',
        proximityRadiusKm: '',
        referenceLatitude: '',
        referenceLongitude: '',
        comparisonLatitude: '',
        comparisonLongitude: '',
        optionalMode: 'none'
    };

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before geospatial modal:', error);
        }
    }

    geospatialModalColumns = buildGeospatialColumnCandidates();

    if (geospatialModalColumns.length < 2) {
        showNotification('Not enough columns detected to configure geospatial analyses. Running with automatic detection instead.', 'warning');
        await addSingleAnalysisCell(analysisType);
        resetGeospatialModalState();
        return;
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipGeospatialModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare geospatial analysis cell. Please try again.', 'error');
        resetGeospatialModalState();
        return;
    }

    currentGeospatialCellId = cellId;

    initializeGeospatialModal();
    attachGeospatialModalLifecycleHandlers();

    const supportsLabel = geospatialAnalysisSupportsLabel(analysisType);
    const recommendations = detectGeospatialRecommendations(geospatialModalColumns, supportsLabel);
    populateGeospatialModal(analysisType, geospatialModalColumns, recommendations, supportsLabel);

    const modalElement = document.getElementById('geospatialAnalysisModal');
    if (!modalElement) {
        showNotification('Geospatial configuration modal is unavailable.', 'error');
        resetGeospatialModalState();
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    if (recommendations.latitude && recommendations.longitude) {
        showNotification(`Confirm latitude/longitude columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
    } else {
        showNotification('Select latitude and longitude columns to continue.', 'info');
    }
}

function initializeGeospatialModal() {
    const modalElement = document.getElementById('geospatialAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const latitudeSelect = document.getElementById('geospatialLatitudeSelect');
    const longitudeSelect = document.getElementById('geospatialLongitudeSelect');
    const labelSelect = document.getElementById('geospatialLabelSelect');
    const confirmBtn = document.getElementById('geospatialModalConfirmBtn');
    const autoBtn = document.getElementById('geospatialModalAutoBtn');
    const radiusInput = document.getElementById('geospatialProximityRadiusInput');
    const referenceLatInput = document.getElementById('geospatialReferenceLatitudeInput');
    const referenceLonInput = document.getElementById('geospatialReferenceLongitudeInput');
    const comparisonLatSelect = document.getElementById('geospatialComparisonLatitudeSelect');
    const comparisonLonSelect = document.getElementById('geospatialComparisonLongitudeSelect');
    const optionalModeRadios = document.querySelectorAll('input[name="geospatialOptionalMode"]');

    if (latitudeSelect) {
        latitudeSelect.addEventListener('change', event => {
            geospatialModalSelection.latitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (longitudeSelect) {
        longitudeSelect.addEventListener('change', event => {
            geospatialModalSelection.longitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (labelSelect) {
        labelSelect.addEventListener('change', event => {
            geospatialModalSelection.label = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (radiusInput) {
        radiusInput.addEventListener('input', event => {
            geospatialModalSelection.proximityRadiusKm = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (referenceLatInput) {
        referenceLatInput.addEventListener('input', event => {
            geospatialModalSelection.referenceLatitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (referenceLonInput) {
        referenceLonInput.addEventListener('input', event => {
            geospatialModalSelection.referenceLongitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (comparisonLatSelect) {
        comparisonLatSelect.addEventListener('change', event => {
            geospatialModalSelection.comparisonLatitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (comparisonLonSelect) {
        comparisonLonSelect.addEventListener('change', event => {
            geospatialModalSelection.comparisonLongitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (optionalModeRadios && optionalModeRadios.length) {
        optionalModeRadios.forEach(radio => {
            radio.addEventListener('change', event => {
                const selectedMode = (event.target.value || '').trim();
                setGeospatialOptionalMode(selectedMode);
            });
        });
    }

    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchGeospatialAnalysis);
    }

    if (autoBtn) {
        autoBtn.addEventListener('click', () => {
            if (!geospatialModalColumns.length) {
                updateGeospatialAutoStatus('No columns available to auto-detect.', 'warning');
                return;
            }
            const supportsLabel = geospatialAnalysisSupportsLabel(currentGeospatialAnalysisType);
            const recommendations = detectGeospatialRecommendations(geospatialModalColumns, supportsLabel);
            applyGeospatialSelection(recommendations, true, supportsLabel);
        });
    }

    modalElement.dataset.initialized = 'true';
}

function attachGeospatialModalLifecycleHandlers() {
    const modalElement = document.getElementById('geospatialAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleGeospatialModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleGeospatialModalHidden() {
    if (geospatialModalConfirmed) {
        geospatialModalConfirmed = false;
        geospatialModalIsRerun = false;
        geospatialModalSelection = {
            latitude: '',
            longitude: '',
            label: '',
            proximityRadiusKm: '',
            referenceLatitude: '',
            referenceLongitude: '',
            comparisonLatitude: '',
            comparisonLongitude: '',
            optionalMode: 'none'
        };
        return;
    }

    if (geospatialModalIsRerun) {
        geospatialModalIsRerun = false;
        currentGeospatialAnalysisType = '';
        geospatialModalSelection = {
            latitude: '',
            longitude: '',
            label: '',
            proximityRadiusKm: '',
            referenceLatitude: '',
            referenceLongitude: '',
            comparisonLatitude: '',
            comparisonLongitude: ''
        };
        showNotification('Geospatial analysis rerun cancelled.', 'info');
        return;
    }

    if (currentGeospatialCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentGeospatialCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        showNotification('Geospatial analysis cancelled.', 'info');
    }

    resetGeospatialModalState();
}

function resetGeospatialModalState() {
    currentGeospatialAnalysisType = '';
    currentGeospatialCellId = '';
    geospatialModalSelection = {
        latitude: '',
        longitude: '',
        label: '',
        proximityRadiusKm: '',
        referenceLatitude: '',
        referenceLongitude: '',
        comparisonLatitude: '',
        comparisonLongitude: '',
        optionalMode: 'none'
    };
    geospatialModalColumns = [];
    geospatialModalConfirmed = false;
    geospatialModalIsRerun = false;
    const radiusInput = document.getElementById('geospatialProximityRadiusInput');
    const referenceLatInput = document.getElementById('geospatialReferenceLatitudeInput');
    const referenceLonInput = document.getElementById('geospatialReferenceLongitudeInput');
    const comparisonLatSelect = document.getElementById('geospatialComparisonLatitudeSelect');
    const comparisonLonSelect = document.getElementById('geospatialComparisonLongitudeSelect');
    if (radiusInput) {
        radiusInput.value = '';
    }
    if (referenceLatInput) {
        referenceLatInput.value = '';
    }
    if (referenceLonInput) {
        referenceLonInput.value = '';
    }
    if (comparisonLatSelect) {
        comparisonLatSelect.value = '';
    }
    if (comparisonLonSelect) {
        comparisonLonSelect.value = '';
    }
    updateGeospatialOptionalModeUI('none');
    updateGeospatialAutoStatus('');
    updateGeospatialSelectionSummary();
}

function buildGeospatialColumnCandidates() {
    const candidates = [];
    const seen = new Set();

    const pushCandidate = (name, meta = {}) => {
        const normalized = typeof name === 'string' ? name.trim() : String(name || '').trim();
        if (!normalized || seen.has(normalized)) {
            return;
        }
        seen.add(normalized);
        const lowerName = normalized.toLowerCase();
        const dataType = typeof meta.dataType === 'string' ? meta.dataType : '';
        const dataCategory = typeof meta.dataCategory === 'string' ? meta.dataCategory : '';
        const numericLikely = typeof meta.numericLikely === 'boolean'
            ? meta.numericLikely
            : /int|float|double|decimal|number/.test(dataType) || dataCategory === 'numeric';
        const categoricalLikely = typeof meta.categoricalLikely === 'boolean'
            ? meta.categoricalLikely
            : dataCategory === 'categorical' || /string|object|category|text/.test(dataType);

        candidates.push({
            name: normalized,
            dataType,
            dataCategory,
            sampleValue: meta.sampleValue,
            isLatitudeHint: GEOSPATIAL_LAT_HINTS.some(hint => lowerName.includes(hint)),
            isLongitudeHint: GEOSPATIAL_LON_HINTS.some(hint => lowerName.includes(hint)),
            isLabelHint: GEOSPATIAL_LABEL_HINTS.some(hint => lowerName.includes(hint)),
            numericLikely,
            categoricalLikely
        });
    };

    if (Array.isArray(columnInsightsData?.column_insights)) {
        columnInsightsData.column_insights.forEach(col => {
            if (!col || col.dropped) {
                return;
            }
            const meta = {
                dataType: (col.data_type || '').toString().toLowerCase(),
                dataCategory: (col.data_category || '').toString().toLowerCase(),
                sampleValue: col.example_value || col.sample_value || null
            };
            pushCandidate(col.name, meta);
        });
    }

    if (candidates.length === 0) {
        let fallbackColumns = [];
        if (window.currentDataFrame && Array.isArray(window.currentDataFrame.columns)) {
            fallbackColumns = window.currentDataFrame.columns;
        } else {
            try {
                const storedColumns = sessionStorage.getItem('datasetColumns');
                if (storedColumns) {
                    const parsed = JSON.parse(storedColumns);
                    if (Array.isArray(parsed)) {
                        fallbackColumns = parsed;
                    }
                }
            } catch (storageError) {
                console.warn('Unable to parse stored dataset columns for geospatial modal:', storageError);
            }
        }

        fallbackColumns.forEach(name => pushCandidate(name));
    }

    candidates.sort((a, b) => {
        const scoreDiff = computeGeospatialScore(b) - computeGeospatialScore(a);
        if (scoreDiff !== 0) {
            return scoreDiff;
        }
        return a.name.localeCompare(b.name);
    });

    return candidates;
}

function computeGeospatialScore(column) {
    let score = 0;
    if (column.isLatitudeHint) {
        score += 12;
    }
    if (column.isLongitudeHint) {
        score += 12;
    }
    if (column.numericLikely) {
        score += 4;
    }
    if (column.isLabelHint) {
        score += 6;
    }
    if (column.categoricalLikely && !column.numericLikely) {
        score += 2;
    }
    return score;
}

function detectGeospatialRecommendations(columns, includeLabel = true) {
    const selection = { latitude: '', longitude: '', label: '' };
    const used = new Set();

    const isNumericCandidate = column => {
        if (column.numericLikely) {
            return true;
        }

        const type = typeof column.dataType === 'string' ? column.dataType : '';
        if (/int|float|double|decimal|number|degree/.test(type)) {
            return true;
        }

        if (typeof column.sampleValue === 'number') {
            return true;
        }
        if (typeof column.sampleValue === 'string') {
            const trimmed = column.sampleValue.trim();
            if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
                return true;
            }
        }

        return false;
    };

    const coordinatePool = columns.filter(col => isNumericCandidate(col) && !col.isLabelHint);

    const pickCoordinate = hintKey => {
        const hinted = coordinatePool.filter(col => col[hintKey] && !used.has(col.name));
        if (hinted.length) {
            used.add(hinted[0].name);
            return hinted[0].name;
        }

        const available = coordinatePool.filter(col => !used.has(col.name));
        if (available.length) {
            used.add(available[0].name);
            return available[0].name;
        }

        const fallbackHinted = columns.filter(col => col[hintKey] && !col.isLabelHint && !used.has(col.name));
        if (fallbackHinted.length) {
            used.add(fallbackHinted[0].name);
            return fallbackHinted[0].name;
        }

        const fallbackAny = columns.filter(col => !col.isLabelHint && !used.has(col.name));
        if (fallbackAny.length) {
            used.add(fallbackAny[0].name);
            return fallbackAny[0].name;
        }

        return '';
    };

    selection.latitude = pickCoordinate('isLatitudeHint');
    selection.longitude = pickCoordinate('isLongitudeHint');

    if (!selection.longitude && selection.latitude) {
        const remaining = coordinatePool.filter(col => !used.has(col.name));
        if (remaining.length) {
            selection.longitude = remaining[0].name;
            used.add(selection.longitude);
        }
    }

    if (includeLabel) {
        const labelCandidates = columns.filter(col =>
            !used.has(col.name) &&
            (col.isLabelHint || col.categoricalLikely)
        );
        if (labelCandidates.length) {
            selection.label = labelCandidates[0].name;
        }
    }

    return selection;
}

function toggleGeospatialLabelVisibility(supportsLabel) {
    const labelGroup = document.getElementById('geospatialLabelGroup');
    const labelSelect = document.getElementById('geospatialLabelSelect');

    if (labelGroup) {
        if (supportsLabel) {
            labelGroup.classList.remove('d-none');
        } else {
            labelGroup.classList.add('d-none');
        }
    }

    if (labelSelect) {
        if (supportsLabel) {
            labelSelect.disabled = false;
        } else {
            labelSelect.value = '';
            labelSelect.disabled = true;
        }
    }
}

function toggleGeospatialProximityOptions(analysisType) {
    const shouldShow = geospatialAnalysisUsesProximityOptions(analysisType);

    if (!shouldShow) {
        setGeospatialOptionalMode('none');
        const optionalGroup = document.getElementById('geospatialOptionalModeGroup');
        if (optionalGroup) {
            optionalGroup.classList.add('d-none');
        }
        return;
    }

    setGeospatialOptionalMode(geospatialModalSelection.optionalMode || 'none', { preserveValues: true });
}

function setGeospatialProximityInputs(selection = {}) {
    const radiusInput = document.getElementById('geospatialProximityRadiusInput');
    const referenceLatInput = document.getElementById('geospatialReferenceLatitudeInput');
    const referenceLonInput = document.getElementById('geospatialReferenceLongitudeInput');

    const radiusValue = selection.proximityRadiusKm !== undefined && selection.proximityRadiusKm !== null
        ? String(selection.proximityRadiusKm).trim()
        : (geospatialModalSelection.proximityRadiusKm || '');
    const refLatValue = selection.referenceLatitude !== undefined && selection.referenceLatitude !== null
        ? String(selection.referenceLatitude).trim()
        : (geospatialModalSelection.referenceLatitude || '');
    const refLonValue = selection.referenceLongitude !== undefined && selection.referenceLongitude !== null
        ? String(selection.referenceLongitude).trim()
        : (geospatialModalSelection.referenceLongitude || '');

    if (radiusInput) {
        radiusInput.value = radiusValue;
    }
    if (referenceLatInput) {
        referenceLatInput.value = refLatValue;
    }
    if (referenceLonInput) {
        referenceLonInput.value = refLonValue;
    }

    geospatialModalSelection = {
        ...geospatialModalSelection,
        proximityRadiusKm: radiusValue,
        referenceLatitude: refLatValue,
        referenceLongitude: refLonValue
    };

    updateGeospatialSelectionSummary();
}

function setGeospatialComparisonInputs(selection = {}) {
    const comparisonLatSelect = document.getElementById('geospatialComparisonLatitudeSelect');
    const comparisonLonSelect = document.getElementById('geospatialComparisonLongitudeSelect');

    const comparisonLatValue = selection.comparisonLatitude !== undefined && selection.comparisonLatitude !== null
        ? String(selection.comparisonLatitude).trim()
        : (geospatialModalSelection.comparisonLatitude || '');
    const comparisonLonValue = selection.comparisonLongitude !== undefined && selection.comparisonLongitude !== null
        ? String(selection.comparisonLongitude).trim()
        : (geospatialModalSelection.comparisonLongitude || '');

    if (comparisonLatSelect) {
        comparisonLatSelect.value = comparisonLatValue;
    }
    if (comparisonLonSelect) {
        comparisonLonSelect.value = comparisonLonValue;
    }

    geospatialModalSelection = {
        ...geospatialModalSelection,
        comparisonLatitude: comparisonLatValue,
        comparisonLongitude: comparisonLonValue
    };

    updateGeospatialSelectionSummary();
}

function updateGeospatialOptionalModeUI(mode) {
    const normalized = mode === 'radius' || mode === 'comparison' ? mode : 'none';
    const optionalGroup = document.getElementById('geospatialOptionalModeGroup');
    if (optionalGroup) {
        const shouldShow = geospatialAnalysisUsesProximityOptions(currentGeospatialAnalysisType);
        optionalGroup.classList.toggle('d-none', !shouldShow);
    }

    const proximityCard = document.getElementById('geospatialProximityConfig');
    const comparisonCard = document.getElementById('geospatialComparisonConfig');
    if (proximityCard) {
        proximityCard.classList.toggle('d-none', normalized !== 'radius');
    }
    if (comparisonCard) {
        comparisonCard.classList.toggle('d-none', normalized !== 'comparison');
    }

    const modeNone = document.getElementById('geospatialOptionalModeNone');
    const modeRadius = document.getElementById('geospatialOptionalModeRadius');
    const modeComparison = document.getElementById('geospatialOptionalModeComparison');

    if (modeNone) {
        modeNone.checked = normalized === 'none';
    }
    if (modeRadius) {
        modeRadius.checked = normalized === 'radius';
    }
    if (modeComparison) {
        modeComparison.checked = normalized === 'comparison';
    }
}

function setGeospatialOptionalMode(mode, options = {}) {
    const normalized = mode === 'radius' || mode === 'comparison' ? mode : 'none';
    const previousMode = geospatialModalSelection.optionalMode;
    geospatialModalSelection = {
        ...geospatialModalSelection,
        optionalMode: normalized
    };

    updateGeospatialOptionalModeUI(normalized);

    const preserveValues = !!options.preserveValues;

    if (!preserveValues) {
        if (normalized !== 'radius') {
            setGeospatialProximityInputs({ proximityRadiusKm: '', referenceLatitude: '', referenceLongitude: '' });
        }
        if (normalized !== 'comparison') {
            setGeospatialComparisonInputs({ comparisonLatitude: '', comparisonLongitude: '' });
        }
        if (normalized === 'none') {
            updateGeospatialSelectionSummary();
        }
        return;
    }

    updateGeospatialSelectionSummary();
}

function populateGeospatialModal(analysisType, columns, initialSelection = null, supportsLabelOverride = null) {
    const modalLabel = document.getElementById('geospatialAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.innerHTML = `<i class="bi bi-geo-alt"></i> <span>Configure ${getAnalysisTypeName(analysisType)}</span>`;
    }

    const subtitle = document.getElementById('geospatialModalSubtitle');
    if (subtitle) {
        if (analysisType === 'spatial_data_quality_analysis') {
            subtitle.textContent = 'Review coordinate completeness, validity, and duplication issues for your dataset.';
        } else if (analysisType === GEOSPATIAL_PROXIMITY_ANALYSIS_ID) {
            subtitle.textContent = 'Select coordinates, then optionally add a radius or manual reference point to measure proximity.';
        } else {
            subtitle.textContent = 'Choose latitude and longitude columns so maps and proximity metrics can run accurately.';
        }
    }

    toggleGeospatialProximityOptions(analysisType);
    if (geospatialAnalysisUsesProximityOptions(analysisType)) {
        const proximitySelection = initialSelection || {};
        const initialMode = proximitySelection.optionalMode || geospatialModalSelection.optionalMode || 'none';
        geospatialModalSelection = {
            ...geospatialModalSelection,
            optionalMode: initialMode
        };
        setGeospatialOptionalMode(initialMode, { preserveValues: true });
        setGeospatialProximityInputs({
            proximityRadiusKm: proximitySelection.proximityRadiusKm ?? proximitySelection.radiusKm ?? '',
            referenceLatitude: proximitySelection.referenceLatitude ?? '',
            referenceLongitude: proximitySelection.referenceLongitude ?? ''
        });
        setGeospatialComparisonInputs({
            comparisonLatitude: proximitySelection.comparisonLatitude ?? '',
            comparisonLongitude: proximitySelection.comparisonLongitude ?? ''
        });
    }

    const latSelect = document.getElementById('geospatialLatitudeSelect');
    const lonSelect = document.getElementById('geospatialLongitudeSelect');
    const labelSelect = document.getElementById('geospatialLabelSelect');
    const comparisonLatSelect = document.getElementById('geospatialComparisonLatitudeSelect');
    const comparisonLonSelect = document.getElementById('geospatialComparisonLongitudeSelect');

    const supportsLabel = typeof supportsLabelOverride === 'boolean'
        ? supportsLabelOverride
        : geospatialAnalysisSupportsLabel(analysisType);

    const defaults = initialSelection || { latitude: '', longitude: '', label: '' };
    geospatialModalSelection = {
        ...geospatialModalSelection,
        latitude: defaults.latitude || '',
        longitude: defaults.longitude || '',
        label: supportsLabel ? (defaults.label || '') : ''
    };

    toggleGeospatialLabelVisibility(supportsLabel);

    populateGeospatialSelect(latSelect, columns, geospatialModalSelection.latitude, 'latitude');
    populateGeospatialSelect(lonSelect, columns, geospatialModalSelection.longitude, 'longitude');

    if (supportsLabel) {
        populateGeospatialSelect(labelSelect, columns, geospatialModalSelection.label, 'label');
    } else if (labelSelect) {
        labelSelect.innerHTML = '<option value="">Label not used for this analysis</option>';
        labelSelect.value = '';
    }

    populateGeospatialSelect(comparisonLatSelect, columns, geospatialModalSelection.comparisonLatitude, 'comparison-latitude');
    populateGeospatialSelect(comparisonLonSelect, columns, geospatialModalSelection.comparisonLongitude, 'comparison-longitude');

    updateGeospatialSelectionSummary();
    updateGeospatialAutoStatus('');
}

function populateGeospatialSelect(selectElement, columns, selectedValue, role) {
    if (!selectElement) {
        return;
    }

    const options = [];

    if (role === 'label') {
        options.push('<option value="">None (skip label)</option>');
    } else if (role === 'comparison-latitude') {
        options.push('<option value="">None (skip comparison latitude)</option>');
    } else if (role === 'comparison-longitude') {
        options.push('<option value="">None (skip comparison longitude)</option>');
    } else {
        const placeholder = role === 'latitude' ? 'Select latitude column' : 'Select longitude column';
        options.push(`<option value="">${placeholder}</option>`);
    }

    columns.forEach(col => {
        const hints = [];
        if (col.isLatitudeHint) {
            hints.push('lat');
        }
        if (col.isLongitudeHint) {
            hints.push('lon');
        }
        if (col.isLabelHint) {
            hints.push('label');
        }

        const hintText = hints.length ? ` • ${hints.join(', ')}` : '';
        const typeText = col.dataType ? ` (${col.dataType})` : '';
        const selectedAttr = selectedValue && col.name === selectedValue ? ' selected' : '';

        options.push(`
            <option value="${escapeHtml(col.name)}"${selectedAttr}>
                ${escapeHtml(col.name)}${hintText}${typeText}
            </option>
        `);
    });

    selectElement.innerHTML = options.join('');

    if (selectedValue) {
        selectElement.value = selectedValue;
    } else if (role === 'label' || role === 'comparison-latitude' || role === 'comparison-longitude') {
        selectElement.value = '';
    }
}

function applyGeospatialSelection(selection, showMessage = false, supportsLabelOverride = null) {
    const supportsLabel = typeof supportsLabelOverride === 'boolean'
        ? supportsLabelOverride
        : geospatialAnalysisSupportsLabel(currentGeospatialAnalysisType);

    geospatialModalSelection = {
        ...geospatialModalSelection,
        latitude: selection.latitude || '',
        longitude: selection.longitude || '',
        label: supportsLabel ? (selection.label || '') : ''
    };

    const latSelect = document.getElementById('geospatialLatitudeSelect');
    const lonSelect = document.getElementById('geospatialLongitudeSelect');
    const labelSelect = document.getElementById('geospatialLabelSelect');

    toggleGeospatialLabelVisibility(supportsLabel);

    if (latSelect) {
        latSelect.value = geospatialModalSelection.latitude;
    }
    if (lonSelect) {
        lonSelect.value = geospatialModalSelection.longitude;
    }
    if (labelSelect) {
        labelSelect.value = supportsLabel ? (geospatialModalSelection.label || '') : '';
    }

    updateGeospatialSelectionSummary();

    if (showMessage) {
        if (geospatialModalSelection.latitude && geospatialModalSelection.longitude) {
            updateGeospatialAutoStatus(`Detected ${geospatialModalSelection.latitude} as latitude and ${geospatialModalSelection.longitude} as longitude.`, 'success');
        } else {
            updateGeospatialAutoStatus('Unable to auto-detect both latitude and longitude.', 'warning');
        }
    }
}

function updateGeospatialSelectionSummary() {
    const summaryElement = document.getElementById('geospatialSelectionSummary');
    const confirmBtn = document.getElementById('geospatialModalConfirmBtn');

    if (!summaryElement) {
        return;
    }

    const {
        latitude,
        longitude,
        label,
        proximityRadiusKm,
        referenceLatitude,
        referenceLongitude,
        comparisonLatitude,
        comparisonLongitude,
        optionalMode
    } = geospatialModalSelection;
    const supportsLabel = geospatialAnalysisSupportsLabel(currentGeospatialAnalysisType);
    const effectiveOptionalMode = optionalMode === 'radius' || optionalMode === 'comparison' ? optionalMode : 'none';

    summaryElement.classList.remove('alert-success', 'alert-warning', 'alert-secondary');

    if (!latitude || !longitude) {
        summaryElement.classList.add('alert-secondary');
        summaryElement.textContent = 'Select latitude and longitude columns to continue.';
        if (confirmBtn) {
            confirmBtn.disabled = true;
        }
        return;
    }

    if (latitude === longitude) {
        summaryElement.classList.add('alert-warning');
        summaryElement.textContent = 'Latitude and longitude must be different columns.';
        if (confirmBtn) {
            confirmBtn.disabled = true;
        }
        return;
    }

    const comparisonSelected = comparisonLatitude && comparisonLongitude;
    if (effectiveOptionalMode === 'comparison' && !comparisonSelected) {
        summaryElement.classList.add('alert-warning');
        summaryElement.textContent = 'Select both comparison latitude and longitude columns to focus on coordinate differences.';
        if (confirmBtn) {
            confirmBtn.disabled = true;
        }
        return;
    }

    if (comparisonLatitude && !comparisonLongitude) {
        summaryElement.classList.add('alert-warning');
        summaryElement.textContent = 'Provide both comparison latitude and longitude columns to calculate differences, or clear both fields.';
        if (confirmBtn) {
            confirmBtn.disabled = true;
        }
        return;
    }

    if (!comparisonLatitude && comparisonLongitude) {
        summaryElement.classList.add('alert-warning');
        summaryElement.textContent = 'Provide both comparison latitude and longitude columns to calculate differences, or clear both fields.';
        if (confirmBtn) {
            confirmBtn.disabled = true;
        }
        return;
    }

    summaryElement.classList.add('alert-success');
    const detailParts = [`${latitude} (latitude)`, `${longitude} (longitude)`];
    if (supportsLabel && label) {
        detailParts.push(`${label} (label)`);
    }
    if (currentGeospatialAnalysisType === GEOSPATIAL_PROXIMITY_ANALYSIS_ID) {
        if (proximityRadiusKm) {
            detailParts.push(`${proximityRadiusKm} km radius`);
        }
        if (referenceLatitude && referenceLongitude) {
            detailParts.push(`reference (${referenceLatitude}, ${referenceLongitude})`);
        }
        if (comparisonLatitude && comparisonLongitude) {
            detailParts.push(`comparison vs ${comparisonLatitude}/${comparisonLongitude}`);
        }
    }

    summaryElement.textContent = `Ready to run with ${detailParts.join(', ')}.`;

    if (confirmBtn) {
        confirmBtn.disabled = false;
    }
}

function updateGeospatialAutoStatus(message, tone = 'muted') {
    const statusElement = document.getElementById('geospatialModalAutoStatus');
    if (!statusElement) {
        return;
    }

    statusElement.textContent = message || '';
    statusElement.classList.remove('text-success', 'text-warning', 'text-danger', 'text-muted');

    if (!message) {
        statusElement.classList.add('text-muted');
        return;
    }

    if (tone === 'success') {
        statusElement.classList.add('text-success');
    } else if (tone === 'warning') {
        statusElement.classList.add('text-warning');
    } else if (tone === 'danger') {
        statusElement.classList.add('text-danger');
    } else {
        statusElement.classList.add('text-muted');
    }
}

async function launchGeospatialAnalysis() {
    if (!currentGeospatialAnalysisType) {
        showNotification('No geospatial analysis selected.', 'error');
        return;
    }

    const {
        latitude,
        longitude,
        label,
        proximityRadiusKm,
        referenceLatitude,
        referenceLongitude,
        comparisonLatitude,
        comparisonLongitude,
        optionalMode
    } = geospatialModalSelection;

    if (!latitude || !longitude) {
        showNotification('Select both latitude and longitude columns to continue.', 'warning');
        return;
    }

    if (latitude === longitude) {
        showNotification('Latitude and longitude must reference different columns.', 'warning');
        return;
    }

    if ((comparisonLatitude && !comparisonLongitude) || (!comparisonLatitude && comparisonLongitude)) {
        showNotification('Provide both comparison latitude and longitude columns or leave both blank.', 'warning');
        return;
    }

    const confirmBtn = document.getElementById('geospatialModalConfirmBtn');
    let previousLabel = null;

    if (confirmBtn) {
        previousLabel = confirmBtn.innerHTML;
        confirmBtn.disabled = true;
        confirmBtn.setAttribute('aria-busy', 'true');
        confirmBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Running…
        `;
    }

    geospatialModalConfirmed = true;
    const modalElement = document.getElementById('geospatialAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    if (modalInstance) {
        modalInstance.hide();
    }

    try {
        const supportsLabel = geospatialAnalysisSupportsLabel(currentGeospatialAnalysisType);
        if (!currentGeospatialCellId) {
            const fallbackCellId = await addSingleAnalysisCell(currentGeospatialAnalysisType, {
                skipGeospatialModal: true
            });
            currentGeospatialCellId = fallbackCellId || '';
        }

        if (!currentGeospatialCellId) {
            showNotification('Unable to start geospatial analysis: no analysis cell available.', 'error');
            return;
        }

        const includeLabel = supportsLabel && !!label;
        const selectedColumns = includeLabel ? [latitude, longitude, label] : [latitude, longitude];
        const columnMapping = {
            latitude,
            longitude
        };
        if (includeLabel) {
            columnMapping.location_label = label;
        }

        const selectionDetails = { latitude, longitude };
        if (includeLabel) {
            selectionDetails.label = label;
        }
        if (currentGeospatialAnalysisType === GEOSPATIAL_PROXIMITY_ANALYSIS_ID) {
            selectionDetails.proximityRadiusKm = proximityRadiusKm || '';
            selectionDetails.referenceLatitude = referenceLatitude || '';
            selectionDetails.referenceLongitude = referenceLongitude || '';
            selectionDetails.comparisonLatitude = comparisonLatitude || '';
            selectionDetails.comparisonLongitude = comparisonLongitude || '';
            selectionDetails.optionalMode = optionalMode || 'none';
        }

        const analysisOptions = {
            overrideSelectedColumns: selectedColumns,
            includeGlobalSelectedColumns: false,
            modalType: 'geospatial',
            modalSelectionPayload: selectionDetails
        };

        if (currentGeospatialAnalysisType === GEOSPATIAL_PROXIMITY_ANALYSIS_ID) {
            const metadata = {};
            const parsedRadius = parseFloat(proximityRadiusKm);
            if (!Number.isNaN(parsedRadius) && Number.isFinite(parsedRadius) && parsedRadius > 0) {
                metadata.proximity_radius_km = parsedRadius;
            }
            const parsedRefLat = parseFloat(referenceLatitude);
            const parsedRefLon = parseFloat(referenceLongitude);
            if (!Number.isNaN(parsedRefLat) && Number.isFinite(parsedRefLat) && !Number.isNaN(parsedRefLon) && Number.isFinite(parsedRefLon)) {
                metadata.reference_latitude = parsedRefLat;
                metadata.reference_longitude = parsedRefLon;
            }
            if (comparisonLatitude && comparisonLongitude) {
                metadata.comparison_latitude_column = comparisonLatitude;
                metadata.comparison_longitude_column = comparisonLongitude;
            }
            const normalizedOptionalMode = optionalMode === 'comparison' || optionalMode === 'radius' ? optionalMode : 'none';
            if (normalizedOptionalMode !== 'none') {
                metadata.proximity_optional_mode = normalizedOptionalMode;
            }
            if (Object.keys(metadata).length > 0) {
                analysisOptions.analysisMetadata = metadata;
            }
        }

        showNotification(`Running ${getAnalysisTypeName(currentGeospatialAnalysisType)} with selected coordinates.`, 'success');

        await generateAndRunAnalysis(
            currentGeospatialCellId,
            currentGeospatialAnalysisType,
            columnMapping,
            analysisOptions
        );
    } catch (error) {
        console.error('Geospatial analysis execution failed:', error);
        showNotification(`Geospatial analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.removeAttribute('aria-busy');
            if (previousLabel !== null) {
                confirmBtn.innerHTML = previousLabel;
            }
        }

        resetGeospatialModalState();
    }
}

async function openGeospatialModalForRerun(cellId, analysisType, previousSelection = [], previousDetails = null) {
    geospatialModalIsRerun = true;
    geospatialModalConfirmed = false;
    currentGeospatialCellId = cellId;
    currentGeospatialAnalysisType = analysisType;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before geospatial rerun:', error);
        }
    }

    geospatialModalColumns = buildGeospatialColumnCandidates();

    if (geospatialModalColumns.length < 2) {
        geospatialModalIsRerun = false;
        showNotification('Not enough columns available to reconfigure geospatial analysis.', 'warning');
        return;
    }

    const supportsLabel = geospatialAnalysisSupportsLabel(analysisType);
    const initialSelection = rebuildGeospatialSelectionFromHistory(previousSelection, previousDetails, supportsLabel);

    initializeGeospatialModal();
    attachGeospatialModalLifecycleHandlers();
    populateGeospatialModal(analysisType, geospatialModalColumns, initialSelection, supportsLabel);

    const modalElement = document.getElementById('geospatialAnalysisModal');
    if (!modalElement) {
        geospatialModalIsRerun = false;
        showNotification('Geospatial configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    showNotification('Adjust the coordinate columns if needed, then rerun the analysis.', 'info');
}

function rebuildGeospatialSelectionFromHistory(previousSelection, previousDetails, includeLabel = true) {
    const selection = {
        latitude: '',
        longitude: '',
        label: '',
        proximityRadiusKm: '',
        referenceLatitude: '',
        referenceLongitude: '',
        comparisonLatitude: '',
        comparisonLongitude: '',
        optionalMode: 'none'
    };

    const includeProximity = geospatialAnalysisUsesProximityOptions(currentGeospatialAnalysisType);

    if (previousDetails && typeof previousDetails === 'object') {
        selection.latitude = typeof previousDetails.latitude === 'string' ? previousDetails.latitude : '';
        selection.longitude = typeof previousDetails.longitude === 'string' ? previousDetails.longitude : '';
        if (includeLabel) {
            selection.label = typeof previousDetails.label === 'string' ? previousDetails.label : '';
        }
        if (includeProximity) {
            if (typeof previousDetails.proximityRadiusKm === 'string' || typeof previousDetails.proximityRadiusKm === 'number') {
                selection.proximityRadiusKm = String(previousDetails.proximityRadiusKm).trim();
            }
            if (typeof previousDetails.referenceLatitude === 'string' || typeof previousDetails.referenceLatitude === 'number') {
                selection.referenceLatitude = String(previousDetails.referenceLatitude).trim();
            }
            if (typeof previousDetails.referenceLongitude === 'string' || typeof previousDetails.referenceLongitude === 'number') {
                selection.referenceLongitude = String(previousDetails.referenceLongitude).trim();
            }
            if (typeof previousDetails.comparisonLatitude === 'string') {
                selection.comparisonLatitude = previousDetails.comparisonLatitude.trim();
            }
            if (typeof previousDetails.comparisonLongitude === 'string') {
                selection.comparisonLongitude = previousDetails.comparisonLongitude.trim();
            }
            if (typeof previousDetails.optionalMode === 'string') {
                selection.optionalMode = previousDetails.optionalMode.trim();
            } else if (typeof previousDetails.optional_mode === 'string') {
                selection.optionalMode = previousDetails.optional_mode.trim();
            }
            if (previousDetails.comparison && typeof previousDetails.comparison === 'object') {
                const comparisonColumns = previousDetails.comparison.columns || {};
                if (!selection.comparisonLatitude && typeof comparisonColumns.latitude === 'string') {
                    selection.comparisonLatitude = comparisonColumns.latitude.trim();
                }
                if (!selection.comparisonLongitude && typeof comparisonColumns.longitude === 'string') {
                    selection.comparisonLongitude = comparisonColumns.longitude.trim();
                }
                if (!selection.optionalMode || selection.optionalMode === 'none') {
                    selection.optionalMode = 'comparison';
                }
            }
            if (!selection.optionalMode || selection.optionalMode === 'none') {
                const meta = previousDetails.metadata;
                if (meta && typeof meta === 'object' && typeof meta.proximity_optional_mode === 'string') {
                    selection.optionalMode = meta.proximity_optional_mode.trim();
                }
            }
        }
    }

    if ((!selection.latitude || !selection.longitude || (includeLabel && !selection.label)) && Array.isArray(previousSelection)) {
        const normalized = previousSelection
            .map(name => (typeof name === 'string' ? name.trim() : String(name || '')))
            .filter(Boolean);
        if (!selection.latitude && normalized.length > 0) {
            selection.latitude = normalized[0];
        }
        if (!selection.longitude && normalized.length > 1) {
            selection.longitude = normalized[1];
        }
        if (includeLabel && !selection.label && normalized.length > 2) {
            selection.label = normalized[2];
        }
    }

    if (!selection.latitude || !selection.longitude || (includeLabel && !selection.label)) {
        const recommendations = detectGeospatialRecommendations(geospatialModalColumns, includeLabel);
        if (!selection.latitude) {
            selection.latitude = recommendations.latitude;
        }
        if (!selection.longitude) {
            selection.longitude = recommendations.longitude;
        }
        if (includeLabel && !selection.label) {
            selection.label = recommendations.label;
        }
    }

    if (!includeLabel) {
        selection.label = '';
    }

    if (!includeProximity) {
        selection.proximityRadiusKm = '';
        selection.referenceLatitude = '';
        selection.referenceLongitude = '';
        selection.comparisonLatitude = '';
        selection.comparisonLongitude = '';
        selection.optionalMode = 'none';
    }

    if (selection.optionalMode !== 'radius' && selection.optionalMode !== 'comparison') {
        selection.optionalMode = 'none';
    }

    return selection;
}


'use strict';

const TEXT_CATEGORY_TOKENS = ['text', 'string', 'nlp', 'freeform', 'comment', 'description', 'feedback', 'review'];
const TEXT_TYPE_TOKENS = ['object', 'string', 'text'];
const TEXT_NAME_HINTS = ['comment', 'description', 'message', 'note', 'review', 'summary', 'text', 'content', 'body', 'feedback', 'title'];

async function prepareTextAnalysisConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    resetTextModalState();
    currentTextAnalysisType = analysisType;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before text modal:', error);
        }
    }

    const columnCandidates = getTextColumnCandidates();
    if (!columnCandidates.length) {
        showNotification('No text-like columns detected. Running analysis with default behaviour instead.', 'info');
        resetTextModalState();
        await addSingleAnalysisCell(analysisType);
        return;
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipTextModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare text analysis cell. Please try again.', 'error');
        resetTextModalState();
        return;
    }

    currentTextCellId = cellId;
    textModalColumns = columnCandidates;

    initializeTextModal();
    attachTextModalLifecycleHandlers();
    populateTextModal(analysisType, columnCandidates);

    const modalElement = document.getElementById('textAnalysisModal');
    if (!modalElement) {
        showNotification('Text configuration modal is unavailable.', 'error');
        resetTextModalState();
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
    showNotification(`Select text columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
}

function resetTextModalState() {
    currentTextAnalysisType = '';
    currentTextCellId = '';
    textModalConfirmed = false;
    textModalSelection = new Set();
    textModalColumns = [];
    textModalRecommendedDefaults = [];
    textModalSearchTerm = '';
    textModalIsRerun = false;
}

function attachTextModalLifecycleHandlers() {
    const modalElement = document.getElementById('textAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleTextModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleTextModalHidden() {
    if (textModalConfirmed) {
        textModalConfirmed = false;
        textModalIsRerun = false;
        return;
    }

    if (textModalIsRerun) {
        textModalIsRerun = false;
        const message = currentTextAnalysisType ? `${getAnalysisTypeName(currentTextAnalysisType)} rerun cancelled.` : 'Text analysis rerun cancelled.';
        showNotification(message, 'info');
        resetTextModalState();
        return;
    }

    if (currentTextCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentTextCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        if (currentTextAnalysisType) {
            showNotification('Text analysis cancelled.', 'info');
        }
    }

    resetTextModalState();
}

function initializeTextModal() {
    const modalElement = document.getElementById('textAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const confirmBtn = document.getElementById('textModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchTextAnalysis);
    }

    const selectAllBtn = document.getElementById('textModalSelectAllBtn');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', selectAllTextColumns);
    }

    const clearBtn = document.getElementById('textModalClearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            textModalSelection = new Set();
            renderTextColumnList();
            updateTextSelectionSummary();
            updateTextChipStates();
        });
    }

    const recommendedBtn = document.getElementById('textModalRecommendBtn');
    if (recommendedBtn) {
        recommendedBtn.addEventListener('click', applyTextRecommendations);
    }

    const searchInput = document.getElementById('textColumnSearch');
    if (searchInput) {
        searchInput.addEventListener('input', event => {
            textModalSearchTerm = (event.target.value || '').toLowerCase();
            renderTextColumnList();
            updateTextChipStates();
        });
    }

    const columnList = document.getElementById('textColumnList');
    if (columnList) {
        columnList.addEventListener('change', handleTextListChange);
    }

    const chipsContainer = document.getElementById('textRecommendationChips');
    if (chipsContainer) {
        chipsContainer.addEventListener('click', handleTextChipClick);
    }

    modalElement.dataset.initialized = 'true';
}

function getTextColumnCandidates() {
    if (!Array.isArray(columnInsightsData?.column_insights)) {
        return [];
    }

    const candidates = [];

    columnInsightsData.column_insights.forEach(col => {
        if (!col || col.dropped) {
            return;
        }

        const normalizedName = (col.name ?? '').toString().trim();
        if (!normalizedName) {
            return;
        }

        const lowerName = normalizedName.toLowerCase();
        const dataCategory = (col.data_category || '').toString().toLowerCase();
        const dataType = (col.data_type || '').toString().toLowerCase();
        const stats = typeof col.statistics === 'object' && col.statistics !== null ? col.statistics : {};

        const uniqueCount = toFiniteNumber(stats.unique_count ?? stats.distinct_count ?? stats.cardinality);
        const missingPct = toFiniteNumber(col.null_percentage ?? stats.null_percentage);
        const avgLength = toFiniteNumber(
            stats.avg_length ?? stats.average_length ?? stats.mean_length ?? stats.mean_char_length ?? stats.avg_char_length
        );
        const medianLength = toFiniteNumber(stats.median_length ?? stats.median_char_length);
        const maxLength = toFiniteNumber(stats.max_length ?? stats.max_char_length ?? stats.max_len);
        const avgWords = toFiniteNumber(stats.avg_word_count ?? stats.average_word_count);
        const alphaShare = toFiniteNumber(stats.alpha_ratio ?? stats.alpha_share ?? stats.alpha_percentage);
        const sampleValues = Array.isArray(stats.sample_values)
            ? stats.sample_values
            : Array.isArray(col.sample_values)
                ? col.sample_values
                : [];

        const sampleSuggestsText = sampleValues.some(value => typeof value === 'string' && value.split(/\s+/).length >= 3);
        const isTextCategory = TEXT_CATEGORY_TOKENS.some(token => dataCategory.includes(token));
        const isTextType = TEXT_TYPE_TOKENS.some(token => dataType.includes(token));
        const nameSuggestsText = TEXT_NAME_HINTS.some(token => lowerName.includes(token));
        const cardinalityReasonable = typeof uniqueCount === 'number' ? uniqueCount > 10 : true;
        const avgLengthReasonable = typeof avgLength === 'number' ? avgLength >= 8 : true;
        const alphaLikelyText = typeof alphaShare === 'number' ? alphaShare >= 0.4 : false;

        if (
            !isTextCategory &&
            !isTextType &&
            !nameSuggestsText &&
            !sampleSuggestsText &&
            !alphaLikelyText
        ) {
            return;
        }

        const reasonParts = [];
        if (typeof uniqueCount === 'number') {
            reasonParts.push(`${uniqueCount} unique`);
        }
        if (typeof avgLength === 'number') {
            reasonParts.push(`${avgLength.toFixed(1)} avg chars`);
        }
        if (typeof avgWords === 'number') {
            reasonParts.push(`${avgWords.toFixed(1)} avg words`);
        }
        if (typeof missingPct === 'number') {
            reasonParts.push(`${missingPct.toFixed(1)}% missing`);
        }
        if (alphaLikelyText) {
            reasonParts.push('Alphabetic rich');
        }
        if (nameSuggestsText) {
            reasonParts.push('Name hint');
        }

        const recommended = Boolean(
            (avgLengthReasonable && cardinalityReasonable) ||
            (sampleSuggestsText && (typeof avgWords !== 'number' || avgWords <= 80))
        );

        candidates.push({
            name: normalizedName,
            dataCategory: dataCategory || dataType || 'text',
            dataType,
            uniqueCount: typeof uniqueCount === 'number' ? uniqueCount : null,
            missingPct: typeof missingPct === 'number' ? missingPct : null,
            avgLength: typeof avgLength === 'number' ? avgLength : null,
            medianLength: typeof medianLength === 'number' ? medianLength : null,
            maxLength: typeof maxLength === 'number' ? maxLength : null,
            avgWords: typeof avgWords === 'number' ? avgWords : null,
            recommended,
            reason: reasonParts.join(' • ')
        });
    });

    candidates.sort((a, b) => {
        if (a.recommended !== b.recommended) {
            return a.recommended ? -1 : 1;
        }
        const aAvg = typeof a.avgLength === 'number' ? a.avgLength : -1;
        const bAvg = typeof b.avgLength === 'number' ? b.avgLength : -1;
        if (aAvg !== bAvg) {
            return bAvg - aAvg;
        }
        return a.name.localeCompare(b.name);
    });

    return candidates;
}

function populateTextModal(analysisType, columns, initialSelection = null) {
    const analysisName = getAnalysisTypeName(analysisType);
    const modalLabel = document.getElementById('textAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.textContent = `Configure ${analysisName}`;
    }

    const subtitle = document.getElementById('textModalSubtitle');
    if (subtitle) {
        subtitle.textContent = 'Pick text columns to review length distributions, token frequencies, and vocabulary richness.';
    }

    const confirmBtn = document.getElementById('textModalConfirmBtn');
    if (confirmBtn) {
        const baseLabel = `Run ${analysisName}`;
        confirmBtn.dataset.baseLabel = baseLabel;
        confirmBtn.textContent = baseLabel;
        confirmBtn.disabled = false;
    }

    const searchInput = document.getElementById('textColumnSearch');
    if (searchInput) {
        searchInput.value = '';
    }

    textModalColumns = Array.isArray(columns) ? [...columns] : [];

    const recommended = textModalColumns.filter(col => col.recommended);
    textModalRecommendedDefaults = recommended.slice(0, 6).map(col => col.name);

    const normalizedInitialSelection = Array.isArray(initialSelection) || initialSelection instanceof Set
        ? Array.from(new Set(initialSelection instanceof Set ? [...initialSelection] : initialSelection)).filter(name =>
            textModalColumns.some(col => col.name === name)
        )
        : [];

    if (normalizedInitialSelection.length > 0) {
        textModalSelection = new Set(normalizedInitialSelection);
    } else if (textModalRecommendedDefaults.length > 0) {
        textModalSelection = new Set(textModalRecommendedDefaults);
    } else {
        const fallback = textModalColumns.slice(0, Math.min(5, textModalColumns.length)).map(col => col.name);
        textModalSelection = new Set(fallback);
    }

    renderTextRecommendations(recommended);
    renderTextColumnList();
    updateTextChipStates();
    updateTextSelectionSummary();
}

function renderTextRecommendations(recommendedColumns) {
    const container = document.getElementById('textRecommendationChips');
    if (!container) {
        return;
    }

    container.innerHTML = '';

    if (!recommendedColumns.length) {
        container.innerHTML = '<span class="text-muted small">No automatic recommendations yet. Select columns manually below.</span>';
        return;
    }

    recommendedColumns.forEach(col => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-outline-primary btn-sm me-2 mb-2';
        button.dataset.column = col.name;
        if (col.reason) {
            button.title = col.reason;
        }
        button.textContent = col.name;
        container.appendChild(button);
    });
}

function renderTextColumnList() {
    const listElement = document.getElementById('textColumnList');
    if (!listElement) {
        return;
    }

    let filtered = [...textModalColumns];
    if (textModalSearchTerm) {
        filtered = filtered.filter(col => col.name.toLowerCase().includes(textModalSearchTerm));
    }

    if (!filtered.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No columns match your search.</div>';
        return;
    }

    const rows = filtered
        .map(col => {
            const checked = textModalSelection.has(col.name) ? 'checked' : '';
            const detailParts = [];
            if (typeof col.avgLength === 'number') {
                detailParts.push(`${col.avgLength.toFixed(1)} avg chars`);
            }
            if (typeof col.avgWords === 'number') {
                detailParts.push(`${col.avgWords.toFixed(1)} avg words`);
            }
            if (typeof col.uniqueCount === 'number') {
                detailParts.push(`${col.uniqueCount} unique`);
            }
            if (typeof col.missingPct === 'number') {
                detailParts.push(`${col.missingPct.toFixed(1)}% missing`);
            }
            const detailText = detailParts.length ? `<small class="text-muted">${detailParts.join(' • ')}</small>` : '';
            const badgeLabel = col.dataCategory || col.dataType || 'text';

            return `
                <label class="list-group-item d-flex align-items-start gap-3">
                    <input class="form-check-input mt-1" type="checkbox" value="${escapeHtml(col.name)}" data-column="${escapeHtml(col.name)}" ${checked}>
                    <div class="flex-grow-1">
                        <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                            <strong>${escapeHtml(col.name)}</strong>
                            <span class="badge text-bg-light text-capitalize">${escapeHtml(badgeLabel)}</span>
                        </div>
                        ${detailText}
                    </div>
                </label>
            `;
        })
        .join('');

    listElement.innerHTML = rows;
}

function updateTextSelectionSummary() {
    const summaryElement = document.getElementById('textSelectionSummary');
    const confirmBtn = document.getElementById('textModalConfirmBtn');
    const count = textModalSelection.size;

    if (summaryElement) {
        if (count === 0) {
            summaryElement.textContent = 'Select one or more text columns to continue.';
        } else {
            const preview = Array.from(textModalSelection).slice(0, 5);
            const remainder = count - preview.length;
            const suffix = remainder > 0 ? `, +${remainder} more` : '';
            summaryElement.textContent = `${count} column${count === 1 ? '' : 's'} selected: ${preview.join(', ')}${suffix}`;
        }
    }

    if (confirmBtn) {
        const baseLabel = confirmBtn.dataset.baseLabel || confirmBtn.textContent || 'Run analysis';
        confirmBtn.textContent = count > 0 ? `${baseLabel} (${count})` : baseLabel;
        confirmBtn.disabled = count === 0;
    }
}

function updateTextChipStates() {
    const chipsContainer = document.getElementById('textRecommendationChips');
    if (!chipsContainer) {
        return;
    }
    chipsContainer.querySelectorAll('[data-column]').forEach(button => {
        const columnName = button.dataset.column;
        button.classList.toggle('active', textModalSelection.has(columnName));
    });
}

function handleTextListChange(event) {
    const target = event.target;
    if (!target || target.type !== 'checkbox') {
        return;
    }

    const columnName = target.dataset.column;
    if (!columnName) {
        return;
    }

    if (target.checked) {
        textModalSelection.add(columnName);
    } else {
        textModalSelection.delete(columnName);
    }

    updateTextSelectionSummary();
    updateTextChipStates();
}

function handleTextChipClick(event) {
    const target = event.target;
    if (!target || !target.dataset?.column) {
        return;
    }

    const columnName = target.dataset.column;
    if (textModalSelection.has(columnName)) {
        textModalSelection.delete(columnName);
    } else {
        textModalSelection.add(columnName);
    }

    renderTextColumnList();
    updateTextSelectionSummary();
    updateTextChipStates();
}

function applyTextRecommendations() {
    if (!textModalRecommendedDefaults.length) {
        showNotification('No recommended text columns were identified yet.', 'info');
        return;
    }

    textModalSelection = new Set(textModalRecommendedDefaults);
    renderTextColumnList();
    updateTextSelectionSummary();
    updateTextChipStates();
}

function selectAllTextColumns() {
    if (!textModalColumns.length) {
        return;
    }

    textModalSelection = new Set(textModalColumns.map(col => col.name));
    renderTextColumnList();
    updateTextSelectionSummary();
    updateTextChipStates();
}

async function launchTextAnalysis() {
    if (!currentTextAnalysisType) {
        showNotification('No text analysis selected.', 'error');
        return;
    }

    const selectedList = Array.from(textModalSelection);
    if (!selectedList.length) {
        showNotification('Select at least one text column to continue.', 'warning');
        return;
    }

    const confirmBtn = document.getElementById('textModalConfirmBtn');
    let previousLabel = null;
    if (confirmBtn) {
        previousLabel = confirmBtn.innerHTML;
        confirmBtn.disabled = true;
        confirmBtn.setAttribute('aria-busy', 'true');
        confirmBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Running…
        `;
    }

    textModalConfirmed = true;
    const modalElement = document.getElementById('textAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    if (modalInstance) {
        modalInstance.hide();
    }

    try {
        if (!currentTextCellId) {
            const fallbackCellId = await addSingleAnalysisCell(currentTextAnalysisType, {
                skipTextModal: true
            });
            currentTextCellId = fallbackCellId || '';
        }

        if (!currentTextCellId) {
            showNotification('Unable to start text analysis: no analysis cell available.', 'error');
            return;
        }

        showNotification(`Running ${getAnalysisTypeName(currentTextAnalysisType)} for ${selectedList.length} column${selectedList.length === 1 ? '' : 's'}.`, 'success');

        await generateAndRunAnalysis(
            currentTextCellId,
            currentTextAnalysisType,
            {},
            {
                overrideSelectedColumns: selectedList,
                includeGlobalSelectedColumns: false,
                modalType: 'text'
            }
        );
    } catch (error) {
        console.error('Text analysis run failed:', error);
        showNotification(`Text analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.removeAttribute('aria-busy');
            if (previousLabel !== null) {
                confirmBtn.innerHTML = previousLabel;
            }
        }
        resetTextModalState();
    }
}

async function openTextModalForRerun(cellId, analysisType, previousSelection = []) {
    textModalIsRerun = true;
    currentTextAnalysisType = analysisType;
    currentTextCellId = cellId;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before text rerun:', error);
        }
    }

    const columnCandidates = getTextColumnCandidates();
    if (!columnCandidates.length) {
        textModalIsRerun = false;
        showNotification('Text columns are unavailable for rerun.', 'warning');
        return;
    }

    textModalColumns = columnCandidates;

    initializeTextModal();
    attachTextModalLifecycleHandlers();
    populateTextModal(analysisType, columnCandidates, previousSelection);

    const modalElement = document.getElementById('textAnalysisModal');
    if (!modalElement) {
        textModalIsRerun = false;
        showNotification('Text configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    if (previousSelection && previousSelection.length > 0) {
        showNotification('Review or adjust your previously selected text columns, then rerun.', 'info');
    } else {
        showNotification(`Select text columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
    }
}


let currentNetworkCellId = '';
let currentNetworkAnalysisType = '';
let networkModalConfirmed = false;
let networkModalIsRerun = false;
let networkModalColumns = [];
let networkModalSelection = new Set();
let networkModalSearchTerm = '';

let currentEntityNetworkCellId = '';
let currentEntityNetworkAnalysisType = '';
let entityNetworkModalConfirmed = false;
let entityNetworkModalIsRerun = false;
let entityNetworkColumns = [];
let entityNetworkSelection = new Set();
let entityNetworkSearchTerm = '';

async function prepareNetworkAnalysisConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    currentNetworkAnalysisType = analysisType;
    currentNetworkCellId = '';
    networkModalConfirmed = false;
    networkModalIsRerun = false;
    networkModalColumns = [];
    networkModalSelection = new Set();
    networkModalSearchTerm = '';

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before network modal:', error);
        }
    }

    const numericCandidates = getNumericColumnCandidates();
    if (!Array.isArray(numericCandidates) || numericCandidates.length < 2) {
        showNotification('Need at least two numeric columns to configure the correlation network. Running default analysis instead.', 'warning');
        await addSingleAnalysisCell(analysisType);
        return;
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipNetworkModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare network analysis cell. Please try again.', 'error');
        return;
    }

    currentNetworkCellId = cellId;
    networkModalColumns = [...numericCandidates];

    initializeNetworkModal();
    attachNetworkModalLifecycleHandlers();
    populateNetworkModal(analysisType, numericCandidates);

    const modalElement = document.getElementById('networkAnalysisModal');
    if (!modalElement) {
        showNotification('Network configuration modal is unavailable.', 'error');
        return;
    }

    bootstrap.Modal.getOrCreateInstance(modalElement).show();
    showNotification(`Select numeric columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
}

function attachNetworkModalLifecycleHandlers() {
    const modalElement = document.getElementById('networkAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleNetworkModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function initializeNetworkModal() {
    const modalElement = document.getElementById('networkAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const confirmBtn = document.getElementById('networkModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchNetworkAnalysis);
    }

    const selectAllBtn = document.getElementById('networkModalSelectAllBtn');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', () => {
            networkModalSelection = new Set(networkModalColumns.map(col => col.name));
            renderNetworkColumnList();
            updateNetworkSelectionSummary();
        });
    }

    const clearBtn = document.getElementById('networkModalClearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            networkModalSelection = new Set();
            renderNetworkColumnList();
            updateNetworkSelectionSummary();
        });
    }

    const searchInput = document.getElementById('networkColumnSearch');
    if (searchInput) {
        searchInput.addEventListener('input', event => {
            networkModalSearchTerm = (event.target.value || '').toLowerCase();
            renderNetworkColumnList();
        });
    }

    const columnList = document.getElementById('networkColumnList');
    if (columnList) {
        columnList.addEventListener('change', handleNetworkListChange);
    }

    modalElement.dataset.initialized = 'true';
}

function populateNetworkModal(analysisType, columns, initialSelection = null) {
    const modalLabel = document.getElementById('networkAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.textContent = `Configure ${getAnalysisTypeName(analysisType)}`;
    }

    const subtitle = document.getElementById('networkModalSubtitle');
    if (subtitle) {
        subtitle.textContent = 'Pick numeric measures to include in the correlation network graph.';
    }

    const confirmBtn = document.getElementById('networkModalConfirmBtn');
    if (confirmBtn) {
        const baseLabel = `Run ${getAnalysisTypeName(analysisType)}`;
        confirmBtn.dataset.baseLabel = baseLabel;
        confirmBtn.textContent = baseLabel;
        confirmBtn.disabled = true;
    }

    networkModalColumns = Array.isArray(columns) ? [...columns] : [];

    const normalizedSelection = Array.isArray(initialSelection) || initialSelection instanceof Set
        ? Array.from(new Set(initialSelection instanceof Set ? [...initialSelection] : initialSelection))
            .filter(name => networkModalColumns.some(col => col.name === name))
        : [];

    if (normalizedSelection.length >= 2) {
        networkModalSelection = new Set(normalizedSelection);
    } else {
        const defaults = networkModalColumns.slice(0, Math.min(4, networkModalColumns.length)).map(col => col.name);
        networkModalSelection = new Set(defaults);
    }

    renderNetworkColumnList();
    updateNetworkSelectionSummary();
}

function renderNetworkColumnList() {
    const listElement = document.getElementById('networkColumnList');
    if (!listElement) {
        return;
    }

    let filtered = [...networkModalColumns];
    if (networkModalSearchTerm) {
        filtered = filtered.filter(col => col.name.toLowerCase().includes(networkModalSearchTerm));
    }

    if (!filtered.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No numeric columns match your search.</div>';
        return;
    }

    const rows = filtered.map(col => {
        const checked = networkModalSelection.has(col.name) ? 'checked' : '';
        const detailParts = [];
        if (typeof col.uniqueCount === 'number') {
            detailParts.push(`${col.uniqueCount} unique`);
        }
        if (typeof col.minValue === 'number' && typeof col.maxValue === 'number') {
            detailParts.push(`range ${col.minValue} – ${col.maxValue}`);
        }
        if (typeof col.missingPct === 'number') {
            detailParts.push(`${col.missingPct.toFixed(1)}% missing`);
        }
        const detailText = detailParts.length ? `<small class="text-muted">${detailParts.join(' • ')}</small>` : '';
        const badgeLabel = col.dataCategory || col.dataType || 'numeric';

        return `
            <label class="list-group-item d-flex align-items-start gap-3">
                <input class="form-check-input mt-1" type="checkbox" value="${escapeHtml(col.name)}" data-column="${escapeHtml(col.name)}" ${checked}>
                <div class="flex-grow-1">
                    <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                        <strong>${escapeHtml(col.name)}</strong>
                        <span class="badge text-bg-light text-capitalize">${escapeHtml(badgeLabel)}</span>
                    </div>
                    ${detailText}
                </div>
            </label>
        `;
    }).join('');

    listElement.innerHTML = rows;
}

function updateNetworkSelectionSummary() {
    const summaryElement = document.getElementById('networkSelectionSummary');
    const confirmBtn = document.getElementById('networkModalConfirmBtn');
    const count = networkModalSelection.size;

    if (summaryElement) {
        if (count < 2) {
            summaryElement.classList.remove('alert-success');
            summaryElement.classList.add('alert-secondary');
            summaryElement.textContent = 'Select at least two numeric columns to build the network.';
        } else {
            summaryElement.classList.remove('alert-secondary');
            summaryElement.classList.add('alert-success');
            const preview = Array.from(networkModalSelection).slice(0, 5);
            const remainder = count - preview.length;
            const suffix = remainder > 0 ? `, +${remainder} more` : '';
            summaryElement.textContent = `${count} column${count === 1 ? '' : 's'} selected: ${preview.join(', ')}${suffix}`;
        }
    }

    if (confirmBtn) {
        const baseLabel = confirmBtn.dataset.baseLabel || confirmBtn.textContent || 'Run analysis';
        confirmBtn.textContent = count > 0 ? `${baseLabel} (${count})` : baseLabel;
        confirmBtn.disabled = count < 2;
    }
}

function handleNetworkListChange(event) {
    const target = event.target;
    if (!target || target.type !== 'checkbox') {
        return;
    }

    const columnName = target.value;
    if (!columnName) {
        return;
    }

    if (target.checked) {
        networkModalSelection.add(columnName);
    } else {
        networkModalSelection.delete(columnName);
    }

    updateNetworkSelectionSummary();
}

async function launchNetworkAnalysis() {
    const selection = Array.from(networkModalSelection);
    if (selection.length < 2) {
        showNotification('Pick at least two numeric columns to form relationships.', 'warning');
        return;
    }

    const confirmBtn = document.getElementById('networkModalConfirmBtn');
    let previousLabel = null;
    if (confirmBtn) {
        previousLabel = confirmBtn.innerHTML;
        confirmBtn.disabled = true;
        confirmBtn.setAttribute('aria-busy', 'true');
        confirmBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Preparing…
        `;
    }

    networkModalConfirmed = true;
    const modalElement = document.getElementById('networkAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    modalInstance?.hide();

    try {
        if (!currentNetworkCellId) {
            const fallbackCellId = await addSingleAnalysisCell(currentNetworkAnalysisType, {
                skipNetworkModal: true
            });
            currentNetworkCellId = fallbackCellId || '';
        }

        if (!currentNetworkCellId) {
            showNotification('Unable to start analysis: no available cell.', 'error');
            return;
        }

        showNotification(`Running ${getAnalysisTypeName(currentNetworkAnalysisType)} for ${selection.length} numeric columns.`, 'success');

        await generateAndRunAnalysis(
            currentNetworkCellId,
            currentNetworkAnalysisType,
            {},
            {
                overrideSelectedColumns: selection,
                includeGlobalSelectedColumns: false,
                modalType: 'network'
            }
        );
    } catch (error) {
        console.error('Network analysis run failed:', error);
        showNotification(`Analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.removeAttribute('aria-busy');
            if (previousLabel !== null) {
                confirmBtn.innerHTML = previousLabel;
            }
        }
        resetNetworkModalState();
    }
}

function handleNetworkModalHidden() {
    if (networkModalConfirmed) {
        networkModalConfirmed = false;
        networkModalIsRerun = false;
        return;
    }

    if (networkModalIsRerun) {
        networkModalIsRerun = false;
        resetNetworkModalState();
        showNotification('Correlation network rerun cancelled.', 'info');
        return;
    }

    if (currentNetworkCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentNetworkCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        if (currentNetworkAnalysisType) {
            showNotification('Correlation network analysis cancelled.', 'info');
        }
    }

    resetNetworkModalState();
}

function resetNetworkModalState() {
    currentNetworkCellId = '';
    currentNetworkAnalysisType = '';
    networkModalConfirmed = false;
    networkModalIsRerun = false;
    networkModalColumns = [];
    networkModalSelection = new Set();
    networkModalSearchTerm = '';
}

async function openNetworkModalForRerun(cellId, analysisType, previousSelection = []) {
    networkModalIsRerun = true;
    currentNetworkCellId = cellId;
    currentNetworkAnalysisType = analysisType;
    networkModalConfirmed = false;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before rerun:', error);
        }
    }

    const numericCandidates = getNumericColumnCandidates();
    if (!Array.isArray(numericCandidates) || numericCandidates.length < 2) {
        showNotification('Not enough numeric columns remain to configure the network. Running default analysis.', 'warning');
        resetNetworkModalState();
        await addSingleAnalysisCell(analysisType);
        return;
    }

    networkModalColumns = [...numericCandidates];
    initializeNetworkModal();
    attachNetworkModalLifecycleHandlers();
    populateNetworkModal(analysisType, numericCandidates, previousSelection);

    const modalElement = document.getElementById('networkAnalysisModal');
    if (!modalElement) {
        showNotification('Network configuration modal is unavailable.', 'error');
        return;
    }

    bootstrap.Modal.getOrCreateInstance(modalElement).show();
}

async function prepareEntityNetworkAnalysisConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    currentEntityNetworkAnalysisType = analysisType;
    currentEntityNetworkCellId = '';
    entityNetworkModalConfirmed = false;
    entityNetworkModalIsRerun = false;
    entityNetworkColumns = [];
    entityNetworkSelection = new Set();
    entityNetworkSearchTerm = '';

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before entity network modal:', error);
        }
    }

    const categoricalCandidates = getCategoricalColumnCandidates();
    if (!Array.isArray(categoricalCandidates) || categoricalCandidates.length < 2) {
        showNotification('Need at least two categorical columns to configure the entity network. Running default analysis instead.', 'warning');
        await addSingleAnalysisCell(analysisType);
        return;
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipEntityNetworkModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare entity network cell. Please try again.', 'error');
        return;
    }

    currentEntityNetworkCellId = cellId;
    entityNetworkColumns = [...categoricalCandidates];

    initializeEntityNetworkModal();
    attachEntityNetworkModalLifecycleHandlers();
    populateEntityNetworkModal(analysisType, categoricalCandidates);

    const modalElement = document.getElementById('entityNetworkAnalysisModal');
    if (!modalElement) {
        showNotification('Entity network configuration modal is unavailable.', 'error');
        return;
    }

    bootstrap.Modal.getOrCreateInstance(modalElement).show();
    showNotification(`Select categorical columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
}

function attachEntityNetworkModalLifecycleHandlers() {
    const modalElement = document.getElementById('entityNetworkAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleEntityNetworkModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function initializeEntityNetworkModal() {
    const modalElement = document.getElementById('entityNetworkAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const confirmBtn = document.getElementById('entityNetworkModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchEntityNetworkAnalysis);
    }

    const selectAllBtn = document.getElementById('entityNetworkModalSelectAllBtn');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', () => {
            entityNetworkSelection = new Set(entityNetworkColumns.map(col => col.name));
            renderEntityNetworkColumnList();
            updateEntityNetworkSelectionSummary();
        });
    }

    const clearBtn = document.getElementById('entityNetworkModalClearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            entityNetworkSelection = new Set();
            renderEntityNetworkColumnList();
            updateEntityNetworkSelectionSummary();
        });
    }

    const searchInput = document.getElementById('entityNetworkColumnSearch');
    if (searchInput) {
        searchInput.addEventListener('input', event => {
            entityNetworkSearchTerm = (event.target.value || '').toLowerCase();
            renderEntityNetworkColumnList();
        });
    }

    const columnList = document.getElementById('entityNetworkColumnList');
    if (columnList) {
        columnList.addEventListener('change', handleEntityNetworkListChange);
    }

    modalElement.dataset.initialized = 'true';
}

function populateEntityNetworkModal(analysisType, columns, initialSelection = null) {
    const modalLabel = document.getElementById('entityNetworkAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.textContent = `Configure ${getAnalysisTypeName(analysisType)}`;
    }

    const subtitle = document.getElementById('entityNetworkModalSubtitle');
    if (subtitle) {
        subtitle.textContent = 'Pick categorical columns to map their co-occurrence as an entity relationship network. Choose two for the strongest focus.';
    }

    const confirmBtn = document.getElementById('entityNetworkModalConfirmBtn');
    if (confirmBtn) {
        const baseLabel = `Run ${getAnalysisTypeName(analysisType)}`;
        confirmBtn.dataset.baseLabel = baseLabel;
        confirmBtn.textContent = baseLabel;
        confirmBtn.disabled = true;
    }

    entityNetworkColumns = Array.isArray(columns) ? [...columns] : [];

    const normalizedSelection = Array.isArray(initialSelection) || initialSelection instanceof Set
        ? Array.from(new Set(initialSelection instanceof Set ? [...initialSelection] : initialSelection))
            .filter(name => entityNetworkColumns.some(col => col.name === name))
        : [];

    if (normalizedSelection.length >= 2) {
        entityNetworkSelection = new Set(normalizedSelection);
    } else {
        const defaults = entityNetworkColumns.slice(0, Math.min(2, entityNetworkColumns.length)).map(col => col.name);
        entityNetworkSelection = new Set(defaults);
    }

    renderEntityNetworkColumnList();
    updateEntityNetworkSelectionSummary();
}

function renderEntityNetworkColumnList() {
    const listElement = document.getElementById('entityNetworkColumnList');
    if (!listElement) {
        return;
    }

    let filtered = [...entityNetworkColumns];
    if (entityNetworkSearchTerm) {
        filtered = filtered.filter(col => col.name.toLowerCase().includes(entityNetworkSearchTerm));
    }

    if (!filtered.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No categorical columns match your search.</div>';
        return;
    }

    const rows = filtered.map(col => {
        const checked = entityNetworkSelection.has(col.name) ? 'checked' : '';
        const detailParts = [];
        if (typeof col.uniqueCount === 'number') {
            detailParts.push(`${col.uniqueCount} unique`);
        }
        if (typeof col.missingPct === 'number') {
            detailParts.push(`${col.missingPct.toFixed(1)}% missing`);
        }
        if (col.reason) {
            detailParts.push(col.reason);
        }
        const detailText = detailParts.length ? `<small class="text-muted">${detailParts.join(' • ')}</small>` : '';
        const badgeLabel = col.dataCategory || col.dataType || 'categorical';

        return `
            <label class="list-group-item d-flex align-items-start gap-3">
                <input class="form-check-input mt-1" type="checkbox" value="${escapeHtml(col.name)}" data-column="${escapeHtml(col.name)}" ${checked}>
                <div class="flex-grow-1">
                    <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                        <strong>${escapeHtml(col.name)}</strong>
                        <span class="badge text-bg-light text-capitalize">${escapeHtml(badgeLabel)}</span>
                    </div>
                    ${detailText}
                </div>
            </label>
        `;
    }).join('');

    listElement.innerHTML = rows;
}

function updateEntityNetworkSelectionSummary() {
    const summaryElement = document.getElementById('entityNetworkSelectionSummary');
    const confirmBtn = document.getElementById('entityNetworkModalConfirmBtn');
    const count = entityNetworkSelection.size;

    if (summaryElement) {
        if (count < 2) {
            summaryElement.classList.remove('alert-success');
            summaryElement.classList.add('alert-secondary');
            summaryElement.textContent = 'Select at least two categorical columns to map entity relationships.';
        } else {
            summaryElement.classList.remove('alert-secondary');
            summaryElement.classList.add('alert-success');
            const preview = Array.from(entityNetworkSelection).slice(0, 4);
            const remainder = count - preview.length;
            const suffix = remainder > 0 ? `, +${remainder} more` : '';
            summaryElement.textContent = `${count} column${count === 1 ? '' : 's'} selected: ${preview.join(', ')}${suffix}`;
        }
    }

    if (confirmBtn) {
        const baseLabel = confirmBtn.dataset.baseLabel || confirmBtn.textContent || 'Run analysis';
        confirmBtn.textContent = count > 0 ? `${baseLabel} (${count})` : baseLabel;
        confirmBtn.disabled = count < 2;
    }
}

function handleEntityNetworkListChange(event) {
    const target = event.target;
    if (!target || target.type !== 'checkbox') {
        return;
    }

    const columnName = target.value;
    if (!columnName) {
        return;
    }

    if (target.checked) {
        entityNetworkSelection.add(columnName);
    } else {
        entityNetworkSelection.delete(columnName);
    }

    updateEntityNetworkSelectionSummary();
}

async function launchEntityNetworkAnalysis() {
    const selection = Array.from(entityNetworkSelection);
    if (selection.length < 2) {
        showNotification('Pick at least two categorical columns to build the entity network.', 'warning');
        return;
    }

    const confirmBtn = document.getElementById('entityNetworkModalConfirmBtn');
    let previousLabel = null;
    if (confirmBtn) {
        previousLabel = confirmBtn.innerHTML;
        confirmBtn.disabled = true;
        confirmBtn.setAttribute('aria-busy', 'true');
        confirmBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Preparing…
        `;
    }

    entityNetworkModalConfirmed = true;
    const modalElement = document.getElementById('entityNetworkAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    modalInstance?.hide();

    try {
        if (!currentEntityNetworkCellId) {
            const fallbackCellId = await addSingleAnalysisCell(currentEntityNetworkAnalysisType, {
                skipEntityNetworkModal: true
            });
            currentEntityNetworkCellId = fallbackCellId || '';
        }

        if (!currentEntityNetworkCellId) {
            showNotification('Unable to start analysis: no available cell.', 'error');
            return;
        }

        const previewColumns = selection.slice(0, 2).join(' & ');
        showNotification(`Running ${getAnalysisTypeName(currentEntityNetworkAnalysisType)} for ${previewColumns}.`, 'success');

        await generateAndRunAnalysis(
            currentEntityNetworkCellId,
            currentEntityNetworkAnalysisType,
            {},
            {
                overrideSelectedColumns: selection,
                includeGlobalSelectedColumns: false,
                modalType: 'entity-network'
            }
        );
    } catch (error) {
        console.error('Entity network analysis run failed:', error);
        showNotification(`Analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.removeAttribute('aria-busy');
            if (previousLabel !== null) {
                confirmBtn.innerHTML = previousLabel;
            }
        }
        resetEntityNetworkModalState();
    }
}

function handleEntityNetworkModalHidden() {
    if (entityNetworkModalConfirmed) {
        entityNetworkModalConfirmed = false;
        entityNetworkModalIsRerun = false;
        return;
    }

    if (entityNetworkModalIsRerun) {
        entityNetworkModalIsRerun = false;
        resetEntityNetworkModalState();
        showNotification('Entity network rerun cancelled.', 'info');
        return;
    }

    if (currentEntityNetworkCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentEntityNetworkCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        if (currentEntityNetworkAnalysisType) {
            showNotification('Entity relationship network analysis cancelled.', 'info');
        }
    }

    resetEntityNetworkModalState();
}

function resetEntityNetworkModalState() {
    currentEntityNetworkCellId = '';
    currentEntityNetworkAnalysisType = '';
    entityNetworkModalConfirmed = false;
    entityNetworkModalIsRerun = false;
    entityNetworkColumns = [];
    entityNetworkSelection = new Set();
    entityNetworkSearchTerm = '';
}

async function openEntityNetworkModalForRerun(cellId, analysisType, previousSelection = []) {
    entityNetworkModalIsRerun = true;
    currentEntityNetworkCellId = cellId;
    currentEntityNetworkAnalysisType = analysisType;
    entityNetworkModalConfirmed = false;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before entity rerun:', error);
        }
    }

    const categoricalCandidates = getCategoricalColumnCandidates();
    if (!Array.isArray(categoricalCandidates) || categoricalCandidates.length < 2) {
        showNotification('Not enough categorical columns remain to configure the entity network. Running default analysis.', 'warning');
        resetEntityNetworkModalState();
        await addSingleAnalysisCell(analysisType);
        return;
    }

    entityNetworkColumns = [...categoricalCandidates];
    initializeEntityNetworkModal();
    attachEntityNetworkModalLifecycleHandlers();
    populateEntityNetworkModal(analysisType, categoricalCandidates, previousSelection);

    const modalElement = document.getElementById('entityNetworkAnalysisModal');
    if (!modalElement) {
        showNotification('Entity network configuration modal is unavailable.', 'error');
        return;
    }

    bootstrap.Modal.getOrCreateInstance(modalElement).show();
}


const TARGET_MODAL_MAX_SELECTION = 3;
const TARGET_NAME_HINTS = [
    'target',
    'label',
    'outcome',
    'response',
    'class',
    'status',
    'result',
    'churn',
    'default',
    'fraud',
    'success',
    'converted',
    'y'
];

async function prepareTargetAnalysisConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    currentTargetAnalysisType = analysisType;
    currentTargetCellId = '';
    targetModalConfirmed = false;
    targetModalIsRerun = false;
    targetModalSelection = new Set();
    targetModalColumns = [];
    targetModalRecommendedDefaults = [];
    targetModalSearchTerm = '';

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before target modal:', error);
        }
    }

    const columnCandidates = getTargetColumnCandidates();
    if (!columnCandidates.length) {
        showNotification('No suitable target candidates detected. Running full analysis instead.', 'info');
        await addSingleAnalysisCell(analysisType);
        return;
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipTargetModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare target analysis cell. Please try again.', 'error');
        return;
    }

    currentTargetCellId = cellId;
    targetModalColumns = columnCandidates;

    initializeTargetModal();
    attachTargetModalLifecycleHandlers();
    populateTargetModal(analysisType, columnCandidates);

    const modalElement = document.getElementById('targetAnalysisModal');
    if (!modalElement) {
        showNotification('Target configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
    showNotification(`Select target columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
}

function attachTargetModalLifecycleHandlers() {
    const modalElement = document.getElementById('targetAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleTargetModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleTargetModalHidden() {
    if (targetModalConfirmed) {
        resetTargetModalState();
        return;
    }

    if (targetModalIsRerun) {
        targetModalIsRerun = false;
        currentTargetCellId = '';
        currentTargetAnalysisType = '';
        targetModalSelection = new Set();
        targetModalColumns = [];
        targetModalRecommendedDefaults = [];
        targetModalSearchTerm = '';
        showNotification('Target analysis rerun cancelled.', 'info');
        return;
    }

    if (currentTargetCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentTargetCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        if (currentTargetAnalysisType) {
            showNotification('Target analysis cancelled.', 'info');
        }
    }

    resetTargetModalState();
}

function resetTargetModalState() {
    targetModalConfirmed = false;
    targetModalIsRerun = false;
    currentTargetCellId = '';
    currentTargetAnalysisType = '';
    targetModalSelection = new Set();
    targetModalColumns = [];
    targetModalRecommendedDefaults = [];
    targetModalSearchTerm = '';
}

function initializeTargetModal() {
    const modalElement = document.getElementById('targetAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const confirmBtn = document.getElementById('targetModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchTargetAnalysis);
    }

    const recommendBtn = document.getElementById('targetModalRecommendBtn');
    if (recommendBtn) {
        recommendBtn.addEventListener('click', () => {
            applyTargetRecommendations();
        });
    }

    const clearBtn = document.getElementById('targetModalClearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            targetModalSelection = new Set();
            renderTargetColumnList();
            updateTargetSelectionSummary();
        });
    }

    const searchInput = document.getElementById('targetColumnSearch');
    if (searchInput) {
        searchInput.addEventListener('input', event => {
            targetModalSearchTerm = (event.target.value || '').toLowerCase();
            renderTargetColumnList();
        });
    }

    const columnList = document.getElementById('targetColumnList');
    if (columnList) {
        columnList.addEventListener('change', handleTargetListChange);
    }

    modalElement.dataset.initialized = 'true';
}

function toFiniteNumber(value) {
    if (value === null || value === undefined) {
        return null;
    }
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}

function getTargetColumnCandidates() {
    if (!Array.isArray(columnInsightsData?.column_insights)) {
        return [];
    }

    const candidates = [];

    columnInsightsData.column_insights.forEach(col => {
        if (!col || col.dropped) {
            return;
        }

        const name = (col.name ?? '').toString().trim();
        if (!name) {
            return;
        }

        const stats = typeof col.statistics === 'object' && col.statistics !== null ? col.statistics : {};
        const rowCount = toFiniteNumber(stats.count ?? stats.row_count ?? stats.total_count) ?? toFiniteNumber(col.row_count);
        const uniqueCount = toFiniteNumber(stats.unique_count ?? stats.distinct_count ?? stats.cardinality);
        const uniqueRatioRaw = toFiniteNumber(stats.unique_ratio ?? stats.distinct_ratio);
        const uniqueRatio = uniqueRatioRaw !== null && uniqueRatioRaw > 1 ? uniqueRatioRaw / 100 : uniqueRatioRaw;
        const missingPctRaw = toFiniteNumber(col.null_percentage ?? stats.missing_ratio);
        const missingPct = missingPctRaw !== null && missingPctRaw <= 1 ? missingPctRaw * 100 : missingPctRaw;
        const dataCategory = (col.data_category || '').toString().toLowerCase();
        const dataType = (col.data_type || '').toString().toLowerCase();

        if (uniqueCount !== null && uniqueCount <= 1) {
            return;
        }

        const normalizedName = name.toLowerCase();
        const hintScore = TARGET_NAME_HINTS.some(hint => normalizedName.includes(hint)) ? 3 : 0;

        const isNumeric = ['int', 'int64', 'float', 'float64', 'numeric', 'number'].some(type => dataType.includes(type)) || dataCategory === 'numeric';
        const isCategorical = ['categorical', 'category', 'object', 'bool', 'boolean', 'text'].some(type => dataCategory.includes(type) || dataType.includes(type));

        let role = 'regression';
        let score = hintScore;
        const uniqueRatioValue = uniqueRatio !== null ? uniqueRatio : (rowCount && uniqueCount !== null ? uniqueCount / rowCount : null);
    const hasLowCardinality = uniqueCount !== null && uniqueCount <= 20;
    const hasBinaryStyleRatio = uniqueRatioValue !== null && uniqueRatioValue <= 0.3;
    const qualifiesByRatio = hasBinaryStyleRatio && (uniqueCount === null || uniqueCount <= 50);

        if (isCategorical || hasLowCardinality || qualifiesByRatio) {
            role = 'classification';
            score += 2;
            if (uniqueCount !== null && (uniqueCount === 2 || uniqueCount === 3)) {
                score += 1;
            }
            if (qualifiesByRatio) {
                score += 1;
            }
        } else if (isNumeric) {
            role = 'regression';
            score += 1.5;
            if (uniqueRatioValue !== null && uniqueRatioValue >= 0.4) {
                score += 0.5;
            }
        }

        if (uniqueRatioValue !== null && uniqueRatioValue > 0.95 && uniqueCount !== null && uniqueCount > 20) {
            score -= 2.5;
        }

        if (missingPct !== null && missingPct > 40) {
            score -= 1;
        }

        const reasonParts = [];
        if (uniqueCount !== null) {
            reasonParts.push(`${uniqueCount} ${uniqueCount === 1 ? 'unique value' : 'unique values'}`);
        }
        if (uniqueRatioValue !== null) {
            const ratioPct = uniqueRatioValue <= 1 ? uniqueRatioValue * 100 : uniqueRatioValue;
            reasonParts.push(`${ratioPct.toFixed(ratioPct < 5 ? 1 : 0)}% unique`);
        }
        if (missingPct !== null) {
            reasonParts.push(`${missingPct.toFixed(1)}% missing`);
        }

        candidates.push({
            name,
            dataCategory,
            dataType,
            uniqueCount: uniqueCount !== null ? uniqueCount : undefined,
            uniqueRatio: uniqueRatioValue !== null ? uniqueRatioValue : undefined,
            missingPct: missingPct !== null ? missingPct : undefined,
            role,
            score,
            reason: reasonParts.join(' • '),
            recommended: score >= 3 || hintScore > 0
        });
    });

    candidates.sort((a, b) => {
        if (a.recommended !== b.recommended) {
            return a.recommended ? -1 : 1;
        }
        if (b.score !== a.score) {
            return b.score - a.score;
        }
        return a.name.localeCompare(b.name);
    });

    return candidates;
}

function populateTargetModal(analysisType, columns, initialSelection = null, selectionDetails = null) {
    const modalLabel = document.getElementById('targetAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.textContent = `Configure ${getAnalysisTypeName(analysisType)}`;
    }

    const subtitle = document.getElementById('targetModalSubtitle');
    if (subtitle) {
        subtitle.textContent = 'Choose target columns to profile class balance or value spread before modelling.';
    }

    const confirmBtn = document.getElementById('targetModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.dataset.baseLabel = `Run ${getAnalysisTypeName(analysisType)}`;
        confirmBtn.textContent = confirmBtn.dataset.baseLabel;
        confirmBtn.disabled = false;
    }

    targetModalColumns = Array.isArray(columns) ? [...columns] : [];
    targetModalRecommendedDefaults = targetModalColumns.filter(col => col.recommended).slice(0, TARGET_MODAL_MAX_SELECTION).map(col => col.name);

    const normalizedInitialSelection = Array.isArray(initialSelection) || initialSelection instanceof Set
        ? Array.from(new Set(initialSelection instanceof Set ? [...initialSelection] : initialSelection)).filter(name =>
            targetModalColumns.some(col => col.name === name)
        )
        : [];

    if (normalizedInitialSelection.length > 0) {
        targetModalSelection = new Set(normalizedInitialSelection);
    } else if (targetModalRecommendedDefaults.length > 0) {
        targetModalSelection = new Set(targetModalRecommendedDefaults);
    } else {
        const fallbackDefaults = targetModalColumns.slice(0, TARGET_MODAL_MAX_SELECTION).map(col => col.name);
        targetModalSelection = new Set(fallbackDefaults);
    }

    if (selectionDetails && Array.isArray(selectionDetails)) {
        selectionDetails.forEach(detail => {
            const columnName = detail?.column;
            const role = detail?.role;
            if (!columnName) {
                return;
            }
            const candidate = targetModalColumns.find(col => col.name === columnName);
            if (candidate && role && (role === 'classification' || role === 'regression')) {
                candidate.role = role;
            }
        });
    }

    renderTargetRecommendations();
    renderTargetColumnList();
    updateTargetSelectionSummary();
}

function renderTargetRecommendations() {
    const container = document.getElementById('targetRecommendationChips');
    if (!container) {
        return;
    }

    container.innerHTML = '';

    if (!targetModalRecommendedDefaults.length) {
        container.innerHTML = '<span class="text-muted small">No automatic recommendations yet. Select columns manually below.</span>';
        return;
    }

    targetModalRecommendedDefaults.forEach(columnName => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'btn btn-outline-primary btn-sm me-2 mb-2';
        chip.dataset.column = columnName;
        chip.textContent = columnName;
        chip.addEventListener('click', () => {
            if (!targetModalSelection.has(columnName)) {
                if (targetModalSelection.size >= TARGET_MODAL_MAX_SELECTION) {
                    showNotification(`Select up to ${TARGET_MODAL_MAX_SELECTION} target columns.`, 'warning');
                    return;
                }
                targetModalSelection.add(columnName);
                renderTargetColumnList();
                updateTargetSelectionSummary();
            }
        });
        container.appendChild(chip);
    });
}

function renderTargetColumnList() {
    const listElement = document.getElementById('targetColumnList');
    if (!listElement) {
        return;
    }

    let filtered = [...targetModalColumns];
    if (targetModalSearchTerm) {
        filtered = filtered.filter(col => col.name.toLowerCase().includes(targetModalSearchTerm));
    }

    if (!filtered.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No columns match your search.</div>';
        return;
    }

    const rows = filtered.map(col => {
        const checked = targetModalSelection.has(col.name) ? 'checked' : '';
        const typeBadge = col.dataCategory
            ? `<span class="badge text-bg-light text-capitalize">${escapeHtml(col.dataCategory || col.dataType || 'column')}</span>`
            : '';
        const detailParts = [];
        if (typeof col.uniqueCount === 'number') {
            detailParts.push(`${col.uniqueCount} ${col.role === 'classification' ? 'classes' : 'unique values'}`);
        }
        if (typeof col.uniqueRatio === 'number') {
            const ratioPct = col.uniqueRatio <= 1 ? col.uniqueRatio * 100 : col.uniqueRatio;
            detailParts.push(`${ratioPct.toFixed(ratioPct < 5 ? 1 : 0)}% unique`);
        }
        if (typeof col.missingPct === 'number') {
            detailParts.push(`${col.missingPct.toFixed(1)}% missing`);
        }
        const reasonText = col.reason ? `<div class="text-muted small">${escapeHtml(col.reason)}</div>` : '';

        return `
            <label class="list-group-item d-flex align-items-start gap-3 target-column-item">
                <input class="form-check-input mt-1" type="checkbox" value="${escapeHtml(col.name)}" data-column="${escapeHtml(col.name)}" ${checked}>
                <div class="flex-grow-1">
                    <div class="d-flex align-items-center gap-2 flex-wrap">
                        <strong>${escapeHtml(col.name)}</strong>
                        ${typeBadge}
                    </div>
                    <div class="text-muted small">${detailParts.join(' • ') || 'Profiling metrics not available yet.'}</div>
                    ${reasonText}
                </div>
            </label>
        `;
    }).join('');

    listElement.innerHTML = rows;
}

function handleTargetListChange(event) {
    if (!event || event.target.type !== 'checkbox') {
        return;
    }

    const columnName = event.target.value;
    if (!columnName) {
        return;
    }

    if (event.target.checked) {
        if (targetModalSelection.size >= TARGET_MODAL_MAX_SELECTION) {
            event.target.checked = false;
            showNotification(`Select up to ${TARGET_MODAL_MAX_SELECTION} target columns.`, 'warning');
            return;
        }
        targetModalSelection.add(columnName);
    } else {
        targetModalSelection.delete(columnName);
    }

    updateTargetSelectionSummary();
}

function applyTargetRecommendations() {
    if (!targetModalRecommendedDefaults.length) {
        showNotification('No recommended target columns at the moment.', 'info');
        return;
    }

    targetModalSelection = new Set(targetModalRecommendedDefaults.slice(0, TARGET_MODAL_MAX_SELECTION));
    renderTargetColumnList();
    updateTargetSelectionSummary();
}

function updateTargetSelectionSummary() {
    const summaryElement = document.getElementById('targetSelectionSummary');
    const confirmBtn = document.getElementById('targetModalConfirmBtn');
    const count = targetModalSelection.size;

    if (summaryElement) {
        if (count === 0) {
            summaryElement.textContent = `Select at least one target column (up to ${TARGET_MODAL_MAX_SELECTION}).`;
        } else {
            const preview = Array.from(targetModalSelection).slice(0, TARGET_MODAL_MAX_SELECTION);
            const remainder = count - preview.length;
            const suffix = remainder > 0 ? `, +${remainder} more` : '';
            summaryElement.textContent = `${count} target column${count === 1 ? '' : 's'} selected: ${preview.join(', ')}${suffix}`;
        }
    }

    if (confirmBtn) {
        const baseLabel = confirmBtn.dataset.baseLabel || confirmBtn.textContent || 'Run analysis';
        confirmBtn.textContent = count > 0 ? `${baseLabel} (${count})` : baseLabel;
        confirmBtn.disabled = count === 0;
    }
}

async function launchTargetAnalysis() {
    if (!currentTargetCellId || !currentTargetAnalysisType) {
        showNotification('Target analysis cell is unavailable. Please try again.', 'error');
        return;
    }

    if (targetModalSelection.size === 0) {
        showNotification('Select at least one target column.', 'warning');
        return;
    }

    targetModalConfirmed = true;

    const selection = Array.from(targetModalSelection);
    const selectionDetails = selection.map(columnName => {
        const candidate = targetModalColumns.find(col => col.name === columnName) || {};
        return {
            column: columnName,
            role: candidate.role || 'classification',
            unique_count: candidate.uniqueCount ?? null,
            unique_ratio: candidate.uniqueRatio ?? null,
            missing_pct: candidate.missingPct ?? null,
            score: candidate.score ?? null
        };
    });

    const modalElement = document.getElementById('targetAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    modalInstance?.hide();

    const analysisOptions = {
        overrideSelectedColumns: selection,
        includeGlobalSelectedColumns: false,
        modalType: 'target',
        modalSelectionPayload: {
            columns: selectionDetails
        }
    };

    await generateAndRunAnalysis(currentTargetCellId, currentTargetAnalysisType, {}, analysisOptions);
    resetTargetModalState();
}

async function openTargetModalForRerun(cellId, analysisType, previousSelection = [], previousDetails = null) {
    if (!analysisType) {
        return;
    }

    currentTargetCellId = cellId;
    currentTargetAnalysisType = analysisType;
    targetModalConfirmed = false;
    targetModalIsRerun = true;
    targetModalSelection = new Set();
    targetModalColumns = [];
    targetModalRecommendedDefaults = [];
    targetModalSearchTerm = '';

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before target rerun modal:', error);
        }
    }

    const columnCandidates = getTargetColumnCandidates();
    if (!columnCandidates.length) {
        showNotification('No suitable target candidates detected. Re-running analysis with previous selection.', 'info');
        const fallbackOptions = {
            overrideSelectedColumns: Array.isArray(previousSelection) ? previousSelection : [],
            includeGlobalSelectedColumns: false,
            modalType: 'target',
            modalSelectionPayload: previousDetails && typeof previousDetails === 'object' ? { ...previousDetails } : null
        };
        executeAnalysisRerun(cellId, analysisType, fallbackOptions);
        targetModalIsRerun = false;
        return;
    }

    targetModalColumns = columnCandidates;
    initializeTargetModal();
    attachTargetModalLifecycleHandlers();
    populateTargetModal(analysisType, columnCandidates, previousSelection, previousDetails?.columns || null);

    const modalElement = document.getElementById('targetAnalysisModal');
    if (!modalElement) {
        showNotification('Target configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
    showNotification(`Adjust target selection for ${getAnalysisTypeName(analysisType)}.`, 'info');
}

window.prepareTargetAnalysisConfiguration = prepareTargetAnalysisConfiguration;
window.openTargetModalForRerun = openTargetModalForRerun;
window.launchTargetAnalysis = launchTargetAnalysis;


const MARKETING_ANALYSIS_CONFIG = {
    'campaign_metrics_analysis': {
        title: 'Campaign Metrics Analysis',
        description: 'Analyze marketing campaign performance metrics including impressions, clicks, conversions, CTR, CPC, and ROAS.',
        features: [
            '✓ Core KPI calculations (CTR, CPC, ROAS)',
            '✓ Performance distribution analysis', 
            '✓ Campaign and channel comparison',
            '✓ Data quality assessment',
            '✓ Actionable recommendations'
        ],
        columns: {
            'impressions': {
                label: 'Impressions',
                description: 'Total impressions/views',
                required: false,
                examples: ['impressions', 'views', 'reach', 'imp_count']
            },
            'clicks': {
                label: 'Clicks', 
                description: 'Total clicks received',
                required: false,
                examples: ['clicks', 'click_count', 'link_clicks']
            },
            'conversions': {
                label: 'Conversions',
                description: 'Total conversions/purchases',
                required: false, 
                examples: ['conversions', 'purchases', 'signups', 'goals']
            },
            'spend': {
                label: 'Spend',
                description: 'Amount spent on campaign', 
                required: false,
                examples: ['spend', 'cost', 'budget', 'investment']
            },
            'revenue': {
                label: 'Revenue',
                description: 'Revenue generated',
                required: false,
                examples: ['revenue', 'sales', 'value', 'income']
            },
            'campaign': {
                label: 'Campaign Name',
                description: 'Campaign identifier',
                required: false,
                examples: ['campaign', 'campaign_name', 'ad_name']
            },
            'channel': {
                label: 'Channel/Source',
                description: 'Marketing channel',
                required: false, 
                examples: ['channel', 'source', 'platform', 'medium']
            }
        }
    },
    'conversion_funnel_analysis': {
        title: 'Conversion Funnel Analysis',
        description: 'Analyze step-by-step user journey, identify drop-off rates and bottlenecks in your conversion funnel.',
        features: [
            '✓ Multi-step funnel analysis',
            '✓ Drop-off rate calculations', 
            '✓ Bottleneck identification',
            '✓ Channel-based segmentation',
            '✓ Optimization recommendations'
        ],
        columns: {
            'session_id': {
                label: 'Session ID',
                description: 'Unique session identifier',
                required: false,
                examples: ['session_id', 'user_id', 'visitor_id']
            },
            'landed': {
                label: 'Landing Page',
                description: 'Visited landing page (1/0)',
                required: false,
                examples: ['landed', 'visited', 'entry', 'step1']
            },
            'viewed_product': {
                label: 'Viewed Product', 
                description: 'Viewed product page (1/0)',
                required: false,
                examples: ['viewed_product', 'product_view', 'step2']
            },
            'added_to_cart': {
                label: 'Added to Cart',
                description: 'Added item to cart (1/0)',
                required: false,
                examples: ['added_to_cart', 'cart', 'step3']
            },
            'started_checkout': {
                label: 'Started Checkout',
                description: 'Started checkout process (1/0)', 
                required: false,
                examples: ['started_checkout', 'checkout', 'step4']
            },
            'completed_purchase': {
                label: 'Completed Purchase',
                description: 'Completed transaction (1/0)',
                required: false,
                examples: ['completed_purchase', 'purchase', 'conversion']
            },
            'channel': {
                label: 'Traffic Channel',
                description: 'Source of traffic',
                required: false,
                examples: ['channel', 'source', 'utm_source', 'referrer']
            }
        }
    },
    'engagement_analysis': {
        title: 'Engagement Analysis', 
        description: 'Analyze user interaction patterns, session quality, and engagement metrics.',
        features: [
            '✓ Session quality assessment',
            '✓ Interaction depth analysis',
            '✓ User loyalty segmentation', 
            '✓ Device and source comparison',
            '✓ Engagement optimization tips'
        ],
        columns: {
            'session_duration': {
                label: 'Session Duration',
                description: 'Time spent on site (seconds/minutes)',
                required: false,
                examples: ['session_duration', 'time_spent', 'duration']
            },
            'pages_viewed': {
                label: 'Pages Viewed',
                description: 'Number of pages visited',
                required: false,
                examples: ['pages_viewed', 'page_views', 'pageviews']
            },
            'bounce': {
                label: 'Bounce',
                description: 'Bounce indicator (1/0)',
                required: false,
                examples: ['bounce', 'bounced', 'single_page']
            },
            'interactions': {
                label: 'Interactions',
                description: 'Number of interactions',
                required: false,
                examples: ['interactions', 'clicks', 'events']
            },
            'device': {
                label: 'Device Type',
                description: 'Device category',
                required: false,
                examples: ['device', 'device_type', 'platform']
            },
            'channel': {
                label: 'Traffic Source',
                description: 'Source of traffic',
                required: false,
                examples: ['channel', 'source', 'medium']
            }
        }
    },
    'channel_performance_analysis': {
        title: 'Channel Performance Analysis',
        description: 'Compare marketing channel effectiveness, ROI, and performance metrics across different acquisition sources.',
        features: [
            '✓ Channel ROI comparison',
            '✓ Traffic quality analysis',
            '✓ Cost per acquisition by channel',
            '✓ Performance benchmarking',
            '✓ Channel optimization tips'
        ],
        columns: {
            'channel': {
                label: 'Channel',
                description: 'Marketing channel/source',
                required: false,
                examples: ['channel', 'source', 'utm_source', 'medium']
            },
            'sessions': {
                label: 'Sessions',
                description: 'Number of sessions',
                required: false,
                examples: ['sessions', 'visits', 'traffic']
            },
            'conversions': {
                label: 'Conversions',
                description: 'Number of conversions',
                required: false,
                examples: ['conversions', 'goals', 'purchases']
            },
            'revenue': {
                label: 'Revenue',
                description: 'Revenue generated',
                required: false,
                examples: ['revenue', 'sales', 'value']
            },
            'spend': {
                label: 'Ad Spend',
                description: 'Amount spent on channel',
                required: false,
                examples: ['spend', 'cost', 'ad_spend', 'budget']
            }
        }
    },
    'audience_segmentation_analysis': {
        title: 'Audience Segmentation Analysis',
        description: 'Segment users by demographics, behavior patterns, and engagement levels for targeted marketing.',
        features: [
            '✓ Demographic segmentation',
            '✓ Behavioral clustering',
            '✓ Engagement-based groups',
            '✓ Segment profiling',
            '✓ Targeting recommendations'
        ],
        columns: {
            'age': {
                label: 'Age',
                description: 'User age',
                required: false,
                examples: ['age', 'user_age', 'age_group']
            },
            'gender': {
                label: 'Gender',
                description: 'User gender',
                required: false,
                examples: ['gender', 'sex', 'demographic_gender']
            },
            'location': {
                label: 'Location',
                description: 'Geographic location',
                required: false,
                examples: ['location', 'country', 'city', 'region']
            },
            'purchase_frequency': {
                label: 'Purchase Frequency',
                description: 'How often user purchases',
                required: false,
                examples: ['purchase_frequency', 'order_count', 'frequency']
            },
            'total_spent': {
                label: 'Total Spent',
                description: 'Total amount spent',
                required: false,
                examples: ['total_spent', 'lifetime_value', 'total_revenue']
            }
        }
    },
    'roi_analysis': {
        title: 'ROI Analysis',
        description: 'Calculate return on investment, profitability metrics, and identify the most cost-effective marketing activities.',
        features: [
            '✓ ROI calculations',
            '✓ ROAS analysis',
            '✓ Profit margin assessment',
            '✓ Cost-benefit analysis',
            '✓ Investment recommendations'
        ],
        columns: {
            'revenue': {
                label: 'Revenue',
                description: 'Revenue generated',
                required: false,
                examples: ['revenue', 'sales', 'income', 'earnings']
            },
            'cost': {
                label: 'Cost/Spend',
                description: 'Amount invested',
                required: false,
                examples: ['cost', 'spend', 'investment', 'budget']
            },
            'impressions': {
                label: 'Impressions',
                description: 'Number of impressions',
                required: false,
                examples: ['impressions', 'views', 'reach']
            },
            'clicks': {
                label: 'Clicks',
                description: 'Number of clicks',
                required: false,
                examples: ['clicks', 'click_count', 'link_clicks']
            },
            'conversions': {
                label: 'Conversions',
                description: 'Number of conversions',
                required: false,
                examples: ['conversions', 'sales_count', 'goals']
            }
        }
    },
    'attribution_analysis': {
        title: 'Attribution Analysis',
        description: 'Understand customer journey touchpoints and assign conversion credit across marketing channels.',
        features: [
            '✓ Multi-touch attribution',
            '✓ Channel contribution analysis',
            '✓ Customer journey mapping',
            '✓ Touchpoint effectiveness',
            '✓ Attribution model comparison'
        ],
        columns: {
            'customer_id': {
                label: 'Customer ID',
                description: 'Unique customer identifier',
                required: false,
                examples: ['customer_id', 'user_id', 'client_id']
            },
            'touchpoint': {
                label: 'Touchpoint',
                description: 'Marketing touchpoint/channel',
                required: false,
                examples: ['touchpoint', 'channel', 'source', 'medium']
            },
            'timestamp': {
                label: 'Timestamp',
                description: 'When interaction occurred',
                required: false,
                examples: ['timestamp', 'date', 'interaction_time']
            },
            'conversion': {
                label: 'Conversion',
                description: 'Conversion indicator (1/0)',
                required: false,
                examples: ['conversion', 'converted', 'purchase']
            },
            'revenue': {
                label: 'Revenue',
                description: 'Revenue from conversion',
                required: false,
                examples: ['revenue', 'value', 'purchase_amount']
            }
        }
    },
    'cohort_analysis': {
        title: 'Cohort Analysis',
        description: 'Track user behavior and retention over time by grouping users into cohorts based on shared characteristics.',
        features: [
            '✓ Retention rate tracking',
            '✓ User lifecycle analysis',
            '✓ Cohort performance comparison',
            '✓ Churn prediction',
            '✓ Lifetime value trends'
        ],
        columns: {
            'user_id': {
                label: 'User ID',
                description: 'Unique user identifier',
                required: false,
                examples: ['user_id', 'customer_id', 'account_id']
            },
            'signup_date': {
                label: 'Signup Date',
                description: 'When user first signed up',
                required: false,
                examples: ['signup_date', 'registration_date', 'first_seen']
            },
            'activity_date': {
                label: 'Activity Date',
                description: 'Date of user activity',
                required: false,
                examples: ['activity_date', 'last_seen', 'event_date']
            },
            'revenue': {
                label: 'Revenue',
                description: 'Revenue generated by user',
                required: false,
                examples: ['revenue', 'purchase_amount', 'value']
            },
            'active': {
                label: 'Active Status',
                description: 'Whether user is active (1/0)',
                required: false,
                examples: ['active', 'is_active', 'retained']
            }
        }
    }
};

if (typeof window !== 'undefined') {
    window.MARKETING_ANALYSIS_CONFIG = MARKETING_ANALYSIS_CONFIG;
}

let currentAnalysisType = '';
let currentMarketingCellId = '';  // Store cell ID for modal completion
let availableColumns = [];
let selectedColumnMapping = {};
let marketingModalConfirmed = false;

// Show marketing analysis modal
function showMarketingAnalysisModal(analysisType, dataColumns = []) {
    console.log('showMarketingAnalysisModal called with:', analysisType, dataColumns);
    
    currentAnalysisType = analysisType;
    selectedColumnMapping = {};
    
    // Try to get real columns from column insights data first
    if (columnInsightsData && columnInsightsData.column_insights) {
        availableColumns = columnInsightsData.column_insights.map(col => col.name);
        console.log('Using real column names from columnInsightsData:', availableColumns);
    } else if (dataColumns && dataColumns.length > 0) {
        availableColumns = dataColumns;
        console.log('Using provided dataColumns:', availableColumns);
    } else {
        // Fallback - get from session storage or use mock data as last resort
        const storedColumns = sessionStorage.getItem('datasetColumns');
        if (storedColumns) {
            try {
                availableColumns = JSON.parse(storedColumns);
                console.log('Using stored column names:', availableColumns);
            } catch (e) {
                console.warn('Could not parse stored columns, using mock data');
                availableColumns = [
                    'campaign_name', 'date', 'impressions', 'clicks', 'conversions', 
                    'spend', 'revenue', 'ctr', 'cpc', 'roas', 'channel', 'age_group',
                    'gender', 'location', 'device_type', 'session_duration', 'bounce_rate'
                ];
            }
        } else {
            console.warn('No real columns available, using mock columns for demo');
            availableColumns = [
                'campaign_name', 'date', 'impressions', 'clicks', 'conversions', 
                'spend', 'revenue', 'ctr', 'cpc', 'roas', 'channel', 'age_group',
                'gender', 'location', 'device_type', 'session_duration', 'bounce_rate'
            ];
        }
    }
    
    console.log('Final available columns for modal:', availableColumns);
    
    // Store columns in session storage for other components to use
    if (availableColumns && availableColumns.length > 0) {
        sessionStorage.setItem('datasetColumns', JSON.stringify(availableColumns));
        console.log('Stored columns in sessionStorage for other components');
    }
    
    const config = MARKETING_ANALYSIS_CONFIG[analysisType];
    if (!config) {
        console.error('Unknown marketing analysis type:', analysisType);
        return;
    }
    
    // Update modal content
    document.getElementById('analysisTitle').textContent = config.title;
    document.getElementById('analysisDescription').textContent = config.description;
    
    // Update features list
    const featuresList = document.getElementById('featuresList');
    featuresList.innerHTML = config.features.map(feature => `<li>${feature}</li>`).join('');
    
    // Generate column mapping interface
    generateColumnMappingInterface(config.columns);
    
    // Add example data format
    generateExampleDataFormat(analysisType);
    
    // Show the modal
    attachMarketingModalLifecycleHandlers();
    const modal = new bootstrap.Modal(document.getElementById('marketingAnalysisModal'));
    modal.show();
}

// Generate column mapping interface with data type filtering
function generateColumnMappingInterface(columnConfig) {
    console.log('Generating column mapping interface with config:', columnConfig);
    console.log('Available columns:', availableColumns);
    console.log('Column insights data:', columnInsightsData);
    
    const container = document.getElementById('columnMappingContainer');
    container.innerHTML = '';
    
    // Get column type information if available
    const columnTypes = {};
    if (columnInsightsData && columnInsightsData.column_insights) {
        columnInsightsData.column_insights.forEach(col => {
            columnTypes[col.name] = {
                data_type: col.data_type,
                data_category: col.data_category,
                is_numeric: col.data_category === 'numeric'
            };
        });
        console.log('Column types extracted:', columnTypes);
    }
    
    Object.entries(columnConfig).forEach(([key, config]) => {
        const columnDiv = document.createElement('div');
        columnDiv.className = 'col-md-6 mb-3';
        
        const isRequired = config.required ? '<span class="text-danger">*</span>' : '';
        const exampleText = config.examples.length > 0 ? `Examples: ${config.examples.join(', ')}` : '';
        
        // Filter columns based on expected data type for this metric
        const expectedNumeric = config.label.toLowerCase().includes('count') || 
                              config.label.toLowerCase().includes('rate') ||
                              config.label.toLowerCase().includes('amount') ||
                              config.label.toLowerCase().includes('spend') ||
                              config.label.toLowerCase().includes('revenue') ||
                              config.label.toLowerCase().includes('impressions') ||
                              config.label.toLowerCase().includes('clicks');
        
        // Generate options for dropdown with type filtering
        let optionsHtml = '<option value="">-- Skip this metric --</option>';
        
        // Create groups for better organization
        let numericCols = [];
        let textCols = [];
        let otherCols = [];
        
        availableColumns.forEach(col => {
            const colType = columnTypes[col];
            if (colType) {
                if (colType.is_numeric || colType.data_category === 'numeric') {
                    numericCols.push(col);
                } else if (colType.data_category === 'text' || colType.data_type === 'object') {
                    textCols.push(col);
                } else {
                    otherCols.push(col);
                }
            } else {
                otherCols.push(col);
            }
        });
        
        // Add numeric columns first (preferred for most metrics)
        if (numericCols.length > 0) {
            optionsHtml += '<optgroup label="📊 Numeric Columns">';
            numericCols.forEach(col => {
                const typeInfo = columnTypes[col] ? ` (${columnTypes[col].data_type})` : '';
                optionsHtml += `<option value="${col}">${col}${typeInfo}</option>`;
            });
            optionsHtml += '</optgroup>';
        }
        
        // Add text columns (useful for categories, names, etc.)
        if (textCols.length > 0) {
            optionsHtml += '<optgroup label="📝 Text Columns">';
            textCols.forEach(col => {
                const typeInfo = columnTypes[col] ? ` (${columnTypes[col].data_type})` : '';
                optionsHtml += `<option value="${col}">${col}${typeInfo}</option>`;
            });
            optionsHtml += '</optgroup>';
        }
        
        // Add other columns
        if (otherCols.length > 0) {
            optionsHtml += '<optgroup label="🔧 Other Columns">';
            otherCols.forEach(col => {
                const typeInfo = columnTypes[col] ? ` (${columnTypes[col].data_type})` : '';
                optionsHtml += `<option value="${col}">${col}${typeInfo}</option>`;
            });
            optionsHtml += '</optgroup>';
        }
        
        // Provide guidance based on expected type
        let typeGuidance = '';
        if (expectedNumeric) {
            typeGuidance = '<small class="text-info"><i class="bi bi-info-circle"></i> This metric typically uses numeric columns</small>';
        } else {
            typeGuidance = '<small class="text-muted"><i class="bi bi-tag"></i> This metric can use text or categorical columns</small>';
        }
        
        columnDiv.innerHTML = `
            <div class="form-group">
                <label class="form-label">
                    <strong>${config.label}</strong> ${isRequired}
                    <small class="text-muted d-block">${config.description}</small>
                </label>
                <select class="form-select form-select-sm" id="column_${key}" onchange="updateColumnMapping('${key}', this.value)">
                    ${optionsHtml}
                </select>
                <small class="text-muted d-block">${exampleText}</small>
                ${typeGuidance}
            </div>
        `;
        
        container.appendChild(columnDiv);
    });
    
    console.log('Column mapping interface generated with', Object.keys(columnConfig).length, 'metrics');
}

// Generate example data format based on analysis type
function generateExampleDataFormat(analysisType) {
    console.log('Generating example data format for:', analysisType);
    
    const exampleDataCard = document.getElementById('exampleDataCard');
    if (!exampleDataCard) {
        console.warn('Example data card not found');
        return;
    }
    
    const config = MARKETING_ANALYSIS_CONFIG[analysisType];
    if (!config) return;
    
    // Generate sample data based on analysis type
    let exampleData = [];
    let recommendedColumns = [];
    
    switch(analysisType) {
        case 'campaign_metrics_analysis':
            exampleData = [
                { campaign_name: 'Summer_Sale_2024', impressions: '25,000', clicks: '1,200', conversions: '45', spend: '$500', revenue: '$2,250' },
                { campaign_name: 'Holiday_Promo', impressions: '18,500', clicks: '890', conversions: '32', spend: '$380', revenue: '$1,600' },
                { campaign_name: 'Spring_Launch', impressions: '22,000', clicks: '1,100', conversions: '55', spend: '$450', revenue: '$2,750' }
            ];
            recommendedColumns = Object.keys(config.columns);
            break;
            
        case 'conversion_funnel_analysis':
            exampleData = [
                { user_id: 'U001', landed: '1', viewed_product: '1', added_to_cart: '1', started_checkout: '1', completed_purchase: '1' },
                { user_id: 'U002', landed: '1', viewed_product: '1', added_to_cart: '0', started_checkout: '0', completed_purchase: '0' },
                { user_id: 'U003', landed: '1', viewed_product: '1', added_to_cart: '1', started_checkout: '0', completed_purchase: '0' }
            ];
            recommendedColumns = ['user_id', 'landed', 'viewed_product', 'added_to_cart', 'started_checkout', 'completed_purchase'];
            break;
            
        case 'roi_analysis':
            exampleData = [
                { campaign: 'Social_Media', revenue: '$5,000', spend: '$1,200', impressions: '50,000', conversions: '85' },
                { campaign: 'Google_Ads', revenue: '$8,500', spend: '$2,000', impressions: '75,000', conversions: '120' },
                { campaign: 'Email_Marketing', revenue: '$3,200', spend: '$400', impressions: '25,000', conversions: '45' }
            ];
            recommendedColumns = ['campaign', 'revenue', 'spend', 'impressions', 'conversions'];
            break;
            
        default:
            // Generic example for other analyses
            exampleData = [
                { category: 'Category_A', metric1: '100', metric2: '50', value: '$1,000' },
                { category: 'Category_B', metric1: '150', metric2: '75', value: '$1,500' },
                { category: 'Category_C', metric1: '80', metric2: '40', value: '$800' }
            ];
            recommendedColumns = Object.keys(config.columns).slice(0, 4);
    }
    
    // Generate HTML for the example table
    if (exampleData.length > 0) {
        const tableHeaders = Object.keys(exampleData[0]);
        const tableHTML = `
            <div class="row">
                <div class="col-md-8">
                    <h6 class="text-primary">Sample ${config.title} Data:</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped">
                            <thead class="table-primary">
                                <tr>
                                    ${tableHeaders.map(header => `<th>${header}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${exampleData.map(row => `
                                    <tr>
                                        ${tableHeaders.map(header => `<td>${row[header]}</td>`).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="col-md-4">
                    <h6 class="text-success">Recommended Columns:</h6>
                    <div class="mb-2">
                        ${recommendedColumns.map(col => `<span class="badge bg-info me-1 mb-1">${col}</span>`).join('')}
                    </div>
                    <div class="mt-3">
                        <small class="text-muted">
                            <strong>Tip:</strong> Your column names don't need to match exactly. 
                            Use the mapping above to connect your data columns to the analysis metrics.
                        </small>
                    </div>
                </div>
            </div>
        `;
        
        exampleDataCard.innerHTML = tableHTML;
    }
}

// Update column mapping when user selects a column
function updateColumnMapping(metric, columnName) {
    console.log(`Updating column mapping: ${metric} -> ${columnName}`);
    
    if (columnName) {
        selectedColumnMapping[metric] = columnName;
    } else {
        delete selectedColumnMapping[metric];
    }
    
    console.log('Current column mapping:', selectedColumnMapping);
    updateConfigurationSummary();
}

// Auto-detect columns based on naming patterns
function autoDetectColumns() {
    console.log('Auto-detecting columns...');
    console.log('Available columns:', availableColumns);
    console.log('Current analysis type:', currentAnalysisType);
    
    const config = MARKETING_ANALYSIS_CONFIG[currentAnalysisType];
    if (!config) {
        console.error('No config found for', currentAnalysisType);
        showNotification('Configuration error for analysis type', 'error');
        return;
    }
    
    console.log('Column config:', config.columns);
    
    // Clear previous selections
    selectedColumnMapping = {};
    
    // Reset all select elements first
    Object.keys(config.columns).forEach(metric => {
        const selectElement = document.getElementById(`column_${metric}`);
        if (selectElement) {
            selectElement.value = '';
        }
    });
    
    let detectedCount = 0;
    
    // Auto-detect based on column names with improved matching
    Object.entries(config.columns).forEach(([metric, metricConfig]) => {
        console.log(`\nTrying to match metric: ${metric}`);
        console.log(`Examples: ${metricConfig.examples}`);
        
        for (const availableCol of availableColumns) {
            const colLower = availableCol.toLowerCase().replace(/[_\s-]/g, '');
            
            // Check if any example patterns match
            for (const example of metricConfig.examples) {
                const exampleLower = example.toLowerCase().replace(/[_\s-]/g, '');
                
                // Multiple matching strategies
                const exactMatch = colLower === exampleLower;
                const containsMatch = colLower.includes(exampleLower) || exampleLower.includes(colLower);
                const partialMatch = colLower.includes(exampleLower.substring(0, 4)) || exampleLower.includes(colLower.substring(0, 4));
                
                if (exactMatch || containsMatch || (partialMatch && exampleLower.length > 3)) {
                    selectedColumnMapping[metric] = availableCol;
                    
                    // Update the select element
                    const selectElement = document.getElementById(`column_${metric}`);
                    if (selectElement) {
                        selectElement.value = availableCol;
                        console.log(`✓ Auto-detected ${metric} -> ${availableCol} (matched with ${example})`);
                        detectedCount++;
                    }
                    
                    break; // Move to next metric after first match
                }
            }
            
            if (selectedColumnMapping[metric]) break; // Break outer loop if found
        }
        
        if (!selectedColumnMapping[metric]) {
            console.log(`✗ No match found for metric: ${metric}`);
        }
    });
    
    updateConfigurationSummary();
    
    // Show feedback
    console.log(`Auto-detection completed. Found ${detectedCount} mappings:`, selectedColumnMapping);
    
    if (detectedCount > 0) {
        showNotification(`✅ Auto-detected ${detectedCount} column mappings!`, 'success');
    } else {
        showNotification('⚠️ Could not auto-detect columns. Please select manually.', 'warning');
    }
}

// Update configuration summary
function updateConfigurationSummary() {
    const summaryDiv = document.getElementById('configurationSummary');
    const summaryList = document.getElementById('mappingSummaryList');
    
    if (Object.keys(selectedColumnMapping).length === 0) {
        summaryDiv.style.display = 'none';
        return;
    }
    
    summaryDiv.style.display = 'block';
    
    const mappingItems = Object.entries(selectedColumnMapping)
        .map(([metric, column]) => `<span class="badge bg-primary me-1">${metric}: ${column}</span>`)
        .join(' ');
    
    summaryList.innerHTML = `<p><strong>Selected mappings:</strong><br>${mappingItems}</p>`;
}

// Generate marketing analysis with selected columns
async function generateMarketingAnalysis() {
    if (!currentAnalysisType) {
        showNotification('❌ No analysis type selected', 'error');
        return;
    }

    const modal = bootstrap.Modal.getInstance(document.getElementById('marketingAnalysisModal'));
    marketingModalConfirmed = true;
    if (modal) modal.hide();

    const mappingCount = Object.keys(selectedColumnMapping).length;
    const marketingConfig = isMarketingAnalysisType(currentAnalysisType) && typeof window !== 'undefined'
        ? window.MARKETING_ANALYSIS_CONFIG?.[currentAnalysisType]
        : null;
    const analysisLabel = marketingConfig?.title || getAnalysisTypeName(currentAnalysisType);

    if (mappingCount === 0) {
        showNotification(`⚠️ Generating ${analysisLabel} with auto-detection (no columns mapped)`, 'warning');
    } else {
        const mappedLabel = mappingCount === 1 ? 'mapped column' : 'mapped columns';
        showNotification(`✅ Generating ${analysisLabel} with ${mappingCount} ${mappedLabel}`, 'success');
    }

    try {
        if (!currentMarketingCellId) {
            const fallbackColumns = await gatherAvailableColumnNames();
            const sanitizedFallbackColumns = Array.from(new Set((fallbackColumns || []).filter(Boolean)));
            const cellId = await addSingleAnalysisCell(currentAnalysisType, {
                skipMarketingModal: true,
                prefetchedColumns: sanitizedFallbackColumns
            });
            currentMarketingCellId = cellId || '';
        }

        if (!currentMarketingCellId) {
            showNotification('Unable to start marketing analysis: no analysis cell available', 'error');
            return;
        }

        await generateAndRunAnalysis(currentMarketingCellId, currentAnalysisType, selectedColumnMapping);
    } catch (error) {
        console.error('Marketing analysis generation failed:', error);
        showNotification(`Marketing analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        currentMarketingCellId = '';
        marketingModalConfirmed = false;
    }
}

// Debug function to help troubleshoot column mapping issues
function debugColumnMapping() {
    console.log('=== DEBUG COLUMN MAPPING ===');
    console.log('Current analysis type:', currentAnalysisType);
    console.log('Available columns:', availableColumns);
    console.log('Column insights data:', columnInsightsData);
    console.log('Selected column mapping:', selectedColumnMapping);
    
    const config = MARKETING_ANALYSIS_CONFIG[currentAnalysisType];
    if (config) {
        console.log('Analysis config:', config);
        console.log('Required columns config:', config.columns);
        
        // Test auto-detect logic for each metric
        console.log('\n=== TESTING AUTO-DETECT LOGIC ===');
        Object.entries(config.columns).forEach(([metric, metricConfig]) => {
            console.log(`\nMetric: ${metric}`);
            console.log(`Examples: ${metricConfig.examples}`);
            
            let bestMatch = null;
            for (const availableCol of availableColumns) {
                const colLower = availableCol.toLowerCase().replace(/[_\s-]/g, '');
                
                for (const example of metricConfig.examples) {
                    const exampleLower = example.toLowerCase().replace(/[_\s-]/g, '');
                    
                    const exactMatch = colLower === exampleLower;
                    const containsMatch = colLower.includes(exampleLower) || exampleLower.includes(colLower);
                    const partialMatch = colLower.includes(exampleLower.substring(0, 4)) || exampleLower.includes(colLower.substring(0, 4));
                    
                    if (exactMatch || containsMatch || (partialMatch && exampleLower.length > 3)) {
                        bestMatch = { column: availableCol, example: example, type: exactMatch ? 'exact' : containsMatch ? 'contains' : 'partial' };
                        console.log(`   ✓ Found match: ${availableCol} (${bestMatch.type} match with ${example})`);
                        break;
                    }
                }
                
                if (bestMatch) break;
            }
            
            if (!bestMatch) {
                console.log(`   ✗ No match found for ${metric}`);
            }
        });
    }
    
    // Show alert with key info
    const summary = `
Debug Info:
- Analysis: ${currentAnalysisType}
- Available columns: ${availableColumns.length}
- Column insights: ${columnInsightsData ? 'Available' : 'Not available'}
- Current mappings: ${Object.keys(selectedColumnMapping).length}

Check console for detailed output.
    `;
    
    alert(summary.trim());
    console.log('=== END DEBUG ===');
}
