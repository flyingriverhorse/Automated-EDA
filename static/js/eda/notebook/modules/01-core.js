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
