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

