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
