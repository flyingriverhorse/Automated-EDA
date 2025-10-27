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
