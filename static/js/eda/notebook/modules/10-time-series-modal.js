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
