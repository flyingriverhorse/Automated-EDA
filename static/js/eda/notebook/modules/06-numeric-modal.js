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

