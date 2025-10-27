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

