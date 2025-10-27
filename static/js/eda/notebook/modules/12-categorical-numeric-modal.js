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
