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
