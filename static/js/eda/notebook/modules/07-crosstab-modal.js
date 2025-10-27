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
