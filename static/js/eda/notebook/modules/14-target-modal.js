const TARGET_MODAL_MAX_SELECTION = 3;
const TARGET_NAME_HINTS = [
    'target',
    'label',
    'outcome',
    'response',
    'class',
    'status',
    'result',
    'churn',
    'default',
    'fraud',
    'success',
    'converted',
    'y'
];

async function prepareTargetAnalysisConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    currentTargetAnalysisType = analysisType;
    currentTargetCellId = '';
    targetModalConfirmed = false;
    targetModalIsRerun = false;
    targetModalSelection = new Set();
    targetModalColumns = [];
    targetModalRecommendedDefaults = [];
    targetModalSearchTerm = '';

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before target modal:', error);
        }
    }

    const columnCandidates = getTargetColumnCandidates();
    if (!columnCandidates.length) {
        showNotification('No suitable target candidates detected. Running full analysis instead.', 'info');
        await addSingleAnalysisCell(analysisType);
        return;
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipTargetModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare target analysis cell. Please try again.', 'error');
        return;
    }

    currentTargetCellId = cellId;
    targetModalColumns = columnCandidates;

    initializeTargetModal();
    attachTargetModalLifecycleHandlers();
    populateTargetModal(analysisType, columnCandidates);

    const modalElement = document.getElementById('targetAnalysisModal');
    if (!modalElement) {
        showNotification('Target configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
    showNotification(`Select target columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
}

function attachTargetModalLifecycleHandlers() {
    const modalElement = document.getElementById('targetAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleTargetModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleTargetModalHidden() {
    if (targetModalConfirmed) {
        resetTargetModalState();
        return;
    }

    if (targetModalIsRerun) {
        targetModalIsRerun = false;
        currentTargetCellId = '';
        currentTargetAnalysisType = '';
        targetModalSelection = new Set();
        targetModalColumns = [];
        targetModalRecommendedDefaults = [];
        targetModalSearchTerm = '';
        showNotification('Target analysis rerun cancelled.', 'info');
        return;
    }

    if (currentTargetCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentTargetCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        if (currentTargetAnalysisType) {
            showNotification('Target analysis cancelled.', 'info');
        }
    }

    resetTargetModalState();
}

function resetTargetModalState() {
    targetModalConfirmed = false;
    targetModalIsRerun = false;
    currentTargetCellId = '';
    currentTargetAnalysisType = '';
    targetModalSelection = new Set();
    targetModalColumns = [];
    targetModalRecommendedDefaults = [];
    targetModalSearchTerm = '';
}

function initializeTargetModal() {
    const modalElement = document.getElementById('targetAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const confirmBtn = document.getElementById('targetModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchTargetAnalysis);
    }

    const recommendBtn = document.getElementById('targetModalRecommendBtn');
    if (recommendBtn) {
        recommendBtn.addEventListener('click', () => {
            applyTargetRecommendations();
        });
    }

    const clearBtn = document.getElementById('targetModalClearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            targetModalSelection = new Set();
            renderTargetColumnList();
            updateTargetSelectionSummary();
        });
    }

    const searchInput = document.getElementById('targetColumnSearch');
    if (searchInput) {
        searchInput.addEventListener('input', event => {
            targetModalSearchTerm = (event.target.value || '').toLowerCase();
            renderTargetColumnList();
        });
    }

    const columnList = document.getElementById('targetColumnList');
    if (columnList) {
        columnList.addEventListener('change', handleTargetListChange);
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

function getTargetColumnCandidates() {
    if (!Array.isArray(columnInsightsData?.column_insights)) {
        return [];
    }

    const candidates = [];

    columnInsightsData.column_insights.forEach(col => {
        if (!col || col.dropped) {
            return;
        }

        const name = (col.name ?? '').toString().trim();
        if (!name) {
            return;
        }

        const stats = typeof col.statistics === 'object' && col.statistics !== null ? col.statistics : {};
        const rowCount = toFiniteNumber(stats.count ?? stats.row_count ?? stats.total_count) ?? toFiniteNumber(col.row_count);
        const uniqueCount = toFiniteNumber(stats.unique_count ?? stats.distinct_count ?? stats.cardinality);
        const uniqueRatioRaw = toFiniteNumber(stats.unique_ratio ?? stats.distinct_ratio);
        const uniqueRatio = uniqueRatioRaw !== null && uniqueRatioRaw > 1 ? uniqueRatioRaw / 100 : uniqueRatioRaw;
        const missingPctRaw = toFiniteNumber(col.null_percentage ?? stats.missing_ratio);
        const missingPct = missingPctRaw !== null && missingPctRaw <= 1 ? missingPctRaw * 100 : missingPctRaw;
        const dataCategory = (col.data_category || '').toString().toLowerCase();
        const dataType = (col.data_type || '').toString().toLowerCase();

        if (uniqueCount !== null && uniqueCount <= 1) {
            return;
        }

        const normalizedName = name.toLowerCase();
        const hintScore = TARGET_NAME_HINTS.some(hint => normalizedName.includes(hint)) ? 3 : 0;

        const isNumeric = ['int', 'int64', 'float', 'float64', 'numeric', 'number'].some(type => dataType.includes(type)) || dataCategory === 'numeric';
        const isCategorical = ['categorical', 'category', 'object', 'bool', 'boolean', 'text'].some(type => dataCategory.includes(type) || dataType.includes(type));

        let role = 'regression';
        let score = hintScore;
        const uniqueRatioValue = uniqueRatio !== null ? uniqueRatio : (rowCount && uniqueCount !== null ? uniqueCount / rowCount : null);
    const hasLowCardinality = uniqueCount !== null && uniqueCount <= 20;
    const hasBinaryStyleRatio = uniqueRatioValue !== null && uniqueRatioValue <= 0.3;
    const qualifiesByRatio = hasBinaryStyleRatio && (uniqueCount === null || uniqueCount <= 50);

        if (isCategorical || hasLowCardinality || qualifiesByRatio) {
            role = 'classification';
            score += 2;
            if (uniqueCount !== null && (uniqueCount === 2 || uniqueCount === 3)) {
                score += 1;
            }
            if (qualifiesByRatio) {
                score += 1;
            }
        } else if (isNumeric) {
            role = 'regression';
            score += 1.5;
            if (uniqueRatioValue !== null && uniqueRatioValue >= 0.4) {
                score += 0.5;
            }
        }

        if (uniqueRatioValue !== null && uniqueRatioValue > 0.95 && uniqueCount !== null && uniqueCount > 20) {
            score -= 2.5;
        }

        if (missingPct !== null && missingPct > 40) {
            score -= 1;
        }

        const reasonParts = [];
        if (uniqueCount !== null) {
            reasonParts.push(`${uniqueCount} ${uniqueCount === 1 ? 'unique value' : 'unique values'}`);
        }
        if (uniqueRatioValue !== null) {
            const ratioPct = uniqueRatioValue <= 1 ? uniqueRatioValue * 100 : uniqueRatioValue;
            reasonParts.push(`${ratioPct.toFixed(ratioPct < 5 ? 1 : 0)}% unique`);
        }
        if (missingPct !== null) {
            reasonParts.push(`${missingPct.toFixed(1)}% missing`);
        }

        candidates.push({
            name,
            dataCategory,
            dataType,
            uniqueCount: uniqueCount !== null ? uniqueCount : undefined,
            uniqueRatio: uniqueRatioValue !== null ? uniqueRatioValue : undefined,
            missingPct: missingPct !== null ? missingPct : undefined,
            role,
            score,
            reason: reasonParts.join(' • '),
            recommended: score >= 3 || hintScore > 0
        });
    });

    candidates.sort((a, b) => {
        if (a.recommended !== b.recommended) {
            return a.recommended ? -1 : 1;
        }
        if (b.score !== a.score) {
            return b.score - a.score;
        }
        return a.name.localeCompare(b.name);
    });

    return candidates;
}

function populateTargetModal(analysisType, columns, initialSelection = null, selectionDetails = null) {
    const modalLabel = document.getElementById('targetAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.textContent = `Configure ${getAnalysisTypeName(analysisType)}`;
    }

    const subtitle = document.getElementById('targetModalSubtitle');
    if (subtitle) {
        subtitle.textContent = 'Choose target columns to profile class balance or value spread before modelling.';
    }

    const confirmBtn = document.getElementById('targetModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.dataset.baseLabel = `Run ${getAnalysisTypeName(analysisType)}`;
        confirmBtn.textContent = confirmBtn.dataset.baseLabel;
        confirmBtn.disabled = false;
    }

    targetModalColumns = Array.isArray(columns) ? [...columns] : [];
    targetModalRecommendedDefaults = targetModalColumns.filter(col => col.recommended).slice(0, TARGET_MODAL_MAX_SELECTION).map(col => col.name);

    const normalizedInitialSelection = Array.isArray(initialSelection) || initialSelection instanceof Set
        ? Array.from(new Set(initialSelection instanceof Set ? [...initialSelection] : initialSelection)).filter(name =>
            targetModalColumns.some(col => col.name === name)
        )
        : [];

    if (normalizedInitialSelection.length > 0) {
        targetModalSelection = new Set(normalizedInitialSelection);
    } else if (targetModalRecommendedDefaults.length > 0) {
        targetModalSelection = new Set(targetModalRecommendedDefaults);
    } else {
        const fallbackDefaults = targetModalColumns.slice(0, TARGET_MODAL_MAX_SELECTION).map(col => col.name);
        targetModalSelection = new Set(fallbackDefaults);
    }

    if (selectionDetails && Array.isArray(selectionDetails)) {
        selectionDetails.forEach(detail => {
            const columnName = detail?.column;
            const role = detail?.role;
            if (!columnName) {
                return;
            }
            const candidate = targetModalColumns.find(col => col.name === columnName);
            if (candidate && role && (role === 'classification' || role === 'regression')) {
                candidate.role = role;
            }
        });
    }

    renderTargetRecommendations();
    renderTargetColumnList();
    updateTargetSelectionSummary();
}

function renderTargetRecommendations() {
    const container = document.getElementById('targetRecommendationChips');
    if (!container) {
        return;
    }

    container.innerHTML = '';

    if (!targetModalRecommendedDefaults.length) {
        container.innerHTML = '<span class="text-muted small">No automatic recommendations yet. Select columns manually below.</span>';
        return;
    }

    targetModalRecommendedDefaults.forEach(columnName => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'btn btn-outline-primary btn-sm me-2 mb-2';
        chip.dataset.column = columnName;
        chip.textContent = columnName;
        chip.addEventListener('click', () => {
            if (!targetModalSelection.has(columnName)) {
                if (targetModalSelection.size >= TARGET_MODAL_MAX_SELECTION) {
                    showNotification(`Select up to ${TARGET_MODAL_MAX_SELECTION} target columns.`, 'warning');
                    return;
                }
                targetModalSelection.add(columnName);
                renderTargetColumnList();
                updateTargetSelectionSummary();
            }
        });
        container.appendChild(chip);
    });
}

function renderTargetColumnList() {
    const listElement = document.getElementById('targetColumnList');
    if (!listElement) {
        return;
    }

    let filtered = [...targetModalColumns];
    if (targetModalSearchTerm) {
        filtered = filtered.filter(col => col.name.toLowerCase().includes(targetModalSearchTerm));
    }

    if (!filtered.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No columns match your search.</div>';
        return;
    }

    const rows = filtered.map(col => {
        const checked = targetModalSelection.has(col.name) ? 'checked' : '';
        const typeBadge = col.dataCategory
            ? `<span class="badge text-bg-light text-capitalize">${escapeHtml(col.dataCategory || col.dataType || 'column')}</span>`
            : '';
        const detailParts = [];
        if (typeof col.uniqueCount === 'number') {
            detailParts.push(`${col.uniqueCount} ${col.role === 'classification' ? 'classes' : 'unique values'}`);
        }
        if (typeof col.uniqueRatio === 'number') {
            const ratioPct = col.uniqueRatio <= 1 ? col.uniqueRatio * 100 : col.uniqueRatio;
            detailParts.push(`${ratioPct.toFixed(ratioPct < 5 ? 1 : 0)}% unique`);
        }
        if (typeof col.missingPct === 'number') {
            detailParts.push(`${col.missingPct.toFixed(1)}% missing`);
        }
        const reasonText = col.reason ? `<div class="text-muted small">${escapeHtml(col.reason)}</div>` : '';

        return `
            <label class="list-group-item d-flex align-items-start gap-3 target-column-item">
                <input class="form-check-input mt-1" type="checkbox" value="${escapeHtml(col.name)}" data-column="${escapeHtml(col.name)}" ${checked}>
                <div class="flex-grow-1">
                    <div class="d-flex align-items-center gap-2 flex-wrap">
                        <strong>${escapeHtml(col.name)}</strong>
                        ${typeBadge}
                    </div>
                    <div class="text-muted small">${detailParts.join(' • ') || 'Profiling metrics not available yet.'}</div>
                    ${reasonText}
                </div>
            </label>
        `;
    }).join('');

    listElement.innerHTML = rows;
}

function handleTargetListChange(event) {
    if (!event || event.target.type !== 'checkbox') {
        return;
    }

    const columnName = event.target.value;
    if (!columnName) {
        return;
    }

    if (event.target.checked) {
        if (targetModalSelection.size >= TARGET_MODAL_MAX_SELECTION) {
            event.target.checked = false;
            showNotification(`Select up to ${TARGET_MODAL_MAX_SELECTION} target columns.`, 'warning');
            return;
        }
        targetModalSelection.add(columnName);
    } else {
        targetModalSelection.delete(columnName);
    }

    updateTargetSelectionSummary();
}

function applyTargetRecommendations() {
    if (!targetModalRecommendedDefaults.length) {
        showNotification('No recommended target columns at the moment.', 'info');
        return;
    }

    targetModalSelection = new Set(targetModalRecommendedDefaults.slice(0, TARGET_MODAL_MAX_SELECTION));
    renderTargetColumnList();
    updateTargetSelectionSummary();
}

function updateTargetSelectionSummary() {
    const summaryElement = document.getElementById('targetSelectionSummary');
    const confirmBtn = document.getElementById('targetModalConfirmBtn');
    const count = targetModalSelection.size;

    if (summaryElement) {
        if (count === 0) {
            summaryElement.textContent = `Select at least one target column (up to ${TARGET_MODAL_MAX_SELECTION}).`;
        } else {
            const preview = Array.from(targetModalSelection).slice(0, TARGET_MODAL_MAX_SELECTION);
            const remainder = count - preview.length;
            const suffix = remainder > 0 ? `, +${remainder} more` : '';
            summaryElement.textContent = `${count} target column${count === 1 ? '' : 's'} selected: ${preview.join(', ')}${suffix}`;
        }
    }

    if (confirmBtn) {
        const baseLabel = confirmBtn.dataset.baseLabel || confirmBtn.textContent || 'Run analysis';
        confirmBtn.textContent = count > 0 ? `${baseLabel} (${count})` : baseLabel;
        confirmBtn.disabled = count === 0;
    }
}

async function launchTargetAnalysis() {
    if (!currentTargetCellId || !currentTargetAnalysisType) {
        showNotification('Target analysis cell is unavailable. Please try again.', 'error');
        return;
    }

    if (targetModalSelection.size === 0) {
        showNotification('Select at least one target column.', 'warning');
        return;
    }

    targetModalConfirmed = true;

    const selection = Array.from(targetModalSelection);
    const selectionDetails = selection.map(columnName => {
        const candidate = targetModalColumns.find(col => col.name === columnName) || {};
        return {
            column: columnName,
            role: candidate.role || 'classification',
            unique_count: candidate.uniqueCount ?? null,
            unique_ratio: candidate.uniqueRatio ?? null,
            missing_pct: candidate.missingPct ?? null,
            score: candidate.score ?? null
        };
    });

    const modalElement = document.getElementById('targetAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    modalInstance?.hide();

    const analysisOptions = {
        overrideSelectedColumns: selection,
        includeGlobalSelectedColumns: false,
        modalType: 'target',
        modalSelectionPayload: {
            columns: selectionDetails
        }
    };

    await generateAndRunAnalysis(currentTargetCellId, currentTargetAnalysisType, {}, analysisOptions);
    resetTargetModalState();
}

async function openTargetModalForRerun(cellId, analysisType, previousSelection = [], previousDetails = null) {
    if (!analysisType) {
        return;
    }

    currentTargetCellId = cellId;
    currentTargetAnalysisType = analysisType;
    targetModalConfirmed = false;
    targetModalIsRerun = true;
    targetModalSelection = new Set();
    targetModalColumns = [];
    targetModalRecommendedDefaults = [];
    targetModalSearchTerm = '';

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before target rerun modal:', error);
        }
    }

    const columnCandidates = getTargetColumnCandidates();
    if (!columnCandidates.length) {
        showNotification('No suitable target candidates detected. Re-running analysis with previous selection.', 'info');
        const fallbackOptions = {
            overrideSelectedColumns: Array.isArray(previousSelection) ? previousSelection : [],
            includeGlobalSelectedColumns: false,
            modalType: 'target',
            modalSelectionPayload: previousDetails && typeof previousDetails === 'object' ? { ...previousDetails } : null
        };
        executeAnalysisRerun(cellId, analysisType, fallbackOptions);
        targetModalIsRerun = false;
        return;
    }

    targetModalColumns = columnCandidates;
    initializeTargetModal();
    attachTargetModalLifecycleHandlers();
    populateTargetModal(analysisType, columnCandidates, previousSelection, previousDetails?.columns || null);

    const modalElement = document.getElementById('targetAnalysisModal');
    if (!modalElement) {
        showNotification('Target configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
    showNotification(`Adjust target selection for ${getAnalysisTypeName(analysisType)}.`, 'info');
}

window.prepareTargetAnalysisConfiguration = prepareTargetAnalysisConfiguration;
window.openTargetModalForRerun = openTargetModalForRerun;
window.launchTargetAnalysis = launchTargetAnalysis;
