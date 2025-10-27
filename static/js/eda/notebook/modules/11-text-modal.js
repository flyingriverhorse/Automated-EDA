'use strict';

const TEXT_CATEGORY_TOKENS = ['text', 'string', 'nlp', 'freeform', 'comment', 'description', 'feedback', 'review'];
const TEXT_TYPE_TOKENS = ['object', 'string', 'text'];
const TEXT_NAME_HINTS = ['comment', 'description', 'message', 'note', 'review', 'summary', 'text', 'content', 'body', 'feedback', 'title'];

async function prepareTextAnalysisConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    resetTextModalState();
    currentTextAnalysisType = analysisType;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before text modal:', error);
        }
    }

    const columnCandidates = getTextColumnCandidates();
    if (!columnCandidates.length) {
        showNotification('No text-like columns detected. Running analysis with default behaviour instead.', 'info');
        resetTextModalState();
        await addSingleAnalysisCell(analysisType);
        return;
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipTextModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare text analysis cell. Please try again.', 'error');
        resetTextModalState();
        return;
    }

    currentTextCellId = cellId;
    textModalColumns = columnCandidates;

    initializeTextModal();
    attachTextModalLifecycleHandlers();
    populateTextModal(analysisType, columnCandidates);

    const modalElement = document.getElementById('textAnalysisModal');
    if (!modalElement) {
        showNotification('Text configuration modal is unavailable.', 'error');
        resetTextModalState();
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
    showNotification(`Select text columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
}

function resetTextModalState() {
    currentTextAnalysisType = '';
    currentTextCellId = '';
    textModalConfirmed = false;
    textModalSelection = new Set();
    textModalColumns = [];
    textModalRecommendedDefaults = [];
    textModalSearchTerm = '';
    textModalIsRerun = false;
}

function attachTextModalLifecycleHandlers() {
    const modalElement = document.getElementById('textAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleTextModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleTextModalHidden() {
    if (textModalConfirmed) {
        textModalConfirmed = false;
        textModalIsRerun = false;
        return;
    }

    if (textModalIsRerun) {
        textModalIsRerun = false;
        const message = currentTextAnalysisType ? `${getAnalysisTypeName(currentTextAnalysisType)} rerun cancelled.` : 'Text analysis rerun cancelled.';
        showNotification(message, 'info');
        resetTextModalState();
        return;
    }

    if (currentTextCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentTextCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        if (currentTextAnalysisType) {
            showNotification('Text analysis cancelled.', 'info');
        }
    }

    resetTextModalState();
}

function initializeTextModal() {
    const modalElement = document.getElementById('textAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const confirmBtn = document.getElementById('textModalConfirmBtn');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchTextAnalysis);
    }

    const selectAllBtn = document.getElementById('textModalSelectAllBtn');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', selectAllTextColumns);
    }

    const clearBtn = document.getElementById('textModalClearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            textModalSelection = new Set();
            renderTextColumnList();
            updateTextSelectionSummary();
            updateTextChipStates();
        });
    }

    const recommendedBtn = document.getElementById('textModalRecommendBtn');
    if (recommendedBtn) {
        recommendedBtn.addEventListener('click', applyTextRecommendations);
    }

    const searchInput = document.getElementById('textColumnSearch');
    if (searchInput) {
        searchInput.addEventListener('input', event => {
            textModalSearchTerm = (event.target.value || '').toLowerCase();
            renderTextColumnList();
            updateTextChipStates();
        });
    }

    const columnList = document.getElementById('textColumnList');
    if (columnList) {
        columnList.addEventListener('change', handleTextListChange);
    }

    const chipsContainer = document.getElementById('textRecommendationChips');
    if (chipsContainer) {
        chipsContainer.addEventListener('click', handleTextChipClick);
    }

    modalElement.dataset.initialized = 'true';
}

function getTextColumnCandidates() {
    if (!Array.isArray(columnInsightsData?.column_insights)) {
        return [];
    }

    const candidates = [];

    columnInsightsData.column_insights.forEach(col => {
        if (!col || col.dropped) {
            return;
        }

        const normalizedName = (col.name ?? '').toString().trim();
        if (!normalizedName) {
            return;
        }

        const lowerName = normalizedName.toLowerCase();
        const dataCategory = (col.data_category || '').toString().toLowerCase();
        const dataType = (col.data_type || '').toString().toLowerCase();
        const stats = typeof col.statistics === 'object' && col.statistics !== null ? col.statistics : {};

        const uniqueCount = toFiniteNumber(stats.unique_count ?? stats.distinct_count ?? stats.cardinality);
        const missingPct = toFiniteNumber(col.null_percentage ?? stats.null_percentage);
        const avgLength = toFiniteNumber(
            stats.avg_length ?? stats.average_length ?? stats.mean_length ?? stats.mean_char_length ?? stats.avg_char_length
        );
        const medianLength = toFiniteNumber(stats.median_length ?? stats.median_char_length);
        const maxLength = toFiniteNumber(stats.max_length ?? stats.max_char_length ?? stats.max_len);
        const avgWords = toFiniteNumber(stats.avg_word_count ?? stats.average_word_count);
        const alphaShare = toFiniteNumber(stats.alpha_ratio ?? stats.alpha_share ?? stats.alpha_percentage);
        const sampleValues = Array.isArray(stats.sample_values)
            ? stats.sample_values
            : Array.isArray(col.sample_values)
                ? col.sample_values
                : [];

        const sampleSuggestsText = sampleValues.some(value => typeof value === 'string' && value.split(/\s+/).length >= 3);
        const isTextCategory = TEXT_CATEGORY_TOKENS.some(token => dataCategory.includes(token));
        const isTextType = TEXT_TYPE_TOKENS.some(token => dataType.includes(token));
        const nameSuggestsText = TEXT_NAME_HINTS.some(token => lowerName.includes(token));
        const cardinalityReasonable = typeof uniqueCount === 'number' ? uniqueCount > 10 : true;
        const avgLengthReasonable = typeof avgLength === 'number' ? avgLength >= 8 : true;
        const alphaLikelyText = typeof alphaShare === 'number' ? alphaShare >= 0.4 : false;

        if (
            !isTextCategory &&
            !isTextType &&
            !nameSuggestsText &&
            !sampleSuggestsText &&
            !alphaLikelyText
        ) {
            return;
        }

        const reasonParts = [];
        if (typeof uniqueCount === 'number') {
            reasonParts.push(`${uniqueCount} unique`);
        }
        if (typeof avgLength === 'number') {
            reasonParts.push(`${avgLength.toFixed(1)} avg chars`);
        }
        if (typeof avgWords === 'number') {
            reasonParts.push(`${avgWords.toFixed(1)} avg words`);
        }
        if (typeof missingPct === 'number') {
            reasonParts.push(`${missingPct.toFixed(1)}% missing`);
        }
        if (alphaLikelyText) {
            reasonParts.push('Alphabetic rich');
        }
        if (nameSuggestsText) {
            reasonParts.push('Name hint');
        }

        const recommended = Boolean(
            (avgLengthReasonable && cardinalityReasonable) ||
            (sampleSuggestsText && (typeof avgWords !== 'number' || avgWords <= 80))
        );

        candidates.push({
            name: normalizedName,
            dataCategory: dataCategory || dataType || 'text',
            dataType,
            uniqueCount: typeof uniqueCount === 'number' ? uniqueCount : null,
            missingPct: typeof missingPct === 'number' ? missingPct : null,
            avgLength: typeof avgLength === 'number' ? avgLength : null,
            medianLength: typeof medianLength === 'number' ? medianLength : null,
            maxLength: typeof maxLength === 'number' ? maxLength : null,
            avgWords: typeof avgWords === 'number' ? avgWords : null,
            recommended,
            reason: reasonParts.join(' • ')
        });
    });

    candidates.sort((a, b) => {
        if (a.recommended !== b.recommended) {
            return a.recommended ? -1 : 1;
        }
        const aAvg = typeof a.avgLength === 'number' ? a.avgLength : -1;
        const bAvg = typeof b.avgLength === 'number' ? b.avgLength : -1;
        if (aAvg !== bAvg) {
            return bAvg - aAvg;
        }
        return a.name.localeCompare(b.name);
    });

    return candidates;
}

function populateTextModal(analysisType, columns, initialSelection = null) {
    const analysisName = getAnalysisTypeName(analysisType);
    const modalLabel = document.getElementById('textAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.textContent = `Configure ${analysisName}`;
    }

    const subtitle = document.getElementById('textModalSubtitle');
    if (subtitle) {
        subtitle.textContent = 'Pick text columns to review length distributions, token frequencies, and vocabulary richness.';
    }

    const confirmBtn = document.getElementById('textModalConfirmBtn');
    if (confirmBtn) {
        const baseLabel = `Run ${analysisName}`;
        confirmBtn.dataset.baseLabel = baseLabel;
        confirmBtn.textContent = baseLabel;
        confirmBtn.disabled = false;
    }

    const searchInput = document.getElementById('textColumnSearch');
    if (searchInput) {
        searchInput.value = '';
    }

    textModalColumns = Array.isArray(columns) ? [...columns] : [];

    const recommended = textModalColumns.filter(col => col.recommended);
    textModalRecommendedDefaults = recommended.slice(0, 6).map(col => col.name);

    const normalizedInitialSelection = Array.isArray(initialSelection) || initialSelection instanceof Set
        ? Array.from(new Set(initialSelection instanceof Set ? [...initialSelection] : initialSelection)).filter(name =>
            textModalColumns.some(col => col.name === name)
        )
        : [];

    if (normalizedInitialSelection.length > 0) {
        textModalSelection = new Set(normalizedInitialSelection);
    } else if (textModalRecommendedDefaults.length > 0) {
        textModalSelection = new Set(textModalRecommendedDefaults);
    } else {
        const fallback = textModalColumns.slice(0, Math.min(5, textModalColumns.length)).map(col => col.name);
        textModalSelection = new Set(fallback);
    }

    renderTextRecommendations(recommended);
    renderTextColumnList();
    updateTextChipStates();
    updateTextSelectionSummary();
}

function renderTextRecommendations(recommendedColumns) {
    const container = document.getElementById('textRecommendationChips');
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

function renderTextColumnList() {
    const listElement = document.getElementById('textColumnList');
    if (!listElement) {
        return;
    }

    let filtered = [...textModalColumns];
    if (textModalSearchTerm) {
        filtered = filtered.filter(col => col.name.toLowerCase().includes(textModalSearchTerm));
    }

    if (!filtered.length) {
        listElement.innerHTML = '<div class="text-muted small px-2 py-3">No columns match your search.</div>';
        return;
    }

    const rows = filtered
        .map(col => {
            const checked = textModalSelection.has(col.name) ? 'checked' : '';
            const detailParts = [];
            if (typeof col.avgLength === 'number') {
                detailParts.push(`${col.avgLength.toFixed(1)} avg chars`);
            }
            if (typeof col.avgWords === 'number') {
                detailParts.push(`${col.avgWords.toFixed(1)} avg words`);
            }
            if (typeof col.uniqueCount === 'number') {
                detailParts.push(`${col.uniqueCount} unique`);
            }
            if (typeof col.missingPct === 'number') {
                detailParts.push(`${col.missingPct.toFixed(1)}% missing`);
            }
            const detailText = detailParts.length ? `<small class="text-muted">${detailParts.join(' • ')}</small>` : '';
            const badgeLabel = col.dataCategory || col.dataType || 'text';

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

function updateTextSelectionSummary() {
    const summaryElement = document.getElementById('textSelectionSummary');
    const confirmBtn = document.getElementById('textModalConfirmBtn');
    const count = textModalSelection.size;

    if (summaryElement) {
        if (count === 0) {
            summaryElement.textContent = 'Select one or more text columns to continue.';
        } else {
            const preview = Array.from(textModalSelection).slice(0, 5);
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

function updateTextChipStates() {
    const chipsContainer = document.getElementById('textRecommendationChips');
    if (!chipsContainer) {
        return;
    }
    chipsContainer.querySelectorAll('[data-column]').forEach(button => {
        const columnName = button.dataset.column;
        button.classList.toggle('active', textModalSelection.has(columnName));
    });
}

function handleTextListChange(event) {
    const target = event.target;
    if (!target || target.type !== 'checkbox') {
        return;
    }

    const columnName = target.dataset.column;
    if (!columnName) {
        return;
    }

    if (target.checked) {
        textModalSelection.add(columnName);
    } else {
        textModalSelection.delete(columnName);
    }

    updateTextSelectionSummary();
    updateTextChipStates();
}

function handleTextChipClick(event) {
    const target = event.target;
    if (!target || !target.dataset?.column) {
        return;
    }

    const columnName = target.dataset.column;
    if (textModalSelection.has(columnName)) {
        textModalSelection.delete(columnName);
    } else {
        textModalSelection.add(columnName);
    }

    renderTextColumnList();
    updateTextSelectionSummary();
    updateTextChipStates();
}

function applyTextRecommendations() {
    if (!textModalRecommendedDefaults.length) {
        showNotification('No recommended text columns were identified yet.', 'info');
        return;
    }

    textModalSelection = new Set(textModalRecommendedDefaults);
    renderTextColumnList();
    updateTextSelectionSummary();
    updateTextChipStates();
}

function selectAllTextColumns() {
    if (!textModalColumns.length) {
        return;
    }

    textModalSelection = new Set(textModalColumns.map(col => col.name));
    renderTextColumnList();
    updateTextSelectionSummary();
    updateTextChipStates();
}

async function launchTextAnalysis() {
    if (!currentTextAnalysisType) {
        showNotification('No text analysis selected.', 'error');
        return;
    }

    const selectedList = Array.from(textModalSelection);
    if (!selectedList.length) {
        showNotification('Select at least one text column to continue.', 'warning');
        return;
    }

    const confirmBtn = document.getElementById('textModalConfirmBtn');
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

    textModalConfirmed = true;
    const modalElement = document.getElementById('textAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    if (modalInstance) {
        modalInstance.hide();
    }

    try {
        if (!currentTextCellId) {
            const fallbackCellId = await addSingleAnalysisCell(currentTextAnalysisType, {
                skipTextModal: true
            });
            currentTextCellId = fallbackCellId || '';
        }

        if (!currentTextCellId) {
            showNotification('Unable to start text analysis: no analysis cell available.', 'error');
            return;
        }

        showNotification(`Running ${getAnalysisTypeName(currentTextAnalysisType)} for ${selectedList.length} column${selectedList.length === 1 ? '' : 's'}.`, 'success');

        await generateAndRunAnalysis(
            currentTextCellId,
            currentTextAnalysisType,
            {},
            {
                overrideSelectedColumns: selectedList,
                includeGlobalSelectedColumns: false,
                modalType: 'text'
            }
        );
    } catch (error) {
        console.error('Text analysis run failed:', error);
        showNotification(`Text analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.removeAttribute('aria-busy');
            if (previousLabel !== null) {
                confirmBtn.innerHTML = previousLabel;
            }
        }
        resetTextModalState();
    }
}

async function openTextModalForRerun(cellId, analysisType, previousSelection = []) {
    textModalIsRerun = true;
    currentTextAnalysisType = analysisType;
    currentTextCellId = cellId;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before text rerun:', error);
        }
    }

    const columnCandidates = getTextColumnCandidates();
    if (!columnCandidates.length) {
        textModalIsRerun = false;
        showNotification('Text columns are unavailable for rerun.', 'warning');
        return;
    }

    textModalColumns = columnCandidates;

    initializeTextModal();
    attachTextModalLifecycleHandlers();
    populateTextModal(analysisType, columnCandidates, previousSelection);

    const modalElement = document.getElementById('textAnalysisModal');
    if (!modalElement) {
        textModalIsRerun = false;
        showNotification('Text configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    if (previousSelection && previousSelection.length > 0) {
        showNotification('Review or adjust your previously selected text columns, then rerun.', 'info');
    } else {
        showNotification(`Select text columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
    }
}
