const GEOSPATIAL_LAT_HINTS = ['latitude', 'lat', 'latitud', 'lat_deg', 'y', 'northing'];
const GEOSPATIAL_LON_HINTS = ['longitude', 'lon', 'lng', 'long', 'lon_deg', 'x', 'easting'];
const GEOSPATIAL_LABEL_HINTS = ['name', 'city', 'region', 'area', 'location', 'place', 'label', 'site'];

const GEOSPATIAL_PROXIMITY_ANALYSIS_ID = 'geospatial_proximity_analysis';

const GEOSPATIAL_LABEL_DISABLED_TYPES = new Set([
    'coordinate_system_projection_check',
    'spatial_distribution_analysis',
    'spatial_data_quality_analysis'
]);

let currentGeospatialAnalysisType = '';
let currentGeospatialCellId = '';
let geospatialModalConfirmed = false;
let geospatialModalSelection = {
    latitude: '',
    longitude: '',
    label: '',
    proximityRadiusKm: '',
    referenceLatitude: '',
    referenceLongitude: '',
    comparisonLatitude: '',
    comparisonLongitude: '',
    optionalMode: 'none'
};
let geospatialModalColumns = [];
let geospatialModalIsRerun = false;

function geospatialAnalysisSupportsLabel(analysisType) {
    if (!analysisType) {
        return true;
    }
    return !GEOSPATIAL_LABEL_DISABLED_TYPES.has(analysisType);
}

function geospatialAnalysisUsesProximityOptions(analysisType) {
    return analysisType === GEOSPATIAL_PROXIMITY_ANALYSIS_ID;
}

async function prepareGeospatialAnalysisConfiguration(analysisType) {
    if (!analysisType) {
        return;
    }

    currentGeospatialAnalysisType = analysisType;
    geospatialModalConfirmed = false;
    geospatialModalIsRerun = false;
    geospatialModalSelection = {
        latitude: '',
        longitude: '',
        label: '',
        proximityRadiusKm: '',
        referenceLatitude: '',
        referenceLongitude: '',
        comparisonLatitude: '',
        comparisonLongitude: '',
        optionalMode: 'none'
    };

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before geospatial modal:', error);
        }
    }

    geospatialModalColumns = buildGeospatialColumnCandidates();

    if (geospatialModalColumns.length < 2) {
        showNotification('Not enough columns detected to configure geospatial analyses. Running with automatic detection instead.', 'warning');
        await addSingleAnalysisCell(analysisType);
        resetGeospatialModalState();
        return;
    }

    const cellId = await addSingleAnalysisCell(analysisType, {
        skipGeospatialModal: true
    });

    if (!cellId) {
        showNotification('Unable to prepare geospatial analysis cell. Please try again.', 'error');
        resetGeospatialModalState();
        return;
    }

    currentGeospatialCellId = cellId;

    initializeGeospatialModal();
    attachGeospatialModalLifecycleHandlers();

    const supportsLabel = geospatialAnalysisSupportsLabel(analysisType);
    const recommendations = detectGeospatialRecommendations(geospatialModalColumns, supportsLabel);
    populateGeospatialModal(analysisType, geospatialModalColumns, recommendations, supportsLabel);

    const modalElement = document.getElementById('geospatialAnalysisModal');
    if (!modalElement) {
        showNotification('Geospatial configuration modal is unavailable.', 'error');
        resetGeospatialModalState();
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    if (recommendations.latitude && recommendations.longitude) {
        showNotification(`Confirm latitude/longitude columns for ${getAnalysisTypeName(analysisType)}.`, 'info');
    } else {
        showNotification('Select latitude and longitude columns to continue.', 'info');
    }
}

function initializeGeospatialModal() {
    const modalElement = document.getElementById('geospatialAnalysisModal');
    if (!modalElement || modalElement.dataset.initialized === 'true') {
        return;
    }

    const latitudeSelect = document.getElementById('geospatialLatitudeSelect');
    const longitudeSelect = document.getElementById('geospatialLongitudeSelect');
    const labelSelect = document.getElementById('geospatialLabelSelect');
    const confirmBtn = document.getElementById('geospatialModalConfirmBtn');
    const autoBtn = document.getElementById('geospatialModalAutoBtn');
    const radiusInput = document.getElementById('geospatialProximityRadiusInput');
    const referenceLatInput = document.getElementById('geospatialReferenceLatitudeInput');
    const referenceLonInput = document.getElementById('geospatialReferenceLongitudeInput');
    const comparisonLatSelect = document.getElementById('geospatialComparisonLatitudeSelect');
    const comparisonLonSelect = document.getElementById('geospatialComparisonLongitudeSelect');
    const optionalModeRadios = document.querySelectorAll('input[name="geospatialOptionalMode"]');

    if (latitudeSelect) {
        latitudeSelect.addEventListener('change', event => {
            geospatialModalSelection.latitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (longitudeSelect) {
        longitudeSelect.addEventListener('change', event => {
            geospatialModalSelection.longitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (labelSelect) {
        labelSelect.addEventListener('change', event => {
            geospatialModalSelection.label = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (radiusInput) {
        radiusInput.addEventListener('input', event => {
            geospatialModalSelection.proximityRadiusKm = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (referenceLatInput) {
        referenceLatInput.addEventListener('input', event => {
            geospatialModalSelection.referenceLatitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (referenceLonInput) {
        referenceLonInput.addEventListener('input', event => {
            geospatialModalSelection.referenceLongitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (comparisonLatSelect) {
        comparisonLatSelect.addEventListener('change', event => {
            geospatialModalSelection.comparisonLatitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (comparisonLonSelect) {
        comparisonLonSelect.addEventListener('change', event => {
            geospatialModalSelection.comparisonLongitude = (event.target.value || '').trim();
            updateGeospatialSelectionSummary();
        });
    }

    if (optionalModeRadios && optionalModeRadios.length) {
        optionalModeRadios.forEach(radio => {
            radio.addEventListener('change', event => {
                const selectedMode = (event.target.value || '').trim();
                setGeospatialOptionalMode(selectedMode);
            });
        });
    }

    if (confirmBtn) {
        confirmBtn.addEventListener('click', launchGeospatialAnalysis);
    }

    if (autoBtn) {
        autoBtn.addEventListener('click', () => {
            if (!geospatialModalColumns.length) {
                updateGeospatialAutoStatus('No columns available to auto-detect.', 'warning');
                return;
            }
            const supportsLabel = geospatialAnalysisSupportsLabel(currentGeospatialAnalysisType);
            const recommendations = detectGeospatialRecommendations(geospatialModalColumns, supportsLabel);
            applyGeospatialSelection(recommendations, true, supportsLabel);
        });
    }

    modalElement.dataset.initialized = 'true';
}

function attachGeospatialModalLifecycleHandlers() {
    const modalElement = document.getElementById('geospatialAnalysisModal');
    if (!modalElement || modalElement.dataset.lifecycleAttached === 'true') {
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', handleGeospatialModalHidden);
    modalElement.dataset.lifecycleAttached = 'true';
}

function handleGeospatialModalHidden() {
    if (geospatialModalConfirmed) {
        geospatialModalConfirmed = false;
        geospatialModalIsRerun = false;
        geospatialModalSelection = {
            latitude: '',
            longitude: '',
            label: '',
            proximityRadiusKm: '',
            referenceLatitude: '',
            referenceLongitude: '',
            comparisonLatitude: '',
            comparisonLongitude: '',
            optionalMode: 'none'
        };
        return;
    }

    if (geospatialModalIsRerun) {
        geospatialModalIsRerun = false;
        currentGeospatialAnalysisType = '';
        geospatialModalSelection = {
            latitude: '',
            longitude: '',
            label: '',
            proximityRadiusKm: '',
            referenceLatitude: '',
            referenceLongitude: '',
            comparisonLatitude: '',
            comparisonLongitude: ''
        };
        showNotification('Geospatial analysis rerun cancelled.', 'info');
        return;
    }

    if (currentGeospatialCellId) {
        const pendingCell = document.querySelector(`[data-cell-id="${currentGeospatialCellId}"]`);
        if (pendingCell) {
            pendingCell.remove();
            updateAnalysisResultsPlaceholder();
        }
        showNotification('Geospatial analysis cancelled.', 'info');
    }

    resetGeospatialModalState();
}

function resetGeospatialModalState() {
    currentGeospatialAnalysisType = '';
    currentGeospatialCellId = '';
    geospatialModalSelection = {
        latitude: '',
        longitude: '',
        label: '',
        proximityRadiusKm: '',
        referenceLatitude: '',
        referenceLongitude: '',
        comparisonLatitude: '',
        comparisonLongitude: '',
        optionalMode: 'none'
    };
    geospatialModalColumns = [];
    geospatialModalConfirmed = false;
    geospatialModalIsRerun = false;
    const radiusInput = document.getElementById('geospatialProximityRadiusInput');
    const referenceLatInput = document.getElementById('geospatialReferenceLatitudeInput');
    const referenceLonInput = document.getElementById('geospatialReferenceLongitudeInput');
    const comparisonLatSelect = document.getElementById('geospatialComparisonLatitudeSelect');
    const comparisonLonSelect = document.getElementById('geospatialComparisonLongitudeSelect');
    if (radiusInput) {
        radiusInput.value = '';
    }
    if (referenceLatInput) {
        referenceLatInput.value = '';
    }
    if (referenceLonInput) {
        referenceLonInput.value = '';
    }
    if (comparisonLatSelect) {
        comparisonLatSelect.value = '';
    }
    if (comparisonLonSelect) {
        comparisonLonSelect.value = '';
    }
    updateGeospatialOptionalModeUI('none');
    updateGeospatialAutoStatus('');
    updateGeospatialSelectionSummary();
}

function buildGeospatialColumnCandidates() {
    const candidates = [];
    const seen = new Set();

    const pushCandidate = (name, meta = {}) => {
        const normalized = typeof name === 'string' ? name.trim() : String(name || '').trim();
        if (!normalized || seen.has(normalized)) {
            return;
        }
        seen.add(normalized);
        const lowerName = normalized.toLowerCase();
        const dataType = typeof meta.dataType === 'string' ? meta.dataType : '';
        const dataCategory = typeof meta.dataCategory === 'string' ? meta.dataCategory : '';
        const numericLikely = typeof meta.numericLikely === 'boolean'
            ? meta.numericLikely
            : /int|float|double|decimal|number/.test(dataType) || dataCategory === 'numeric';
        const categoricalLikely = typeof meta.categoricalLikely === 'boolean'
            ? meta.categoricalLikely
            : dataCategory === 'categorical' || /string|object|category|text/.test(dataType);

        candidates.push({
            name: normalized,
            dataType,
            dataCategory,
            sampleValue: meta.sampleValue,
            isLatitudeHint: GEOSPATIAL_LAT_HINTS.some(hint => lowerName.includes(hint)),
            isLongitudeHint: GEOSPATIAL_LON_HINTS.some(hint => lowerName.includes(hint)),
            isLabelHint: GEOSPATIAL_LABEL_HINTS.some(hint => lowerName.includes(hint)),
            numericLikely,
            categoricalLikely
        });
    };

    if (Array.isArray(columnInsightsData?.column_insights)) {
        columnInsightsData.column_insights.forEach(col => {
            if (!col || col.dropped) {
                return;
            }
            const meta = {
                dataType: (col.data_type || '').toString().toLowerCase(),
                dataCategory: (col.data_category || '').toString().toLowerCase(),
                sampleValue: col.example_value || col.sample_value || null
            };
            pushCandidate(col.name, meta);
        });
    }

    if (candidates.length === 0) {
        let fallbackColumns = [];
        if (window.currentDataFrame && Array.isArray(window.currentDataFrame.columns)) {
            fallbackColumns = window.currentDataFrame.columns;
        } else {
            try {
                const storedColumns = sessionStorage.getItem('datasetColumns');
                if (storedColumns) {
                    const parsed = JSON.parse(storedColumns);
                    if (Array.isArray(parsed)) {
                        fallbackColumns = parsed;
                    }
                }
            } catch (storageError) {
                console.warn('Unable to parse stored dataset columns for geospatial modal:', storageError);
            }
        }

        fallbackColumns.forEach(name => pushCandidate(name));
    }

    candidates.sort((a, b) => {
        const scoreDiff = computeGeospatialScore(b) - computeGeospatialScore(a);
        if (scoreDiff !== 0) {
            return scoreDiff;
        }
        return a.name.localeCompare(b.name);
    });

    return candidates;
}

function computeGeospatialScore(column) {
    let score = 0;
    if (column.isLatitudeHint) {
        score += 12;
    }
    if (column.isLongitudeHint) {
        score += 12;
    }
    if (column.numericLikely) {
        score += 4;
    }
    if (column.isLabelHint) {
        score += 6;
    }
    if (column.categoricalLikely && !column.numericLikely) {
        score += 2;
    }
    return score;
}

function detectGeospatialRecommendations(columns, includeLabel = true) {
    const selection = { latitude: '', longitude: '', label: '' };
    const used = new Set();

    const isNumericCandidate = column => {
        if (column.numericLikely) {
            return true;
        }

        const type = typeof column.dataType === 'string' ? column.dataType : '';
        if (/int|float|double|decimal|number|degree/.test(type)) {
            return true;
        }

        if (typeof column.sampleValue === 'number') {
            return true;
        }
        if (typeof column.sampleValue === 'string') {
            const trimmed = column.sampleValue.trim();
            if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
                return true;
            }
        }

        return false;
    };

    const coordinatePool = columns.filter(col => isNumericCandidate(col) && !col.isLabelHint);

    const pickCoordinate = hintKey => {
        const hinted = coordinatePool.filter(col => col[hintKey] && !used.has(col.name));
        if (hinted.length) {
            used.add(hinted[0].name);
            return hinted[0].name;
        }

        const available = coordinatePool.filter(col => !used.has(col.name));
        if (available.length) {
            used.add(available[0].name);
            return available[0].name;
        }

        const fallbackHinted = columns.filter(col => col[hintKey] && !col.isLabelHint && !used.has(col.name));
        if (fallbackHinted.length) {
            used.add(fallbackHinted[0].name);
            return fallbackHinted[0].name;
        }

        const fallbackAny = columns.filter(col => !col.isLabelHint && !used.has(col.name));
        if (fallbackAny.length) {
            used.add(fallbackAny[0].name);
            return fallbackAny[0].name;
        }

        return '';
    };

    selection.latitude = pickCoordinate('isLatitudeHint');
    selection.longitude = pickCoordinate('isLongitudeHint');

    if (!selection.longitude && selection.latitude) {
        const remaining = coordinatePool.filter(col => !used.has(col.name));
        if (remaining.length) {
            selection.longitude = remaining[0].name;
            used.add(selection.longitude);
        }
    }

    if (includeLabel) {
        const labelCandidates = columns.filter(col =>
            !used.has(col.name) &&
            (col.isLabelHint || col.categoricalLikely)
        );
        if (labelCandidates.length) {
            selection.label = labelCandidates[0].name;
        }
    }

    return selection;
}

function toggleGeospatialLabelVisibility(supportsLabel) {
    const labelGroup = document.getElementById('geospatialLabelGroup');
    const labelSelect = document.getElementById('geospatialLabelSelect');

    if (labelGroup) {
        if (supportsLabel) {
            labelGroup.classList.remove('d-none');
        } else {
            labelGroup.classList.add('d-none');
        }
    }

    if (labelSelect) {
        if (supportsLabel) {
            labelSelect.disabled = false;
        } else {
            labelSelect.value = '';
            labelSelect.disabled = true;
        }
    }
}

function toggleGeospatialProximityOptions(analysisType) {
    const shouldShow = geospatialAnalysisUsesProximityOptions(analysisType);

    if (!shouldShow) {
        setGeospatialOptionalMode('none');
        const optionalGroup = document.getElementById('geospatialOptionalModeGroup');
        if (optionalGroup) {
            optionalGroup.classList.add('d-none');
        }
        return;
    }

    setGeospatialOptionalMode(geospatialModalSelection.optionalMode || 'none', { preserveValues: true });
}

function setGeospatialProximityInputs(selection = {}) {
    const radiusInput = document.getElementById('geospatialProximityRadiusInput');
    const referenceLatInput = document.getElementById('geospatialReferenceLatitudeInput');
    const referenceLonInput = document.getElementById('geospatialReferenceLongitudeInput');

    const radiusValue = selection.proximityRadiusKm !== undefined && selection.proximityRadiusKm !== null
        ? String(selection.proximityRadiusKm).trim()
        : (geospatialModalSelection.proximityRadiusKm || '');
    const refLatValue = selection.referenceLatitude !== undefined && selection.referenceLatitude !== null
        ? String(selection.referenceLatitude).trim()
        : (geospatialModalSelection.referenceLatitude || '');
    const refLonValue = selection.referenceLongitude !== undefined && selection.referenceLongitude !== null
        ? String(selection.referenceLongitude).trim()
        : (geospatialModalSelection.referenceLongitude || '');

    if (radiusInput) {
        radiusInput.value = radiusValue;
    }
    if (referenceLatInput) {
        referenceLatInput.value = refLatValue;
    }
    if (referenceLonInput) {
        referenceLonInput.value = refLonValue;
    }

    geospatialModalSelection = {
        ...geospatialModalSelection,
        proximityRadiusKm: radiusValue,
        referenceLatitude: refLatValue,
        referenceLongitude: refLonValue
    };

    updateGeospatialSelectionSummary();
}

function setGeospatialComparisonInputs(selection = {}) {
    const comparisonLatSelect = document.getElementById('geospatialComparisonLatitudeSelect');
    const comparisonLonSelect = document.getElementById('geospatialComparisonLongitudeSelect');

    const comparisonLatValue = selection.comparisonLatitude !== undefined && selection.comparisonLatitude !== null
        ? String(selection.comparisonLatitude).trim()
        : (geospatialModalSelection.comparisonLatitude || '');
    const comparisonLonValue = selection.comparisonLongitude !== undefined && selection.comparisonLongitude !== null
        ? String(selection.comparisonLongitude).trim()
        : (geospatialModalSelection.comparisonLongitude || '');

    if (comparisonLatSelect) {
        comparisonLatSelect.value = comparisonLatValue;
    }
    if (comparisonLonSelect) {
        comparisonLonSelect.value = comparisonLonValue;
    }

    geospatialModalSelection = {
        ...geospatialModalSelection,
        comparisonLatitude: comparisonLatValue,
        comparisonLongitude: comparisonLonValue
    };

    updateGeospatialSelectionSummary();
}

function updateGeospatialOptionalModeUI(mode) {
    const normalized = mode === 'radius' || mode === 'comparison' ? mode : 'none';
    const optionalGroup = document.getElementById('geospatialOptionalModeGroup');
    if (optionalGroup) {
        const shouldShow = geospatialAnalysisUsesProximityOptions(currentGeospatialAnalysisType);
        optionalGroup.classList.toggle('d-none', !shouldShow);
    }

    const proximityCard = document.getElementById('geospatialProximityConfig');
    const comparisonCard = document.getElementById('geospatialComparisonConfig');
    if (proximityCard) {
        proximityCard.classList.toggle('d-none', normalized !== 'radius');
    }
    if (comparisonCard) {
        comparisonCard.classList.toggle('d-none', normalized !== 'comparison');
    }

    const modeNone = document.getElementById('geospatialOptionalModeNone');
    const modeRadius = document.getElementById('geospatialOptionalModeRadius');
    const modeComparison = document.getElementById('geospatialOptionalModeComparison');

    if (modeNone) {
        modeNone.checked = normalized === 'none';
    }
    if (modeRadius) {
        modeRadius.checked = normalized === 'radius';
    }
    if (modeComparison) {
        modeComparison.checked = normalized === 'comparison';
    }
}

function setGeospatialOptionalMode(mode, options = {}) {
    const normalized = mode === 'radius' || mode === 'comparison' ? mode : 'none';
    const previousMode = geospatialModalSelection.optionalMode;
    geospatialModalSelection = {
        ...geospatialModalSelection,
        optionalMode: normalized
    };

    updateGeospatialOptionalModeUI(normalized);

    const preserveValues = !!options.preserveValues;

    if (!preserveValues) {
        if (normalized !== 'radius') {
            setGeospatialProximityInputs({ proximityRadiusKm: '', referenceLatitude: '', referenceLongitude: '' });
        }
        if (normalized !== 'comparison') {
            setGeospatialComparisonInputs({ comparisonLatitude: '', comparisonLongitude: '' });
        }
        if (normalized === 'none') {
            updateGeospatialSelectionSummary();
        }
        return;
    }

    updateGeospatialSelectionSummary();
}

function populateGeospatialModal(analysisType, columns, initialSelection = null, supportsLabelOverride = null) {
    const modalLabel = document.getElementById('geospatialAnalysisModalLabel');
    if (modalLabel) {
        modalLabel.innerHTML = `<i class="bi bi-geo-alt"></i> <span>Configure ${getAnalysisTypeName(analysisType)}</span>`;
    }

    const subtitle = document.getElementById('geospatialModalSubtitle');
    if (subtitle) {
        if (analysisType === 'spatial_data_quality_analysis') {
            subtitle.textContent = 'Review coordinate completeness, validity, and duplication issues for your dataset.';
        } else if (analysisType === GEOSPATIAL_PROXIMITY_ANALYSIS_ID) {
            subtitle.textContent = 'Select coordinates, then optionally add a radius or manual reference point to measure proximity.';
        } else {
            subtitle.textContent = 'Choose latitude and longitude columns so maps and proximity metrics can run accurately.';
        }
    }

    toggleGeospatialProximityOptions(analysisType);
    if (geospatialAnalysisUsesProximityOptions(analysisType)) {
        const proximitySelection = initialSelection || {};
        const initialMode = proximitySelection.optionalMode || geospatialModalSelection.optionalMode || 'none';
        geospatialModalSelection = {
            ...geospatialModalSelection,
            optionalMode: initialMode
        };
        setGeospatialOptionalMode(initialMode, { preserveValues: true });
        setGeospatialProximityInputs({
            proximityRadiusKm: proximitySelection.proximityRadiusKm ?? proximitySelection.radiusKm ?? '',
            referenceLatitude: proximitySelection.referenceLatitude ?? '',
            referenceLongitude: proximitySelection.referenceLongitude ?? ''
        });
        setGeospatialComparisonInputs({
            comparisonLatitude: proximitySelection.comparisonLatitude ?? '',
            comparisonLongitude: proximitySelection.comparisonLongitude ?? ''
        });
    }

    const latSelect = document.getElementById('geospatialLatitudeSelect');
    const lonSelect = document.getElementById('geospatialLongitudeSelect');
    const labelSelect = document.getElementById('geospatialLabelSelect');
    const comparisonLatSelect = document.getElementById('geospatialComparisonLatitudeSelect');
    const comparisonLonSelect = document.getElementById('geospatialComparisonLongitudeSelect');

    const supportsLabel = typeof supportsLabelOverride === 'boolean'
        ? supportsLabelOverride
        : geospatialAnalysisSupportsLabel(analysisType);

    const defaults = initialSelection || { latitude: '', longitude: '', label: '' };
    geospatialModalSelection = {
        ...geospatialModalSelection,
        latitude: defaults.latitude || '',
        longitude: defaults.longitude || '',
        label: supportsLabel ? (defaults.label || '') : ''
    };

    toggleGeospatialLabelVisibility(supportsLabel);

    populateGeospatialSelect(latSelect, columns, geospatialModalSelection.latitude, 'latitude');
    populateGeospatialSelect(lonSelect, columns, geospatialModalSelection.longitude, 'longitude');

    if (supportsLabel) {
        populateGeospatialSelect(labelSelect, columns, geospatialModalSelection.label, 'label');
    } else if (labelSelect) {
        labelSelect.innerHTML = '<option value="">Label not used for this analysis</option>';
        labelSelect.value = '';
    }

    populateGeospatialSelect(comparisonLatSelect, columns, geospatialModalSelection.comparisonLatitude, 'comparison-latitude');
    populateGeospatialSelect(comparisonLonSelect, columns, geospatialModalSelection.comparisonLongitude, 'comparison-longitude');

    updateGeospatialSelectionSummary();
    updateGeospatialAutoStatus('');
}

function populateGeospatialSelect(selectElement, columns, selectedValue, role) {
    if (!selectElement) {
        return;
    }

    const options = [];

    if (role === 'label') {
        options.push('<option value="">None (skip label)</option>');
    } else if (role === 'comparison-latitude') {
        options.push('<option value="">None (skip comparison latitude)</option>');
    } else if (role === 'comparison-longitude') {
        options.push('<option value="">None (skip comparison longitude)</option>');
    } else {
        const placeholder = role === 'latitude' ? 'Select latitude column' : 'Select longitude column';
        options.push(`<option value="">${placeholder}</option>`);
    }

    columns.forEach(col => {
        const hints = [];
        if (col.isLatitudeHint) {
            hints.push('lat');
        }
        if (col.isLongitudeHint) {
            hints.push('lon');
        }
        if (col.isLabelHint) {
            hints.push('label');
        }

        const hintText = hints.length ? ` • ${hints.join(', ')}` : '';
        const typeText = col.dataType ? ` (${col.dataType})` : '';
        const selectedAttr = selectedValue && col.name === selectedValue ? ' selected' : '';

        options.push(`
            <option value="${escapeHtml(col.name)}"${selectedAttr}>
                ${escapeHtml(col.name)}${hintText}${typeText}
            </option>
        `);
    });

    selectElement.innerHTML = options.join('');

    if (selectedValue) {
        selectElement.value = selectedValue;
    } else if (role === 'label' || role === 'comparison-latitude' || role === 'comparison-longitude') {
        selectElement.value = '';
    }
}

function applyGeospatialSelection(selection, showMessage = false, supportsLabelOverride = null) {
    const supportsLabel = typeof supportsLabelOverride === 'boolean'
        ? supportsLabelOverride
        : geospatialAnalysisSupportsLabel(currentGeospatialAnalysisType);

    geospatialModalSelection = {
        ...geospatialModalSelection,
        latitude: selection.latitude || '',
        longitude: selection.longitude || '',
        label: supportsLabel ? (selection.label || '') : ''
    };

    const latSelect = document.getElementById('geospatialLatitudeSelect');
    const lonSelect = document.getElementById('geospatialLongitudeSelect');
    const labelSelect = document.getElementById('geospatialLabelSelect');

    toggleGeospatialLabelVisibility(supportsLabel);

    if (latSelect) {
        latSelect.value = geospatialModalSelection.latitude;
    }
    if (lonSelect) {
        lonSelect.value = geospatialModalSelection.longitude;
    }
    if (labelSelect) {
        labelSelect.value = supportsLabel ? (geospatialModalSelection.label || '') : '';
    }

    updateGeospatialSelectionSummary();

    if (showMessage) {
        if (geospatialModalSelection.latitude && geospatialModalSelection.longitude) {
            updateGeospatialAutoStatus(`Detected ${geospatialModalSelection.latitude} as latitude and ${geospatialModalSelection.longitude} as longitude.`, 'success');
        } else {
            updateGeospatialAutoStatus('Unable to auto-detect both latitude and longitude.', 'warning');
        }
    }
}

function updateGeospatialSelectionSummary() {
    const summaryElement = document.getElementById('geospatialSelectionSummary');
    const confirmBtn = document.getElementById('geospatialModalConfirmBtn');

    if (!summaryElement) {
        return;
    }

    const {
        latitude,
        longitude,
        label,
        proximityRadiusKm,
        referenceLatitude,
        referenceLongitude,
        comparisonLatitude,
        comparisonLongitude,
        optionalMode
    } = geospatialModalSelection;
    const supportsLabel = geospatialAnalysisSupportsLabel(currentGeospatialAnalysisType);
    const effectiveOptionalMode = optionalMode === 'radius' || optionalMode === 'comparison' ? optionalMode : 'none';

    summaryElement.classList.remove('alert-success', 'alert-warning', 'alert-secondary');

    if (!latitude || !longitude) {
        summaryElement.classList.add('alert-secondary');
        summaryElement.textContent = 'Select latitude and longitude columns to continue.';
        if (confirmBtn) {
            confirmBtn.disabled = true;
        }
        return;
    }

    if (latitude === longitude) {
        summaryElement.classList.add('alert-warning');
        summaryElement.textContent = 'Latitude and longitude must be different columns.';
        if (confirmBtn) {
            confirmBtn.disabled = true;
        }
        return;
    }

    const comparisonSelected = comparisonLatitude && comparisonLongitude;
    if (effectiveOptionalMode === 'comparison' && !comparisonSelected) {
        summaryElement.classList.add('alert-warning');
        summaryElement.textContent = 'Select both comparison latitude and longitude columns to focus on coordinate differences.';
        if (confirmBtn) {
            confirmBtn.disabled = true;
        }
        return;
    }

    if (comparisonLatitude && !comparisonLongitude) {
        summaryElement.classList.add('alert-warning');
        summaryElement.textContent = 'Provide both comparison latitude and longitude columns to calculate differences, or clear both fields.';
        if (confirmBtn) {
            confirmBtn.disabled = true;
        }
        return;
    }

    if (!comparisonLatitude && comparisonLongitude) {
        summaryElement.classList.add('alert-warning');
        summaryElement.textContent = 'Provide both comparison latitude and longitude columns to calculate differences, or clear both fields.';
        if (confirmBtn) {
            confirmBtn.disabled = true;
        }
        return;
    }

    summaryElement.classList.add('alert-success');
    const detailParts = [`${latitude} (latitude)`, `${longitude} (longitude)`];
    if (supportsLabel && label) {
        detailParts.push(`${label} (label)`);
    }
    if (currentGeospatialAnalysisType === GEOSPATIAL_PROXIMITY_ANALYSIS_ID) {
        if (proximityRadiusKm) {
            detailParts.push(`${proximityRadiusKm} km radius`);
        }
        if (referenceLatitude && referenceLongitude) {
            detailParts.push(`reference (${referenceLatitude}, ${referenceLongitude})`);
        }
        if (comparisonLatitude && comparisonLongitude) {
            detailParts.push(`comparison vs ${comparisonLatitude}/${comparisonLongitude}`);
        }
    }

    summaryElement.textContent = `Ready to run with ${detailParts.join(', ')}.`;

    if (confirmBtn) {
        confirmBtn.disabled = false;
    }
}

function updateGeospatialAutoStatus(message, tone = 'muted') {
    const statusElement = document.getElementById('geospatialModalAutoStatus');
    if (!statusElement) {
        return;
    }

    statusElement.textContent = message || '';
    statusElement.classList.remove('text-success', 'text-warning', 'text-danger', 'text-muted');

    if (!message) {
        statusElement.classList.add('text-muted');
        return;
    }

    if (tone === 'success') {
        statusElement.classList.add('text-success');
    } else if (tone === 'warning') {
        statusElement.classList.add('text-warning');
    } else if (tone === 'danger') {
        statusElement.classList.add('text-danger');
    } else {
        statusElement.classList.add('text-muted');
    }
}

async function launchGeospatialAnalysis() {
    if (!currentGeospatialAnalysisType) {
        showNotification('No geospatial analysis selected.', 'error');
        return;
    }

    const {
        latitude,
        longitude,
        label,
        proximityRadiusKm,
        referenceLatitude,
        referenceLongitude,
        comparisonLatitude,
        comparisonLongitude,
        optionalMode
    } = geospatialModalSelection;

    if (!latitude || !longitude) {
        showNotification('Select both latitude and longitude columns to continue.', 'warning');
        return;
    }

    if (latitude === longitude) {
        showNotification('Latitude and longitude must reference different columns.', 'warning');
        return;
    }

    if ((comparisonLatitude && !comparisonLongitude) || (!comparisonLatitude && comparisonLongitude)) {
        showNotification('Provide both comparison latitude and longitude columns or leave both blank.', 'warning');
        return;
    }

    const confirmBtn = document.getElementById('geospatialModalConfirmBtn');
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

    geospatialModalConfirmed = true;
    const modalElement = document.getElementById('geospatialAnalysisModal');
    const modalInstance = modalElement ? bootstrap.Modal.getInstance(modalElement) : null;
    if (modalInstance) {
        modalInstance.hide();
    }

    try {
        const supportsLabel = geospatialAnalysisSupportsLabel(currentGeospatialAnalysisType);
        if (!currentGeospatialCellId) {
            const fallbackCellId = await addSingleAnalysisCell(currentGeospatialAnalysisType, {
                skipGeospatialModal: true
            });
            currentGeospatialCellId = fallbackCellId || '';
        }

        if (!currentGeospatialCellId) {
            showNotification('Unable to start geospatial analysis: no analysis cell available.', 'error');
            return;
        }

        const includeLabel = supportsLabel && !!label;
        const selectedColumns = includeLabel ? [latitude, longitude, label] : [latitude, longitude];
        const columnMapping = {
            latitude,
            longitude
        };
        if (includeLabel) {
            columnMapping.location_label = label;
        }

        const selectionDetails = { latitude, longitude };
        if (includeLabel) {
            selectionDetails.label = label;
        }
        if (currentGeospatialAnalysisType === GEOSPATIAL_PROXIMITY_ANALYSIS_ID) {
            selectionDetails.proximityRadiusKm = proximityRadiusKm || '';
            selectionDetails.referenceLatitude = referenceLatitude || '';
            selectionDetails.referenceLongitude = referenceLongitude || '';
            selectionDetails.comparisonLatitude = comparisonLatitude || '';
            selectionDetails.comparisonLongitude = comparisonLongitude || '';
            selectionDetails.optionalMode = optionalMode || 'none';
        }

        const analysisOptions = {
            overrideSelectedColumns: selectedColumns,
            includeGlobalSelectedColumns: false,
            modalType: 'geospatial',
            modalSelectionPayload: selectionDetails
        };

        if (currentGeospatialAnalysisType === GEOSPATIAL_PROXIMITY_ANALYSIS_ID) {
            const metadata = {};
            const parsedRadius = parseFloat(proximityRadiusKm);
            if (!Number.isNaN(parsedRadius) && Number.isFinite(parsedRadius) && parsedRadius > 0) {
                metadata.proximity_radius_km = parsedRadius;
            }
            const parsedRefLat = parseFloat(referenceLatitude);
            const parsedRefLon = parseFloat(referenceLongitude);
            if (!Number.isNaN(parsedRefLat) && Number.isFinite(parsedRefLat) && !Number.isNaN(parsedRefLon) && Number.isFinite(parsedRefLon)) {
                metadata.reference_latitude = parsedRefLat;
                metadata.reference_longitude = parsedRefLon;
            }
            if (comparisonLatitude && comparisonLongitude) {
                metadata.comparison_latitude_column = comparisonLatitude;
                metadata.comparison_longitude_column = comparisonLongitude;
            }
            const normalizedOptionalMode = optionalMode === 'comparison' || optionalMode === 'radius' ? optionalMode : 'none';
            if (normalizedOptionalMode !== 'none') {
                metadata.proximity_optional_mode = normalizedOptionalMode;
            }
            if (Object.keys(metadata).length > 0) {
                analysisOptions.analysisMetadata = metadata;
            }
        }

        showNotification(`Running ${getAnalysisTypeName(currentGeospatialAnalysisType)} with selected coordinates.`, 'success');

        await generateAndRunAnalysis(
            currentGeospatialCellId,
            currentGeospatialAnalysisType,
            columnMapping,
            analysisOptions
        );
    } catch (error) {
        console.error('Geospatial analysis execution failed:', error);
        showNotification(`Geospatial analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.removeAttribute('aria-busy');
            if (previousLabel !== null) {
                confirmBtn.innerHTML = previousLabel;
            }
        }

        resetGeospatialModalState();
    }
}

async function openGeospatialModalForRerun(cellId, analysisType, previousSelection = [], previousDetails = null) {
    geospatialModalIsRerun = true;
    geospatialModalConfirmed = false;
    currentGeospatialCellId = cellId;
    currentGeospatialAnalysisType = analysisType;

    if (!columnInsightsData || !Array.isArray(columnInsightsData.column_insights)) {
        try {
            await loadColumnInsights();
        } catch (error) {
            console.warn('Unable to refresh column insights before geospatial rerun:', error);
        }
    }

    geospatialModalColumns = buildGeospatialColumnCandidates();

    if (geospatialModalColumns.length < 2) {
        geospatialModalIsRerun = false;
        showNotification('Not enough columns available to reconfigure geospatial analysis.', 'warning');
        return;
    }

    const supportsLabel = geospatialAnalysisSupportsLabel(analysisType);
    const initialSelection = rebuildGeospatialSelectionFromHistory(previousSelection, previousDetails, supportsLabel);

    initializeGeospatialModal();
    attachGeospatialModalLifecycleHandlers();
    populateGeospatialModal(analysisType, geospatialModalColumns, initialSelection, supportsLabel);

    const modalElement = document.getElementById('geospatialAnalysisModal');
    if (!modalElement) {
        geospatialModalIsRerun = false;
        showNotification('Geospatial configuration modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    showNotification('Adjust the coordinate columns if needed, then rerun the analysis.', 'info');
}

function rebuildGeospatialSelectionFromHistory(previousSelection, previousDetails, includeLabel = true) {
    const selection = {
        latitude: '',
        longitude: '',
        label: '',
        proximityRadiusKm: '',
        referenceLatitude: '',
        referenceLongitude: '',
        comparisonLatitude: '',
        comparisonLongitude: '',
        optionalMode: 'none'
    };

    const includeProximity = geospatialAnalysisUsesProximityOptions(currentGeospatialAnalysisType);

    if (previousDetails && typeof previousDetails === 'object') {
        selection.latitude = typeof previousDetails.latitude === 'string' ? previousDetails.latitude : '';
        selection.longitude = typeof previousDetails.longitude === 'string' ? previousDetails.longitude : '';
        if (includeLabel) {
            selection.label = typeof previousDetails.label === 'string' ? previousDetails.label : '';
        }
        if (includeProximity) {
            if (typeof previousDetails.proximityRadiusKm === 'string' || typeof previousDetails.proximityRadiusKm === 'number') {
                selection.proximityRadiusKm = String(previousDetails.proximityRadiusKm).trim();
            }
            if (typeof previousDetails.referenceLatitude === 'string' || typeof previousDetails.referenceLatitude === 'number') {
                selection.referenceLatitude = String(previousDetails.referenceLatitude).trim();
            }
            if (typeof previousDetails.referenceLongitude === 'string' || typeof previousDetails.referenceLongitude === 'number') {
                selection.referenceLongitude = String(previousDetails.referenceLongitude).trim();
            }
            if (typeof previousDetails.comparisonLatitude === 'string') {
                selection.comparisonLatitude = previousDetails.comparisonLatitude.trim();
            }
            if (typeof previousDetails.comparisonLongitude === 'string') {
                selection.comparisonLongitude = previousDetails.comparisonLongitude.trim();
            }
            if (typeof previousDetails.optionalMode === 'string') {
                selection.optionalMode = previousDetails.optionalMode.trim();
            } else if (typeof previousDetails.optional_mode === 'string') {
                selection.optionalMode = previousDetails.optional_mode.trim();
            }
            if (previousDetails.comparison && typeof previousDetails.comparison === 'object') {
                const comparisonColumns = previousDetails.comparison.columns || {};
                if (!selection.comparisonLatitude && typeof comparisonColumns.latitude === 'string') {
                    selection.comparisonLatitude = comparisonColumns.latitude.trim();
                }
                if (!selection.comparisonLongitude && typeof comparisonColumns.longitude === 'string') {
                    selection.comparisonLongitude = comparisonColumns.longitude.trim();
                }
                if (!selection.optionalMode || selection.optionalMode === 'none') {
                    selection.optionalMode = 'comparison';
                }
            }
            if (!selection.optionalMode || selection.optionalMode === 'none') {
                const meta = previousDetails.metadata;
                if (meta && typeof meta === 'object' && typeof meta.proximity_optional_mode === 'string') {
                    selection.optionalMode = meta.proximity_optional_mode.trim();
                }
            }
        }
    }

    if ((!selection.latitude || !selection.longitude || (includeLabel && !selection.label)) && Array.isArray(previousSelection)) {
        const normalized = previousSelection
            .map(name => (typeof name === 'string' ? name.trim() : String(name || '')))
            .filter(Boolean);
        if (!selection.latitude && normalized.length > 0) {
            selection.latitude = normalized[0];
        }
        if (!selection.longitude && normalized.length > 1) {
            selection.longitude = normalized[1];
        }
        if (includeLabel && !selection.label && normalized.length > 2) {
            selection.label = normalized[2];
        }
    }

    if (!selection.latitude || !selection.longitude || (includeLabel && !selection.label)) {
        const recommendations = detectGeospatialRecommendations(geospatialModalColumns, includeLabel);
        if (!selection.latitude) {
            selection.latitude = recommendations.latitude;
        }
        if (!selection.longitude) {
            selection.longitude = recommendations.longitude;
        }
        if (includeLabel && !selection.label) {
            selection.label = recommendations.label;
        }
    }

    if (!includeLabel) {
        selection.label = '';
    }

    if (!includeProximity) {
        selection.proximityRadiusKm = '';
        selection.referenceLatitude = '';
        selection.referenceLongitude = '';
        selection.comparisonLatitude = '';
        selection.comparisonLongitude = '';
        selection.optionalMode = 'none';
    }

    if (selection.optionalMode !== 'radius' && selection.optionalMode !== 'comparison') {
        selection.optionalMode = 'none';
    }

    return selection;
}
