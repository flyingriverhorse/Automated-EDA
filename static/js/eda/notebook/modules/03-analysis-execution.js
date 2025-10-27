async function addSelectedAnalysisCells() {
    console.log('addSelectedAnalysisCells called with selections:', selectedAnalysisTypes);
    
    if (selectedAnalysisTypes.size === 0) {
        showNotification('Please select analysis types from the categories above', 'info');
        return;
    }
    
    // Check if any marketing analyses are selected
    const marketingAnalyses = [];
    const nonMarketingAnalyses = [];
    
    for (const analysisType of selectedAnalysisTypes) {
        if (MARKETING_ANALYSIS_CONFIG[analysisType]) {
            marketingAnalyses.push(analysisType);
        } else {
            nonMarketingAnalyses.push(analysisType);
        }
    }
    
    // If there are marketing analyses, handle them specially
    if (marketingAnalyses.length > 0) {
        if (marketingAnalyses.length === 1 && selectedAnalysisTypes.size === 1) {
            // Single marketing analysis - show modal directly
            await prepareMarketingAnalysisConfiguration(marketingAnalyses[0]);
        } else {
            // Multiple analyses with at least one marketing - show informative message
            showNotification(
                `Marketing analyses (${marketingAnalyses.map(a => getAnalysisTypeName(a)).join(', ')}) require column mapping configuration. Please add them individually for the best experience.`,
                'info',
                8000
            );
            
            // Process only non-marketing analyses for bulk add
            if (nonMarketingAnalyses.length > 0) {
                let addedCount = 0;
                const errors = [];
                
                for (const analysisType of nonMarketingAnalyses) {
                    try {
                        console.log(`Adding analysis result for: ${analysisType}`);
                        await addSingleAnalysisCell(analysisType);
                        addedCount++;
                    } catch (error) {
                        console.error(`Failed to add analysis result for ${analysisType}:`, error);
                        errors.push(`${getAnalysisTypeName(analysisType)}: ${error.message}`);
                    }
                }
                
                if (addedCount > 0) {
                    showNotification(`Added ${addedCount} non-marketing analysis result(s). Add marketing analyses individually.`, 'success');
                }
                
                if (errors.length > 0) {
                    showNotification(`Some analyses failed to add: ${errors.join(', ')}`, 'warning');
                }
            }
        }
        
        // Don't clear selections if marketing analyses need to be added individually
        if (marketingAnalyses.length === 1 && selectedAnalysisTypes.size === 1) {
            clearAllSelections();
        }
        
        return;
    }
    
    // No marketing analyses - process normally
    let addedCount = 0;
    const errors = [];
    
    // Process each selected analysis type
    for (const analysisType of selectedAnalysisTypes) {
        try {
            console.log(`Adding analysis result for: ${analysisType}`);
            await addSingleAnalysisCell(analysisType);
            addedCount++;
        } catch (error) {
            console.error(`Failed to add analysis result for ${analysisType}:`, error);
            errors.push(`${getAnalysisTypeName(analysisType)}: ${error.message}`);
        }
    }
    
    // Show results
    if (addedCount > 0) {
        showNotification(`Added ${addedCount} analysis result(s)`, 'success');
        
        // Clear selections after successful addition
        clearAllSelections();
    }
    
    if (errors.length > 0) {
        showNotification(`Some analyses failed to add: ${errors.join(', ')}`, 'warning');
    }
}

(function initializeEDAChatBridge() {
    const existingBridge = typeof window.EDAChatBridge === 'object' && window.EDAChatBridge !== null
        ? window.EDAChatBridge
        : {};

    const analysisCells = existingBridge.__analysisCells instanceof Map ? existingBridge.__analysisCells : new Map();
    const customCells = existingBridge.__customCells instanceof Map ? existingBridge.__customCells : new Map();

    let activeContext = existingBridge.__activeContext || null;
    let notebookMeta = existingBridge.__notebookMeta || null;
    let lastUpdate = existingBridge.lastUpdate || null;

    const MAX_TEXT_LENGTH = 6000;
    const MAX_PREVIEW_LENGTH = 180;
    const MAX_TABLE_ROWS = 10;
    const MAX_TABLES = 3;
    const MAX_CHARTS = 4;

    function truncateText(value, maxLength = MAX_TEXT_LENGTH) {
        if (typeof value !== 'string') {
            return value;
        }
        if (value.length <= maxLength) {
            return value;
        }
        return `${value.slice(0, maxLength)}…`;
    }

    function buildMetricSummary(structuredResults) {
        if (!Array.isArray(structuredResults) || structuredResults.length === 0) {
            return null;
        }
        const primaryResult = structuredResults[0] || {};
        const metrics = Array.isArray(primaryResult.metrics) ? primaryResult.metrics : [];
        if (metrics.length === 0) {
            return null;
        }
        const metricHighlights = metrics.slice(0, 3).map(metric => {
            const label = metric.label || metric.name || 'Metric';
            const value = metric.value;
            const unit = metric.unit ? ` ${metric.unit}` : '';
            if (typeof value === 'number') {
                return `${label}: ${Number(value.toFixed(4))}${unit}`;
            }
            return `${label}: ${value}${unit}`;
        });
        return metricHighlights.join('; ');
    }

    function buildInsightSummary(structuredResults) {
        if (!Array.isArray(structuredResults) || structuredResults.length === 0) {
            return null;
        }
        const primaryResult = structuredResults[0] || {};
        const insights = Array.isArray(primaryResult.insights) ? primaryResult.insights : [];
        if (insights.length === 0) {
            return null;
        }
        const highlights = insights
            .filter(insight => insight && insight.text)
            .slice(0, 2)
            .map(insight => {
                const level = insight.level ? `${insight.level.toUpperCase()}: ` : '';
                return `${level}${truncateText(insight.text, 300)}`;
            });
        return highlights.length ? highlights.join(' | ') : null;
    }

    function buildAnalysisLLMSummary(entry) {
        if (!entry) {
            return null;
        }
        const parts = [];
        const metaSummary = truncateText(entry.metaSummary, 600);
        if (metaSummary) {
            parts.push(metaSummary);
        }

        const metricSummary = buildMetricSummary(entry.structuredResults);
        if (metricSummary) {
            parts.push(`Key metrics: ${metricSummary}`);
        }

        const insightSummary = buildInsightSummary(entry.structuredResults);
        if (insightSummary) {
            parts.push(`Insights: ${insightSummary}`);
        }

        const request = entry.request || {};
        if (!parts.length && Array.isArray(request.selected_columns) && request.selected_columns.length) {
            const preview = request.selected_columns.slice(0, 5);
            const remainder = request.selected_columns.length - preview.length;
            parts.push(`Columns analysed: ${preview.join(', ')}${remainder > 0 ? ` (+${remainder})` : ''}`);
        }

        const legacyStdout = entry.legacyOutput?.stdout;
        if (legacyStdout) {
            parts.push(`Console: ${truncateText(legacyStdout, 400)}`);
        }

        const warnings = entry.responsePreview?.warnings;
        if (warnings) {
            const warningText = Array.isArray(warnings) ? warnings.join(' | ') : warnings;
            parts.push(`Warnings: ${truncateText(warningText, 400)}`);
        }

        return parts.length ? parts.join('\n') : null;
    }

    function buildCustomLLMSummary(entry) {
        if (!entry) {
            return null;
        }
        const parts = [];
        if (entry.textOutput) {
            parts.push(truncateText(entry.textOutput, 600));
        }
        if (entry.error) {
            parts.push(`Error: ${truncateText(entry.error, 400)}`);
        }
        if (entry.plotCount) {
            parts.push(`Plots generated: ${entry.plotCount}`);
        }
        if (entry.responsePreview?.rate_limit_info) {
            parts.push('Rate limit info available for review.');
        }
        return parts.length ? parts.join('\n') : null;
    }

    function buildNotebookSummary() {
        const dataset = getDatasetInfo();
        const analysisCount = analysisCells.size;
        const customCount = customCells.size;
        const parts = [];

        if (dataset.name) {
            parts.push(`Dataset: ${dataset.name}`);
        }
        if (analysisCount) {
            parts.push(`Analyses run: ${analysisCount}`);
        }
        if (customCount) {
            parts.push(`Custom cells run: ${customCount}`);
        }
        if (activeContext?.cellId) {
            parts.push(`Focus on ${activeContext.scope} cell ${activeContext.cellId}`);
        }
        if (lastUpdate) {
            try {
                const timestamp = new Date(lastUpdate);
                if (!Number.isNaN(timestamp.valueOf())) {
                    parts.push(`Context updated ${timestamp.toLocaleString()}`);
                }
            } catch (e) {
                // ignore timestamp parsing issues
            }
        }

        return parts.join(' • ');
    }

    function buildAnalysisHighlights(limit = 4) {
        const sortedEntries = Array.from(analysisCells.values()).sort((a, b) => {
            const aTime = a.completedAt || a.startedAt || '';
            const bTime = b.completedAt || b.startedAt || '';
            if (aTime === bTime) {
                return 0;
            }
            return aTime > bTime ? -1 : 1;
        });
        return sortedEntries.slice(0, limit).map(entry => ({
            cellId: entry.cellId,
            analysisType: entry.analysisType,
            analysisName: entry.analysisName,
            status: entry.status,
            summary: truncateText(entry.llmSummary || buildAnalysisLLMSummary(entry), 800)
        }));
    }

    function buildCustomHighlights(limit = 3) {
        const sortedEntries = Array.from(customCells.values()).sort((a, b) => {
            const aTime = a.completedAt || a.startedAt || '';
            const bTime = b.completedAt || b.startedAt || '';
            if (aTime === bTime) {
                return 0;
            }
            return aTime > bTime ? -1 : 1;
        });
        return sortedEntries.slice(0, limit).map(entry => ({
            cellId: entry.cellId,
            status: entry.status,
            summary: truncateText(entry.llmSummary || buildCustomLLMSummary(entry), 600)
        }));
    }

    function sanitizeStructuredResults(results) {
        if (!Array.isArray(results)) {
            return [];
        }
        return results.slice(0, 6).map(result => {
            const sanitized = {
                analysis_id: result.analysis_id || result.id || null,
                title: result.title || null,
                summary: truncateText(result.summary, 1000),
                status: result.status || 'success'
            };

            if (Array.isArray(result.metrics)) {
                sanitized.metrics = result.metrics.slice(0, 6).map(metric => ({
                    label: metric.label || metric.name || '',
                    value: typeof metric.value === 'number' ? metric.value : truncateText(String(metric.value), 200),
                    unit: metric.unit || null,
                    description: truncateText(metric.description, 400),
                    trend: metric.trend || null
                }));
            }

            if (Array.isArray(result.insights)) {
                sanitized.insights = result.insights.slice(0, 8).map(insight => ({
                    level: insight.level || 'info',
                    text: truncateText(insight.text, 600)
                }));
            }

            if (Array.isArray(result.tables)) {
                sanitized.tables = result.tables.slice(0, MAX_TABLES).map(table => ({
                    title: table.title || null,
                    description: truncateText(table.description, 400),
                    columns: Array.isArray(table.columns) ? table.columns.slice(0, 12) : [],
                    rows: Array.isArray(table.rows)
                        ? table.rows.slice(0, MAX_TABLE_ROWS).map(row => {
                            const rowCopy = {};
                            Object.entries(row || {}).forEach(([key, value]) => {
                                rowCopy[key] = typeof value === 'number' ? value : truncateText(String(value), 300);
                            });
                            return rowCopy;
                        })
                        : []
                }));
            }

            if (Array.isArray(result.charts)) {
                sanitized.charts = result.charts.slice(0, MAX_CHARTS).map(chart => ({
                    title: chart.title || null,
                    description: truncateText(chart.description, 400),
                    hasImage: Boolean(chart.image),
                    imagePreview: chart.image ? truncateText(chart.image, MAX_PREVIEW_LENGTH) : null
                }));
            }

            if (result.details && typeof result.details === 'object') {
                try {
                    sanitized.details = JSON.parse(JSON.stringify(result.details));
                } catch (detailError) {
                    sanitized.details = result.details;
                }
            }

            return sanitized;
        });
    }

    function sanitizeLegacyOutput(output) {
        if (!output || typeof output !== 'object') {
            return null;
        }

        const sanitized = {};
        if (output.stdout) {
            sanitized.stdout = truncateText(output.stdout, MAX_TEXT_LENGTH);
        }
        if (output.stderr) {
            sanitized.stderr = truncateText(output.stderr, 2000);
        }
        if (Array.isArray(output.data_frames) && output.data_frames.length > 0) {
            sanitized.data_frames = output.data_frames.slice(0, 2).map(frame => ({
                name: frame.name || null,
                preview: frame.html ? truncateText(frame.html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim(), 1000) : null
            }));
        }
        if (Array.isArray(output.plots) && output.plots.length > 0) {
            sanitized.plots = {
                count: output.plots.length,
                preview: truncateText(String(output.plots[0]), MAX_PREVIEW_LENGTH)
            };
        }
        return sanitized;
    }

    function sanitizeRequestPayload(payload) {
        if (!payload || typeof payload !== 'object') {
            return null;
        }
        const sanitized = {
            analysis_ids: Array.isArray(payload.analysis_ids) ? [...payload.analysis_ids] : null,
            source_id: payload.source_id || null
        };
        if (Array.isArray(payload.selected_columns)) {
            sanitized.selected_columns = payload.selected_columns.slice(0, 100);
        }
        if (payload.column_mapping && typeof payload.column_mapping === 'object') {
            sanitized.column_mapping = payload.column_mapping;
        }
        if (payload.preprocessing && typeof payload.preprocessing === 'object') {
            sanitized.preprocessing = payload.preprocessing;
        }
        if (payload.analysis_metadata && typeof payload.analysis_metadata === 'object') {
            sanitized.analysis_metadata = payload.analysis_metadata;
        }
        return sanitized;
    }

    function sanitizePreprocessingReport(report) {
        if (!report || typeof report !== 'object') {
            return null;
        }
        const { applied_steps, dropped_columns, imputations, summary } = report;
        return {
            summary: truncateText(summary || '', 1000),
            applied_steps: Array.isArray(applied_steps) ? applied_steps.slice(0, 10) : null,
            dropped_columns: Array.isArray(dropped_columns) ? dropped_columns.slice(0, 20) : null,
            imputations: Array.isArray(imputations) ? imputations.slice(0, 10) : null
        };
    }

    function sanitizeCustomResponse(response) {
        if (!response || typeof response !== 'object') {
            return null;
        }
        const sanitized = {
            success: Boolean(response.success)
        };
        if (response.error) {
            sanitized.error = truncateText(response.error, 800);
        }
        if (response.output && typeof response.output === 'string') {
            sanitized.stdout = truncateText(response.output, 600);
        } else if (response.output && typeof response.output === 'object') {
            sanitized.stdout = truncateText(response.output.stdout || '', 600);
            if (response.output.stderr) {
                sanitized.stderr = truncateText(response.output.stderr, 400);
            }
        }
        if (Array.isArray(response.plots)) {
            sanitized.plot_count = response.plots.length;
        }
        if (response.rate_limit_info) {
            sanitized.rate_limit_info = response.rate_limit_info;
        }
        return sanitized;
    }

    function buildResponsePreview(response) {
        if (!response || typeof response !== 'object') {
            return null;
        }
        const preview = {
            success: Boolean(response.success)
        };
        if (response.analysis_count !== undefined) {
            preview.analysis_count = response.analysis_count;
        }
        if (response.metadata && typeof response.metadata === 'object') {
            const executionTime = response.metadata.execution_time || response.metadata.runtime_seconds;
            if (executionTime !== undefined) {
                preview.execution_time = executionTime;
            }
        }
        if (response.warnings) {
            preview.warnings = Array.isArray(response.warnings) ? response.warnings.slice(0, 5) : truncateText(String(response.warnings), 500);
        }
        return preview;
    }

    function getSelectedColumnsSnapshot() {
        try {
            if (typeof selectedColumns !== 'undefined' && selectedColumns && typeof selectedColumns.size === 'number') {
                return Array.from(selectedColumns);
            }
        } catch (e) {
            console.warn('Unable to snapshot selected columns:', e);
        }
        return null;
    }

    function getDatasetInfo() {
        const sourceId = typeof initSourceId === 'function' ? initSourceId() : (existingBridge.sourceId || null);
        let name = null;
        const nameEl = document.getElementById('dataset-name') || document.getElementById('datasetName');
        if (nameEl && nameEl.textContent) {
            name = nameEl.textContent.trim();
        } else if (window.currentDataset?.name) {
            name = window.currentDataset.name;
        }
        return {
            sourceId,
            name,
            info: window.currentDataset?.info || null
        };
    }

    function getPreprocessingContext() {
        try {
            const applied = typeof preprocessingApplied !== 'undefined' ? Boolean(preprocessingApplied) : null;
            const dirty = typeof preprocessingDirty !== 'undefined' ? Boolean(preprocessingDirty) : null;
            const summaryFn = typeof buildPreprocessingSummary === 'function' ? buildPreprocessingSummary : null;
            const summary = summaryFn && lastPreprocessingReport ? summaryFn(lastPreprocessingReport) : null;
            return {
                applied,
                pendingChanges: dirty,
                summary,
                lastReport: sanitizePreprocessingReport(lastPreprocessingReport)
            };
        } catch (error) {
            console.warn('Unable to build preprocessing context for LLM chat:', error);
            return null;
        }
    }

    function emitContextUpdate() {
        lastUpdate = new Date().toISOString();
        if (window.EDAChatBridge) {
            window.EDAChatBridge.lastUpdate = lastUpdate;
            window.EDAChatBridge.__analysisCells = analysisCells;
            window.EDAChatBridge.__customCells = customCells;
            window.EDAChatBridge.__activeContext = activeContext;
            window.EDAChatBridge.__notebookMeta = notebookMeta;
        }
        try {
            const context = buildContext();
            window.dispatchEvent(new CustomEvent('eda-llm-context-updated', { detail: context }));
        } catch (eventError) {
            console.warn('Failed to dispatch EDA LLM context event:', eventError);
        }
    }

    function buildContext() {
        return {
            type: 'eda_notebook',
            dataset: getDatasetInfo(),
            preprocessing: getPreprocessingContext(),
            selectedColumns: getSelectedColumnsSnapshot(),
            activeContext,
            lastUpdate,
            analysisCells: Array.from(analysisCells.values()),
            customAnalyses: Array.from(customCells.values()),
            analysisHighlights: buildAnalysisHighlights(),
            customHighlights: buildCustomHighlights(),
            summary: buildNotebookSummary(),
            meta: notebookMeta || {}
        };
    }

    function ensureAnalysisEntry(cellId) {
        const existingEntry = analysisCells.get(cellId);
        if (existingEntry) {
            return existingEntry;
        }
        const entry = {
            cellId,
            scope: 'analysis',
            status: 'queued',
            runCount: 0
        };
        analysisCells.set(cellId, entry);
        return entry;
    }

    function ensureCustomEntry(cellId) {
        const existingEntry = customCells.get(cellId);
        if (existingEntry) {
            return existingEntry;
        }
        const entry = {
            cellId,
            scope: 'custom',
            status: 'queued',
            runCount: 0
        };
        customCells.set(cellId, entry);
        return entry;
    }

    function recordAnalysisPending(cellId, payload = {}) {
        const entry = ensureAnalysisEntry(cellId);
        const now = new Date().toISOString();
        const updated = {
            ...entry,
            analysisType: payload.analysisType || entry.analysisType || null,
            analysisName: payload.analysisName || entry.analysisName || (payload.analysisType ? getAnalysisTypeName(payload.analysisType) : null),
            status: 'running',
            startedAt: now,
            completedAt: null,
            request: sanitizeRequestPayload(payload.requestPayload) || entry.request || null,
            selectedColumnsSnapshot: Array.isArray(payload.selectedColumns) ? payload.selectedColumns : entry.selectedColumnsSnapshot || getSelectedColumnsSnapshot() || null
        };
        if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        analysisCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function recordAnalysisSuccess(cellId, payload = {}) {
        const entry = ensureAnalysisEntry(cellId);
        const updated = {
            ...entry,
            analysisType: payload.analysisType || entry.analysisType || null,
            analysisName: payload.analysisName || entry.analysisName || (payload.analysisType ? getAnalysisTypeName(payload.analysisType) : null),
            status: 'success',
            completedAt: new Date().toISOString(),
            runCount: (entry.runCount || 0) + 1,
            structuredResults: sanitizeStructuredResults(payload.structuredResults),
            legacyOutput: sanitizeLegacyOutput(payload.legacyOutput),
            request: sanitizeRequestPayload(payload.requestPayload) || entry.request || null,
            selectedColumnsSnapshot: Array.isArray(payload.selectedColumns) ? payload.selectedColumns : entry.selectedColumnsSnapshot || getSelectedColumnsSnapshot() || null,
            metaSummary: payload.metaSummary ? truncateText(payload.metaSummary, 1200) : entry.metaSummary || null,
            responsePreview: buildResponsePreview(payload.rawResponse)
        };
        if (!updated.startedAt) {
            updated.startedAt = entry.startedAt || new Date().toISOString();
        }
        const summary = buildAnalysisLLMSummary(updated);
        if (summary) {
            updated.llmSummary = summary;
        } else if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        analysisCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function recordAnalysisError(cellId, payload = {}) {
        const entry = ensureAnalysisEntry(cellId);
        const updated = {
            ...entry,
            analysisType: payload.analysisType || entry.analysisType || null,
            analysisName: payload.analysisName || entry.analysisName || (payload.analysisType ? getAnalysisTypeName(payload.analysisType) : null),
            status: 'error',
            completedAt: new Date().toISOString(),
            error: truncateText(payload.errorMessage || payload.error || 'Analysis failed.', 1200),
            request: sanitizeRequestPayload(payload.requestPayload) || entry.request || null,
            selectedColumnsSnapshot: Array.isArray(payload.selectedColumns) ? payload.selectedColumns : entry.selectedColumnsSnapshot || getSelectedColumnsSnapshot() || null
        };
        if (!updated.startedAt) {
            updated.startedAt = entry.startedAt || new Date().toISOString();
        }
        const errorSummary = truncateText(updated.error, 600);
        if (errorSummary) {
            updated.llmSummary = errorSummary;
        } else if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        analysisCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function removeAnalysisCell(cellId) {
        if (analysisCells.has(cellId)) {
            analysisCells.delete(cellId);
            if (activeContext?.scope === 'analysis' && activeContext.cellId === cellId) {
                activeContext = null;
            }
            emitContextUpdate();
        }
    }

    function recordCustomPending(cellId, payload = {}) {
        const entry = ensureCustomEntry(cellId);
        const now = new Date().toISOString();
        const updated = {
            ...entry,
            status: 'running',
            startedAt: now,
            completedAt: null,
            code: truncateText(payload.code || entry.code || '', MAX_TEXT_LENGTH)
        };
        if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        customCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function recordCustomSuccess(cellId, payload = {}) {
        const entry = ensureCustomEntry(cellId);
        const updated = {
            ...entry,
            status: 'success',
            completedAt: new Date().toISOString(),
            runCount: (entry.runCount || 0) + 1,
            code: truncateText(payload.code || entry.code || '', MAX_TEXT_LENGTH),
            textOutput: truncateText(payload.textOutput || '', MAX_TEXT_LENGTH),
            plotCount: Array.isArray(payload.plots) ? payload.plots.length : entry.plotCount || 0,
            plotsPreview: Array.isArray(payload.plots) && payload.plots.length > 0 ? truncateText(String(payload.plots[0]), MAX_PREVIEW_LENGTH) : entry.plotsPreview || null,
            responsePreview: sanitizeCustomResponse(payload.rawResponse)
        };
        if (!updated.startedAt) {
            updated.startedAt = entry.startedAt || new Date().toISOString();
        }
        const summary = buildCustomLLMSummary(updated);
        if (summary) {
            updated.llmSummary = summary;
        } else if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        customCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function recordCustomError(cellId, payload = {}) {
        const entry = ensureCustomEntry(cellId);
        const updated = {
            ...entry,
            status: 'error',
            completedAt: new Date().toISOString(),
            code: truncateText(payload.code || entry.code || '', MAX_TEXT_LENGTH),
            error: truncateText(payload.errorMessage || payload.error || 'Execution failed.', 2000),
            responsePreview: sanitizeCustomResponse(payload.rawResponse)
        };
        if (!updated.startedAt) {
            updated.startedAt = entry.startedAt || new Date().toISOString();
        }
        const summary = buildCustomLLMSummary(updated);
        if (summary) {
            updated.llmSummary = summary;
        } else if (updated.llmSummary) {
            delete updated.llmSummary;
        }
        customCells.set(cellId, updated);
        emitContextUpdate();
        return updated;
    }

    function removeCustomCell(cellId) {
        if (customCells.has(cellId)) {
            customCells.delete(cellId);
            if (activeContext?.scope === 'custom' && activeContext.cellId === cellId) {
                activeContext = null;
            }
            emitContextUpdate();
        }
    }

    function setActiveCell(cellId, scope = 'analysis') {
        activeContext = { scope, cellId, timestamp: new Date().toISOString() };
        emitContextUpdate();
    }

    function setNotebookMeta(meta) {
        if (!meta || typeof meta !== 'object') {
            return;
        }
        notebookMeta = { ...(notebookMeta || {}), ...meta };
        emitContextUpdate();
    }

    window.EDAChatBridge = Object.assign(existingBridge, {
        getContext: buildContext,
        setActiveCell,
        recordAnalysisPending,
        recordAnalysisSuccess,
        recordAnalysisError,
        removeAnalysisCell,
        recordCustomPending,
        recordCustomSuccess,
        recordCustomError,
        removeCustomCell,
        setNotebookMeta,
        lastUpdate,
        __analysisCells: analysisCells,
        __customCells: customCells,
        __activeContext: activeContext,
        __notebookMeta: notebookMeta
    });
})();

function openAskAIForCell(cellId) {
    if (window.EDAChatBridge && typeof window.EDAChatBridge.setActiveCell === 'function') {
        window.EDAChatBridge.setActiveCell(cellId, 'analysis');
    }

    if (window.LLMChat && typeof window.LLMChat.showModal === 'function') {
        window.LLMChat.showModal();
    } else {
        showNotification('AI assistant is currently unavailable.', 'warning');
    }
}

// Add a single analysis result card (extracted from original addAnalysisCell function)
async function addSingleAnalysisCell(analysisType, options = {}) {
    console.log('Adding single analysis result for:', analysisType);

    const {
        skipMarketingModal = false,
        skipCategoricalModal = false,
        skipNumericModal = false,
        skipCategoricalNumericModal = false,
        skipCrossTabModal = false,
    skipGeospatialModal = false,
        skipTimeSeriesModal = false,
        skipTextModal = false,
        skipTargetModal = false,
        skipNetworkModal = false,
        skipEntityNetworkModal = false,
        prefetchedColumns = [],
        analysisOptions = {}
    } = options || {};

    const cellId = `cell-${cellCounter++}`;
    console.log('Result card ID:', cellId);

    // Create and add cell immediately (shows loading state)
    const cellHTML = createAnalysisCell(cellId, analysisType);
    console.log('Cell HTML created');

    const cellsContainer = document.getElementById('notebookCells');
    console.log('Cells container:', cellsContainer);

    if (cellsContainer) {
        if (cellsContainer.firstElementChild) {
            cellsContainer.insertAdjacentHTML('afterbegin', cellHTML);
            console.log('Cell HTML inserted at top');
        } else {
            cellsContainer.insertAdjacentHTML('beforeend', cellHTML);
            console.log('Cell HTML inserted as first item');
        }
        updateAnalysisResultsPlaceholder();
    } else {
        console.error('notebookCells container not found!');
        return null;
    }

    // Scroll to new cell
    const newCell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (newCell) {
        newCell.scrollIntoView({ behavior: 'smooth', block: 'center' });
        console.log('Scrolled to new cell');
    } else {
        console.error('New cell not found for scrolling:', cellId);
    }

    // Generate and run analysis
    try {
        if (isMarketingAnalysisType(analysisType)) {
            console.log('Marketing analysis detected, preparing configuration for:', analysisType);

            let columns = Array.isArray(prefetchedColumns) ? [...prefetchedColumns] : [];

            if (columns.length === 0) {
                if (window.currentDataFrame && Array.isArray(window.currentDataFrame.columns) && window.currentDataFrame.columns.length > 0) {
                    columns = [...window.currentDataFrame.columns];
                } else {
                    const storedColumns = sessionStorage.getItem('datasetColumns');
                    if (storedColumns) {
                        try {
                            const parsed = JSON.parse(storedColumns);
                            if (Array.isArray(parsed)) {
                                columns = parsed;
                            }
                        } catch (e) {
                            console.warn('Could not parse stored columns:', e);
                        }
                    }
                }
            }

            const sanitizedColumns = Array.from(new Set((columns || []).filter(Boolean)));

            if (sanitizedColumns.length > 0) {
                try {
                    sessionStorage.setItem('datasetColumns', JSON.stringify(sanitizedColumns));
                } catch (storageError) {
                    console.warn('Failed to persist dataset columns for marketing modal:', storageError);
                }
            }

            currentMarketingCellId = cellId;

            if (skipMarketingModal) {
                return cellId;
            }

            showMarketingAnalysisModal(analysisType, sanitizedColumns);
        } else if (isCategoricalAnalysisType(analysisType) && skipCategoricalModal) {
            return cellId;
        } else if (isNumericFrequencyAnalysisType(analysisType) && skipNumericModal) {
            return cellId;
        } else if (isCategoricalNumericAnalysisType(analysisType) && skipCategoricalNumericModal) {
            return cellId;
        } else if (isCrossTabAnalysisType(analysisType) && skipCrossTabModal) {
            return cellId;
        } else if (isGeospatialAnalysisType(analysisType) && skipGeospatialModal) {
            return cellId;
        } else if (isTimeSeriesAnalysisType(analysisType) && skipTimeSeriesModal) {
            return cellId;
        } else if (isTargetAnalysisType(analysisType) && skipTargetModal) {
            return cellId;
        } else if (isTextAnalysisType(analysisType) && skipTextModal) {
            return cellId;
        } else if (isNetworkAnalysisType(analysisType) && skipNetworkModal) {
            return cellId;
        } else if (isEntityNetworkAnalysisType(analysisType) && skipEntityNetworkModal) {
            return cellId;
        } else {
            await generateAndRunAnalysis(cellId, analysisType, {}, analysisOptions);
        }
    } catch (error) {
        console.error('Analysis execution failed:', error);
        showNotification(`Analysis execution failed for ${getAnalysisTypeName(analysisType)}`, 'error');
    }

    return cellId;
}

// Legacy function for compatibility (redirects to new multi-select system)
async function addAnalysisCell() {
    console.log('addAnalysisCell called - redirecting to multi-select system');
    return await addSelectedAnalysisCells();
}

// Generate and run analysis via backend
async function generateAndRunAnalysis(cellId, analysisType, columnMapping = {}, analysisOptions = {}) {
    // Handle case where cellId is actually analysisType (for marketing modal calls)
    if (typeof cellId === 'string' && !analysisType) {
        analysisType = cellId;
        cellId = `analysis-${Date.now()}`;
        
        // Create a new cell for the analysis
        const newCell = addAnalysisCell();
        cellId = newCell.getAttribute('data-cell-id');
        
        console.log('Created new cell for marketing analysis:', cellId);
    }
    
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    
    console.log('generateAndRunAnalysis called with:', { cellId, analysisType, columnMapping });
    
    if (!analysisType) {
        console.error('No analysis type provided!');
        showNotification('Please select an analysis type first', 'error');
        return;
    }
    const outputArea = document.getElementById(`output-${cellId}`);
    const resultDiv = document.getElementById(`result-${cellId}`);
    const loadingDiv = cell ? cell.querySelector('.loading-indicator') : null;
    const statusBadge = cell ? cell.querySelector('.analysis-run-indicator') : null;
    const optionsForRun = analysisOptions || {};
    const overrideSelection = Array.isArray(optionsForRun.overrideSelectedColumns)
        ? optionsForRun.overrideSelectedColumns.filter(name => name !== null && name !== undefined)
        : null;
    const normalizedOverrideSelection = overrideSelection && overrideSelection.length > 0
        ? Array.from(new Set(
            overrideSelection
                .map(name => (typeof name === 'string' ? name : String(name)).trim())
                .filter(Boolean)
        ))
        : [];
    const selectionPayload = optionsForRun.modalSelectionPayload || null;
    const analysisMetadataOverride = optionsForRun.analysisMetadata;
    const shouldIncludeGlobalSelection = optionsForRun.includeGlobalSelectedColumns !== false;

    let requestPayload = null;
    let runSelectedColumns = null;

    try {
        // Show executing state
        if (cell) {
            cell.classList.add('executing');
            cell.classList.remove('error');
        }
        updateRunIndicator(statusBadge, 'running', 'Running…');
        updateAnalysisMeta(cellId, `Running ${getAnalysisTypeName(analysisType)} • ${new Date().toLocaleTimeString()}`);
        clearAnalysisAlerts(cellId);

        if (loadingDiv) {
            loadingDiv.style.display = 'block';
            loadingDiv.innerHTML = `
                <div class="text-center text-muted py-4">
                    <div class="loading-spinner"></div>
                    <p class="mt-2 mb-0">Generating and running analysis…</p>
                </div>
            `;
        }

        if (outputArea) {
            outputArea.style.display = 'none';
        }

        if (resultDiv) {
            resultDiv.innerHTML = '';
        }
        
        console.log(`Calling granular analysis execution API: /advanced-eda/components/run-analysis`);
        console.log(`Analysis IDs: [${analysisType}]`);
        
        // Call backend API for granular analysis execution
        // Include selected columns if available
        requestPayload = {
            analysis_ids: [analysisType],
            source_id: sourceId
        };
        
        // Add column mapping for marketing analyses
        if (columnMapping && Object.keys(columnMapping).length > 0) {
            requestPayload.column_mapping = columnMapping;
            console.log('Including marketing column mapping:', columnMapping);
        }
        
        if (normalizedOverrideSelection.length > 0) {
            requestPayload.selected_columns = [...normalizedOverrideSelection];
            console.log('Using modal-selected columns in analysis:', requestPayload.selected_columns);
        } else if (shouldIncludeGlobalSelection && selectedColumns.size > 0) {
            requestPayload.selected_columns = Array.from(selectedColumns);
            console.log('Including selected columns in analysis:', requestPayload.selected_columns);
        }

        if (analysisMetadataOverride && typeof analysisMetadataOverride === 'object' && Object.keys(analysisMetadataOverride).length > 0) {
            requestPayload.analysis_metadata = analysisMetadataOverride;
            console.log('Including analysis metadata overrides:', analysisMetadataOverride);
        }

        if (preprocessingApplied) {
            const preprocessingPayload = buildPreprocessingPayload();
            if (preprocessingPayload) {
                requestPayload.preprocessing = preprocessingPayload;
                console.log('Including preprocessing payload:', preprocessingPayload);
            }
        }

        runSelectedColumns = Array.isArray(requestPayload.selected_columns)
            ? [...requestPayload.selected_columns]
            : (selectedColumns && selectedColumns.size > 0 ? Array.from(selectedColumns) : null);

        if (window.EDAChatBridge && typeof window.EDAChatBridge.recordAnalysisPending === 'function') {
            window.EDAChatBridge.recordAnalysisPending(cellId, {
                analysisType,
                analysisName: getAnalysisTypeName(analysisType),
                requestPayload,
                selectedColumns: runSelectedColumns || undefined
            });
        }
        
        const response = await fetch(`/advanced-eda/components/run-analysis`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestPayload)
        });

        console.log(`Response status: ${response.status}`);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('API Error Response:', errorText);
            console.error('Request payload was:', requestPayload);
            console.error('Response status:', response.status);
            console.error('Response headers:', [...response.headers.entries()]);
            
            // Try to parse error as JSON for better debugging
            try {
                const errorJson = JSON.parse(errorText);
                console.error('Parsed error JSON:', errorJson);
                if (errorJson.debug_info) {
                    console.error('Debug info:', errorJson.debug_info);
                }
            } catch (e) {
                console.error('Error response is not JSON:', errorText);
            }
            
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('API Result:', result);

        if (result.success) {
            updateRunIndicator(statusBadge, 'success', `Run #${executionCounter++}`);
            if (cell) {
                cell.classList.remove('executing');
            }

            if (loadingDiv) {
                loadingDiv.style.display = 'none';
            }
            if (outputArea) {
                outputArea.style.display = 'block';
            }

            const structuredResults = getStructuredResultsFromResponse(result);
            const legacyOutput = result.output || result.execution_result;

            const preprocessingReportPayload = result.preprocessing_report || result.metadata?.preprocessing_report;
            if (preprocessingReportPayload) {
                handlePreprocessingReport(preprocessingReportPayload, { preserveAppliedState: true });
            }

            if (resultDiv) {
                if (structuredResults.length > 0) {
                    renderStructuredAnalysis(resultDiv, structuredResults);
                } else if (legacyOutput && (legacyOutput.stdout || legacyOutput.stderr || legacyOutput.plots?.length > 0 || legacyOutput.data_frames?.length > 0)) {
                    renderLegacyCellOutput(resultDiv, legacyOutput);
                } else {
                    resultDiv.innerHTML = '<p class="text-muted">Analysis completed successfully (no output to display).</p>';
                }
            }

            let metaSummary = buildAnalysisMetaText(structuredResults, result.analysis_count);
            updateAnalysisMeta(cellId, metaSummary);

            if (window.EDAChatBridge && typeof window.EDAChatBridge.recordAnalysisSuccess === 'function') {
                window.EDAChatBridge.recordAnalysisSuccess(cellId, {
                    analysisType,
                    analysisName: getAnalysisTypeName(analysisType),
                    structuredResults,
                    legacyOutput,
                    requestPayload,
                    selectedColumns: runSelectedColumns || undefined,
                    metaSummary,
                    rawResponse: result
                });
            }

            if (result.warnings) {
                pushAnalysisAlert(cellId, 'warning', result.warnings);
            }

            if (Array.isArray(result.incompatible_analyses) && result.incompatible_analyses.length > 0) {
                pushAnalysisAlert(
                    cellId,
                    'info',
                    `Skipped incompatible analyses: ${result.incompatible_analyses.join(', ')}`
                );
            }

            const executionWarnings = legacyOutput?.warnings;
            if (executionWarnings) {
                const warningMessage = Array.isArray(executionWarnings)
                    ? executionWarnings.join('<br>')
                    : executionWarnings;
                pushAnalysisAlert(cellId, 'warning', warningMessage);
            }

            // Surface warning or danger insights in alerts area for quick scanning
            if (structuredResults.length > 0) {
                structuredResults.forEach(res => {
                    if (Array.isArray(res.insights)) {
                        res.insights
                            .filter(insight => ['warning', 'danger'].includes((insight.level || '').toLowerCase()))
                            .forEach(insight => pushAnalysisAlert(cellId, insight.level === 'danger' ? 'error' : 'warning', insight.text));
                    }
                });
            }

            if (cell) {
                if (optionsForRun.modalType) {
                    cell.dataset.modalType = optionsForRun.modalType;
                } else {
                    delete cell.dataset.modalType;
                }

                if (normalizedOverrideSelection.length > 0) {
                    cell.dataset.modalSelection = JSON.stringify(normalizedOverrideSelection);
                } else {
                    delete cell.dataset.modalSelection;
                }

                if (selectionPayload) {
                    try {
                        cell.dataset.modalSelectionDetails = JSON.stringify(selectionPayload);
                    } catch (detailError) {
                        console.warn('Unable to persist modal selection details:', detailError);
                    }
                } else {
                    delete cell.dataset.modalSelectionDetails;
                }
            }

            showNotification(`${getAnalysisTypeName(analysisType)} completed!`, 'success');

        } else {
            updateRunIndicator(statusBadge, 'error', 'Failed');
            if (cell) {
                cell.classList.add('error');
                cell.classList.remove('executing');
            }
            const errorMessage = result.error || 'Analysis generation failed';
            if (loadingDiv) {
                loadingDiv.style.display = 'block';
                loadingDiv.innerHTML = `<div class="error-message"><strong>Error:</strong> ${errorMessage}</div>`;
            }
            updateAnalysisMeta(cellId, 'Execution failed. Review the error details below.');
            pushAnalysisAlert(cellId, 'error', errorMessage);
            showNotification(errorMessage, 'error');

            if (window.EDAChatBridge && typeof window.EDAChatBridge.recordAnalysisError === 'function') {
                window.EDAChatBridge.recordAnalysisError(cellId, {
                    analysisType,
                    analysisName: getAnalysisTypeName(analysisType),
                    errorMessage,
                    requestPayload,
                    selectedColumns: runSelectedColumns || undefined
                });
            }
            return;
        }
        
    } catch (error) {
        console.error('Analysis generation error:', error);
        if (cell) {
            cell.classList.remove('executing');
            cell.classList.add('error');
        }
        updateRunIndicator(statusBadge, 'error', 'Failed');
        if (loadingDiv) {
            loadingDiv.style.display = 'block';
            loadingDiv.innerHTML = `<div class="error-message"><strong>Error:</strong> ${error.message}</div>`;
        }
        if (outputArea) {
            outputArea.style.display = 'none';
        }
        updateAnalysisMeta(cellId, 'Execution failed. Check the message above for details.');
        pushAnalysisAlert(cellId, 'error', error.message || 'Analysis failed to complete.');
        showNotification('Analysis failed to complete', 'error');

        if (window.EDAChatBridge && typeof window.EDAChatBridge.recordAnalysisError === 'function') {
            window.EDAChatBridge.recordAnalysisError(cellId, {
                analysisType,
                analysisName: getAnalysisTypeName(analysisType),
                errorMessage: error?.message,
                requestPayload,
                selectedColumns: runSelectedColumns || undefined
            });
        }
    }
}

// Create modern analysis result card
function createAnalysisCell(cellId, analysisType) {
    const analysisName = getAnalysisTypeName(analysisType);
    const defaultDescription = 'Loading component details…';
    
    return `
    <article class="analysis-result-card" data-cell-type="analysis" data-cell-id="${cellId}" data-analysis-type="${analysisType}">
        <header class="analysis-result-header">
            <div class="analysis-header-text">
                <div class="analysis-name-row">
                    <span class="analysis-title">${analysisName}</span>
                </div>
                <p class="analysis-description" id="description-${cellId}">${defaultDescription}</p>
            </div>
            <div class="analysis-actions">
                <span class="analysis-run-indicator status-queued" aria-live="polite">Queued</span>
                <button class="btn btn-sm btn-primary ask-ai-btn ms-2" onclick="openAskAIForCell('${cellId}')" title="Ask AI about this result">
                    <i class="bi bi-stars"></i> Ask AI
                </button>
                <button class="btn btn-sm btn-outline-secondary" onclick="moveCellUp('${cellId}')" title="Move result up">
                    <i class="bi bi-arrow-up"></i>
                </button>
                <button class="btn btn-sm btn-outline-secondary" onclick="moveCellDown('${cellId}')" title="Move result down">
                    <i class="bi bi-arrow-down"></i>
                </button>
                <button class="btn btn-sm btn-outline-secondary" onclick="openAnalysisCodeFromCell('${cellId}', '${analysisType}')" title="View code example">
                    <i class="bi bi-code-slash"></i> Code
                </button>
                <button class="btn btn-sm btn-outline-secondary" onclick="rerunAnalysis('${cellId}', '${analysisType}')" title="Run again">
                    <i class="bi bi-arrow-clockwise"></i> Rerun
                </button>
                <button class="btn btn-sm btn-outline-danger" onclick="deleteCell('${cellId}')" title="Remove result">
                    <i class="bi bi-trash"></i>
                </button>
            </div>
        </header>
        <div class="analysis-meta-row">
            <div class="analysis-meta" id="meta-${cellId}">Ready to execute. Results will appear below.</div>
        </div>
        <div class="analysis-tags" id="tags-${cellId}"></div>
        <div class="analysis-body">
            <div class="loading-indicator">
                <div class="text-center text-muted py-4">
                    <div class="loading-spinner"></div>
                    <p class="mt-2 mb-0">Preparing analysis...</p>
                </div>
            </div>
            <div class="analysis-output cell-output" id="output-${cellId}" style="display: none;">
                <div id="result-${cellId}"></div>
            </div>
        </div>
    </article>`;
}

async function loadAnalysisMetadata(cellId, analysisType) {
    if (!analysisType) {
        return;
    }

    if (analysisMetadataCache.has(analysisType)) {
        applyAnalysisMetadata(cellId, analysisMetadataCache.get(analysisType));
        return;
    }

    try {
        const response = await fetch(`/advanced-eda/components/${analysisType}/info`);
        if (!response.ok) {
            throw new Error(`Metadata request failed with status ${response.status}`);
        }

        const payload = await response.json();
        const metadata = payload.component || payload.metadata || {};
        analysisMetadataCache.set(analysisType, metadata);
        applyAnalysisMetadata(cellId, metadata);
    } catch (error) {
        console.warn(`Failed to load analysis metadata for ${analysisType}:`, error);
        const descriptionElement = document.getElementById(`description-${cellId}`);
        if (descriptionElement && descriptionElement.textContent === 'Loading component details…') {
            descriptionElement.textContent = 'Detailed description is unavailable for this component.';
        }
    }
}

function applyAnalysisMetadata(cellId, metadata = {}) {
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (!cell) {
        return;
    }

    const titleElement = cell.querySelector('.analysis-title');
    if (titleElement && metadata.name) {
        titleElement.textContent = metadata.name;
    }

    const descriptionElement = document.getElementById(`description-${cellId}`);
    if (descriptionElement && metadata.description) {
        descriptionElement.textContent = metadata.description;
    }

    const metaElement = document.getElementById(`meta-${cellId}`);
    if (metaElement) {
        const category = formatAnalysisCategory(metadata.category);
        const complexity = (metadata.complexity || 'intermediate').replace('_', ' ');
        const runtime = metadata.estimated_runtime || '1-5 seconds';
        metaElement.textContent = `${category} • ${complexity.charAt(0).toUpperCase() + complexity.slice(1)} • Est. runtime ${runtime}`;
    }

    const tagsElement = document.getElementById(`tags-${cellId}`);
    if (tagsElement) {
        const tags = Array.isArray(metadata.tags) ? metadata.tags : [];
        if (tags.length) {
            tagsElement.innerHTML = tags
                .slice(0, 8)
                .map(tag => `<span class="analysis-tag">#${tag}</span>`)
                .join('');
        } else {
            tagsElement.innerHTML = '';
        }
    }
}

// Rerun analysis
async function rerunAnalysis(cellId, analysisType) {
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (!cell) {
        showNotification('Analysis cell could not be found for rerun.', 'error');
        return;
    }

    const modalType = cell.dataset.modalType || '';
    let previousSelection = [];
    let previousSelectionDetails = null;

    if (cell.dataset.modalSelection) {
        try {
            const parsed = JSON.parse(cell.dataset.modalSelection);
            if (Array.isArray(parsed)) {
                previousSelection = parsed;
            }
        } catch (parseError) {
            console.warn('Unable to parse stored modal selection for rerun:', parseError);
        }
    }

    if (cell.dataset.modalSelectionDetails) {
        try {
            const parsedDetails = JSON.parse(cell.dataset.modalSelectionDetails);
            if (parsedDetails && typeof parsedDetails === 'object') {
                previousSelectionDetails = parsedDetails;
            }
        } catch (detailError) {
            console.warn('Unable to parse stored modal selection details for rerun:', detailError);
        }
    }

    if (modalType && previousSelection.length > 0) {
        openRerunChoiceModal(cellId, analysisType, modalType, previousSelection, previousSelectionDetails);
        return;
    }

    await executeAnalysisRerun(cellId, analysisType, {});
}

async function executeAnalysisRerun(cellId, analysisType, analysisOptions = {}) {
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (!cell) {
        showNotification('Analysis cell could not be found for rerun.', 'error');
        return;
    }

    const outputArea = document.getElementById(`output-${cellId}`);
    const loadingDiv = cell.querySelector('.loading-indicator');

    cell.classList.remove('error');
    if (outputArea) {
        outputArea.style.display = 'none';
    }
    if (loadingDiv) {
        loadingDiv.style.display = 'block';
        loadingDiv.innerHTML = '<div class="text-center text-muted py-4"><div class="loading-spinner"></div><p class="mt-2 mb-0">Re-running analysis...</p></div>';
    }

    showNotification('Re-running analysis...', 'info');

    if (window.EDAChatBridge && typeof window.EDAChatBridge.recordAnalysisPending === 'function') {
        window.EDAChatBridge.recordAnalysisPending(cellId, {
            analysisType,
            analysisName: getAnalysisTypeName(analysisType),
            requestPayload: analysisOptions?.requestPayload,
            selectedColumns: analysisOptions?.selectedColumns
        });
    }

    try {
        const columnMapping = analysisOptions && typeof analysisOptions.columnMapping === 'object'
            ? { ...analysisOptions.columnMapping }
            : {};
        const optionsForRun = { ...analysisOptions };
        if (optionsForRun.columnMapping) {
            delete optionsForRun.columnMapping;
        }
        await generateAndRunAnalysis(cellId, analysisType, columnMapping, optionsForRun);
    } catch (error) {
        console.error('Analysis rerun failed:', error);
        showNotification('Analysis rerun failed', 'error');
    }
}

function openRerunChoiceModal(cellId, analysisType, modalType, previousSelection, previousDetails = null) {
    const modalElement = document.getElementById('analysisRerunModal');
    if (!modalElement) {
        console.warn('Rerun choice modal is unavailable; defaulting to reuse of previous selection.');
        const fallbackOptions = buildAnalysisOptionsForModal(modalType, previousSelection, previousDetails);
        executeAnalysisRerun(cellId, analysisType, fallbackOptions);
        return;
    }

    rerunModalState.cellId = cellId;
    rerunModalState.analysisType = analysisType;
    rerunModalState.modalType = modalType;
    rerunModalState.previousSelection = Array.isArray(previousSelection) ? [...previousSelection] : [];
    rerunModalState.previousSelectionDetails = previousDetails && typeof previousDetails === 'object'
        ? { ...previousDetails }
        : null;

    const analysisName = getAnalysisTypeName(analysisType);
    const messageElement = document.getElementById('rerunModalMessage');
    if (messageElement) {
        messageElement.textContent = `Would you like to rerun ${analysisName} with the previous column selection or choose a new set?`;
    }

    const previewElement = document.getElementById('rerunModalSelectionPreview');
    if (previewElement) {
        if (rerunModalState.previousSelection.length === 0) {
            previewElement.textContent = 'No previous column selection was stored.';
        } else {
            const preview = rerunModalState.previousSelection.slice(0, 6);
            const remainder = rerunModalState.previousSelection.length - preview.length;
            previewElement.textContent = remainder > 0
                ? `${preview.join(', ')} (+${remainder} more)`
                : preview.join(', ');
        }
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
}

function handleRerunReuse() {
    const { cellId, analysisType, modalType, previousSelection, previousSelectionDetails } = rerunModalState;
    const modalElement = document.getElementById('analysisRerunModal');
    if (modalElement) {
        const instance = bootstrap.Modal.getInstance(modalElement);
        instance?.hide();
    }

    if (!cellId || !analysisType) {
        resetRerunModalState();
        return;
    }

    const selection = Array.isArray(previousSelection) ? [...previousSelection] : [];
    const detailPayload = previousSelectionDetails && typeof previousSelectionDetails === 'object'
        ? { ...previousSelectionDetails }
        : null;
    resetRerunModalState();
    const rerunOptions = buildAnalysisOptionsForModal(modalType, selection, detailPayload);
    executeAnalysisRerun(cellId, analysisType, rerunOptions);
}

function handleRerunReconfigure() {
    const { cellId, analysisType, modalType, previousSelection, previousSelectionDetails } = rerunModalState;
    const modalElement = document.getElementById('analysisRerunModal');
    if (modalElement) {
        const instance = bootstrap.Modal.getInstance(modalElement);
        instance?.hide();
    }

    if (!cellId || !analysisType) {
        resetRerunModalState();
        return;
    }

    const selection = Array.isArray(previousSelection) ? [...previousSelection] : [];
    const detailPayload = previousSelectionDetails && typeof previousSelectionDetails === 'object'
        ? { ...previousSelectionDetails }
        : null;
    resetRerunModalState();

    if (modalType === 'categorical') {
        openCategoricalModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'numeric') {
        openNumericModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'categorical-numeric') {
        openCategoricalNumericModalForRerun(cellId, analysisType, selection, detailPayload);
    } else if (modalType === 'crosstab') {
        openCrossTabModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'time-series') {
        openTimeSeriesModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'text') {
        openTextModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'network') {
        openNetworkModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'entity-network') {
        openEntityNetworkModalForRerun(cellId, analysisType, selection);
    } else if (modalType === 'target') {
        openTargetModalForRerun(cellId, analysisType, selection, detailPayload);
    } else if (modalType === 'geospatial') {
        openGeospatialModalForRerun(cellId, analysisType, selection, detailPayload);
    } else {
        const fallbackOptions = buildAnalysisOptionsForModal(modalType, selection, detailPayload);
        executeAnalysisRerun(cellId, analysisType, fallbackOptions);
    }
}

function resetRerunModalState() {
    rerunModalState.cellId = '';
    rerunModalState.analysisType = '';
    rerunModalState.modalType = '';
    rerunModalState.previousSelection = [];
    rerunModalState.previousSelectionDetails = null;
}

function buildAnalysisOptionsForModal(modalType, selection, previousDetails) {
    let normalizedSelection = Array.isArray(selection)
        ? selection
            .map(name => (typeof name === 'string' ? name.trim() : String(name || '')))
            .filter(Boolean)
        : [];

    const options = {
        overrideSelectedColumns: normalizedSelection,
        includeGlobalSelectedColumns: false
    };

    if (modalType) {
        options.modalType = modalType;
    }

    if (previousDetails && typeof previousDetails === 'object') {
        options.modalSelectionPayload = { ...previousDetails };

        if (modalType === 'categorical-numeric' && typeof buildCategoricalNumericAnalysisMetadata === 'function') {
            try {
                options.analysisMetadata = buildCategoricalNumericAnalysisMetadata(previousDetails);
            } catch (metadataError) {
                console.warn('Unable to rebuild categorical vs numeric metadata for rerun fallback:', metadataError);
            }
        } else if (modalType === 'geospatial') {
            const latitude = typeof previousDetails.latitude === 'string' ? previousDetails.latitude.trim() : '';
            const longitude = typeof previousDetails.longitude === 'string' ? previousDetails.longitude.trim() : '';
            const label = typeof previousDetails.label === 'string' ? previousDetails.label.trim() : '';

            if (latitude && longitude) {
                options.columnMapping = {
                    latitude,
                    longitude
                };
                if (label) {
                    options.columnMapping.location_label = label;
                }

                if (normalizedSelection.length === 0) {
                    normalizedSelection = label ? [latitude, longitude, label] : [latitude, longitude];
                    options.overrideSelectedColumns = normalizedSelection;
                }
            }
        }
    }

    options.overrideSelectedColumns = normalizedSelection;

    return options;
}

function openAnalysisCode(event, analysisType) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    if (!analysisType) {
        showNotification('Select an analysis type to view code.', 'warning');
        return;
    }

    const selectedCols = selectedColumns.size > 0 ? Array.from(selectedColumns) : null;
    showAnalysisCodeModal(analysisType, { selectedColumns: selectedCols });
}

function openAnalysisCodeFromCell(cellId, analysisType) {
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    let selectedCols = null;

    if (cell && cell.dataset.modalSelection) {
        try {
            const parsed = JSON.parse(cell.dataset.modalSelection);
            if (Array.isArray(parsed) && parsed.length > 0) {
                selectedCols = parsed;
            }
        } catch (error) {
            console.warn('Unable to parse stored modal selection for code preview:', error);
        }
    }

    if (!selectedCols && selectedColumns.size > 0) {
        selectedCols = Array.from(selectedColumns);
    }

    showAnalysisCodeModal(analysisType, { selectedColumns: selectedCols });
}

async function showAnalysisCodeModal(analysisType, options = {}) {
    const modalElement = document.getElementById('analysisCodeModal');
    if (!modalElement) {
        showNotification('Code preview modal is unavailable.', 'error');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();

    const titleElement = document.getElementById('analysisCodeModalLabel');
    if (titleElement) {
        titleElement.textContent = `Code snippet • ${getAnalysisTypeName(analysisType)}`;
    }

    const statusElement = document.getElementById('analysisCodeStatus');
    const codeWrapper = document.getElementById('analysisCodeBlock');
    const codeElement = document.getElementById('analysisCodeContent');
    const copyButton = document.getElementById('analysisCodeCopyBtn');

    if (copyButton) {
        copyButton.disabled = true;
    }

    if (statusElement) {
        statusElement.classList.remove('d-none');
        statusElement.textContent = 'Generating notebook-ready code...';
    }
    if (codeWrapper) {
        codeWrapper.classList.add('d-none');
    }
    if (codeElement) {
        codeElement.textContent = '';
    }

    analysisCodeModalCurrentAnalysis = analysisType;
    analysisCodeModalCurrentCode = '';

    const payload = {
        analysis_ids: [analysisType],
        source_id: initSourceId()
    };

    if (!payload.source_id) {
        if (statusElement) {
            statusElement.textContent = 'Source identifier is missing. Load a dataset before generating code.';
        }
        return;
    }

    if (Array.isArray(options.selectedColumns) && options.selectedColumns.length > 0) {
        payload.selected_columns = options.selectedColumns;
    }

    if (options.columnMapping && typeof options.columnMapping === 'object') {
        payload.column_mapping = options.columnMapping;
    }

    try {
        const response = await fetch('/advanced-eda/components/generate-code', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Code generation failed with status ${response.status}`);
        }

        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || 'Code generation failed.');
        }

        analysisCodeModalCurrentCode = result.code || '';

        if (statusElement) {
            statusElement.classList.add('d-none');
        }
        if (codeWrapper) {
            codeWrapper.classList.remove('d-none');
        }
        if (codeElement) {
            codeElement.textContent = analysisCodeModalCurrentCode || '# No code generated.';
        }
        if (copyButton) {
            copyButton.disabled = !analysisCodeModalCurrentCode;
        }
    } catch (error) {
        console.error('Failed to generate analysis code:', error);
        if (statusElement) {
            statusElement.classList.remove('d-none');
            statusElement.textContent = error.message || 'Unable to generate code for this analysis.';
        }
        if (copyButton) {
            copyButton.disabled = true;
        }
    }
}

function copyAnalysisCodeToClipboard() {
    if (!analysisCodeModalCurrentCode) {
        showNotification('No code available to copy.', 'warning');
        return;
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(analysisCodeModalCurrentCode)
            .then(() => showNotification('Analysis code copied to clipboard.', 'success'))
            .catch(error => {
                console.warn('Clipboard copy failed:', error);
                fallbackCopyAnalysisCode();
            });
    } else {
        fallbackCopyAnalysisCode();
    }
}

function fallbackCopyAnalysisCode() {
    const textarea = document.createElement('textarea');
    textarea.value = analysisCodeModalCurrentCode;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'absolute';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();
    try {
        document.execCommand('copy');
        showNotification('Analysis code copied to clipboard.', 'success');
    } catch (error) {
        console.warn('Fallback clipboard copy failed:', error);
        showNotification('Unable to copy code automatically. Select and copy manually.', 'warning');
    } finally {
        document.body.removeChild(textarea);
    }
}

// Structured analysis rendering helpers
function getStructuredResultsFromResponse(result) {
    if (Array.isArray(result.analysis_results) && result.analysis_results.length > 0) {
        return result.analysis_results;
    }

    if (result.output && Array.isArray(result.output.analysis_results) && result.output.analysis_results.length > 0) {
        return result.output.analysis_results;
    }

    return [];
}

function renderStructuredAnalysis(container, analysisResults) {
    if (!Array.isArray(analysisResults) || analysisResults.length === 0) {
        container.innerHTML = '<p class="text-muted">Analysis completed successfully (no structured results to display).</p>';
        return;
    }

    const sections = analysisResults.map(renderStructuredSection).join('');
    container.innerHTML = sections;
}

function buildAnalysisMetaText(structuredResults, analysisCount) {
    const timestamp = new Date().toLocaleTimeString();

    if (!structuredResults || structuredResults.length === 0) {
        const count = analysisCount || 1;
        const suffix = count === 1 ? 'component' : 'components';
        return `Completed ${timestamp} • ${count} ${suffix}`;
    }

    const counts = structuredResults.reduce((acc, result) => {
        const status = (result.status || 'success').toLowerCase();
        acc[status] = (acc[status] || 0) + 1;
        return acc;
    }, {});

    const summary = Object.entries(counts)
        .map(([status, count]) => `${count} ${status}`)
        .join(', ');

    return `Completed ${timestamp} • ${summary}`;
}

function renderStructuredSection(result) {
    const title = escapeHtml(result.title || result.analysis_id || 'Analysis Result');
    const summary = result.summary ? `<p class="analysis-section-summary">${escapeHtml(result.summary)}</p>` : '';
    const metrics = renderAnalysisMetrics(result.metrics);
    const insights = renderAnalysisInsights(result.insights);
    const tables = renderAnalysisTables(result.tables);
    const charts = renderAnalysisCharts(result.charts);
    const details = renderAnalysisDetails(result.details);
    const statusBadge = renderAnalysisStatusBadge(result.status);

    return `
        <section class="analysis-structured-section">
            <header class="structured-section-header">
                <div class="structured-section-heading">
                    <h5>${title}</h5>
                    ${summary}
                </div>
                ${statusBadge}
            </header>
            ${metrics}
            ${insights}
            ${tables}
            ${charts}
            ${details}
        </section>
    `;
}

function renderAnalysisStatusBadge(status) {
    const normalized = (status || 'success').toLowerCase();
    const label = normalized.charAt(0).toUpperCase() + normalized.slice(1);
    return `<span class="analysis-status-badge status-${escapeHtml(normalized)}">${escapeHtml(label)}</span>`;
}

function renderAnalysisMetrics(metrics) {
    if (!Array.isArray(metrics) || metrics.length === 0) {
        return '';
    }

    const cards = metrics
        .map(metric => {
            const label = escapeHtml(metric.label || 'Metric');
            const value = formatMetricValue(metric.value);
            const unit = metric.unit ? `<span class="metric-unit">${escapeHtml(metric.unit)}</span>` : '';
            const description = metric.description ? `<div class="metric-description">${escapeHtml(metric.description)}</div>` : '';
            const trendIcon = metric.trend === 'up' ? '<i class="bi bi-arrow-up-right text-success"></i>' : metric.trend === 'down' ? '<i class="bi bi-arrow-down-right text-danger"></i>' : '';
            return `
                <div class="analysis-metric-card">
                    <div class="metric-header">
                        <span class="metric-label">${label}</span>
                        ${trendIcon}
                    </div>
                    <div class="metric-value">${value}${unit}</div>
                    ${description}
                </div>
            `;
        })
        .join('');

    return `
        <section class="analysis-section analysis-section-metrics">
            <h6>Key metrics</h6>
            <div class="analysis-metrics-grid">
                ${cards}
            </div>
        </section>
    `;
}

function renderAnalysisInsights(insights) {
    if (!Array.isArray(insights) || insights.length === 0) {
        return '';
    }

    const items = insights
        .map(insight => {
            const level = escapeHtml((insight.level || 'info').toLowerCase());
            const text = escapeHtml(insight.text || '');
            const icon = level === 'success' ? 'bi-check-circle' : level === 'warning' ? 'bi-exclamation-triangle' : level === 'danger' ? 'bi-x-circle' : 'bi-info-circle';
            return `
                <li class="analysis-insight-item insight-${level}">
                    <span class="insight-badge">
                        <i class="bi ${icon}"></i>
                        ${level.charAt(0).toUpperCase() + level.slice(1)}
                    </span>
                    <span class="insight-text">${text}</span>
                </li>
            `;
        })
        .join('');

    return `
        <section class="analysis-section analysis-section-insights">
            <h6>Insights</h6>
            <ul class="analysis-insights-list">
                ${items}
            </ul>
        </section>
    `;
}

function renderAnalysisTables(tables) {
    if (!Array.isArray(tables) || tables.length === 0) {
        return '';
    }

    const sections = tables
        .map(table => {
            const title = escapeHtml(table.title || 'Table');
            const description = table.description ? `<p class="analysis-section-note">${escapeHtml(table.description)}</p>` : '';
            const tableHtml = createTableHtml(table.columns, table.rows);
            return `
                <div class="analysis-table-wrapper">
                    <div class="analysis-table-header">
                        <h6>${title}</h6>
                        ${description}
                    </div>
                    <div class="table-responsive analysis-table-responsive">
                        ${tableHtml}
                    </div>
                </div>
            `;
        })
        .join('');

    return `
        <section class="analysis-section analysis-section-tables">
            ${sections}
        </section>
    `;
}

function renderAnalysisCharts(charts) {
    if (!Array.isArray(charts) || charts.length === 0) {
        return '';
    }

    const items = charts
        .map((chart, index) => {
            const rawTitle = chart.title || `Chart ${index + 1}`;
            const rawDescription = chart.description || '';
            const title = escapeHtml(rawTitle);
            const description = rawDescription ? `<p class="analysis-section-note">${escapeHtml(rawDescription)}</p>` : '';
            const imageSrc = chart.image || '';
            const datasetTitle = escapeHtml(rawTitle);
            const datasetDescription = escapeHtml(rawDescription);
            const datasetSrc = escapeHtml(imageSrc);

            return `
                <figure class="analysis-chart" data-chart-src="${datasetSrc}" data-chart-title="${datasetTitle}" data-chart-description="${datasetDescription}">
                    <button type="button" class="chart-expand-button" data-chart-expand-trigger aria-label="Expand chart ${title}">
                        <i class="bi bi-arrows-fullscreen" aria-hidden="true"></i>
                        <span class="visually-hidden">Expand chart</span>
                    </button>
                    <img src="${imageSrc}" alt="${title}" class="analysis-chart-image" data-chart-expand-trigger />
                    <figcaption>
                        <strong>${title}</strong>
                        ${description}
                    </figcaption>
                </figure>
            `;
        })
        .join('');

    return `
        <section class="analysis-section analysis-section-charts">
            <h6>Visualizations</h6>
            <div class="analysis-charts-grid">
                ${items}
            </div>
        </section>
    `;
}

function renderAnalysisDetails(details) {
    if (!details || typeof details !== 'object' || Object.keys(details).length === 0) {
        return '';
    }

    const json = JSON.stringify(details, null, 2);
    return `
        <section class="analysis-section analysis-section-details">
            <h6>Additional details</h6>
            <pre class="analysis-details-pre">${escapeHtml(json)}</pre>
        </section>
    `;
}

function openChartExpandModal(imageSrc, title, description) {
    if (!imageSrc) {
        console.warn('Cannot open chart modal without an image source');
        return;
    }

    const modalElement = document.getElementById('chartExpandModal');
    const modalImage = document.getElementById('chartExpandModalImage');
    const modalTitle = document.getElementById('chartExpandModalTitle');
    const modalDescription = document.getElementById('chartExpandModalDescription');

    if (!modalElement || !modalImage || !modalTitle || !modalDescription) {
        console.warn('Chart expand modal elements not found in the DOM');
        return;
    }

    modalImage.src = imageSrc;
    modalImage.alt = title || 'Expanded chart';
    modalTitle.textContent = title || 'Chart visualization';

    if (description) {
        modalDescription.textContent = description;
        modalDescription.classList.remove('d-none');
    } else {
        modalDescription.textContent = '';
        modalDescription.classList.add('d-none');
    }

    if (typeof bootstrap === 'undefined' || !bootstrap.Modal) {
        console.warn('Bootstrap modal library is not available for chart expansion');
        return;
    }

    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
    modalInstance.show();
}

document.addEventListener('click', event => {
    const trigger = event.target.closest('[data-chart-expand-trigger]');
    if (!trigger) {
        return;
    }

    const chartFigure = trigger.closest('.analysis-chart');
    if (!chartFigure) {
        return;
    }

    const { chartSrc, chartTitle, chartDescription } = chartFigure.dataset;
    if (!chartSrc) {
        console.warn('No chart source found for expand interaction');
        return;
    }

    event.preventDefault();
    openChartExpandModal(chartSrc, chartTitle, chartDescription);
});

function createTableHtml(columns = [], rows = []) {
    if (!Array.isArray(columns) || columns.length === 0) {
        return '<p class="text-muted">No table data available.</p>';
    }

    const headerHtml = columns.map(column => `<th>${escapeHtml(column)}</th>`).join('');
    const bodyHtml = rows
        .map(row => {
            const cells = columns
                .map(column => {
                    const cellValue = row[column];
                    if (cellValue === null || cellValue === undefined) {
                        return '<td><span class="text-muted">—</span></td>';
                    }
                    return `<td>${escapeHtml(String(cellValue))}</td>`;
                })
                .join('');
            return `<tr>${cells}</tr>`;
        })
        .join('');

    return `
        <table class="table table-sm table-striped">
            <thead>
                <tr>${headerHtml}</tr>
            </thead>
            <tbody>
                ${bodyHtml}
            </tbody>
        </table>
    `;
}

function formatMetricValue(value) {
    if (typeof value === 'number') {
        return value % 1 === 0 ? value.toString() : value.toFixed(2);
    }
    if (value === null || value === undefined) {
        return '—';
    }
    return escapeHtml(String(value));
}

function escapeHtml(value) {
    if (typeof value !== 'string') {
        return value;
    }
    return value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// Legacy output renderer (fallback)
function renderLegacyCellOutput(container, output) {
    let html = '';

    if (output.stdout) {
        html += `<div class="output-section">
            <pre class="stdout-output">${escapeHtml(output.stdout)}</pre>
        </div>`;
    }

    if (output.stderr) {
        html += `<div class="output-section">
            <div class="alert alert-warning">
                <strong>Warning/Error Output:</strong>
                <pre class="stderr-output">${escapeHtml(output.stderr)}</pre>
            </div>
        </div>`;
    }

    if (output.plots && output.plots.length > 0) {
        html += '<div class="output-section"><h6>Generated Plots:</h6><div class="plots-container">';
        output.plots.forEach((plot, index) => {
            html += `<div class="chart-container mb-3">
                <img src="data:image/png;base64,${plot}" alt="Plot ${index + 1}" class="notebook-plot-image">
            </div>`;
        });
        html += '</div></div>';
    }

    if (output.data_frames && output.data_frames.length > 0) {
        html += '<div class="output-section"><h6>Generated DataFrames:</h6>';
        output.data_frames.forEach(df => {
            html += `<div class="dataframe-container mb-3">
                <div class="table-responsive">
                    ${df.html}
                </div>
            </div>`;
        });
        html += '</div>';
    }

    if (!html) {
        html = '<p class="text-muted">No output generated</p>';
    }

    container.innerHTML = html;
}

// Delete analysis result card
function deleteCell(cellId) {
    if (confirm('Remove this analysis result?')) {
        const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
        if (cell) {
            cell.remove();
            if (window.EDAChatBridge && typeof window.EDAChatBridge.removeAnalysisCell === 'function') {
                window.EDAChatBridge.removeAnalysisCell(cellId);
            }
            showNotification('Analysis result removed.', 'info');
            updateAnalysisResultsPlaceholder();
        }
    }
}

// Get friendly analysis type name
function getAnalysisTypeName(analysisType) {
    const names = {
        // Data Quality & Structure
        'dataset_shape_analysis': 'Dataset Shape Analysis',
        'data_range_validation': 'Data Range Validation',
        'data_types_validation': 'Data Types Validation',
        'missing_value_analysis': 'Missing Value Analysis',
        'duplicate_detection': 'Duplicate Detection',
        
        // Univariate Analysis - Numeric
        'summary_statistics': 'Summary Statistics',
    'numeric_frequency_analysis': 'Numeric Frequency Analysis',
        'distribution_plots': 'Distribution Plots',
        'histogram_plots': 'Histogram Plots',
        'box_plots': 'Box Plots',
        'violin_plots': 'Violin Plots',
        'kde_plots': 'KDE (Density) Plots',
        'skewness_analysis': 'Skewness Analysis',
        'skewness_statistics': 'Skewness Statistics',
        'skewness_visualization': 'Skewness Visualization',
        'normality_test': 'Normality Test',
        
        // Univariate Analysis - Categorical
        'categorical_frequency_analysis': 'Categorical Frequency Analysis',
    'categorical_cardinality_profile': 'Categorical Cardinality Profile',
    'rare_category_detection': 'Rare Category Detection',
        'frequency_analysis': 'Frequency Analysis',
        'categorical_visualization': 'Categorical Visualization',
        'categorical_bar_charts': 'Categorical Bar Charts',
        'categorical_pie_charts': 'Categorical Pie Charts',
        
        // Bivariate/Multivariate Analysis
        'correlation_analysis': 'Correlation Analysis',
        'pearson_correlation': 'Pearson Correlation',
        'spearman_correlation': 'Spearman Correlation',
    'scatter_plot_analysis': 'Scatter Plot Analysis',
    'cross_tabulation_analysis': 'Cross Tabulation Analysis',
    'categorical_numeric_relationships': 'Categorical vs Numeric Explorer',
        
        // Outlier & Anomaly Detection
        'iqr_outlier_detection': 'IQR Outlier Detection',
        'iqr_detection': 'IQR Outlier Detection',
        'zscore_outlier_detection': 'Z-Score Outlier Detection',
        'zscore_detection': 'Z-Score Outlier Detection',
        'visual_outlier_inspection': 'Visual Outlier Inspection',
        'visual_inspection': 'Visual Outlier Inspection',
        'outlier_distribution_visualization': 'Outlier Distribution Plots',
        'outlier_scatter_matrix': 'Outlier Scatter Matrix',
        
        // Time-Series Exploration
        'temporal_trend_analysis': 'Temporal Trend Analysis',
        'temporal_trends': 'Temporal Trends',
        'seasonality_detection': 'Seasonality Detection',
    'datetime_feature_extraction': 'Datetime Feature Extraction',

        // Geospatial Analysis
        'coordinate_system_projection_check': 'Coordinate System & Projection Check',
        'spatial_distribution_analysis': 'Spatial Distribution Analysis',
        'spatial_relationships_analysis': 'Spatial Relationships Analysis',

        // Text Analysis
        'text_length_distribution': 'Text Length Distribution',
        'text_token_frequency': 'Text Token Frequency',
    'text_vocabulary_summary': 'Text Vocabulary Summary',
    'text_feature_engineering_profile': 'Text Feature Engineering Profile',
    'text_nlp_profile': 'Advanced NLP Profile',
        
        // Relationship Exploration
        'multicollinearity_analysis': 'Multicollinearity Analysis',
        'pca_dimensionality_reduction': 'PCA Analysis',
        'network_analysis': 'Network Analysis',
        'entity_relationship_network': 'Entity Network Analysis',
        'pca_analysis': 'PCA Analysis',
        'pca_scree_plot': 'PCA Scree Plot',
        'pca_cumulative_variance': 'PCA Cumulative Variance',
        'pca_visualization': 'PCA Visualization',
        'pca_biplot': 'PCA Biplot',
        'pca_heatmap': 'PCA Loadings Heatmap',
    'cluster_tendency_analysis': 'Cluster Tendency Analysis',
    'cluster_segmentation_analysis': 'Cluster Segmentation',
        
        // Marketing Analysis Components
        'campaign_metrics_analysis': 'Campaign Metrics Analysis',
        'conversion_funnel_analysis': 'Conversion Funnel Analysis',
        'engagement_analysis': 'Engagement Analysis',
        'channel_performance_analysis': 'Channel Performance Analysis',
        'audience_segmentation_analysis': 'Audience Segmentation Analysis',
        'roi_analysis': 'ROI Analysis',
        'attribution_analysis': 'Attribution Analysis',
        'cohort_analysis': 'Cohort Analysis',
        
        // Legacy names for backward compatibility
        'data_quality_structure': 'Data Quality & Structure Check',
        'missing_data_analysis': 'Missing Data Analysis',
        'data_profiling': 'Comprehensive Data Profiling',
        'univariate_numeric': 'Numeric Variables Analysis',
        'univariate_categorical': 'Categorical Variables Analysis',
        'distribution_analysis': 'Distribution Analysis',
        'bivariate_multivariate': 'Relationships & Interactions',
        'relationship_exploration': 'Advanced Relationships',
        'advanced_outlier_detection': 'Advanced Outlier Detection',
        'outlier_detection': 'Basic Outlier Detection',
        'time_series_exploration': 'Time Series Analysis',
        'data_overview': 'Data Overview & Summary',
        'data_types': 'Data Types & Info',
        'missing_values': 'Missing Values Analysis',
        'descriptive_stats': 'Descriptive Statistics',
        'feature_importance': 'Feature Importance',
        'clustering_analysis': 'Clustering Analysis',
        'classification_eda': 'Classification EDA',
        'regression_eda': 'Regression EDA',
        'time_series_eda': 'Time Series EDA',
        'fraud_detection_eda': 'Fraud Detection EDA'
    };
    
    return names[analysisType] || analysisType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Add markdown cell
function addMarkdownCell() {
    console.log('addMarkdownCell called');
    const cellId = `markdown-${cellCounter++}`;
    const markdownHTML = `
    <div class="notebook-cell" data-cell-type="markdown" data-cell-id="${cellId}">
        <div class="cell-container">
            <div class="cell-header">
                <div class="cell-type-indicator markdown-cell">
                    <i class="bi bi-markdown"></i> Markdown
                </div>
                <div class="cell-actions">
                    <button class="btn btn-sm btn-outline-secondary" onclick="moveCellUp('${cellId}')" title="Move Up">
                        <i class="bi bi-arrow-up"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="moveCellDown('${cellId}')" title="Move Down">
                        <i class="bi bi-arrow-down"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="editMarkdownCell('${cellId}')">
                        <i class="bi bi-pencil"></i> Edit
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteCell('${cellId}')">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
            <div class="cell-content">
                <textarea class="markdown-editor markdown-editor-min-height" id="markdown-${cellId}" placeholder="Enter markdown content...">## Analysis Notes

Add your markdown content here...</textarea>
                <div class="markdown-preview d-none" id="preview-${cellId}">
                    <h2>Analysis Notes</h2>
                    <p>Add your markdown content here...</p>
                </div>
            </div>
        </div>
    </div>`;
    
    const cellsContainer = document.getElementById('notebookCells');
    cellsContainer.insertAdjacentHTML('afterbegin', markdownHTML);
    updateAnalysisResultsPlaceholder();
    
    showNotification('Markdown cell added at top!', 'success');
    
    // Scroll to new cell
    const newCell = document.querySelector(`[data-cell-id="${cellId}"]`);
    newCell.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Edit markdown cell
function editMarkdownCell(cellId) {
    const editor = document.getElementById(`markdown-${cellId}`);
    const preview = document.getElementById(`preview-${cellId}`);
    
    if (editor.style.display === 'none') {
        // Switch to edit mode
        editor.style.display = 'block';
        preview.style.display = 'none';
    } else {
        // Switch to preview mode
        editor.style.display = 'none';
        preview.style.display = 'block';
        // Here you would typically convert markdown to HTML
        preview.innerHTML = `<pre>${editor.value}</pre>`;
    }
}

let lastDomainDetectionResult = null;
let isDomainDetectionLoading = false;

// Helper to format domain labels consistently
function formatDomainLabel(domain) {
    if (!domain) {
        return 'Unknown';
    }

    try {
        return domain
            .toString()
            .replace(/[_-]+/g, ' ')
            .replace(/\s+/g, ' ')
            .trim()
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    } catch (error) {
        console.warn('Unable to format domain label:', domain, error);
        return String(domain);
    }
}

function handleDomainButtonClick(event) {
    const wantsRefresh = event && (event.shiftKey || event.metaKey || event.ctrlKey);

    if (isDomainDetectionLoading) {
        console.log('Domain detection is already running, ignoring click');
        return;
    }

    if (wantsRefresh || !lastDomainDetectionResult) {
        loadDomainDetection(true, { forceRefresh: Boolean(wantsRefresh) });
        return;
    }

    openDomainModalWithResult(lastDomainDetectionResult);
}

// Load and display domain detection results
async function loadDomainDetection(manualTrigger = false, options = {}) {
    const { forceRefresh = false } = options;
    console.log('loadDomainDetection called with sourceId:', sourceId, 'manualTrigger:', manualTrigger, 'forceRefresh:', forceRefresh);

    const detectButton = document.getElementById('domainDetectButton');
    const labelEl = document.getElementById('domainButtonLabel');
    const confidenceEl = document.getElementById('domainButtonConfidence');
    const separatorEl = document.getElementById('domainButtonSeparator');
    const spinnerEl = document.getElementById('domainButtonSpinner');

    if (!detectButton || !labelEl) {
        console.warn('Domain detection UI not found');
        return null;
    }

    if (!sourceId) {
        labelEl.textContent = 'No dataset';
        if (confidenceEl) confidenceEl.classList.add('d-none');
        if (separatorEl) separatorEl.classList.add('d-none');
        detectButton.disabled = true;
        detectButton.classList.add('domain-compact-button--error');
        return null;
    }

    isDomainDetectionLoading = true;
    detectButton.disabled = true;
    detectButton.classList.remove('domain-compact-button--error');
    detectButton.classList.add('domain-compact-button--loading');
    detectButton.setAttribute('data-loading', 'true');

    if (spinnerEl) {
        spinnerEl.classList.remove('d-none');
    }

    labelEl.textContent = manualTrigger ? (forceRefresh ? 'Refreshing…' : 'Rechecking…') : 'Detecting…';
    if (confidenceEl) confidenceEl.classList.add('d-none');
    if (separatorEl) separatorEl.classList.add('d-none');

    try {
        console.log(`Calling domain detection API: /advanced-eda/api/detect-domain/${sourceId}`);
        const response = await fetch(`/advanced-eda/api/detect-domain/${sourceId}`);
        console.log('Domain detection response status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Domain detection API error:', errorText);

            if (response.status === 404) {
                throw new Error('Dataset not found. Please verify the data source.');
            }

            throw new Error(`Failed with status ${response.status}`);
        }

        const result = await response.json();
        console.log('Domain detection result:', result);

        if (result.success && (result.primary_domain || result.domain)) {
            const domainNameRaw = result.primary_domain || result.domain;
            const formattedDomain = formatDomainLabel(domainNameRaw);
            const confidenceScore = result.primary_confidence ?? result.confidence ?? 0;
            const confidencePercent = Math.round(confidenceScore * 100);

            labelEl.textContent = formattedDomain;

            if (confidenceEl) {
                confidenceEl.textContent = `${confidencePercent}% confidence`;
                confidenceEl.classList.remove('d-none');
            }

            if (separatorEl) {
                separatorEl.classList.remove('d-none');
            }

            detectButton.dataset.domain = formattedDomain;
            detectButton.dataset.confidence = confidencePercent;
            detectButton.setAttribute('data-has-domain', 'true');
            detectButton.title = `Detected domain: ${formattedDomain}. Click to view details. Hold Shift to refresh.`;

            lastDomainDetectionResult = result;

            if (Array.isArray(result.recommendations)) {
                displayDomainSpecificRecommendations(result.recommendations, formattedDomain);
            }

            if (manualTrigger) {
                openDomainModalWithResult(result);
            }

            return result;
        }

        throw new Error(result.error || 'Domain detection failed');
    } catch (error) {
        console.error('Domain detection error:', error);
        lastDomainDetectionResult = null;
    delete detectButton.dataset.domain;
    delete detectButton.dataset.confidence;
    detectButton.removeAttribute('data-has-domain');
        labelEl.textContent = 'Retry detection';
        if (confidenceEl) {
            confidenceEl.textContent = 'Tap to try again';
            confidenceEl.classList.remove('d-none');
        }
        if (separatorEl) {
            separatorEl.classList.remove('d-none');
        }
        detectButton.classList.add('domain-compact-button--error');
        detectButton.title = `Domain detection failed: ${error.message}`;
        throw error;
    } finally {
        isDomainDetectionLoading = false;
        detectButton.disabled = false;
        detectButton.classList.remove('domain-compact-button--loading');
        detectButton.removeAttribute('data-loading');
        if (spinnerEl) {
            spinnerEl.classList.add('d-none');
        }
    }
}

// Display domain-specific recommendations in the sidebar (compact tracking)
function displayDomainSpecificRecommendations(recommendations, domainName) {
    console.log('Domain-specific recommendations available for', domainName, recommendations);

    const detectButton = document.getElementById('domainDetectButton');
    if (detectButton) {
        detectButton.dataset.recommendationsCount = Array.isArray(recommendations) ? recommendations.length : 0;
    }
}

function cleanupDomainModalArtifacts() {
    try {
        document.body.classList.remove('modal-open');
    } catch (err) {
        // ignore
    }

    try {
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(backdrop => backdrop.remove());
    } catch (err) {
        // ignore
    }
}

function openDomainModalWithResult(result) {
    if (!result) {
        console.warn('Cannot open domain modal without a result');
        return;
    }

    const modalElement = document.getElementById('domainRecommendationsModal');
    const modalContent = document.getElementById('domainRecommendationsContent');

    if (!modalElement || !modalContent) {
        console.warn('Domain recommendations modal elements not found');
        return;
    }

    modalContent.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading domain summary...</span>
            </div>
            <p class="mt-3 text-muted">Preparing domain summary…</p>
        </div>
    `;

    let modalInstance = null;
    try {
        if (typeof bootstrap === 'undefined' || !bootstrap.Modal) {
            throw new Error('Bootstrap modal library not available');
        }

        modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
        modalInstance.show();
    } catch (err) {
        console.error('Failed to open domain recommendations modal:', err);
        cleanupDomainModalArtifacts();
        if (typeof showNotification === 'function') {
            showNotification('Domain recommendations are unavailable. Please try again in a moment.', 'error');
        }
        return;
    }

    modalElement.addEventListener('hidden.bs.modal', () => {
        cleanupDomainModalArtifacts();
    }, { once: true });

    requestAnimationFrame(() => {
        try {
            displayDetailedDomainRecommendations(result);
        } catch (renderError) {
            console.error('Failed to render domain recommendations:', renderError);
            modalContent.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    Unable to render domain recommendations. Please try again.
                </div>
            `;
        }
    });
}

function forceRefreshDomainDetection() {
    if (isDomainDetectionLoading) {
        return;
    }

    loadDomainDetection(true, { forceRefresh: true });
}

// Load and display domain analysis
async function loadDomainAnalysis(templateName = 'data_quality_structure') {
    console.log('loadDomainAnalysis called with sourceId:', sourceId, 'template:', templateName);
    if (!sourceId) {
        alert('No dataset ID available for domain analysis');
        return;
    }

    const domainSection = document.getElementById('domainAnalysisSection');
    if (domainSection) {
        domainSection.classList.remove('d-none');
        domainSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    try {
        console.log(`Calling domain analysis API: /advanced-eda/api/domain-analysis/${sourceId}?template_name=${templateName}`);
        const response = await fetch(`/advanced-eda/api/domain-analysis/${sourceId}?template_name=${templateName}&analysis_depth=intermediate`);
        console.log('Domain analysis response status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Domain analysis API error:', errorText);
            throw new Error(`Failed to load domain analysis: ${response.status}`);
        }

        const result = await response.json();
        console.log('Domain analysis result:', result);

        if (result.success) {
            displayDomainAnalysisResults(result);
        } else {
            throw new Error(result.error || 'Domain analysis failed');
        }
    } catch (error) {
        console.error('Domain analysis error:', error);
        const analysisContent = document.getElementById('domainAnalysisContent');
        if (analysisContent) {
            analysisContent.innerHTML = `<div class="alert alert-danger">Failed to load domain analysis: ${error.message}</div>`;
        }
    }
}

// Load and display domain recommendations
async function loadDomainRecommendations() {
    console.log('loadDomainRecommendations called with sourceId:', sourceId);

    if (isDomainDetectionLoading) {
        console.log('Domain detection currently running; please wait until it finishes.');
        return;
    }

    if (!sourceId) {
        alert('No dataset ID available for domain recommendations');
        return;
    }

    if (lastDomainDetectionResult) {
        openDomainModalWithResult(lastDomainDetectionResult);
        return;
    }

    try {
        const result = await loadDomainDetection(true, { forceRefresh: true });
        if (result) {
            openDomainModalWithResult(result);
        }
    } catch (error) {
        console.error('Unable to load domain recommendations:', error);
        showNotification(`Domain recommendations failed: ${error.message}`, 'error');
    }
}

// Cell movement functions
function moveCellUp(cellId) {
    console.log('Move up called for cell:', cellId);
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (cell) {
        const previousSibling = cell.previousElementSibling;
        if (previousSibling) {
            cell.parentNode.insertBefore(cell, previousSibling);
            showNotification('Cell moved up', 'success');
        } else {
            showNotification('Cell is already at the top', 'info');
        }
    } else {
        console.error('Cell not found with ID:', cellId);
        showNotification('Error: Cell not found', 'error');
    }
}

function moveCellDown(cellId) {
    console.log('Move down called for cell:', cellId);
    const cell = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (cell) {
        const nextSibling = cell.nextElementSibling;
        if (nextSibling) {
            cell.parentNode.insertBefore(nextSibling, cell);
            showNotification('Cell moved down', 'success');
        } else {
            showNotification('Cell is already at the bottom', 'info');
        }
    } else {
        console.error('Cell not found with ID:', cellId);
        showNotification('Error: Cell not found', 'error');
    }
}

function updateAnalysisResultsPlaceholder() {
    const container = document.getElementById('notebookCells');
    const placeholder = document.getElementById('analysisResultsPlaceholder');

    if (!container || !placeholder) {
        return;
    }

    const hasCells = container.children.length > 0;
    placeholder.classList.toggle('d-none', hasCells);
    placeholder.setAttribute('aria-hidden', hasCells ? 'true' : 'false');
}

function clearAllCells() {
    const cellsContainer = document.getElementById('notebookCells');
    if (cellsContainer) {
        if (confirm('Clear all analysis results? This action cannot be undone.')) {
            cellsContainer.innerHTML = '';
            cellCounter = 1;
            executionCounter = 1;
            showNotification('Analysis results cleared. Dataset and preprocessing stay as-is.', 'success');
            updateAnalysisResultsPlaceholder();
            updatePreprocessingStatusBadge(lastPreprocessingReport);
            refreshCategoryLocks();
        }
    }
}

// Display detailed domain recommendations in modal
function displayDetailedDomainRecommendations(domainResult) {
    const recommendationsContent = document.getElementById('domainRecommendationsContent');
    if (!recommendationsContent) return;
    
    const primaryDomain = domainResult.primary_domain || domainResult.domain || 'general';
    const primaryConfidence = domainResult.primary_confidence || domainResult.confidence || 0;
    const secondaryDomains = domainResult.secondary_domains || [];
    const recommendations = domainResult.recommendations || [];
    const allScores = domainResult.all_scores || domainResult.domain_scores || {};
    const formattedPrimaryDomain = formatDomainLabel(primaryDomain);
    const primaryConfidencePercent = Math.round(primaryConfidence * 100);
    
    let html = '';
    
    html += `
        <div class="d-flex flex-column flex-md-row justify-content-between align-items-start align-items-md-center gap-3 mb-4">
            <div>
                <h5 class="mb-1">Domain overview</h5>
                <p class="text-muted small mb-0">Confidence scores and recommended flows for ${formattedPrimaryDomain} data.</p>
            </div>
            <div class="d-flex gap-2">
                <button class="btn btn-outline-secondary btn-sm" onclick="forceRefreshDomainDetection()">
                    <i class="bi bi-arrow-clockwise"></i> Re-run detection
                </button>
            </div>
        </div>
    `;
    
    // Safety disclaimer with exclamation icon
    html += `
        <div class="alert alert-warning domain-warning-alert d-flex align-items-start mb-4" role="alert">
            <i class="bi bi-exclamation-triangle-fill text-warning me-3 fs-4" style="margin-top: 2px;"></i>
            <div>
                <strong>Important Notice:</strong> Domain predictions are not 100% accurate and should be used as guidance only. 
                Please investigate the results carefully for your safety and verify the appropriateness of the recommended analysis 
                methods for your specific dataset and use case.
            </div>
        </div>
    `;
    
    // Primary and Secondary Domain Detection Summary
    html += `
        <div class="mb-4">
            <div class="card border-primary domain-detection-card">
                <div class="card-header bg-primary text-white py-2 domain-detection-card__header">
                    <h6 class="mb-0">
                        <i class="bi bi-bullseye"></i> Domain Detection Results
                    </h6>
                </div>
                <div class="card-body py-3">
                    <!-- Primary Domain -->
                    <div class="mb-3">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="fw-bold text-primary">Primary Domain: ${formattedPrimaryDomain}</span>
                            <span class="badge bg-primary">${primaryConfidencePercent}% confidence</span>
                        </div>
                        <div class="progress domain-confidence-progress">
                            <div class="progress-bar bg-primary" style="width: ${primaryConfidencePercent}%"></div>
                        </div>
                    </div>
    `;
    
    // Secondary Domains
    if (secondaryDomains.length > 0) {
        html += `
                    <div class="border-top pt-3">
                        <h6 class="small text-muted mb-3">Secondary Domain Predictions:</h6>
        `;
        
        secondaryDomains.forEach((secondary, index) => {
            const nameValue = secondary.domain || secondary.name || secondary;
            const confidence = secondary.confidence || secondary.score || 0;
            const formattedName = formatDomainLabel(nameValue);
            const confidencePercent = Math.round(confidence * 100);
            html += `
                        <div class="mb-2">
                            <div class="d-flex justify-content-between align-items-center mb-1">
                                <span class="small">${formattedName}</span>
                                <span class="small text-muted">${confidencePercent}%</span>
                            </div>
                            <div class="progress secondary-domain-progress">
                                <div class="progress-bar bg-secondary" style="width: ${confidencePercent}%"></div>
                            </div>
                        </div>
            `;
        });
        
        html += `
                    </div>
        `;
    }
    
    html += `
                </div>
            </div>
        </div>
    `;
    
    // Domain-specific recommendations from the analyzer
    if (recommendations.length > 0) {
        html += `
            <div class="mb-4">
                <h6 class="domain-recommendations-title"><i class="bi bi-lightbulb"></i> Domain-Specific Recommendations</h6>
                <div class="card border-success domain-recommendations-card">
                    <div class="card-body py-3">
                        <ul class="list-group list-group-flush domain-recommendations-list">
        `;
        
        recommendations.forEach((rec, index) => {
            html += `
                <li class="list-group-item border-0 py-2 px-0">
                    <div class="d-flex align-items-start">
                        <i class="bi bi-check-circle-fill text-success me-2 mt-1 flex-shrink-0"></i>
                        <span class="small">${rec}</span>
                    </div>
                </li>
            `;
        });
        
        html += `
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Analysis workflow recommendations based on primary domain
    const workflowRecommendations = getDomainSpecificWorkflow(primaryDomain);
    if (workflowRecommendations.length > 0) {
        html += `
            <div class="mb-4">
                <h6><i class="bi bi-diagram-3"></i> Recommended Analysis Workflow</h6>
                <div class="row">
        `;
        
        workflowRecommendations.forEach((step, index) => {
            html += `
                <div class="col-md-6 mb-2">
                    <div class="card border-light h-100 recommendation-card recommendation-card-clickable" onclick="applyRecommendation('${step.template}')">
                        <div class="card-body py-2">
                            <div class="d-flex align-items-center">
                                <div class="badge bg-primary rounded-pill me-2 flex-shrink-0">${index + 1}</div>
                                <div>
                                    <h6 class="card-title small mb-1">${step.title}</h6>
                                    <p class="card-text small text-muted mb-0">${step.description}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += `</div></div>`;
    }
    
    // Show domain scores if available
    if (Object.keys(allScores).length > 0) {
        const topScores = Object.entries(allScores)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5);
        const maxScore = topScores.length ? Math.max(...topScores.map(([, score]) => score)) : 1;
        const safeMaxScore = maxScore === 0 ? 1 : maxScore;
            
        html += `
            <div class="mb-3">
                <h6><i class="bi bi-bar-chart"></i> Domain Detection Scores</h6>
                <div class="small">
        `;
        
        topScores.forEach(([domain, score]) => {
            const percentage = (score / safeMaxScore) * 100;
            const formattedDomain = formatDomainLabel(domain);
            html += `
                <div class="mb-2">
                    <div class="d-flex justify-content-between">
                        <span>${formattedDomain}</span>
                        <span>${score.toFixed(1)}</span>
                    </div>
                    <div class="progress domain-score-progress">
                        <div class="progress-bar ${formatDomainLabel(domain) === formattedPrimaryDomain ? 'bg-primary' : 'bg-secondary'}" style="width: ${Math.min(percentage, 100)}%"></div>
                    </div>
                </div>
            `;
        });
        
        html += `</div></div>`;
    }
    
    recommendationsContent.innerHTML = html;
}

// Get domain-specific workflow recommendations
function getDomainSpecificWorkflow(domainName) {
    const workflows = {
        'fraud': [
            { title: "Data Quality Check", description: "Check for missing values and anomalies", template: "data_quality_structure" },
            { title: "Advanced Outlier Detection", description: "Detect fraudulent transactions using advanced methods", template: "advanced_outlier_detection" },
            { title: "Feature Analysis", description: "Analyze transaction patterns and amounts", template: "univariate_numeric" },
            { title: "Relationship Exploration", description: "Discover hidden fraud patterns using clustering", template: "relationship_exploration" },
            { title: "Correlation Analysis", description: "Find relationships between fraud indicators", template: "correlation_analysis" },
            { title: "Time Series Analysis", description: "Analyze fraud trends over time", template: "time_series_exploration" }
        ],
        'healthcare': [
            { title: "Data Quality Assessment", description: "Analyze healthcare data quality and completeness", template: "data_quality_structure" },
            { title: "Patient Demographics", description: "Analyze patient population characteristics", template: "univariate_categorical" },
            { title: "Treatment Outcomes", description: "Examine treatment effectiveness patterns", template: "distribution_analysis" },
            { title: "Medical Correlations", description: "Find relationships between symptoms and outcomes", template: "bivariate_multivariate" },
            { title: "Missing Data Analysis", description: "Check medical record completeness", template: "missing_data_analysis" },
            { title: "Pattern Discovery", description: "Discover patient clusters and treatment patterns", template: "relationship_exploration" }
        ],
        'finance': [
            { title: "Data Quality Assessment", description: "Comprehensive data quality validation", template: "data_quality_structure" },
            { title: "Financial Metrics Analysis", description: "Comprehensive numeric analysis of financial data", template: "univariate_numeric" },
            { title: "Time Series Analysis", description: "Analyze financial trends and patterns over time", template: "time_series_exploration" },
            { title: "Market Correlations", description: "Find relationships between financial metrics", template: "bivariate_multivariate" },
            { title: "Advanced Outlier Detection", description: "Detect anomalous financial transactions", template: "advanced_outlier_detection" },
            { title: "Pattern Discovery", description: "Discover market patterns using advanced analytics", template: "relationship_exploration" }
        ],
        'retail': [
            { title: "Data Quality Check", description: "Validate retail data integrity", template: "data_quality_structure" },
            { title: "Sales Patterns", description: "Analyze sales and customer behavior", template: "univariate_numeric" },
            { title: "Customer Analysis", description: "Product performance and categorical analysis", template: "univariate_categorical" },
            { title: "Time Series Analysis", description: "Identify seasonal patterns and trends", template: "time_series_exploration" },
            { title: "Customer Relationships", description: "Discover customer segments and buying patterns", template: "relationship_exploration" },
            { title: "Market Correlations", description: "Find relationships between products and sales", template: "bivariate_multivariate" }
        ],
        'marketing': [
            { title: "Data Quality Assessment", description: "Validate marketing data completeness", template: "data_quality_structure" },
            { title: "Campaign Analysis", description: "Analyze campaign performance metrics", template: "univariate_numeric" },
            { title: "Customer Segmentation", description: "Group customers by demographics and behavior", template: "univariate_categorical" },
            { title: "Multi-channel Analysis", description: "Analyze relationships between marketing channels", template: "bivariate_multivariate" },
            { title: "Customer Journey", description: "Track customer interactions over time", template: "time_series_exploration" },
            { title: "Pattern Discovery", description: "Discover hidden customer patterns and clusters", template: "relationship_exploration" }
        ],
        'iot': [
            { title: "Sensor Data Quality", description: "Check sensor data quality and completeness", template: "data_quality_structure" },
            { title: "Time Series Analysis", description: "Analyze sensor readings over time", template: "time_series_exploration" },
            { title: "Anomaly Detection", description: "Detect sensor anomalies and failures", template: "advanced_outlier_detection" },
            { title: "Sensor Correlations", description: "Find relationships between sensor readings", template: "bivariate_multivariate" },
            { title: "Pattern Discovery", description: "Discover operational patterns in IoT data", template: "relationship_exploration" },
            { title: "Missing Data Analysis", description: "Handle sensor data gaps and outages", template: "missing_data_analysis" }
        ]
    };
    
    return workflows[domainName] || [
        { title: "Data Quality Assessment", description: "Start with comprehensive data quality and structure analysis", template: "data_quality_structure" },
        { title: "Numeric Analysis", description: "Comprehensive analysis of numeric variables", template: "univariate_numeric" },
        { title: "Categorical Analysis", description: "Analyze categorical variables and distributions", template: "univariate_categorical" },
        { title: "Relationship Analysis", description: "Discover relationships between variables", template: "bivariate_multivariate" },
        { title: "Pattern Discovery", description: "Advanced pattern discovery and clustering", template: "relationship_exploration" },
        { title: "Outlier Detection", description: "Identify anomalies and outliers", template: "advanced_outlier_detection" },
        { title: "Time Series Analysis", description: "Analyze temporal patterns (if time data exists)", template: "time_series_exploration" }
    ];
}

// Display domain recommendations (legacy function - updated to use new approach)
function displayDomainRecommendations(result) {
    const recommendationsContent = document.getElementById('domainRecommendationsContent');
    if (!recommendationsContent) return;
    
    let html = '';
    
    // Analysis recommendations based on detected domain
    html += `
        <div class="mb-3">
            <h6><i class="bi bi-lightbulb"></i> Recommended Analysis Approaches</h6>
            <p class="text-muted small">Based on your detected domain type and data characteristics</p>
        </div>
    `;
    
    // Template-based recommendations
    if (result.template_name) {
        html += `
            <div class="mb-3">
                <div class="card border-primary">
                    <div class="card-header bg-primary text-white py-2">
                        <h6 class="mb-0">Primary Recommendation: ${result.template_name}</h6>
                    </div>
                    <div class="card-body py-2">
                        <p class="small mb-0">This template is most suitable for your dataset based on domain analysis.</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Generate specific recommendations based on common analysis patterns
    const recommendations = [
        {
            title: "Data Quality Assessment",
            description: "Start with comprehensive data quality and structure analysis",
            template: "data_quality_structure"
        },
        {
            title: "Numeric Variable Analysis", 
            description: "Analyze numeric variables with advanced statistics and distributions",
            template: "univariate_numeric"
        },
        {
            title: "Categorical Analysis",
            description: "Comprehensive analysis of categorical variables and their distributions", 
            template: "univariate_categorical"
        },
        {
            title: "Missing Data Analysis",
            description: "Identify missing values and data quality issues",
            template: "missing_data_analysis"
        },
        {
            title: "Correlation Analysis",
            description: "Discover relationships between variables",
            template: "correlation_analysis"
        },
        {
            title: "Distribution Analysis",
            description: "Analyze the distribution patterns of your variables",
            template: "distribution_analysis"
        }
    ];
    
    html += `
        <div class="mb-3">
            <h6><i class="bi bi-list-check"></i> Suggested Analysis Steps</h6>
            <div class="row">
    `;
    
    recommendations.forEach(rec => {
        html += `
            <div class="col-md-6 mb-2">
                <div class="card border-light h-100 recommendation-card recommendation-card-clickable" onclick="applyRecommendation('${rec.template}')">
                    <div class="card-body py-2">
                        <div class="d-flex align-items-center">
                            <div>
                                <h6 class="card-title small mb-1">${rec.title}</h6>
                                <p class="card-text small text-muted mb-0">${rec.description}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += `</div></div>`;
    
    // Additional domain-specific tips
    html += `
        <div class="mb-3">
            <h6><i class="bi bi-info-circle"></i> Tips for Better Analysis</h6>
            <ul class="small mb-0">
                <li>Start with data exploration to understand your dataset structure</li>
                <li>Check for missing values and outliers before advanced analysis</li>
                <li>Use appropriate visualization techniques for your data types</li>
                <li>Consider domain-specific metrics and KPIs</li>
            </ul>
        </div>
    `;
    
    recommendationsContent.innerHTML = html;
}

// Apply a recommendation (select analysis type and add cell)
function applyRecommendation(templateName) {
    console.log('Applying recommendation:', templateName);
    
    // Find and select the analysis type
    const analysisOption = document.querySelector(`[data-value="${templateName}"]`);
    if (analysisOption) {
        if (typeof selectAnalysisType === 'function') {
            selectAnalysisType(analysisOption);
        } else {
            triggerAnalysisRun(templateName, analysisOption);
        }
        
        // Show success message
        showNotification(`Applied recommendation: ${analysisOption.getAttribute('data-name')}`, 'success');
        
        // Close recommendations modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('domainRecommendationsModal'));
        if (modal) {
            modal.hide();
        }
        
        // Scroll to analysis types section after modal is hidden
        setTimeout(() => {
            const analysisGrid = document.getElementById('analysisGrid');
            if (analysisGrid) {
                analysisGrid.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 300);
    }
}

// Close domain recommendations section
function closeDomainRecommendations() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('domainRecommendationsModal'));
    if (modal) {
        modal.hide();
    }
}

// Export functions for global access
// ============================================================================
// COLUMN INSIGHTS FUNCTIONALITY
// ============================================================================

// Global variables for column insights
let columnInsightsData = null;
