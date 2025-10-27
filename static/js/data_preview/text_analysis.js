/**
 * text_analysis.js
 * ================
 * 
 * Text Analysis Module for Data Preview Page
 * 
 * Purpose:
 *  - Render text analysis reports
 *  - Display text column statistics
 *  - Show NLP recommendations
 *  - Provide text categorization insights
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    global.DI.textAnalysis = global.DI.textAnalysis || {};

    function escapeHtml(value) {
        if (value == null) {
            return '';
        }
        return String(value).replace(/[&<>"']/g, function(match) {
            const escapeMap = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;'
            };
            return escapeMap[match] || match;
        });
    }

    function asNumber(value) {
        const num = Number(value);
        return Number.isFinite(num) ? num : null;
    }

    function numberOr(value, fallback = 0) {
        const num = asNumber(value);
        return num === null ? fallback : num;
    }

    function formatPercentageDisplay(value, fractionDigits = 1, fallback = '0%') {
        const num = asNumber(value);
        if (num === null) {
            return fallback;
        }
        const clamped = Math.min(100, Math.max(0, num));
        return `${clamped.toFixed(fractionDigits)}%`;
    }

    function formatRange(minValue, maxValue) {
        const min = numberOr(minValue, 0);
        const max = numberOr(maxValue, min);
        return `${min.toLocaleString()} - ${max.toLocaleString()}`;
    }

    function formatColumnList(columns, formatter) {
        if (!Array.isArray(columns) || columns.length === 0) {
            return '';
        }

        return columns.map(col => {
            const label = formatter ? formatter(col) : (col && col.name);
            return `<span class="column-chip">${escapeHtml(label || '—')}</span>`;
        }).join('');
    }

    function formatColumnNameChips(columnNames) {
        if (!Array.isArray(columnNames) || columnNames.length === 0) {
            return '';
        }

        return columnNames.map(name => `<span class="column-chip">${escapeHtml(name)}</span>`).join('');
    }

    function getPatternIcon(patternType) {
        const map = {
            email: 'fas fa-envelope',
            url: 'fas fa-link',
            phone: 'fas fa-phone',
            date: 'fas fa-calendar-alt'
        };
        return map[patternType] || 'fas fa-search';
    }
    
    /**
     * Render text analysis in the specified container
     */
    function renderTextAnalysis(qualityReport, containerId) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('Container not found:', containerId);
            return;
        }
        
        const textSummary = qualityReport.text_analysis_summary || {};
        const columnDetails = qualityReport.quality_metrics?.column_details || [];
        const textColumns = columnDetails.filter(col => col.data_category === 'text');
        
        let html = `
            <div class="quality-section text-analysis-section">
                <h4><i class="fas fa-font"></i> Text Column Analysis</h4>
        `;
        
        // Text Analysis Summary
        html += renderTextSummary(textSummary);
        
        // Text Column Details
        if (textColumns.length > 0) {
            html += renderTextColumnDetails(textColumns);
        } else {
            html += `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i>
                    No text columns detected in this dataset.
                </div>
            `;
        }
        
        // Aggregated pattern & quality insights
        html += renderDetectedPatternSummary(textSummary);
        html += renderTextQualityAlerts(textSummary);

        // Column-level governance notices (PII)
        html += renderPiiWarnings(textSummary);

        // Text Processing Recommendations
        html += renderTextProcessingRecommendations(textColumns);
        
        html += '</div>';
        
        container.innerHTML = html;
    }
    
    /**
     * Render text analysis summary
     */
    function renderTextSummary(textSummary) {
        const totalTextColumns = numberOr(textSummary.total_text_columns, 0);
        const freeTextColumns = numberOr(textSummary.free_text_columns, 0);
        const categoricalTextColumns = numberOr(textSummary.categorical_text_columns, 0);

        let html = `
            <div class="text-analysis-summary">
                <div class="text-stat-grid">
                    <div class="text-stat-card">
                        <div class="text-stat-value">${totalTextColumns.toLocaleString()}</div>
                        <div class="text-stat-label">Text Columns</div>
                    </div>
                    <div class="text-stat-card nlp-card">
                        <div class="text-stat-value">${freeTextColumns.toLocaleString()}</div>
                        <div class="text-stat-label">NLP Candidates</div>
                    </div>
                    <div class="text-stat-card categorical-card">
                        <div class="text-stat-value">${categoricalTextColumns.toLocaleString()}</div>
                        <div class="text-stat-label">Categorical Text</div>
                    </div>
                </div>
            </div>
        `;
        
        return html;
    }
    
    /**
     * Render text column details
     */
    function renderTextColumnDetails(textColumns) {
        let html = `
            <div class="text-columns-details">
                <h5><i class="fas fa-list"></i> Text Column Details</h5>
                <div class="text-columns-grid">
        `;
        
        textColumns.forEach(col => {
            const isNLP = col.text_category === 'free_text' || col.text_category === 'descriptive_text';
            const isCategorical = col.text_category === 'categorical';
            const cardClass = isNLP ? 'text-column-card nlp-candidate' : 
                             isCategorical ? 'text-column-card categorical-text' : 
                             'text-column-card mixed-text';

            const columnName = escapeHtml(col.name || 'Unknown column');
            const textCategory = escapeHtml(col.text_category || 'text');
            const avgLength = numberOr(col.avg_text_length, 0);
            const rangeLabel = formatRange(col.min_text_length, col.max_text_length);
            const uniqueCount = numberOr(col.unique_count, 0);
            const uniquePercentageDisplay = col.unique_percentage != null
                ? formatPercentageDisplay(col.unique_percentage)
                : null;
            const completenessDisplay = formatPercentageDisplay(100 - numberOr(col.null_percentage, 0));
            const columnPatterns = Array.isArray(col.text_patterns) ? col.text_patterns : [];
            const columnFlags = Array.isArray(col.text_quality_flags) ? col.text_quality_flags : [];

            html += `
                <div class="${cardClass}">
                    <div class="text-column-header">
                        <h6><i class="fas fa-font"></i> ${columnName}</h6>
                        <span class="text-category-badge">${textCategory}</span>
                    </div>
                    <div class="text-column-metrics">
                        <div class="text-metric">
                            <span class="metric-label">Avg Length:</span>
                            <span class="metric-value">${avgLength.toLocaleString()} chars</span>
                        </div>
                        <div class="text-metric">
                            <span class="metric-label">Range:</span>
                            <span class="metric-value">${escapeHtml(rangeLabel)}</span>
                        </div>
                        <div class="text-metric">
                            <span class="metric-label">Unique Values:</span>
                            <span class="metric-value">${uniqueCount.toLocaleString()}</span>
                            ${uniquePercentageDisplay ? `<small class="metric-subtle">(${uniquePercentageDisplay})</small>` : ''}
                        </div>
                        <div class="text-metric">
                            <span class="metric-label">Completeness:</span>
                            <span class="metric-value">${completenessDisplay}</span>
                        </div>
                    </div>
                    ${renderTextRecommendation(col, isNLP, isCategorical)}
                    ${renderColumnPatterns(columnPatterns)}
                    ${renderColumnQualityFlags(columnFlags)}
                </div>
            `;
        });
        
        html += '</div></div>';
        
        return html;
    }

    function renderColumnPatterns(patterns) {
        if (!Array.isArray(patterns) || patterns.length === 0) {
            return '';
        }

        const items = patterns
            .filter(pattern => pattern && pattern.type)
            .map(pattern => {
                const percentDisplay = pattern.percentage != null
                    ? ` (${formatPercentageDisplay(pattern.percentage, 1)})`
                    : '';
                const countDisplay = pattern.count != null
                    ? ` - ${pattern.count.toLocaleString()} matches`
                    : '';
                return `
                    <li>
                        <i class="${getPatternIcon(pattern.type)}"></i>
                        <strong>${escapeHtml(pattern.type)}</strong>${percentDisplay}${countDisplay}
                    </li>
                `;
            })
            .join('');

        if (!items) {
            return '';
        }

        return `
            <div class="text-column-insight">
                <div class="insight-title"><i class="fas fa-search"></i> Detected Patterns</div>
                <ul class="insight-list">${items}</ul>
            </div>
        `;
    }

    function renderColumnQualityFlags(flags) {
        if (!Array.isArray(flags) || flags.length === 0) {
            return '';
        }

        const badges = flags
            .map(flag => `<span class="quality-flag-badge">${escapeHtml(flag)}</span>`)
            .join('');

        return `
            <div class="text-column-insight quality-flags">
                <div class="insight-title"><i class="fas fa-flag"></i> Text Quality Flags</div>
                <div class="quality-flag-badges">${badges}</div>
            </div>
        `;
    }

    function renderDetectedPatternSummary(textSummary) {
        const patterns = Array.isArray(textSummary.detected_patterns) ? textSummary.detected_patterns : [];
        if (patterns.length === 0) {
            return '';
        }

        const items = patterns.map(pattern => {
            const type = escapeHtml(pattern.pattern_type || 'pattern');
            const count = numberOr(pattern.total_count, 0);
            const columns = formatColumnNameChips(pattern.columns || []);
            return `
                <div class="pattern-card">
                    <div class="pattern-header">
                        <i class="${getPatternIcon(pattern.pattern_type)}"></i>
                        <span class="pattern-type">${type}</span>
                        <span class="pattern-count">${count.toLocaleString()} matches</span>
                    </div>
                    ${columns ? `<div class="pattern-columns"><strong>Columns:</strong> ${columns}</div>` : ''}
                </div>
            `;
        }).join('');

        return `
            <div class="text-pattern-summary">
                <h5><i class="fas fa-search"></i> Detected Text Patterns</h5>
                <div class="pattern-grid">${items}</div>
            </div>
        `;
    }

    function renderTextQualityAlerts(textSummary) {
        const flags = Array.isArray(textSummary.text_quality_flags) ? textSummary.text_quality_flags : [];
        if (flags.length === 0) {
            return '';
        }

        const cards = flags.map(entry => {
            const flagLabel = escapeHtml(entry.flag || 'quality flag');
            const columns = formatColumnNameChips(entry.columns || []);
            return `
                <div class="quality-flag-card">
                    <div class="quality-flag-header">
                        <i class="fas fa-exclamation-circle"></i>
                        <span class="quality-flag-label">${flagLabel}</span>
                    </div>
                    ${columns ? `<div class="quality-flag-columns"><strong>Columns:</strong> ${columns}</div>` : ''}
                </div>
            `;
        }).join('');

        return `
            <div class="text-quality-alerts">
                <h5><i class="fas fa-flag"></i> Text Quality Alerts</h5>
                <div class="quality-flag-grid">${cards}</div>
            </div>
        `;
    }

    function renderPiiWarnings(textSummary) {
        const piiColumns = Array.isArray(textSummary.pii_columns) ? textSummary.pii_columns : [];
        if (piiColumns.length === 0) {
            return '';
        }

        return `
            <div class="pii-warning">
                <div class="pii-warning-header">
                    <i class="fas fa-user-shield"></i>
                    <span>Sensitive Data Detected</span>
                </div>
                <p>Columns likely contain personally identifiable information. Apply masking, hashing, or access controls before sharing.</p>
                <div class="pii-columns">${formatColumnNameChips(piiColumns)}</div>
            </div>
        `;
    }
    
    /**
     * Render text recommendation for a column
     */
    function renderTextRecommendation(col, isNLP, isCategorical) {
        let html = '';
        
        if (isNLP) {
            html += `
                <div class="text-recommendation nlp-recommendation">
                    <i class="fas fa-brain"></i> 
                    Suitable for NLP analysis, sentiment analysis, or topic modeling
                </div>
            `;
        }
        
        if (isCategorical) {
            html += `
                <div class="text-recommendation categorical-recommendation">
                    <i class="fas fa-tags"></i> 
                    Good candidate for label encoding or one-hot encoding
                </div>
            `;
        }
        
        // Additional recommendations based on characteristics
        if (numberOr(col.avg_text_length, 0) > 100) {
            html += `
                <div class="text-recommendation processing-recommendation">
                    <i class="fas fa-cut"></i> 
                    Consider text preprocessing: tokenization, stemming, or summarization
                </div>
            `;
        }
        
        const uniquePercentage = asNumber(col.unique_percentage);
        if (uniquePercentage !== null && uniquePercentage > 90) {
            html += `
                <div class="text-recommendation uniqueness-recommendation">
                    <i class="fas fa-fingerprint"></i> 
                    High uniqueness - potential identifier or free-form text
                </div>
            `;
        }
        
        return html;
    }
    
    /**
     * Render text processing recommendations
     */
    function renderTextProcessingRecommendations(textColumns) {
        if (textColumns.length === 0) {
            return '';
        }
        
        let html = `
            <div class="text-processing-recommendations">
                <h5><i class="fas fa-cogs"></i> Text Processing Recommendations</h5>
                <div class="recommendation-cards">
        `;
        
        // Preprocessing recommendations
        const longTextColumns = textColumns.filter(col => numberOr(col.avg_text_length, 0) > 50);
        if (longTextColumns.length > 0) {
            html += `
                <div class="recommendation-card">
                    <div class="recommendation-header">
                        <i class="fas fa-cut"></i>
                        <h6>Text Preprocessing</h6>
                    </div>
                    <div class="recommendation-content">
                        <p>Long text columns detected. Consider:</p>
                        <ul>
                            <li>Tokenization and normalization</li>
                            <li>Stop word removal</li>
                            <li>Stemming or lemmatization</li>
                        </ul>
                        <div class="affected-columns">
                            <strong>Columns:</strong> ${formatColumnList(longTextColumns)}
                        </div>
                    </div>
                </div>
            `;
        }
        
        // NLP analysis recommendations
        const nlpColumns = textColumns.filter(col => 
            col.text_category === 'free_text' || col.text_category === 'descriptive_text'
        );
        if (nlpColumns.length > 0) {
            html += `
                <div class="recommendation-card">
                    <div class="recommendation-header">
                        <i class="fas fa-brain"></i>
                        <h6>NLP Analysis</h6>
                    </div>
                    <div class="recommendation-content">
                        <p>Free text columns are suitable for:</p>
                        <ul>
                            <li>Sentiment analysis</li>
                            <li>Topic modeling (LDA, NMF)</li>
                            <li>Named entity recognition</li>
                            <li>Text classification</li>
                        </ul>
                        <div class="affected-columns">
                            <strong>Columns:</strong> ${formatColumnList(nlpColumns)}
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Categorical encoding recommendations
        const categoricalColumns = textColumns.filter(col => col.text_category === 'categorical');
        if (categoricalColumns.length > 0) {
            html += `
                <div class="recommendation-card">
                    <div class="recommendation-header">
                        <i class="fas fa-tags"></i>
                        <h6>Categorical Encoding</h6>
                    </div>
                    <div class="recommendation-content">
                        <p>Categorical text columns can be encoded using:</p>
                        <ul>
                            <li>One-hot encoding (for low cardinality)</li>
                            <li>Label encoding (for ordinal categories)</li>
                            <li>Target encoding (for high cardinality)</li>
                        </ul>
                        <div class="affected-columns">
                            <strong>Columns:</strong> ${formatColumnList(categoricalColumns)}
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Missing data handling for text
        const missingTextColumns = textColumns.filter(col => numberOr(col.null_percentage, 0) > 5);
        if (missingTextColumns.length > 0) {
            html += `
                <div class="recommendation-card">
                    <div class="recommendation-header">
                        <i class="fas fa-exclamation-triangle"></i>
                        <h6>Missing Text Data</h6>
                    </div>
                    <div class="recommendation-content">
                        <p>Text columns with missing values. Consider:</p>
                        <ul>
                            <li>Imputation with "Unknown" or "Missing"</li>
                            <li>Dropping rows with missing text (if critical)</li>
                            <li>Using indicator variables for missingness</li>
                        </ul>
                        <div class="affected-columns">
                            <strong>Columns:</strong> ${formatColumnList(
                                missingTextColumns,
                                (col) => `${col.name} (${formatPercentageDisplay(numberOr(col.null_percentage, 0))} missing)`
                            )}
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += '</div></div>';
        
        return html;
    }
    
    // Export functions
    global.DI.textAnalysis.renderTextAnalysis = renderTextAnalysis;
    
})(window);
