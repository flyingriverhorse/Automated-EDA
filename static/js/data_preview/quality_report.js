/**
 * quality_report.js
 * =================
 * 
 * Quality Report Module for Data Preview Page
 * 
 * Purpose:
 *  - Render comprehensive data quality reports
 *  - Display quality metrics and statistics
 *  - Show data completeness and issues
 *  - Provide quality visualizations
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    global.DI.qualityReport = global.DI.qualityReport || {};
    
    /**
     * Render quality report in the specified container
     */
    function renderQualityReport(qualityReport, containerId) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('Container not found:', containerId);
            return;
        }
        
        if (!qualityReport) {
            console.error('No quality report data provided');
            container.innerHTML = '<div class="alert alert-danger">No quality report data available</div>';
            return;
        }

        try {
            const metadata = qualityReport.basic_metadata;
            const quality = qualityReport.quality_metrics;
            const issues = qualityReport.potential_issues || [];
            
            let html = `<div class="quality-report">`;
            
            // Basic Metadata Section
            html += renderBasicMetadata(metadata);
            
            // Data Quality Metrics Section
            html += renderQualityMetrics(quality, qualityReport.overall_quality_score);
            
            // Missing Values Tabbed Interface
            html += renderMissingValuesTabs(quality.column_details);
            
            // Enhanced Column Details
            html += renderEnhancedColumnDetails(quality.column_details);
            
            // Potential Issues Section
            //if (issues.length) {
            //    html += renderPotentialIssues(issues);
            //}
            
            html += '</div>';
            
            container.innerHTML = html;

        } catch (error) {
            console.error('Error rendering quality report:', error);
            container.innerHTML = `<div class="alert alert-danger">Error rendering quality report: ${error.message}</div>`;
        }
    }
    
    /**
     * Render basic metadata section
     */
    function renderBasicMetadata(metadata) {
        let html = `
            <div class="quality-section">
                <h4><i class="fas fa-info-circle"></i> Basic Metadata</h4>
                <div class="metadata-grid">
        `;
        
        html += `
            <div class="metadata-card interactive-sample-card">
                <div class="metadata-value">
              <input type="number" 
                  id="sampleSizeInput" 
                  value="${metadata.sample_rows}" 
                  min="100" 
                  max="1000000" 
                           step="100"
                           class="sample-size-input"
                           onchange="updateSampleSize(this.value)"
                           oninput="debounceUpdateSampleSize(this.value)">
                </div>
                <div class="metadata-label">Sample Rows (Interactive)</div>
                <div class="sample-size-status" id="sampleSizeStatus"></div>
                ${metadata.estimated_total_rows && metadata.estimated_total_rows > 0 ? 
                    `<div class="total-rows-info">Total: ${metadata.estimated_total_rows.toLocaleString()} rows</div>` : 
                    ''}
            </div>
        `;
        
        if (metadata.estimated_total_rows && metadata.estimated_total_rows !== metadata.sample_rows) {
            html += `
                <div class="metadata-card">
                    <div class="metadata-value">${metadata.estimated_total_rows.toLocaleString()}</div>
                    <div class="metadata-label">Estimated Total</div>
                </div>
            `;
        }
        
        html += `
            <div class="metadata-card">
                <div class="metadata-value">${metadata.total_columns}</div>
                <div class="metadata-label">Columns</div>
            </div>
        `;
        
        if (metadata.file_size_bytes) {
            html += `
                <div class="metadata-card">
                    <div class="metadata-value">${formatFileSize(metadata.file_size_bytes)}</div>
                    <div class="metadata-label">File Size</div>
                </div>
            `;
        }
        
        html += `
            <div class="metadata-card">
                <div class="metadata-value">${formatFileSize(metadata.memory_usage_bytes)}</div>
                <div class="metadata-label">Memory Usage</div>
            </div>
        `;
        
        html += '</div>';
        
        // Data Types Section
        html += renderDataTypes(metadata.data_types);
        
        html += '</div>';
        
        return html;
    }
    
    /**
     * Render data types section
     */
    function renderDataTypes(dataTypes) {
        let html = `
            <div class="data-types-section">
                <h5><i class="fas fa-tags"></i> Data Types</h5>
                <div class="data-types-grid">
        `;
        
        html += Object.entries(dataTypes).map(([dtype, columns]) => {
            // Clean dtype for CSS class (remove special characters, convert to lowercase)
            const dtypeClass = dtype.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
            return `<div class="dtype-card">
                <div class="dtype-name">
                    <span class="dtype-badge dtype-${dtypeClass}">${dtype}</span>
                </div>
                <div class="dtype-count">${columns.length} column${columns.length !== 1 ? 's' : ''}</div>
                <div class="dtype-columns">${columns.slice(0,5).join(', ')}${columns.length>5?' and '+(columns.length-5)+' more...':''}</div>
            </div>`;
        }).join('');
        
        html += '</div></div>';
        
        return html;
    }
    
    /**
     * Render quality metrics section
     */
    function renderQualityMetrics(quality, overallQualityScore) {
        let html = `
            <div class="quality-section">
                <h4><i class="fas fa-chart-line"></i> Data Quality Metrics</h4>
                <div class="quality-summary">
        `;
        
        // Overall Quality Score
        if (overallQualityScore !== undefined) {
            const scoreClass = overallQualityScore >= 80 ? 'excellent' : 
                             overallQualityScore >= 60 ? 'good' : 'needs-attention';
            html += `
                <div class="quality-stat ${scoreClass}">
                    <div class="quality-stat-value">${overallQualityScore.toFixed(0)}%</div>
                    <div class="quality-stat-label">Overall Quality Score</div>
                </div>
            `;
        }
        
        html += `
            <div class="quality-stat ${quality.overall_completeness > 90 ? 'excellent' : quality.overall_completeness > 70 ? 'good' : 'needs-attention'}">
                <div class="quality-stat-value">${quality.overall_completeness.toFixed(1)}%</div>
                <div class="quality-stat-label">Data Completeness</div>
            </div>
            <div class="quality-stat">
                <div class="quality-stat-value">${quality.columns_with_missing}</div>
                <div class="quality-stat-label">Columns with Missing Values</div>
            </div>
            <div class="quality-stat ${quality.high_cardinality_columns > 0 ? 'warning' : 'good'}">
                <div class="quality-stat-value">${quality.high_cardinality_columns}</div>
                <div class="quality-stat-label">High Cardinality Columns</div>
            </div>
        `;
        
        html += '</div></div>';
        
        return html;
    }
    
    /**
     * Render missing values tabbed interface
     */
    function renderMissingValuesTabs(columnDetails) {
        let html = `
            <div class="missing-tabs-container">
                <div class="missing-values-tabs">
                    <button class="missing-tab-btn active" data-tab="chart" onclick="switchMissingTab('chart')">
                        <i class="fas fa-chart-bar"></i> Missing Values by Column
                    </button>
                </div>
                
                <div class="missing-tab-content active" id="missingChartTab">
                    ${renderMissingValuesChartContent(columnDetails)}
                </div>
            </div>
        `;
        
        return html;
    }

    /**
     * Render missing values chart content (without wrapper)
     */
    function renderMissingValuesChartContent(columnDetails) {
        const allColumns = columnDetails || [];
        
        // Separate columns with and without missing values
        const columnsWithMissing = allColumns
            .filter(col => col.null_percentage > 0)
            .sort((a, b) => b.null_percentage - a.null_percentage);
            
        const columnsWithoutMissing = allColumns.filter(col => col.null_percentage === 0);
        
        let html = `<h5><i class="fas fa-chart-bar"></i> Missing Values by Column</h5>`;
        
        if (!allColumns.length) {
            html += `
                <div class="no-missing-values">
                    <i class="fas fa-question-circle"></i> 
                    No column data available!
                </div>
            `;
        } else if (!columnsWithMissing.length) {
            html += `
                <div class="no-missing-values">
                    <i class="fas fa-check-circle"></i> 
                    Excellent! All ${columnsWithoutMissing.length} columns are complete with no missing values!
                </div>
            `;
        } else {
            // Show summary of complete columns
            if (columnsWithoutMissing.length > 0) {
                html += `
                    <div class="missing-values-summary">
                        <i class="fas fa-info-circle"></i> 
                        ${columnsWithoutMissing.length} column${columnsWithoutMissing.length !== 1 ? 's have' : ' has'} no missing values. 
                        Showing ${columnsWithMissing.length} column${columnsWithMissing.length !== 1 ? 's' : ''} with missing data below:
                    </div>
                `;
            }
            
            html += `<div class="missing-values-bars-container">`;
            html += `<div class="missing-values-bars">`;
            
            columnsWithMissing.forEach(col => {
                const nullPercentage = col.null_percentage || 0;
                html += `
                    <div class="missing-bar-row">
                        <div class="missing-bar-label">${col.name}</div>
                        <div class="missing-bar-container">
                            <div class="missing-bar" style="width: ${nullPercentage}%"></div>
                        </div>
                        <div class="missing-bar-value">${nullPercentage.toFixed(1)}%</div>
                    </div>
                `;
            });
            
            html += '</div>';
            html += '</div>';
        }
        
        return html;
    }
    
    /**
     * Render enhanced column details
     */
    function renderEnhancedColumnDetails(columnDetails) {
        let html = `
            <div class="column-details-section">
                <h5><i class="fas fa-columns"></i> Column Details</h5>
                <div class="column-details-table-container">
                    <table class="column-details-table">
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Data Type</th>
                                <th>Category</th>
                                <th>Completeness</th>
                                <th>Uniqueness</th>
                                <th>Text Info</th>
                                <th>Memory</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        columnDetails.forEach(col => {
            const rowClass = col.null_percentage > 30 ? 'high-missing' : 
                           col.null_percentage > 10 ? 'medium-missing' : '';
            
            const categoryBadge = col.data_category ? 
                `<span class="category-badge category-${col.data_category}">${col.data_category}</span>` : 
                '<span class="category-badge">unknown</span>';
            
            let textInfo = '-';
            if (col.data_category === 'text') {
                const avgLen = col.avg_text_length || 0;
                const category = col.text_category || 'mixed';
                textInfo = `
                    <div class="text-info">
                        <span class="text-category-mini">${category}</span>
                        <small>${avgLen.toFixed(0)} avg chars</small>
                    </div>
                `;
            }
            
            html += `
                <tr class="${rowClass}">
                    <td class="column-name">${col.name}</td>
                    <td class="dtype-badge dtype-${col.dtype.split('(')[0]}">${col.dtype}</td>
                    <td class="category-cell">${categoryBadge}</td>
                    <td class="completeness-cell">
                        <span class="completeness-percentage ${col.null_percentage > 30 ? 'poor' : col.null_percentage > 10 ? 'fair' : 'good'}">
                            ${(100 - col.null_percentage).toFixed(1)}%
                        </span>
                        ${col.null_percentage > 0 ? `<small>(${col.null_count} missing)</small>` : ''}
                    </td>
                    <td class="uniqueness-cell">
                        <span class="uniqueness-percentage ${col.unique_percentage > 95 ? 'very-high' : col.unique_percentage > 50 ? 'high' : 'normal'}">
                            ${col.unique_percentage.toFixed(1)}%
                        </span>
                        <small>(${col.unique_count} unique)</small>
                    </td>
                    <td class="text-info-cell">${textInfo}</td>
                    <td class="memory-cell">${formatFileSize(col.memory_usage)}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        return html;
    }

    function escapeHtml(text) {
        if (!text) return '';
        return String(text).replace(/[&<>"]+/g, function(s) {
            return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[s]);
        });
    }

    /**
     * Render potential issues section (enhanced)
     * Uses native <details> so items are accessible and collapsible without extra JS.
     */
    /**
    function renderPotentialIssues(issues) {
        let html = `
            <div class="quality-section potential-issues-section">
                <h4><i class="fas fa-exclamation-triangle"></i> Potential Issues</h4>
                <div class="issues-list">
        `;

        issues.forEach(issue => {
            const iconClass = issue.type === 'warning' ? 'fa-exclamation-triangle' : 
                              issue.type === 'info' ? 'fa-info-circle' : 'fa-times-circle';
            const severity = issue.type || 'info';

            html += `
                <details class="issue-item issue-${severity}">
                    <summary class="issue-summary">
                        <div class="issue-icon"><i class="fas ${iconClass}"></i></div>
                        <div class="issue-main">
                            <div class="issue-message">${issue.message}</div>
                            ${issue.column ? `<div class="issue-column">Column: ${issue.column}</div>` : ''}
                        </div>
                        <div class="issue-meta">
                            <span class="issue-badge ${'issue-'+severity}">${(severity||'info').toUpperCase()}</span>
                        </div>
                    </summary>
                    <div class="issue-details">
                        ${issue.detail ? `<div class="issue-detail-text">${issue.detail}</div>` : ''}
                        ${issue.suggestion ? `<div class="issue-suggestion"><strong>Suggested fix:</strong> ${issue.suggestion}</div>` : ''}
                        ${issue.sample ? `<pre class="issue-sample">${String(issue.sample)}</pre>` : ''}
                    </div>
                </details>
            `;
        });

        html += '</div></div>';

        return html;
    }
     */
    /**
     * Helper function to format file sizes
     */
    function formatFileSize(bytes) {
        if (!bytes) return '0 B';
        
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Switch between missing values tabs
     */
    function switchMissingTab(tabName) {
        // Remove active class from all tab buttons
        const tabBtns = document.querySelectorAll('.missing-tab-btn');
        tabBtns.forEach(btn => btn.classList.remove('active'));
        
        // Remove active class from all tab contents
        const tabContents = document.querySelectorAll('.missing-tab-content');
        tabContents.forEach(content => content.classList.remove('active'));
        
        // Add active class to clicked tab button
        const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
        if (activeBtn) activeBtn.classList.add('active');
        
        // Show corresponding tab content
        if (tabName === 'chart') {
            const targetContent = document.getElementById('missingChartTab');
            if (targetContent) {
                targetContent.classList.add('active');
            }
        }
    }
    
    // Sample size update functionality
    let sampleSizeUpdateTimeout;
    
    /**
     * Debounced sample size update function
     */
    function debounceUpdateSampleSize(newSize) {
        clearTimeout(sampleSizeUpdateTimeout);
        sampleSizeUpdateTimeout = setTimeout(() => {
            updateSampleSize(newSize);
        }, 1000); // 1 second delay
    }
    
    /**
     * Update sample size and refresh all analyses
     */
    function updateSampleSize(newSize) {
        const sampleSize = parseInt(newSize);
        if (isNaN(sampleSize) || sampleSize < 100 || sampleSize > 1000000) {
            showSampleSizeStatus('Invalid sample size. Please enter a value between 100 and 1,000,000.', 'error');
            return;
        }
        
        showSampleSizeStatus('Updating analyses...', 'loading');
        
        // Update global sample size
        window.currentSampleSize = sampleSize;
        
        // Refresh all analyses with new sample size
        refreshAllAnalyses(sampleSize);
    }
    
    /**
     * Refresh all analyses with new sample size
     */
    function refreshAllAnalyses(sampleSize) {
        if (!window.DI || !window.DI.previewPage || !window.DI.previewPage.getCurrentSourceId) {
            console.error('Preview page not properly initialized');
            showSampleSizeStatus('Error: Page not properly initialized', 'error');
            return;
        }
        
        const sourceId = window.DI.previewPage.getCurrentSourceId();
        if (!sourceId) {
            console.error('No source ID available');
            showSampleSizeStatus('Error: No data source selected', 'error');
            return;
        }
        
        const apiUrl = `${window.DATA_PREVIEW_CONFIG.apiBasePath}/${sourceId}/quality-report?sample_size=${sampleSize}`;
        
        fetch(apiUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    // Get the actual sample size used (might be different from requested)
                    const actualSampleSize = data.actual_sample_size || sampleSize;
                    
                    // Update the input field to show actual sample size if different
                    const sampleInput = document.getElementById('sampleSizeInput');
                    if (sampleInput && actualSampleSize !== sampleSize) {
                        sampleInput.value = actualSampleSize;
                        window.currentSampleSize = actualSampleSize;
                    }
                    
                    // Update current quality report data
                    window.DI.previewPage.setCurrentQualityReport(data.quality_report);
                    
                    // Re-render quality report tab if it's currently active
                    const qualityTabContent = document.getElementById('qualityTabContent');
                    const qualityTab = document.querySelector('[data-tab="quality"]');
                    if (qualityTab && qualityTab.classList.contains('active')) {
                        renderQualityReport(data.quality_report, 'qualityTabContent');
                    }
                    
                    // Re-render text analysis tab if it's currently active
                    const textTabContent = document.getElementById('textTabContent');
                    const textTab = document.querySelector('[data-tab="text"]');
                    if (textTab && textTab.classList.contains('active')) {
                        if (window.DI.textAnalysis && window.DI.textAnalysis.renderTextAnalysis) {
                            window.DI.textAnalysis.renderTextAnalysis(data.quality_report, 'textTabContent');
                        }
                    }
                    
                    // Re-render recommendations tab if it's currently active
                    const recommendationsTabContent = document.getElementById('recommendationsTabContent');
                    const recommendationsTab = document.querySelector('[data-tab="recommendations"]');
                    if (recommendationsTab && recommendationsTab.classList.contains('active')) {
                        if (window.DI.recommendations && window.DI.recommendations.renderRecommendations) {
                            window.DI.recommendations.renderRecommendations(data.quality_report, 'recommendationsTabContent');
                        }
                    }
                    
                    // Show success message with actual sample size
                    const displaySize = actualSampleSize !== sampleSize ? 
                        `${actualSampleSize.toLocaleString()} rows (capped from ${sampleSize.toLocaleString()})` : 
                        `${actualSampleSize.toLocaleString()} rows`;
                    showSampleSizeStatus(`Updated with ${displaySize}`, 'success');
                } else {
                    throw new Error(data.error || 'Unknown error');
                }
            })
            .catch(error => {
                console.error('Error updating sample size:', error);
                showSampleSizeStatus('Error updating analyses', 'error');
            });
    }
    
    /**
     * Show sample size update status
     */
    function showSampleSizeStatus(message, type) {
        const statusElement = document.getElementById('sampleSizeStatus');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `sample-size-status ${type}`;
            
            if (type === 'success') {
                setTimeout(() => {
                    statusElement.textContent = '';
                    statusElement.className = 'sample-size-status';
                }, 3000);
            }
        }
    }
    
    // Export functions
    global.DI.qualityReport.renderQualityReport = renderQualityReport;
    global.DI.qualityReport.switchMissingTab = switchMissingTab;
    global.DI.qualityReport.updateSampleSize = updateSampleSize;
    global.DI.qualityReport.debounceUpdateSampleSize = debounceUpdateSampleSize;
    
    // Make functions globally accessible for inline event handlers
    global.updateSampleSize = updateSampleSize;
    global.debounceUpdateSampleSize = debounceUpdateSampleSize;
    
})(window);
