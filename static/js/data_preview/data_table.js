/**
 * data_table.js
 * =============
 * 
 * Data Table Module for Data Preview Page
 * 
 * Purpose:
 *  - Render interactive data tables
 *  - Handle table pagination and sorting
 *  - Provide data export functionality
 *  - Display sample data with statistics
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    global.DI.dataTable = global.DI.dataTable || {};
    
    /**
     * Render data table in the specified container
     */
    function renderDataTable(previewData, containerId) {
        console.log('=== renderDataTable called ===');
        console.log('previewData:', previewData);
        console.log('containerId:', containerId);
        
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('Container not found:', containerId);
            return;
        }
        
        if (!previewData || !previewData.columns || !previewData.sample_data) {
            console.error('Invalid preview data:', previewData);
            container.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    No preview data available.
                </div>
            `;
            return;
        }
        
        console.log('Preview data is valid, proceeding with render...');
        console.log('Sample data length:', previewData.sample_data.length);
        console.log('Columns:', previewData.columns);
        
        // Show loading indicator for very large datasets only
        const rowCount = previewData.sample_data.length;
        const columnCount = previewData.columns.length;
        
        if (rowCount > 1000 || columnCount > 50) {
            container.innerHTML = `
                <div class="data-preview-loading">
                    <div class="text-center py-4">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-2 text-muted">
                            Preparing ${rowCount.toLocaleString()} rows for display...
                        </p>
                    </div>
                </div>
            `;
            
            // Reduced timeout for better perceived performance
            setTimeout(() => {
                renderTableContent(previewData, container);
            }, 10);
        } else {
            // Render immediately for normal datasets
            renderTableContent(previewData, container);
        }
    }
    
    /**
     * Render the actual table content
     */
    function renderTableContent(previewData, container) {
        console.log('=== renderTableContent called ===');
        console.log('previewData:', previewData);
        
        // First render the basic table with data
        const basicTableHtml = renderBasicTableWithData(previewData);
        
        /** 
        let html = `
            <div class="data-preview-section">
                <div class="data-preview-header">
                    <h4><i class="fas fa-table"></i> Data Sample</h4>
                    <div class="data-preview-stats">
                        ${renderDataStats(previewData)}
                    </div>
                    <div class="data-preview-actions-below-stats">
                        ${renderColumnInfoButton()}
                    </div>
                </div>
                ${basicTableHtml}
            </div>
        `;
        */

        let html = `
            <div class="data-preview-section">
                <div class="data-preview-header d-flex justify-content-between align-items-center">
                    <h4><i class="fas fa-table"></i> Data Sample</h4>
                    ${renderColumnInfoButton()}
                </div>
                ${basicTableHtml}
            </div>
        `;

        console.log('Setting container HTML...');
        container.innerHTML = html;
        
        console.log('Container HTML set, initializing DataTable...');
        
        // Initialize DataTable immediately for better performance
        console.log('Starting DataTable initialization...');
        
        // Check if jQuery and DataTables are available
        if (typeof $ === 'undefined') {
            console.error('jQuery not loaded');
            return;
        }
        
        if (!$.fn.DataTable) {
            console.error('DataTables not loaded');
            return;
        }
        
        console.log('jQuery and DataTables are available, proceeding...');
        initializeDataTableFromHTML(previewData);
    }
    
    /**
     * Render basic table with data already populated
     */
    function renderBasicTableWithData(previewData) {
        console.log('=== renderBasicTableWithData ===');
        console.log('previewData.sample_data:', previewData.sample_data);
        console.log('previewData.columns:', previewData.columns);
        
        const tableId = 'dataPreviewTable';
        const sampleData = previewData.sample_data;
        const totalRows = previewData.shape ? previewData.shape[0] : sampleData.length;
        const isFirstLastMode = previewData.sample_mode === 'first_last';
        
        if (!sampleData || sampleData.length === 0) {
            console.error('No sample data to render!');
            return `<div class="alert alert-warning">No data available</div>`;
        }
        
        console.log('Rendering table with', sampleData.length, 'rows');
        
        let html = `
            <div class="table-container">
                <div class="table-responsive">
                    <table id="${tableId}" class="table table-striped table-bordered data-preview-table">
                        <thead class="table-dark">
                            <tr>
                                <th class="row-number-header">#</th>
        `;
        
        // Add column headers with type indicators  
        const columnHeaders = previewData.columns.map(col => {
            const colType = detectColumnType(previewData.sample_data, col);
            const typeIcon = getTypeIcon(colType);
            
            return `
                <th class="data-column-header" data-column="${col}" data-type="${colType}">
                    <div class="column-header-content">
                        <span class="column-name">${col}</span>
                        <span class="column-type" title="${colType}">
                            <i class="${typeIcon}"></i>
                        </span>
                    </div>
                </th>
            `;
        }).join('');
        
        html += columnHeaders;
        
        html += `
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        // Add data rows - optimize with array join for better performance
        const rowsHtml = sampleData.map((row, index) => {
            // Calculate actual row index for display
            let actualRowIndex = index;
            if (isFirstLastMode && previewData.first_rows && index >= previewData.first_rows) {
                const positionInLastSection = index - previewData.first_rows;
                actualRowIndex = totalRows - (previewData.last_rows || 50) + positionInLastSection;
            }
            
            const cellsHtml = previewData.columns.map(col => {
                const value = row[col];
                const cellClass = getCellClass(value);
                const displayValue = formatCellValue(value);
                return `<td class="${cellClass}" title="${value}">${displayValue}</td>`;
            }).join('');
            
            return `<tr><td class="row-number">${actualRowIndex + 1}</td>${cellsHtml}</tr>`;
        }).join('');
        
        html += rowsHtml;
        
        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        /** 
        // Add special notice for first/last mode
        if (isFirstLastMode && totalRows > sampleData.length) {
            const firstRows = previewData.first_rows || 50;
            const lastRows = previewData.last_rows || 50;
            html += `
                <div class="first-last-notice mt-2">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle"></i>
                        <strong>First & Last Row Preview:</strong> Showing first ${firstRows} and last ${lastRows} rows 
                        from ${totalRows.toLocaleString()} total rows. Use search and filters to explore the data.
                    </div>
                </div>
            `;
        }
        */
        console.log('Generated HTML table length:', html.length);
        console.log('HTML preview:', html.substring(0, 500) + '...');
        
        return html;
    }
    
    /**
     * Initialize DataTables from existing HTML table
     */
    function initializeDataTableFromHTML(previewData) {
        const tableId = 'dataPreviewTable';
        const table = document.getElementById(tableId);
        
        console.log('=== initializeDataTableFromHTML ===');
        console.log('Table element:', table);
        console.log('Table HTML preview:', table ? table.outerHTML.substring(0, 300) + '...' : 'NULL');
        
        if (!table) {
            console.error('Table element not found');
            return;
        }
        
        // Check table structure
        const thead = table.querySelector('thead');
        const tbody = table.querySelector('tbody');
        const rows = tbody ? tbody.querySelectorAll('tr') : [];
        
        console.log('Table structure check:');
        console.log('- thead found:', !!thead);
        console.log('- tbody found:', !!tbody);
        console.log('- rows in tbody:', rows.length);
        
        if (rows.length === 0) {
            console.error('No data rows found in table!');
            return;
        }
        
        // Destroy existing DataTable if present
        if ($.fn.DataTable.isDataTable(`#${tableId}`)) {
            console.log('Destroying existing DataTable');
            $(`#${tableId}`).DataTable().destroy();
        }
        
        try {
            const totalRows = previewData.shape ? previewData.shape[0] : previewData.sample_data.length;
            const isFirstLastMode = previewData.sample_mode === 'first_last';
            
            console.log('DataTable configuration:');
            console.log('- totalRows:', totalRows);
            console.log('- isFirstLastMode:', isFirstLastMode);
            
            console.log('Creating DataTable from HTML...');
            const dataTable = $(`#${tableId}`).DataTable({
                pageLength: 25,
                lengthMenu: [[10, 25, 50, 100], [10, 25, 50, 100]],
                searching: true,
                ordering: true,
                info: true,
                paging: true,
                processing: false, // Disable processing indicator for faster rendering
                deferRender: true, // Improve performance for large datasets
                stateSave: false, // Disable state saving for better performance
                autoWidth: false, // Disable auto width calculation
                columnDefs: [
                    {
                        targets: 0, // First column (row numbers)
                        orderable: false,
                        searchable: false,
                        className: 'row-number',
                        width: '60px'
                    }
                ],
                language: {
                    search: "Search data:",
                    lengthMenu: "Show _MENU_ rows",
                    info: `Showing _START_ to _END_ of _TOTAL_ preview rows${isFirstLastMode ? ` (from ${totalRows.toLocaleString()} total)` : ''}`,
                    infoEmpty: "No data available",
                    infoFiltered: "(filtered from _MAX_ total records)"
                },
                dom: '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6"f>>' +
                     '<"row"<"col-sm-12"tr>>' +
                     '<"row"<"col-sm-12 col-md-5"i><"col-sm-12 col-md-7"p>>'
            });
            
            console.log('DataTable created successfully from HTML:', dataTable);
            console.log('DataTable info:', dataTable.page.info());
            return dataTable;
            
        } catch (error) {
            console.error('Failed to initialize DataTable from HTML:', error);
            console.error('Error details:', error.stack);
            console.log('Falling back to basic table...');
        }
    }
    
    /**
     * Render data statistics
    function renderDataStats(previewData) {
        const sampleRows = previewData.sample_data.length;
        const totalRows = previewData.shape ? previewData.shape[0] : sampleRows;
        const totalCols = previewData.shape ? previewData.shape[1] : previewData.columns.length;
        const estimatedTotal = previewData.estimated_total_rows || null;
        const isFirstLastMode = previewData.sample_mode === 'first_last';
        
        let html = `
            <div class="data-stats-inline">
                <div class="stat-item">
                    <span class="stat-value">${sampleRows}</span>
                    <span class="stat-label">Preview Rows</span>
                </div>
        `;
        
        if (estimatedTotal && estimatedTotal > sampleRows) {
            html += `
                <div class="stat-item">
                    <span class="stat-value">${estimatedTotal.toLocaleString()}</span>
                    <span class="stat-label">Estimated Total</span>
                </div>
            `;
        } else if (totalRows > sampleRows) {
            html += `
                <div class="stat-item">
                    <span class="stat-value">${totalRows.toLocaleString()}</span>
                    <span class="stat-label">Total Rows</span>
                </div>
            `;
        }
        
        html += `
                <div class="stat-item">
                    <span class="stat-value">${totalCols}</span>
                    <span class="stat-label">Columns</span>
                </div>
        `;
        
        // Add mode indicator
        if (isFirstLastMode && totalRows > sampleRows) {
            html += `
                <div class="stat-item info">
                    <span class="stat-value">📋</span>
                    <span class="stat-label">First & Last Rows</span>
                </div>
            `;
        }
        
        html += `
            </div>
        `;
        
        return html;
    }

    /**
     * Fallback: Render basic table without DataTables
     */
    function renderBasicTable(previewData, tableId) {
        const table = document.getElementById(tableId);
        if (!table) return;
        
        const sampleData = previewData.sample_data;
        const totalRows = previewData.shape ? previewData.shape[0] : sampleData.length;
        const isFirstLastMode = previewData.sample_mode === 'first_last';
        
        let tbody = '';
        sampleData.forEach((row, index) => {
            let actualRowIndex;
            if (isFirstLastMode && previewData.first_rows && index >= previewData.first_rows) {
                const positionInLastSection = index - previewData.first_rows;
                actualRowIndex = totalRows - (previewData.last_rows || 50) + positionInLastSection;
            } else {
                actualRowIndex = index;
            }
            
            tbody += `<tr>`;
            tbody += `<td class="row-number">${actualRowIndex + 1}</td>`;
            
            previewData.columns.forEach(col => {
                const value = row[col];
                const cellClass = getCellClass(value);
                const displayValue = formatCellValue(value);
                tbody += `<td class="${cellClass}" title="${value}">${displayValue}</td>`;
            });
            
            tbody += `</tr>`;
        });
        
        table.querySelector('tbody').innerHTML = tbody;
        console.log('Basic table rendered as fallback');
    }
    
    /**
     * Initialize DataTables on the preview table
     */
    function initializeDataTable(previewData) {
        const tableId = 'dataPreviewTable';
        const table = document.getElementById(tableId);
        
        console.log('=== initializeDataTable ===');
        console.log('Table element:', table);
        console.log('Preview data:', previewData);
        
        if (!table) {
            console.error('Table element not found');
            return;
        }
        
        // Check if we have data
        if (!previewData || !previewData.sample_data || previewData.sample_data.length === 0) {
            console.error('No preview data available for DataTable');
            return;
        }
        
        // Destroy existing DataTable if present
        if ($.fn.DataTable.isDataTable(`#${tableId}`)) {
            console.log('Destroying existing DataTable');
            $(`#${tableId}`).DataTable().destroy();
        }
        
        try {
            // Prepare data for DataTables
            const sampleData = previewData.sample_data;
            const totalRows = previewData.shape ? previewData.shape[0] : sampleData.length;
            const isFirstLastMode = previewData.sample_mode === 'first_last';
            
            console.log('Sample data:', sampleData);
            console.log('Sample data length:', sampleData.length);
            console.log('First sample row:', sampleData[0]);
            
            // Simplified data transformation for DataTables
            const tableData = sampleData.map((row, index) => {
                // Calculate actual row index for display
                let actualRowIndex = index;
                if (isFirstLastMode && previewData.first_rows && index >= previewData.first_rows) {
                    const positionInLastSection = index - previewData.first_rows;
                    actualRowIndex = totalRows - (previewData.last_rows || 50) + positionInLastSection;
                }
                
                // Create array starting with row number
                const rowData = [actualRowIndex + 1];
                
                // Add data for each column
                previewData.columns.forEach(col => {
                    const value = row[col];
                    // Use simple value without HTML for now
                    rowData.push(value != null ? String(value) : '');
                });
                
                return rowData;
            });
            
            console.log('Transformed table data:', tableData);
            console.log('Table data length:', tableData.length);
            
            // Simple column definitions
            const columns = [
                {
                    title: '#',
                    orderable: false,
                    searchable: false,
                    className: 'row-number'
                }
            ];
            
            previewData.columns.forEach(col => {
                columns.push({
                    title: col,
                    className: 'data-cell'
                });
            });
            
            console.log('Column definitions:', columns);
            
            // Initialize DataTable with minimal options
            console.log('Creating DataTable...');
            const dataTable = $(`#${tableId}`).DataTable({
                data: tableData,
                columns: columns,
                pageLength: 10,
                lengthMenu: [[10, 25, 50, 100], [10, 25, 50, 100]],
                searching: true,
                ordering: true,
                info: true,
                language: {
                    search: "Search data:",
                    lengthMenu: "Show _MENU_ rows",
                    info: `Showing _START_ to _END_ of _TOTAL_ preview rows${isFirstLastMode ? ` (from ${totalRows.toLocaleString()} total)` : ''}`,
                    infoEmpty: "No data available",
                    infoFiltered: "(filtered from _MAX_ total records)"
                }
            });
            
            console.log('DataTable created successfully:', dataTable);
            return dataTable;
            
        } catch (error) {
            console.error('Failed to initialize DataTable:', error);
            console.error('Error details:', error.stack);
            // Fallback to basic table rendering
            renderBasicTable(previewData, tableId);
        }
    }
    
    /**
     * Render table actions
     */
    function renderTableActions(previewData) {
        // Expose actions via global functions but do not render export/refresh buttons
        // The UI now shows only a small Column Info button below the stats area.
        return '';
    }

    /**
     * Render a small Column Info button to show next to the title
     */
    function renderColumnInfoButton() {
        return `
            <button class="btn btn-sm column-info-btn theme-styled-btn" onclick="showColumnInfo()" aria-label="Column information">
                <i class="fas fa-info-circle me-1"></i>
                <span>Column Details</span>
            </button>
        `;
    }
    
    /**
     * Initialize DataTable with performance optimizations
     */
    function initializeDataTable() {
        // Wait a bit for DOM to be ready
        setTimeout(() => {
            if (window.$ && window.$.fn.DataTable) {
                try {
                    const table = $('#dataPreviewTable').DataTable({
                        pageLength: 10, // Start with smaller page size
                        lengthMenu: [[10, 25, 50, 100], [10, 25, 50, 100]], // Remove "All" option to prevent freezing
                        scrollX: true,
                        scrollY: '400px',
                        scrollCollapse: true,
                        order: [], // No initial sorting
                        deferRender: true, // Improve performance for large datasets
                        processing: true, // Show processing indicator
                        stateSave: true, // Remember user preferences
                        columnDefs: [
                            {
                                targets: 0, // Row number column
                                orderable: false,
                                searchable: false,
                                className: 'row-number-cell',
                                width: '50px'
                            },
                            {
                                targets: '_all', // All other columns
                                render: function(data, type, row, meta) {
                                    if (type === 'display' && data && data.length > 100) {
                                        // Truncate long text for display performance
                                        return data.substring(0, 100) + '...';
                                    }
                                    return data;
                                }
                            }
                        ],
                        language: {
                            search: "Search in data:",
                            lengthMenu: "Show _MENU_ rows per page",
                            info: "Showing _START_ to _END_ of _TOTAL_ rows",
                            infoFiltered: "(filtered from _MAX_ total rows)",
                            processing: "Loading data...",
                            paginate: {
                                first: "First",
                                last: "Last", 
                                next: "Next",
                                previous: "Previous"
                            },
                            emptyTable: "No data available"
                        },
                        dom: '<"row table-controls"<"col-sm-12 col-md-4"l><"col-sm-12 col-md-4 text-center"<"table-info">><"col-sm-12 col-md-4"f>>' +
                             '<"row"<"col-sm-12"tr>>' +
                             '<"row table-footer"<"col-sm-12 col-md-5"i><"col-sm-12 col-md-7"p>>',
                        responsive: {
                            details: {
                                type: 'column',
                                target: 'tr'
                            }
                        },
                        drawCallback: function(settings) {
                            // Add performance info after each draw
                            const info = this.api().page.info();
                            $('.table-info').html(`
                                <small class="text-muted">
                                    <i class="fas fa-table"></i> 
                                    ${info.recordsTotal} rows loaded
                                </small>
                            `);
                        }
                    });
                    
                    // Add column click handlers with debouncing
                    let clickTimeout;
                    $('#dataPreviewTable thead th').click(function() {
                        clearTimeout(clickTimeout);
                        clickTimeout = setTimeout(() => {
                            const columnName = $(this).data('column');
                            if (columnName) {
                                showColumnDetails(columnName);
                            }
                        }, 300); // Debounce clicks
                    });
                    
                    // Store table reference globally for external access
                    window.dataPreviewTable = table;
                    
                } catch (error) {
                    console.warn('DataTable initialization failed:', error);
                    // Fallback to basic table if DataTable fails
                    initializeBasicTable();
                }
            } else {
                // Fallback if DataTable is not available
                initializeBasicTable();
            }
        }, 100);
    }
    
    /**
     * Detect column type from sample data
     */
    function detectColumnType(sampleData, columnName) {
        if (!sampleData || sampleData.length === 0) return 'unknown';
        
        const values = sampleData.map(row => row[columnName]).filter(val => val !== null && val !== undefined && val !== '');
        
        if (values.length === 0) return 'empty';
        
        // Check if all values are numbers
        const numericValues = values.filter(val => !isNaN(val) && !isNaN(parseFloat(val)));
        if (numericValues.length === values.length) {
            // Check if all are integers
            const integerValues = numericValues.filter(val => Number.isInteger(parseFloat(val)));
            return integerValues.length === numericValues.length ? 'integer' : 'float';
        }
        
        // Check if values look like dates
        const dateValues = values.filter(val => !isNaN(Date.parse(val)));
        if (dateValues.length > values.length * 0.8) {
            return 'datetime';
        }
        
        // Check if boolean-like
        const booleanValues = values.filter(val => 
            val.toString().toLowerCase() === 'true' || 
            val.toString().toLowerCase() === 'false' ||
            val === '1' || val === '0'
        );
        if (booleanValues.length === values.length) {
            return 'boolean';
        }
        
        // Check uniqueness for categorical
        const uniqueValues = new Set(values);
        if (uniqueValues.size < values.length * 0.1 && uniqueValues.size < 20) {
            return 'categorical';
        }
        
        return 'text';
    }
    
    /**
     * Get icon for column type
     */
    function getTypeIcon(colType) {
        const iconMap = {
            'integer': 'fas fa-hashtag',
            'float': 'fas fa-percent',
            'text': 'fas fa-font',
            'categorical': 'fas fa-tags',
            'datetime': 'fas fa-calendar',
            'boolean': 'fas fa-check-square',
            'empty': 'fas fa-minus',
            'unknown': 'fas fa-question'
        };
        
        return iconMap[colType] || 'fas fa-question';
    }
    
    /**
     * Get CSS class for cell based on value
     */
    function getCellClass(value) {
        if (value === null || value === undefined || value === '') {
            return 'cell-null';
        }
        
        if (typeof value === 'number') {
            return 'cell-numeric';
        }
        
        if (typeof value === 'boolean') {
            return 'cell-boolean';
        }
        
        if (!isNaN(Date.parse(value))) {
            return 'cell-date';
        }
        
        return 'cell-text';
    }
    
    /**
     * Format cell value for display
     */
    function formatCellValue(value) {
        if (value === null || value === undefined) {
            return '<em class="null-value">null</em>';
        }
        
        if (value === '') {
            return '<em class="empty-value">empty</em>';
        }
        
        if (typeof value === 'number') {
            if (Number.isInteger(value)) {
                return value.toLocaleString();
            } else {
                return parseFloat(value.toFixed(4)).toString();
            }
        }
        
        if (typeof value === 'boolean') {
            return value ? '<span class="boolean-true">true</span>' : '<span class="boolean-false">false</span>';
        }
        
        // Truncate long text
        const stringValue = value.toString();
        if (stringValue.length > 100) {
            return `<span title="${stringValue}">${stringValue.substring(0, 97)}...</span>`;
        }
        
        return stringValue;
    }
    
    /**
     * Action functions
     */
    function exportTableData(format) {
        const sourceId = window.DI?.previewPage?.getCurrentSourceId();
        if (!sourceId) {
            showNotification('No data source available for export', 'error');
            return;
        }
        
        showNotification(`Exporting data as ${format.toUpperCase()}...`, 'info');
        
        // Use existing export functionality if available
        if (window.showExportModal) {
            window.showExportModal(sourceId, 'source', format);
        } else {
            showNotification(`${format.toUpperCase()} export functionality not available`, 'warning');
        }
    }
    
    function refreshTableData() {
        showNotification('Refreshing table data...', 'info');
        
        // Re-load the preview tab
        if (window.DI?.previewPage) {
            // Force reload by clearing cache
            if (window.loadDataForTab) {
                window.loadDataForTab('preview');
            }
        }
    }
    
    function showColumnInfo() {
        const sourceId = window.DI?.previewPage?.getCurrentSourceId();
        if (!sourceId) {
            showNotification('No data source available', 'error');
            return;
        }
        
        // Show a modal with column information
        const previewData = window.DI?.previewPage?.getCurrentPreviewData();
        if (previewData && previewData.columns) {
            showColumnInfoModal(previewData);
        } else {
            showNotification('Column information not available', 'warning');
        }
    }
    
    function showColumnDetails(columnName) {
        showNotification(`Column details for "${columnName}" - feature coming soon!`, 'info');
    }
    
    /**
     * Show column information modal
     */
    function showColumnInfoModal(previewData) {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content column-info-modal">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-info-circle"></i> Column Information
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="table-responsive">
                            <table class="table table-sm column-info-table">
                                <thead>
                                    <tr>
                                        <th>Column</th>
                                        <th>Type</th>
                                        <th>Sample Values</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${previewData.columns.map(col => {
                                        const colType = detectColumnType(previewData.sample_data, col);
                                        const sampleValues = previewData.sample_data
                                            .slice(0, 3)
                                            .map(row => row[col])
                                            .filter(val => val !== null && val !== undefined)
                                            .join(', ');
                                        
                                        return `
                                            <tr>
                                                <td><strong>${col}</strong></td>
                                                <td>
                                                    <i class="${getTypeIcon(colType)}"></i> ${colType}
                                                </td>
                                                <td class="text-muted">${sampleValues || 'No values'}</td>
                                            </tr>
                                        `;
                                    }).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Show modal using Bootstrap if available
        if (window.bootstrap && window.bootstrap.Modal) {
            const bsModal = new window.bootstrap.Modal(modal);
            bsModal.show();
            
            // Remove modal from DOM when hidden
            modal.addEventListener('hidden.bs.modal', () => {
                modal.remove();
            });
        } else {
            // Fallback: show as alert
            alert('Column information feature requires Bootstrap modal component');
            modal.remove();
        }
    }
    
    /**
     * Utility function to show notifications
     */
    function showNotification(message, type) {
        if (window.DI && window.DI.utilities && window.DI.utilities.notifications) {
            window.DI.utilities.notifications.showNotification(message, type);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }
    
    /**
     * Initialize basic table fallback (when DataTable is not available)
     */
    function initializeBasicTable() {
        console.log('Initializing basic table fallback');
        
        const table = document.getElementById('dataPreviewTable');
        if (!table) return;
        
        // Add basic styling and interactions
        table.classList.add('basic-table-initialized');
        
        // Add click handlers for column headers
        const headers = table.querySelectorAll('th[data-column]');
        headers.forEach(header => {
            header.style.cursor = 'pointer';
            header.addEventListener('click', () => {
                const columnName = header.getAttribute('data-column');
                if (columnName) {
                    showColumnDetails(columnName);
                }
            });
        });
        
        // Add simple pagination for large tables
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        
        if (rows.length > 50) {
            addBasicPagination(table, rows, 25);
        }
    }
    
    /**
     * Add basic pagination to prevent performance issues
     */
    function addBasicPagination(table, rows, pageSize) {
        const totalPages = Math.ceil(rows.length / pageSize);
        let currentPage = 1;
        
        // Create pagination controls
        const paginationDiv = document.createElement('div');
        paginationDiv.className = 'basic-pagination mt-3';
        paginationDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div class="pagination-info">
                    <small class="text-muted">
                        Page <span id="currentPageSpan">1</span> of ${totalPages} 
                        (${rows.length} total rows)
                    </small>
                </div>
                <div class="pagination-controls">
                    <button class="btn btn-sm btn-outline-secondary" id="prevPageBtn" disabled>
                        <i class="fas fa-chevron-left"></i> Previous
                    </button>
                    <button class="btn btn-sm btn-outline-secondary ms-2" id="nextPageBtn">
                        Next <i class="fas fa-chevron-right"></i>
                    </button>
                </div>
            </div>
        `;
        
        // Insert pagination after table
        table.parentNode.insertBefore(paginationDiv, table.nextSibling);
        
        // Pagination functions
        function showPage(page) {
            // Hide all rows
            rows.forEach(row => row.style.display = 'none');
            
            // Show current page rows
            const start = (page - 1) * pageSize;
            const end = start + pageSize;
            rows.slice(start, end).forEach(row => row.style.display = '');
            
            // Update controls
            document.getElementById('currentPageSpan').textContent = page;
            document.getElementById('prevPageBtn').disabled = page === 1;
            document.getElementById('nextPageBtn').disabled = page === totalPages;
            
            currentPage = page;
        }
        
        // Event listeners
        document.getElementById('prevPageBtn').addEventListener('click', () => {
            if (currentPage > 1) showPage(currentPage - 1);
        });
        
        document.getElementById('nextPageBtn').addEventListener('click', () => {
            if (currentPage < totalPages) showPage(currentPage + 1);
        });
        
        // Show first page
        showPage(1);
    }
    
    // Export functions
    global.DI.dataTable.renderDataTable = renderDataTable;
    global.DI.dataTable.showColumnInfo = showColumnInfo;
    global.DI.dataTable.showColumnDetails = showColumnDetails;
    
    // Export action functions for global access
    global.exportTableData = exportTableData;
    global.refreshTableData = refreshTableData;
    global.showColumnInfo = showColumnInfo;
    global.showColumnDetails = showColumnDetails;
    
})(window);
