const MARKETING_ANALYSIS_CONFIG = {
    'campaign_metrics_analysis': {
        title: 'Campaign Metrics Analysis',
        description: 'Analyze marketing campaign performance metrics including impressions, clicks, conversions, CTR, CPC, and ROAS.',
        features: [
            '✓ Core KPI calculations (CTR, CPC, ROAS)',
            '✓ Performance distribution analysis', 
            '✓ Campaign and channel comparison',
            '✓ Data quality assessment',
            '✓ Actionable recommendations'
        ],
        columns: {
            'impressions': {
                label: 'Impressions',
                description: 'Total impressions/views',
                required: false,
                examples: ['impressions', 'views', 'reach', 'imp_count']
            },
            'clicks': {
                label: 'Clicks', 
                description: 'Total clicks received',
                required: false,
                examples: ['clicks', 'click_count', 'link_clicks']
            },
            'conversions': {
                label: 'Conversions',
                description: 'Total conversions/purchases',
                required: false, 
                examples: ['conversions', 'purchases', 'signups', 'goals']
            },
            'spend': {
                label: 'Spend',
                description: 'Amount spent on campaign', 
                required: false,
                examples: ['spend', 'cost', 'budget', 'investment']
            },
            'revenue': {
                label: 'Revenue',
                description: 'Revenue generated',
                required: false,
                examples: ['revenue', 'sales', 'value', 'income']
            },
            'campaign': {
                label: 'Campaign Name',
                description: 'Campaign identifier',
                required: false,
                examples: ['campaign', 'campaign_name', 'ad_name']
            },
            'channel': {
                label: 'Channel/Source',
                description: 'Marketing channel',
                required: false, 
                examples: ['channel', 'source', 'platform', 'medium']
            }
        }
    },
    'conversion_funnel_analysis': {
        title: 'Conversion Funnel Analysis',
        description: 'Analyze step-by-step user journey, identify drop-off rates and bottlenecks in your conversion funnel.',
        features: [
            '✓ Multi-step funnel analysis',
            '✓ Drop-off rate calculations', 
            '✓ Bottleneck identification',
            '✓ Channel-based segmentation',
            '✓ Optimization recommendations'
        ],
        columns: {
            'session_id': {
                label: 'Session ID',
                description: 'Unique session identifier',
                required: false,
                examples: ['session_id', 'user_id', 'visitor_id']
            },
            'landed': {
                label: 'Landing Page',
                description: 'Visited landing page (1/0)',
                required: false,
                examples: ['landed', 'visited', 'entry', 'step1']
            },
            'viewed_product': {
                label: 'Viewed Product', 
                description: 'Viewed product page (1/0)',
                required: false,
                examples: ['viewed_product', 'product_view', 'step2']
            },
            'added_to_cart': {
                label: 'Added to Cart',
                description: 'Added item to cart (1/0)',
                required: false,
                examples: ['added_to_cart', 'cart', 'step3']
            },
            'started_checkout': {
                label: 'Started Checkout',
                description: 'Started checkout process (1/0)', 
                required: false,
                examples: ['started_checkout', 'checkout', 'step4']
            },
            'completed_purchase': {
                label: 'Completed Purchase',
                description: 'Completed transaction (1/0)',
                required: false,
                examples: ['completed_purchase', 'purchase', 'conversion']
            },
            'channel': {
                label: 'Traffic Channel',
                description: 'Source of traffic',
                required: false,
                examples: ['channel', 'source', 'utm_source', 'referrer']
            }
        }
    },
    'engagement_analysis': {
        title: 'Engagement Analysis', 
        description: 'Analyze user interaction patterns, session quality, and engagement metrics.',
        features: [
            '✓ Session quality assessment',
            '✓ Interaction depth analysis',
            '✓ User loyalty segmentation', 
            '✓ Device and source comparison',
            '✓ Engagement optimization tips'
        ],
        columns: {
            'session_duration': {
                label: 'Session Duration',
                description: 'Time spent on site (seconds/minutes)',
                required: false,
                examples: ['session_duration', 'time_spent', 'duration']
            },
            'pages_viewed': {
                label: 'Pages Viewed',
                description: 'Number of pages visited',
                required: false,
                examples: ['pages_viewed', 'page_views', 'pageviews']
            },
            'bounce': {
                label: 'Bounce',
                description: 'Bounce indicator (1/0)',
                required: false,
                examples: ['bounce', 'bounced', 'single_page']
            },
            'interactions': {
                label: 'Interactions',
                description: 'Number of interactions',
                required: false,
                examples: ['interactions', 'clicks', 'events']
            },
            'device': {
                label: 'Device Type',
                description: 'Device category',
                required: false,
                examples: ['device', 'device_type', 'platform']
            },
            'channel': {
                label: 'Traffic Source',
                description: 'Source of traffic',
                required: false,
                examples: ['channel', 'source', 'medium']
            }
        }
    },
    'channel_performance_analysis': {
        title: 'Channel Performance Analysis',
        description: 'Compare marketing channel effectiveness, ROI, and performance metrics across different acquisition sources.',
        features: [
            '✓ Channel ROI comparison',
            '✓ Traffic quality analysis',
            '✓ Cost per acquisition by channel',
            '✓ Performance benchmarking',
            '✓ Channel optimization tips'
        ],
        columns: {
            'channel': {
                label: 'Channel',
                description: 'Marketing channel/source',
                required: false,
                examples: ['channel', 'source', 'utm_source', 'medium']
            },
            'sessions': {
                label: 'Sessions',
                description: 'Number of sessions',
                required: false,
                examples: ['sessions', 'visits', 'traffic']
            },
            'conversions': {
                label: 'Conversions',
                description: 'Number of conversions',
                required: false,
                examples: ['conversions', 'goals', 'purchases']
            },
            'revenue': {
                label: 'Revenue',
                description: 'Revenue generated',
                required: false,
                examples: ['revenue', 'sales', 'value']
            },
            'spend': {
                label: 'Ad Spend',
                description: 'Amount spent on channel',
                required: false,
                examples: ['spend', 'cost', 'ad_spend', 'budget']
            }
        }
    },
    'audience_segmentation_analysis': {
        title: 'Audience Segmentation Analysis',
        description: 'Segment users by demographics, behavior patterns, and engagement levels for targeted marketing.',
        features: [
            '✓ Demographic segmentation',
            '✓ Behavioral clustering',
            '✓ Engagement-based groups',
            '✓ Segment profiling',
            '✓ Targeting recommendations'
        ],
        columns: {
            'age': {
                label: 'Age',
                description: 'User age',
                required: false,
                examples: ['age', 'user_age', 'age_group']
            },
            'gender': {
                label: 'Gender',
                description: 'User gender',
                required: false,
                examples: ['gender', 'sex', 'demographic_gender']
            },
            'location': {
                label: 'Location',
                description: 'Geographic location',
                required: false,
                examples: ['location', 'country', 'city', 'region']
            },
            'purchase_frequency': {
                label: 'Purchase Frequency',
                description: 'How often user purchases',
                required: false,
                examples: ['purchase_frequency', 'order_count', 'frequency']
            },
            'total_spent': {
                label: 'Total Spent',
                description: 'Total amount spent',
                required: false,
                examples: ['total_spent', 'lifetime_value', 'total_revenue']
            }
        }
    },
    'roi_analysis': {
        title: 'ROI Analysis',
        description: 'Calculate return on investment, profitability metrics, and identify the most cost-effective marketing activities.',
        features: [
            '✓ ROI calculations',
            '✓ ROAS analysis',
            '✓ Profit margin assessment',
            '✓ Cost-benefit analysis',
            '✓ Investment recommendations'
        ],
        columns: {
            'revenue': {
                label: 'Revenue',
                description: 'Revenue generated',
                required: false,
                examples: ['revenue', 'sales', 'income', 'earnings']
            },
            'cost': {
                label: 'Cost/Spend',
                description: 'Amount invested',
                required: false,
                examples: ['cost', 'spend', 'investment', 'budget']
            },
            'impressions': {
                label: 'Impressions',
                description: 'Number of impressions',
                required: false,
                examples: ['impressions', 'views', 'reach']
            },
            'clicks': {
                label: 'Clicks',
                description: 'Number of clicks',
                required: false,
                examples: ['clicks', 'click_count', 'link_clicks']
            },
            'conversions': {
                label: 'Conversions',
                description: 'Number of conversions',
                required: false,
                examples: ['conversions', 'sales_count', 'goals']
            }
        }
    },
    'attribution_analysis': {
        title: 'Attribution Analysis',
        description: 'Understand customer journey touchpoints and assign conversion credit across marketing channels.',
        features: [
            '✓ Multi-touch attribution',
            '✓ Channel contribution analysis',
            '✓ Customer journey mapping',
            '✓ Touchpoint effectiveness',
            '✓ Attribution model comparison'
        ],
        columns: {
            'customer_id': {
                label: 'Customer ID',
                description: 'Unique customer identifier',
                required: false,
                examples: ['customer_id', 'user_id', 'client_id']
            },
            'touchpoint': {
                label: 'Touchpoint',
                description: 'Marketing touchpoint/channel',
                required: false,
                examples: ['touchpoint', 'channel', 'source', 'medium']
            },
            'timestamp': {
                label: 'Timestamp',
                description: 'When interaction occurred',
                required: false,
                examples: ['timestamp', 'date', 'interaction_time']
            },
            'conversion': {
                label: 'Conversion',
                description: 'Conversion indicator (1/0)',
                required: false,
                examples: ['conversion', 'converted', 'purchase']
            },
            'revenue': {
                label: 'Revenue',
                description: 'Revenue from conversion',
                required: false,
                examples: ['revenue', 'value', 'purchase_amount']
            }
        }
    },
    'cohort_analysis': {
        title: 'Cohort Analysis',
        description: 'Track user behavior and retention over time by grouping users into cohorts based on shared characteristics.',
        features: [
            '✓ Retention rate tracking',
            '✓ User lifecycle analysis',
            '✓ Cohort performance comparison',
            '✓ Churn prediction',
            '✓ Lifetime value trends'
        ],
        columns: {
            'user_id': {
                label: 'User ID',
                description: 'Unique user identifier',
                required: false,
                examples: ['user_id', 'customer_id', 'account_id']
            },
            'signup_date': {
                label: 'Signup Date',
                description: 'When user first signed up',
                required: false,
                examples: ['signup_date', 'registration_date', 'first_seen']
            },
            'activity_date': {
                label: 'Activity Date',
                description: 'Date of user activity',
                required: false,
                examples: ['activity_date', 'last_seen', 'event_date']
            },
            'revenue': {
                label: 'Revenue',
                description: 'Revenue generated by user',
                required: false,
                examples: ['revenue', 'purchase_amount', 'value']
            },
            'active': {
                label: 'Active Status',
                description: 'Whether user is active (1/0)',
                required: false,
                examples: ['active', 'is_active', 'retained']
            }
        }
    }
};

if (typeof window !== 'undefined') {
    window.MARKETING_ANALYSIS_CONFIG = MARKETING_ANALYSIS_CONFIG;
}

let currentAnalysisType = '';
let currentMarketingCellId = '';  // Store cell ID for modal completion
let availableColumns = [];
let selectedColumnMapping = {};
let marketingModalConfirmed = false;

// Show marketing analysis modal
function showMarketingAnalysisModal(analysisType, dataColumns = []) {
    console.log('showMarketingAnalysisModal called with:', analysisType, dataColumns);
    
    currentAnalysisType = analysisType;
    selectedColumnMapping = {};
    
    // Try to get real columns from column insights data first
    if (columnInsightsData && columnInsightsData.column_insights) {
        availableColumns = columnInsightsData.column_insights.map(col => col.name);
        console.log('Using real column names from columnInsightsData:', availableColumns);
    } else if (dataColumns && dataColumns.length > 0) {
        availableColumns = dataColumns;
        console.log('Using provided dataColumns:', availableColumns);
    } else {
        // Fallback - get from session storage or use mock data as last resort
        const storedColumns = sessionStorage.getItem('datasetColumns');
        if (storedColumns) {
            try {
                availableColumns = JSON.parse(storedColumns);
                console.log('Using stored column names:', availableColumns);
            } catch (e) {
                console.warn('Could not parse stored columns, using mock data');
                availableColumns = [
                    'campaign_name', 'date', 'impressions', 'clicks', 'conversions', 
                    'spend', 'revenue', 'ctr', 'cpc', 'roas', 'channel', 'age_group',
                    'gender', 'location', 'device_type', 'session_duration', 'bounce_rate'
                ];
            }
        } else {
            console.warn('No real columns available, using mock columns for demo');
            availableColumns = [
                'campaign_name', 'date', 'impressions', 'clicks', 'conversions', 
                'spend', 'revenue', 'ctr', 'cpc', 'roas', 'channel', 'age_group',
                'gender', 'location', 'device_type', 'session_duration', 'bounce_rate'
            ];
        }
    }
    
    console.log('Final available columns for modal:', availableColumns);
    
    // Store columns in session storage for other components to use
    if (availableColumns && availableColumns.length > 0) {
        sessionStorage.setItem('datasetColumns', JSON.stringify(availableColumns));
        console.log('Stored columns in sessionStorage for other components');
    }
    
    const config = MARKETING_ANALYSIS_CONFIG[analysisType];
    if (!config) {
        console.error('Unknown marketing analysis type:', analysisType);
        return;
    }
    
    // Update modal content
    document.getElementById('analysisTitle').textContent = config.title;
    document.getElementById('analysisDescription').textContent = config.description;
    
    // Update features list
    const featuresList = document.getElementById('featuresList');
    featuresList.innerHTML = config.features.map(feature => `<li>${feature}</li>`).join('');
    
    // Generate column mapping interface
    generateColumnMappingInterface(config.columns);
    
    // Add example data format
    generateExampleDataFormat(analysisType);
    
    // Show the modal
    attachMarketingModalLifecycleHandlers();
    const modal = new bootstrap.Modal(document.getElementById('marketingAnalysisModal'));
    modal.show();
}

// Generate column mapping interface with data type filtering
function generateColumnMappingInterface(columnConfig) {
    console.log('Generating column mapping interface with config:', columnConfig);
    console.log('Available columns:', availableColumns);
    console.log('Column insights data:', columnInsightsData);
    
    const container = document.getElementById('columnMappingContainer');
    container.innerHTML = '';
    
    // Get column type information if available
    const columnTypes = {};
    if (columnInsightsData && columnInsightsData.column_insights) {
        columnInsightsData.column_insights.forEach(col => {
            columnTypes[col.name] = {
                data_type: col.data_type,
                data_category: col.data_category,
                is_numeric: col.data_category === 'numeric'
            };
        });
        console.log('Column types extracted:', columnTypes);
    }
    
    Object.entries(columnConfig).forEach(([key, config]) => {
        const columnDiv = document.createElement('div');
        columnDiv.className = 'col-md-6 mb-3';
        
        const isRequired = config.required ? '<span class="text-danger">*</span>' : '';
        const exampleText = config.examples.length > 0 ? `Examples: ${config.examples.join(', ')}` : '';
        
        // Filter columns based on expected data type for this metric
        const expectedNumeric = config.label.toLowerCase().includes('count') || 
                              config.label.toLowerCase().includes('rate') ||
                              config.label.toLowerCase().includes('amount') ||
                              config.label.toLowerCase().includes('spend') ||
                              config.label.toLowerCase().includes('revenue') ||
                              config.label.toLowerCase().includes('impressions') ||
                              config.label.toLowerCase().includes('clicks');
        
        // Generate options for dropdown with type filtering
        let optionsHtml = '<option value="">-- Skip this metric --</option>';
        
        // Create groups for better organization
        let numericCols = [];
        let textCols = [];
        let otherCols = [];
        
        availableColumns.forEach(col => {
            const colType = columnTypes[col];
            if (colType) {
                if (colType.is_numeric || colType.data_category === 'numeric') {
                    numericCols.push(col);
                } else if (colType.data_category === 'text' || colType.data_type === 'object') {
                    textCols.push(col);
                } else {
                    otherCols.push(col);
                }
            } else {
                otherCols.push(col);
            }
        });
        
        // Add numeric columns first (preferred for most metrics)
        if (numericCols.length > 0) {
            optionsHtml += '<optgroup label="📊 Numeric Columns">';
            numericCols.forEach(col => {
                const typeInfo = columnTypes[col] ? ` (${columnTypes[col].data_type})` : '';
                optionsHtml += `<option value="${col}">${col}${typeInfo}</option>`;
            });
            optionsHtml += '</optgroup>';
        }
        
        // Add text columns (useful for categories, names, etc.)
        if (textCols.length > 0) {
            optionsHtml += '<optgroup label="📝 Text Columns">';
            textCols.forEach(col => {
                const typeInfo = columnTypes[col] ? ` (${columnTypes[col].data_type})` : '';
                optionsHtml += `<option value="${col}">${col}${typeInfo}</option>`;
            });
            optionsHtml += '</optgroup>';
        }
        
        // Add other columns
        if (otherCols.length > 0) {
            optionsHtml += '<optgroup label="🔧 Other Columns">';
            otherCols.forEach(col => {
                const typeInfo = columnTypes[col] ? ` (${columnTypes[col].data_type})` : '';
                optionsHtml += `<option value="${col}">${col}${typeInfo}</option>`;
            });
            optionsHtml += '</optgroup>';
        }
        
        // Provide guidance based on expected type
        let typeGuidance = '';
        if (expectedNumeric) {
            typeGuidance = '<small class="text-info"><i class="bi bi-info-circle"></i> This metric typically uses numeric columns</small>';
        } else {
            typeGuidance = '<small class="text-muted"><i class="bi bi-tag"></i> This metric can use text or categorical columns</small>';
        }
        
        columnDiv.innerHTML = `
            <div class="form-group">
                <label class="form-label">
                    <strong>${config.label}</strong> ${isRequired}
                    <small class="text-muted d-block">${config.description}</small>
                </label>
                <select class="form-select form-select-sm" id="column_${key}" onchange="updateColumnMapping('${key}', this.value)">
                    ${optionsHtml}
                </select>
                <small class="text-muted d-block">${exampleText}</small>
                ${typeGuidance}
            </div>
        `;
        
        container.appendChild(columnDiv);
    });
    
    console.log('Column mapping interface generated with', Object.keys(columnConfig).length, 'metrics');
}

// Generate example data format based on analysis type
function generateExampleDataFormat(analysisType) {
    console.log('Generating example data format for:', analysisType);
    
    const exampleDataCard = document.getElementById('exampleDataCard');
    if (!exampleDataCard) {
        console.warn('Example data card not found');
        return;
    }
    
    const config = MARKETING_ANALYSIS_CONFIG[analysisType];
    if (!config) return;
    
    // Generate sample data based on analysis type
    let exampleData = [];
    let recommendedColumns = [];
    
    switch(analysisType) {
        case 'campaign_metrics_analysis':
            exampleData = [
                { campaign_name: 'Summer_Sale_2024', impressions: '25,000', clicks: '1,200', conversions: '45', spend: '$500', revenue: '$2,250' },
                { campaign_name: 'Holiday_Promo', impressions: '18,500', clicks: '890', conversions: '32', spend: '$380', revenue: '$1,600' },
                { campaign_name: 'Spring_Launch', impressions: '22,000', clicks: '1,100', conversions: '55', spend: '$450', revenue: '$2,750' }
            ];
            recommendedColumns = Object.keys(config.columns);
            break;
            
        case 'conversion_funnel_analysis':
            exampleData = [
                { user_id: 'U001', landed: '1', viewed_product: '1', added_to_cart: '1', started_checkout: '1', completed_purchase: '1' },
                { user_id: 'U002', landed: '1', viewed_product: '1', added_to_cart: '0', started_checkout: '0', completed_purchase: '0' },
                { user_id: 'U003', landed: '1', viewed_product: '1', added_to_cart: '1', started_checkout: '0', completed_purchase: '0' }
            ];
            recommendedColumns = ['user_id', 'landed', 'viewed_product', 'added_to_cart', 'started_checkout', 'completed_purchase'];
            break;
            
        case 'roi_analysis':
            exampleData = [
                { campaign: 'Social_Media', revenue: '$5,000', spend: '$1,200', impressions: '50,000', conversions: '85' },
                { campaign: 'Google_Ads', revenue: '$8,500', spend: '$2,000', impressions: '75,000', conversions: '120' },
                { campaign: 'Email_Marketing', revenue: '$3,200', spend: '$400', impressions: '25,000', conversions: '45' }
            ];
            recommendedColumns = ['campaign', 'revenue', 'spend', 'impressions', 'conversions'];
            break;
            
        default:
            // Generic example for other analyses
            exampleData = [
                { category: 'Category_A', metric1: '100', metric2: '50', value: '$1,000' },
                { category: 'Category_B', metric1: '150', metric2: '75', value: '$1,500' },
                { category: 'Category_C', metric1: '80', metric2: '40', value: '$800' }
            ];
            recommendedColumns = Object.keys(config.columns).slice(0, 4);
    }
    
    // Generate HTML for the example table
    if (exampleData.length > 0) {
        const tableHeaders = Object.keys(exampleData[0]);
        const tableHTML = `
            <div class="row">
                <div class="col-md-8">
                    <h6 class="text-primary">Sample ${config.title} Data:</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped">
                            <thead class="table-primary">
                                <tr>
                                    ${tableHeaders.map(header => `<th>${header}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${exampleData.map(row => `
                                    <tr>
                                        ${tableHeaders.map(header => `<td>${row[header]}</td>`).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="col-md-4">
                    <h6 class="text-success">Recommended Columns:</h6>
                    <div class="mb-2">
                        ${recommendedColumns.map(col => `<span class="badge bg-info me-1 mb-1">${col}</span>`).join('')}
                    </div>
                    <div class="mt-3">
                        <small class="text-muted">
                            <strong>Tip:</strong> Your column names don't need to match exactly. 
                            Use the mapping above to connect your data columns to the analysis metrics.
                        </small>
                    </div>
                </div>
            </div>
        `;
        
        exampleDataCard.innerHTML = tableHTML;
    }
}

// Update column mapping when user selects a column
function updateColumnMapping(metric, columnName) {
    console.log(`Updating column mapping: ${metric} -> ${columnName}`);
    
    if (columnName) {
        selectedColumnMapping[metric] = columnName;
    } else {
        delete selectedColumnMapping[metric];
    }
    
    console.log('Current column mapping:', selectedColumnMapping);
    updateConfigurationSummary();
}

// Auto-detect columns based on naming patterns
function autoDetectColumns() {
    console.log('Auto-detecting columns...');
    console.log('Available columns:', availableColumns);
    console.log('Current analysis type:', currentAnalysisType);
    
    const config = MARKETING_ANALYSIS_CONFIG[currentAnalysisType];
    if (!config) {
        console.error('No config found for', currentAnalysisType);
        showNotification('Configuration error for analysis type', 'error');
        return;
    }
    
    console.log('Column config:', config.columns);
    
    // Clear previous selections
    selectedColumnMapping = {};
    
    // Reset all select elements first
    Object.keys(config.columns).forEach(metric => {
        const selectElement = document.getElementById(`column_${metric}`);
        if (selectElement) {
            selectElement.value = '';
        }
    });
    
    let detectedCount = 0;
    
    // Auto-detect based on column names with improved matching
    Object.entries(config.columns).forEach(([metric, metricConfig]) => {
        console.log(`\nTrying to match metric: ${metric}`);
        console.log(`Examples: ${metricConfig.examples}`);
        
        for (const availableCol of availableColumns) {
            const colLower = availableCol.toLowerCase().replace(/[_\s-]/g, '');
            
            // Check if any example patterns match
            for (const example of metricConfig.examples) {
                const exampleLower = example.toLowerCase().replace(/[_\s-]/g, '');
                
                // Multiple matching strategies
                const exactMatch = colLower === exampleLower;
                const containsMatch = colLower.includes(exampleLower) || exampleLower.includes(colLower);
                const partialMatch = colLower.includes(exampleLower.substring(0, 4)) || exampleLower.includes(colLower.substring(0, 4));
                
                if (exactMatch || containsMatch || (partialMatch && exampleLower.length > 3)) {
                    selectedColumnMapping[metric] = availableCol;
                    
                    // Update the select element
                    const selectElement = document.getElementById(`column_${metric}`);
                    if (selectElement) {
                        selectElement.value = availableCol;
                        console.log(`✓ Auto-detected ${metric} -> ${availableCol} (matched with ${example})`);
                        detectedCount++;
                    }
                    
                    break; // Move to next metric after first match
                }
            }
            
            if (selectedColumnMapping[metric]) break; // Break outer loop if found
        }
        
        if (!selectedColumnMapping[metric]) {
            console.log(`✗ No match found for metric: ${metric}`);
        }
    });
    
    updateConfigurationSummary();
    
    // Show feedback
    console.log(`Auto-detection completed. Found ${detectedCount} mappings:`, selectedColumnMapping);
    
    if (detectedCount > 0) {
        showNotification(`✅ Auto-detected ${detectedCount} column mappings!`, 'success');
    } else {
        showNotification('⚠️ Could not auto-detect columns. Please select manually.', 'warning');
    }
}

// Update configuration summary
function updateConfigurationSummary() {
    const summaryDiv = document.getElementById('configurationSummary');
    const summaryList = document.getElementById('mappingSummaryList');
    
    if (Object.keys(selectedColumnMapping).length === 0) {
        summaryDiv.style.display = 'none';
        return;
    }
    
    summaryDiv.style.display = 'block';
    
    const mappingItems = Object.entries(selectedColumnMapping)
        .map(([metric, column]) => `<span class="badge bg-primary me-1">${metric}: ${column}</span>`)
        .join(' ');
    
    summaryList.innerHTML = `<p><strong>Selected mappings:</strong><br>${mappingItems}</p>`;
}

// Generate marketing analysis with selected columns
async function generateMarketingAnalysis() {
    if (!currentAnalysisType) {
        showNotification('❌ No analysis type selected', 'error');
        return;
    }

    const modal = bootstrap.Modal.getInstance(document.getElementById('marketingAnalysisModal'));
    marketingModalConfirmed = true;
    if (modal) modal.hide();

    const mappingCount = Object.keys(selectedColumnMapping).length;
    const marketingConfig = isMarketingAnalysisType(currentAnalysisType) && typeof window !== 'undefined'
        ? window.MARKETING_ANALYSIS_CONFIG?.[currentAnalysisType]
        : null;
    const analysisLabel = marketingConfig?.title || getAnalysisTypeName(currentAnalysisType);

    if (mappingCount === 0) {
        showNotification(`⚠️ Generating ${analysisLabel} with auto-detection (no columns mapped)`, 'warning');
    } else {
        const mappedLabel = mappingCount === 1 ? 'mapped column' : 'mapped columns';
        showNotification(`✅ Generating ${analysisLabel} with ${mappingCount} ${mappedLabel}`, 'success');
    }

    try {
        if (!currentMarketingCellId) {
            const fallbackColumns = await gatherAvailableColumnNames();
            const sanitizedFallbackColumns = Array.from(new Set((fallbackColumns || []).filter(Boolean)));
            const cellId = await addSingleAnalysisCell(currentAnalysisType, {
                skipMarketingModal: true,
                prefetchedColumns: sanitizedFallbackColumns
            });
            currentMarketingCellId = cellId || '';
        }

        if (!currentMarketingCellId) {
            showNotification('Unable to start marketing analysis: no analysis cell available', 'error');
            return;
        }

        await generateAndRunAnalysis(currentMarketingCellId, currentAnalysisType, selectedColumnMapping);
    } catch (error) {
        console.error('Marketing analysis generation failed:', error);
        showNotification(`Marketing analysis failed to start: ${error.message || error}`, 'error');
    } finally {
        currentMarketingCellId = '';
        marketingModalConfirmed = false;
    }
}

// Debug function to help troubleshoot column mapping issues
function debugColumnMapping() {
    console.log('=== DEBUG COLUMN MAPPING ===');
    console.log('Current analysis type:', currentAnalysisType);
    console.log('Available columns:', availableColumns);
    console.log('Column insights data:', columnInsightsData);
    console.log('Selected column mapping:', selectedColumnMapping);
    
    const config = MARKETING_ANALYSIS_CONFIG[currentAnalysisType];
    if (config) {
        console.log('Analysis config:', config);
        console.log('Required columns config:', config.columns);
        
        // Test auto-detect logic for each metric
        console.log('\n=== TESTING AUTO-DETECT LOGIC ===');
        Object.entries(config.columns).forEach(([metric, metricConfig]) => {
            console.log(`\nMetric: ${metric}`);
            console.log(`Examples: ${metricConfig.examples}`);
            
            let bestMatch = null;
            for (const availableCol of availableColumns) {
                const colLower = availableCol.toLowerCase().replace(/[_\s-]/g, '');
                
                for (const example of metricConfig.examples) {
                    const exampleLower = example.toLowerCase().replace(/[_\s-]/g, '');
                    
                    const exactMatch = colLower === exampleLower;
                    const containsMatch = colLower.includes(exampleLower) || exampleLower.includes(colLower);
                    const partialMatch = colLower.includes(exampleLower.substring(0, 4)) || exampleLower.includes(colLower.substring(0, 4));
                    
                    if (exactMatch || containsMatch || (partialMatch && exampleLower.length > 3)) {
                        bestMatch = { column: availableCol, example: example, type: exactMatch ? 'exact' : containsMatch ? 'contains' : 'partial' };
                        console.log(`   ✓ Found match: ${availableCol} (${bestMatch.type} match with ${example})`);
                        break;
                    }
                }
                
                if (bestMatch) break;
            }
            
            if (!bestMatch) {
                console.log(`   ✗ No match found for ${metric}`);
            }
        });
    }
    
    // Show alert with key info
    const summary = `
Debug Info:
- Analysis: ${currentAnalysisType}
- Available columns: ${availableColumns.length}
- Column insights: ${columnInsightsData ? 'Available' : 'Not available'}
- Current mappings: ${Object.keys(selectedColumnMapping).length}

Check console for detailed output.
    `;
    
    alert(summary.trim());
    console.log('=== END DEBUG ===');
}
