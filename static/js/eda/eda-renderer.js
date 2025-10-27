/**
 * EDA Renderer - Dynamic Chart Generation and Visualization Engine
 * Handles visualization rendering, template charts, and interactive graphics
 */

// Prevent redeclaration errors
if (typeof window.EDARenderer === 'undefined') {
    class EDARenderer {
        constructor() {
            this.chartInstances = new Map();
            this.renderingQueue = [];
            this.templates = {};
            this.colorPalettes = {
                default: ['#4facfe', '#00f2fe', '#43e97b', '#38f9d7', '#667eea', '#764ba2'],
                healthcare: ['#ff7b7b', '#667eea', '#4facfe', '#00f2fe'],
                finance: ['#56ab2f', '#a8e6cf', '#667eea', '#764ba2'],
                tech: ['#4facfe', '#00f2fe', '#667eea', '#764ba2']
            };
            this.defaultOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                }
            }
        };
    }

    /**
     * Initialize the EDA renderer
     */
    initialize() {
        console.log('EDARenderer: Initializing...');
        
        // Load chart libraries
        this.loadChartLibraries();
        
        // Set up visualization templates
        this.initializeTemplates();
        
        // Set up event listeners
        this.setupEventListeners();
        
        console.log('EDARenderer: Initialized successfully');
    }

    /**
     * Load required chart libraries
     */
    loadChartLibraries() {
        // Chart.js should already be loaded via CDN
        if (typeof Chart !== 'undefined') {
            // Configure Chart.js defaults
            Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
            Chart.defaults.font.size = 12;
        }
        
        // Plotly should already be loaded via CDN
        if (typeof Plotly !== 'undefined') {
            console.log('EDARenderer: Plotly.js loaded');
        }
    }

    /**
     * Initialize visualization templates
     */
    initializeTemplates() {
        this.templates = {
            // Distribution plots
            histogram: {
                type: 'histogram',
                library: 'chartjs',
                config: {
                    type: 'bar',
                    options: {
                        ...this.defaultOptions,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: { display: true, text: 'Frequency' }
                            },
                            x: {
                                title: { display: true, text: 'Values' }
                            }
                        }
                    }
                }
            },
            
            // Correlation plots
            heatmap: {
                type: 'heatmap',
                library: 'plotly',
                config: {
                    type: 'heatmap',
                    colorscale: 'RdBu',
                    showscale: true
                }
            },
            
            // Scatter plots
            scatter: {
                type: 'scatter',
                library: 'chartjs',
                config: {
                    type: 'scatter',
                    options: {
                        ...this.defaultOptions,
                        scales: {
                            x: { type: 'linear', position: 'bottom' },
                            y: { type: 'linear' }
                        }
                    }
                }
            },
            
            // Box plots
            boxplot: {
                type: 'boxplot',
                library: 'plotly',
                config: {
                    type: 'box',
                    boxpoints: 'outliers'
                }
            },
            
            // Time series
            timeseries: {
                type: 'timeseries',
                library: 'chartjs',
                config: {
                    type: 'line',
                    options: {
                        ...this.defaultOptions,
                        scales: {
                            x: {
                                type: 'time',
                                time: {
                                    displayFormats: {
                                        quarter: 'MMM YYYY'
                                    }
                                }
                            }
                        }
                    }
                }
            },
            
            // Pie charts
            pie: {
                type: 'pie',
                library: 'chartjs',
                config: {
                    type: 'doughnut',
                    options: {
                        ...this.defaultOptions,
                        cutout: '30%'
                    }
                }
            }
        };
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Listen for window resize to update responsive charts
        window.addEventListener('resize', () => {
            this.resizeAllCharts();
        });
        
        // Listen for tab changes to refresh visible charts
        document.addEventListener('tabActivated', (event) => {
            this.refreshTabCharts(event.detail.tabId);
        });
    }

    /**
     * Render a chart with given configuration
     */
    async renderChart(containerId, chartConfig, data) {
        try {
            console.log(`EDARenderer: Rendering chart in ${containerId}`);
            
            const container = document.getElementById(containerId);
            if (!container) {
                throw new Error(`Container ${containerId} not found`);
            }
            
            // Destroy existing chart if present
            this.destroyChart(containerId);
            
            // Determine chart type and library
            const template = this.templates[chartConfig.type] || chartConfig;
            const library = template.library || 'chartjs';
            
            let chartInstance;
            
            switch (library) {
                case 'chartjs':
                    chartInstance = await this.renderChartJS(container, template, data);
                    break;
                    
                case 'plotly':
                    chartInstance = await this.renderPlotly(container, template, data);
                    break;
                    
                default:
                    throw new Error(`Unsupported chart library: ${library}`);
            }
            
            // Store chart instance
            this.chartInstances.set(containerId, {
                instance: chartInstance,
                library: library,
                config: template,
                data: data
            });
            
            return chartInstance;
            
        } catch (error) {
            console.error('EDARenderer: Error rendering chart:', error);
            this.showChartError(containerId, error.message);
            throw error;
        }
    }

    /**
     * Render chart using Chart.js
     */
    async renderChartJS(container, template, data) {
        if (typeof Chart === 'undefined') {
            throw new Error('Chart.js not loaded');
        }
        
        // Create canvas element
        const canvas = document.createElement('canvas');
        container.innerHTML = '';
        container.appendChild(canvas);
        
        // Prepare chart configuration
        const config = {
            ...template.config,
            data: this.formatDataForChartJS(data, template.type)
        };
        
        // Create chart
        const chart = new Chart(canvas, config);
        
        return chart;
    }

    /**
     * Render chart using Plotly
     */
    async renderPlotly(container, template, data) {
        if (typeof Plotly === 'undefined') {
            throw new Error('Plotly.js not loaded');
        }
        
        // Prepare plotly data and layout
        const plotlyData = this.formatDataForPlotly(data, template.type);
        const layout = {
            responsive: true,
            autosize: true,
            margin: { t: 40, r: 40, b: 40, l: 40 },
            ...template.layout
        };
        
        // Create plot
        await Plotly.newPlot(container, plotlyData, layout, { displayModeBar: false });
        
        return { container: container, type: 'plotly' };
    }

    /**
     * Format data for Chart.js
     */
    formatDataForChartJS(data, chartType) {
        const colors = this.getColorPalette();
        
        switch (chartType) {
            case 'histogram':
                return {
                    labels: data.bins || data.labels,
                    datasets: [{
                        label: data.label || 'Frequency',
                        data: data.values || data.data,
                        backgroundColor: colors[0] + '80',
                        borderColor: colors[0],
                        borderWidth: 1
                    }]
                };
                
            case 'scatter':
                return {
                    datasets: [{
                        label: data.label || 'Data',
                        data: data.points || data.data,
                        backgroundColor: colors[0],
                        borderColor: colors[0]
                    }]
                };
                
            case 'pie':
                return {
                    labels: data.labels,
                    datasets: [{
                        data: data.values || data.data,
                        backgroundColor: colors,
                        borderWidth: 2,
                        borderColor: '#fff'
                    }]
                };
                
            case 'timeseries':
                return {
                    labels: data.dates || data.labels,
                    datasets: [{
                        label: data.label || 'Time Series',
                        data: data.values || data.data,
                        borderColor: colors[0],
                        backgroundColor: colors[0] + '20',
                        fill: true,
                        tension: 0.4
                    }]
                };
                
            default:
                return {
                    labels: data.labels,
                    datasets: [{
                        label: data.label || 'Data',
                        data: data.values || data.data,
                        backgroundColor: colors[0] + '80',
                        borderColor: colors[0],
                        borderWidth: 1
                    }]
                };
        }
    }

    /**
     * Format data for Plotly
     */
    formatDataForPlotly(data, chartType) {
        switch (chartType) {
            case 'heatmap':
                return [{
                    z: data.matrix || data.data,
                    x: data.xLabels || data.columns,
                    y: data.yLabels || data.index,
                    type: 'heatmap',
                    colorscale: 'RdBu',
                    showscale: true
                }];
                
            case 'boxplot':
                if (Array.isArray(data.data) && Array.isArray(data.data[0])) {
                    // Multiple box plots
                    return data.data.map((values, index) => ({
                        y: values,
                        type: 'box',
                        name: data.labels ? data.labels[index] : `Box ${index + 1}`,
                        boxpoints: 'outliers'
                    }));
                } else {
                    // Single box plot
                    return [{
                        y: data.values || data.data,
                        type: 'box',
                        name: data.label || 'Values',
                        boxpoints: 'outliers'
                    }];
                }
                
            case 'scatter3d':
                return [{
                    x: data.x,
                    y: data.y,
                    z: data.z,
                    mode: 'markers',
                    type: 'scatter3d',
                    marker: {
                        size: 5,
                        color: data.color || this.getColorPalette()[0]
                    }
                }];
                
            default:
                return [{
                    x: data.x || data.labels,
                    y: data.y || data.values || data.data,
                    type: chartType,
                    name: data.label || 'Data'
                }];
        }
    }

    /**
     * Get color palette based on current theme/domain
     */
    getColorPalette(domain = 'default') {
        return this.colorPalettes[domain] || this.colorPalettes.default;
    }

    /**
     * Render multiple charts for analysis results
     */
    async renderAnalysisCharts(results, containerPrefix = 'chart-') {
        const promises = [];
        
        results.visualizations.forEach((viz, index) => {
            const containerId = `${containerPrefix}${index}`;
            promises.push(this.renderChart(containerId, viz.config, viz.data));
        });
        
        try {
            await Promise.all(promises);
            console.log('EDARenderer: All analysis charts rendered successfully');
        } catch (error) {
            console.error('EDARenderer: Error rendering analysis charts:', error);
        }
    }

    /**
     * Create a quick visualization based on data type
     */
    async renderQuickViz(containerId, data, vizType = 'auto') {
        let chartType = vizType;
        
        if (vizType === 'auto') {
            chartType = this.detectChartType(data);
        }
        
        const config = { type: chartType };
        return await this.renderChart(containerId, config, data);
    }

    /**
     * Detect appropriate chart type based on data
     */
    detectChartType(data) {
        // Simple heuristics for chart type detection
        if (data.matrix) return 'heatmap';
        if (data.bins) return 'histogram';
        if (data.points) return 'scatter';
        if (data.dates) return 'timeseries';
        if (data.labels && data.values && data.labels.length < 10) return 'pie';
        
        return 'histogram'; // Default fallback
    }

    /**
     * Update chart data
     */
    updateChart(containerId, newData) {
        const chartInfo = this.chartInstances.get(containerId);
        if (!chartInfo) {
            console.warn(`EDARenderer: Chart ${containerId} not found for update`);
            return;
        }
        
        if (chartInfo.library === 'chartjs') {
            const formattedData = this.formatDataForChartJS(newData, chartInfo.config.type);
            chartInfo.instance.data = formattedData;
            chartInfo.instance.update();
        } else if (chartInfo.library === 'plotly') {
            const plotlyData = this.formatDataForPlotly(newData, chartInfo.config.type);
            Plotly.redraw(chartInfo.instance.container, plotlyData);
        }
        
        // Update stored data
        chartInfo.data = newData;
    }

    /**
     * Destroy a chart instance
     */
    destroyChart(containerId) {
        const chartInfo = this.chartInstances.get(containerId);
        if (chartInfo) {
            if (chartInfo.library === 'chartjs' && chartInfo.instance.destroy) {
                chartInfo.instance.destroy();
            } else if (chartInfo.library === 'plotly') {
                Plotly.purge(chartInfo.instance.container);
            }
            
            this.chartInstances.delete(containerId);
        }
    }

    /**
     * Resize all charts
     */
    resizeAllCharts() {
        this.chartInstances.forEach((chartInfo, containerId) => {
            if (chartInfo.library === 'chartjs') {
                chartInfo.instance.resize();
            } else if (chartInfo.library === 'plotly') {
                Plotly.Plots.resize(chartInfo.instance.container);
            }
        });
    }

    /**
     * Refresh charts in a specific tab
     */
    refreshTabCharts(tabId) {
        const tabContainer = document.getElementById(tabId);
        if (!tabContainer) return;
        
        // Find all chart containers in the tab
        const chartContainers = tabContainer.querySelectorAll('[id^="chart-"]');
        
        chartContainers.forEach(container => {
            const chartInfo = this.chartInstances.get(container.id);
            if (chartInfo) {
                setTimeout(() => {
                    if (chartInfo.library === 'chartjs') {
                        chartInfo.instance.resize();
                    } else if (chartInfo.library === 'plotly') {
                        Plotly.Plots.resize(container);
                    }
                }, 100);
            }
        });
    }

    /**
     * Show chart error
     */
    showChartError(containerId, message) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="d-flex justify-content-center align-items-center chart-error-state">
                    <div class="text-center text-muted">
                        <i class="bi bi-exclamation-triangle h1"></i>
                        <p class="mb-2">Chart Error</p>
                        <small>${message}</small>
                        <br>
                        <button class="btn btn-sm btn-outline-secondary mt-2" onclick="EDARenderer.retryChart('${containerId}')">
                            <i class="bi bi-arrow-clockwise"></i> Retry
                        </button>
                    </div>
                </div>
            `;
        }
    }

    /**
     * Retry chart rendering
     */
    retryChart(containerId) {
        const chartInfo = this.chartInstances.get(containerId);
        if (chartInfo) {
            this.renderChart(containerId, chartInfo.config, chartInfo.data);
        }
    }

    /**
     * Export chart as image
     */
    exportChart(containerId, format = 'png') {
        const chartInfo = this.chartInstances.get(containerId);
        if (!chartInfo) {
            console.warn(`EDARenderer: Chart ${containerId} not found for export`);
            return null;
        }
        
        if (chartInfo.library === 'chartjs') {
            const canvas = chartInfo.instance.canvas;
            return canvas.toDataURL(`image/${format}`);
        } else if (chartInfo.library === 'plotly') {
            return Plotly.toImage(chartInfo.instance.container, {
                format: format,
                width: 800,
                height: 600
            });
        }
        
        return null;
    }

    /**
     * Set chart theme/palette
     */
    setTheme(domain) {
        if (this.colorPalettes[domain]) {
            // Update all existing charts with new colors
            this.chartInstances.forEach((chartInfo, containerId) => {
                if (chartInfo.library === 'chartjs') {
                    // Update chart colors
                    const newData = this.formatDataForChartJS(chartInfo.data, chartInfo.config.type);
                    chartInfo.instance.data = newData;
                    chartInfo.instance.update();
                }
            });
        }
    }

    /**
     * Get all active chart instances
     */
    getAllCharts() {
        return Array.from(this.chartInstances.entries()).map(([containerId, chartInfo]) => ({
            containerId,
            library: chartInfo.library,
            type: chartInfo.config.type,
            hasData: !!chartInfo.data
        }));
    }

    /**
     * Clear all charts
     */
    clearAllCharts() {
        this.chartInstances.forEach((chartInfo, containerId) => {
            this.destroyChart(containerId);
        });
        
        this.chartInstances.clear();
    }

    /**
     * Generate chart from analysis result
     */
    async renderFromAnalysis(containerId, analysisResult) {
        const { chart_type, data, title, description } = analysisResult;
        
        // Create enhanced config with analysis metadata
        const config = {
            type: chart_type,
            title: title,
            description: description
        };
        
        return await this.renderChart(containerId, config, data);
    }
}

// Create global instance
window.edaRenderer = new EDARenderer();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EDARenderer;
}

}