/**
 * Tab Manager for Advanced EDA Platform
 * Handles tab switching, state management, and coordination between tabs
 */

// Prevent redeclaration errors
if (typeof window.TabManager === 'undefined') {
    class TabManager {
        constructor() {
            this.activeTab = 'notebook-content';
            this.tabStates = {
                'notebook-content': { initialized: false, data: null },
                'custom-analysis-content': { initialized: false, data: null },
                'quality-content': { initialized: false, data: null },
                'text-content': { initialized: false, data: null },
                'recommendations-content': { initialized: false, data: null },
                'data-preview-content': { initialized: false, data: null }
            };
            this.sharedData = {};
            this.previewTabMap = {
                'quality-content': 'quality',
                'text-content': 'text',
                'recommendations-content': 'recommendations',
                'data-preview-content': 'preview'
            };
            this.previewInitialized = false;
        }

    /**
     * Initialize the tab manager
     */
    initialize() {
        console.log('TabManager: Initializing...');
        
        // Set up tab event listeners
        this.setupTabListeners();
        
        // Initialize Bootstrap tabs
        this.initializeBootstrapTabs();
        
        // Set initial active tab
        this.setActiveTab('notebook-content');
        
        console.log('TabManager: Initialized successfully');
    }

    /**
     * Set up event listeners for tab changes
     */
    setupTabListeners() {
        const tabButtons = document.querySelectorAll('[data-bs-toggle="tab"]');
        
        tabButtons.forEach(button => {
            button.addEventListener('show.bs.tab', (event) => {
                this.handleTabShow(event);
            });
            
            button.addEventListener('shown.bs.tab', (event) => {
                this.handleTabShown(event);
            });
            
            button.addEventListener('hide.bs.tab', (event) => {
                this.handleTabHide(event);
            });
        });
    }

    /**
     * Initialize Bootstrap tab functionality
     */
    initializeBootstrapTabs() {
        if (typeof bootstrap !== 'undefined') {
            // Enable Bootstrap tabs
            const triggerTabList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tab"]'));
            triggerTabList.forEach(function (triggerEl) {
                const tabTrigger = new bootstrap.Tab(triggerEl);
            });
        }
    }

    /**
     * Handle tab show event (before tab becomes visible)
     */
    handleTabShow(event) {
        const targetTabId = event.target.getAttribute('data-bs-target').replace('#', '');
        console.log(`TabManager: Showing tab ${targetTabId}`);
        
        // Show loading state if needed
        this.showTabLoading(targetTabId);
        
        // Prepare tab data if needed
        this.prepareTabData(targetTabId);
    }

    /**
     * Handle tab shown event (after tab becomes visible)
     */
    handleTabShown(event) {
        const targetTabId = event.target.getAttribute('data-bs-target').replace('#', '');
        const previousTabId = event.relatedTarget?.getAttribute('data-bs-target')?.replace('#', '');
        
        console.log(`TabManager: Tab ${targetTabId} shown (previous: ${previousTabId})`);
        
        // Update active tab
        this.activeTab = targetTabId;
        
        // Initialize tab if not already initialized
        if (!this.tabStates[targetTabId].initialized) {
            this.initializeTab(targetTabId);
        }
        
        // Hide loading state
        this.hideTabLoading(targetTabId);
        
        // Trigger tab-specific events
        this.triggerTabEvents(targetTabId);
        
        // Update URL hash
        this.updateUrlHash(targetTabId);
    }

    /**
     * Handle tab hide event
     */
    handleTabHide(event) {
        const sourceTabId = event.target.getAttribute('data-bs-target').replace('#', '');
        console.log(`TabManager: Hiding tab ${sourceTabId}`);
        
        // Save tab state if needed
        this.saveTabState(sourceTabId);
    }

    /**
     * Set active tab programmatically
     */
    setActiveTab(tabId) {
        const tabButton = document.querySelector(`[data-bs-target="#${tabId}"]`);
        if (tabButton && typeof bootstrap !== 'undefined') {
            const tab = new bootstrap.Tab(tabButton);
            tab.show();
        }
    }

    /**
     * Show loading state for a tab
     */
    showTabLoading(tabId) {
        if (this.previewTabMap?.[tabId]) {
            return;
        }
        const tabContent = document.getElementById(tabId);
        if (tabContent && !this.tabStates[tabId].initialized) {
            // Add loading overlay if tab has complex initialization
            const loadingOverlay = document.createElement('div');
            loadingOverlay.className = 'tab-loading-overlay';
            loadingOverlay.innerHTML = `
                <div class="d-flex justify-content-center align-items-center loading-overlay-centered">
                    <div class="text-center">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-3 text-muted">Initializing tab...</p>
                    </div>
                </div>
            `;
            
            // Only add if not already present
            if (!tabContent.querySelector('.tab-loading-overlay')) {
                tabContent.appendChild(loadingOverlay);
            }
        }
    }

    /**
     * Hide loading state for a tab
     */
    hideTabLoading(tabId) {
        const tabContent = document.getElementById(tabId);
        if (tabContent) {
            const loadingOverlay = tabContent.querySelector('.tab-loading-overlay');
            if (loadingOverlay) {
                loadingOverlay.remove();
            }
        }
    }

    /**
     * Prepare data needed for a tab
     */
    prepareTabData(tabId) {
        // Check if we have the required data for this tab
        if (!this.sharedData.dataset && currentDataset?.source_id) {
            // Fetch dataset information if needed
            this.fetchDatasetInfo();
        }
    }

    /**
     * Initialize a specific tab
     */
    initializeTab(tabId) {
        console.log(`TabManager: Initializing tab ${tabId}`);
        
        try {
            switch (tabId) {
                case 'notebook-content':
                    if (typeof loadColumnInsights === 'function' && !window.columnInsightsData && !window.columnInsightsLoading) {
                        try {
                            loadColumnInsights();
                        } catch (innerError) {
                            console.warn('TabManager: Column insights load failed during init:', innerError);
                        }
                    }
                    break;

                case 'custom-analysis-content':
                    break;

                case 'quality-content':
                case 'text-content':
                case 'recommendations-content':
                case 'data-preview-content':
                    this.initializePreviewTab(tabId);
                    break;
            }
            
            // Mark as initialized
            this.tabStates[tabId].initialized = true;
            
        } catch (error) {
            console.error(`TabManager: Error initializing tab ${tabId}:`, error);
            this.showTabError(tabId, error.message);
        }
    }

    initializePreviewTab(tabId) {
        const previewKey = this.previewTabMap[tabId];
        if (!previewKey) {
            return;
        }

        if (!this.ensurePreviewCoreReady()) {
            console.warn('TabManager: Preview modules not ready at tab initialization, deferring load.');
            setTimeout(() => {
                if (this.ensurePreviewCoreReady() && window.DI?.previewPage?.loadDataForTab) {
                    try {
                        window.DI.previewPage.loadDataForTab(previewKey);
                    } catch (error) {
                        console.error(`TabManager: Deferred load failed for preview tab ${tabId}:`, error);
                    }
                }
            }, 300);
            return;
        }

        try {
            if (window.DI?.previewPage?.loadDataForTab) {
                window.DI.previewPage.loadDataForTab(previewKey);
            }
        } catch (error) {
            console.error(`TabManager: Failed to load preview tab ${tabId}:`, error);
            throw error;
        }
    }

    ensurePreviewCoreReady() {
        if (window.DI?.previewPage?.isInitialized?.()) {
            this.previewInitialized = true;
        }

        if (this.previewInitialized && window.DI?.previewPage?.loadDataForTab) {
            return true;
        }

        if (window.DI?.previewPage?.loadDataForTab) {
            this.previewInitialized = true;
            return true;
        }

        try {
            if (window.DI?.previewPage?.initialize) {
                window.DI.previewPage.initialize();
            } else if (typeof window.initializeDataPreviewPage === 'function') {
                window.initializeDataPreviewPage();
            }
        } catch (error) {
            console.error('TabManager: Error initializing data preview modules:', error);
            return false;
        }

        if (window.DI?.previewPage?.loadDataForTab) {
            this.previewInitialized = true;
            return true;
        }

        console.warn('TabManager: Data preview modules not ready yet.');
        return false;
    }

    /**
     * Show error state for a tab
     */
    showTabError(tabId, message) {
        const tabContent = document.getElementById(tabId);
        if (tabContent) {
            const errorHtml = `
                <div class="alert alert-danger m-4">
                    <h6><i class="bi bi-exclamation-triangle"></i> Tab Initialization Error</h6>
                    <p class="mb-0">${message}</p>
                    <button class="btn btn-sm btn-outline-danger mt-2" onclick="TabManager.retryInitialization('${tabId}')">
                        <i class="bi bi-arrow-clockwise"></i> Retry
                    </button>
                </div>
            `;
            
            // Add error message if not already present
            if (!tabContent.querySelector('.alert-danger')) {
                tabContent.insertAdjacentHTML('afterbegin', errorHtml);
            }
        }
    }

    /**
     * Retry tab initialization
     */
    retryInitialization(tabId) {
        // Remove error message
        const tabContent = document.getElementById(tabId);
        if (tabContent) {
            const errorAlert = tabContent.querySelector('.alert-danger');
            if (errorAlert) {
                errorAlert.remove();
            }
        }
        
        // Reset initialization state
        this.tabStates[tabId].initialized = false;
        
        // Show loading and re-initialize
        this.showTabLoading(tabId);
        setTimeout(() => {
            this.initializeTab(tabId);
            this.hideTabLoading(tabId);
        }, 500);
    }

    /**
     * Trigger tab-specific events
     */
    triggerTabEvents(tabId) {
        // Dispatch custom events for tab activation
        const event = new CustomEvent('tabActivated', {
            detail: { tabId: tabId, sharedData: this.sharedData }
        });
        document.dispatchEvent(event);
        
        // Tab-specific triggers
        if (tabId === 'notebook-content' && typeof loadColumnInsights === 'function' && !window.columnInsightsLoading && !window.columnInsightsData) {
            setTimeout(() => {
                try {
                    loadColumnInsights();
                } catch (error) {
                    console.warn('TabManager: Unable to refresh column insights on tab activation:', error);
                }
            }, 120);
        }

        if (this.previewTabMap[tabId]) {
            const previewKey = this.previewTabMap[tabId];
            try {
                if (this.ensurePreviewCoreReady() && window.DI?.previewPage?.loadDataForTab) {
                    window.DI.previewPage.loadDataForTab(previewKey);
                }
            } catch (error) {
                console.warn(`TabManager: Unable to refresh preview tab ${tabId}:`, error);
            }
        }
    }

    /**
     * Save current state of a tab
     */
    saveTabState(tabId) {
        try {
            switch (tabId) {
                case 'notebook-content':
                    if (typeof getSelectedColumns === 'function') {
                        this.tabStates[tabId].data = {
                            selectedColumns: getSelectedColumns(),
                            timestamp: new Date().toISOString()
                        };
                    }
                    break;
                    
            }
        } catch (error) {
            console.warn(`TabManager: Could not save state for tab ${tabId}:`, error);
        }
    }

    /**
     * Get tab state
     */
    getTabState(tabId) {
        return this.tabStates[tabId];
    }

    /**
     * Update shared data
     */
    updateSharedData(key, value) {
        this.sharedData[key] = value;
        
        // Notify all tabs of shared data update
        const event = new CustomEvent('sharedDataUpdated', {
            detail: { key: key, value: value, allData: this.sharedData }
        });
        document.dispatchEvent(event);
    }

    /**
     * Get shared data
     */
    getSharedData(key = null) {
        return key ? this.sharedData[key] : this.sharedData;
    }

    /**
     * Fetch dataset information and update shared data
     */
    async fetchDatasetInfo() {
        if (!currentDataset?.source_id) {
            console.warn('TabManager: No dataset source_id available');
            return;
        }
        
        try {
            const response = await fetch(`/advanced-eda/api/data-sources/${currentDataset.source_id}/info`);
            const data = await response.json();
            
            if (data.success) {
                this.updateSharedData('dataset', data);
            } else {
                console.error('TabManager: Dataset info error:', data.error);
            }
            
        } catch (error) {
            console.error('TabManager: Error fetching dataset info:', error);
        }
    }

    /**
     * Update URL hash based on active tab
     */
    updateUrlHash(tabId) {
        const tabName = tabId.replace('-content', '');
        if (history.pushState) {
            history.pushState(null, null, `#${tabName}`);
        }
    }

    /**
     * Handle browser back/forward navigation
     */
    handleHashChange() {
        const hash = window.location.hash.replace('#', '');
        if (hash) {
            const targetTabId = `${hash}-content`;
            if (this.tabStates[targetTabId]) {
                this.setActiveTab(targetTabId);
            }
        }
    }

    /**
     * Export data for cross-tab communication
     */
    exportToTab(targetTabId, data, dataType = 'general') {
        // Store data for target tab
        if (!this.tabStates[targetTabId].data) {
            this.tabStates[targetTabId].data = {};
        }
        
        this.tabStates[targetTabId].data[`import_${dataType}`] = {
            data: data,
            timestamp: new Date(),
            sourceTab: this.activeTab
        };
        
        // Switch to target tab
        this.setActiveTab(targetTabId);
        
        // Trigger import event after tab is shown
        setTimeout(() => {
            const event = new CustomEvent('dataImported', {
                detail: { data: data, dataType: dataType, sourceTab: this.activeTab }
            });
            document.getElementById(targetTabId).dispatchEvent(event);
        }, 500);
    }

    /**
     * Get current active tab
     */
    getActiveTab() {
        return this.activeTab;
    }

    /**
     * Check if a tab is initialized
     */
    isTabInitialized(tabId) {
        return this.tabStates[tabId]?.initialized || false;
    }

    /**
     * Reset all tab states (useful for new dataset)
     */
    resetAllTabs() {
        Object.keys(this.tabStates).forEach(tabId => {
            this.tabStates[tabId].initialized = false;
            this.tabStates[tabId].data = null;
        });
        
        this.sharedData = {};
        this.previewInitialized = false;
        
        // Reinitialize current active tab
        if (this.activeTab) {
            this.initializeTab(this.activeTab);
        }
    }
}

// Handle browser navigation  
window.addEventListener('hashchange', () => {
    window.tabManager.handleHashChange();
});

// Handle initial hash on page load
document.addEventListener('DOMContentLoaded', () => {
    // Check for initial hash
    if (window.location.hash) {
        tabManagerInstance.handleHashChange();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TabManager;
}

// Make TabManager globally available
window.TabManager = TabManager;

// Create global instance  
const tabManagerInstance = new TabManager();
window.tabManager = tabManagerInstance;

} // End of redeclaration prevention if statement