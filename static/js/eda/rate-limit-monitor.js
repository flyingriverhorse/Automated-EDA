/**
 * Rate Limit Monitor for Custom Analysis
 * Displays current rate limit status and manages user feedback
 */

class RateLimitMonitor {
    constructor() {
        this.isMonitoring = false;
        this.rateLimitInfo = null;
        this.systemStats = null;
        this.updateInterval = null;
        this.sessionStartTime = null;
        this.init();
    }

    init() {
        console.log('Initializing rate limit monitor...');
        this.createRateLimitDisplay();
        
        // Force show the monitor
        setTimeout(() => {
            const display = document.getElementById('rateLimitDisplay');
            if (display) {
                display.classList.remove('d-none');
                display.style.display = 'block';
                console.log('Rate limit display shown');
            } else {
                console.warn('Rate limit display element not found');
            }
        }, 100);
        
        // Start with empty data instead of mock data
        this.useEmptyRateLimitData();
        this.useEmptySystemStats();
        
        this.startMonitoring();
        
        // Update display every 5 seconds for responsive monitoring (not too frequent to avoid button interference)
        this.updateInterval = setInterval(() => {
            console.log('🔄 Auto-refreshing rate limit data (5s interval)');
            this.fetchRateLimitStatus();
            this.fetchSystemStats();
        }, 5000);
    }

    createRateLimitDisplay() {
        // Check if rate limit display already exists
        const existingDisplay = document.getElementById('rateLimitDisplay');
        if (existingDisplay) {
            // Populate the existing container
            this.populateRateLimitDisplay(existingDisplay);
            return;
        }

        // If no existing container, create a new one
        this.createNewRateLimitDisplay();
    }

    populateRateLimitDisplay(container) {
        console.log('Populating existing rate limit display container');
        
        // Add CSS class instead of inline styles
        container.style.display = 'block';
        container.classList.add('rate-limit-container-custom');
        
        container.innerHTML = this.getRateLimitHTML();
        container.classList.remove('d-none');
        
        console.log('Rate limit display populated successfully');
        
        // Add CSS for rate limit display
        // Note: CSS styles moved to eda.css for cleaner architecture
        // this.addRateLimitStyles();
        this.sessionStartTime = Date.now();
    }

    createNewRateLimitDisplay() {
        console.log('Creating new rate limit display');
        const rateLimitHtml = `<div id="rateLimitDisplay" class="rate-limit-display mb-3">${this.getRateLimitHTML()}</div>`;

        // Insert in different locations
        let inserted = false;
        
        // First try: after the toolbar
        const toolbar = document.querySelector('.analysis-toolbar');
        if (toolbar) {
            toolbar.insertAdjacentHTML('afterend', rateLimitHtml);
            inserted = true;
        }
        
        // Second try: after examples panel
        if (!inserted) {
            const examplesPanel = document.querySelector('#examplesPanel');
            if (examplesPanel) {
                examplesPanel.insertAdjacentHTML('afterend', rateLimitHtml);
                inserted = true;
            }
        }
        
        // Third try: at the beginning of custom analysis container
        if (!inserted) {
            const container = document.querySelector('.custom-analysis-container, #customAnalysisTabPane, .tab-pane.fade.active, .tab-pane');
            if (container) {
                container.insertAdjacentHTML('afterbegin', rateLimitHtml);
                inserted = true;
            }
        }

        // Fourth try: Before code cells
        if (!inserted) {
            const codeCells = document.querySelector('#customCodeCells');
            if (codeCells) {
                codeCells.insertAdjacentHTML('beforebegin', rateLimitHtml);
                inserted = true;
            }
        }

        if (!inserted) {
            console.warn('Could not find suitable location for rate limit display');
        }

        // Add CSS for rate limit display
        // Note: CSS styles moved to eda.css for cleaner architecture
        // this.addRateLimitStyles();
        this.sessionStartTime = Date.now();
    }

    getRateLimitHTML() {
        return `
            <div class="alert alert-primary py-2 px-3 mb-2 border-0 rate-limit-header-gradient">
                <div class="d-flex align-items-center justify-content-between">
                    <div class="d-flex align-items-center">
                        <span class="badge bg-light text-dark me-2">
                            Executions: <span id="rateLimitCurrent">0</span>/<span id="rateLimitMax">20</span>
                        </span>
                        <span class="badge bg-light text-dark me-2">
                            Concurrent: <span id="rateLimitConcurrent">0</span>/<span id="rateLimitMaxConcurrent">1</span>
                        </span>
                    </div>
                    <div class="d-flex align-items-center">
                         <span class="badge bg-light text-dark me-2">
                            Users: <span id="activeUsers">0</span>
                        </span>
                        <span class="badge bg-light text-dark">
                            CPU: <span id="systemCPU">0%</span> | Memory: <span id="systemMemory">0%</span>
                        </span>
                    </div>
                    <div class="d-flex align-items-center">
                        <button class="btn btn-sm btn-outline-light" onclick="window.rateLimitMonitor?.forceRefresh()" title="Refresh Now">
                            <i class="bi bi-arrow-clockwise"></i>
                        </button>
                    </div>
                </div>
                <div class="progress mt-2 rate-limit-progress-thin">
                    <div class="progress-bar bg-light rate-limit-progress-bar-fill" id="executionProgressBar" role="progressbar"></div>
                </div>
            </div>
        `;
    }

    // CSS styles moved to eda.css for cleaner architecture
    // addRateLimitStyles() function removed

    startMonitoring() {
        this.isMonitoring = true;
        
        // Fetch real data immediately 
        this.fetchRateLimitStatus();
        this.fetchSystemStats();
        
        // Initial display update
        setTimeout(() => {
            this.updateDisplay();
        }, 500);
    }

    async fetchRateLimitStatus() {
        try {
            const response = await fetch('/advanced-eda/api/rate-limit-status');
            if (response.ok) {
                this.rateLimitInfo = await response.json();
                console.log('✅ Rate limit info fetched from API:', this.rateLimitInfo);
                this.updateDisplay();
            } else {
                console.warn(`⚠️ Rate limit API returned: ${response.status} ${response.statusText}`);
                console.log('📝 Using empty default state (API not available)');
                this.useEmptyRateLimitData();
            }
        } catch (error) {
            console.warn('❌ Failed to fetch rate limit status:', error);
            console.log('📝 Using empty default state (no executions yet)');
            this.useEmptyRateLimitData();
        }
    }

    async fetchSystemStats() {
        try {
            const response = await fetch('/advanced-eda/api/system-stats');
            if (response.ok) {
                this.systemStats = await response.json();
                console.log('✅ System stats fetched from API:', this.systemStats);
                this.updateDisplay();
            } else {
                console.warn(`⚠️ System stats API returned: ${response.status} ${response.statusText}`);
                console.log('📝 Using empty default state (API not available)');
                this.useEmptySystemStats();
            }
        } catch (error) {
            console.warn('❌ Failed to fetch system stats:', error);
            console.log('📝 Using empty default state (no activity yet)');
            this.useEmptySystemStats();
        }
    }

    useEmptyRateLimitData() {
        // Show realistic empty state (no executions yet)
        this.rateLimitInfo = {
            max_executions_per_minute: 20,  // Default limit from rate_limiter.py
            remaining_executions: 20,
            max_concurrent_executions: 1,   // Default limit from rate_limiter.py
            concurrent_executions: 0,
            executions_last_minute: 0,
            window_start: new Date().toISOString(),
            status: 'No executions yet'
        };
        console.log('📊 Empty rate limit data (no activity):', this.rateLimitInfo);
        this.updateDisplay();
    }

    useEmptySystemStats() {
        // Show realistic empty system state
        this.systemStats = {
            memory_usage: 0,
            cpu_usage: 0,
            active_users: 0,
            total_executions: 0,
            system_load: 0,
            timestamp: new Date().toISOString(),
            status: 'No system activity'
        };
        console.log('💻 Empty system stats (no activity):', this.systemStats);
        this.updateDisplay();
    }

    useMockRateLimitData() {
        // Redirect to empty data instead of mock data
        console.log('📊 Mock data requested, using empty data instead');
        this.useEmptyRateLimitData();
    }

    useMockSystemStats() {
        // Redirect to empty data instead of mock data
        console.log('💻 Mock system stats requested, using empty data instead');
        this.useEmptySystemStats();
    }

    updateDisplay() {
        this.updateRateLimitDisplay();
        this.updateSessionDuration();
        this.updateLastRefreshTime();
    }

    updateLastRefreshTime() {
        const timeElement = document.getElementById('lastUpdateTime');
        if (timeElement) {
            const now = new Date();
            const timeString = now.toLocaleTimeString('en-US', { 
                hour12: false, 
                hour: '2-digit', 
                minute: '2-digit',
                second: '2-digit'
            });
            timeElement.textContent = `Last updated: ${timeString}`;
        }
    }

    updateRateLimitDisplay() {
        if (!this.rateLimitInfo) return;

        // Update current usage
        const currentExecutions = this.rateLimitInfo.max_executions_per_minute - this.rateLimitInfo.remaining_executions;
        
        const rateLimitCurrent = document.getElementById('rateLimitCurrent');
        const rateLimitMax = document.getElementById('rateLimitMax');
        const rateLimitConcurrent = document.getElementById('rateLimitConcurrent');
        const rateLimitMaxConcurrent = document.getElementById('rateLimitMaxConcurrent');
        
        if (rateLimitCurrent) rateLimitCurrent.textContent = currentExecutions;
        if (rateLimitMax) rateLimitMax.textContent = this.rateLimitInfo.max_executions_per_minute;
        if (rateLimitConcurrent) rateLimitConcurrent.textContent = this.rateLimitInfo.concurrent_executions;
        if (rateLimitMaxConcurrent) rateLimitMaxConcurrent.textContent = this.rateLimitInfo.max_concurrent_executions;

        // Update progress bar
        const progressBar = document.getElementById('executionProgressBar');
        if (progressBar) {
            const percentage = (currentExecutions / this.rateLimitInfo.max_executions_per_minute) * 100;
            progressBar.style.width = `${Math.min(percentage, 100)}%`;
            
            // Update progress bar color based on usage
            if (percentage >= 90) {
                progressBar.className = 'progress-bar bg-danger';
            } else if (percentage >= 70) {
                progressBar.className = 'progress-bar bg-warning';
            } else {
                progressBar.className = 'progress-bar bg-light';
            }
        }

        // Update system stats in the compact view
        if (this.systemStats) {
            const memoryEl = document.getElementById('systemMemory');
            const cpuEl = document.getElementById('systemCPU');
            const usersEl = document.getElementById('activeUsers');

            if (memoryEl) memoryEl.textContent = `${Math.round(this.systemStats.memory_usage)}%`;
            if (cpuEl) cpuEl.textContent = `${Math.round(this.systemStats.cpu_usage)}%`;
            if (usersEl) usersEl.textContent = this.systemStats.active_users || 0;
        }
    }

    // Remove the detailed view methods since we don't need them anymore
    updateDetailedView() {
        // No longer needed in simplified version
    }

    updateSessionDuration() {
        if (!this.rateLimitInfo) return;

        // Update remaining executions
        const remainingEl = document.getElementById('remainingExecutions');
        if (remainingEl) {
            remainingEl.textContent = this.rateLimitInfo.remaining_executions;
        }

        // Update active processes
        const activeEl = document.getElementById('activeProcesses');
        if (activeEl) {
            activeEl.textContent = this.rateLimitInfo.concurrent_executions;
        }

        // Update system stats if available
        if (this.systemStats) {
            const memoryEl = document.getElementById('systemMemory');
            const cpuEl = document.getElementById('systemCPU');
            const usersEl = document.getElementById('activeUsers');

            if (memoryEl) memoryEl.textContent = `${Math.round(this.systemStats.memory_usage)}%`;
            if (cpuEl) cpuEl.textContent = `${Math.round(this.systemStats.cpu_usage)}%`;
            if (usersEl) usersEl.textContent = this.systemStats.active_users || 0;
        }
    }

    updateSessionDuration() {
        if (!this.sessionStartTime) return;
        
        const duration = Date.now() - this.sessionStartTime;
        const minutes = Math.floor(duration / 60000);
        const seconds = Math.floor((duration % 60000) / 1000);
        
        const sessionEl = document.getElementById('sessionDuration');
        if (sessionEl) {
            sessionEl.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    toggleDetailedView() {
        const details = document.getElementById('rateLimitDetails');
        const icon = document.getElementById('toggleDetailsIcon');
        
        if (details) {
            if (details.classList.contains('d-none')) {
                details.classList.remove('d-none');
                if (icon) icon.className = 'bi bi-chevron-up';
            } else {
                details.classList.add('d-none');
                if (icon) icon.className = 'bi bi-chevron-down';
            }
        }
    }

    forceRefresh() {
        console.log('Forcing refresh of rate limit monitor...');
        this.fetchRateLimitStatus();
        this.fetchSystemStats();
        this.showRefreshIndicator();
    }

    showRefreshIndicator() {
        // Show a subtle refresh indicator without DOM manipulation that interferes with buttons
        const refreshBtn = document.querySelector('[onclick*="forceRefresh"]');
        if (refreshBtn) {
            const icon = refreshBtn.querySelector('i');
            if (icon) {
                // Add spinning animation briefly
                icon.style.animation = 'spin 0.5s linear';
                setTimeout(() => {
                    icon.style.animation = '';
                }, 500);
            }
        }
    }

    canExecute() {
        if (!this.rateLimitInfo) return true; // Allow if no rate limit info yet
        
        return this.rateLimitInfo.remaining_executions > 0 && 
               this.rateLimitInfo.concurrent_executions < this.rateLimitInfo.max_concurrent_executions;
    }

    getRemainingExecutions() {
        return this.rateLimitInfo ? this.rateLimitInfo.remaining_executions : 0;
    }

    handleRateLimitError(rateLimitInfo) {
        this.rateLimitInfo = rateLimitInfo;
        this.updateDisplay();
        
        // Show warning message
        const display = document.getElementById('rateLimitDisplay');
        if (display) {
            display.classList.add('rate-limit-error');
            display.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    // Method to track execution start
    trackExecutionStart() {
        if (this.rateLimitInfo) {
            this.rateLimitInfo.concurrent_executions++;
            this.updateDisplay();
        }
    }

    // Method to track execution end
    trackExecutionEnd() {
        if (this.rateLimitInfo && this.rateLimitInfo.concurrent_executions > 0) {
            this.rateLimitInfo.concurrent_executions--;
            this.updateDisplay();
        }
    }

}

// Global rate limit monitor instance
let rateLimitMonitor = null;

// Initialize rate limit monitoring when DOM is ready or when tab becomes active
document.addEventListener('DOMContentLoaded', function() {
    initRateLimitMonitor();
});

// Also initialize when custom analysis tab is shown
document.addEventListener('shown.bs.tab', function(e) {
    if (e.target.id === 'custom-analysis-tab' || e.target.getAttribute('data-bs-target') === '#custom-analysis-content') {
        initRateLimitMonitor();
    }
});

function initRateLimitMonitor() {
    console.log('=== Attempting to initialize rate limit monitor ===');
    
    // Check for required elements
    const customAnalysisContent = document.getElementById('custom-analysis-content');
    const rateLimitDisplay = document.getElementById('rateLimitDisplay');
    
    console.log('Element check:', {
        'custom-analysis-content': !!customAnalysisContent,
        'rateLimitDisplay': !!rateLimitDisplay
    });
    
    // Initialize if we're on the EDA page with custom analysis tab
    if (customAnalysisContent) {
        if (!window.rateLimitMonitor) {
            console.log('Creating new RateLimitMonitor instance...');
            rateLimitMonitor = new RateLimitMonitor();
            
            // Make it available globally for other scripts
            window.rateLimitMonitor = rateLimitMonitor;
            console.log('✅ Rate limit monitor initialized and made available globally');
        } else {
            console.log('Rate limit monitor already exists');
        }
    } else {
        console.log('⚠️ Custom analysis content not found, retrying in 500ms...');
        setTimeout(initRateLimitMonitor, 500);
    }
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RateLimitMonitor;
}

// Make RateLimitMonitor globally available
window.RateLimitMonitor = RateLimitMonitor;

// Ensure global initialization function is available
window.initRateLimitMonitor = function() {
    console.log('Global initRateLimitMonitor called');
    if (!window.rateLimitMonitor) {
        window.rateLimitMonitor = new RateLimitMonitor();
        console.log('✅ Rate limit monitor initialized globally');
    } else {
        console.log('Rate limit monitor already exists, restarting...');
        window.rateLimitMonitor.startMonitoring();
    }
};