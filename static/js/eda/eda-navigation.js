// Global EDA Navigation System - Simple Working Version
// Provides floating scroll navigation for all EDA tabs

function addGlobalNavigationStyles() {
    // Check if styles already added
    if (document.getElementById('edaNavigationStyles')) {
        return;
    }
    
    const style = document.createElement('style');
    style.id = 'edaNavigationStyles';
    style.textContent = `
        /* Global floating navigation */
        #edaScrollNav {
            position: fixed;
            left: 24px;
            bottom: 32px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 999;
        }

        #edaScrollNav .eda-scroll-btn {
            width: 42px;
            height: 42px;
            border-radius: 50%;
            border: none;
            background: #343a40;
            color: #fff;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: pointer;
        }

        #edaScrollNav .eda-scroll-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.3);
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            #edaScrollNav {
                left: unset;
                right: 18px;
            }
        }
    `;
    document.head.appendChild(style);
}

function ensureFloatingNavigation() {
    // Remove any existing navigation to avoid duplicates
    const existingNav = document.getElementById('edaScrollNav');
    if (existingNav) {
        existingNav.remove();
    }
    
    // Create new navigation
    const nav = document.createElement('div');
    nav.id = 'edaScrollNav';
    nav.innerHTML = `
        <button id="edaScrollToTopBtn" class="eda-scroll-btn" title="Scroll to top">
            <i class="bi bi-arrow-up"></i>
        </button>
        <button id="edaScrollToBottomBtn" class="eda-scroll-btn" title="Scroll to bottom">
            <i class="bi bi-arrow-down"></i>
        </button>
    `;
    document.body.appendChild(nav);

    // Attach event listeners with unique IDs to avoid conflicts
    const scrollToTopBtn = document.getElementById('edaScrollToTopBtn');
    if (scrollToTopBtn) {
        scrollToTopBtn.onclick = () => {
            console.log('Scrolling to top');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        };
    }

    const scrollToBottomBtn = document.getElementById('edaScrollToBottomBtn');
    if (scrollToBottomBtn) {
        scrollToBottomBtn.onclick = () => {
            console.log('Scrolling to bottom');
            window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' });
        };
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('EDA Navigation initializing...');
    addGlobalNavigationStyles();
    ensureFloatingNavigation();
    
    // Listen for tab switches to reinitialize navigation
    const tabButtons = document.querySelectorAll('[data-bs-toggle="tab"]');
    tabButtons.forEach(button => {
        button.addEventListener('shown.bs.tab', function() {
            console.log('Tab switched, reinitializing navigation...');
            setTimeout(() => {
                ensureFloatingNavigation();
            }, 100);
        });
    });
    
    // Also reinitialize on window resize
    window.addEventListener('resize', function() {
        setTimeout(() => {
            ensureFloatingNavigation();
        }, 100);
    });
});

// Export functions for manual usage
window.ensureFloatingNavigation = ensureFloatingNavigation;
window.addGlobalNavigationStyles = addGlobalNavigationStyles;