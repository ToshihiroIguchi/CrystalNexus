/**
 * CrystalNexus Utility Functions
 * Common utility functions used across the application
 */

/**
 * Smart clipboard copy function with fallback support
 * @param {string} text - Text to copy to clipboard
 * @returns {Promise<Object>} Result object with success status and method used
 */
async function smartCopyToClipboard(text) {
    try {
        // Modern browsers with Clipboard API
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
            return { success: true, method: 'modern' };
        } else {
            // Fallback for older browsers or non-HTTPS
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.cssText = 'position:fixed;opacity:0;pointer-events:none;left:-9999px;';
            document.body.appendChild(textarea);
            textarea.select();
            textarea.setSelectionRange(0, text.length);
            
            const success = document.execCommand('copy');
            document.body.removeChild(textarea);
            
            if (success) {
                return { success: true, method: 'legacy' };
            } else {
                throw new Error('execCommand copy failed');
            }
        }
    } catch (error) {
        console.error('Copy operation failed:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Show copy feedback tooltip
 * @param {HTMLElement} button - Button element to show tooltip near
 * @param {boolean} success - Whether the copy operation was successful
 */
function showCopyFeedback(button, success) {
    const tooltip = document.createElement('div');
    tooltip.className = 'copy-tooltip';
    tooltip.textContent = success ? 'Copied' : 'Copy failed';
    tooltip.style.cssText = `
        position: absolute;
        background: ${success ? '#27ae60' : '#e74c3c'};
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        white-space: nowrap;
        z-index: 1000;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.2s;
    `;
    
    // Position tooltip above button
    const rect = button.getBoundingClientRect();
    tooltip.style.left = (rect.left + window.scrollX - 10) + 'px';
    tooltip.style.top = (rect.top + window.scrollY - 30) + 'px';
    
    document.body.appendChild(tooltip);
    
    // Show tooltip
    requestAnimationFrame(() => {
        tooltip.style.opacity = '1';
    });
    
    // Hide and remove tooltip
    setTimeout(() => {
        tooltip.style.opacity = '0';
        setTimeout(() => {
            if (tooltip.parentNode) {
                document.body.removeChild(tooltip);
            }
        }, 200);
    }, 1500);
}

/**
 * Show notification toast message
 * @param {string} message - Message to display
 * @param {string} type - Notification type: 'success', 'error', 'warning', 'info'
 */
function showNotification(message, type = 'info') {
    // Remove existing notification if present
    const existing = document.querySelector('.notification');
    if (existing) {
        existing.remove();
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    
    // Set colors based on type
    const colors = {
        'success': { bg: '#2ecc71', text: 'white' },
        'error': { bg: '#e74c3c', text: 'white' },
        'warning': { bg: '#f39c12', text: 'white' },
        'info': { bg: '#3498db', text: 'white' }
    };
    
    const color = colors[type] || colors.info;
    
    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        backgroundColor: color.bg,
        color: color.text,
        padding: '12px 20px',
        borderRadius: '4px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        zIndex: '10000',
        fontSize: '14px',
        maxWidth: '300px',
        opacity: '0',
        transform: 'translateY(-20px)',
        transition: 'all 0.3s ease'
    });
    
    // Add to page
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateY(0)';
    }, 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-20px)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

/**
 * Clear status messages from the interface
 */
function clearStatusMessage() {
    const statusElements = document.querySelectorAll('.status-message');
    statusElements.forEach(element => {
        element.style.display = 'none';
    });
}

/**
 * Format execution time for display
 * @param {number} executionTime - Execution time in seconds
 * @returns {string} Formatted time string
 */
function formatExecutionTime(executionTime) {
    if (executionTime < 1) {
        return `${Math.round(executionTime * 1000)}ms`;
    } else if (executionTime < 60) {
        return `${executionTime.toFixed(1)}s`;
    } else {
        const minutes = Math.floor(executionTime / 60);
        const seconds = Math.round(executionTime % 60);
        return `${minutes}m ${seconds}s`;
    }
}

/**
 * Validate supercell input values
 * @param {number} a - A dimension
 * @param {number} b - B dimension  
 * @param {number} c - C dimension
 * @returns {Object} Validation result with isValid boolean and error message
 */
function validateSupercellInputs(a, b, c) {
    const errors = [];
    
    if (!Number.isInteger(a) || a < 1 || a > 10) {
        errors.push('A dimension must be between 1 and 10');
    }
    if (!Number.isInteger(b) || b < 1 || b > 10) {
        errors.push('B dimension must be between 1 and 10');
    }
    if (!Number.isInteger(c) || c < 1 || c > 10) {
        errors.push('C dimension must be between 1 and 10');
    }
    
    return {
        isValid: errors.length === 0,
        errors: errors
    };
}

/**
 * Debounce function to limit the rate of function execution
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Safe JSON parse with error handling
 * @param {string} jsonString - JSON string to parse
 * @param {*} defaultValue - Default value if parsing fails
 * @returns {*} Parsed object or default value
 */
function safeJSONParse(jsonString, defaultValue = null) {
    try {
        return JSON.parse(jsonString);
    } catch (error) {
        console.warn('JSON parse failed:', error);
        return defaultValue;
    }
}