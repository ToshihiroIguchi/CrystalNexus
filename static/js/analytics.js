
// Local helper (analytics.html does not load utils.js):
// escape HTML special characters to prevent DOM XSS when inserting via innerHTML
function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
}

// require_analytics_access (main.py) accepts either an X-Analytics-Token header
// or a ?token= query parameter. A browser cannot set a header on an address-bar
// navigation, so the dashboard is always reached as /analytics?token=... when a
// token is configured; forward that same token on the three XHRs below, which
// would otherwise 401 -- even from localhost.
function getAnalyticsToken() {
    try {
        return new URLSearchParams(window.location.search).get('token') || '';
    } catch (e) {
        return '';
    }
}

async function analyticsFetch(path) {
    const token = getAnalyticsToken();
    const options = { credentials: 'same-origin' };
    if (token) {
        options.headers = { 'X-Analytics-Token': token };
    }
    const response = await fetch(path, options);
    if (response.status === 401) {
        showAnalyticsError(
            'Analytics access denied (401). Open this page as ' +
            '/analytics?token=YOUR_TOKEN using the token printed by ' +
            'start_crystalnexus.py.'
        );
    }
    return response;
}

function showAnalyticsError(message) {
    const containerSelector = '.analytics-container';
    const container = document.querySelector(containerSelector) || document.body;
    let banner = document.getElementById('analyticsErrorBanner');
    if (banner) {
        // Idempotent: three failing fetches must produce exactly one banner.
        return;
    }
    banner = document.createElement('div');
    banner.id = 'analyticsErrorBanner';
    banner.style.cssText = 'background:#fee;color:#900;border:1px solid #c00;' +
        'padding:12px 16px;margin-bottom:16px;border-radius:4px;font-weight:bold;';
    banner.textContent = message; // textContent, not innerHTML -- message may include user-influenced text
    container.insertBefore(banner, container.firstChild);
}

document.addEventListener('DOMContentLoaded', async () => {
    // Fetch and render data
    await loadSummaryData();
    await loadPopularSamples();
    await loadRecentActivity();
});

async function loadSummaryData() {
    try {
        const response = await analyticsFetch('/api/analytics/summary');
        if (!response.ok) throw new Error('Failed to fetch summary');
        const data = await response.json();

        renderDailyAccessChart(data.daily_access);
        renderFeatureUsageChart(data.feature_usage);
    } catch (error) {
        console.error('Error loading summary:', error);
    }
}

async function loadPopularSamples() {
    try {
        const response = await analyticsFetch('/api/analytics/ranking');
        if (!response.ok) throw new Error('Failed to fetch ranking');
        const data = await response.json();

        const tbody = document.querySelector('#popularSamplesTable tbody');
        tbody.innerHTML = '';

        data.popular_samples.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${escapeHtml(item.name || 'Unknown')}</td>
                <td>${escapeHtml(item.count)}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        console.error('Error loading popular samples:', error);
    }
}

async function loadRecentActivity() {
    try {
        const response = await analyticsFetch('/api/analytics/recent');
        if (!response.ok) throw new Error('Failed to fetch activity');
        const data = await response.json();

        const tbody = document.querySelector('#recentActivityTable tbody');
        tbody.innerHTML = '';

        data.recent_events.forEach(event => {
            const tr = document.createElement('tr');
            const date = new Date(event.timestamp).toLocaleString();

            let details = '';
            if (event.parameters) {
                // If parameters is a string, parse it
                let params = event.parameters;
                if (typeof params === 'string') {
                    try { params = JSON.parse(params); } catch (e) { }
                }

                if (event.event_type === 'relax') {
                    details = `Steps: ${params.steps}, fmax: ${params.fmax}`;
                } else {
                    details = JSON.stringify(params).substring(0, 50);
                }
            } else if (event.execution_time) {
                details = `${event.execution_time.toFixed(2)}s`;
            }

            tr.innerHTML = `
                <td>${escapeHtml(date)}</td>
                <td><span class="status-badge status-success">${escapeHtml(event.event_type)}</span></td>
                <td>${escapeHtml(event.filename || event.formula || '-')}</td>
                <td>${escapeHtml(details)}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        console.error('Error loading recent activity:', error);
    }
}

function renderDailyAccessChart(data) {
    const ctx = document.getElementById('dailyAccessChart').getContext('2d');

    // Sort by date just in case
    data.sort((a, b) => new Date(a.date) - new Date(b.date));

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.date),
            datasets: [{
                label: 'Page Views / API Calls',
                data: data.map(d => d.count),
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

function renderFeatureUsageChart(data) {
    const ctx = document.getElementById('featureUsageChart').getContext('2d');

    const colors = [
        'rgba(255, 99, 132, 0.7)',
        'rgba(54, 162, 235, 0.7)',
        'rgba(255, 206, 86, 0.7)',
        'rgba(75, 192, 192, 0.7)',
        'rgba(153, 102, 255, 0.7)',
    ];

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.map(d => d.type),
            datasets: [{
                data: data.map(d => d.count),
                backgroundColor: colors.slice(0, data.length)
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
        }
    });
}
