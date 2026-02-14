
(async () => {
    // Profiling helper
    const measure = async (name, fn) => {
        const start = performance.now();
        await fn();
        const end = performance.now();
        return end - start;
    };

    // 1. Measure pure chart update cost
    if (!window.autoModeChart) {
        // Initialize if missing (should be there if page loaded correctly)
        if (document.getElementById('auto-mode-chart')) {
            window.autoModeChart = new AutoModeChart('auto-mode-chart');
        } else {
            return { error: "Chart canvas not found" };
        }
    }

    const chartUpdateTimes = [];
    for (let i = 0; i < 50; i++) { // Simulate 50 updates
        const time = await measure('chartUpdate', async () => {
            window.autoModeChart.addDataPoint(`Step ${i}`, -8.0 + (i * 0.01));
        });
        chartUpdateTimes.push(time);
    }

    const avgChartUpdate = chartUpdateTimes.reduce((a, b) => a + b, 0) / chartUpdateTimes.length;
    const maxChartUpdate = Math.max(...chartUpdateTimes);

    // 2. Measure DOM update cost (simulated)
    const statusDiv = document.getElementById('execute-status-text');
    const domUpdateTimes = [];
    if (statusDiv) {
        for (let i = 0; i < 50; i++) {
            const time = await measure('domUpdate', async () => {
                statusDiv.innerText = `Testing atom ${i}...`;
                // Force layout reflow?
                statusDiv.offsetHeight;
            });
            domUpdateTimes.push(time);
        }
    }
    const avgDomUpdate = domUpdateTimes.length ? domUpdateTimes.reduce((a, b) => a + b, 0) / domUpdateTimes.length : 0;

    // 3. Check for any heavy event listeners or loops in main code
    // This is harder to script purely from outside, but we can look at the loop structure in our report.

    return {
        avgChartUpdateMs: avgChartUpdate.toFixed(3),
        maxChartUpdateMs: maxChartUpdate.toFixed(3),
        avgDomUpdateMs: avgDomUpdate.toFixed(3),
        totalChartOverheadFor8AtomsMs: (avgChartUpdate * 8).toFixed(3),
        message: "Profiling complete. Chart update is likely very fast on its own."
    };
})();
