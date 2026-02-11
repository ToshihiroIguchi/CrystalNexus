class AutoModeChart {
    constructor(canvasId) {
        this.canvasId = canvasId;
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.chart = null;
        this.dataPoints = [];
        this.maxDataPoints = 50; // Sliding window size

        this.initChart();
    }

    initChart() {
        this.chart = new Chart(this.ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Energy (eV/atom)',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    fill: true,
                    tension: 0.1 // Slight curve for smoother look
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0
                },
                hover: {
                    animationDuration: 0
                },
                responsiveAnimationDuration: 0,
                plugins: {
                    legend: {
                        display: false // Hide legend
                    },
                    title: {
                        display: false // Hide title
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function (context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y.toFixed(6) + ' eV';
                                }
                                return label;
                            },
                            title: function (context) {
                                return context[0].label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: false // Hide x-axis entirely
                    },
                    y: {
                        display: false // Hide y-axis entirely
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                },
                layout: {
                    padding: 5 // Minimal padding
                }
            }
        });
    }

    reset() {
        this.dataPoints = [];
        this.chart.data.labels = [];
        this.chart.data.datasets[0].data = [];
        this.chart.update();
    }

    addDataPoint(label, energy) {
        // Add new data point
        this.dataPoints.push({ label, energy });

        // Update chart data
        this.chart.data.labels.push(label);
        this.chart.data.datasets[0].data.push(energy);

        // Maintain sliding window
        if (this.chart.data.labels.length > this.maxDataPoints) {
            this.chart.data.labels.shift();
            this.chart.data.datasets[0].data.shift();
        }

        // Efficient update
        this.chart.update('none'); // 'none' mode prevents animation for performance
    }

    // Handle resize if needed
    resize() {
        this.chart.resize();
    }

    destroy() {
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
    }
}

// Global instance variable
window.autoModeChart = null;

// Initialize chart when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    // Check if canvas exists, if not, it will be initialized when the Auto Mode UI is shown
    const chartCanvas = document.getElementById('auto-mode-chart');
    if (chartCanvas) {
        window.autoModeChart = new AutoModeChart('auto-mode-chart');
    }
});
