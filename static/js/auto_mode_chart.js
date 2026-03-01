/**
 * AutoModeChart - Raw Canvas Implementation
 * Replaces Chart.js to ensure strict 100% vertical fill of the container.
 */
class AutoModeChart {
    constructor(canvasId) {
        this.canvasId = canvasId;
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.dataPoints = []; // Array of {label, energy}
        this.maxDataPoints = 50;

        // Handle high DPI displays
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    reset(totalPoints = 50) {
        this.maxDataPoints = totalPoints;
        this.dataPoints = [];
        this.clear();
    }

    addDataPoint(label, energy) {
        // Enforce resize check in case it was hidden during init
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        if ((this.canvas.width !== rect.width * dpr || this.canvas.height !== rect.height * dpr) && rect.width > 0 && rect.height > 0) {
            this.resize();
        }

        this.dataPoints.push({ label, energy });
        this.draw();
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        const dpr = window.devicePixelRatio || 1;

        // Only resize if dimensions changed to avoid flickering/clearing
        if (this.canvas.width !== rect.width * dpr || this.canvas.height !== rect.height * dpr) {
            this.canvas.width = rect.width * dpr;
            this.canvas.height = rect.height * dpr;
            this.ctx.scale(dpr, dpr);
        }

        this.draw();
    }

    draw() {
        // Get dimensions in CSS pixels
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        this.clear();

        if (this.dataPoints.length === 0) return;

        // 1. Calculate Min/Max for strict scaling
        let min = Infinity;
        let max = -Infinity;
        for (const p of this.dataPoints) {
            if (p.energy < min) min = p.energy;
            if (p.energy > max) max = p.energy;
        }

        // 2. Setup Drawing Parameters
        const padding = 2; // Keep line width inside canvas
        const drawHeight = height - (padding * 2);
        const yRange = max - min;

        // X-axis step (based on maxDataPoints window)
        // We always span the full width based on maxDataPoints? 
        // Or do we fill as we go? 
        // Previous behavior: "Sliding window". 
        // Let's assume the x-axis represents the full window (0 to maxDataPoints-1)
        const xStep = width / (this.maxDataPoints > 1 ? this.maxDataPoints - 1 : 1);

        this.ctx.beginPath();
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = 'rgb(75, 192, 192)';
        this.ctx.lineJoin = 'round';

        // 3. Generate Path
        const points = this.dataPoints.map((p, i) => {
            let x = i * xStep;

            // Normalize Y: (value - min) / range -> 0..1
            // Invert because Canvas Y is 0 at top: 1 - normalized
            let normalizedY = 0.5; // Default middle
            if (yRange > 0) {
                normalizedY = (p.energy - min) / yRange;
            }

            let y = height - padding - (normalizedY * drawHeight);
            return { x, y };
        });

        // Move to first point
        if (points.length > 0) {
            this.ctx.moveTo(points[0].x, points[0].y);
            for (let i = 1; i < points.length; i++) {
                this.ctx.lineTo(points[i].x, points[i].y);
            }
        }
        this.ctx.stroke();

        // 4. Draw Gradient Fill (Optional, matches original look)
        if (points.length > 1) {
            this.ctx.save();
            this.ctx.lineTo(points[points.length - 1].x, height);
            this.ctx.lineTo(points[0].x, height);
            this.ctx.closePath();

            const gradient = this.ctx.createLinearGradient(0, 0, 0, height);
            gradient.addColorStop(0, 'rgba(75, 192, 192, 0.4)');
            gradient.addColorStop(1, 'rgba(75, 192, 192, 0.0)');
            this.ctx.fillStyle = gradient;
            this.ctx.fill();
            this.ctx.restore();
        }

        // 5. Draw Dots
        this.ctx.fillStyle = 'rgb(75, 192, 192)';
        for (const p of points) {
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    destroy() {
        this.dataPoints = [];
        this.ctx = null;
    }
}
