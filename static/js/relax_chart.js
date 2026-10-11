/**
 * RelaxLineChart - Raw Canvas line chart for live optimization progress.
 * Minimal sparkline of {x, y} points (x = optimizer step): no axes, ticks or
 * grid. Optional log-scale Y and an optional faint dashed reference line
 * (fmax target). Visual style follows AutoModeChart.
 */
class RelaxLineChart {
    constructor(canvasId, { yLabel = '', logY = false, refLine = null } = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.yLabel = yLabel;
        this.logY = logY;
        this.refLine = refLine;
        this.points = []; // Array of {x, y}

        this.resize();
        // Keep the bound handler so destroy() can remove exactly this listener.
        this._onResize = () => this.resize();
        window.addEventListener('resize', this._onResize);
    }

    reset() {
        this.points = [];
        this.draw();
    }

    addPoints(points) {
        if (!this.ctx || !points || points.length === 0) return;
        for (const p of points) {
            // Non-finite values arrive as null from the server; skip them.
            if (typeof p.x === 'number' && typeof p.y === 'number' && isFinite(p.y)) {
                this.points.push({ x: p.x, y: p.y });
            }
        }
        // The canvas may have been hidden (0x0) when constructed.
        this.resize();
    }

    setRefLine(value) {
        this.refLine = (typeof value === 'number' && isFinite(value)) ? value : null;
        this.draw();
    }

    resize() {
        if (!this.ctx) return; // destroyed
        const rect = this.canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        const dpr = window.devicePixelRatio || 1;
        const w = Math.round(rect.width * dpr);
        const h = Math.round(rect.height * dpr);
        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.canvas.width = w;
            this.canvas.height = h;
        }
        this.draw();
    }

    draw() {
        if (!this.ctx) return;
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        if (width === 0 || height === 0) return;

        const ctx = this.ctx;
        const dpr = window.devicePixelRatio || 1;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width, height);

        // Log scale cannot show non-positive values; they are skipped.
        const pts = this.logY ? this.points.filter(p => p.y > 0) : this.points;
        if (pts.length === 0) return;

        const f = this.logY ? Math.log10 : (v => v);
        const vals = pts.map(p => f(p.y));
        const hasRef = this.refLine !== null && (!this.logY || this.refLine > 0);
        const all = hasRef ? vals.concat(f(this.refLine)) : vals;
        const min = Math.min(...all);
        const max = Math.max(...all);
        const range = max - min;

        const padding = 4;
        const drawHeight = height - padding * 2;
        const toY = v => range > 0
            ? height - padding - ((f(v) - min) / range) * drawHeight
            : height / 2;

        const xs = pts.map(p => p.x);
        const xMin = Math.min(...xs);
        const xSpan = Math.max(Math.max(...xs) - xMin, 1);
        const usableW = width - padding * 2;
        const toX = v => padding + ((v - xMin) / xSpan) * usableW;

        // Very faint reference line (fmax target)
        if (hasRef) {
            const y = toY(this.refLine);
            ctx.strokeStyle = 'rgba(75, 192, 192, 0.35)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
            ctx.setLineDash([]);
        }

        const drawn = pts.map(p => ({ x: toX(p.x), y: toY(p.y) }));

        ctx.beginPath();
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgb(75, 192, 192)';
        ctx.lineJoin = 'round';
        ctx.moveTo(drawn[0].x, drawn[0].y);
        for (let i = 1; i < drawn.length; i++) ctx.lineTo(drawn[i].x, drawn[i].y);
        ctx.stroke();

        // Gradient fill under the line
        if (drawn.length > 1) {
            ctx.save();
            ctx.lineTo(drawn[drawn.length - 1].x, height);
            ctx.lineTo(drawn[0].x, height);
            ctx.closePath();
            const gradient = ctx.createLinearGradient(0, 0, 0, height);
            gradient.addColorStop(0, 'rgba(75, 192, 192, 0.4)');
            gradient.addColorStop(1, 'rgba(75, 192, 192, 0.0)');
            ctx.fillStyle = gradient;
            ctx.fill();
            ctx.restore();
        }

        // Dot at the latest point only
        const last = drawn[drawn.length - 1];
        ctx.fillStyle = 'rgb(75, 192, 192)';
        ctx.beginPath(); ctx.arc(last.x, last.y, 2.5, 0, Math.PI * 2); ctx.fill();
    }

    destroy() {
        window.removeEventListener('resize', this._onResize);
        this.points = [];
        this.ctx = null;
    }
}
