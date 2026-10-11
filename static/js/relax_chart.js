/**
 * RelaxLineChart - Raw Canvas line chart for live optimization progress.
 * Plots {x, y} points (x = optimizer step) with labelled axes and ticks,
 * an optional log-scale Y axis and an optional dashed horizontal reference
 * line (used for the fmax target). Style follows AutoModeChart.
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

    _colors() {
        const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        return dark
            ? { axis: '#9aa0a6', grid: 'rgba(255,255,255,0.12)', text: '#c4c7c5', line: 'rgb(75, 192, 192)', ref: '#ff8a80' }
            : { axis: '#5f6368', grid: 'rgba(0,0,0,0.10)', text: '#444746', line: 'rgb(38, 150, 150)', ref: '#d93025' };
    }

    // Pick "nice" linear ticks covering [min, max].
    _linearTicks(min, max, count) {
        if (!(max > min)) {
            const pad = Math.abs(min) * 0.05 || 0.5;
            min -= pad;
            max += pad;
        }
        const raw = (max - min) / count;
        const mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const norm = raw / mag;
        const step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
        const lo = Math.floor(min / step + 1e-9) * step;
        const hi = Math.ceil(max / step - 1e-9) * step;
        const ticks = [];
        for (let v = lo; v <= hi + step * 0.5; v += step) ticks.push(v);
        return { ticks, lo, hi, step };
    }

    _fmt(v, step) {
        if (v !== 0 && (Math.abs(v) >= 1e5 || Math.abs(v) < 1e-3)) return v.toExponential(0);
        const decimals = step > 0 ? Math.max(0, Math.min(6, -Math.floor(Math.log10(step)))) : 2;
        return v.toFixed(decimals);
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

        const c = this._colors();
        const left = 50, right = 10, top = 5, bottom = 28;
        const pw = width - left - right;
        const ph = height - top - bottom;
        if (pw <= 10 || ph <= 10) return;

        // Y values to scale against: data plus the reference line
        let ys = this.points.map(p => p.y);
        if (this.refLine !== null) ys.push(this.refLine);
        if (this.logY) ys = ys.filter(v => v > 0); // log axis cannot show <= 0
        const xs = this.points.map(p => p.x);

        const hasData = ys.length > 0 && xs.length > 0;
        const yMin = hasData ? Math.min(...ys) : (this.logY ? 0.01 : 0);
        const yMax = hasData ? Math.max(...ys) : 1;
        const xMax = xs.length > 0 ? Math.max(...xs, 5) : 5;

        // Build Y ticks and value->pixel mapping
        let yTicks, yFmt, toY;
        if (this.logY) {
            const eLo = Math.floor(Math.log10(yMin));
            let eHi = Math.ceil(Math.log10(yMax));
            if (eHi === eLo) eHi = eLo + 1;
            yTicks = [];
            for (let e = eLo; e <= eHi; e++) yTicks.push(Math.pow(10, e));
            toY = v => top + ph - ((Math.log10(Math.max(v, 1e-300)) - eLo) / (eHi - eLo)) * ph;
            yFmt = v => (v >= 0.01 && v < 1e4) ? String(parseFloat(v.toPrecision(3))) : v.toExponential(0);
        } else {
            const t = this._linearTicks(yMin, yMax, 3);
            yTicks = t.ticks;
            toY = v => top + ph - ((v - t.lo) / (t.hi - t.lo)) * ph;
            yFmt = v => this._fmt(v, t.step);
        }
        const xt = this._linearTicks(0, xMax, 5);
        const toX = v => left + ((v - xt.lo) / (xt.hi - xt.lo)) * pw;

        ctx.font = '11px sans-serif';
        ctx.lineWidth = 1;

        // Grid + Y ticks
        ctx.fillStyle = c.text;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (const v of yTicks) {
            const y = Math.round(toY(v)) + 0.5;
            ctx.strokeStyle = c.grid;
            ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + pw, y); ctx.stroke();
            ctx.fillText(yFmt(v), left - 6, y);
        }
        // X ticks
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (const v of xt.ticks) {
            const x = Math.round(toX(v)) + 0.5;
            ctx.strokeStyle = c.grid;
            ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + ph); ctx.stroke();
            ctx.fillText(String(Math.round(v)), x, top + ph + 4);
        }

        // Axes
        ctx.strokeStyle = c.axis;
        ctx.beginPath();
        ctx.moveTo(left + 0.5, top);
        ctx.lineTo(left + 0.5, top + ph + 0.5);
        ctx.lineTo(left + pw, top + ph + 0.5);
        ctx.stroke();

        // Axis labels
        ctx.fillStyle = c.text;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText('Step', left + pw / 2, height - 1);
        // Y label is shown in the HTML title above each chart (saves width).

        ctx.save();
        ctx.beginPath();
        ctx.rect(left, top, pw, ph);
        ctx.clip();

        // Reference line (dashed)
        if (this.refLine !== null && (!this.logY || this.refLine > 0)) {
            const y = toY(this.refLine);
            ctx.strokeStyle = c.ref;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 4]);
            ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + pw, y); ctx.stroke();
            ctx.setLineDash([]);
        }

        // Data line; on a log axis non-positive values break the line
        ctx.strokeStyle = c.line;
        ctx.fillStyle = c.line;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        let pen = false;
        const drawn = [];
        for (const p of this.points) {
            if (this.logY && !(p.y > 0)) { pen = false; continue; }
            const x = toX(p.x), y = toY(p.y);
            if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
            pen = true;
            drawn.push({ x, y });
        }
        ctx.stroke();
        if (drawn.length <= 60) {
            for (const p of drawn) {
                ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, Math.PI * 2); ctx.fill();
            }
        }
        ctx.restore();
    }

    destroy() {
        window.removeEventListener('resize', this._onResize);
        this.points = [];
        this.ctx = null;
    }
}
