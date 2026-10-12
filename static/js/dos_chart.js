/**
 * DOSChart - Raw Canvas chart for a predicted electronic density of states.
 * x = E - E_F (eV), y = DOS (states/eV). Filled area, Fermi-level line at
 * E = 0, axis ticks and a hover crosshair
 * with a value tooltip. Visual style follows RelaxLineChart (teal series).
 */
class DOSChart {
    constructor(canvasId, { yLabel = 'DOS (states/eV)' } = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.yLabel = yLabel;
        this.x = [];       // E - E_F (eV)
        this.y = [];       // DOS
        this.hoverIndex = -1;

        this._onResize = () => this.resize();
        this._onMove = (e) => this._handleMove(e);
        this._onLeave = () => { this.hoverIndex = -1; this.draw(); };
        window.addEventListener('resize', this._onResize);
        this.canvas.addEventListener('mousemove', this._onMove);
        this.canvas.addEventListener('mouseleave', this._onLeave);
        this.resize();
    }

    setData(x, y) {
        this.x = Array.isArray(x) ? x : [];
        this.y = Array.isArray(y) ? y : [];
        this.hoverIndex = -1;
        this.resize();
    }

    setY(y) {
        this.y = Array.isArray(y) ? y : [];
        this.draw();
    }

    setYLabel(label) {
        this.yLabel = label;
        this.draw();
    }

    clear() {
        this.x = [];
        this.y = [];
        this.hoverIndex = -1;
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

    // Plot-area margins in CSS pixels (room for tick labels and axis titles).
    _margins() {
        return { left: 58, right: 14, top: 12, bottom: 38 };
    }

    _layout() {
        const rect = this.canvas.getBoundingClientRect();
        const m = this._margins();
        const n = this.x.length;
        if (rect.width === 0 || rect.height === 0 || n === 0) return null;
        const xMin = this.x[0];
        const xMax = this.x[n - 1];
        let yMax = 0;
        for (const v of this.y) if (typeof v === 'number' && isFinite(v) && v > yMax) yMax = v;
        if (yMax <= 0) yMax = 1;
        yMax *= 1.05;
        const pw = rect.width - m.left - m.right;
        const ph = rect.height - m.top - m.bottom;
        return {
            width: rect.width, height: rect.height, m, pw, ph, xMin, xMax, yMax,
            toX: v => m.left + ((v - xMin) / (xMax - xMin || 1)) * pw,
            toY: v => m.top + ph - (v / yMax) * ph,
        };
    }

    // "Nice" tick step (1, 2, 5 x 10^k) giving roughly `count` ticks.
    _niceStep(range, count) {
        const raw = range / Math.max(count, 1);
        const mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const norm = raw / mag;
        const f = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10;
        return f * mag;
    }

    _formatTick(v, step) {
        const decimals = Math.max(0, -Math.floor(Math.log10(step)));
        return v.toFixed(Math.min(decimals, 4));
    }

    draw() {
        if (!this.ctx) return;
        const L = this._layout();
        const ctx = this.ctx;
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);
        if (!L) return;

        const { m, pw, ph, toX, toY, xMin, xMax, yMax } = L;
        const teal = 'rgb(75, 192, 192)';
        ctx.font = '11px sans-serif';
        ctx.fillStyle = '#555';
        ctx.strokeStyle = '#e1e4e8';
        ctx.lineWidth = 1;

        // Grid + ticks
        const xStep = this._niceStep(xMax - xMin, 8);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (let v = Math.ceil(xMin / xStep) * xStep; v <= xMax + 1e-9; v += xStep) {
            const px = toX(v);
            ctx.strokeStyle = '#eef0f2';
            ctx.beginPath(); ctx.moveTo(px, m.top); ctx.lineTo(px, m.top + ph); ctx.stroke();
            ctx.fillStyle = '#555';
            ctx.fillText(this._formatTick(Math.abs(v) < 1e-9 ? 0 : v, xStep), px, m.top + ph + 4);
        }
        const yStep = this._niceStep(yMax, 5);
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let v = 0; v <= yMax + 1e-9; v += yStep) {
            const py = toY(v);
            ctx.strokeStyle = '#eef0f2';
            ctx.beginPath(); ctx.moveTo(m.left, py); ctx.lineTo(m.left + pw, py); ctx.stroke();
            ctx.fillStyle = '#555';
            ctx.fillText(this._formatTick(v, yStep), m.left - 6, py);
        }

        // Axes frame
        ctx.strokeStyle = '#999';
        ctx.beginPath();
        ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + ph); ctx.lineTo(m.left + pw, m.top + ph);
        ctx.stroke();

        // Filled DOS area + outline (clipped to the plot area)
        ctx.save();
        ctx.beginPath();
        ctx.rect(m.left, m.top, pw, ph);
        ctx.clip();
        ctx.beginPath();
        ctx.moveTo(toX(this.x[0]), toY(0));
        for (let i = 0; i < this.x.length; i++) {
            const v = (typeof this.y[i] === 'number' && isFinite(this.y[i])) ? this.y[i] : 0;
            ctx.lineTo(toX(this.x[i]), toY(v));
        }
        ctx.lineTo(toX(this.x[this.x.length - 1]), toY(0));
        ctx.closePath();
        const gradient = ctx.createLinearGradient(0, m.top, 0, m.top + ph);
        gradient.addColorStop(0, 'rgba(75, 192, 192, 0.45)');
        gradient.addColorStop(1, 'rgba(75, 192, 192, 0.05)');
        ctx.fillStyle = gradient;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.lineJoin = 'round';
        ctx.strokeStyle = teal;
        ctx.stroke();
        ctx.restore();

        // Fermi level (E - E_F = 0)
        if (xMin <= 0 && xMax >= 0) {
            const fx = toX(0);
            ctx.save();
            ctx.strokeStyle = '#e74c3c';
            ctx.lineWidth = 1.2;
            ctx.setLineDash([5, 4]);
            ctx.beginPath(); ctx.moveTo(fx, m.top); ctx.lineTo(fx, m.top + ph); ctx.stroke();
            ctx.restore();
            ctx.fillStyle = '#e74c3c';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('E_F', fx + 4, m.top + 2);
        }

        // Axis titles
        ctx.fillStyle = '#333';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText('E - E_F (eV)', m.left + pw / 2, L.height - 2);
        ctx.save();
        ctx.translate(12, m.top + ph / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textBaseline = 'top';
        ctx.fillText(this.yLabel, 0, 0);
        ctx.restore();

        this._drawHover(L);
    }

    _drawHover(L) {
        const i = this.hoverIndex;
        if (i < 0 || i >= this.x.length) return;
        const ctx = this.ctx;
        const { m, ph, pw, toX, toY } = L;
        const px = toX(this.x[i]);
        const val = this.y[i];
        ctx.save();
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.35)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(px, m.top); ctx.lineTo(px, m.top + ph); ctx.stroke();
        if (typeof val === 'number' && isFinite(val)) {
            ctx.fillStyle = 'rgb(75, 192, 192)';
            ctx.beginPath(); ctx.arc(px, toY(val), 3.5, 0, Math.PI * 2); ctx.fill();
        }
        const text = `E−E_F = ${this.x[i].toFixed(2)} eV, DOS = ${
            (typeof val === 'number' && isFinite(val)) ? val.toPrecision(4) : 'n/a'}`;
        ctx.font = '12px sans-serif';
        const tw = ctx.measureText(text).width + 12;
        let bx = px + 10;
        if (bx + tw > m.left + pw) bx = px - 10 - tw;
        const by = m.top + 18;
        ctx.fillStyle = 'rgba(44, 62, 80, 0.92)';
        ctx.fillRect(bx, by, tw, 22);
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, bx + 6, by + 11);
        ctx.restore();
    }

    _handleMove(e) {
        const L = this._layout();
        if (!L) return;
        const rect = this.canvas.getBoundingClientRect();
        const px = e.clientX - rect.left;
        if (px < L.m.left || px > L.m.left + L.pw) {
            if (this.hoverIndex !== -1) { this.hoverIndex = -1; this.draw(); }
            return;
        }
        const target = L.xMin + ((px - L.m.left) / (L.pw || 1)) * (L.xMax - L.xMin);
        // Grid is uniform; binary search keeps this correct regardless.
        let lo = 0, hi = this.x.length - 1;
        while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (this.x[mid] < target) lo = mid; else hi = mid;
        }
        const idx = Math.abs(this.x[lo] - target) <= Math.abs(this.x[hi] - target) ? lo : hi;
        if (idx !== this.hoverIndex) {
            this.hoverIndex = idx;
            this.draw();
        }
    }

    destroy() {
        window.removeEventListener('resize', this._onResize);
        this.canvas.removeEventListener('mousemove', this._onMove);
        this.canvas.removeEventListener('mouseleave', this._onLeave);
        this.x = [];
        this.y = [];
        this.ctx = null;
    }
}
