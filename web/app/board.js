// board.js — canvas Go board: stones, last-move marker, hover ghost,
// click → move index (y*b+x). No game logic lives here.
export class Board {
    constructor(canvas, onClick) {
        this.cv = canvas;
        this.cx = canvas.getContext("2d");
        canvas.addEventListener("click", (e) => {
            if (!this.b || !this.onClick)
                return;
            const r = canvas.getBoundingClientRect();
            const x = (e.clientX - r.left) * (canvas.width / r.width);
            const y = (e.clientY - r.top) * (canvas.height / r.height);
            const gx = Math.round((x - this.pad) / this.cell);
            const gy = Math.round((y - this.pad) / this.cell);
            if (gx >= 0 && gx < this.b && gy >= 0 && gy < this.b)
                this.onClick((this.b - 1 - gy) * this.b + gx);
        });
        canvas.addEventListener("mousemove", (e) => {
            if (!this.b)
                return;
            const r = canvas.getBoundingClientRect();
            const x = (e.clientX - r.left) * (canvas.width / r.width);
            const y = (e.clientY - r.top) * (canvas.height / r.height);
            const gx = Math.round((x - this.pad) / this.cell);
            const gy = Math.round((y - this.pad) / this.cell);
            this.hover = gx >= 0 && gx < this.b && gy >= 0 && gy < this.b
                ? { x: gx, y: gy } : null;
            this.draw();
        });
        canvas.addEventListener("mouseleave", () => {
            this.hover = null;
            this.draw();
        });
        this.onClick = onClick;
        this.b = 0;
    }

    set(b, stones, lastMove, ghostColor) {
        this.b = b;
        this.stones = stones;          // Uint8Array b*b: 0 empty 1 black 2 white
        this.last = lastMove;
        this.ghostColor = ghostColor;  // hover stone color: 0 none, 1 black, 2 white
        const S = this.cv.width;
        this.pad = S / (b + 1);
        this.cell = (S - 2 * this.pad) / (b - 1);
        this.draw();
    }

    draw() {
        const { cx, b } = this;
        if (!b)
            return;
        const S = this.cv.width;
        cx.fillStyle = "#dcB35c";
        cx.fillStyle = "#dcb35c";
        cx.fillRect(0, 0, S, S);

        // grid
        cx.strokeStyle = "#3a2f1b";
        cx.lineWidth = 1;
        for (let i = 0; i < b; i++) {
            const p = this.pad + i * this.cell;
            cx.beginPath();
            cx.moveTo(this.pad, p);
            cx.lineTo(S - this.pad, p);
            cx.moveTo(p, this.pad);
            cx.lineTo(p, S - this.pad);
            cx.stroke();
        }
        // star points
        const stars = b >= 13 ? [3, Math.floor(b / 2), b - 4] :
                      b >= 9 ? [2, 4, 6].filter((v) => v < b) :
                      b >= 7 ? [2, 4].filter((v) => v < b) : [];
        cx.fillStyle = "#3a2f1b";
        for (const sy of stars)
            for (const sx of stars) {
                cx.beginPath();
                cx.arc(this.pad + sx * this.cell, this.pad + sy * this.cell,
                       Math.max(2, this.cell * 0.06), 0, 7);
                cx.fill();
            }

        // hover ghost — the human player's color, half transparent
        if (this.hover && this.ghostColor) {
            const { x, y } = this.hover;
            if (this.stones[(b - 1 - y) * b + x] === 0) {
                cx.globalAlpha = 0.5;
                this.stone(x, y, this.ghostColor);
                cx.globalAlpha = 1;
            }
        }

        // stones — index row 0 is GTP row 1 = the BOTTOM row: render it
        // at the bottom of the canvas (canvas gy runs top-down)
        for (let iy = 0; iy < b; iy++)
            for (let ix = 0; ix < b; ix++) {
                const s = this.stones[iy * b + ix];
                if (s)
                    this.stone(ix, b - 1 - iy, s);
            }

        // last-move marker
        if (this.last != null && this.last < b * b) {
            const x = this.last % b, y = (this.last / b) | 0;
            const s = this.stones[y * b + x];
            if (s) {
                cx.strokeStyle = s === 1 ? "#fff" : "#000";
                cx.lineWidth = 2;
                cx.beginPath();
                cx.arc(this.pad + x * this.cell,
                       this.pad + (b - 1 - y) * this.cell,
                       this.cell * 0.22, 0, 7);
                cx.stroke();
            }
        }
    }

    stone(x, y, color) {
        const cx = this.cx, px = this.pad + x * this.cell,
              py = this.pad + y * this.cell, r = this.cell * 0.44;
        const g = cx.createRadialGradient(px - r / 3, py - r / 3, r / 8,
                                          px, py, r);
        if (color === 1) {
            g.addColorStop(0, "#666");
            g.addColorStop(1, "#000");
        } else {
            g.addColorStop(0, "#fff");
            g.addColorStop(1, "#ccc");
        }
        cx.fillStyle = g;
        cx.beginPath();
        cx.arc(px, py, r, 0, 7);
        cx.fill();
        cx.strokeStyle = "rgba(0,0,0,.4)";
        cx.lineWidth = 0.5;
        cx.stroke();
    }
}
