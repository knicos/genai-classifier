export const CANVAS_SIZE = 58;
export const CANVAS_BG = `rgb(${viridisColor(0).join(',')})`;
const FIXED_MIN_DB = -100;
const FIXED_MAX_DB = -30;

// Windowing/interpolation controls.
const WINDOW_BLOCK_X = 2;
const WINDOW_BLOCK_Y = 2;
const ENABLE_WINDOWED_RENDER = true;

// Visual frequency-range controls (rendering only).
const ENABLE_VOCAL_BAND_LIMIT = true;
const VOCAL_MIN_HZ = 80;
const VOCAL_MAX_HZ = 4000;

function clamp01(x: number): number {
    return Math.max(0, Math.min(1, x));
}

function viridisColor(t: number): [number, number, number] {
    const x = clamp01(t);

    const r =
        0.280268003 +
        x * (-0.143510503 + x * (2.225793877 + x * (-14.815088879 + x * (25.212752309 + x * -11.772589584))));
    const g =
        -0.002117546 + x * (1.617109353 + x * (-1.90930507 + x * (2.701152864 + x * (-1.685288385 + x * 0.178738871))));
    const b =
        0.300805501 +
        x * (2.614650302 + x * (-12.01913909 + x * (28.93355911 + x * (-33.49129477 + x * 13.762053843))));

    return [Math.round(255 * clamp01(r)), Math.round(255 * clamp01(g)), Math.round(255 * clamp01(b))];
}

function bilinearDb(frames: Float32Array[], framePos: number, binPos: number): number {
    const f0 = Math.floor(framePos);
    const f1 = Math.min(frames.length - 1, f0 + 1);
    const tf = framePos - f0;

    const frameSize = frames[0].length;
    const b0 = Math.floor(binPos);
    const b1 = Math.min(frameSize - 1, b0 + 1);
    const tb = binPos - b0;

    const v00 = frames[f0][b0];
    const v01 = frames[f0][b1];
    const v10 = frames[f1][b0];
    const v11 = frames[f1][b1];

    const v0 = v00 * (1 - tb) + v01 * tb;
    const v1 = v10 * (1 - tb) + v11 * tb;
    return v0 * (1 - tf) + v1 * tf;
}

function getVisualBinRange(frameSize: number, sampleRateHz: number): { minBin: number; maxBin: number } {
    if (!ENABLE_VOCAL_BAND_LIMIT) {
        return { minBin: 0, maxBin: frameSize - 1 };
    }

    const sr = sampleRateHz;
    const nyquist = sr > 0 ? sr / 2 : 0;
    if (!(nyquist > 0)) {
        return { minBin: 0, maxBin: frameSize - 1 };
    }

    const minHz = Math.max(0, Math.min(VOCAL_MIN_HZ, nyquist));
    const maxHz = Math.max(minHz, Math.min(VOCAL_MAX_HZ, nyquist));

    const minBin = Math.round((minHz / nyquist) * (frameSize - 1));
    const maxBin = Math.round((maxHz / nyquist) * (frameSize - 1));

    if (maxBin <= minBin) {
        return { minBin: 0, maxBin: frameSize - 1 };
    }
    return { minBin, maxBin };
}

export function renderFramesToCanvas(
    frames: Float32Array[],
    canvas: HTMLCanvasElement,
    frameDurationMillis: number,
    sampleRateHz: number,
    fromX = 0,
    clear = false
): number {
    const ctx = canvas.getContext('2d');
    if (!ctx) return 0;

    const width = canvas.width;
    const height = canvas.height;

    if (clear) {
        ctx.fillStyle = CANVAS_BG;
        ctx.fillRect(0, 0, width, height);
    }

    if (frames.length === 0) return 0;

    const msPerFrame = frameDurationMillis;
    const audioDurationSec = msPerFrame > 0 ? (frames.length * msPerFrame) / 1000 : 0;

    // Total timeline width at CANVAS_SIZE px/sec (not clamped to canvas width).
    const totalWidth = Math.max(1, Math.round(audioDurationSec * CANVAS_SIZE));
    const prevTotalWidth = Math.max(0, fromX);

    const frameCount = frames.length;
    const frameSize = frames[0].length;
    const { minBin, maxBin } = getVisualBinRange(frameSize, sampleRateHz);
    const binSpan = Math.max(1e-6, maxBin - minBin);
    const denom = Math.max(1e-6, FIXED_MAX_DB - FIXED_MIN_DB);

    // Viewport in global timeline coords.
    // Negative means the content is right-aligned with blank space on the left.
    const viewportStart = totalWidth - width;
    const prevViewportStart = prevTotalWidth - width;

    // Shift existing pixels if viewport moved right (new audio arrived).
    if (!clear && viewportStart > prevViewportStart) {
        const shift = Math.min(width, viewportStart - prevViewportStart);
        if (shift > 0) {
            ctx.drawImage(canvas, shift, 0, width - shift, height, 0, 0, width - shift, height);
            ctx.fillStyle = CANVAS_BG;
            ctx.fillRect(width - shift, 0, shift, height);
        }
    }

    // Render only newly available global timeline region.
    const visibleGlobalStart = Math.max(0, viewportStart);
    const renderGlobalStart = Math.max(prevTotalWidth, visibleGlobalStart);
    const renderGlobalEnd = totalWidth;
    if (renderGlobalEnd <= renderGlobalStart) {
        return totalWidth;
    }

    const updateWidth = renderGlobalEnd - renderGlobalStart;
    const destStartX = renderGlobalStart - viewportStart; // supports right-aligned startup

    const imageData = ctx.createImageData(updateWidth, height);
    const data = imageData.data;

    // Precompute coarse window grid for interpolation over global X.
    let grid: number[][] | null = null;
    let gridW = 0;
    let gridH = 0;
    let gridX0 = 0;
    if (ENABLE_WINDOWED_RENDER) {
        gridX0 = Math.floor(renderGlobalStart / WINDOW_BLOCK_X);
        const gridX1 = Math.ceil((renderGlobalEnd - 1) / WINDOW_BLOCK_X) + 1;

        gridW = gridX1 - gridX0 + 1;
        gridH = Math.ceil(height / WINDOW_BLOCK_Y) + 1;
        grid = Array.from({ length: gridH }, () => new Array<number>(gridW));

        for (let gy = 0; gy < gridH; gy++) {
            const yAnchor = Math.min(height - 1, gy * WINDOW_BLOCK_Y);

            for (let gx = 0; gx < gridW; gx++) {
                const gxGlobal = gridX0 + gx;
                const xAnchorGlobal = Math.min(totalWidth - 1, Math.max(0, gxGlobal * WINDOW_BLOCK_X));

                const tSec = xAnchorGlobal / CANVAS_SIZE;
                const framePos = Math.max(0, Math.min(frameCount - 1, (tSec * 1000) / Math.max(1e-6, msPerFrame)));
                const yNorm = height === 1 ? 0 : (height - 1 - yAnchor) / (height - 1);
                const binPos = minBin + yNorm * binSpan;

                grid[gy][gx] = bilinearDb(frames, framePos, binPos);
            }
        }
    }

    for (let localX = 0; localX < updateWidth; localX++) {
        const xGlobal = renderGlobalStart + localX;

        for (let y = 0; y < height; y++) {
            let db: number;

            if (grid) {
                const gx = xGlobal / WINDOW_BLOCK_X - gridX0;
                const gy = y / WINDOW_BLOCK_Y;

                const x0 = Math.floor(gx);
                const y0 = Math.floor(gy);
                const x1 = Math.min(gridW - 1, x0 + 1);
                const y1 = Math.min(gridH - 1, y0 + 1);

                const tx = gx - x0;
                const ty = gy - y0;

                const v00 = grid[y0][x0];
                const v01 = grid[y0][x1];
                const v10 = grid[y1][x0];
                const v11 = grid[y1][x1];

                const v0 = v00 * (1 - tx) + v01 * tx;
                const v1 = v10 * (1 - tx) + v11 * tx;
                db = v0 * (1 - ty) + v1 * ty;
            } else {
                const tSec = xGlobal / CANVAS_SIZE;
                const framePos = Math.max(0, Math.min(frameCount - 1, (tSec * 1000) / Math.max(1e-6, msPerFrame)));
                const yNorm = height === 1 ? 0 : (height - 1 - y) / (height - 1);
                const binPos = minBin + yNorm * binSpan;
                db = bilinearDb(frames, framePos, binPos);
            }

            const t = clamp01((db - FIXED_MIN_DB) / denom);
            const [r, g, b] = viridisColor(t);

            const idx = (y * updateWidth + localX) * 4;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = 255;
        }
    }

    ctx.putImageData(imageData, destStartX, 0);
    return totalWidth;
}
