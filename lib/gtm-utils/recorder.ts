import EE from 'eventemitter3';
import { CANVAS_BG, CANVAS_SIZE, renderFramesToCanvas } from './spectrogram';
import { normalizeFloat32Array } from '@base/speech-commands/browser_fft_utils';

export interface SpectrogramData {
    /**
     * The float32 data for the spectrogram.
     *
     * Stored frame by frame. For example, the first N elements
     * belong to the first time frame and the next N elements belong
     * to the second time frame, and so forth.
     */
    data: Float32Array;

    /**
     * Number of points per frame, i.e., FFT length per frame.
     */
    frameSize: number;

    /**
     * Duration of each frame in milliseconds.
     */
    frameDurationMillis?: number;

    /**
     * Index to the key frame (0-based).
     *
     * A key frame is a frame in the spectrogram that belongs to
     * the utterance of interest. It is used to distinguish the
     * utterance part from the background-noise part.
     *
     * A typical use of key frame index: when multiple training examples are
     * extracted from a spectroram, every example is guaranteed to include
     * the key frame.
     *
     * Key frame is not required. If it is missing, heuristics algorithms
     * (e.g., finding the highest-intensity frame) can be used to calculate
     * the key frame.
     */
    keyFrameIndex?: number;
}

export interface RawAudioData {
    /** Samples of the snippet. */
    data: Float32Array;

    /** Sampling rate, in Hz. */
    sampleRateHz: number;
}

export interface AudioExample {
    /** A label for the example. */
    label: string;

    /** Spectrogram data. */
    spectrogram: SpectrogramData;

    /**
     * A 224x224 heatmap image of this 1-second spectrogram example.
     */
    spectrogramCanvas?: HTMLCanvasElement;

    /**
     * Raw audio in PCM (pulse-code modulation) format.
     *
     * Optional.
     */
    rawAudio?: RawAudioData;
}

export interface RecordingProgress {
    progress: number;
    elapsedTimeMillis: number;
}

type SoundRecorderEvents = 'example' | 'update' | 'start' | 'stop' | 'error';

export interface RecordExampleOptions {
    /** The size of each frame in the spectrogram, i.e., the FFT length. */
    frameSize: number;

    /**
     * Whether to include raw audio data in the example.
     *
     * Default: false.
     */
    includeRawAudio?: boolean;

    durationMillis?: number;

    /** Sampling rate of the raw audio data, in Hz. */
    sampleRateHz: number;

    /**
     * Optional initial warmup period (ms).
     * During this period, no examples are emitted, but visualization continues.
     *
     * Default: 0.
     */
    warmupMillis?: number;

    columnTruncateLength?: number;

    overlapFactor?: number;

    includeCanvas?: boolean;
}

function flattenFrames(frames: Float32Array[], frameSize: number): Float32Array {
    const out = new Float32Array(frames.length * frameSize);
    for (let i = 0; i < frames.length; i++) {
        out.set(frames[i], i * frameSize);
    }
    return out;
}

function computeKeyFrameIndex(frames: Float32Array[]): number {
    if (frames.length === 0) return 0;
    let bestIdx = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < frames.length; i++) {
        const f = frames[i];
        let sum = 0;
        for (let j = 0; j < f.length; j++) sum += f[j];
        const avg = sum / f.length;
        if (avg > bestScore) {
            bestScore = avg;
            bestIdx = i;
        }
    }
    return bestIdx;
}

function isPowerOfTwo(n: number): boolean {
    return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

export default class SoundRecorder extends EE<SoundRecorderEvents> {
    public canvas: HTMLCanvasElement | null = null;

    private stream: MediaStream | null = null;
    private audioContext: AudioContext | null = null;
    private analyser: AnalyserNode | null = null;
    private source: MediaStreamAudioSourceNode | null = null;
    private frameIntervalTask: number | null = null;

    private isRecording = false;
    private stopRequested = false;

    private allFreqFrames: Float32Array[] = [];
    private lastRenderTime = 0;
    private lastMainDrawWidth = 0;

    private currentFrameDurationMillis = 0;
    private currentSampleRateHz = 0;
    private lastEmitTime = 0;
    private startTime = 0;

    private finish(err?: unknown) {
        if (!this.isRecording) return;

        if (this.frameIntervalTask != null) {
            clearInterval(this.frameIntervalTask);
            this.frameIntervalTask = null;
        }

        try {
            this.analyser?.disconnect();
            this.analyser = null;
            this.source?.disconnect();
            this.source = null;
        } catch {
            // Ignore any error during disconnection/cleanup.
        }

        if (this.stream) {
            for (const track of this.stream.getTracks()) {
                track.stop();
            }
        }
        this.stream = null;

        if (this.audioContext) {
            void this.audioContext.close();
        }
        this.audioContext = null;

        this.renderSpectrogram(true);

        this.isRecording = false;
        this.stopRequested = false;
        this.emit('stop');

        if (err != null) {
            this.emit('error', err);
        }
    }

    private emitOneSecondExample(
        freqData: Float32Array[],
        frameDurationMillis: number,
        word: string,
        options: RecordExampleOptions
    ) {
        //const freqFrames = currentExampleFreq.splice(0, framesPerExample);
        const specData = flattenFrames(freqData, options.columnTruncateLength ?? options.frameSize);

        let exampleCanvas: HTMLCanvasElement | undefined;
        if (options.includeCanvas) {
            exampleCanvas = document.createElement('canvas');
            exampleCanvas.width = CANVAS_SIZE;
            exampleCanvas.height = CANVAS_SIZE;

            // Take the image from the main canvas
            if (this.canvas) {
                const mainCtx = this.canvas.getContext('2d');
                const exampleCtx = exampleCanvas.getContext('2d');
                if (mainCtx && exampleCtx) {
                    const sx = this.canvas.width - CANVAS_SIZE;
                    const sy = 0;
                    const sWidth = CANVAS_SIZE;
                    const sHeight = CANVAS_SIZE;
                    const dx = 0;
                    const dy = 0;
                    const dWidth = CANVAS_SIZE;
                    const dHeight = CANVAS_SIZE;
                    exampleCtx.drawImage(this.canvas, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight);
                }
            } else {
                // Fill with white if main canvas is not available for some reason.
                const exampleCtx = exampleCanvas.getContext('2d');
                if (exampleCtx) {
                    exampleCtx.fillStyle = CANVAS_BG;
                    exampleCtx.fillRect(0, 0, exampleCanvas.width, exampleCanvas.height);
                }
            }
        }

        const example: AudioExample = {
            label: word,
            spectrogram: {
                data: normalizeFloat32Array(specData),
                frameSize: options.columnTruncateLength ?? options.frameSize,
                frameDurationMillis,
                keyFrameIndex: computeKeyFrameIndex(freqData),
            },
            spectrogramCanvas: exampleCanvas,
        };

        /*if (includeRawAudio) {
                        const timeFrames = currentExampleTime.splice(0, framesPerExample);
                        example.rawAudio = {
                            data: flattenFrames(timeFrames, options.frameSize),
                            sampleRateHz: actualSampleRateHz,
                        };
                    }*/

        this.emit('example', example);
    }

    async startRecording(word: string, options: RecordExampleOptions): Promise<void> {
        if (this.isRecording) {
            throw new Error('SoundRecorder is already recording.');
        }
        if (!options || !options.frameSize || options.frameSize <= 0) {
            throw new Error(`Invalid frameSize: ${options?.frameSize}`);
        }
        if (!isPowerOfTwo(options.frameSize)) {
            throw new Error(`frameSize must be a power of 2, but got ${options.frameSize}`);
        }
        if (options.durationMillis !== undefined && !(options.durationMillis > 0)) {
            throw new Error(`Invalid durationMillis: ${options.durationMillis}`);
        }
        if (!(options.sampleRateHz > 0)) {
            throw new Error(`Invalid sampleRateHz: ${options.sampleRateHz}`);
        }
        if (options.warmupMillis != null && options.warmupMillis < 0) {
            throw new Error(`Invalid warmupMillis: ${options.warmupMillis}`);
        }

        //const includeRawAudio = !!options.includeRawAudio;
        const warmupMillis = Math.max(0, options.warmupMillis ?? 0);

        this.isRecording = true;
        this.stopRequested = false;
        this.allFreqFrames = [];
        this.lastRenderTime = 0;
        this.lastMainDrawWidth = 0;

        if (this.canvas) {
            // Pre-fill white once; incremental renderer only draws changed region.
            const mainCtx = this.canvas.getContext('2d');
            if (mainCtx) {
                mainCtx.fillStyle = CANVAS_BG;
                mainCtx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            }
        }

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: options.sampleRateHz,
                },
                video: false,
            });
            this.audioContext = new AudioContext({ sampleRate: options.sampleRateHz });
            this.source = this.audioContext.createMediaStreamSource(this.stream);
            this.analyser = this.audioContext.createAnalyser();

            this.analyser.fftSize = options.frameSize * 2;
            this.analyser.smoothingTimeConstant = 0;
            this.source.connect(this.analyser);

            const actualSampleRateHz = this.audioContext.sampleRate;
            this.currentSampleRateHz = actualSampleRateHz;
            const frameDurationMillis = (options.frameSize / actualSampleRateHz) * 1e3;
            this.currentFrameDurationMillis = frameDurationMillis;

            const periodMs = Math.max(1, Math.round(frameDurationMillis));
            const emitPeriodMs = 1000 * (1 - (options.overlapFactor ?? 0.5));
            const framesPerExample = Math.max(1, Math.round(1000 / frameDurationMillis));
            const now = performance.now();
            const endTime = now + (options.durationMillis ?? 60 * 60 * 1000);
            const warmupEndTime = now + warmupMillis;
            this.startTime = now;

            const freqScratch = new Float32Array(options.frameSize);
            //const timeScratch = new Float32Array(this.analyser.fftSize);

            const currentExampleFreq: Float32Array[] = [];
            //const currentExampleTime: Float32Array[] = [];

            this.emit('start');

            this.frameIntervalTask = window.setInterval(() => {
                try {
                    if (this.stopRequested || performance.now() >= endTime) {
                        this.finish();
                        return;
                    }
                    if (!this.analyser) return;

                    this.analyser.getFloatFrequencyData(freqScratch);
                    if (!Number.isFinite(freqScratch[0])) return;

                    const freqFrame = freqScratch.slice(0, options.columnTruncateLength ?? options.frameSize);
                    this.allFreqFrames.push(freqFrame);

                    const now = performance.now();
                    const inWarmup = now < warmupEndTime;
                    const progress: RecordingProgress = {
                        progress:
                            options.durationMillis === undefined
                                ? 1
                                : Math.min(1, (now - (warmupEndTime - warmupMillis)) / options.durationMillis),
                        elapsedTimeMillis: now - this.startTime,
                    };

                    const shouldEmit =
                        !inWarmup &&
                        currentExampleFreq.length === framesPerExample &&
                        now - this.lastEmitTime >= emitPeriodMs;

                    // Throttle canvas updates to reduce overhead.
                    if (shouldEmit || now - this.lastRenderTime >= 100) {
                        this.renderSpectrogram(false);
                        this.lastRenderTime = now;
                        this.emit('update', progress);
                    }

                    if (inWarmup) {
                        // Keep visualization flowing, but discard example accumulation.
                        currentExampleFreq.length = 0;
                        /*if (includeRawAudio) {
                            currentExampleTime.length = 0;
                        }*/
                    } else {
                        currentExampleFreq.push(freqFrame);

                        /*if (includeRawAudio) {
                            this.analyser.getFloatTimeDomainData(timeScratch);
                            currentExampleTime.push(timeScratch.slice(timeScratch.length - options.frameSize));
                        }*/

                        // Keep it at a constant length
                        if (currentExampleFreq.length > framesPerExample) {
                            currentExampleFreq.shift();
                        }

                        if (shouldEmit) {
                            this.lastEmitTime = now;
                            this.emitOneSecondExample(currentExampleFreq, frameDurationMillis, word, options);
                        }
                    }
                } catch (err) {
                    this.finish(err);
                }
            }, periodMs);
        } catch (err) {
            this.isRecording = false;
            this.stopRequested = false;
            this.currentSampleRateHz = 0;
            throw err;
        }
    }

    stopRecording() {
        if (!this.isRecording) return;
        this.stopRequested = true;
    }

    private renderSpectrogram(force = false): void {
        if (!this.canvas) return;
        if (this.canvas.width <= 0) this.canvas.width = CANVAS_SIZE;
        if (this.canvas.height <= 0) this.canvas.height = CANVAS_SIZE;

        const ctx = this.canvas.getContext('2d');
        if (!ctx) return;

        if (this.allFreqFrames.length === 0) {
            ctx.fillStyle = CANVAS_BG;
            ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            this.lastMainDrawWidth = 0;
            return;
        }

        // Incremental update: render only newly exposed x-range.
        this.lastMainDrawWidth = renderFramesToCanvas(
            this.allFreqFrames,
            this.canvas,
            this.currentFrameDurationMillis,
            this.currentSampleRateHz,
            this.lastMainDrawWidth,
            false
        );

        if (force) {
            this.emit('update', {
                progress: 1,
                elapsedTimeMillis: performance.now() - this.startTime,
            });
        }
    }
}
