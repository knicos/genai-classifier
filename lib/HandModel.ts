import { TrainingParameters as HandTrainingParams } from './gtm-hand/teachable-handpose';
import {
    TeachableHandPose,
    type Metadata as HandMetadata,
    createTeachable as createHand,
    drawHandKeypoints,
    drawHandSkeleton,
} from './gtm-hand';
import * as tf from '@tensorflow/tfjs';
import type { TeachableModel, ExplainedPredictionsOutput, TMType } from './TeachableModel';
import { AudioExample } from './gtm-utils/recorder';

const NULLARRAY: string[] = [];

interface TrainingParameters extends HandTrainingParams {}

interface BaseMetadata {
    modelBaseUrl?: string;
}

export type Metadata = BaseMetadata & HandMetadata;

type HandLike = { keypoints: { x: number; y: number; score?: number; name?: string }[] };

export default class HandModel implements TeachableModel {
    protected model?: TeachableHandPose;
    protected _ready?: Promise<boolean>;
    protected trained = false;
    protected busy = false;
    protected imageSize = 224;
    protected _disposed = false;
    public variant: TMType = 'hand';
    public explained?: HTMLCanvasElement;
    modelBaseUrl = 'https://tmstore.blob.core.windows.net/models';
    private lastHands: HandLike[] = [];

    constructor(type: TMType, metadata?: Metadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer) {
        if (type !== 'hand') {
            throw new Error(`Invalid type for HandModel: ${type}`);
        }

        this.variant = type;

        if (metadata?.modelBaseUrl) {
            this.modelBaseUrl = metadata.modelBaseUrl;
        }

        this._ready = this.load(metadata, model, weights).then(() => {
            return true;
        });
    }

    public getVariant(): TMType {
        return this.variant;
    }

    public setXAICanvas(canvas: HTMLCanvasElement) {
        if (this.model) {
            this.explained = canvas;
            return;
        }
        throw new Error('no_model');
    }

    public setXAIClass() {}

    protected async load(metadata?: HandMetadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer) {
        await tf.ready();

        if (metadata && model && weights) {
            this.trained = true;
            const tmmodel = await createHand(metadata);
            tmmodel.model = await tf.loadLayersModel({
                load: async () => ({
                    modelTopology: model.modelTopology,
                    weightData: weights,
                    weightSpecs: model.weightsManifest[0].weights,
                }),
            });
            this.model = tmmodel;
        } else {
            const tmmodel = await createHand({ tfjsVersion: tf.version.tfjs });
            this.model = tmmodel;
            tmmodel.setName('My Model');
        }

        this.imageSize = this.model.getMetadata()?.imageSize || 257;
    }

    public async ready() {
        return this.isReady() || this._ready || false;
    }

    public draw(image: HTMLCanvasElement) {
        const ctx = image.getContext('2d');
        if (!ctx) return image;

        if (this.model && this.lastHands.length > 0) {
            try {
                for (const hand of this.lastHands) {
                    drawHandKeypoints(hand.keypoints, ctx, 0.5);
                    drawHandSkeleton(hand.keypoints, ctx);
                }
            } catch (error) {
                console.error(error);
            }
        }

        return image;
    }

    public async estimate(image: HTMLCanvasElement): Promise<HTMLCanvasElement> {
        if (this.model && !this.busy) {
            this.busy = true;
            try {
                const { allHands } = await this.model.estimateHand(image);
                this.lastHands = allHands;
            } catch (error) {
                console.error('Estimation error', error);
                this.lastHands = [];
            }
            this.busy = false;
        }
        return image;
    }

    public async predict(image: HTMLCanvasElement): Promise<ExplainedPredictionsOutput> {
        if (!this.trained || this._disposed) return { predictions: [] };
        if (!this.model || this.busy) return { predictions: [] };
        this.busy = true;

        let allHands: HandLike[];
        let jointHandOutput: Float32Array;

        try {
            const result = await this.model.estimateHand(image);
            allHands = result.allHands;
            jointHandOutput = result.jointHandOutput;
            this.lastHands = allHands;
        } catch {
            this.lastHands = [];
            this.busy = false;
            return { predictions: [] };
        }
        this.busy = false;

        if (this._disposed || !this.model) return { predictions: [] };

        try {
            const predictions = await this.model.predict(jointHandOutput);
            return { predictions };
        } catch {
            return { predictions: [] };
        }
    }

    public async train(params: TrainingParameters, callbacks: tf.CustomCallbackArgs) {
        this.trained = false;
        if (this.model) {
            this.busy = true;
            return this.model.train(params, callbacks).then((m) => {
                this.trained = true;
                this.busy = false;
                return m;
            });
        }
        throw new Error('no_model');
    }

    public async addExample(className: number, image: HTMLCanvasElement | AudioExample) {
        if (!this.model) {
            throw new Error('no_model');
        }

        if (!(image instanceof HTMLCanvasElement)) {
            throw new Error('invalid_sample_type');
        }

        let result = await this.model.estimateHand(image, false);

        if (!result.allHands.length) {
            result = await this.model.estimateHand(image, true);
        }

        this.model.addExample(className, result.jointHandOutput);
    }

    public dispose() {
        this._disposed = true;

        if (this.model) {
            try {
                this.model.dispose();
            } catch (error) {
                console.warn('Error disposing hand model:', error);
            }
        }

        this.model = undefined;
        this.lastHands = [];
    }

    public setName(name: string): void {
        if (this.model) {
            this.model.setName(name);
        }
    }

    public getModel(): TeachableHandPose | undefined {
        return this.model;
    }

    public getImageSize() {
        return this.imageSize;
    }

    public isTrained() {
        return this.trained;
    }

    public isReady() {
        return !!this.model;
    }

    public setSeed(seed: string) {
        if (this.model) {
            this.model.setSeed(seed);
        }
    }

    public getMetadata() {
        if (this.model) {
            return this.model.getMetadata();
        }
    }

    public async save(handler: tf.io.IOHandler) {
        if (this.model) {
            return this.model.save(handler);
        }
        throw new Error('no_model');
    }

    public setLabels(labels: string[]) {
        if (this.model) {
            this.model.setLabels(labels);
        }
    }

    public getLabels(): string[] {
        if (this.model) {
            return this.model.getLabels();
        }
        return NULLARRAY;
    }

    public getLabel(ix: number): string {
        if (this.model) {
            return this.model.getLabel(ix);
        }
        return '';
    }

    public getNumExamples(): number {
        if (this.model) {
            return this.model.examples.reduce((t, e) => t + e.length, 0);
        }
        return 0;
    }

    public getExamplesPerClass(): number[] {
        if (this.model) {
            return this.model.examples.map((e) => e.length);
        }
        return [];
    }

    public getNumValidation(): number {
        if (this.model) {
            return this.model.examples.reduce((t, e) => t + Math.ceil(e.length * 0.15), 0);
        }
        return 0;
    }

    public calculateAccuracy() {
        if (this.model) {
            return this.model.calculateAccuracyPerClass();
        }
        throw new Error('no_model');
    }
}
