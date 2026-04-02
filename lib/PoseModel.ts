import { TrainingParameters as PoseTrainingParams } from './gtm-pose/teachable-posenet';
import {
    TeachablePoseNet,
    Metadata as PoseMetadata,
    drawKeypoints,
    drawSkeleton,
    createTeachable as createPose,
} from './gtm-pose';
import * as tf from '@tensorflow/tfjs';
import { renderHeatmap, renderPoseXAI } from './heatmap';
import { CAM } from './xai';
import type { TeachableModel, ExplainedPredictionsOutput, TMType } from './TeachableModel';

const NULLARRAY: string[] = [];

type Vector2D = {
    y: number;
    x: number;
};

type Keypoint = {
    score: number;
    position: Vector2D;
    part: string;
};

type Pose = {
    keypoints: Keypoint[];
    score: number;
};

interface TrainingParameters extends PoseTrainingParams {}

interface BaseMetadata {
    modelBaseUrl?: string;
}

export type Metadata = BaseMetadata & PoseMetadata;

export default class PoseModel implements TeachableModel {
    protected model?: TeachablePoseNet;
    protected _ready?: Promise<boolean>;
    protected trained = false;
    protected busy = false;
    protected imageSize = 224;
    protected _disposed = false;
    public variant: TMType = 'pose';
    public explained?: HTMLCanvasElement;
    modelBaseUrl = 'https://tmstore.blob.core.windows.net/models';
    private lastPose?: Pose;
    private CAMModel?: CAM;

    constructor(type: TMType, metadata?: Metadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer) {
        if (type !== 'pose') {
            throw new Error(`Invalid type for PoseModel: ${type}`);
        }

        this.variant = type;

        if (metadata?.modelBaseUrl) {
            this.modelBaseUrl = metadata.modelBaseUrl;
        }

        this._ready = this.load(metadata, model, weights).then(() => {
            return true;
        });
    }

    getVariant(): TMType {
        return this.variant;
    }

    public setXAICanvas(canvas: HTMLCanvasElement) {
        if (this.model) {
            this.explained = canvas;
            if (!this.CAMModel) {
                this.CAMModel = new CAM(this.model);
            }
            return;
        }
        throw new Error('no_model');
    }

    public setXAIClass(className: string | number | null) {
        if (this.CAMModel) {
            if (className === null) {
                this.CAMModel.setSelectedIndex(null);
                return;
            }
            const ix = typeof className === 'number' ? className : (this.model?.getLabels() || []).indexOf(className);
            this.CAMModel.setSelectedIndex(ix === undefined || ix === -1 ? null : ix);
        }
    }

    protected async load(metadata?: PoseMetadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer) {
        await tf.ready();
        if (metadata && model && weights) {
            this.trained = true;
            const tmmodel = await createPose(metadata);
            tmmodel.model = await tf.loadLayersModel({
                load: async () => {
                    return {
                        modelTopology: model.modelTopology,
                        weightData: weights,
                        weightSpecs: model.weightsManifest[0].weights,
                    };
                },
            });
            this.model = tmmodel;
        } else {
            const tmmodel = await createPose({ tfjsVersion: tf.version.tfjs });
            this.model = tmmodel;
            tmmodel.setName('My Model');
        }

        this.imageSize = (this.model.getMetadata().modelSettings as any)?.posenet?.inputResolution || 257;
    }

    public async ready() {
        return this.isReady() || this._ready || false;
    }

    /**
     * If a pose is available, draw the keypoints and skeleton.
     *
     * @param image Image to draw the pose into.
     */
    public draw(image: HTMLCanvasElement) {
        if (this.model && this.lastPose) {
            const ctx = image.getContext('2d');
            if (this.lastPose && ctx) {
                try {
                    drawKeypoints(this.lastPose.keypoints, 0.5, ctx);
                    drawSkeleton(this.lastPose.keypoints, 0.5, ctx);
                } catch (e) {
                    console.error(e);
                }
            }
        }
        return image;
    }

    /**
     * Estimate pose if this is a PoseNet model, otherwise do nothing.
     * This caches the pose so draw() can use it without re-estimating.
     *
     * @param image Input image at correct resolution
     */
    public async estimate(image: HTMLCanvasElement): Promise<HTMLCanvasElement> {
        if (this.model && !this.busy) {
            this.busy = true;
            try {
                const poseData = await this.model.estimatePose(image);
                this.lastPose = poseData.pose;
            } catch (e) {
                console.error('Estimation error', e);
                this.lastPose = undefined;
            }
            this.busy = false;
        }
        return image;
    }

    /* Preechakul et al., Improved image classification explainability with high-accuracy heatmaps, iScience 25, March 18, 2022. https://doi.org/10.1016/j.isci.2022.103933 */

    public async predict(image: HTMLCanvasElement): Promise<ExplainedPredictionsOutput> {
        if (!this.trained || this._disposed) return { predictions: [] };
        if (this.model) {
            // Pose path: wrap entirely so disposal during any await is silent
            let pose:
                | { keypoints: { score: number; position: { y: number; x: number }; part: string }[]; score: number }
                | undefined;
            let posenetOutput: Float32Array;
            try {
                const result = await this.model.estimatePose(image);
                pose = result.pose;
                posenetOutput = result.posenetOutput;
                this.lastPose = pose;
            } catch {
                this.lastPose = undefined;
                return { predictions: [] };
            }

            if (this._disposed || !this.model) return { predictions: [] };
            if (!posenetOutput || posenetOutput.length === 0) {
                this.lastPose = undefined;
                return { predictions: [] };
            }

            // XAI path
            if (this.explained && this.CAMModel && pose) {
                const cam = this.CAMModel;
                try {
                    const camResult = await cam.createPoseCAM(image, posenetOutput);
                    if (cam.isDisposed()) this.CAMModel = undefined;
                    if (this._disposed || !this.model) return { predictions: [] };
                    if (this.explained && camResult.keypointImportance) {
                        renderPoseXAI(image, this.explained, pose.keypoints, camResult.keypointImportance, 0.3);
                    } else if (this.explained && camResult.heatmapData.length > 0) {
                        renderHeatmap(image, this.explained, camResult.heatmapData);
                    }
                    return { predictions: camResult.predictions };
                } catch (error) {
                    // Disposal during switch: silent. Genuine failure: warn and fall through.
                    if (!this._disposed) console.warn('XAI (pose) failed, falling back to standard predict:', error);
                }
            }

            // Plain predict (no XAI or XAI failed/disposed)
            if (this._disposed || !this.model) return { predictions: [] };
            try {
                const predictions = await this.model.predict(posenetOutput);
                return { predictions };
            } catch {
                return { predictions: [] };
            }
        }
        return { predictions: [] };
    }

    /**
     * Predict directly from pose output data (for validation/internal use)
     */
    public async predictFromPoseData(poseData: Float32Array): Promise<ExplainedPredictionsOutput> {
        if (!this.model) {
            console.warn('VALIDATION ERROR: Pose model not initialized');
            return { predictions: [] };
        }
        if (!this.model.model) {
            console.warn('VALIDATION ERROR: Pose model.model is null');
            return { predictions: [] };
        }
        if (this.model.model.layers.length === 0) {
            console.warn('VALIDATION ERROR: Pose model has no layers');
            return { predictions: [] };
        }

        try {
            const predictions = await this.model.predict(poseData);

            if (!predictions || predictions.length === 0) {
                console.warn('VALIDATION ERROR: Pose model returned empty predictions');
            }
            return { predictions };
        } catch (error) {
            console.error('VALIDATION ERROR during pose prediction:', error);
            return { predictions: [] };
        }
    }

    public async train(params: TrainingParameters, callbacks: tf.CustomCallbackArgs) {
        this.trained = false;
        if (this.model) {
            return this.model.train(params, callbacks).then((m) => {
                if (this.model) {
                    if (this.CAMModel) this.CAMModel.dispose();
                    this.CAMModel = new CAM(this.model);
                }
                this.trained = true;
                return m;
            });
        }
        throw new Error('no_model');
    }

    public async addExample(className: number, image: HTMLCanvasElement) {
        if (this.model) {
            const { heatmapScores, offsets } = await this.model.estimatePoseOutputs(image);
            const posenetOutput = await this.model.poseOutputsToAray(heatmapScores, offsets);
            return this.model.addExample(className, posenetOutput);
        }
    }

    public dispose() {
        this._disposed = true;
        // Dispose CAM first before disposing models, since CAM reference model layers
        if (this.CAMModel) {
            try {
                this.CAMModel.dispose();
            } catch (error) {
                console.warn('Error disposing CAM model:', error);
            }
        }

        if (this.model) {
            try {
                if (this.model.isTrained) {
                    this.model.dispose();
                } else {
                    this.model.model?.dispose();
                }
            } catch (error) {
                console.warn('Error disposing pose model:', error);
            }
        }
        this.model = undefined;
        this.lastPose = undefined;
        this.CAMModel = undefined;
    }

    public setName(name: string): void {
        if (this.model) {
            this.model.setName(name);
        }
    }

    public getModel(): TeachablePoseNet | undefined {
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
        } else {
            throw new Error('setLabels is only supported for image and pose models');
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
        } else {
            throw new Error('no_model');
        }
    }
}
