import { TeachableMobileNet, Metadata as ImageMetadata, createTeachable as createImage } from './gtm-image';
import { TrainingParameters as ImageTrainingParams } from './gtm-image/teachable-mobilenet';
import { TrainingParameters as PoseTrainingParams } from './gtm-pose/teachable-posenet';
import {
    TeachablePoseNet,
    Metadata as PoseMetadata,
    drawKeypoints,
    drawSkeleton,
    createTeachable as createPose,
} from './gtm-pose';
import * as tf from '@tensorflow/tfjs';
import { renderHeatmap } from './heatmap';
import { CAM } from './xai';

export type TMType = 'image' | 'pose';

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

export interface PredictionsOutput {
    className: string;
    probability: number;
}

export interface ExplainedPredictionsOutput {
    predictions: PredictionsOutput[];
    heatmap?: number[][];
}

interface TrainingParameters extends ImageTrainingParams, PoseTrainingParams {}

export type Metadata = ImageMetadata | PoseMetadata;

const NULLARRAY: string[] = [];

export default class TeachableModel {
    private imageModel?: TeachableMobileNet;
    private poseModel?: TeachablePoseNet;
    private _ready?: Promise<boolean>;
    private trained = false;
    private lastPose?: Pose;
    private lastPoseOut?: Float32Array;
    private busy = false;
    private imageSize = 224;
    public variant: TMType = 'image';
    public explained?: HTMLCanvasElement;
    private CAMModel?: CAM;

    constructor(type: TMType, metadata?: Metadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer) {
        this._ready = new Promise((resolve) => {
            let atype = type;
            if (metadata?.packageName) {
                if (metadata.packageName === '@teachablemachine/pose') {
                    atype = 'pose';
                } else if (metadata.packageName === '@teachablemachine/image') {
                    atype = 'image';
                }
            }

            this.variant = atype;

            if (atype === 'image') {
                this.loadImage(metadata, model, weights).then(() => {
                    resolve(true);
                });
            } else if (atype === 'pose') {
                this.loadPose(metadata as PoseMetadata, model, weights).then(() => resolve(true));
            } else {
                resolve(false);
            }
        });
    }

    public setName(name: string) {
        if (this.imageModel) {
            this.imageModel.setName(name);
        } else if (this.poseModel) {
            this.poseModel.setName(name);
        }
    }

    public getVariant() {
        return this.variant;
    }

    public getImageModel() {
        return this.imageModel;
    }

    public getPoseModel() {
        return this.poseModel;
    }

    public getImageSize() {
        return this.imageSize;
    }

    public isTrained() {
        return this.trained;
    }

    private async loadImage(metadata?: ImageMetadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer) {
        await tf.ready();
        if (metadata && model && weights) {
            const tmmodel = await createImage(metadata, {
                version: 2,
                alpha: 0.35,
                modelBaseUrl: 'https://tmstore.blob.core.windows.net/models',
            });
            tmmodel.model = await tf.loadLayersModel({
                load: async () => {
                    return {
                        modelTopology: model.modelTopology,
                        weightData: weights,
                        weightSpecs: model.weightsManifest[0].weights,
                    };
                },
            });
            this.imageModel = tmmodel;
            this.trained = true;
        } else {
            const tmmodel = await createImage({ tfjsVersion: tf.version.tfjs }, { version: 2, alpha: 0.35 });
            this.imageModel = tmmodel;
            tmmodel.setName('My Model');
        }

        this.imageSize = this.imageModel.getMetadata().imageSize || 224;
    }

    private async loadPose(metadata?: PoseMetadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer) {
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
            this.poseModel = tmmodel;
        } else {
            const tmmodel = await createPose({ tfjsVersion: tf.version.tfjs });
            this.poseModel = tmmodel;
            tmmodel.setName('My Model');
        }

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        this.imageSize = (this.poseModel.getMetadata().modelSettings as any)?.posenet?.inputResolution || 257;
    }

    public async ready() {
        return this.isReady() || this._ready || false;
    }

    public isReady() {
        return !!(this.imageModel || this.poseModel);
    }

    public setSeed(seed: string) {
        if (this.imageModel) {
            this.imageModel.setSeed(seed);
        } else if (this.poseModel) {
            this.poseModel.setSeed(seed);
        }
    }

    public getMetadata() {
        if (this.imageModel) {
            return this.imageModel.getMetadata();
        } else if (this.poseModel) {
            return this.poseModel.getMetadata();
        }
    }

    public async save(handler: tf.io.IOHandler) {
        if (this.imageModel) {
            return this.imageModel.save(handler);
        } else if (this.poseModel) {
            return this.poseModel.save(handler);
        }
    }

    /**
     * If a pose is available, draw the keypoints and skeleton.
     *
     * @param image Image to draw the pose into.
     */
    public draw(image: HTMLCanvasElement) {
        if (this.poseModel && this.lastPose) {
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
    }

    /**
     * Estimate pose if this is a PoseNet model, otherwise do nothing.
     * This must be called for prediction to work with PoseNet.
     *
     * @param image Input image at correct resolution
     */
    public async estimate(image: HTMLCanvasElement): Promise<void> {
        if (this.poseModel && !this.busy) {
            // Only allow one estimate at a time.
            this.busy = true;
            try {
                const { pose, posenetOutput } = await this.poseModel.estimatePose(image);
                this.lastPose = pose;
                this.lastPoseOut = posenetOutput;
            } catch (e) {
                console.error('Estimation error', e);
            }
            this.busy = false;
        }
    }

    /* Preechakul et al., Improved image classification explainability with high-accuracy heatmaps, iScience 25, March 18, 2022. https://doi.org/10.1016/j.isci.2022.103933 */

    public async predict(image: HTMLCanvasElement): Promise<ExplainedPredictionsOutput> {
        if (!this.trained) return { predictions: [] };

        if (this.imageModel) {
            if (this.explained && this.CAMModel) {
                const camResult = await this.CAMModel.createCAM(image);
                renderHeatmap(image, this.explained, camResult.heatmapData);
                return { predictions: camResult.predictions };
            } else {
                const predictions = await this.imageModel.predict(image);
                return { predictions };
            }
        } else if (this.poseModel) {
            // Force an estimate if we are not generating one already
            if (!this.busy) {
                // Note: doesn't wait for estimate promise.
                this.estimate(image);
            }
            if (!this.lastPoseOut) return { predictions: [] };
            const result = { predictions: await this.poseModel.predict(this.lastPoseOut) };
            return result;
        }
        return { predictions: [] };
    }

    public async train(params: TrainingParameters, callbacks: tf.CustomCallbackArgs) {
        this.trained = false;
        if (this.imageModel) {
            return this.imageModel.train(params, callbacks).then((m) => {
                if (this.imageModel) {
                    if (this.CAMModel) this.CAMModel.dispose();
                    this.CAMModel = new CAM(this.imageModel);
                }
                this.trained = true;
                return m;
            });
        } else if (this.poseModel) {
            return this.poseModel.train(params, callbacks).then((m) => {
                this.trained = true;
                return m;
            });
        }
    }

    public async addExample(className: number, image: HTMLCanvasElement) {
        if (this.imageModel) {
            return this.imageModel.addExample(className, image);
        } else if (this.poseModel) {
            const { heatmapScores, offsets } = await this.poseModel.estimatePoseOutputs(image);
            const posenetOutput = this.poseModel.poseOutputsToAray(heatmapScores, offsets);
            return this.poseModel.addExample(className, posenetOutput);
        }
    }

    public setLabels(labels: string[]) {
        if (this.imageModel) {
            this.imageModel.setLabels(labels);
        } else if (this.poseModel) {
            this.poseModel.setLabels(labels);
        }
    }

    public dispose() {
        if (this.imageModel) {
            if (this.imageModel.isTrained) {
                this.imageModel.dispose();
            } else {
                this.imageModel.model?.dispose();
            }
        }
        if (this.poseModel) {
            if (this.poseModel.isTrained) {
                this.poseModel.dispose();
            } else {
                this.poseModel.model?.dispose();
            }
        }
        if (this.CAMModel) {
            this.CAMModel.dispose();
        }
        this.imageModel = undefined;
        this.poseModel = undefined;
        this.lastPose = undefined;
        this.lastPoseOut = undefined;
    }

    public getLabels(): string[] {
        if (this.imageModel) {
            return this.imageModel.getLabels();
        } else if (this.poseModel) {
            return this.poseModel.getLabels();
        }
        return NULLARRAY;
    }

    public getLabel(ix: number): string {
        if (this.imageModel) {
            return this.imageModel.getLabel(ix);
        } else if (this.poseModel) {
            return this.poseModel.getLabel(ix);
        }
        return '';
    }

    public getNumExamples(): number {
        if (this.imageModel) {
            return this.imageModel.examples.reduce((t, e) => t + e.length, 0);
        } else if (this.poseModel) {
            return this.poseModel.examples.reduce((t, e) => t + e.length, 0);
        }
        return 0;
    }

    public getExamplesPerClass(): number[] {
        if (this.imageModel) {
            return this.imageModel.examples.map((e) => e.length);
        } else if (this.poseModel) {
            return this.poseModel.examples.map((e) => e.length);
        }
        return [];
    }

    public getNumValidation(): number {
        if (this.imageModel) {
            return this.imageModel.examples.reduce((t, e) => t + Math.ceil(e.length * 0.15), 0);
        } else if (this.poseModel) {
            return this.poseModel.examples.reduce((t, e) => t + Math.ceil(e.length * 0.15), 0);
        }
        return 0;
    }

    public calculateAccuracy() {
        if (this.imageModel) {
            return this.imageModel.calculateAccuracyPerClass();
        } else if (this.poseModel) {
            return this.poseModel.calculateAccuracyPerClass();
        } else {
            throw new Error('no_model');
        }
    }
}
