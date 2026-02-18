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
import { renderHeatmap, renderPoseXAI } from './heatmap';
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

interface BaseMetadata {
    modelBaseUrl?: string;
}

export type Metadata = BaseMetadata & (ImageMetadata | PoseMetadata);

const NULLARRAY: string[] = [];

export default class TeachableModel {
    private imageModel?: TeachableMobileNet;
    private poseModel?: TeachablePoseNet;
    private _ready?: Promise<boolean>;
    private trained = false;
    private lastPose?: Pose;
    private busy = false;
    private imageSize = 224;
    public variant: TMType = 'image';
    public explained?: HTMLCanvasElement;
    private CAMModel?: CAM;
    private modelBaseUrl = 'https://tmstore.blob.core.windows.net/models';

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

            if (metadata?.modelBaseUrl) {
                this.modelBaseUrl = metadata.modelBaseUrl;
            }

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

    public setXAICanvas(canvas: HTMLCanvasElement) {
        if (this.imageModel) {
            this.explained = canvas;
            if (!this.CAMModel) {
                this.CAMModel = new CAM(this.imageModel);
            }
            return;
        } else if (this.poseModel) {
            this.explained = canvas;
            if (!this.CAMModel) {
                this.CAMModel = new CAM(this.poseModel);
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
            const ix = typeof className === 'number' 
                ? className 
                : (this.imageModel?.getLabels() || this.poseModel?.getLabels() || []).indexOf(className);
            this.CAMModel.setSelectedIndex(ix === undefined || ix === -1 ? null : ix);
        }
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
                modelBaseUrl: this.modelBaseUrl,
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
     * This caches the pose so draw() can use it without re-estimating.
     *
     * @param image Input image at correct resolution
     */
    public async estimate(image: HTMLCanvasElement): Promise<void> {
        if (this.poseModel && !this.busy) {
            this.busy = true;
            try {
                const poseData = await this.poseModel.estimatePose(image);
                this.lastPose = poseData.pose;
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
            // For validation: always get fresh pose estimation, don't rely on cached lastPoseOut
            // This ensures accurate predictions even when called rapidly or out of sequence
            const { pose, posenetOutput } = await this.poseModel.estimatePose(image);
            
            if (!posenetOutput || posenetOutput.length === 0) {
                console.warn('Failed to extract pose from image');
                return { predictions: [] };
            }
            
            // Always draw pose skeleton on canvas for pose models
            if (pose) {
                const ctx = image.getContext('2d');
                if (ctx) {
                    drawKeypoints(pose.keypoints, 0.5, ctx);
                    drawSkeleton(pose.keypoints, 0.5, ctx);
                }
            }
            
            // If XAI is enabled, generate explanations
            if (this.explained && this.CAMModel && pose) {
                try {
                    const camResult = await this.CAMModel.createPoseCAM(image, posenetOutput);
                    // Use keypoint-importance visualization if available, otherwise fall back to heatmap
                    if (camResult.keypointImportance) {
                        renderPoseXAI(image, this.explained, pose.keypoints, camResult.keypointImportance, 0.3);
                    } else {
                        renderHeatmap(image, this.explained, camResult.heatmapData);
                    }
                    return { predictions: camResult.predictions };
                } catch (error) {
                    console.warn('XAI generation failed, falling back to standard prediction:', error);
                    const predictions = await this.poseModel.predict(posenetOutput);
                    return { predictions };
                }
            } else {
                const predictions = await this.poseModel.predict(posenetOutput);
                return { predictions };
            }
        }
        return { predictions: [] };
    }

    /**
     * Predict directly from pose output data (for validation/internal use)
     */
    public async predictFromPoseData(poseData: Float32Array): Promise<ExplainedPredictionsOutput> {
        console.log('predictFromPoseData called with data length:', poseData?.length);
        
        if (!this.poseModel) {
            console.warn('VALIDATION ERROR: Pose model not initialized');
            return { predictions: [] };
        }
        
        console.log('Pose model exists, checking model.model...');
        
        if (!this.poseModel.model) {
            console.warn('VALIDATION ERROR: Pose model.model is null');
            return { predictions: [] };
        }
        
        console.log('Model exists with layers:', this.poseModel.model.layers.length);
        
        if (this.poseModel.model.layers.length === 0) {
            console.warn('VALIDATION ERROR: Pose model has no layers');
            return { predictions: [] };
        }
        
        try {
            console.log('About to call poseModel.predict...');
            console.log('Model labels:', this.poseModel.getMetadata().labels);
            
            const predictions = await this.poseModel.predict(poseData);
            
            console.log('Predictions returned:', predictions);
            console.log('Predictions length:', predictions?.length);
            
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
                if (this.poseModel) {
                    if (this.CAMModel) this.CAMModel.dispose();
                    this.CAMModel = new CAM(this.poseModel);
                }
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
        // Pose state is no longer cached
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
