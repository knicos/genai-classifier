import { TeachableMobileNet, Metadata as ImageMetadata, createTeachable as createImage } from './gtm-image';
import { TrainingParameters as ImageTrainingParams } from './gtm-image/teachable-mobilenet';
import * as tf from '@tensorflow/tfjs';
import { renderHeatmap } from './heatmap';
import { CAM } from './xai';
import { TeachableModel, ExplainedPredictionsOutput, TMType } from './TeachableModel';

const NULLARRAY: string[] = [];

interface TrainingParameters extends ImageTrainingParams {}

interface BaseMetadata {
    modelBaseUrl?: string;
}

export type Metadata = BaseMetadata & ImageMetadata;

export default class ImageModel implements TeachableModel {
    protected model?: TeachableMobileNet;
    protected _ready?: Promise<boolean>;
    protected trained = false;
    protected busy = false;
    protected imageSize = 224;
    protected _disposed = false;
    public variant: TMType = 'image';
    public explained?: HTMLCanvasElement;
    modelBaseUrl = 'https://tmstore.blob.core.windows.net/models';
    private CAMModel?: CAM;

    constructor(type: TMType, metadata?: Metadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer) {
        if (type !== 'image') {
            throw new Error(`Invalid type for ImageModel: ${type}`);
        }

        if (metadata?.packageName) {
            if (metadata.packageName !== '@teachablemachine/image') {
                throw new Error(`Invalid packageName for ImageModel: ${metadata.packageName}`);
            }
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

    public async estimate(image: HTMLCanvasElement): Promise<HTMLCanvasElement> {
        return image;
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

    protected async load(metadata?: ImageMetadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer) {
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
            this.model = tmmodel;
            this.trained = true;
        } else {
            const tmmodel = await createImage({ tfjsVersion: tf.version.tfjs }, { version: 2, alpha: 0.35 });
            this.model = tmmodel;
            tmmodel.setName('My Model');
        }

        this.imageSize = this.model.getMetadata().imageSize || 224;
    }

    public async ready() {
        return this.isReady() || this._ready || false;
    }

    /* Preechakul et al., Improved image classification explainability with high-accuracy heatmaps, iScience 25, March 18, 2022. https://doi.org/10.1016/j.isci.2022.103933 */
    public async predict(image: HTMLCanvasElement): Promise<ExplainedPredictionsOutput> {
        if (!this.trained || this._disposed) return { predictions: [] };

        if (this.model) {
            if (this.explained && this.CAMModel) {
                try {
                    const cam = this.CAMModel;
                    const camResult = await cam.createCAM(image);
                    if (cam.isDisposed()) this.CAMModel = undefined;
                    if (this.explained && camResult.heatmapData.length > 0) {
                        renderHeatmap(image, this.explained, camResult.heatmapData);
                    }
                    return { predictions: camResult.predictions };
                } catch (error) {
                    // Disposal during switch: silent. Genuine failure: warn and fall through.
                    if (!this._disposed) console.warn('XAI (image) failed, falling back to standard predict:', error);
                }
            }
            if (this._disposed || !this.model) return { predictions: [] };
            try {
                const predictions = await this.model.predict(image);
                return { predictions };
            } catch {
                return { predictions: [] };
            }
        }
        return { predictions: [] };
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
            return this.model.addExample(className, image);
        }
        throw new Error('no_model');
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

        // Then dispose the actual models
        if (this.model) {
            try {
                if (this.model.isTrained) {
                    this.model.dispose();
                } else {
                    this.model.model?.dispose();
                }
            } catch (error) {
                console.warn('Error disposing image model:', error);
            }
        }

        this.model = undefined;
        this.CAMModel = undefined;
        // Pose state is no longer cached
    }

    public setName(name: string): void {
        if (this.model) {
            this.model.setName(name);
        }
    }

    public getModel(): TeachableMobileNet | undefined {
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
        this.model?.setSeed(seed);
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

    /**
     * If a pose is available, draw the keypoints and skeleton.
     *
     * @param image Image to draw the pose into.
     */
    public draw(image: HTMLCanvasElement) {
        return image;
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
