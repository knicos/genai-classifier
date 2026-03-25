import * as tf from '@tensorflow/tfjs';
import { ExplainedPredictionsOutput, TMType, TeachableModel } from './TeachableModel';
import TeachableSpeechCommands, {
    SoundTrainingParams,
    TeachableSpeechCommandsMetadata,
} from './speech-commands/TeachableSpeechCommands';
import { AudioExample } from './gtm-utils/recorder';

const NULLARRAY: string[] = [];

export default class SpeechModel implements TeachableModel {
    protected model?: TeachableSpeechCommands;
    protected _ready?: Promise<boolean>;
    protected trained = false;
    protected busy = false;
    protected imageSize = 224;
    protected _disposed = false;
    public variant: TMType = 'speech';
    public explained?: HTMLCanvasElement;
    modelBaseUrl = 'https://tmstore.blob.core.windows.net/models';

    constructor(type: TMType) {
        if (type !== 'speech') {
            throw new Error(`Invalid type for SpeechModel: ${type}`);
        }
        this._ready = this.load().then(() => true);
        console.log('SpeechModel initialized');
    }

    public setXAICanvas() {
        if (this.model) {
            return;
        }
        throw new Error('no_model');
    }

    public setXAIClass() {}

    protected async load(metadata?: TeachableSpeechCommandsMetadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer) {
        await tf.ready();
        if (metadata && model && weights) {
            // TODO
        } else {
            this.model = new TeachableSpeechCommands(metadata || {});
            console.log('Loading model...', this.model);
            await this.model.ready;
        }

        this.imageSize = 0;
    }

    public async ready() {
        return this.isReady() || this._ready || false;
    }

    public async predict(input: AudioExample): Promise<ExplainedPredictionsOutput> {
        if (!this.trained) {
            throw new Error('Model is not trained yet.');
        }
        await this.model?.transferRecognizer?.ensureModelLoaded();
        const result = await this.model?.transferRecognizer?.recognize(input.spectrogram.data);
        if (result) {
            const labels = this.getLabels();
            const predictions: ExplainedPredictionsOutput = {
                predictions: Array.from(result.scores as Float32Array).map((score, i) => ({
                    className: labels[i] || `class-${i}`,
                    probability: score as number,
                })),
            };
            return predictions;
        }
        return { predictions: [] };
    }

    public async train(params: SoundTrainingParams, callbacks: tf.CustomCallbackArgs) {
        this.trained = false;
        if (this.model) {
            return this.model.train(params, callbacks).then((m) => {
                this.trained = true;
                return m;
            });
        }
        throw new Error('no_model');
    }

    public draw(image: HTMLCanvasElement): HTMLCanvasElement {
        return image;
    }

    public async estimate(image: HTMLCanvasElement): Promise<HTMLCanvasElement> {
        return image;
    }

    public async addExample(_: number, example: AudioExample) {
        if (this.model) {
            return this.model.addExample(example.label, example);
        }
        throw new Error('no_model');
    }

    public dispose() {
        this._disposed = true;

        this.model?.transferRecognizer?.dispose();

        this.model = undefined;
    }

    public setName(): void {
        // TODO
    }

    public getModel(): TeachableSpeechCommands | undefined {
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

    public setSeed() {
        // Not needed
    }

    public getMetadata() {
        if (this.model) {
            return this.model.getMetadata();
        }
    }

    public async save() {
        if (this.model) {
            //return this.model.save(handler);
            return undefined;
        }
        throw new Error('no_model');
    }

    public setLabels() {
        // Not needed.
    }

    public getLabels(): string[] {
        if (this.model && this.model.transferRecognizer) {
            return this.model.transferRecognizer.wordLabels();
        }
        return NULLARRAY;
    }

    public getLabel(ix: number): string {
        if (this.model && this.model.transferRecognizer) {
            return this.model.transferRecognizer.wordLabels()[ix] || '';
        }
        return '';
    }

    public getNumExamples(): number {
        if (this.model && this.model.transferRecognizer) {
            const counts = this.model.transferRecognizer.countExamples();
            return Object.values(counts).reduce((a, b) => a + b, 0);
        }
        return 0;
    }

    public getExamplesPerClass(): number[] {
        if (this.model && this.model.transferRecognizer) {
            const counts = this.model.transferRecognizer.countExamples();
            return this.model.transferRecognizer.wordLabels().map((label) => counts[label] || 0);
        }
        return [];
    }

    public getNumValidation(): number {
        if (this.model && this.model.transferRecognizer) {
            const counts = this.model.transferRecognizer.countExamples();
            return Object.values(counts).reduce((t, e) => t + Math.ceil(e * 0.15), 0);
        }
        return 0;
    }

    public async calculateAccuracy() {
        if (this.model) {
            return { reference: null, predictions: tf.tensor([]) };
        } else {
            throw new Error('no_model');
        }
    }
}
