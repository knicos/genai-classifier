import * as tf from '@tensorflow/tfjs';
import { ExplainedPredictionsOutput, TMType, TeachableModel } from './TeachableModel';
import { AudioExample } from './gtm-utils/recorder';
import {
    create,
    SpeechCommandRecognizer,
    SpeechCommandRecognizerMetadata,
    TransferLearnConfig,
    TransferSpeechCommandRecognizer,
} from './speech-commands';

const NULLARRAY: string[] = [];

export default class SpeechModel implements TeachableModel {
    private recognizer?: SpeechCommandRecognizer;
    private transferRecognizer?: TransferSpeechCommandRecognizer;
    private metadata: SpeechCommandRecognizerMetadata;
    protected _ready?: Promise<boolean>;
    protected trained = false;
    protected busy = false;
    protected imageSize = 224;
    protected _disposed = false;
    public variant: TMType = 'speech';
    public explained?: HTMLCanvasElement;
    modelBaseUrl = 'https://tmstore.blob.core.windows.net/models';
    private labels: string[] = [];

    constructor(
        type: TMType,
        metadata?: SpeechCommandRecognizerMetadata,
        model?: tf.io.ModelJSON,
        weights?: ArrayBuffer
    ) {
        if (type !== 'speech') {
            throw new Error(`Invalid type for SpeechModel: ${type}`);
        }

        this.metadata = metadata || {
            wordLabels: [],
            tfjsSpeechCommandsVersion: '0.4.0',
        };

        this._ready = this.load(metadata, model, weights).then(() => true);
    }

    getModel() {
        return this.transferRecognizer ? this.transferRecognizer : this.recognizer;
    }

    getVariant(): TMType {
        return this.variant;
    }

    public setXAICanvas() {
        return;
    }

    public setXAIClass() {}

    protected async load(metadata?: SpeechCommandRecognizerMetadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer) {
        await tf.ready();
        try {
            this.recognizer = create(
                'BROWSER_FFT',
                undefined,
                model
                    ? {
                          modelTopology: model.modelTopology,
                          weightData: weights,
                          weightSpecs: model.weightsManifest[0].weights,
                      }
                    : undefined,
                model ? metadata : undefined
            );
            await this.recognizer
                .ensureModelLoaded()
                .then(() => {
                    this.transferRecognizer = model ? undefined : this.recognizer?.createTransfer('my-transfer-model');
                })
                .then(() => true)
                .catch((err: unknown) => {
                    console.error('Error loading model:', err);
                    return false;
                });
            this.trained = !!model;
        } catch (err) {
            console.error('Error loading custom model:', err);
            throw err;
        }

        this.imageSize = 0;
    }

    public getRecognizerModel() {
        return this.transferRecognizer ? this.transferRecognizer : this.recognizer;
    }

    public countExamples() {
        if (this.transferRecognizer) {
            return this.transferRecognizer.countExamples();
        }
        return {};
    }

    public async ready() {
        return this.isReady() || this._ready || false;
    }

    public async predict(input: AudioExample): Promise<ExplainedPredictionsOutput> {
        if (!this.trained) {
            throw new Error('Model is not trained yet.');
        }
        await this.getRecognizerModel()?.ensureModelLoaded();
        const result = await this.getRecognizerModel()?.recognize(input.spectrogram.data);
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

    public async train(params: TransferLearnConfig, callbacks: tf.CustomCallbackArgs) {
        this.trained = false;
        if (this.transferRecognizer) {
            await this.transferRecognizer.train({
                epochs: params.epochs,
                batchSize: params.batchSize,
                callback: callbacks,
            });

            this.trained = true;
            return;
        }
        throw new Error('no_model');
    }

    public draw(image: HTMLCanvasElement): HTMLCanvasElement {
        return image;
    }

    public async estimate(image: HTMLCanvasElement): Promise<HTMLCanvasElement> {
        return image;
    }

    public async addExample(label: number, example: AudioExample) {
        example.label = this.labels[label];
        if (this.transferRecognizer) {
            return this.transferRecognizer.addExample(example);
        }
        throw new Error('no_model');
    }

    public dispose() {
        this._disposed = true;
        if (this.transferRecognizer) {
            this.transferRecognizer.dispose();
        }
        if (this.recognizer) {
            this.recognizer.dispose();
        }
    }

    public setName(): void {
        // TODO
    }

    public getImageSize() {
        return this.imageSize;
    }

    public isTrained() {
        return this.trained;
    }

    public isReady() {
        return !!this.getRecognizerModel();
    }

    public setSeed() {
        // Not needed
    }

    public getMetadata() {
        this.metadata.wordLabels = this.getRecognizerModel()?.wordLabels() || [];
        return this.metadata;
    }

    public async save(handler: tf.io.IOHandler) {
        if (this.transferRecognizer) {
            return this.transferRecognizer.save(handler);
        }
        throw new Error('no_model');
    }

    public setLabels(labels: string[]) {
        this.labels = labels;
    }

    public getLabels(): string[] {
        return this.getRecognizerModel()?.wordLabels() || NULLARRAY;
    }

    public getLabel(ix: number): string {
        return this.getRecognizerModel()?.wordLabels()[ix] || '';
    }

    public getNumExamples(): number {
        if (this.transferRecognizer) {
            const counts = this.transferRecognizer.countExamples();
            return Object.values(counts).reduce((a, b) => a + b, 0);
        }
        return 0;
    }

    public getExamplesPerClass(): number[] {
        if (this.transferRecognizer) {
            const counts = this.transferRecognizer.countExamples();
            return this.transferRecognizer.wordLabels().map((label) => counts[label] || 0);
        }
        return [];
    }

    public getNumValidation(): number {
        if (this.transferRecognizer) {
            const counts = this.transferRecognizer.countExamples();
            return Object.values(counts).reduce((t, e) => t + Math.ceil(e * 0.15), 0);
        }
        return 0;
    }

    public async calculateAccuracy() {
        if (this.transferRecognizer) {
            return { reference: null, predictions: tf.tensor([]) };
        } else {
            throw new Error('no_model');
        }
    }
}
