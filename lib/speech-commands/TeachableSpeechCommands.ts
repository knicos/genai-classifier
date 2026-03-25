import { Example, SpeechCommandRecognizer, TransferSpeechCommandRecognizer } from './types';
import { create } from './index';
import { CustomCallbackArgs } from '@tensorflow/tfjs';

export interface TeachableSpeechCommandsMetadata {
    modelBaseUrl?: string;
    name?: string;
    packageName?: string;
}

export interface SoundTrainingParams {
    epochs: number;
}

export default class TeachableSpeechCommands {
    private recognizer: SpeechCommandRecognizer;
    public transferRecognizer?: TransferSpeechCommandRecognizer;
    private labels: string[] = [];
    private metadata: TeachableSpeechCommandsMetadata;
    public ready: Promise<boolean>;

    constructor(metadata: TeachableSpeechCommandsMetadata) {
        this.metadata = metadata;
        this.recognizer = create('BROWSER_FFT');
        this.ready = this.recognizer
            .ensureModelLoaded()
            .then(() => {
                this.transferRecognizer = this.recognizer.createTransfer('my-transfer-model');
                //return this.transferRecognizer.ensureModelLoaded();
            })
            .then(() => true);
    }

    public setName(name: string) {
        this.metadata.name = name;
    }

    public getMetadata() {
        return this.metadata;
    }

    public getExamples(label: string): Example[] {
        if (!this.transferRecognizer) {
            return [];
        }
        return this.transferRecognizer.getExamples(label).map((example) => example.example) || [];
    }

    public async addExample(label: string, example: Example) {
        if (!this.transferRecognizer) {
            throw new Error('Transfer recognizer is not initialized yet.');
        }
        if (label !== example.label) {
            throw new Error(`Example label "${example.label}" does not match the provided label "${label}".`);
        }
        if (!this.labels.includes(example.label)) {
            this.labels.push(example.label);
        }
        this.transferRecognizer.addExample(example);
    }

    async train(params: SoundTrainingParams, callbacks: CustomCallbackArgs) {
        if (!this.transferRecognizer) {
            throw new Error('Transfer recognizer is not initialized yet.');
        }
        await this.transferRecognizer.train({
            epochs: params.epochs,
            callback: callbacks,
        });
    }
}
