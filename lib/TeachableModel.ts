import { TeachableMobileNet, Metadata as ImageMetadata } from './gtm-image';
import { TrainingParameters as ImageTrainingParams } from './gtm-image/teachable-mobilenet';
import { TrainingParameters as PoseTrainingParams } from './gtm-pose/teachable-posenet';
import { TeachablePoseNet, Metadata as PoseMetadata } from './gtm-pose';
import * as tf from '@tensorflow/tfjs';
import { AudioExample } from './gtm-utils/recorder';
import TeachableSpeechCommands, {
    SoundTrainingParams,
    TeachableSpeechCommandsMetadata,
} from './speech-commands/TeachableSpeechCommands';

export type TMType = 'image' | 'pose' | 'speech' | 'text';

export interface PredictionsOutput {
    className: string;
    probability: number;
}

export interface ExplainedPredictionsOutput {
    predictions: PredictionsOutput[];
    heatmap?: number[][];
}

interface TrainingParameters extends ImageTrainingParams, PoseTrainingParams, SoundTrainingParams {}

interface BaseMetadata {
    modelBaseUrl?: string;
}

export type Metadata = BaseMetadata & (ImageMetadata | PoseMetadata | TeachableSpeechCommandsMetadata);

export interface TeachableModel {
    readonly variant: TMType;
    explained?: HTMLCanvasElement;
    readonly modelBaseUrl: string;

    // constructor(type: TMType, metadata?: Metadata, model?: tf.io.ModelJSON, weights?: ArrayBuffer);

    setXAICanvas(canvas: HTMLCanvasElement): void;

    setXAIClass(className: string | number | null): void;

    setName(name: string): void;

    getModel(): TeachableMobileNet | TeachablePoseNet | TeachableSpeechCommands | undefined;

    getImageSize(): number;

    isTrained(): boolean;

    ready(): Promise<boolean>;

    isReady(): boolean;

    setSeed(seed: string): void;

    getMetadata(): Metadata | undefined;

    save(handler: tf.io.IOHandler): Promise<tf.io.SaveResult | undefined>;

    draw(image: HTMLCanvasElement): HTMLCanvasElement;

    estimate(image: HTMLCanvasElement): Promise<HTMLCanvasElement>;

    predict(image: HTMLCanvasElement | AudioExample): Promise<ExplainedPredictionsOutput>;

    train(params: TrainingParameters, callbacks: tf.CustomCallbackArgs): Promise<unknown>;

    addExample(className: number, image: HTMLCanvasElement | AudioExample): Promise<void>;

    setLabels(labels: string[]): void;

    dispose(): void;

    getLabels(): string[];

    getLabel(ix: number): string;

    getNumExamples(): number;

    getExamplesPerClass(): number[];

    getNumValidation(): number;

    calculateAccuracy(): Promise<{ reference: any; predictions: tf.Tensor }>;
}
