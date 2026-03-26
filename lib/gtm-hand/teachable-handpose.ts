/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import { util } from '@tensorflow/tfjs';
import type { TensorContainer } from '@tensorflow/tfjs-core/dist/tensor_types';
import type { CustomCallbackArgs } from '@tensorflow/tfjs';
import type { Initializer } from '@tensorflow/tfjs-layers/dist/initializers';
import seedrandom from 'seedrandom';

import { CustomHandPose, type Metadata, MULTI_HAND_FEATURE_SIZE, loadHandDetector } from './custom-handpose';

const VALIDATION_FRACTION = 0.15;

export interface TrainingParameters {
    denseUnits: number;
    epochs: number;
    learningRate: number;
    batchSize: number;
}

interface Sample {
    data: Float32Array;
    label: number[];
}

/**
 * One-hot encode an integer label into an array of length `numClasses`.
 */
function flatOneHot(label: number, numClasses: number): number[] {
    const labelOneHot = new Array(numClasses).fill(0) as number[];
    labelOneHot[label] = 1;
    return labelOneHot;
}

/**
 * Fisher-Yates shuffle. Optionally accepts a seeded PRNG for reproducibility.
 */
function fisherYates(array: Float32Array[] | Sample[], seed?: seedrandom.prng): Float32Array[] | Sample[] {
    const length = array.length;
    const shuffled = array.slice();
    for (let i = length - 1; i > 0; i -= 1) {
        const randomIndex = seed ? Math.floor(seed() * (i + 1)) : Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[randomIndex]] = [shuffled[randomIndex], shuffled[i]];
    }
    return shuffled;
}

export class TeachableHandPose extends CustomHandPose {
    private trainDataset!: tf.data.Dataset<TensorContainer>;
    private validationDataset!: tf.data.Dataset<TensorContainer>;
    private __stopTrainingResolve?: () => void;

    /** Array of training examples per class, each element is a class bucket */
    public examples: Float32Array[][] = [];

    private seed?: seedrandom.prng;

    /** True once the model has been trained (more than 2 layers built) */
    public get isTrained(): boolean {
        return !!this.model && this.model.layers && this.model.layers.length > 2;
    }

    public get isPrepared(): boolean {
        return !!this.trainDataset;
    }

    public get numClasses(): number {
        return this._metadata.labels.length;
    }

    /**
     * Add a sample of data under the provided class index.
     * @param className Integer index for the class this example belongs to
     * @param sample Feature vector from `handOutputsToArray()`
     */
    public addExample(className: number, sample: Float32Array): void {
        this.examples[className].push(sample);
    }

    /**
     * Classify a hand feature vector using the trained model.
     * Returns full probability distribution across all classes.
     */
    public async predict(handOutput: Float32Array) {
        if (!this.model) {
            throw new Error('Model has not been trained yet, call train() first');
        }
        return super.predict(handOutput);
    }

    /**
     * Classify a hand feature vector and return the top-K predictions.
     */
    public async predictTopK(handOutput: Float32Array, maxPredictions = 3) {
        if (!this.model) {
            throw new Error('Model has not been trained yet, call train() first');
        }
        return super.predictTopK(handOutput, maxPredictions);
    }

    /**
     * Pre-process collected examples into train / validation tf.data.Datasets.
     * Must be called before `train()` or will be called automatically.
     */
    public prepare(): void {
        for (const cls of this.examples) {
            if (cls.length === 0) {
                throw new Error('Add some examples before training');
            }
        }
        const { trainDataset, validationDataset } = this.convertToTfDataset();
        this.trainDataset = trainDataset;
        this.validationDataset = validationDataset;
    }

    private convertToTfDataset() {
        // Shuffle each class individually
        for (let i = 0; i < this.examples.length; i++) {
            this.examples[i] = fisherYates(this.examples[i], this.seed) as Float32Array[];
        }

        let trainSamples: Sample[] = [];
        let valSamples: Sample[] = [];

        for (let i = 0; i < this.examples.length; i++) {
            const y = flatOneHot(i, this.numClasses);
            const classLength = this.examples[i].length;
            const numVal = Math.ceil(VALIDATION_FRACTION * classLength);
            const numTrain = classLength - numVal;

            const classTrain = this.examples[i].slice(0, numTrain).map((data) => ({ data, label: y }));
            const classVal = this.examples[i].slice(numTrain).map((data) => ({ data, label: y }));

            trainSamples = trainSamples.concat(classTrain);
            valSamples = valSamples.concat(classVal);
        }

        trainSamples = fisherYates(trainSamples, this.seed) as Sample[];
        valSamples = fisherYates(valSamples, this.seed) as Sample[];

        const trainDataset = tf.data.zip({
            xs: tf.data.array(trainSamples.map((s) => s.data)),
            ys: tf.data.array(trainSamples.map((s) => s.label)),
        });
        const validationDataset = tf.data.zip({
            xs: tf.data.array(valSamples.map((s) => s.data)),
            ys: tf.data.array(valSamples.map((s) => s.label)),
        });

        return { trainDataset, validationDataset };
    }

    /**
     * Build and train the Dense classifier head on the collected examples.
     *
     * Architecture:
     *   Dense(denseUnits, relu) → BatchNorm → Dropout(0.3)
     *   → Dense(denseUnits/2, relu) → BatchNorm → Dropout(0.2)
     *   → Dense(numClasses, softmax)
     *
     * @param params Training hyper-parameters
     * @param callbacks Keras-compatible training callbacks
     */
    public async train(params: TrainingParameters, callbacks: CustomCallbackArgs = {}) {
        const originalOnTrainEnd = callbacks.onTrainEnd || (() => {});
        callbacks.onTrainEnd = (logs: tf.Logs | undefined) => {
            if (this.__stopTrainingResolve) {
                this.__stopTrainingResolve();
                this.__stopTrainingResolve = undefined;
            }
            originalOnTrainEnd(logs);
        };

        if (!this.isPrepared) {
            this.prepare();
        }

        const numLabels = this.getLabels().length;
        util.assert(
            numLabels === this.numClasses,
            () => `Cannot train: has ${numLabels} labels but ${this.numClasses} classes`
        );

        let varianceScaling: Initializer;
        if (this.seed) {
            varianceScaling = tf.initializers.varianceScaling({ seed: 3.14 }) as Initializer;
        } else {
            varianceScaling = tf.initializers.varianceScaling({}) as Initializer;
        }

        this.model = tf.sequential({
            layers: [
                // Layer 1 — dense hidden layer
                tf.layers.dense({
                    inputShape: [MULTI_HAND_FEATURE_SIZE],
                    units: params.denseUnits,
                    activation: 'relu',
                    kernelInitializer: varianceScaling,
                    kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
                    useBias: true,
                }),
                // Layer 2 — stabilise hidden representation
                tf.layers.batchNormalization(),
                // Layer 3 — dropout for regularisation
                tf.layers.dropout({ rate: 0.3 }),
                // Layer 4 — second hidden layer
                tf.layers.dense({
                    units: Math.max(32, Math.floor(params.denseUnits / 2)),
                    activation: 'relu',
                    kernelInitializer: varianceScaling,
                    kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
                    useBias: true,
                }),
                // Layer 5 — stabilise hidden representation
                tf.layers.batchNormalization(),
                // Layer 6 — dropout for regularisation
                tf.layers.dropout({ rate: 0.2 }),
                // Layer 7 — softmax output
                tf.layers.dense({
                    units: this.numClasses,
                    kernelInitializer: varianceScaling,
                    useBias: false,
                    activation: 'softmax',
                }),
            ],
        });

        const optimizer = tf.train.adam(params.learningRate);
        this.model.compile({
            optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });

        if (!(params.batchSize > 0)) {
            throw new Error('Batch size is 0 or NaN. Please choose a non-zero value.');
        }

        const trainData = this.trainDataset.batch(params.batchSize);
        const validationData = this.validationDataset.batch(params.batchSize);

        await this.model.fitDataset(trainData, {
            epochs: params.epochs,
            validationData,
            callbacks,
        });

        optimizer.dispose();
        return this.model;
    }

    /**
     * Save the trained model to a given IO handler or URL.
     */
    public async save(handlerOrURL: tf.io.IOHandler | string, config?: tf.io.SaveConfig): Promise<tf.io.SaveResult> {
        return this.model.save(handlerOrURL, config);
    }

    /** Initialise the examples array (one bucket per class). Must be called after setLabels(). */
    public prepareDataset(): void {
        for (let i = 0; i < this.numClasses; i++) {
            this.examples[i] = [];
        }
    }

    public stopTraining(): Promise<void> {
        return new Promise<void>((resolve) => {
            this.model.stopTraining = true;
            this.__stopTrainingResolve = resolve;
        });
    }

    public dispose(): void {
        this.model.dispose();
        super.dispose();
    }

    public setLabel(index: number, label: string): void {
        this._metadata.labels[index] = label;
    }

    public setLabels(labels: string[]): void {
        this._metadata.labels = labels;
        this.prepareDataset();
    }

    public getLabel(index: number): string {
        return this._metadata.labels[index];
    }

    public getLabels(): string[] {
        return this._metadata.labels;
    }

    public setName(name: string): void {
        this._metadata.modelName = name;
    }

    public getName(): string | undefined {
        return this._metadata.modelName;
    }

    /** Optional seed for reproducible data shuffling. */
    public setSeed(seed: string): void {
        this.seed = seedrandom(seed);
    }

    /**
     * Calculate per-class accuracy on the held-out validation set.
     * Returns tensors with predicted and ground-truth class indices.
     */
    public async calculateAccuracyPerClass() {
        const validationXs = this.validationDataset.mapAsync(async (dataset: TensorContainer) => {
            return (dataset as { xs: TensorContainer; ys: TensorContainer }).xs;
        });
        const validationYs = this.validationDataset.mapAsync(async (dataset: TensorContainer) => {
            return (dataset as { xs: TensorContainer; ys: TensorContainer }).ys;
        });

        const batchSize = Math.min(validationYs.size, 32);
        const iterations = Math.ceil(validationYs.size / batchSize);

        const batchesX = validationXs.batch(batchSize);
        const batchesY = validationYs.batch(batchSize);
        const itX = await batchesX.iterator();
        const itY = await batchesY.iterator();
        const allX: tf.Tensor[] = [];
        const allY: tf.Tensor[] = [];

        for (let i = 0; i < iterations; i++) {
            const batchedXTensor = await itX.next();
            const batchedXPredictionTensor = this.model.predict(batchedXTensor.value as tf.Tensor) as tf.Tensor;
            allX.push(batchedXPredictionTensor.argMax(1));

            const batchedYTensor = await itY.next();
            allY.push((batchedYTensor.value as tf.Tensor).argMax(1));

            (batchedXTensor.value as tf.Tensor).dispose();
            batchedXPredictionTensor.dispose();
            (batchedYTensor.value as tf.Tensor).dispose();
        }

        const predictions = tf.concat(allX);
        const reference = tf.concat(allY);

        if (iterations !== 1) {
            for (let i = 0; i < allX.length; i++) {
                allX[i].dispose();
                allY[i].dispose();
            }
        }

        return { reference, predictions };
    }
}

/**
 * Create a new TeachableHandPose instance ready for training.
 * @param metadata Partial metadata (labels can be added later via setLabels)
 */
export async function createTeachable(metadata: Partial<Metadata>): Promise<TeachableHandPose> {
    const handDetector = await loadHandDetector(metadata.modelSettings);
    return new TeachableHandPose(tf.sequential(), handDetector, metadata);
}
