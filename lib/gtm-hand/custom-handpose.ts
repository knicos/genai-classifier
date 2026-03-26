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

import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';
import type {
    Hand,
    HandDetector,
    Keypoint,
    MediaPipeHandsTfjsModelConfig,
} from '@tensorflow-models/hand-pose-detection';
import * as tf from '@tensorflow/tfjs';
import { version } from './version';

/**
 * Number of keypoints returned by MediaPipe Hands (wrist + 5 fingers × 4 joints)
 */
export const NUM_HAND_KEYPOINTS = 21;

/**
 * Size of the feature vector: 21 keypoints × (x, y) normalised to [0, 1]
 */
export const HAND_FEATURE_SIZE = NUM_HAND_KEYPOINTS * 2;

/** Default input image size used by hand models (same as PoseNet) */
export const DEFAULT_IMAGE_SIZE = 257;

/**
 * The metadata to describe the model's creation,
 * includes the labels associated with the classes
 * and versioning information from training.
 */
export interface Metadata {
    tfjsVersion: string;
    tmVersion?: string;
    packageVersion: string;
    packageName: string;
    modelName?: string;
    timeStamp?: string;
    imageSize?: number;
    labels: string[];
    userMetadata?: unknown;
    modelSettings: Partial<HandModelSettings>;
}

export interface HandModelSettings {
    maxHands: number;
    modelType: 'lite' | 'full';
    detectorModelUrl?: string;
    landmarkModelUrl?: string;
}

export type ClassifierInputSource = handPoseDetection.HandDetectorInput;

export const MAX_HANDS = 2;
export const MULTI_HAND_FEATURE_SIZE = HAND_FEATURE_SIZE * MAX_HANDS + MAX_HANDS + 1;
const MAX_PREDICTIONS = 3;
const MODEL_TYPE = 'full';
const DEFAULT_FULL_DETECTOR_MODEL_URL = 'https://store.gen-ai.fi/tm/models/handpose_full_detector/model.json';
const DEFAULT_FULL_LANDMARK_MODEL_URL = 'https://store.gen-ai.fi/tm/models/handpose_full_landmark/model.json';
const DEFAULT_LITE_DETECTOR_MODEL_URL = 'https://store.gen-ai.fi/tm/models/handpose_lite_detector/model.json';
const DEFAULT_LITE_LANDMARK_MODEL_URL = 'https://store.gen-ai.fi/tm/models/handpose_lite_landmark/model.json';

const fillMetadata = (data: Partial<Metadata>): Metadata => {
    data.packageVersion = data.packageVersion || version;
    data.packageName = '@teachablemachine/hand';
    data.timeStamp = data.timeStamp || new Date().toISOString();
    data.userMetadata = data.userMetadata || {};
    data.modelName = data.modelName || 'untitled';
    data.imageSize = data.imageSize || DEFAULT_IMAGE_SIZE;
    data.labels = data.labels || [];
    data.modelSettings = fillConfig(data.modelSettings);
    return data as Metadata;
};

const isMetadata = (c: unknown): c is Metadata => !!c && typeof c === 'object' && Array.isArray((c as Metadata).labels);

const processMetadata = async (metadata: string | Metadata): Promise<Metadata> => {
    let metadataJSON: Metadata;
    if (typeof metadata === 'string') {
        const response = await fetch(metadata);
        metadataJSON = await response.json();
    } else if (isMetadata(metadata)) {
        metadataJSON = metadata;
    } else {
        throw new Error('Invalid Metadata provided');
    }
    return fillMetadata(metadataJSON);
};

const fillConfig = (config: Partial<HandModelSettings> = {}): HandModelSettings => {
    const modelType = config.modelType ?? MODEL_TYPE;
    const fullDetectorModelUrl = config.detectorModelUrl || DEFAULT_FULL_DETECTOR_MODEL_URL;
    const fullLandmarkModelUrl = config.landmarkModelUrl || DEFAULT_FULL_LANDMARK_MODEL_URL;
    const liteDetectorModelUrl = config.detectorModelUrl || DEFAULT_LITE_DETECTOR_MODEL_URL;
    const liteLandmarkModelUrl = config.landmarkModelUrl || DEFAULT_LITE_LANDMARK_MODEL_URL;

    return {
        maxHands: config.maxHands ?? MAX_HANDS,
        modelType,
        detectorModelUrl: modelType === 'lite' ? liteDetectorModelUrl : fullDetectorModelUrl,
        landmarkModelUrl: modelType === 'lite' ? liteLandmarkModelUrl : fullLandmarkModelUrl,
    };
};

const errorDetails = (error: unknown) => {
    if (error instanceof Error) {
        return {
            name: error.name,
            message: error.message,
            stack: error.stack,
        };
    }
    return { error };
};

/**
 * Computes the probabilities of the top K classes given logits by sorting
 * the softmax probabilities.
 */
export async function getTopKClasses(labels: string[], logits: tf.Tensor<tf.Rank>, topK = MAX_PREDICTIONS) {
    const values = await logits.data();
    return tf.tidy(() => {
        topK = Math.min(topK, values.length);
        const valuesAndIndices: { value: number; index: number }[] = [];
        for (let i = 0; i < values.length; i++) {
            valuesAndIndices.push({ value: values[i], index: i });
        }
        valuesAndIndices.sort((a, b) => b.value - a.value);

        const topkValues = new Float32Array(topK);
        const topkIndices = new Int32Array(topK);
        for (let i = 0; i < topK; i++) {
            topkValues[i] = valuesAndIndices[i].value;
            topkIndices[i] = valuesAndIndices[i].index;
        }

        const topClassesAndProbs = [];
        for (let i = 0; i < topkIndices.length; i++) {
            topClassesAndProbs.push({
                className: labels[topkIndices[i]],
                probability: topkValues[i],
            });
        }
        return topClassesAndProbs;
    });
}

/**
 * Get image pixel dimensions from any supported input type.
 * Returns [width, height].
 */
function getInputDimensions(input: ClassifierInputSource): [number, number] {
    if (input instanceof HTMLCanvasElement) {
        return [input.width, input.height];
    }
    if (input instanceof HTMLImageElement) {
        return [input.naturalWidth, input.naturalHeight];
    }
    if (input instanceof HTMLVideoElement) {
        return [input.videoWidth, input.videoHeight];
    }
    if (input instanceof ImageData) {
        return [input.width, input.height];
    }
    // tf.Tensor3D — shape is [height, width, channels]
    const t = input as tf.Tensor3D;
    return [t.shape[1], t.shape[0]];
}

export class CustomHandPose {
    protected _metadata: Metadata;

    constructor(
        public model: tf.LayersModel,
        public handDetector: HandDetector,
        metadata: Partial<Metadata>
    ) {
        this._metadata = fillMetadata(metadata);
    }

    public getMetadata() {
        return this._metadata;
    }

    public getLabels(): string[] {
        return this._metadata.labels;
    }

    public getLabel(index: number): string {
        return this._metadata.labels[index];
    }

    /**
     * Run hand detection on the input image.
     * Returns the first detected hand, its flattened feature array, and the
     * full list of detected hands.
     *
     * The feature array is 42 floats: 21 keypoints × (x/imageWidth, y/imageHeight).
     * All values are in [0, 1]. If no hand is detected the array is all zeros.
     *
     * @param image Input image compatible with MediaPipe Hands
     * @param flipHorizontal Whether to mirror the result (e.g. webcam stream)
     */
    public async estimateHand(
        image: ClassifierInputSource,
        flipHorizontal = false
    ): Promise<{
        hand: Hand | null;
        handOutput: Float32Array;
        allHands: Hand[];
        allHandOutputs: Float32Array[];
        jointHandOutput: Float32Array;
    }> {
        const staticImageMode = true;

        if (staticImageMode) {
            const detectorWithReset = this.handDetector as HandDetector & { reset?: () => void };
            if (typeof detectorWithReset.reset === 'function') {
                try {
                    detectorWithReset.reset();
                } catch (resetError) {
                    console.warn('[HandPose] Failed to reset detector state before static-image inference', resetError);
                }
            }
        }

        const detectedHands = await this.handDetector.estimateHands(image, { flipHorizontal, staticImageMode });
        const allHands = detectedHands
            .slice()
            .sort((left, right) => (left.keypoints[0]?.x ?? 0) - (right.keypoints[0]?.x ?? 0));
        const hand = allHands[0] ?? null;
        const [width, height] = getInputDimensions(image);
        const allHandOutputs = allHands.map((detectedHand) =>
            this.handOutputsToArray(detectedHand.keypoints, width, height)
        );
        const handOutput = hand ? allHandOutputs[0] : new Float32Array(HAND_FEATURE_SIZE);
        const jointHandOutput = this.handOutputsToJointArray(allHandOutputs);

        return { hand, handOutput, allHands, allHandOutputs, jointHandOutput };
    }

    public handOutputsToJointArray(allHandOutputs: Float32Array[]): Float32Array {
        const arr = new Float32Array(MULTI_HAND_FEATURE_SIZE);
        const handCount = Math.min(allHandOutputs.length, MAX_HANDS);

        for (let handIndex = 0; handIndex < handCount; handIndex++) {
            arr.set(allHandOutputs[handIndex], handIndex * HAND_FEATURE_SIZE);
        }

        const presenceOffset = HAND_FEATURE_SIZE * MAX_HANDS;
        for (let handIndex = 0; handIndex < MAX_HANDS; handIndex++) {
            arr[presenceOffset + handIndex] = handIndex < handCount ? 1 : 0;
        }

        arr[presenceOffset + MAX_HANDS] = handCount / MAX_HANDS;
        return arr;
    }

    /**
     * Flatten and normalise hand keypoints into a fixed-size Float32Array.
     * Each keypoint's x is divided by imageWidth, y by imageHeight → [0, 1].
     *
     * @param keypoints 21 keypoints from a detected hand
     * @param imageWidth Width of the source image in pixels
     * @param imageHeight Height of the source image in pixels
     */
    public handOutputsToArray(keypoints: Keypoint[], imageWidth: number, imageHeight: number): Float32Array {
        const arr = new Float32Array(HAND_FEATURE_SIZE);

        const referenceKeypoint = keypoints[0] || keypoints.find((kp) => !!kp);
        if (!referenceKeypoint) {
            return arr;
        }

        const centerX = referenceKeypoint.x;
        const centerY = referenceKeypoint.y;
        let scale = 0;

        let minX = Number.POSITIVE_INFINITY;
        let minY = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let maxY = Number.NEGATIVE_INFINITY;

        for (let i = 0; i < NUM_HAND_KEYPOINTS; i++) {
            const kp = keypoints[i];
            if (kp) {
                const deltaX = kp.x - centerX;
                const deltaY = kp.y - centerY;
                scale = Math.max(scale, Math.hypot(deltaX, deltaY));

                minX = Math.min(minX, kp.x);
                minY = Math.min(minY, kp.y);
                maxX = Math.max(maxX, kp.x);
                maxY = Math.max(maxY, kp.y);
            }
        }

        if (!Number.isFinite(scale) || scale < 1e-6) {
            const boxWidth = Number.isFinite(maxX - minX) ? maxX - minX : imageWidth;
            const boxHeight = Number.isFinite(maxY - minY) ? maxY - minY : imageHeight;
            scale = Math.hypot(Math.max(boxWidth, 1), Math.max(boxHeight, 1));
        }

        if (!Number.isFinite(scale) || scale < 1e-6) {
            scale = Math.max(Math.hypot(imageWidth, imageHeight), 1);
        }

        for (let i = 0; i < NUM_HAND_KEYPOINTS; i++) {
            const kp = keypoints[i];
            if (kp) {
                arr[i * 2] = (kp.x - centerX) / scale;
                arr[i * 2 + 1] = (kp.y - centerY) / scale;
            }
        }

        return arr;
    }

    /**
     * Given a hand feature vector, return the full probability distribution
     * across all classes.
     * @param handOutput Float32Array of size HAND_FEATURE_SIZE
     */
    public async predict(handOutput: Float32Array) {
        const embeddings = tf.tensor([Array.from(handOutput)]);
        const logits = this.model.predict(embeddings) as tf.Tensor;
        const values = await logits.data();

        const classes = [];
        for (let i = 0; i < values.length; i++) {
            classes.push({
                className: this._metadata.labels[i],
                probability: values[i],
            });
        }

        embeddings.dispose();
        logits.dispose();

        return classes;
    }

    /**
     * Given a hand feature vector, return the top-K class predictions.
     * @param handOutput Float32Array of size HAND_FEATURE_SIZE
     * @param maxPredictions Maximum number of top predictions (default 3)
     */
    public async predictTopK(handOutput: Float32Array, maxPredictions = MAX_PREDICTIONS) {
        const embeddings = tf.tensor([Array.from(handOutput)]);
        const logits = this.model.predict(embeddings) as tf.Tensor;
        const topKClasses = await getTopKClasses(this._metadata.labels, logits, maxPredictions);

        embeddings.dispose();
        logits.dispose();

        return topKClasses;
    }

    public dispose() {
        this.handDetector.dispose();
    }
}

/**
 * Create and initialise a MediaPipe Hands detector using the TFJS runtime.
 */
export async function loadHandDetector(config: Partial<HandModelSettings> = {}): Promise<HandDetector> {
    const settings = fillConfig(config);

    const detectorConfig: MediaPipeHandsTfjsModelConfig = {
        runtime: 'tfjs',
        maxHands: settings.maxHands,
        modelType: settings.modelType,
        detectorModelUrl: settings.detectorModelUrl,
        landmarkModelUrl: settings.landmarkModelUrl,
    };

    try {
        const detector = await handPoseDetection.createDetector(
            handPoseDetection.SupportedModels.MediaPipeHands,
            detectorConfig
        );
        return detector;
    } catch (primaryError) {
        console.error('[HandPose] Detector initialization failed', {
            error: errorDetails(primaryError),
            detectorModelUrl: detectorConfig.detectorModelUrl,
            landmarkModelUrl: detectorConfig.landmarkModelUrl,
        });

        if (detectorConfig.modelType === 'full') {
            const liteConfig: MediaPipeHandsTfjsModelConfig = {
                ...detectorConfig,
                modelType: 'lite',
                detectorModelUrl: DEFAULT_LITE_DETECTOR_MODEL_URL,
                landmarkModelUrl: DEFAULT_LITE_LANDMARK_MODEL_URL,
            };

            console.warn('[HandPose] Retrying detector initialization with lite model', {
                modelType: liteConfig.modelType,
                detectorModelUrl: liteConfig.detectorModelUrl,
                landmarkModelUrl: liteConfig.landmarkModelUrl,
            });

            try {
                const liteDetector = await handPoseDetection.createDetector(
                    handPoseDetection.SupportedModels.MediaPipeHands,
                    liteConfig
                );
                return liteDetector;
            } catch (fallbackError) {
                console.error('[HandPose] Lite fallback detector initialization failed', {
                    error: errorDetails(fallbackError),
                    detectorModelUrl: liteConfig.detectorModelUrl,
                    landmarkModelUrl: liteConfig.landmarkModelUrl,
                });
            }
        }

        throw primaryError;
    }
}

/**
 * Load a trained CustomHandPose model from a URL checkpoint.
 * @param checkpoint URL of the model.json file
 * @param metadata Optional metadata URL or Metadata object
 */
export async function load(checkpoint: string, metadata?: string | Metadata): Promise<CustomHandPose> {
    const customModel = await tf.loadLayersModel(checkpoint);
    const metadataJSON = metadata ? await processMetadata(metadata) : null;
    const handDetector = await loadHandDetector(metadataJSON?.modelSettings);
    return new CustomHandPose(customModel, handDetector, metadataJSON || {});
}

/**
 * Load a trained CustomHandPose model from local File objects (browser).
 * @param json model.json File
 * @param weights model.weights.bin File
 * @param metadata metadata.json File
 */
export async function loadFromFiles(json: File, weights: File, metadata: File): Promise<CustomHandPose> {
    const customModel = await tf.loadLayersModel(tf.io.browserFiles([json, weights]));
    const metadataFile = await new Response(metadata).json();
    const metadataJSON = metadataFile ? await processMetadata(metadataFile) : null;
    const handDetector = await loadHandDetector(metadataJSON?.modelSettings);
    return new CustomHandPose(customModel, handDetector, metadataJSON || {});
}
