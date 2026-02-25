import { TeachableMobileNet } from './gtm-image';
import { TeachablePoseNet } from './gtm-pose';
import * as tf from '@tensorflow/tfjs';
import { cropTo } from './gtm-utils/canvas';

export function cropTensor(img: tf.Tensor3D): tf.Tensor3D {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - size / 2;
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - size / 2;

    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

function capture(rasterElement: HTMLCanvasElement) {
    return tf.tidy(() => {
        const pixels = tf.browser.fromPixels(rasterElement);

        // crop the image so we're using the center square
        const cropped = cropTensor(pixels);

        // Expand the outer most dimension so we have a batch size of 1
        const batchedImage = cropped.expandDims(0);

        // Normalize the image between -1 and a1. The image comes in between 0-255
        // so we divide by 127 and subtract 1.
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
}

export class CAM {
    private model: TeachableMobileNet | TeachablePoseNet;
    private modelType: 'image' | 'pose';
    private activationModel!: tf.Sequential;
    private exposedMobileNet?: tf.LayersModel;
    private selectedIndex: number | null = null;
    private _disposed = false;

    constructor(model: TeachableMobileNet | TeachablePoseNet) {
        this.model = model;
        
        // Detect model type
        this.modelType = 'posenetModel' in model ? 'pose' : 'image';
        
        if (this.modelType === 'image') {
            this.initializeImageModel(model as TeachableMobileNet);
        } else {
            this.initializePoseModel(model as TeachablePoseNet);
        }
    }

    private initializeImageModel(mobileNet: TeachableMobileNet) {
         
        const tModel = (mobileNet.model.layers[1] as any).model as tf.Sequential;

        // Clone the dense layer weights so the activationModel owns its tensors
        const imgDense1Weights = (tModel.layers[1].getWeights() || []).map((w) => w.clone());
        const imgDense2Weights = (tModel.layers[2].getWeights() || []).map((w) => w.clone());

        this.activationModel = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [1280],
                    units: 100,
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling',
                    useBias: true,
                    weights: imgDense1Weights,
                }),
                tf.layers.dense({
                    kernelInitializer: 'varianceScaling',
                    useBias: false,
                    activation: 'relu', // 'softmax',
                    units: tModel.outputs[0].shape[1] || 1,
                    weights: imgDense2Weights,
                }),
            ],
        });

        //const model = this.mobileNet.truncatedModel.layers[0] as tf.Sequential;
        const model = (mobileNet.model.layers[0] as tf.Sequential).layers[0] as tf.Sequential;
        //const conv1bn = model.getLayer('out_relu');

        this.exposedMobileNet = tf.model({
            inputs: mobileNet.model.input,
            outputs: [
                model.output as tf.SymbolicTensor,
                ((mobileNet.model.layers[0] as tf.Sequential).layers[1] as tf.Sequential)
                    .output as tf.SymbolicTensor,
            ],
        });
    }

    private initializePoseModel(poseNet: TeachablePoseNet) {
        const tModel = poseNet.model;
        
        // The pose model has: Dense (relu) -> Dropout -> Dense (softmax)
        // We need the weights from the dense layers
        const denseLayer1 = tModel.layers[0] as tf.layers.Layer;
        const denseLayer2 = tModel.layers[2] as tf.layers.Layer; // Skip dropout layer

        // Clone the dense layer weights so the activationModel owns its tensors
        const poseDense1Weights = (denseLayer1.getWeights() || []).map((w) => w.clone());
        const poseDense2Weights = (denseLayer2.getWeights() || []).map((w) => w.clone());

        this.activationModel = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: poseDense1Weights[0].shape[0] as any,
                    units: poseDense1Weights[0].shape[1] as number,
                    activation: 'relu',
                    useBias: true,
                    weights: poseDense1Weights,
                }),
                tf.layers.dense({
                    units: poseDense2Weights[0].shape[1] as number,
                    activation: 'relu',
                    useBias: false, // Second layer has no bias in the trained model
                    weights: poseDense2Weights,
                }),
            ],
        });

        // For pose models, we'll use the baseModel.predict() to get CNN features
        // The heatmap output contains spatial feature information from the CNN layers
        // We store a reference to use during CAM calculation
        // Note: exposedMobileNet will remain undefined, we'll use baseModel directly
    }

    public dispose() {
        this._disposed = true;
        // activationModel owns cloned weights — safe to dispose independently
        this.activationModel?.dispose();
        // DO NOT dispose exposedMobileNet: it shares layers with the main model.
        // The main model's dispose() handles those layers.
        this.exposedMobileNet = undefined;
    }

    public isDisposed(): boolean {
        return this._disposed;
    }

    public setSelectedIndex(index: number | null) {
        this.selectedIndex = index;
    }

    private async mapTensorToPredictions(data: tf.Tensor) {
        if (this._disposed) return [];
        let features: tf.Tensor;
        try {
            // Use activationModel for both image and pose: it owns cloned weights
            // and is independent from the main model lifecycle
            features = tf.tidy(() => {
                if (this._disposed) throw new Error('disposed');
                const classes = this.activationModel.predict(data);
                if (Array.isArray(classes)) throw new Error('unexpected_array');
                return classes.softmax();
            });
        } catch (error) {
            // Disposal race: silent. Any other error: log it.
            if (error instanceof Error && error.message !== 'disposed') {
                console.warn('CAM prediction failed:', error);
            }
            return [];
        }
        const values = await features.data();
        features.dispose();
        if (this._disposed) return [];
        const labels = this.model.getMetadata().labels;
        if (!labels || labels.length === 0) {
            console.warn('CAM: model has no labels defined');
            return [];
        }
        const classes = [];
        for (let i = 0; i < Math.min(values.length, labels.length); i++) {
            classes.push({ className: labels[i], probability: values[i] });
        }
        return classes;
    }

    private calculateActivations(finalConv: tf.Tensor, classIndex: number) {
        if (this._disposed) throw new Error('disposed');
        return tf.tidy(() => {
            // 1. Reshape to a batch of features instead of GAP
            const batchFeatures = finalConv.reshape([7 * 7, 1280]);
            // 2. Get the linear class predictions for each convolution
            const convFeatures = this.activationModel.predict(batchFeatures);
            if (Array.isArray(convFeatures)) throw new Error('unexpected_array');

            const classFeatures = convFeatures.gather([classIndex], 1);
            const min = classFeatures.min();
            const max = classFeatures.max().sub(min);
            const normDist = classFeatures.sub(min).div(max);
            // Extract the selected class and reshape back to the input dimensions
            const cam = normDist.reshape([7, 7, 1]);
            return cam;
        });
    }

    /**
     * Calculate importance score for each keypoint using class activation mapping
     * Returns array of 17 normalized importance values (one per keypoint)
     */
    private async calculateKeypointImportance(classIndex: number, poseInput: Float32Array): Promise<Float32Array> {
        if (this._disposed) throw new Error('Model is disposed');
        return tf.tidy(() => {
            // Get the weights from both dense layers
            const layer1Weights = this.activationModel.layers[0].getWeights()[0]; // [34, 50]
            const layer2Weights = this.activationModel.layers[1].getWeights()[0]; // [50, numClasses]
            
            // Get weights for the target class [50, 1]
            const classWeights = layer2Weights.slice([0, classIndex], [-1, 1]);
            
            // For each keypoint (x,y pair), calculate its total influence on this class
            const keypointImportances = [];
            
            for (let i = 0; i < 17; i++) {
                // Get first layer weights for this keypoint's x and y coordinates
                const xWeights = layer1Weights.slice([i * 2, 0], [1, -1]); // [1, 50]
                const yWeights = layer1Weights.slice([i * 2 + 1, 0], [1, -1]); // [1, 50]
                
                // Get the actual input values
                const xValue = poseInput[i * 2];
                const yValue = poseInput[i * 2 + 1];
                
                // Calculate weighted contribution: input * first_layer_weights * class_weights
                const xContribution = tf.mul(tf.mul(xWeights, xValue), classWeights.transpose());
                const yContribution = tf.mul(tf.mul(yWeights, yValue), classWeights.transpose());
                
                // Sum the absolute contributions
                const totalContribution = tf.add(
                    tf.abs(xContribution.sum()),
                    tf.abs(yContribution.sum())
                );
                
                keypointImportances.push(totalContribution);
            }
            
            // Stack and normalize
            const stacked = tf.stack(keypointImportances);
            const min = stacked.min();
            const max = stacked.max();
            const range = max.sub(min);
            
            // Normalize to [0, 1]
            const normalized = stacked.sub(min).div(range.add(1e-7));
            
            return normalized.dataSync() as Float32Array;
        });
    }

    /**
     * Calculate CAM from PoseNet heatmap scores for pose model
     * heatmapScores shape: [height, width, 17] where 17 is the number of keypoints
     */
    private calculatePoseActivations(heatmapScores: tf.Tensor3D, classIndex: number): tf.Tensor {
        if (this._disposed) throw new Error('Model is disposed');
        return tf.tidy(() => {
            // Get dense layer weights to determine keypoint importance
            const firstLayerWeights = this.activationModel.layers[0].getWeights()[0];
            const secondLayerWeights = this.activationModel.layers[1].getWeights()[0];
            const classWeights = secondLayerWeights.slice([0, classIndex], [-1, 1]).squeeze();
            
            // Calculate importance score for each keypoint
            // by seeing how much each keypoint (x,y) contributes to this class
            const keypointImportances = [];
            for (let i = 0; i < 17; i++) {
                // Get weights connecting keypoint coordinates to hidden layer
                const xWeights = firstLayerWeights.slice([i * 2, 0], [1, -1]).squeeze();
                const yWeights = firstLayerWeights.slice([i * 2 + 1, 0], [1, -1]).squeeze();
                
                // Weight by class-specific importance (second layer)
                const xContribution = tf.mul(xWeights, classWeights);
                const yContribution = tf.mul(yWeights, classWeights);
                
                // Sum contributions and take absolute value
                const totalContribution = tf.add(
                    tf.abs(xContribution.sum()), 
                    tf.abs(yContribution.sum())
                );
                
                keypointImportances.push(totalContribution);
            }
            
            // Stack and normalize keypoint importance weights [17]
            const keypointWeights = tf.stack(keypointImportances);
            const normalizedWeights = tf.softmax(keypointWeights);
            
            // Weight heatmap channels by keypoint importance
            // High values = important keypoint detected here
            const weighted = tf.mul(heatmapScores, normalizedWeights.reshape([1, 1, 17]));
            
            // Sum across keypoint channels
            const cam = weighted.sum(-1);
            
            // Apply ReLU-like behavior - only keep positive activations
            const positive = tf.maximum(cam, 0);
            
            // Normalize to [0, 1]
            const max = positive.max();
            const normalized = positive.div(max.add(1e-7));
            
            return normalized;
        });
    }

    public async createCAM(image: HTMLCanvasElement) {
        if (this.modelType === 'image') {
            return this.createImageCAM(image);
        } else {
            throw new Error('Use createPoseCAM for pose models');
        }
    }

    private async createImageCAM(image: HTMLCanvasElement) {
        if (this._disposed || !this.exposedMobileNet) throw new Error('exposedMobileNet not initialized');

        const emptyHeatmap = Array(image.height).fill(0).map(() => Array(image.width).fill(0));
        try {
            const mobileNet = this.model as TeachableMobileNet;
            const croppedImage = cropTo(image, mobileNet.getMetadata().imageSize || 0, false);
            const imageTensor = capture(croppedImage);
            if (this._disposed) { imageTensor.dispose(); throw new Error('disposed'); }
            const layerOutputs = this.exposedMobileNet.predict(imageTensor);
            imageTensor.dispose();
            if (!Array.isArray(layerOutputs)) throw new Error('not_array');

            const predictions = await this.mapTensorToPredictions(layerOutputs[1]);
            if (this._disposed || predictions.length === 0) {
                layerOutputs.forEach((o) => o.dispose());
                return { predictions: [], classIndex: 0, heatmapData: emptyHeatmap };
            }

            // Get best index 
            const nameOfMax = predictions.reduce((prev, val) => (val.probability > prev.probability ? val : prev));
            const ix = predictions.indexOf(nameOfMax);
            const cam = this.calculateActivations(layerOutputs[0], this.selectedIndex !== null ? this.selectedIndex : ix);
            const maxProbability = predictions[ix].probability;
            const normPredictions =
                this.selectedIndex === null
                    ? predictions
                    : predictions.map((p) => ({
                          className: p.className,
                          probability: p.probability / maxProbability,
                      }));

            layerOutputs.forEach((output) => output.dispose());

            if (this._disposed) { cam.dispose(); return { predictions: [], classIndex: 0, heatmapData: emptyHeatmap }; }

            const resized = tf.tidy(() => {
                const finalSum = cam.resizeBilinear([image.width, image.height], false, true);
                const final = finalSum.mul(
                    normPredictions[this.selectedIndex !== null ? this.selectedIndex : ix].probability
                );
                return final.reshape([image.width, image.height]);
            });
            cam.dispose();

            const finalData = (await resized.array()) as number[][];
            resized.dispose();
            return { predictions, classIndex: ix, heatmapData: finalData };
        } catch (error) {
            if (!this._disposed) {
                console.warn('CAM (image) generation failed:', error);
            }
            return { predictions: [], classIndex: 0, heatmapData: emptyHeatmap };
        }
    }

    /**
     * Create CAM explanation for a pose classification using CNN feature maps
     * @param image The input image (used to extract CNN features from PoseNet backbone)
     * @param poseOutput The pose output from PoseNet (17 keypoints x 2 = 34 values)
     */
    public async createPoseCAM(image: HTMLCanvasElement, poseOutput: Float32Array) {
        if (this._disposed) return { predictions: [], classIndex: 0, heatmapData: [] as number[][] };
        if (this.modelType !== 'pose') throw new Error('Use createCAM for image models');

        const emptyHeatmap = Array(image.height).fill(0).map(() => Array(image.width).fill(0));
        const poseNet = this.model as TeachablePoseNet;

        // Step 1: get predictions via activationModel (owns cloned weights, safe after dispose)
        let predictions: Awaited<ReturnType<typeof this.mapTensorToPredictions>>;
        try {
            const inputTensor = tf.tensor2d([Array.from(poseOutput)]);
            predictions = await this.mapTensorToPredictions(inputTensor);
            inputTensor.dispose();
        } catch {
            return { predictions: [], classIndex: 0, heatmapData: emptyHeatmap };
        }

        // Guard: dispose() may have fired during the await above
        if (this._disposed || !predictions || predictions.length === 0) {
            return { predictions: [], classIndex: 0, heatmapData: emptyHeatmap };
        }

        const nameOfMax = predictions.reduce((prev, val) => (val.probability > prev.probability ? val : prev));
        const ix = predictions.indexOf(nameOfMax);
        const targetClass = this.selectedIndex !== null ? this.selectedIndex : ix;

        // Step 2: CNN feature extraction — silently bail on any disposal error
        try {
            // Guard: dispose() may have fired during the predictions await
            if (this._disposed) return { predictions, classIndex: targetClass, heatmapData: emptyHeatmap };

            const inputResolution = poseNet.posenetModel.inputResolution;
            const resWidth = Array.isArray(inputResolution) ? inputResolution[0] : inputResolution;
            const resHeight = Array.isArray(inputResolution) ? inputResolution[1] : inputResolution;

            const resizedCanvas = document.createElement('canvas');
            resizedCanvas.width = resWidth;
            resizedCanvas.height = resHeight;
            const ctx = resizedCanvas.getContext('2d');
            if (!ctx) throw new Error('canvas_context');
            ctx.drawImage(image, 0, 0, resWidth, resHeight);

            const imageTensor = tf.browser.fromPixels(resizedCanvas);
            const preprocessed = poseNet.posenetModel.baseModel.preprocessInput(tf.cast(imageTensor, 'float32') as tf.Tensor3D);
            const { heatmapScores } = poseNet.posenetModel.baseModel.predict(preprocessed);
            imageTensor.dispose();
            preprocessed.dispose();

            if (this._disposed) { heatmapScores.dispose(); return { predictions, classIndex: targetClass, heatmapData: emptyHeatmap }; }

            const cam = this.calculatePoseActivations(heatmapScores, targetClass);
            heatmapScores.dispose();

            const maxProbability = predictions[ix].probability;
            const normPredictions =
                this.selectedIndex === null
                    ? predictions
                    : predictions.map((p) => ({
                          className: p.className,
                          probability: p.probability / maxProbability,
                      }));

            const resized = tf.tidy(() => {
                const cam3d = cam.expandDims(2) as tf.Tensor3D;
                const finalSum = cam3d.resizeBilinear([image.height, image.width], false, true);
                const final = finalSum.mul(normPredictions[targetClass].probability);
                return final.reshape([image.height, image.width]);
            });
            cam.dispose();

            const finalData = (await resized.array()) as number[][];
            resized.dispose();

            // Guard after the last await
            if (this._disposed) return { predictions, classIndex: targetClass, heatmapData: emptyHeatmap };

            const keypointImportance = await this.calculateKeypointImportance(targetClass, poseOutput);

            return { predictions: normPredictions, classIndex: targetClass, heatmapData: finalData, keypointImportance };
        } catch (error) {
            // Disposal during model switch: silent. Genuine failure: warn.
            if (!this._disposed) {
                console.warn('CAM (pose) generation failed:', error);
            }
            return { predictions, classIndex: targetClass, heatmapData: emptyHeatmap };
        }
    }
}

