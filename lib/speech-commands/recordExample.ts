import { BrowserFftFeatureExtractor, SpectrogramCallback } from './browser_fft_extractor';
import { normalize, normalizeFloat32Array } from './browser_fft_utils';
import { concatenateFloat32Arrays } from './generic_utils';
import { Example, ExampleCollectionOptions, SpectrogramData } from './types';
import * as tf from '@tensorflow/tfjs-core';

interface CollectionOptions extends ExampleCollectionOptions {
    frameSize: number;
    batchSize: number;
    fftSize: number;
    sampleRateHz: number;
}

export async function collectExample(word: string, options: CollectionOptions): Promise<Example> {
    tf.util.assert(
        word != null && typeof word === 'string' && word.length > 0,
        () => `Must provide a non-empty string when collecting transfer-` + `learning example`
    );

    if (options.durationMultiplier != null && options.durationSec != null) {
        throw new Error(`durationMultiplier and durationSec are mutually exclusive, ` + `but are both specified.`);
    }

    let numFramesPerSpectrogram: number;
    if (options.durationSec != null) {
        tf.util.assert(options.durationSec > 0, () => `Expected durationSec to be > 0, but got ${options.durationSec}`);
        const frameDurationSec = (options.fftSize ?? 0) / (options.sampleRateHz ?? 1);
        numFramesPerSpectrogram = Math.ceil(options.durationSec / frameDurationSec);
    } else if (options.durationMultiplier != null) {
        tf.util.assert(
            options.durationMultiplier >= 1,
            () => `Expected duration multiplier to be >= 1, ` + `but got ${options.durationMultiplier}`
        );
        numFramesPerSpectrogram = Math.round(options.batchSize * options.durationMultiplier);
    } else {
        numFramesPerSpectrogram = options.batchSize;
    }

    if (options.snippetDurationSec != null) {
        tf.util.assert(
            options.snippetDurationSec > 0,
            () => `snippetDurationSec is expected to be > 0, but got ` + `${options.snippetDurationSec}`
        );
        tf.util.assert(
            options.onSnippet != null,
            () => `onSnippet must be provided if snippetDurationSec ` + `is provided.`
        );
    }
    if (options.onSnippet != null) {
        tf.util.assert(
            options.snippetDurationSec != null,
            () => `snippetDurationSec must be provided if onSnippet ` + `is provided.`
        );
    }
    const frameDurationSec = (options.fftSize ?? 0) / (options.sampleRateHz ?? 1);
    const totalDurationSec = frameDurationSec * numFramesPerSpectrogram;

    return new Promise<Example>((resolve) => {
        const stepFactor = options.snippetDurationSec == null ? 1 : options.snippetDurationSec / totalDurationSec;
        const overlapFactor = 1 - stepFactor;
        const callbackCountTarget = Math.round(1 / stepFactor);
        let callbackCount = 0;
        let lastIndex = -1;
        const spectrogramSnippets: Float32Array[] = [];

        const spectrogramCallback: SpectrogramCallback = async (freqData: tf.Tensor, timeData?: tf.Tensor) => {
            // TODO(cais): can we consolidate the logic in the two branches?
            if (options.onSnippet == null) {
                const normalizedX = normalize(freqData);
                const example: Example = {
                    label: word,
                    spectrogram: {
                        data: (await normalizedX.data()) as Float32Array,
                        frameSize: options.frameSize,
                    },
                    rawAudio: options.includeRawAudio
                        ? {
                              data: (await timeData?.data()) as Float32Array,
                              sampleRateHz: audioDataExtractor.sampleRateHz,
                          }
                        : undefined,
                };
                normalizedX.dispose();
                await audioDataExtractor.stop();

                /*resolve({
                        data: (await freqData.data()) as Float32Array,
                        frameSize: options.frameSize,
                    });*/

                resolve(example);
            } else {
                const data = (await freqData.data()) as Float32Array;
                if (lastIndex === -1) {
                    lastIndex = data.length;
                }
                let i = lastIndex - 1;
                while (data[i] !== 0 && i >= 0) {
                    i--;
                }
                const increment = lastIndex - i - 1;
                lastIndex = i + 1;
                const snippetData = data.slice(data.length - increment, data.length);
                spectrogramSnippets.push(snippetData);

                if (options.onSnippet != null) {
                    options.onSnippet({ data: snippetData, frameSize: options.frameSize });
                }

                if (callbackCount++ === callbackCountTarget) {
                    await audioDataExtractor.stop();

                    const normalized = normalizeFloat32Array(concatenateFloat32Arrays(spectrogramSnippets));
                    const finalSpectrogram: SpectrogramData = {
                        data: normalized,
                        frameSize: options.frameSize,
                    };
                    const example: Example = {
                        label: word,
                        spectrogram: finalSpectrogram,
                        rawAudio: options.includeRawAudio
                            ? {
                                  data: (await timeData?.data()) as Float32Array,
                                  sampleRateHz: audioDataExtractor.sampleRateHz,
                              }
                            : undefined,
                    };
                    // TODO(cais): Fix 1-tensor memory leak.
                    resolve(example);
                }
            }
            return false;
        };
        const audioDataExtractor = new BrowserFftFeatureExtractor({
            sampleRateHz: options.sampleRateHz,
            numFramesPerSpectrogram,
            columnTruncateLength: options.frameSize,
            suppressionTimeMillis: 0,
            spectrogramCallback,
            overlapFactor,
            includeRawAudio: options.includeRawAudio,
        });
        audioDataExtractor.start(options.audioTrackConstraints);
    });
}
