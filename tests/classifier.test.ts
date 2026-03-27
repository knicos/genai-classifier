import { describe, it } from 'vitest';
import ClassifierApp, { AudioExample, createModel } from '../lib/main';
import ImageModel from '../lib/ImageModel';
import SpeechModel from '../lib/SpeechModel';
import PoseModel from '../lib/PoseModel';

describe('Can create a new classifier', () => {
    it('can create a new image classifier', { timeout: 10000 }, async ({ expect }) => {
        const classifier = createModel('image');
        expect(classifier).toBeInstanceOf(ImageModel);

        await classifier.ready();

        expect(classifier.isReady()).toBe(true);
        expect(classifier.getModel()).toBeTruthy();
        expect(classifier.getImageSize()).toBe(224);

        classifier.dispose();
    });

    it('can create a new pose classifier', { timeout: 10000 }, async ({ expect }) => {
        const classifier = createModel('pose');
        expect(classifier).toBeInstanceOf(PoseModel);

        await classifier.ready();

        expect(classifier.isReady()).toBe(true);
        expect(classifier.getModel()).toBeTruthy();
        expect(classifier.getImageSize()).toBe(257);

        classifier.dispose();
    });

    it('can create a new speech classifier', { timeout: 10000 }, async ({ expect }) => {
        const classifier = createModel('speech');
        expect(classifier).toBeInstanceOf(SpeechModel);

        await classifier.ready();

        expect(classifier.isReady()).toBe(true);
        expect(classifier.getModel()).toBeTruthy();

        classifier.dispose();
    });
});

function makeExampleCanvas(): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    return canvas;
}

describe('Can save and load a classifier', () => {
    it('can save and load an image classifier', { timeout: 20000 }, async ({ expect }) => {
        const classifier = await ClassifierApp.create('image');

        const data1 = makeExampleCanvas();
        const data2 = makeExampleCanvas();
        const data3 = makeExampleCanvas();
        const data4 = makeExampleCanvas();
        await classifier.train(
            ['red', 'blue'],
            [
                [
                    { data: data1, id: 'red-1' },
                    { data: data2, id: 'red-2' },
                ],
                [
                    { data: data3, id: 'blue-1' },
                    { data: data4, id: 'blue-2' },
                ],
            ],
            { epochs: 1, learningRate: 0.01, batchSize: 1 }
        );
        const modelData = await classifier.save();
        expect(modelData.size).toBeGreaterThan(0);

        const loadedClassifier = await ClassifierApp.load(modelData);
        expect(loadedClassifier?.model).toBeInstanceOf(ImageModel);
        expect(loadedClassifier.getLabels()).toEqual(['red', 'blue']);
    });

    it('can save and load a speech classifier', { timeout: 20000 }, async ({ expect }) => {
        const classifier = await ClassifierApp.create('speech');

        const data1: AudioExample = {
            spectrogram: {
                data: new Float32Array(232 * 43),
                frameSize: 232,
            },
            label: 'red1',
        };
        const data2: AudioExample = {
            spectrogram: {
                data: new Float32Array(232 * 43),
                frameSize: 232,
            },
            label: 'red2',
        };
        const data3: AudioExample = {
            spectrogram: {
                data: new Float32Array(232 * 43),
                frameSize: 232,
            },
            label: 'blue3',
        };
        const data4: AudioExample = {
            spectrogram: {
                data: new Float32Array(232 * 43),
                frameSize: 232,
            },
            label: 'blue4',
        };
        await classifier.train(
            ['red', 'blue'],
            [
                [
                    { data: data1, id: 'red-1' },
                    { data: data2, id: 'red-2' },
                ],
                [
                    { data: data3, id: 'blue-1' },
                    { data: data4, id: 'blue-2' },
                ],
            ],
            { epochs: 1, learningRate: 0.01, batchSize: 1 }
        );
        const modelData = await classifier.save();
        expect(modelData.size).toBeGreaterThan(0);

        const loadedClassifier = await ClassifierApp.load(modelData);
        expect(loadedClassifier?.model).toBeInstanceOf(SpeechModel);
        expect(loadedClassifier.getLabels()).toEqual(['blue', 'red']);
    });
});
