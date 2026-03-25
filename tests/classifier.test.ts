import { describe, it } from 'vitest';
import { createModel } from '../lib/main';
import ImageModel from '../lib/ImageModel';
import SpeechModel from '../lib/SpeechModel';

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

    it('can create a new speech classifier', { timeout: 10000 }, async ({ expect }) => {
        const classifier = createModel('speech');
        expect(classifier).toBeInstanceOf(SpeechModel);

        await classifier.ready();

        expect(classifier.isReady()).toBe(true);
        expect(classifier.getModel()).toBeTruthy();

        classifier.dispose();
    });
});
