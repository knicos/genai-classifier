import { describe, it } from 'vitest';
import { TeachableModel } from '../lib/main';

describe('Can create a new classifier', () => {
    it('can create a new image classifier', { timeout: 10000 }, async ({ expect }) => {
        const classifier = new TeachableModel('image');
        expect(classifier).toBeInstanceOf(TeachableModel);

        await classifier.ready();

        expect(classifier.isReady()).toBe(true);
        expect(classifier.getImageModel()).toBeTruthy;
        expect(classifier.getImageSize()).toBe(224);

        classifier.dispose();
    });
});
