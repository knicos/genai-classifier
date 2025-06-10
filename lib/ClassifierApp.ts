import { BehaviourType } from './behaviours';
import TeachableModel, { ExplainedPredictionsOutput, TMType } from './TeachableModel';
import * as tf from '@tensorflow/tfjs';
import JSZip from 'jszip';
import EE from 'eventemitter3';

export interface ISample {
    data: HTMLCanvasElement;
    id: string;
}

export interface IClassification {
    label: string;
    samples: ISample[];
}

interface ProjectTemp {
    modelJson?: string;
    modelWeights?: ArrayBuffer;
    metadata?: string;
    behaviours?: string;
    samples: string[][];
}

interface ModelContents {
    behaviours?: string;
    zip?: Blob;
    model?: string;
    metadata?: string;
    weights?: ArrayBuffer;
}

export interface TrainingSettings {
    epochs: number;
    learningRate: number;
    batchSize: number;
}

export interface PredictionsOutput extends ExplainedPredictionsOutput {
    nameOfMax: string;
    indexOfMax: number;
    failed?: boolean;
}

type ClassifierAppEvents = 'loading' | 'ready' | 'epoch' | 'training' | 'trainingcomplete' | 'error' | 'action';

export default class ClassifierApp extends EE<ClassifierAppEvents> {
    public model?: TeachableModel;
    public behaviours: BehaviourType[];
    public samples: ISample[][];
    public projectId?: string;
    public readonly variant: TMType;

    constructor(variant: TMType, model?: TeachableModel, behaviours?: BehaviourType[], samples?: ISample[][]) {
        super();
        this.variant = variant;
        this.model = model;
        this.behaviours = behaviours || [];
        this.samples = samples || [];
    }

    public getImageSize() {
        return this.model?.getImageSize() || 224;
    }

    public isReady() {
        return !!this.model && this.model.isReady();
    }

    public draw(image: HTMLCanvasElement) {
        if (this.model) {
            this.model.draw(image);
        }
    }

    public async estimate(image: HTMLCanvasElement): Promise<void> {
        if (this.model) {
            await this.model.estimate(image);
        }
    }

    public setXAICanvas(image: HTMLCanvasElement) {
        if (this.model) {
            this.model.setXAICanvas(image);
        }
    }

    public setXAIClass(className: string | number | null) {
        if (this.model) {
            this.model.setXAIClass(className);
        }
    }

    public async predict(image: HTMLCanvasElement): Promise<PredictionsOutput> {
        if (this.model) {
            const predictions = await this.model.predict(image);

            const nameOfMax = predictions.predictions.reduce((prev, val) =>
                val.probability > prev.probability ? val : prev
            );
            const indexOfMax = predictions.predictions.indexOf(nameOfMax);

            return { ...predictions, nameOfMax: nameOfMax.className, indexOfMax, failed: false };
        }
        return { predictions: [], nameOfMax: '', indexOfMax: -1, failed: true };
    }

    public getLabels(): string[] {
        if (this.model) {
            return this.model.getLabels();
        }
        return [];
    }

    public getLabel(index: number): string {
        if (this.model) {
            return this.model.getLabel(index);
        }
        return '';
    }

    public async train(labels: string[], samples: ISample[][], settings: TrainingSettings) {
        if (this.model) {
            this.model.dispose();
        }

        this.emit('training');

        const newModel = new TeachableModel(this.variant);

        this.emit('loading');
        await newModel.ready();

        newModel.setLabels(labels);

        samples.forEach((s, ix) => {
            s.forEach((ss) => {
                newModel.addExample(ix, ss.data);
            });
        });

        try {
            await newModel.train(
                {
                    denseUnits: 100,
                    epochs: settings.epochs,
                    learningRate: settings.learningRate,
                    batchSize: settings.batchSize,
                },
                {
                    onEpochEnd: (epoch) => {
                        this.emit('epoch', epoch);
                    },
                }
            );
        } catch (e) {
            console.error('Training failed', e);
            this.emit('error', e);
            return;
        }

        this.model = newModel;
        this.emit('trainingcomplete');
        this.emit('ready');
        return newModel;
    }

    static async create(variant: TMType) {
        return new ClassifierApp(variant);
    }

    public async save(): Promise<Blob> {
        const zip = new JSZip();
        if (this.samples.length > 0) {
            const folder = zip.folder('samples');
            if (folder) {
                for (let j = 0; j < this.samples.length; ++j) {
                    const s = this.samples[j];
                    for (let i = 0; i < s.length; ++i) {
                        const ss = s[i];
                        folder.file(`${j}_${i}.png`, ss.data.toDataURL('image/png').split(';base64,')[1], {
                            base64: true,
                        });
                    }
                }
            }
        }

        const contents: ModelContents = {};

        if (this.behaviours.length > 0) {
            contents.behaviours = JSON.stringify({
                behaviours: this.behaviours,
                version: 1,
            });
            zip.file('behaviours.json', contents.behaviours);
        }

        let zipData: Blob = new Blob();
        if (this.model) {
            contents.metadata = JSON.stringify({ ...this.model.getMetadata(), projectId: this.projectId });
            zip.file('metadata.json', contents.metadata);

            await this.model.save({
                save: async (artifact: tf.io.ModelArtifacts) => {
                    if (artifact.weightData && !Array.isArray(artifact.weightData)) {
                        contents.weights = artifact.weightData;
                        zip.file('weights.bin', artifact.weightData);
                    }
                    if (artifact.modelTopology) {
                        contents.model = JSON.stringify({
                            modelTopology: artifact.modelTopology,
                            weightsManifest: [{ paths: ['./weights.bin'], weights: artifact.weightSpecs }],
                        });
                        zip.file('model.json', contents.model);
                    }

                    zipData = await zip.generateAsync({ type: 'blob' });
                    return {
                        modelArtifactsInfo: {
                            dateSaved: new Date(),
                            modelTopologyType: 'JSON',
                        },
                    } as tf.io.SaveResult;
                },
            });
        } else {
            console.warn('No model to save');
        }

        contents.zip = zipData;
        return contents.zip;
    }

    static async load(file: string | Blob): Promise<ClassifierApp> {
        const project: ProjectTemp = {
            samples: [],
        };

        const blob = typeof file === 'string' ? await fetch(file).then((r) => r.blob()) : file;

        const zip = await JSZip.loadAsync(blob);
        const promises: Promise<void>[] = [];

        zip.forEach((_: string, data: JSZip.JSZipObject) => {
            if (data.name === 'model.json') {
                promises.push(
                    data.async('string').then((r) => {
                        project.modelJson = r;
                    })
                );
            } else if (data.name === 'weights.bin') {
                promises.push(
                    data.async('arraybuffer').then((r) => {
                        project.modelWeights = r;
                    })
                );
            } else if (data.name === 'behaviours.json') {
                promises.push(
                    data.async('string').then((r) => {
                        project.behaviours = r;
                    })
                );
            } else if (data.name === 'metadata.json') {
                promises.push(
                    data.async('string').then((r) => {
                        project.metadata = r;
                    })
                );
            } else {
                const parts = data.name.split('/');
                if (parts.length === 2 && !!parts[1] && parts[0] === 'samples') {
                    const split1 = parts[1].split('.');
                    if (split1.length === 2) {
                        const split2 = split1[0].split('_');
                        if (split2.length === 2) {
                            const ix1 = parseInt(split2[0]);
                            const ix2 = parseInt(split2[1]);
                            while (project.samples.length <= ix1) project.samples.push([]);
                            while (project.samples[ix1].length <= ix2) project.samples[ix1].push('');
                            promises.push(
                                data.async('base64').then((r) => {
                                    project.samples[ix1][ix2] = `data:image/png;base64,${r}`;
                                })
                            );
                        }
                    }
                }
            }
        });

        await Promise.all(promises);

        if (project.metadata && project.modelJson && project.modelWeights) {
            const meta = JSON.parse(project.metadata);

            const parsedModel = JSON.parse(project.modelJson) as tf.io.ModelJSON;

            const model = new TeachableModel('image', meta, parsedModel, project.modelWeights);
            await model.ready();

            const samplePromises: Promise<HTMLCanvasElement>[] = [];

            for (const item of project.samples) {
                for (const s of item) {
                    samplePromises.push(
                        new Promise((resolve) => {
                            const canvas = document.createElement('canvas');
                            canvas.width = 224;
                            canvas.height = 224;
                            canvas.style.width = '58px';
                            canvas.style.height = '58px';
                            const ctx = canvas.getContext('2d');
                            const img = new Image();
                            img.onload = () => {
                                ctx?.drawImage(img, 0, 0);
                                resolve(canvas);
                            };
                            img.src = s;
                        })
                    );
                }
            }

            const canvases = await Promise.all(samplePromises);

            const samples: ISample[][] = [];

            let base = 0;
            for (let i = 0; i < project.samples.length; ++i) {
                const newImage: HTMLCanvasElement[] = [];
                for (let j = 0; j < project.samples[i].length; ++j) {
                    newImage.push(canvases[base++]);
                }
                samples.push(newImage.map((i) => ({ data: i, id: '' })));
            }

            const tm = new TeachableModel('image', meta, parsedModel, project.modelWeights);

            await tm.ready();

            const app = new ClassifierApp(
                'image',
                tm,
                project.behaviours ? JSON.parse(project.behaviours).behaviours : [],
                samples
            );
            if (meta.projectId) {
                app.projectId = meta.projectId;
            }
            return app;
        }

        throw new Error('Invalid project file');
    }
}
