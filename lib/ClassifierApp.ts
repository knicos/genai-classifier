import { BehaviourType } from './behaviours';
import { TeachableModel, ExplainedPredictionsOutput, TMType } from './TeachableModel';
import * as tf from '@tensorflow/tfjs';
import JSZip from 'jszip';
import EE from 'eventemitter3';
import ImageModel from './ImageModel';
import PoseModel from './PoseModel';
import SoundModel from './SpeechModel';
import HandModel from './HandModel';
import { AudioExample } from './gtm-utils/recorder';

export interface ISample {
    data: HTMLCanvasElement | AudioExample;
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

interface AudioDataJSON {
    label: string;
    frameSize: number;
    sampleRateHz?: number;
    keyFrameIndex?: number;
    frameDurationMillis?: number;
}

interface LoadedSampleFiles {
    pngBase64?: string;
    audioMeta?: AudioDataJSON;
    spectrogramBuffer?: ArrayBuffer;
    pcmBuffer?: ArrayBuffer;
}

type ClassifierAppEvents = 'loading' | 'ready' | 'epoch' | 'training' | 'trainingcomplete' | 'error' | 'action';

export function createModel(
    variant: TMType,
    metadata?: any,
    model?: tf.io.ModelJSON,
    weights?: ArrayBuffer
): TeachableModel {
    switch (variant) {
        case 'image':
            return new ImageModel(variant, metadata, model, weights);
        case 'pose':
            return new PoseModel(variant, metadata, model, weights);
        case 'speech':
            return new SoundModel(variant, metadata, model, weights);
        case 'hand':
            return new HandModel(variant, metadata, model, weights);
        default:
            throw new Error(`Unsupported model variant: ${variant}`);
    }
}

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
        return this.model?.draw(image);
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

    public async predict(image: HTMLCanvasElement | AudioExample): Promise<PredictionsOutput> {
        if (this.model) {
            const predictions = await this.model.predict(image);

            if (image instanceof HTMLCanvasElement) {
                this.draw(image);
            }

            if (!predictions.predictions.length) {
                return { ...predictions, nameOfMax: '', indexOfMax: -1, failed: true };
            }

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

        const newModel = createModel(this.variant);

        this.emit('loading');
        await newModel.ready();

        newModel.setLabels(labels);

        const promises = samples.map((s, ix) => {
            return s.map((ss) => {
                return newModel.addExample(ix, ss.data);
            });
        });

        await Promise.all(promises.flat());

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

    public async saveComponents(): Promise<ModelContents> {
        const zip = new JSZip();
        if (this.samples.length > 0) {
            const folder = zip.folder('samples');
            if (folder) {
                for (let j = 0; j < this.samples.length; ++j) {
                    const s = this.samples[j];
                    for (let i = 0; i < s.length; ++i) {
                        const ss = s[i];
                        if (ss.data instanceof HTMLCanvasElement) {
                            folder.file(`${j}_${i}.png`, ss.data.toDataURL('image/png').split(';base64,')[1], {
                                base64: true,
                            });
                        } else {
                            if (ss.data.spectrogramCanvas) {
                                folder.file(
                                    `${j}_${i}.png`,
                                    ss.data.spectrogramCanvas.toDataURL('image/png').split(';base64,')[1],
                                    {
                                        base64: true,
                                    }
                                );
                            }
                            if (ss.data.rawAudio) {
                                const f32 = ss.data.rawAudio.data;
                                const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
                                folder.file(`${j}_${i}.pcm`, bytes, { binary: true });
                            }

                            const f32 = ss.data.spectrogram.data;
                            const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
                            folder.file(`${j}_${i}.spectrogram`, bytes, { binary: true });

                            const meta: AudioDataJSON = {
                                label: ss.data.label,
                                frameSize: ss.data.spectrogram.frameSize,
                                sampleRateHz: ss.data.rawAudio?.sampleRateHz,
                                keyFrameIndex: ss.data.spectrogram.keyFrameIndex,
                                frameDurationMillis: ss.data.spectrogram.frameDurationMillis,
                            };
                            folder.file(`${j}_${i}.json`, JSON.stringify(meta));
                        }
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
        return contents;
    }

    public async save(): Promise<Blob> {
        return (await this.saveComponents()).zip || new Blob();
    }

    static async load(file: string | Blob): Promise<ClassifierApp> {
        const project: ProjectTemp = {
            samples: [],
        };

        const sampleFiles = new Map<string, LoadedSampleFiles>();

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
                const m = data.name.match(/^samples\/(\d+)_(\d+)\.(png|json|spectrogram|pcm)$/);
                if (!m) return;

                const key = `${m[1]}_${m[2]}`;
                const ext = m[3];
                const entry = sampleFiles.get(key) ?? {};
                sampleFiles.set(key, entry);

                if (ext === 'png') {
                    promises.push(
                        data.async('base64').then((r) => {
                            entry.pngBase64 = r;
                        })
                    );
                } else if (ext === 'json') {
                    promises.push(
                        data.async('string').then((r) => {
                            entry.audioMeta = JSON.parse(r) as AudioDataJSON;
                        })
                    );
                } else if (ext === 'spectrogram') {
                    promises.push(
                        data.async('arraybuffer').then((r) => {
                            entry.spectrogramBuffer = r;
                        })
                    );
                } else if (ext === 'pcm') {
                    promises.push(
                        data.async('arraybuffer').then((r) => {
                            entry.pcmBuffer = r;
                        })
                    );
                }
            }
        });

        await Promise.all(promises);

        if (project.metadata && project.modelJson && project.modelWeights) {
            const meta = JSON.parse(project.metadata);

            const parsedModel = JSON.parse(project.modelJson) as tf.io.ModelJSON;

            let type: TMType = 'image';
            if ('tfjsSpeechCommandsVersion' in meta) {
                type = 'speech';
            } else if ('poseNetArchitecture' in meta) {
                type = 'pose';
            }

            const canvasFromBase64 = (base64: string) =>
                new Promise<HTMLCanvasElement>((resolve) => {
                    const canvas = document.createElement('canvas');
                    canvas.width = 224;
                    canvas.height = 224;
                    canvas.style.width = '58px';
                    canvas.style.height = '58px';
                    const ctx = canvas.getContext('2d');
                    const img = new Image();
                    img.onload = () => {
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx?.drawImage(img, 0, 0);
                        resolve(canvas);
                    };
                    img.src = `data:image/png;base64,${base64}`;
                });

            const parsedEntries = Array.from(sampleFiles.entries()).map(([key, files]) => {
                const [jStr, iStr] = key.split('_');
                return { j: Number(jStr), i: Number(iStr), files };
            });

            const samples: ISample[][] = [];

            for (const { j, i, files } of parsedEntries) {
                while (samples.length <= j) samples.push([]);
                while (samples[j].length <= i) samples[j].push({ data: document.createElement('canvas'), id: '' });

                const isAudio = !!files.audioMeta && !!files.spectrogramBuffer;

                if (isAudio) {
                    const spectrogramData = new Float32Array(files.spectrogramBuffer!);

                    const audioExample: AudioExample = {
                        label: files.audioMeta!.label,
                        spectrogram: {
                            data: spectrogramData,
                            frameSize: files.audioMeta!.frameSize,
                            keyFrameIndex: files.audioMeta!.keyFrameIndex,
                            frameDurationMillis: files.audioMeta!.frameDurationMillis,
                        },
                    };

                    if (files.pngBase64) {
                        audioExample.spectrogramCanvas = await canvasFromBase64(files.pngBase64);
                    }

                    if (files.pcmBuffer && files.audioMeta!.sampleRateHz) {
                        audioExample.rawAudio = {
                            data: new Float32Array(files.pcmBuffer),
                            sampleRateHz: files.audioMeta!.sampleRateHz,
                        };
                    }

                    samples[j][i] = { data: audioExample, id: '' };
                } else if (files.pngBase64) {
                    const canvas = await canvasFromBase64(files.pngBase64);
                    samples[j][i] = { data: canvas, id: '' };
                } else {
                    throw new Error(`Sample ${j}_${i} has no recognizable payload`);
                }
            }

            const tm = createModel(type, meta, parsedModel, project.modelWeights);

            await tm.ready();

            const app = new ClassifierApp(
                type,
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
