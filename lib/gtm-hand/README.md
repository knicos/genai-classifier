# Teachable Machine Library - Hand

Library for using hand-pose models created with Teachable Machine.

### Model checkpoints

There is one link related to your model that will be provided by Teachable Machine

`https://teachablemachine.withgoogle.com/models/MODEL_ID/`

Which you can use to access:

- The model topology: `https://teachablemachine.withgoogle.com/models/MODEL_ID/model.json`
- The model metadata: `https://teachablemachine.withgoogle.com/models/MODEL_ID/metadata.json`

## Usage

There are two ways to easily use the model provided by Teachable Machine in your Javascript project: by using this library via script tags or by installing this library from NPM (and using a build tool like Parcel, WebPack, or Rollup).

### via Script Tag

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/hand@2.0.0/dist/teachablemachine-hand.min.js"></script>
```

### via NPM

[NPM Package](https://www.npmjs.com/package/@teachablemachine/hand)

```bash
npm i @tensorflow/tfjs
npm i @teachablemachine/hand
```

```js
import * as tf from '@tensorflow/tfjs';
import * as tmHand from '@teachablemachine/hand';
```

### Sample snippet

```html
<div>Teachable Machine Hand Model</div>
<button
    type="button"
    onclick="init()"
>
    Start
</button>
<canvas id="canvas"></canvas>
<div id="label-container"></div>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/hand@2.0.0/dist/teachablemachine-hand.min.js"></script>
<script>
    const URL = '{{URL}}';
    let model, video, canvas, ctx, labelContainer;

    async function init() {
        const modelURL = URL + 'model.json';
        const metadataURL = URL + 'metadata.json';

        model = await tmHand.load(modelURL, metadataURL);

        video = document.createElement('video');
        video.autoplay = true;
        video.playsInline = true;
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 224, height: 224 } });
        video.srcObject = stream;
        await video.play();

        canvas = document.getElementById('canvas');
        canvas.width = 224;
        canvas.height = 224;
        ctx = canvas.getContext('2d');

        labelContainer = document.getElementById('label-container');

        window.requestAnimationFrame(loop);
    }

    async function loop() {
        await predict();
        window.requestAnimationFrame(loop);
    }

    async function predict() {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // estimateHand returns:
        // - allHands: keypoints for drawing
        // - jointHandOutput: fixed-size feature for classification
        const { allHands, jointHandOutput } = await model.estimateHand(canvas, false);

        const prediction = await model.predict(jointHandOutput);

        // draw all detected hands
        for (const hand of allHands) {
            tmHand.drawHandKeypoints(hand.keypoints, ctx, 0.5);
            tmHand.drawHandSkeleton(hand.keypoints, ctx);
        }

        // render predictions
        labelContainer.innerHTML = prediction.map((p) => `${p.className}: ${p.probability.toFixed(2)}`).join('<br/>');
    }
</script>
```

## API

### Loading the model - url checkpoints

`tmHand` is the module name, which is automatically included when you use the `<script src>` method. It gets added as an object to your window so you can access via `window.tmHand` or simply `tmHand`.

```ts
tmHand.load(
  checkpoint: string,
  metadata?: string | Metadata
)
```

Args:

- **checkpoint**: a URL to a json file that contains the model topology and a reference to a bin file (model weights)
- **metadata**: a URL to a json file that contains the text labels of your model and additional information

Usage:

```js
const model = await tmHand.load(checkpointURL, metadataURL);
```

### Loading the model - browser files

You can upload your model files from a local hard drive by using a file picker and the File interface.

```ts
tmHand.loadFromFiles(
  model: File,
  weights: File,
  metadata: File
)
```

Args:

- **model**: a File object that contains the model topology (.json)
- **weights**: a File object with the model weights (.bin)
- **metadata**: a File object that contains the text labels of your model and additional information (.json)

Usage:

```js
const model = await tmHand.loadFromFiles(uploadModel.files[0], uploadWeights.files[0], uploadMetadata.files[0]);
```

### Hand detector - estimateHand

Run MediaPipe Hands + feature extraction.

```ts
model.estimateHand(
  sample: ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | tf.Tensor3D,
  flipHorizontal?: boolean
)
```

Returns:

- **hand**: first detected hand (or `null`)
- **handOutput**: single-hand feature vector (first hand)
- **allHands**: all detected hands
- **allHandOutputs**: per-hand feature vectors
- **jointHandOutput**: fixed-size multi-hand feature vector used by the classifier

Notes:

- Hand order is normalized left-to-right by wrist x-coordinate.
- Inference runs in static mode for every call.

### Teachable Machine model - predict

Classify a hand feature vector.

```ts
model.predict(handOutput: Float32Array)
```

Usage (recommended with current multi-hand pipeline):

```js
const { jointHandOutput } = await model.estimateHand(input);
const prediction = await model.predict(jointHandOutput);
```

### Teachable Machine model - predictTopK

```ts
model.predictTopK(handOutput: Float32Array, maxPredictions?: number)
```

Usage:

```js
const { jointHandOutput } = await model.estimateHand(input);
const prediction = await model.predictTopK(jointHandOutput, 3);
```

### Drawing utilities

```ts
tmHand.drawHandKeypoints(keypoints, ctx, minScore?)
tmHand.drawHandSkeleton(keypoints, ctx)
tmHand.drawHandPoint(ctx, x, y)
```

### Training model API (internal package usage)

`createTeachable` returns a `TeachableHandPose` instance used by the app training pipeline.

```ts
createTeachable(metadata: Partial<Metadata>)
```

Key methods:

```ts
model.setLabels(labels: string[])
model.addExample(classIndex: number, sample: Float32Array)
model.train({ denseUnits, epochs, learningRate, batchSize }, callbacks)
```

## Exports

Main exports from this package:

- `load`, `loadFromFiles`
- `CustomHandPose`, `TeachableHandPose`, `createTeachable`
- `drawHandKeypoints`, `drawHandSkeleton`, `drawHandPoint`, `HAND_CONNECTIONS`
- `version`
