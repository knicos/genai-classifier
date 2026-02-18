// expected hue range: [0, 360)
// expected saturation range: [0, 1]
// expected lightness range: [0, 1]
function hslToRgb(hue: number, saturation: number, lightness: number) {
    // based on algorithm from http://en.wikipedia.org/wiki/HSL_and_HSV#Converting_to_RGB
    if (hue == undefined) {
        return [0, 0, 0];
    }

    const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
    let huePrime = hue / 60;
    const secondComponent = chroma * (1 - Math.abs((huePrime % 2) - 1));

    huePrime = Math.floor(huePrime);
    let red = 0;
    let green = 0;
    let blue = 0;

    if (huePrime === 0) {
        red = chroma;
        green = secondComponent;
        blue = 0;
    } else if (huePrime === 1) {
        red = secondComponent;
        green = chroma;
        blue = 0;
    } else if (huePrime === 2) {
        red = 0;
        green = chroma;
        blue = secondComponent;
    } else if (huePrime === 3) {
        red = 0;
        green = secondComponent;
        blue = chroma;
    } else if (huePrime === 4) {
        red = secondComponent;
        green = 0;
        blue = chroma;
    } else if (huePrime === 5) {
        red = chroma;
        green = 0;
        blue = secondComponent;
    }

    const lightnessAdjustment = lightness - chroma / 2;
    red += lightnessAdjustment;
    green += lightnessAdjustment;
    blue += lightnessAdjustment;

    return [Math.round(red * 255), Math.round(green * 255), Math.round(blue * 255)];
}

/**
 * Convert importance value (0-1) to a color from blue (low) to red (high)
 */
function importanceToColor(importance: number): string {
    // Hue: 240 (blue) to 0 (red)
    const hue = (1 - importance) * 240;
    const [r, g, b] = hslToRgb(hue, 1, 0.5);
    return `rgb(${r}, ${g}, ${b})`;
}

export async function renderHeatmap(input: HTMLCanvasElement, output: HTMLCanvasElement, data: number[][]) {
    if (!output) return;
    const ctx = output.getContext('2d');
    if (ctx) {
        ctx.drawImage(input, 0, 0);
        const imageData = ctx.createImageData(data.length, data.length);
        let ix = 0;
        for (let y = 0; y < data.length; ++y) {
            for (let x = 0; x < data.length; ++x) {
                const v = data[y][x];
                const [r, g, b] = hslToRgb((1 - v) * 240, 1, 0.5);
                imageData.data[ix] = r;
                ++ix;
                imageData.data[ix] = g;
                ++ix;
                imageData.data[ix] = b;
                ++ix;
                imageData.data[ix] = 128;
                ++ix;
            }
        }
        ctx.drawImage(await createImageBitmap(imageData), 0, 0);
    }
}

/**
 * Render pose XAI visualization by drawing keypoints with colors indicating importance
 * @param input Original canvas with the pose image
 * @param output Canvas where the visualization will be rendered
 * @param keypoints Array of detected keypoints
 * @param featureImportance Array of importance values for each feature in the pose output
 * @param minConfidence Minimum confidence to draw a keypoint
 */
export function renderPoseXAI(
    input: HTMLCanvasElement,
    output: HTMLCanvasElement,
    keypoints: any[],
    featureImportance: Float32Array,
    minConfidence: number = 0.5
) {
    if (!output) return;
    const ctx = output.getContext('2d');
    if (!ctx) return;

    // Ensure output canvas matches input size for proper scaling
    if (output.width !== input.width || output.height !== input.height) {
        output.width = input.width;
        output.height = input.height;
    }

    // Draw the original image
    ctx.drawImage(input, 0, 0);

    // The pose output is concatenated heatmaps and offsets
    // First half: heatmaps (17 keypoints), second half: offsets (2 * 17 = 34 values)
    // So we need to extract the importance for each keypoint from the heatmap portion
    const numKeypoints = keypoints.length;
    
    // Calculate average importance per keypoint
    // The feature importance corresponds to the flattened pose output
    // which has structure [heatmaps..., offsets_x..., offsets_y...]
    const keypointImportance: number[] = [];
    
    for (let i = 0; i < numKeypoints; i++) {
        // Average the importance across heatmap and both offset dimensions for this keypoint
        let importance = 0;
        let count = 0;
        
        // Heatmap importance for this keypoint
        if (i < featureImportance.length) {
            importance += featureImportance[i];
            count++;
        }
        
        // Offset X importance
        if (numKeypoints + i < featureImportance.length) {
            importance += featureImportance[numKeypoints + i];
            count++;
        }
        
        // Offset Y importance
        if (2 * numKeypoints + i < featureImportance.length) {
            importance += featureImportance[2 * numKeypoints + i];
            count++;
        }
        
        keypointImportance[i] = count > 0 ? importance / count : 0;
    }

    // Normalize keypoint importance
    const maxImportance = Math.max(...keypointImportance);
    const minImportance = Math.min(...keypointImportance);
    const range = maxImportance - minImportance;
    
    // Draw keypoints with importance-based colors
    for (let i = 0; i < keypoints.length; i++) {
        const keypoint = keypoints[i];
        
        if (keypoint.score < minConfidence) {
            continue;
        }
        
        const { y, x } = keypoint.position;
        const normalizedImportance = range > 0 ? (keypointImportance[i] - minImportance) / range : 0.5;
        const color = importanceToColor(normalizedImportance);
        
        // Use fixed size for all keypoints
        const size = 5;
        
        ctx.fillStyle = color;
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
    }
    
    // Draw skeleton with importance-based colors
    const adjacentKeyPointPairs = getAdjacentKeyPointPairs(keypoints, minConfidence);
    
    for (const [kp1, kp2] of adjacentKeyPointPairs) {
        const importance1 = keypointImportance[keypoints.indexOf(kp1)];
        const importance2 = keypointImportance[keypoints.indexOf(kp2)];
        const avgImportance = (importance1 + importance2) / 2;
        const normalizedImportance = range > 0 ? (avgImportance - minImportance) / range : 0.5;
        const color = importanceToColor(normalizedImportance);
        
        const lineWidth = 2 + normalizedImportance * 2;
        
        ctx.beginPath();
        ctx.moveTo(kp1.position.x, kp1.position.y);
        ctx.lineTo(kp2.position.x, kp2.position.y);
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = color;
        ctx.stroke();
    }
}

/**
 * Helper function to get adjacent keypoint pairs for skeleton drawing
 */
function getAdjacentKeyPointPairs(keypoints: any[], minConfidence: number): [any, any][] {
    // Define skeleton connections (same as in pose-draw.ts)
    const adjacentPairs: [string, string][] = [
        ['leftHip', 'leftShoulder'],
        ['leftElbow', 'leftShoulder'],
        ['leftElbow', 'leftWrist'],
        ['leftHip', 'leftKnee'],
        ['leftKnee', 'leftAnkle'],
        ['rightHip', 'rightShoulder'],
        ['rightElbow', 'rightShoulder'],
        ['rightElbow', 'rightWrist'],
        ['rightHip', 'rightKnee'],
        ['rightKnee', 'rightAnkle'],
        ['leftShoulder', 'rightShoulder'],
        ['leftHip', 'rightHip'],
    ];
    
    const result: [any, any][] = [];
    
    for (const [partName1, partName2] of adjacentPairs) {
        const kp1 = keypoints.find(kp => kp.part === partName1);
        const kp2 = keypoints.find(kp => kp.part === partName2);
        
        if (kp1 && kp2 && kp1.score >= minConfidence && kp2.score >= minConfidence) {
            result.push([kp1, kp2]);
        }
    }
    
    return result;
}

