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

import type { Keypoint } from '@tensorflow-models/hand-pose-detection';

const FILL_COLOR = 'aqua';
const STROKE_COLOR = 'aqua';
const KEYPOINT_SIZE = 4;
const LINE_WIDTH = 2;

/**
 * MediaPipe Hands connections between the 21 keypoints.
 * Each tuple is [startIdx, endIdx].
 *
 * Keypoint indices:
 *  0 = wrist
 *  1-4 = thumb (cmc, mcp, ip, tip)
 *  5-8 = index finger (mcp, pip, dip, tip)
 *  9-12 = middle finger (mcp, pip, dip, tip)
 * 13-16 = ring finger (mcp, pip, dip, tip)
 * 17-20 = pinky (mcp, pip, dip, tip)
 */
export const HAND_CONNECTIONS: [number, number][] = [
    // Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Index finger
    [0, 5], [5, 6], [6, 7], [7, 8],
    // Middle finger
    [0, 9], [9, 10], [10, 11], [11, 12],
    // Ring finger
    [0, 13], [13, 14], [14, 15], [15, 16],
    // Pinky
    [0, 17], [17, 18], [18, 19], [19, 20],
    // Palm cross-connections
    [5, 9], [9, 13], [13, 17],
];

/**
 * Draw a single keypoint dot on the canvas.
 */
export function drawHandPoint(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    radius: number = KEYPOINT_SIZE,
    fillColor: string = FILL_COLOR,
    strokeColor: string = STROKE_COLOR
) {
    ctx.fillStyle = fillColor;
    ctx.strokeStyle = strokeColor;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
}

/**
 * Draw all 21 hand keypoints onto a canvas.
 * @param keypoints Array of 21 keypoints from hand-pose-detection
 * @param ctx Canvas 2D rendering context
 * @param minScore Minimum score to render a keypoint (defaults to 0)
 * @param scale Optional scale factor for coordinates
 */
export function drawHandKeypoints(
    keypoints: Keypoint[],
    ctx: CanvasRenderingContext2D,
    minScore = 0,
    keypointSize: number = KEYPOINT_SIZE,
    fillColor: string = FILL_COLOR,
    strokeColor: string = STROKE_COLOR,
    scale = 1
) {
    for (const kp of keypoints) {
        if (kp.score !== undefined && kp.score < minScore) continue;
        drawHandPoint(ctx, kp.x * scale, kp.y * scale, keypointSize, fillColor, strokeColor);
    }
}

/**
 * Draw the hand skeleton (connections between keypoints) onto a canvas.
 * @param keypoints Array of 21 keypoints from hand-pose-detection
 * @param ctx Canvas 2D rendering context
 * @param scale Optional scale factor for coordinates
 */
export function drawHandSkeleton(
    keypoints: Keypoint[],
    ctx: CanvasRenderingContext2D,
    lineWidth: number = LINE_WIDTH,
    strokeColor: string = STROKE_COLOR,
    scale = 1
) {
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;

    for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        const start = keypoints[startIdx];
        const end = keypoints[endIdx];
        if (!start || !end) continue;

        ctx.beginPath();
        ctx.moveTo(start.x * scale, start.y * scale);
        ctx.lineTo(end.x * scale, end.y * scale);
        ctx.stroke();
    }
}
