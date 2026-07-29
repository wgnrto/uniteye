/*
 * UnitEye web core — platform-independent gaze-consumer logic ported from the Unity C#.
 * These are faithful JS ports of RidgeRegression, the One-Euro filter, the AOI hit-tests, and the
 * eye-crop geometry / preprocessing, so a browser (WebGL) build can share the same behaviour the
 * native pipeline uses. Loaded as a classic script (works from file:// for the self-test); attaches
 * to globalThis.UnitEyeCore.
 */
(function (root) {
  "use strict";

  // ---- Linear solve (Gaussian elimination w/ partial pivoting) — for in-browser ridge calibration ----
  function solveLinear(A, b) {
    const n = b.length;
    // augmented copy
    const M = A.map((row, i) => row.slice().concat([b[i]]));
    for (let col = 0; col < n; col++) {
      // pivot
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
      const d = M[col][col] || 1e-12;
      for (let c = col; c <= n; c++) M[col][c] /= d;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col];
        for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
      }
    }
    return M.map(row => row[n]);
  }

  /**
   * Ridge regression model — mirrors RidgeRegression.cs (predict = W · [1, standardized features]).
   * Load a shipped Reg_X.json/Reg_Y.json via RidgeModel.fromJson, or train() in-browser.
   */
  class RidgeModel {
    constructor(w, affine, featureMean, featureStd) {
      this.w = w;
      this.affine = affine !== false;
      this.featureMean = featureMean || null;
      this.featureStd = featureStd || null;
    }
    static fromJson(obj) {
      return new RidgeModel(obj.W, obj.Affine !== false, obj.FeatureMean || null, obj.FeatureStd || null);
    }
    predict(features) {
      const xs = [];
      if (this.affine) xs.push(1.0);
      if (this.featureMean && this.featureStd && this.featureMean.length === features.length) {
        for (let i = 0; i < features.length; i++) xs.push((features[i] - this.featureMean[i]) / this.featureStd[i]);
      } else {
        for (let i = 0; i < features.length; i++) xs.push(features[i]);
      }
      let y = 0;
      for (let i = 0; i < this.w.length && i < xs.length; i++) y += this.w[i] * xs[i];
      return y;
    }
  }

  /**
   * Trains one ridge model per axis from calibration samples (standardized, bias term, X'X + λI).
   * @param {number[][]} features  one feature vector per sample
   * @param {number[]} targets     normalized target per sample (0..1)
   * @param {number} lambda
   * @returns {RidgeModel}
   */
  function trainRidge(features, targets, lambda) {
    const n = features.length, d = features[0].length;
    const mean = new Array(d).fill(0), std = new Array(d).fill(0);
    for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += features[i][j]; mean[j] = s / n; }
    for (let j = 0; j < d; j++) {
      let s = 0; for (let i = 0; i < n; i++) { const v = features[i][j] - mean[j]; s += v * v; }
      std[j] = Math.sqrt(s / n); if (std[j] < 1e-6) std[j] = 1;
    }
    const D = d + 1;
    const A = Array.from({ length: D }, () => new Array(D).fill(0));
    const bvec = new Array(D).fill(0);
    for (let i = 0; i < n; i++) {
      const row = new Array(D); row[0] = 1.0;
      for (let j = 0; j < d; j++) row[j + 1] = (features[i][j] - mean[j]) / std[j];
      for (let r = 0; r < D; r++) { bvec[r] += row[r] * targets[i]; for (let c = 0; c < D; c++) A[r][c] += row[r] * row[c]; }
    }
    for (let r = 0; r < D; r++) A[r][r] += lambda;
    const w = solveLinear(A, bvec);
    return new RidgeModel(w, true, mean, std);
  }

  // ---- One-Euro filter (2D), ported from OneEuroFilter.cs ----
  function alpha(cutoff, freq) {
    const te = 1.0 / freq;
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }
  class LowPass {
    constructor() { this.y = 0; this.s = 0; this.init = false; }
    filter(x, a) { this.y = x; this.s = this.init ? a * x + (1 - a) * this.s : x; this.init = true; return this.s; }
    hasLast() { return this.init; }
    last() { return this.y; }
  }
  class OneEuro {
    constructor(freq, mincutoff, beta, dcutoff) {
      this.freq = freq; this.mincutoff = mincutoff != null ? mincutoff : 1.0;
      this.beta = beta || 0.0; this.dcutoff = dcutoff != null ? dcutoff : 1.0;
      this.x = new LowPass(); this.dx = new LowPass(); this.lasttime = -1;
    }
    filterScalar(value, timestamp) {
      if (this.lasttime !== -1 && timestamp !== -1) this.freq = 1.0 / Math.max(1e-6, timestamp - this.lasttime);
      this.lasttime = timestamp;
      const dvalue = this.x.hasLast() ? (value - this.x.last()) * this.freq : 0.0;
      const edvalue = this.dx.filter(dvalue, alpha(this.dcutoff, this.freq));
      const cutoff = this.mincutoff + this.beta * Math.abs(edvalue);
      return this.x.filter(value, alpha(cutoff, this.freq));
    }
  }
  class OneEuro2D {
    constructor(freq, mincutoff, beta, dcutoff) { this.fx = new OneEuro(freq, mincutoff, beta, dcutoff); this.fy = new OneEuro(freq, mincutoff, beta, dcutoff); }
    filter(x, y, timestamp) { return { x: this.fx.filterScalar(x, timestamp), y: this.fy.filterScalar(y, timestamp) }; }
  }

  // ---- AOI hit-tests (normalized coords, (0,0) top-left .. (1,1) bottom-right), ported from AOIBox/AOICircle ----
  function pointInBox(px, py, sx, sy, ex, ey, inverted) {
    const inside = px >= Math.min(sx, ex) && px <= Math.max(sx, ex) && py >= Math.min(sy, ey) && py <= Math.max(sy, ey);
    return inverted ? !inside : inside;
  }
  function pointInCircle(px, py, cx, cy, radius, aspect, inverted) {
    // aspect = screen width/height; circle is defined in normalized space (aspect-influenced, like Unity)
    const dx = (px - cx), dy = (py - cy) / (aspect || 1);
    const inside = (dx * dx + dy * dy) <= radius * radius;
    return inverted ? !inside : inside;
  }

  /**
   * Eye-crop rectangle in source-image pixels, ported from HomulerFunctions.GetEyeTexture geometry.
   * landmarks: array of {x,y} in normalized [0,1] (MediaPipe FaceLandmarker, y top-left origin).
   * Argument order matches the native calls: left eye (362, 263), right eye (33, 133) — leftIdx is
   * the smaller-x corner. The native code works in a bottom-left origin (it y-flips landmarks and
   * uses GetPixels, whose y=0 is the bottom), so the reference row is computed in bottom-left space
   * and converted to the top-left y that canvas drawImage expects.
   * Returns {x, y, size} (square crop, top-left origin) or null if degenerate.
   */
  function eyeCropRect(landmarks, leftIdx, rightIdx, srcW, srcH) {
    const l = landmarks[leftIdx], r = landmarks[rightIdx];
    let eyeLength = r.x - l.x;
    if (eyeLength <= 0) return null;
    const xShift = eyeLength * 0.2;
    eyeLength += 2 * xShift;
    const yShift = eyeLength * 0.5;
    // native: y-flipped (bottom-left) landmark ys, yRef = avg - 2*yShift, clamped; GetPixels bottom edge
    let yRefBl = ((1 - l.y) + (1 - r.y)) * 0.5 - 2 * yShift;
    yRefBl = Math.min(1, Math.max(0, yRefBl));
    const size = Math.trunc(eyeLength * srcW);
    const x = Math.trunc((l.x - xShift) * srcW);
    const yBot = Math.trunc(yRefBl * srcH);           // bottom edge, bottom-left origin (native yBot)
    const y = srcH - yBot - size;                      // convert to top-left origin for drawImage
    if (size <= 0) return null;
    return { x, y, size };
  }

  // ---- EyeMU preprocessing: RGB in [0,1] -> [-0.5, 0.5] (PreprocessEyeMU.compute) ----
  // Fills an NHWC Float32Array (1,128,128,3) from an ImageData-like {data:Uint8ClampedArray,width,height}.
  function eyeCropToTensor(imageData, out) {
    const { data, width, height } = imageData; // expected 128x128
    let o = 0;
    for (let i = 0; i < width * height; i++) {
      out[o++] = data[i * 4] / 255 - 0.5;      // R
      out[o++] = data[i * 4 + 1] / 255 - 0.5;  // G
      out[o++] = data[i * 4 + 2] / 255 - 0.5;  // B
    }
    return out;
  }

  // ---- EyeMU calibration feature vector ----
  // Faithful port of HomulerEyeMURunner.FillEyeMUFeatures + HomulerFunctions.FillIrisFeatures, so the
  // browser feeds its calibration the SAME 19 features (same order) as the native pipeline. Layout:
  //   [0..3]   EyeMU embedding (dense_7)
  //   [4..10]  polynomial of the NORMALIZED raw gaze point: gx, gy, gx², gy², gx·gy, gx³, gy³
  //   [11..14] head pose: yaw, pitch, roll, area
  //   [15..18] per-eye normalized iris offsets (appended last, so the head-pose slots keep their
  //            indices — the native calibration's head-pose jitter targets 11/12/13 by index)
  // A per-axis LINEAR ridge over just [gx, gy] fits the centre slope and compresses the corners; the
  // quadratic/cross/cubic terms are what let the fit reach the screen corners. Changing this length
  // stales saved calibrations (they NaN and fall back to raw gaze) — recalibrate.
  const EYEMU_FEATURE_COUNT = 19;
  const EYEMU_IRIS_FEATURE_START = 15;

  // MediaPipe landmark indices — identical to the constants in HomulerFunctions.
  // Each iris is paired with the eye it ACTUALLY lies in: MediaPipe names its two 5-point iris blocks
  // (468.., 473..) by IMAGE side while the eye-corner indices use SUBJECT side, so the two namings are
  // mirrored and pairing them by name pairs each iris with the wrong eye. Verified on a real detection:
  // index 468 lies between corners 33 and 133; index 473 lies between corners 362 and 263.
  const IRIS_LANDMARKS = {
    leftEyeInnerCorner: 362, leftEyeOuterCorner: 263,
    rightEyeOuterCorner: 33, rightEyeInnerCorner: 133,
    leftIrisCenter: 473, rightIrisCenter: 468,
  };

  function fillEyeMUFeatures(f, embedding, gx, gy, headYaw, headPitch, headRoll, headArea) {
    f[0] = embedding[0];
    f[1] = embedding[1];
    f[2] = embedding[2];
    f[3] = embedding[3];
    f[4] = gx;
    f[5] = gy;
    f[6] = gx * gx;
    f[7] = gy * gy;
    f[8] = gx * gy;
    f[9] = gx * gx * gx;
    f[10] = gy * gy * gy;
    f[11] = headYaw;
    f[12] = headPitch;
    f[13] = headRoll;
    f[14] = headArea;
    return f;
  }

  // Iris center relative to the eye-corner midpoint, divided by the corner distance — the classic
  // direct webcam gaze cue, made scale-invariant (head distance / face size cancel out). Zero when an
  // eye is degenerate. Symmetric in the two corners, so their argument order does not affect the result.
  function fillOneEyeIrisOffset(cornerA, cornerB, iris, dest, index) {
    const midX = (cornerA.x + cornerB.x) * 0.5;
    const midY = (cornerA.y + cornerB.y) * 0.5;
    const dx = cornerB.x - cornerA.x, dy = cornerB.y - cornerA.y;
    const cornerDistance = Math.sqrt(dx * dx + dy * dy);
    if (cornerDistance < 1e-5) { dest[index] = dest[index + 1] = 0; return; }
    dest[index] = (iris.x - midX) / cornerDistance;
    dest[index + 1] = (iris.y - midY) / cornerDistance;
  }

  function fillIrisFeatures(landmarks, dest, start) {
    const I = IRIS_LANDMARKS;
    // Guard on the highest index actually read — which iris constant is larger depends on the mapping.
    if (!landmarks || landmarks.length <= Math.max(I.leftIrisCenter, I.rightIrisCenter)) {
      dest[start] = dest[start + 1] = dest[start + 2] = dest[start + 3] = 0;
      return dest;
    }
    fillOneEyeIrisOffset(landmarks[I.leftEyeInnerCorner], landmarks[I.leftEyeOuterCorner],
      landmarks[I.leftIrisCenter], dest, start);
    fillOneEyeIrisOffset(landmarks[I.rightEyeOuterCorner], landmarks[I.rightEyeInnerCorner],
      landmarks[I.rightIrisCenter], dest, start + 2);
    return dest;
  }

  /**
   * Build the full 19-feature EyeMU calibration vector.
   * gx, gy are the model's NORMALIZED gaze output (0..1), not pixels. pose is [yaw, pitch, roll, area].
   * `out` is an optional reuse buffer — as in the native runner it is valid only until the next call,
   * so a caller that retains the vector (e.g. calibration capture) must copy it.
   */
  function buildEyeMUFeatures(embedding, gx, gy, pose, landmarks, out) {
    const f = out || new Array(EYEMU_FEATURE_COUNT);
    fillEyeMUFeatures(f, embedding, gx, gy, pose[0], pose[1], pose[2], pose[3]);
    fillIrisFeatures(landmarks, f, EYEMU_IRIS_FEATURE_START);
    return f;
  }

  root.UnitEyeCore = {
    RidgeModel, trainRidge, OneEuro, OneEuro2D,
    pointInBox, pointInCircle, eyeCropRect, eyeCropToTensor,
    fillEyeMUFeatures, fillIrisFeatures, buildEyeMUFeatures,
    EYEMU_FEATURE_COUNT, EYEMU_IRIS_FEATURE_START, IRIS_LANDMARKS,
    _solveLinear: solveLinear,
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
