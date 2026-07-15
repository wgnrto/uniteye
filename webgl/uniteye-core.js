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

  root.UnitEyeCore = {
    RidgeModel, trainRidge, OneEuro, OneEuro2D,
    pointInBox, pointInCircle, eyeCropRect, eyeCropToTensor,
    _solveLinear: solveLinear,
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
