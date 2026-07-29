/*
 * uniteye-cv.js — browser gaze pipeline (ES module).
 *
 * Real mode: getUserMedia -> MediaPipe FaceLandmarker (@mediapipe/tasks-vision) -> eye crops ->
 * EyeMU (onnxruntime-web) -> raw gaze; then the SHARED consumer layer (uniteye-core.js): ridge
 * calibration -> One-Euro filter -> your onGaze callback (drive a gaze dot, AOI hit-tests, logging,
 * or SendMessage into a Unity WebGL build).
 *
 * Mock mode ({mock:true}): skips all CV and uses the mouse as the raw gaze, so the entire consumer
 * plumbing (calibration/filter/AOI/logging or the Unity bridge) can be exercised without a webcam.
 *
 * Depends on globalThis.UnitEyeCore (load uniteye-core.js first).
 * Verified in-browser: EyeMU runs in onnxruntime-web (correct I/O names, finite output) and
 * FaceLandmarker initializes; the model/CDN versions below are the ones confirmed working.
 *
 * MediaPipe is pinned to tasks-vision 1.0.0 (Google's first stable release, July 2026). The bump from
 * 0.10.35 was verified to be behaviour-preserving: on the same face image both versions return 478
 * landmarks whose coordinates are bit-identical (the .task model is pinned, so only the runtime moved).
 * Pin the version deliberately — @latest can break without notice.
 */

const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.mjs';
const ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const MP_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.0/vision_bundle.mjs';
const MP_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.0/wasm';
const MP_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// MediaPipe canonical face-mesh indices (match the native UnitEye eye-corner + EAR indices)
const IDX = {
  // eye corners: right {outer 33, inner 133}, left {inner 362, outer 263}
  rOuter: 33, rInner: 133, lInner: 362, lOuter: 263,
  // right-eye EAR verticals/horizontal (HomulerEyeHelper)
  rEar: [[160, 144], [159, 145], [158, 153], [33, 133]],
  lEar: [[385, 380], [386, 374], [387, 373], [362, 263]],
  // head pose landmarks (FaceMeshSolution)
  yaw: [50, 280], pitch: [10, 168], roll: [6, 151],
};

function dist2(ax, ay, bx, by, sw, sh) {
  const dx = (ax - bx) * sw, dy = (ay - by) * sh;
  return Math.sqrt(dx * dx + dy * dy);
}

export class UnitEyeWeb {
  constructor(opts = {}) {
    const C = globalThis.UnitEyeCore;
    if (!C) throw new Error('UnitEyeCore not loaded (include uniteye-core.js first)');
    this.C = C;
    this.mock = !!opts.mock;
    this.modelUrl = opts.modelUrl || 'models/eyemu.onnx';
    this.onGaze = opts.onGaze || function () {};
    this.screenW = opts.screenW || window.innerWidth;
    this.screenH = opts.screenH || window.innerHeight;
    this.blinkThreshold = opts.blinkThreshold ?? 0.15;
    this.calibX = null; this.calibY = null;       // UnitEyeCore.RidgeModel per axis (optional)
    // One-Euro tuned for coarse, stable output; adjust to taste
    this.filter = new C.OneEuro2D(60, opts.mincutoff ?? 0.5, opts.beta ?? 0.01, opts.dcutoff ?? 1.0);
    this._running = false;
    this._lastVideoTime = -1;
    this._mouse = { x: this.screenW / 2, y: this.screenH / 2 };
    this._crop = document.createElement('canvas'); this._crop.width = 128; this._crop.height = 128;
    this._cropCtx = this._crop.getContext('2d', { willReadFrequently: true });
    // Reused calibration-feature buffer (mirrors the native runner's reused float[]); valid only until
    // the next frame — a consumer that retains the vector must copy it.
    this._features = new Array(C.EYEMU_FEATURE_COUNT);
  }

  /** Load a shipped Reg_X.json / Reg_Y.json (or in-browser trained models) to calibrate. */
  setCalibration(regX, regY) {
    this.calibX = regX ? this.C.RidgeModel.fromJson(regX) : null;
    this.calibY = regY ? this.C.RidgeModel.fromJson(regY) : null;
  }

  async init() {
    if (this.mock) return;
    // EyeMU on onnxruntime-web
    const ort = await import(ORT_URL);
    ort.env.wasm.wasmPaths = ORT_WASM;
    ort.env.wasm.numThreads = 1; // avoids requiring cross-origin isolation (SharedArrayBuffer)
    this.ort = ort;
    this.session = await ort.InferenceSession.create(this.modelUrl, { executionProviders: ['wasm'] });
    // MediaPipe FaceLandmarker
    const vision = await import(MP_URL);
    const fileset = await vision.FilesetResolver.forVisionTasks(MP_WASM);
    this.faceLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MP_MODEL },
      runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false,
    });
    // camera
    this.video = document.createElement('video');
    this.video.autoplay = true; this.video.playsInline = true; this.video.muted = true;
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    this.video.srcObject = stream;
    await new Promise(res => { this.video.onloadedmetadata = () => { this.video.play(); res(); }; });
  }

  start() {
    this._running = true;
    if (this.mock) {
      window.addEventListener('mousemove', e => { this._mouse.x = e.clientX; this._mouse.y = e.clientY; });
      const tickMock = () => {
        if (!this._running) return;
        const t = performance.now() / 1000;
        this._emit(this._mouse.x, this._mouse.y, null, false, true, t);
        requestAnimationFrame(tickMock);
      };
      requestAnimationFrame(tickMock);
      return;
    }
    const tick = async () => {
      if (!this._running) return;
      try { await this._processFrame(); } catch (e) { console.warn('UnitEye frame error', e); }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  stop() {
    this._running = false;
    if (this.video && this.video.srcObject) this.video.srcObject.getTracks().forEach(t => t.stop());
  }

  async _processFrame() {
    const now = performance.now();
    // requestAnimationFrame often runs faster than the webcam. Do not turn a single camera frame into
    // multiple identical observations: that distorts calibration/evaluation sample weighting and adds
    // misleading filter updates without new visual evidence.
    if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        this.video.currentTime === this._lastVideoTime) return;
    this._lastVideoTime = this.video.currentTime;
    const res = this.faceLandmarker.detectForVideo(this.video, now);
    if (!res || !res.faceLandmarks || res.faceLandmarks.length === 0) {
      this._emit(0, 0, null, false, false, now / 1000);
      return;
    }
    const lm = res.faceLandmarks[0]; // 478 {x,y,z} normalized, y top-left
    const vw = this.video.videoWidth, vh = this.video.videoHeight;

    // eye crops (128x128, NHWC, RGB-0.5). Left eye is flipped to match EyeMU (native flips left).
    // Argument order matches the native GetEyeTexture calls: left (362=lInner, 263=lOuter),
    // right (33=rOuter, 133=rInner) — first index is the smaller-x corner, so eyeLength > 0.
    const leftTensor = this._eyeTensor(lm, IDX.lInner, IDX.lOuter, vw, vh, true);
    const rightTensor = this._eyeTensor(lm, IDX.rOuter, IDX.rInner, vw, vh, false);
    if (!leftTensor || !rightTensor) { this._emit(0, 0, null, false, false, now / 1000); return; }

    // eye corners (8) and head pose (4)
    const corners = Float32Array.from([
      lm[IDX.lOuter].x, lm[IDX.lOuter].y, lm[IDX.lInner].x, lm[IDX.lInner].y,
      lm[IDX.rOuter].x, lm[IDX.rOuter].y, lm[IDX.rInner].x, lm[IDX.rInner].y,
    ]);
    const pose = this._headPose(lm);

    const ort = this.ort;
    const feeds = {
      'input_1:0': new ort.Tensor('float32', leftTensor, [1, 128, 128, 3]),
      'input_2:0': new ort.Tensor('float32', rightTensor, [1, 128, 128, 3]),
      'input_4': new ort.Tensor('float32', corners, [1, 8]),
      'input_5': new ort.Tensor('float32', pose, [1, 4]),
    };
    const out = await this.session.run(feeds);
    const emb = out['dense_7'].data;   // 4
    const gaze = out['dense_8'].data;  // 2 (normalized 0..1)

    const rawX = gaze[0] * this.screenW;
    const rawY = gaze[1] * this.screenH;

    // 19-feature vector, matching HomulerEyeMURunner.Features / FeatureCount exactly:
    // [embedding 4, polynomial of the normalized gaze point 7, head pose 4, iris offsets 4].
    // NOTE these use gaze[0]/gaze[1] — the model's NORMALIZED 0..1 output, not the pixel values:
    // the native runner divides its pixel NetworkOutput back by Screen.width/height for exactly this
    // reason (keeps the squared/cubed terms in a sane range). The old vector fed raw pixels plus two
    // constant screen-size features, which standardize to zero and carry no signal.
    const features = this.C.buildEyeMUFeatures(emb, gaze[0], gaze[1], pose, lm, this._features);

    const blink = this._blink(lm, vw, vh);
    this._emit(rawX, rawY, features, blink, true, now / 1000);
  }

  _eyeTensor(lm, aIdx, bIdx, vw, vh, flip) {
    const r = this.C.eyeCropRect(lm, aIdx, bIdx, vw, vh);
    if (!r) return null;
    const ctx = this._cropCtx;
    ctx.save();
    if (flip) { ctx.translate(128, 0); ctx.scale(-1, 1); }
    // draw the source crop scaled to 128x128
    ctx.drawImage(this.video, r.x, r.y, r.size, r.size, 0, 0, 128, 128);
    ctx.restore();
    const img = ctx.getImageData(0, 0, 128, 128);
    return this.C.eyeCropToTensor(img, new Float32Array(128 * 128 * 3));
  }

  _headPose(lm) {
    const [y0, y1] = IDX.yaw, [p0, p1] = IDX.pitch, [r0, r1] = IDX.roll;
    const yaw = Math.atan((lm[y0].z - lm[y1].z) / ((lm[y0].x - lm[y1].x) || 1e-6));
    const pitch = Math.atan((lm[p0].z - lm[p1].z) / ((lm[p1].y - lm[p0].y) || 1e-6));
    let roll = Math.atan2(lm[r1].x - lm[r0].x, lm[r0].y - lm[r1].y);
    roll = (roll >= 0 ? roll - Math.PI : roll + Math.PI) / 2;
    // head "area": landmark bounding-box area (proxy for FaceRects area)
    let minx = 1, miny = 1, maxx = 0, maxy = 0;
    for (const p of lm) { if (p.x < minx) minx = p.x; if (p.x > maxx) maxx = p.x; if (p.y < miny) miny = p.y; if (p.y > maxy) maxy = p.y; }
    const area = (maxx - minx) * (maxy - miny);
    return Float32Array.from([yaw, pitch, roll, area]);
  }

  _blink(lm, vw, vh) {
    const ear = (spec) => {
      const d1 = dist2(lm[spec[0][0]].x, lm[spec[0][0]].y, lm[spec[0][1]].x, lm[spec[0][1]].y, vw, vh);
      const d2 = dist2(lm[spec[1][0]].x, lm[spec[1][0]].y, lm[spec[1][1]].x, lm[spec[1][1]].y, vw, vh);
      const d3 = dist2(lm[spec[2][0]].x, lm[spec[2][0]].y, lm[spec[2][1]].x, lm[spec[2][1]].y, vw, vh);
      const D = dist2(lm[spec[3][0]].x, lm[spec[3][0]].y, lm[spec[3][1]].x, lm[spec[3][1]].y, vw, vh);
      return (d1 + d2 + d3) / (3 * (D || 1e-6));
    };
    return ((ear(IDX.rEar) + ear(IDX.lEar)) / 2) < this.blinkThreshold;
  }

  _emit(rawX, rawY, features, blink, facePresent, t) {
    let x = rawX, y = rawY;
    // calibration (if loaded and we have a full feature vector)
    if (this.calibX && this.calibY && features) {
      x = this.calibX.predict(features) * this.screenW;
      y = this.calibY.predict(features) * this.screenH;
    }
    const f = this.filter.filter(x, y, t);
    this.onGaze({ x: f.x, y: f.y }, { x: rawX, y: rawY }, blink, facePresent, features);
  }
}
