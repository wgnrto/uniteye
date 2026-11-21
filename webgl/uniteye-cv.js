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
 * MediaPipe is pinned to tasks-vision 1.2.3 (1.x stable line). The bump from
 * 0.10.35 / 1.0.0 was verified to be behaviour-preserving: on the same face image the runtime returns 478
 * landmarks whose coordinates are bit-identical (the .task model is pinned, so only the runtime moved).
 * Pin the version deliberately — @latest can break without notice.
 */

const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.mjs';
const ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
const MP_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs';
const MP_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm';
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
    this._fixedScreen = !!(opts.screenW && opts.screenH);
    this._canvas = opts.canvas || null;
    this.screenW = opts.screenW || window.innerWidth;
    this.screenH = opts.screenH || window.innerHeight;
    this._updateViewportSize();
    this.blinkThreshold = opts.blinkThreshold ?? 0.15;
    this.calibX = null; this.calibY = null;       // UnitEyeCore.RidgeModel per axis (optional)
    // One-Euro tuned for coarse, stable output; adjust to taste
    this.filter = new C.OneEuro2D(60, opts.mincutoff ?? 0.5, opts.beta ?? 0.01, opts.dcutoff ?? 1.0);
    this._running = false;
    this._paused = false;
    this._lastVideoTime = -1;
    this._mouse = { x: this.screenW / 2, y: this.screenH / 2 };
    this._onMouseMove = null;
    this._crop = document.createElement('canvas'); this._crop.width = 128; this._crop.height = 128;
    this._cropCtx = this._crop.getContext('2d', { willReadFrequently: true });
    // Reused typed-array buffers so each frame allocates zero EyeMU input arrays.
    this._leftBuf = new Float32Array(128 * 128 * 3);
    this._rightBuf = new Float32Array(128 * 128 * 3);
    this._cornersBuf = new Float32Array(8);
    this._poseBuf = new Float32Array(4);
    this._feeds = null;
    // Reused calibration-feature buffer (mirrors the native runner's reused float[]); valid only until
    // the next frame — a consumer that retains the vector must copy it.
    this._features = new Array(C.EYEMU_FEATURE_COUNT);
  }

  _updateViewportSize() {
    if (this._fixedScreen) return;
    const canvas = this._canvas || (typeof document !== 'undefined' && document.querySelector('#unity-canvas'));
    if (canvas && canvas.width > 0 && canvas.height > 0) {
      this._canvas = canvas;
      this.screenW = canvas.width;
      this.screenH = canvas.height;
    } else if (typeof window !== 'undefined') {
      this.screenW = window.innerWidth || this.screenW;
      this.screenH = window.innerHeight || this.screenH;
    }
  }

  setPaused(paused) {
    this._paused = !!paused;
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
    // Blendshapes (8 direct eyeLook* gaze cues + eyeBlink) and the facial transformation matrix
    // (metric head rotation + TRANSLATION) feed the 36-feature calibration vector, mirroring native.
    this.faceLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MP_MODEL },
      runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
    });
    // camera. 1920x1080: at ~60cm, 1 deg of gaze moves the iris ~0.3px at 480p-class capture — the
    // per-frame signal is SUB-PIXEL, and landmark jitter is the binding accuracy constraint, so capture
    // resolution is the single cheapest accuracy lever (native has captured at 1920 for a while; this
    // path was still asking for 640x480 = 3x less px/deg). 60fps = 2x samples per fixation + shorter
    // exposure (sharper iris); the browser falls back to the closest supported mode.
    this.video = document.createElement('video');
    this.video.autoplay = true; this.video.playsInline = true; this.video.muted = true;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 60 } },
      audio: false,
    });
    this.video.srcObject = stream;
    await new Promise(res => { this.video.onloadedmetadata = () => { this.video.play(); res(); }; });
  }

  start() {
    this._running = true;
    this._paused = false;
    if (this.mock) {
      if (this._onMouseMove) window.removeEventListener('mousemove', this._onMouseMove);
      this._onMouseMove = e => {
        this._updateViewportSize();
        if (this._canvas) {
          const r = this._canvas.getBoundingClientRect();
          if (r.width > 0 && r.height > 0) {
            this._mouse.x = (e.clientX - r.left) * (this.screenW / r.width);
            this._mouse.y = (e.clientY - r.top) * (this.screenH / r.height);
            return;
          }
        }
        this._mouse.x = e.clientX;
        this._mouse.y = e.clientY;
      };
      window.addEventListener('mousemove', this._onMouseMove);
      const tickMock = () => {
        if (!this._running) return;
        if (!this._paused) {
          this._updateViewportSize();
          const t = performance.now() / 1000;
          this._emit(this._mouse.x, this._mouse.y, null, false, true, t);
        }
        requestAnimationFrame(tickMock);
      };
      requestAnimationFrame(tickMock);
      return;
    }
    const tick = async () => {
      if (!this._running) return;
      if (!this._paused) {
        try { await this._processFrame(); } catch (e) { console.warn('UnitEye frame error', e); }
      }
      if (this._running) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  stop() {
    this._running = false;
    if (this._onMouseMove) {
      window.removeEventListener('mousemove', this._onMouseMove);
      this._onMouseMove = null;
    }
    if (this.video && this.video.srcObject) {
      this.video.srcObject.getTracks().forEach(t => t.stop());
      this.video.srcObject = null;
    }
    if (this._feeds) {
      for (const k of Object.keys(this._feeds)) {
        try { this._feeds[k]?.dispose?.(); } catch (_) {}
      }
      this._feeds = null;
    }
    try { this.faceLandmarker?.close?.(); } catch (_) {}
    this.faceLandmarker = null;
    try { this.session?.release?.(); } catch (_) {}
    this.session = null;
  }

  async _processFrame() {
    this._updateViewportSize();
    const now = performance.now();
    if (!this.video || !this.faceLandmarker || !this.session) return;
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

    const leftTensor = this._eyeTensor(lm, IDX.lInner, IDX.lOuter, vw, vh, true, this._leftBuf);
    const rightTensor = this._eyeTensor(lm, IDX.rOuter, IDX.rInner, vw, vh, false, this._rightBuf);
    if (!leftTensor || !rightTensor) { this._emit(0, 0, null, false, false, now / 1000); return; }

    const corners = this._cornersBuf;
    corners[0] = lm[IDX.lOuter].x; corners[1] = lm[IDX.lOuter].y;
    corners[2] = lm[IDX.lInner].x; corners[3] = lm[IDX.lInner].y;
    corners[4] = lm[IDX.rOuter].x; corners[5] = lm[IDX.rOuter].y;
    corners[6] = lm[IDX.rInner].x; corners[7] = lm[IDX.rInner].y;
    const pose = this._headPose(lm, res, this._poseBuf);

    if (!this._feeds) {
      const ort = this.ort;
      this._feeds = {
        'input_1:0': new ort.Tensor('float32', leftTensor, [1, 128, 128, 3]),
        'input_2:0': new ort.Tensor('float32', rightTensor, [1, 128, 128, 3]),
        'input_4': new ort.Tensor('float32', corners, [1, 8]),
        'input_5': new ort.Tensor('float32', pose, [1, 4]),
      };
    }
    const out = await this.session.run(this._feeds);
    if (!this._running || this._paused) {
      try { out['dense_7']?.dispose?.(); out['dense_8']?.dispose?.(); } catch (_) {}
      return;
    }

    const embData = out['dense_7'].data;
    const gazeData = out['dense_8'].data;
    const gx = gazeData[0], gy = gazeData[1];
    const emb0 = embData[0], emb1 = embData[1], emb2 = embData[2], emb3 = embData[3];
    try { out['dense_7']?.dispose?.(); out['dense_8']?.dispose?.(); } catch (_) {}

    const rawX = gx * this.screenW;
    const rawY = gy * this.screenH;

    const embArr = this._embScratch || (this._embScratch = [0, 0, 0, 0]);
    embArr[0] = emb0; embArr[1] = emb1; embArr[2] = emb2; embArr[3] = emb3;
    const features = this.C.buildEyeMUFeatures(embArr, gx, gy, pose, lm, this._features,
      this._contextFromResult(res));

    const blinkScore = this._blinkFromResult(res);
    const blink = blinkScore !== null ? blinkScore > 0.5 : this._blink(lm, vw, vh);
    this._emit(rawX, rawY, features, blink, true, now / 1000);
  }

  // Max of the two eyeBlink blendshape scores, or null when blendshapes are unavailable (or the
  // name->index cache hasn't been built yet — _contextFromResult builds it each frame before this runs).
  _blinkFromResult(res) {
    const shapes = res.faceBlendshapes && res.faceBlendshapes[0];
    if (!shapes || !shapes.categories || !this._blinkIndex) return null;
    const [l, r] = this._blinkIndex;
    if (l < 0 || r < 0) return null;
    return Math.max(shapes.categories[l].score, shapes.categories[r].score);
  }

  // Context source for the shared 17-feature block (mirrors HomulerFunctions.FillTailContext): head
  // translation/depth in METRES from the facial transformation matrix plus the 8 eyeLook* blendshape
  // scores resolved by category name (order matches the native EyeLookNames).
  // MATRIX LAYOUT: @mediapipe/tasks-vision returns the 4x4 COLUMN-major (OpenGL convention, same layout
  // Unity's Matrix4x4 constructor consumes on the native side) — the translation column is data[12..14].
  // Reading it row-major (data[3]/[7]/[11]) picks the fourth column of each ROW, i.e. three zeros from
  // the rotation block, silently zeroing the tx/ty/dist features.
  _contextFromResult(res) {
    const ctx = { tx: 0, ty: 0, dist: 0, eyeLook: this._eyeLookScratch || (this._eyeLookScratch = new Array(8).fill(0)) };
    const m = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
    if (m && m.data && m.data.length >= 16) {
      ctx.tx = m.data[12] * 0.01;
      ctx.ty = m.data[13] * 0.01;
      ctx.dist = Math.abs(m.data[14]) * 0.01;
    }
    ctx.eyeLook.fill(0);
    const shapes = res.faceBlendshapes && res.faceBlendshapes[0];
    if (shapes && shapes.categories) {
      if (!this._eyeLookIndex) {
        const names = ['eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeLookDownLeft',
                       'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight', 'eyeLookDownRight'];
        this._eyeLookIndex = names.map(n => shapes.categories.findIndex(c => c.categoryName === n));
        this._blinkIndex = ['eyeBlinkLeft', 'eyeBlinkRight']
          .map(n => shapes.categories.findIndex(c => c.categoryName === n));
      }
      for (let i = 0; i < 8; i++) {
        const idx = this._eyeLookIndex[i];
        if (idx >= 0) ctx.eyeLook[i] = shapes.categories[idx].score;
      }
    }
    return ctx;
  }

  _eyeTensor(lm, aIdx, bIdx, vw, vh, flip, outBuf) {
    const r = this.C.eyeCropRect(lm, aIdx, bIdx, vw, vh);
    if (!r) return null;
    if (r.x < 0 || r.y < 0 || r.x + r.size > vw || r.y + r.size > vh) return null;
    const ctx = this._cropCtx;
    ctx.save();
    if (flip) { ctx.translate(128, 0); ctx.scale(-1, 1); }
    ctx.drawImage(this.video, r.x, r.y, r.size, r.size, 0, 0, 128, 128);
    ctx.restore();
    const img = ctx.getImageData(0, 0, 128, 128);
    return this.C.eyeCropToTensor(img, outBuf || new Float32Array(128 * 128 * 3));
  }

  _headPose(lm, res, outBuf) {
    let yaw, pitch, roll;
    const m = res && res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
    if (m && m.data && m.data.length >= 16) {
      // Match FaceMeshSolution.ExtractTransformMatrix() (Unity Quaternion.LookRotation(col2, col1).eulerAngles)
      let fx = m.data[8], fy = m.data[9], fz = m.data[10];
      let ux = m.data[4], uy = m.data[5], uz = m.data[6];
      const fLen = Math.hypot(fx, fy, fz);
      const uLen = Math.hypot(ux, uy, uz);
      if (fLen > 1e-4 && uLen > 1e-4) {
        fx /= fLen; fy /= fLen; fz /= fLen;
        let rx = uy * fz - uz * fy;
        let ry = uz * fx - ux * fz;
        let rz = ux * fy - uy * fx;
        const rLen = Math.hypot(rx, ry, rz);
        if (rLen > 1e-4) {
          rx /= rLen; ry /= rLen; rz /= rLen;
          const oux = fy * rz - fz * ry;
          const ouy = fz * rx - fx * rz;
          pitch = Math.asin(Math.max(-1, Math.min(1, -fy)));
          const cosPitch = Math.cos(pitch);
          if (cosPitch > 1e-4) {
            yaw = Math.atan2(fx, fz);
            roll = Math.atan2(ry, ouy);
          } else {
            yaw = Math.atan2(-rx, oux);
            roll = 0;
          }
        }
      }
    }
    if (yaw === undefined) {
      const [y0, y1] = IDX.yaw, [p0, p1] = IDX.pitch, [r0, r1] = IDX.roll;
      yaw = Math.atan((lm[y0].z - lm[y1].z) / ((lm[y0].x - lm[y1].x) || 1e-6));
      pitch = Math.atan((lm[p0].z - lm[p1].z) / ((lm[p1].y - lm[p0].y) || 1e-6));
      roll = Math.atan2(lm[r1].x - lm[r0].x, lm[r0].y - lm[r1].y);
      roll = (roll >= 0 ? roll - Math.PI : roll + Math.PI) / 2;
    }
    // Canonical 468-mesh bounding-box area (0..467, matching FaceMeshSolution.FaceBoundsNormalized)
    let minx = 1, miny = 1, maxx = 0, maxy = 0;
    const meshCount = Math.min(468, lm.length);
    for (let i = 0; i < meshCount; i++) {
      const p = lm[i];
      if (p.x < minx) minx = p.x; if (p.x > maxx) maxx = p.x;
      if (p.y < miny) miny = p.y; if (p.y > maxy) maxy = p.y;
    }
    const area = Math.max(0, maxx - minx) * Math.max(0, maxy - miny);
    const out = outBuf || new Float32Array(4);
    out[0] = yaw; out[1] = pitch; out[2] = roll; out[3] = area;
    return out;
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
