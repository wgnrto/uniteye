# Running UnitEye in a WebGL (browser) build

**Short version:** the native pipeline cannot run on WebGL, but the game logic can. Do the computer
vision in the browser (JavaScript) and feed the gaze point into Unity. Everything downstream —
calibration, filtering, AOI hit-testing, CSV logging — is shared with the native build through the
`IGazeProvider` seam and does not change.

## Status: implemented in `webgl/` (and browser-verified)

This is no longer just a plan — a working browser pipeline is in **`webgl/`** (see `webgl/README.md`):
`uniteye-core.js` (shared-logic ports), `uniteye-cv.js` (getUserMedia + MediaPipe FaceLandmarker + EyeMU
on onnxruntime-web), `index.html` (a gaze + AOI-logging demo with a mock/mouse mode), a `test.html`
self-test, and the Unity glue (`uniteye/Plugins/WebGL/UnitEyeWebGL.jslib`, `WebGLGazeReceiver.cs`,
`uniteye-webgl-boot.js`).

**Verified in a real browser:** the shared ports (ridge predict+train, One-Euro, AOI, geometry) pass a
14/14 self-test; **EyeMU runs in `onnxruntime-web`** with the correct I/O names and finite output;
**MediaPipe FaceLandmarker initializes**; and the full mock → filter → AOI → **log-when-looked-at**
plumbing works end-to-end.

**Unity WebGL builds are verified too** — the gaze scene builds on **Unity 6.3 and 6.5** (batch, via
`UnitEyeWebGLBuild.Build`) and the built players were run in a browser with mock gaze: the complete
JS → jslib → `SendMessage` → `WebGLGazeReceiver` → `WebGLGazeProvider` → `HomulerGaze` chain works
(console markers `UNITEYE_JSLIB_STARTED` → `UNITEYE_PIPELINE_STARTED` → `UNITEYE_WEBGL_BRIDGE_OK` →
`UNITEYE_WEBGL_PROVIDER_OK`). Building required excluding the native MediaPipe assemblies from WebGL
(`Mediapipe.Runtime.asmdef` + the vendored `UniUlm.UnitEye.MediapipeVendored.asmdef` exclude WebGL,
plus `#if !UNITY_WEBGL || UNITY_EDITOR` guards on the native-only UnitEye files) — otherwise IL2CPP
fails to link the `mediapipe_c` P/Invoke symbols. The remaining human-only check: **real gaze accuracy
with a camera + face** (the webcam mode of the browser pipeline, and calibration quality).

## The hard constraint

Unity WebGL builds compile to WebAssembly and **cannot load native plugins** (`.dll`/`.so`/`.aar`). The
homuler MediaPipe plugin is native C++, so it throws `DllNotFoundException` on WebGL and will never work
there (homuler issue #49 has been open since 2021). The moment you target WebGL, the face-landmark stage
must move into the browser's own JS/WASM ML stack. (The EyeMU ONNX model is *not* the blocker — Inference
Engine does run on WebGL, see below — but the landmark source is.)

## The seam: what's shared vs. what forks per platform

```
[ webcam → landmarks → EyeMU → raw gaze (x,y) + blink ]   ← IGazeProvider (forks per platform)
                    ▼
[ calibration (Ridge/MLP) → One-Euro filter → AOI hit-test → CSV log ]   ← shared C#, unchanged
```

`IGazeProvider` (`Scripts/Runtime/GazeProvider/IGazeProvider.cs`) is the seam:

- `NativeGazeProvider` — desktop: homuler MediaPipe + EyeMU on Inference Engine. Done.
- `WebGLGazeProvider` — a stub (guarded by `#if UNITY_WEBGL`) that receives the browser-computed gaze via
  a `.jslib` bridge calling `WebGLGazeProvider.ReportGaze(x, y, facePresent, blinking)`.

`HomulerGaze` picks the provider at `Start()` via `#if UNITY_WEBGL && !UNITY_EDITOR`. The AOI/logging
("log when people look at objects") is the shared side — build it once, it runs the same in an `.exe`
and in a browser tab.

## Recommended architecture

Do the CV in the browser; keep Unity as the app/logic host (or drop Unity from the CV path entirely):

1. **Capture:** JS `getUserMedia()` → hidden `<video>` element (over **HTTPS** — mandatory for the camera).
2. **Landmarks:** MediaPipe **FaceLandmarker** (`@mediapipe/tasks-vision`, WASM) — *not* the deprecated
   `facemesh` CDN package.
3. **EyeMU:** run the model in the browser — the existing EyeMU **TF.js** graph (it shipped a web demo,
   which is what `Resources/ONNX/EyeMUBaseJs.onnx` is a remnant of) or **onnxruntime-web**. Eye-corner
   indices: right 33/133, left 362/263; two 128×128 eye crops.
4. **Bridge:** a `UnitEyeWebGL.jslib` plugin pushes the final gaze `(x,y)` (and optionally the feature
   vector) into Unity via `SendMessage` / an emscripten callback → `WebGLGazeProvider.ReportGaze(...)`.
5. **Downstream:** the shared C# calibration/filter/AOI/CSV runs unchanged. (Optionally reimplement
   calibration in JS too — it's a small algorithm — if you'd rather keep all CV client-side.)

**Golden rule:** compute gaze in JS and pass only the final coordinates across the JS↔WASM boundary.
**Never stream webcam frames into WASM per-frame** — that is the performance killer.

## Inference Engine on WebGL (if you keep EyeMU inside Unity)

Possible, but the slow path — usually not worth it vs. running EyeMU in JS:

- **WebGL2:** only `BackendType.CPU` (Burst→WASM, slow) or `BackendType.GPUPixel` (fragment-shader; no
  random-write so some ops fail, plus known memory leaks). `GPUCompute` does **not** work on WebGL2.
- **`GPUCompute`** only works via the **WebGPU** graphics API, which is **experimental** in Unity 6.3 and
  browser-gated. Gate on `SystemInfo.supportsComputeShaders` with a `GPUPixel`/CPU fallback.

## Gotchas (all paths)

- **HTTPS / secure context** required for `getUserMedia` (or `localhost` for dev).
- Unity-WASM is effectively single-threaded — anything ML you run *inside* Unity is slower than native.
- Unity WebGL's memory/startup footprint is heavy for what is essentially a webcam widget — if you don't
  need a Unity-rendered scene, consider doing the whole thing in the browser and skipping Unity for the CV.
- The only maintained MediaPipe-for-Unity-WebGL asset is a new **paid** one ("MediaPipe for Unity Web",
  2026) — thinly documented and unproven; trial before betting on it. Otherwise the `.jslib` bridge is DIY.

## Multiplayer / "online" note

Networked/multiplayer does **not** affect any of this — eye tracking is client-local. Each player's client
tracks their own webcam and logs locally (or POSTs the log lines to your server). "Online game" only forces
the browser rearchitecture if it means a **WebGL** build; a native networked build uses `NativeGazeProvider`
as-is.
