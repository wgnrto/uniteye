# UnitEye Web (browser / WebGL gaze pipeline)

A browser implementation of the UnitEye gaze pipeline, so eye tracking + "log when people look at
objects" works in a **WebGL** Unity build (or as a standalone web app). The native MediaPipe plugin
can't run on WebGL, so the computer vision runs in the browser here and feeds gaze into Unity.

## Files

| File | What it is | Verified? |
|---|---|---|
| `uniteye-core.js` | Shared consumer logic ported from the C#: ridge calibration (predict + train), One-Euro filter, AOI hit-tests, eye-crop geometry, preprocessing. Classic script (`file://`-safe). | ✅ browser self-test (14/14) |
| `test.html` | Self-test for `uniteye-core.js`. Open it; it shows **ALL PASS**. | ✅ |
| `uniteye-cv.js` | The pipeline: getUserMedia → MediaPipe FaceLandmarker → eye crops → EyeMU (onnxruntime-web) → raw gaze. Plus a **mock/mouse mode**. ES module. | ⚠️ CV needs a webcam; model inference + landmarker init verified |
| `index.html` | Demo: gaze dot + 4 AOIs that highlight and log when looked at. Mock mode auto-starts. | ✅ mock plumbing verified in browser |
| `uniteye-webgl-boot.js` | Glue for a Unity WebGL build — defines `window.UnitEyeStartPipeline` (supports `?mock=1`). | ✅ verified inside real Unity 6.3 + 6.5 WebGL builds (full marker chain) |
| `postprocess-build.ps1` | Copies a Unity WebGL build here + injects the pipeline scripts + model. | ✅ |
| `models/eyemu.onnx` | The EyeMU gaze model (0.7 MB), served to onnxruntime-web. | ✅ runs in ORT |
| `serve-webgl.ps1` | Tiny static server (no node/python needed): `powershell -File serve-webgl.ps1` → http://localhost:8123/ | ✅ |

## Run the demo

Serve this folder over HTTP (the camera needs a secure context — `localhost` counts) and open it:

```
powershell -ExecutionPolicy Bypass -File serve-webgl.ps1      # serves this folder on :8123
# then open http://localhost:8123/index.html
```

- **Mock mode** auto-starts — move your mouse; the dot follows and the AOIs log "look START: …".
- Click **Start (webcam)** for real tracking. It downloads MediaPipe + the EyeMU model, asks for camera
  permission, and tracks your gaze. **Accuracy needs a calibration** (the raw model is uncalibrated);
  wire `engine.setCalibration(regX, regY)` with per-user ridge models the same way the Unity path does.

## What's verified vs. what needs a webcam

Verified in a real browser (via the self-test + `onnxruntime-web`/MediaPipe loads): the shared math
ports, that **EyeMU runs in the browser** (`onnxruntime-web`, correct I/O names `input_1:0`/`input_2:0`/
`input_4`/`input_5` → `dense_8`, finite output), that **MediaPipe FaceLandmarker initializes**, and the
full **mock → filter → AOI → log** plumbing. **Not** verifiable without a camera + face: actual gaze
accuracy, and the eye-crop framing/normalization against a real face.

## Unity WebGL integration

1. Add `uniteye/Plugins/WebGL/UnitEyeWebGL.jslib` (in this package) — the C#↔JS bridge.
2. On WebGL, `WebGLGazeProvider` (picked automatically by `HomulerGaze`) spawns `WebGLGazeReceiver`,
   which calls the jslib to start the pipeline and receives gaze via `SendMessage`. The rest of UnitEye
   (calibration, filter, AOI, CSV logging) runs in C# unchanged.
3. In your WebGL **template**, include `uniteye-core.js`, `uniteye-cv.js`, and `uniteye-webgl-boot.js`,
   and serve `models/eyemu.onnx` next to the build (see `uniteye-webgl-boot.js` for the path).
4. Build with the WebGL module (`Unity Hub → add WebGL Build Support`), serve over HTTPS.

## Building the Unity WebGL player (batch)

```
Unity.exe -batchmode -nographics -buildTarget WebGL -projectPath [host project] ^
  -executeMethod UnitEyeWebGLBuild.Build -logFile build.log
```
Output lands in `[project]/BuildWebGL` (uncompressed, Development). Then wire in the gaze pipeline and
copy it into this served folder:

```
powershell -ExecutionPolicy Bypass -File postprocess-build.ps1 -BuildDir [project]\BuildWebGL -TargetName build63
# open http://localhost:8123/build63/index.html?mock=1   (mock gaze, no webcam needed)
# open http://localhost:8123/build63/index.html          (real webcam pipeline)
```

Console markers prove the chain end-to-end: `UNITEYE_JSLIB_STARTED` (receiver → jslib) →
`UNITEYE_PIPELINE_STARTED` (browser pipeline up) → `UNITEYE_WEBGL_BRIDGE_OK` (first sample reached C#) →
`UNITEYE_WEBGL_PROVIDER_OK` (HomulerGaze consumed it).

Note: the scene logs missing-script warnings for the `Mediapipe`/annotation objects on WebGL — expected,
those are the native-only components excluded from WebGL builds; the browser pipeline replaces them.

## Calibration does not transfer across platforms

Calibrate **per platform** (browser users calibrate in the browser; desktop users in the desktop build).
The raw gaze and most features match the native pipeline, but two features have slightly different
distributions in the browser (head-area uses a landmark bounding box instead of MediaPipe's expanded
face ROI, and blink/EAR scaling differs) — so a calibration file trained natively will carry a bias if
loaded in the browser, and vice versa. Per-user, per-setup calibration is required for accuracy anyway.

## Pinned versions (confirmed working)

- `onnxruntime-web@1.20.1`
- `@mediapipe/tasks-vision@0.10.35` + MediaPipe `face_landmarker.task` (float16/1)
