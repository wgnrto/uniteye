# Barracuda → Inference Engine migration (Path B)

UnitEye moved off Unity's deprecated **Barracuda** inference library onto Unity's **Inference Engine**
(`com.unity.ai.inference`, formerly Sentis), and off **HolisticBarracuda** onto the **native homuler
MediaPipe** plugin for face landmarks. This removed Barracuda entirely (Barracuda and Inference Engine
cannot coexist — both register an importer for `.onnx`).

## What changed

- **Landmarks:** HolisticBarracuda (Barracuda `.onnx`) → native homuler MediaPipe `FaceMeshSolution`
  (468 landmarks + iris via the attention model). No ONNX, no Barracuda for landmarks.
- **EyeMU gaze model:** Barracuda `NNModel` + `IWorker` → Inference Engine `ModelAsset` + `Worker`
  (`HomulerEyeMURunner`).
- **The whole Holistic path was deleted** (`Gaze`, `EyeMURunner`, `EyeHelper`, `Visuallizer`,
  `GazeCalibration`, `GazeEvaluation`, the `Gaze` editor, the 5 Holistic scenes, `UnitEye.prefab`).
  `HomulerGaze` / `HomulerGazeScene` is now the primary pipeline; `UnitEyeAPI` targets it.
- **Package deps:** `com.unity.barracuda` + `jp.ikep.mediapipe.holistic` + the `jp.keijiro`/`jp.ikep`
  scoped registries are gone; `com.unity.ai.inference` (2.6.x, needs Unity 6) was added.

## EyeMU model facts (learned by introspecting the model in-editor)

- Active model: `Resources/ONNX/EyeMUEmbedding.onnx` (referenced by `Resources/EyeMU.asset`).
- **Input names differ from Barracuda:** `input_1:0`, `input_2:0` (the two 128×128 eye crops, with a
  `:0` suffix), `input_4` (8 eye-corner floats), `input_5` (4 head-pose floats).
- **Image inputs are NHWC** `(1, 128, 128, 3)`. `TextureConverter` defaults to NCHW, so the runner uses
  `new TextureTransform().SetTensorLayout(TensorLayout.NHWC)`. (Barracuda's `new Tensor(rt, 3)` was NHWC,
  which is why the old path worked.)
- Outputs: `dense_7` (embedding, 4), `dense_8` (gaze, 2). `dense_6` exists but is unused.

## Two setup steps (required, or the build produces no gaze)

1. **Rebind the model asset.** The `.onnx` re-imports under Inference Engine with a new fileID, so the
   reference in `EyeMU.asset` must be reassigned: run `UnitEye ▸ Rebind EyeMU Model Asset`
   (`EyeMUAssetRebinder`), or `-executeMethod EyeMUAssetRebinder.Rebind` in batch.
2. **Install the MediaPipe StreamingAssets.** homuler loads its `.bytes` models from `StreamingAssets`
   at runtime and throws `FileNotFoundException` on desktop if they are missing. Run
   `UnitEye ▸ Install MediaPipe StreamingAssets` (`MediaPipeAssetInstaller`), or
   `-executeMethod MediaPipeAssetInstaller.Install`. Copies `face_detection_short_range.bytes`,
   `face_landmark_with_attention.bytes`, `face_landmark.bytes` into `Assets/StreamingAssets/`.

## Verified vs. needs-a-camera

**Verified automatically (67-check editor smoke suite, `UnitEyeSmokeTests`):** compiles clean; Barracuda
gone with a single `.onnx` importer; the model imports as `ModelAsset`, binds, loads, and **executes on
CPU with the correct names + NHWC layout returning finite output**; ridge/split/filter/quantizer logic;
scene/prefab scan.

**Needs a human + webcam (cannot be done headless — no camera/GPU/display in batch):** actual gaze
accuracy; the eye-crop **pixel normalization** and **RGB-vs-BGR channel order** (these ride on the
preprocessing shader + `TextureConverter` and only show up as wrong-but-stable gaze); that
`HomulerGazeScene` runs end-to-end and native MediaPipe initializes on the target machine.

## Platform seam (added for native + WebGL)

The CV producer is abstracted behind `IGazeProvider` (`Scripts/Runtime/GazeProvider/`): `NativeGazeProvider`
(this migration) on desktop, `WebGLGazeProvider` (a stub) on WebGL. Everything downstream — calibration,
One-Euro filter, AOI hit-testing, CSV logging — is shared and platform-independent. See `docs/WEBGL.md`.

## Unity version compatibility

Verified green (full 67-check smoke suite) on **Unity 6.3 LTS (6000.3.13f1)** and **Unity 6.5
(6000.5.3f1)**, each in its own host project against the same package folders.

One breaking change surfaced on 6.5 and was fixed: `Object.GetInstanceID()` is hard-deprecated
(CS0619 compile **error**) and `EntityId`'s implicit int conversion is likewise deprecated. The two
affected call sites in the vendored `Scripts/Runtime/Mediapipe/Common/GraphRunner.cs` (write-only
instance tables) now use a local static counter for instance keys instead of Unity identity APIs —
version-proof on both editors. If you vendor more homuler code later, expect the same pattern to need
the same fix.

## Not done / caveats

- homuler stays at **0.12** (0.15 removed the legacy Solution API this code uses).
- **WebGL is not supported** by the native path (homuler is native-only). See `docs/WEBGL.md`.
- `TextureConverter.ToTensor(tex,int,int,int)` is deprecated; the runner uses the pre-allocated-tensor +
  `TextureTransform` form, which is the recommended API.
