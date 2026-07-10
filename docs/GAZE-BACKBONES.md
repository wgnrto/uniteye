# Gaze backbones (EyeMU vs. a direction-based model)

UnitEye's native provider runs a swappable **gaze backbone** behind the shared MediaPipe FaceMesh +
blink/drowsy/distance stack. Pick it on the `HomulerGaze` component → **Gaze Backbone (model)**, or in
code via the `GazeBackbone` enum. Everything downstream (calibration, filtering, AOI, CSV) consumes the
backbone's `RawGaze` + `Features` unchanged, so switching backbones is transparent to your game — **but
each backbone has its own feature vector, so you must recalibrate after switching.**

Seam: [`IGazeBackbone`](../uniteye/Scripts/Runtime/GazeProvider/IGazeBackbone.cs) — `PerformInference` →
`RawGaze` (pixels) + `Features` (for calibration) + debug textures.

## EyeMU (default, verified)

[`HomulerEyeMURunner`](../uniteye/Scripts/Runtime/Utility/HomulerEyeMURunner.cs). Eye crops + eye corners
+ head geometry → a **screen point** directly (12-feature vector). Ships with the package; this is the
tested path.

## GazeMobileOne / GazeMobileNetV2 (yakhyo/gaze-estimation — integrated; needs a webcam accuracy check)

[`GazeEstimationRunner`](../uniteye/Scripts/Runtime/GazeProvider/GazeEstimationRunner.cs) runs a
direction-based model from [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation). Both
models the project ships (`Resources/ONNX/GazeEstimation/mobileone_s0_gaze.onnx` and
`mobilenetv2_gaze.onnx`) are **already wired and selectable** — pick **GazeMobileOne** or
**GazeMobileNetV2** on `HomulerGaze → Gaze Backbone (model)` before play, **or switch at runtime** with
the **Model:** button in the Gaze UI (Webcam & Model controls) / `HomulerGaze.SetBackbone(...)`. Switching
rebuilds only the model (the shared face-mesh/blink/distance stack stays); calibration is per-backbone, so
gaze falls back to raw until you recalibrate. With a direction model the debug crop is a **face** crop
(one thumbnail), so the Gaze UI shows a **FaceCrop** toggle instead of Eyecrops.

The I/O was introspected from the actual ONNX (menu **UnitEye ▸ Inspect Gaze Models**), so the runner is
coded to the real contract, not a guess — both models are identical:

| | |
|---|---|
| input | `input` — `(1, 3, 448, 448)` NCHW, RGB, **ImageNet-normalized** |
| outputs | `yaw`, `pitch` — each `(1, 90)` per-bin logits |
| decode | softmax over bins → expectation → `index*4° − 180°` → radians (L2CS/Gaze360) |

Per frame the runner: GPU-crops the face from the FaceMesh rect → ImageNet-normalizes via
`PreprocessGazeEstimation.compute` into the `(1,3,448,448)` tensor → runs the model → decodes both heads
→ maps to a rough on-screen `RawGaze` and emits `Features = [pitch, yaw, headYaw, headPitch, headRoll,
headArea, screenW, screenH]` for calibration to refine (so the rough raw mapping needn't be accurate).

**Verified headlessly** (smoke suite): both models import with exactly that I/O and execute on CPU, and
the decode math is correct. **Not yet verified: runtime gaze accuracy** — the face-crop orientation, the
end-to-end normalization, and whether the 90-bin convention matches these specific weights can only be
confirmed with a webcam.

### Hand-test

1. Select **GazeMobileOne** (fastest) on `HomulerGaze` and enter play mode.
2. **Show Eyecrops** → the thumbnail should show your **face**, upright.
3. Move your gaze around and confirm the raw dot tracks the right direction (with `Calibration → None`).
4. **Calibrate** (Ridge or ML) — accuracy comes from calibration, not the raw output. Each backbone has
   its own feature vector, so **recalibrate after switching**.
5. If the uncalibrated dot moves the wrong way or barely/too much: the knobs in `GazeEstimationRunner`
   are `BIN_WIDTH_DEG` / `ANGLE_OFFSET_DEG` (if these weights use a different binning than Gaze360's
   4°/180°) and `ANGLE_TO_SCREEN_GAIN` (pre-calibration travel only). If the face crop looks wrong,
   check the FaceMesh face rect / the Blit UV in `PerformInference`.

### Trade-off

Direction-based models can generalize better across people/distances, but they need the face-crop +
angle→screen step above, and their own calibration. EyeMU's direct screen-point regression is why it
drops straight into this pipeline. If accuracy (not model novelty) is the goal, better calibration data
with EyeMU is the cheaper lever.
