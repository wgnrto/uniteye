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

## GazeEstimation (yakhyo/gaze-estimation — needs setup + hand-test)

[`GazeEstimationRunner`](../uniteye/Scripts/Runtime/GazeProvider/GazeEstimationRunner.cs) integrates a
direction-based model such as [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation)
(ResNet / MobileNet / **MobileOne**). These take a **face crop** and output a gaze **direction**
(pitch, yaw) rather than a screen point. The runner:

1. crops the face on the GPU from the FaceMesh face rect (no separate face detector needed);
2. runs the ONNX;
3. decodes (pitch, yaw) and turns it into a rough on-screen `RawGaze`;
4. exposes `Features = [pitch, yaw, headYaw, headPitch, headRoll, headArea, screenW, screenH]` so the
   RidgeRegression / SimpleMLP calibration does the real, per-user angle→screen mapping (just as it
   refines EyeMU's raw output — so the rough raw mapping doesn't need to be accurate).

**Why it's a scaffold, not turnkey:** I can't fetch the ONNX or webcam-test it here, and the exact I/O
depends on which model/variant you export. The runner compiles and self-disables (logs once) if the
model is absent, so the project keeps running on EyeMU.

### Steps to enable it

1. **Export/obtain the ONNX** from yakhyo/gaze-estimation (or L2CS-Net). Prefer a MobileOne variant for
   speed.
2. **Place it** at `uniteye/Resources/ONNX/GazeEstimation.onnx` (Unity imports it as a `ModelAsset`).
3. **Reconcile the model-specific constants** in `GazeEstimationRunner` against your export:
   - `INPUT_SIZE` (448 for L2CS/most yakhyo models; smaller for some MobileOne variants).
   - **Normalization**: if your ONNX does *not* bake in ImageNet mean/std, add it before `ToTensor`
     (a preprocess compute shader like EyeMU's `PreprocessEyeMU.compute`), and confirm NCHW vs NHWC.
   - **Output decoding**: the scaffold assumes one output whose first two values are `(pitch, yaw)` in
     radians. L2CS-style exports instead emit per-bin **logits** needing softmax + expectation — decode
     those if that's your export. Check units (radians vs degrees) and axis order.
   - `ANGLE_TO_SCREEN_GAIN` only affects the pre-calibration `RawGaze`; leave it unless the uncalibrated
     dot barely moves or flies off-screen.
4. **Select** GazeBackbone = GazeEstimation on `HomulerGaze` and enter play mode.
5. **Verify with a webcam**: the face-crop thumbnail (Show Eyecrops) should show your face; move your
   gaze and confirm the raw dot tracks direction; then **run a calibration** (Ridge or ML) — accuracy
   comes from calibration, not the raw output.

### Trade-off

Direction-based models can generalize better across people/distances, but they need the face-crop +
angle→screen step above, and their own calibration. EyeMU's direct screen-point regression is why it
drops straight into this pipeline. If accuracy (not model novelty) is the goal, better calibration data
with EyeMU is the cheaper lever.
