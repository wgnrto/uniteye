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
direction-based model from [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) (the same
weights uniface serves as "MobileGaze"). Three models ship in
`Resources/ONNX/GazeEstimation/` — `mobileone_s0_gaze.onnx` (fastest), `mobilenetv2_gaze.onnx`, and
`resnet34_gaze.onnx` (largest / most accurate; uniface's default) — all **already wired and selectable**:
pick **GazeMobileOne**, **GazeMobileNetV2**, or **GazeResNet34** on `HomulerGaze → Gaze Backbone (model)`
before play, **or switch at runtime** with
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
→ maps to a rough on-screen `RawGaze` and emits a **polynomial** calibration feature vector for the
calibrator to refine (the rough raw mapping needn't be accurate).

### Why the features are a polynomial (fixing "crosshair stuck in the centre, corners bad")

A gaze *direction* maps to a flat screen non-linearly — `screen_x ≈ distance · tan(yaw + headYaw)`, with a
genuine yaw–pitch coupling toward the corners. UnitEye's calibrators are **per-axis linear** (RidgeRegression
fits X and Y separately; even the MLP trains best on a good basis), so feeding them the *raw* angles let them
fit only the centre slope and **compress the corners inward** — the classic regression-to-the-centre symptom.
The direction backbones therefore emit a low-order polynomial basis instead of the raw angles:

`Features = [yaw, pitch, yaw², pitch², yaw·pitch, yaw³, pitch³, headYaw, headPitch, headRoll, headArea]`
([`GazeEstimationRunner.FillGazeFeatures`](../uniteye/Scripts/Runtime/GazeProvider/GazeEstimationRunner.cs))

The `yaw²/pitch²` + `yaw·pitch` terms are the standard 2nd-order eye-tracking calibration polynomial (they
give the off-centre asymmetry and the diagonal coupling); the `yaw³/pitch³` terms are a pole-free stand-in for
`tan` (`tan x ≈ x + x³/3`) that extends reach at the corners. The two old constant `screenW/screenH` features
were dropped (they standardize to zero — no signal). Nothing in the calibration math changed: RidgeRegression
sizes to the vector length and the MLP infers its input count, and both capture and inference read this same
vector, so train and predict always agree. **This changes the direction backbones' feature length, so
recalibrate them once after updating.**

Sampling was rebalanced to match: the calibration now leads with a **CornerPreset that dwells** (~2 s) at the
four corners + four edge midpoints so the extremes get sustained-fixation samples (previously two of three
presets never left a central box and no point dwelled, so the fit had almost no corner leverage), and the
capture no longer drops downward-gaze frames as false "blinks".

**Verified headlessly** (smoke suite): both models import with exactly that I/O and execute on CPU, and
the decode math is correct.

**Verified against the reference implementation:** these releases live on as **"MobileGaze"** inside
[yakhyo/uniface](https://github.com/yakhyo/uniface) — the author's successor library that consolidates
the earlier repos and downloads the *same* ONNX files from the gaze-estimation releases. Its
`uniface/gaze/models.py` pins the exact conventions this runner uses: 448×448 RGB, ImageNet mean/std,
90 bins, softmax + soft-argmax `* 4° − 180°` → radians, pitch/yaw. So the preprocessing and decode are no
longer guesses. Their demo passes the **raw rectangular detector bbox** (no padding or squaring — the
resize stretch is tolerated); our squared `FACE_CROP_SCALE`-padded landmark bbox emulates that detector
framing without the stretch, and remains the one webcam-tuning knob alongside crop orientation.

uniface also lists **ResNet-18/34/50** gaze variants (larger, more accurate; ResNet-34 is uniface's
default) served from the same releases page — they share this exact I/O contract, so adding one to
UnitEye is only: drop the `.onnx` into `Resources/ONNX/GazeEstimation/`, add a `GazeBackbone` enum entry,
and map it in `NativeGazeProvider.CreateBackbone`. Its other modules (RetinaFace detection, PIPNet
landmarks, head pose) duplicate what MediaPipe FaceLandmarker already provides here, and the Python
library itself is not consumable from Unity.

**Not yet verified: runtime gaze accuracy** — the face-crop framing/orientation on a live webcam.

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
